"""Utilities for the OpenAI Parameter Golf challenge."""

from __future__ import annotations

import glob
import io
import json
import math
import os
import random
import statistics
import zlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
from torch import Tensor, nn

from .data import TokenBatch
from .dsl import ArchitectureSpec, ParameterGolfConfig

CONTROL_TENSOR_NAME_PATTERNS = (
    "attn_scale",
    "attn_scales",
    "mlp_scale",
    "mlp_scales",
    "resid_mix",
    "resid_mixes",
    "q_gain",
    "skip_weight",
    "skip_weights",
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = CONTROL_TENSOR_NAME_PATTERNS
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0
DEFAULT_PARAMETER_GOLF_ROOTS = (
    Path("/workspace/tevo_runs"),
)
DEFAULT_PARAMETER_GOLF_CALIBRATION_GLOBS = (
    "runs/runpod_parameter_golf/**/*summary.json",
)
DEFAULT_STANDARD_EVAL_MODE = "standard"
DEFAULT_SLIDING_EVAL_MODE = "sliding64"
DEFAULT_SLIDING_EVAL_STRIDE = 64
MIXED_I5_I6_LARGE_TENSOR_NUMEL = 131_072


def _dtype_nbytes(dtype: torch.dtype) -> int:
    return int(torch.empty((), dtype=dtype).element_size())


def _parameter_golf_roots() -> list[Path]:
    roots: list[Path] = []
    seen: set[Path] = set()
    for env_name in ("TEVO_PARAMETER_GOLF_ROOT", "TEVO_PACKED_ROOT"):
        raw = os.environ.get(env_name)
        if not raw:
            continue
        path = Path(raw).expanduser()
        if path in seen:
            continue
        seen.add(path)
        roots.append(path)
    for path in DEFAULT_PARAMETER_GOLF_ROOTS:
        resolved = path.expanduser()
        if not resolved.exists():
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        roots.append(resolved)
    return roots


def _candidate_paths(path: str | Path) -> list[Path]:
    raw = Path(str(path)).expanduser()
    if raw.is_absolute():
        return [raw]

    candidates: list[Path] = [Path.cwd() / raw]
    for root in _parameter_golf_roots():
        if raw.parts and raw.parts[0] == "runs":
            candidates.append(root.joinpath(*raw.parts[1:]))
        else:
            candidates.append(root / raw)

    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def resolve_parameter_golf_path(path: str | Path) -> Path:
    """Resolve a file path across local cwd and optional Parameter Golf roots."""
    candidates = _candidate_paths(path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    tried = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Missing Parameter Golf path {path!r}; tried: {tried}")


def resolve_parameter_golf_glob(pattern: str | Path) -> list[Path]:
    """Resolve a shard glob across local cwd and optional Parameter Golf roots."""
    tried: list[str] = []
    for candidate in _candidate_paths(pattern):
        tried.append(str(candidate))
        paths = [Path(item) for item in sorted(glob.glob(str(candidate)))]
        if paths:
            return paths
    tried_str = ", ".join(tried)
    raise FileNotFoundError(f"No files found for pattern {pattern!r}; tried: {tried_str}")


def _resolve_glob(pattern: str) -> list[Path]:
    paths = resolve_parameter_golf_glob(pattern)
    return paths


def load_parameter_golf_shard(file: Path) -> Tensor:
    """Load one challenge shard using the published binary header contract."""
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


def load_parameter_golf_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    """Load the full validation token stream and trim it to a whole number of sequences."""
    tokens = torch.cat(
        [load_parameter_golf_shard(file) for file in _resolve_glob(pattern)]
    ).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for seq_len={seq_len}")
    return tokens[: usable + 1]


def build_sentencepiece_luts(
    tokenizer_path: str | Path,
    vocab_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build tokenizer byte-accounting lookup tables exactly like the challenge baseline."""
    sp = spm.SentencePieceProcessor()
    resolved_tokenizer = resolve_parameter_golf_path(tokenizer_path)
    loaded = sp.load(str(resolved_tokenizer))
    if not loaded:
        raise ValueError(f"Failed to load SentencePiece tokenizer from {tokenizer_path}")

    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, int(vocab_size))
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def _trim_eval_tokens(val_tokens: Tensor, seq_len: int, total_eval_tokens: int | None) -> Tensor:
    if total_eval_tokens is None:
        return val_tokens
    usable_tokens = max(int(seq_len), int(total_eval_tokens))
    usable_tokens = min(usable_tokens, int(val_tokens.numel()) - 1)
    usable = (usable_tokens // seq_len) * seq_len
    if usable <= 0:
        return val_tokens[: seq_len + 1]
    return val_tokens[: usable + 1]


def _eval_accumulator_dtype(device: torch.device) -> torch.dtype:
    if device.type == "mps":
        return torch.float32
    return torch.float64


def _score_token_bytes(
    prev_ids: Tensor,
    tgt_ids: Tensor,
    *,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> Tensor:
    token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
    token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(
        dtype=torch.int16
    )
    return token_bytes


def _eval_parameter_golf_val_standard(
    model: nn.Module,
    *,
    seq_len: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    device: torch.device,
    batch_tokens: int,
) -> tuple[float, float]:
    if batch_tokens < seq_len:
        raise ValueError(
            "batch_tokens must be at least seq_len; "
            f"got batch_tokens={batch_tokens}, seq_len={seq_len}"
        )
    batch_seqs = max(1, batch_tokens // seq_len)
    total_seqs = (val_tokens.numel() - 1) // seq_len
    accum_dtype = _eval_accumulator_dtype(device)
    val_loss_sum = torch.zeros((), device=device, dtype=accum_dtype)
    val_token_count = torch.zeros((), device=device, dtype=accum_dtype)
    val_byte_count = torch.zeros((), device=device, dtype=accum_dtype)
    criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")

    with torch.inference_mode():
        for batch_seq_start in range(0, total_seqs, batch_seqs):
            batch_seq_end = min(batch_seq_start + batch_seqs, total_seqs)
            raw_start = batch_seq_start * seq_len
            raw_end = batch_seq_end * seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            logits = model(x)
            batch_loss = criterion(
                logits.view(-1, logits.size(-1)),
                y.reshape(-1),
            )
            val_loss_sum += batch_loss.to(accum_dtype)
            val_token_count += float(y.numel())

            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = _score_token_bytes(
                prev_ids,
                tgt_ids,
                base_bytes_lut=base_bytes_lut,
                has_leading_space_lut=has_leading_space_lut,
                is_boundary_token_lut=is_boundary_token_lut,
            )
            val_byte_count += token_bytes.to(accum_dtype).sum()

    if val_token_count.item() <= 0.0 or val_byte_count.item() <= 0.0:
        return 1e9, 1e9
    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


def _eval_parameter_golf_val_sliding(
    model: nn.Module,
    *,
    seq_len: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    device: torch.device,
    stride_tokens: int,
) -> tuple[float, float]:
    total_pred_tokens = int(val_tokens.numel()) - 1
    if total_pred_tokens <= 0:
        return 1e9, 1e9

    accum_dtype = _eval_accumulator_dtype(device)
    val_loss_sum = torch.zeros((), device=device, dtype=accum_dtype)
    val_token_count = torch.zeros((), device=device, dtype=accum_dtype)
    val_byte_count = torch.zeros((), device=device, dtype=accum_dtype)
    criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    prev_scored = 0
    score_ends = list(range(min(seq_len, total_pred_tokens), total_pred_tokens + 1, stride_tokens))
    if not score_ends or score_ends[-1] != total_pred_tokens:
        score_ends.append(total_pred_tokens)

    with torch.inference_mode():
        for score_end in score_ends:
            begin = max(score_end - seq_len, 0)
            local = val_tokens[begin : score_end + 1].to(device=device, dtype=torch.int64)
            if local.numel() <= 1:
                continue
            x = local[:-1].unsqueeze(0)
            y = local[1:].unsqueeze(0)
            trg_len = max(1, int(score_end) - int(prev_scored))
            score_mask = torch.zeros_like(y, dtype=torch.bool)
            score_mask[:, -trg_len:] = True
            labels = y.masked_fill(~score_mask, -100)
            logits = model(x)
            batch_loss = criterion(
                logits.view(-1, logits.size(-1)),
                labels.reshape(-1),
            )
            val_loss_sum += batch_loss.to(accum_dtype)
            val_token_count += float(trg_len)

            prev_ids = x[:, -trg_len:].reshape(-1)
            tgt_ids = y[:, -trg_len:].reshape(-1)
            token_bytes = _score_token_bytes(
                prev_ids,
                tgt_ids,
                base_bytes_lut=base_bytes_lut,
                has_leading_space_lut=has_leading_space_lut,
                is_boundary_token_lut=is_boundary_token_lut,
            )
            val_byte_count += token_bytes.to(accum_dtype).sum()
            prev_scored = int(score_end)

    if val_token_count.item() <= 0.0 or val_byte_count.item() <= 0.0:
        return 1e9, 1e9
    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


def eval_parameter_golf_val(
    model: nn.Module,
    *,
    seq_len: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    device: torch.device,
    batch_tokens: int,
    eval_mode: str = DEFAULT_STANDARD_EVAL_MODE,
    total_eval_tokens: int | None = None,
    stride_tokens: int = DEFAULT_SLIDING_EVAL_STRIDE,
) -> tuple[float, float]:
    """Compute exact challenge-style validation loss and bits-per-byte."""
    trimmed_tokens = _trim_eval_tokens(val_tokens, seq_len, total_eval_tokens)
    was_training = bool(getattr(model, "training", False))

    try:
        model.eval()
        if eval_mode == DEFAULT_SLIDING_EVAL_MODE:
            return _eval_parameter_golf_val_sliding(
                model,
                seq_len=seq_len,
                val_tokens=trimmed_tokens,
                base_bytes_lut=base_bytes_lut,
                has_leading_space_lut=has_leading_space_lut,
                is_boundary_token_lut=is_boundary_token_lut,
                device=device,
                stride_tokens=max(1, int(stride_tokens)),
            )
        return _eval_parameter_golf_val_standard(
            model,
            seq_len=seq_len,
            val_tokens=trimmed_tokens,
            base_bytes_lut=base_bytes_lut,
            has_leading_space_lut=has_leading_space_lut,
            is_boundary_token_lut=is_boundary_token_lut,
            device=device,
            batch_tokens=batch_tokens,
        )
    finally:
        if was_training:
            model.train()


def tensor_nbytes(tensor: Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _should_keep_fp16_tied_embedding(
    name: str,
    tensor: Tensor,
    parameter_golf: ParameterGolfConfig | None,
) -> bool:
    if parameter_golf is None:
        return False
    if str(getattr(parameter_golf, "tied_embedding_export_dtype", "int8") or "int8") != "fp16":
        return False
    if name not in {"embed.weight", "lm_head.weight", "tok_emb.weight"}:
        return False
    return tensor.ndim == 2


def _tensor_alias_key(tensor: Tensor) -> tuple[object, ...] | None:
    try:
        storage = tensor.untyped_storage()
        return (
            int(storage.data_ptr()),
            int(tensor.storage_offset()),
            tuple(int(dim) for dim in tensor.shape),
            tuple(int(step) for step in tensor.stride()),
            str(tensor.dtype),
        )
    except Exception:
        return None


def _export_quant_bits_for_tensor(
    name: str,
    tensor: Tensor,
    parameter_golf: ParameterGolfConfig | None,
) -> int:
    mode = "int8"
    if parameter_golf is not None:
        mode = str(getattr(parameter_golf, "export_quant_mode", "int8") or "int8").lower()
    if _should_keep_fp16_tied_embedding(name, tensor, parameter_golf):
        return 16
    if mode == "int8":
        return 8
    if mode == "int6":
        return 6
    if mode == "int5":
        return 5
    if mode == "mixed_i5_i6":
        if tensor.ndim == 2 and int(tensor.numel()) >= MIXED_I5_I6_LARGE_TENSOR_NUMEL:
            return 5
        return 6
    return 8


def keep_float_tensor(
    name: str,
    tensor: Tensor,
    passthrough_orig_dtypes: dict[str, str],
    *,
    parameter_golf: ParameterGolfConfig | None = None,
) -> Tensor:
    if _should_keep_fp16_tied_embedding(name, tensor, parameter_golf):
        return tensor.to(dtype=torch.float16).contiguous()
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return tensor.float().contiguous()
    if tensor.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(tensor.dtype).removeprefix("torch.")
        return tensor.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return tensor


def _pack_unsigned_values(unsigned: np.ndarray, bits: int) -> torch.Tensor:
    if unsigned.size == 0:
        return torch.empty((0,), dtype=torch.uint8)
    bit_positions = np.arange(bits, dtype=np.uint16)
    bit_planes = ((unsigned[:, None] >> bit_positions) & 1).astype(np.uint8, copy=False)
    flat_bits = bit_planes.reshape(-1)
    pad = (-int(flat_bits.size)) % 8
    if pad:
        flat_bits = np.pad(flat_bits, (0, pad))
    packed = np.packbits(flat_bits.reshape(-1, 8)[:, ::-1], axis=1, bitorder="big").reshape(-1)
    return torch.from_numpy(packed.copy())


def _unpack_unsigned_values(packed: torch.Tensor, *, bits: int, count: int) -> np.ndarray:
    if count <= 0:
        return np.empty((0,), dtype=np.uint8)
    raw = packed.detach().to("cpu", dtype=torch.uint8).numpy()
    unpacked = np.unpackbits(raw, bitorder="big").reshape(-1, 8)[:, ::-1].reshape(-1)
    unpacked = unpacked[: count * bits].reshape(count, bits)
    weights = (1 << np.arange(bits, dtype=np.uint16)).reshape(1, bits)
    values = (unpacked.astype(np.uint16) * weights).sum(axis=1)
    return values.astype(np.uint16, copy=False)


def quantize_float_tensor(tensor: Tensor, *, bits: int = 8) -> tuple[Tensor, Tensor]:
    t32 = tensor.float()
    qmax = float((1 << (int(bits) - 1)) - 1)
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / qmax).clamp_min(1.0 / qmax)
        quantized = torch.clamp(torch.round(clipped / scale[:, None]), -qmax, qmax)
        return quantized.to(torch.int8).contiguous(), scale.to(
            dtype=INT8_PER_ROW_SCALE_DTYPE
        ).contiguous()

    clip_abs = (
        float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    )
    scale = torch.tensor(clip_abs / qmax if clip_abs > 0 else 1.0, dtype=torch.float32)
    quantized = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -qmax, qmax)
    return quantized.to(torch.int8).contiguous(), scale


def quantize_state_dict_int8(
    state_dict: dict[str, Tensor],
    *,
    parameter_golf: ParameterGolfConfig | None = None,
) -> tuple[dict[str, object], dict[str, int]]:
    """Quantize a checkpoint using the challenge's published int8 export format."""
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    aliases: dict[str, str] = {}
    seen_alias_keys: dict[tuple[object, ...], str] = {}
    stats = dict.fromkeys(
        (
            "param_count",
            "num_tensors",
            "num_float_tensors",
            "num_nonfloat_tensors",
            "baseline_tensor_bytes",
            "int8_payload_bytes",
        ),
        0,
    )

    for name, tensor in state_dict.items():
        alias_key = _tensor_alias_key(tensor)
        if alias_key is not None:
            source_name = seen_alias_keys.get(alias_key)
            if source_name is not None:
                aliases[name] = source_name
                continue
            seen_alias_keys[alias_key] = name
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue

        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(
                name,
                t,
                passthrough_orig_dtypes,
                parameter_golf=parameter_golf,
            )
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        if _should_keep_fp16_tied_embedding(name, t, parameter_golf):
            kept = t.to(dtype=torch.float16).contiguous()
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        bits = _export_quant_bits_for_tensor(name, t, parameter_golf)
        quantized_tensor, scale = quantize_float_tensor(t, bits=bits)
        if scale.ndim > 0:
            qmeta[name] = {
                "scheme": "per_row",
                "axis": 0,
                "bits": bits,
                "shape": [int(dim) for dim in t.shape],
            }
        else:
            qmeta[name] = {
                "scheme": "per_tensor",
                "bits": bits,
                "shape": [int(dim) for dim in t.shape],
            }
        quantized[name] = quantized_tensor
        scales[name] = scale
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        packed_bytes = int(math.ceil(float(t.numel()) * float(bits) / 8.0))
        stats["int8_payload_bytes"] += packed_bytes + tensor_nbytes(scale)

    payload: dict[str, object] = {
        "__quant_format__": "packed_low_precision_v2",
        "quantized": {},
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    packed_quantized: dict[str, Tensor] = {}
    for name, tensor in quantized.items():
        bits = int(qmeta.get(name, {}).get("bits", 8))
        if bits >= 8:
            packed_quantized[name] = tensor
            continue
        qmax = (1 << (bits - 1)) - 1
        unsigned = (tensor.to(torch.int16).reshape(-1) + int(qmax)).numpy().astype(np.uint16)
        packed_quantized[name] = _pack_unsigned_values(unsigned, bits)
    payload["quantized"] = packed_quantized
    if qmeta:
        payload["qmeta"] = qmeta
    if aliases:
        payload["aliases"] = aliases
    if passthrough_orig_dtypes:
        payload["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return payload, {key: int(value) for key, value in stats.items()}


def dequantize_state_dict_int8(payload: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = payload.get("qmeta", {})
    aliases = payload.get("aliases", {})
    passthrough_orig_dtypes = payload.get("passthrough_orig_dtypes", {})
    quantized = payload.get("quantized", {})
    scales = payload.get("scales", {})
    dtypes = payload.get("dtypes", {})
    passthrough = payload.get("passthrough", {})
    if (
        not isinstance(quantized, dict)
        or not isinstance(scales, dict)
        or not isinstance(dtypes, dict)
    ):
        raise ValueError("Invalid quantized payload")

    for name, tensor in quantized.items():
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Quantized tensor {name} is not a torch.Tensor")
        dtype_name = dtypes.get(name)
        scale = scales.get(name)
        if not isinstance(dtype_name, str) or not isinstance(scale, torch.Tensor):
            raise ValueError(f"Quantized tensor {name} is missing dtype or scale metadata")
        dtype = getattr(torch, dtype_name)
        bits = int(qmeta.get(name, {}).get("bits", 8)) if isinstance(qmeta, dict) else 8
        restored_quant = tensor
        if bits < 8:
            target_shape = None
            meta = qmeta.get(name, {}) if isinstance(qmeta, dict) else {}
            if isinstance(meta, dict):
                target_shape = meta.get("shape")
            if target_shape is None:
                raise ValueError(f"Quantized tensor {name} is missing a target shape")
            shape_list = [int(dim) for dim in target_shape]
            count = int(np.prod(shape_list))
            unpacked = _unpack_unsigned_values(tensor, bits=bits, count=count)
            qmax = (1 << (bits - 1)) - 1
            restored_quant = torch.from_numpy(
                unpacked.astype(np.int16, copy=False) - int(qmax)
            ).reshape(*shape_list)
        if (
            isinstance(qmeta, dict) and qmeta.get(name, {}).get("scheme") == "per_row"
        ) or scale.ndim > 0:
            row_scale = scale.to(dtype=torch.float32)
            out[name] = (
                (
                    restored_quant.float()
                    * row_scale.view(restored_quant.shape[0], *([1] * (restored_quant.ndim - 1)))
                )
                .to(dtype=dtype)
                .contiguous()
            )
        else:
            out[name] = (
                restored_quant.float() * float(scale.item())
            ).to(dtype=dtype).contiguous()

    if not isinstance(passthrough, dict):
        raise ValueError("Invalid passthrough payload")
    for name, tensor in passthrough.items():
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Passthrough tensor {name} is not a torch.Tensor")
        restored = tensor.detach().to("cpu").contiguous()
        if isinstance(passthrough_orig_dtypes, dict):
            orig_dtype = passthrough_orig_dtypes.get(name)
            if isinstance(orig_dtype, str):
                restored = restored.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = restored
    if isinstance(aliases, dict):
        for alias_name, source_name in aliases.items():
            if not isinstance(alias_name, str) or not isinstance(source_name, str):
                continue
            source_tensor = out.get(source_name)
            if source_tensor is None:
                raise ValueError(f"Alias {alias_name} references missing tensor {source_name}")
            out[alias_name] = source_tensor
    return out


def measure_quantized_artifact(
    state_dict: dict[str, Tensor],
    *,
    code_bytes: int,
    artifact_budget_bytes: int,
    parameter_golf: ParameterGolfConfig | None = None,
) -> dict[str, float]:
    """Return exact int8+zlib artifact sizes for a checkpoint."""
    quant_payload, stats = quantize_state_dict_int8(
        state_dict,
        parameter_golf=parameter_golf,
    )
    quant_buf = io.BytesIO()
    torch.save(quant_payload, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    zlib_bytes = len(quant_blob)
    total_bytes = int(zlib_bytes) + int(code_bytes)
    return {
        "artifact_code_bytes": float(code_bytes),
        "artifact_budget_bytes": float(artifact_budget_bytes),
        "artifact_payload_bytes": float(stats["int8_payload_bytes"]),
        "artifact_raw_torch_bytes": float(len(quant_raw)),
        "artifact_zlib_bytes": float(zlib_bytes),
        "artifact_total_bytes": float(total_bytes),
        "artifact_over_budget_bytes": float(max(0, total_bytes - int(artifact_budget_bytes))),
    }


def estimate_quantized_payload_bytes_from_state_dict(
    state_dict: dict[str, Tensor],
    *,
    parameter_golf: ParameterGolfConfig | None = None,
) -> int:
    """Estimate int8 payload bytes from shapes and dtypes without serializing values."""
    total = 0
    seen_alias_keys: set[tuple[object, ...]] = set()
    for name, tensor in state_dict.items():
        alias_key = _tensor_alias_key(tensor)
        if alias_key is not None:
            if alias_key in seen_alias_keys:
                continue
            seen_alias_keys.add(alias_key)
        t = tensor.detach().to("cpu")
        if not t.is_floating_point():
            total += tensor_nbytes(t.contiguous())
            continue
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            if _should_keep_fp16_tied_embedding(name, t, parameter_golf):
                total += int(t.numel()) * _dtype_nbytes(torch.float16)
            elif any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
                total += int(t.numel()) * _dtype_nbytes(torch.float32)
            elif t.dtype in {torch.float32, torch.bfloat16}:
                total += int(t.numel()) * _dtype_nbytes(INT8_KEEP_FLOAT_STORE_DTYPE)
            else:
                total += tensor_nbytes(t.contiguous())
            continue
        if _should_keep_fp16_tied_embedding(name, t, parameter_golf):
            total += int(t.numel()) * _dtype_nbytes(torch.float16)
            continue
        bits = _export_quant_bits_for_tensor(name, t, parameter_golf)
        total += int(math.ceil(float(t.numel()) * float(bits) / 8.0))
        if t.ndim == 2:
            total += int(t.shape[0]) * _dtype_nbytes(INT8_PER_ROW_SCALE_DTYPE)
        else:
            total += _dtype_nbytes(torch.float32)
    return total


def estimate_artifact_total_bytes_for_spec(spec: ArchitectureSpec) -> tuple[int, int]:
    """Estimate challenge payload and total bytes by inspecting a freshly initialized model."""
    if spec.parameter_golf is None:
        raise ValueError("ArchitectureSpec.parameter_golf must be configured to estimate artifacts")
    from .models import EvolutionModel

    model = EvolutionModel(spec.model)
    payload_bytes = estimate_quantized_payload_bytes_from_state_dict(
        model.state_dict(),
        parameter_golf=spec.parameter_golf,
    )
    total_bytes = payload_bytes + int(spec.parameter_golf.code_bytes)
    return payload_bytes, total_bytes


def _read_calibration_summary(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    preflight = payload.get("preflight")
    metrics = payload.get("metrics")
    if not isinstance(preflight, dict) or not isinstance(metrics, dict):
        return None
    est_total = preflight.get("artifact_total_bytes_est")
    est_payload = preflight.get("artifact_payload_bytes_est")
    exact_total = metrics.get("artifact_total_bytes")
    exact_payload = metrics.get("artifact_payload_bytes")
    if not all(isinstance(value, (int, float)) and float(value) > 0 for value in (est_total, est_payload, exact_total, exact_payload)):
        return None
    return {
        "path": str(path),
        "tied_embedding_export_dtype": str(preflight.get("tied_embedding_export_dtype") or "int8"),
        "export_quant_mode": str(preflight.get("export_quant_mode") or "int8"),
        "est_total": float(est_total),
        "est_payload": float(est_payload),
        "exact_total": float(exact_total),
        "exact_payload": float(exact_payload),
    }


def artifact_size_calibration_table() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    seen: set[Path] = set()
    for pattern in DEFAULT_PARAMETER_GOLF_CALIBRATION_GLOBS:
        for raw_path in sorted(Path.cwd().glob(pattern)):
            path = raw_path.resolve()
            if path in seen:
                continue
            seen.add(path)
            row = _read_calibration_summary(path)
            if row is not None:
                rows.append(row)

    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        group_key = (
            f"{row['tied_embedding_export_dtype']}::{row['export_quant_mode']}"
        )
        grouped.setdefault(group_key, []).append(row)

    def _group_summary(items: list[dict[str, object]]) -> dict[str, float]:
        total_ratios = [float(item["exact_total"]) / float(item["est_total"]) for item in items]
        payload_ratios = [
            float(item["exact_payload"]) / float(item["est_payload"]) for item in items
        ]
        return {
            "sample_count": float(len(items)),
            "total_ratio_median": float(statistics.median(total_ratios)),
            "payload_ratio_median": float(statistics.median(payload_ratios)),
        }

    groups = {key: _group_summary(items) for key, items in grouped.items()}
    overall = _group_summary(rows) if rows else {
        "sample_count": 0.0,
        "total_ratio_median": 1.0,
        "payload_ratio_median": 1.0,
    }
    return {
        "sample_count": len(rows),
        "groups": groups,
        "overall": overall,
    }


def estimate_calibrated_artifact_total_bytes_for_spec(spec: ArchitectureSpec) -> tuple[int, int]:
    payload_bytes, total_bytes = estimate_artifact_total_bytes_for_spec(spec)
    if spec.parameter_golf is None:
        return payload_bytes, total_bytes
    table = artifact_size_calibration_table()
    groups = table.get("groups", {})
    key = (
        f"{spec.parameter_golf.tied_embedding_export_dtype}::"
        f"{spec.parameter_golf.export_quant_mode}"
    )
    summary = groups.get(key) if isinstance(groups, dict) else None
    if not isinstance(summary, dict) or float(summary.get("sample_count", 0.0) or 0.0) < 1.0:
        summary = table.get("overall", {})
    if not isinstance(summary, dict):
        return payload_bytes, total_bytes
    payload_ratio = float(summary.get("payload_ratio_median", 1.0) or 1.0)
    total_ratio = float(summary.get("total_ratio_median", 1.0) or 1.0)
    calibrated_payload = max(0, int(round(float(payload_bytes) * payload_ratio)))
    calibrated_total = max(0, int(round(float(total_bytes) * total_ratio)))
    return calibrated_payload, calibrated_total


@dataclass
class _TokenStream:
    files: list[Path]
    file_idx: int = 0
    tokens: Tensor | None = None
    pos: int = 0

    def __post_init__(self) -> None:
        if not self.files:
            raise FileNotFoundError("Parameter Golf token stream has no shard files")
        self.tokens = load_parameter_golf_shard(self.files[0])

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_parameter_golf_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        if self.tokens is None:
            raise RuntimeError("Parameter Golf token stream is not initialized")
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            available = self.tokens.numel() - self.pos
            if available <= 0:
                self._advance_file()
                continue
            count = min(remaining, available)
            chunks.append(self.tokens[self.pos : self.pos + count])
            self.pos += count
            remaining -= count
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class ParameterGolfDataModule:
    """Sequential challenge-shard reader that yields training and eval batches."""

    class _BatchIterable:
        def __init__(
            self,
            module: ParameterGolfDataModule,
            *,
            split: str,
            max_tokens: int | None,
        ) -> None:
            self.module = module
            self.split = split
            self.max_tokens = max_tokens
            self._iter: Iterator[TokenBatch] | None = None

        def __iter__(self) -> Iterator[TokenBatch]:
            self._iter = self.module._batch_generator(split=self.split, max_tokens=self.max_tokens)
            return self

        def __next__(self) -> TokenBatch:
            if self._iter is None:
                self._iter = self.module._batch_generator(
                    split=self.split, max_tokens=self.max_tokens
                )
            return next(self._iter)

    def __init__(
        self,
        cfg: ParameterGolfConfig,
        *,
        seq_len: int,
        batch_size: int,
        seed: int = 0,
    ) -> None:
        self.cfg = cfg
        self.seq_len = int(seq_len)
        self.batch_size = max(1, int(batch_size))
        self._seed = int(seed)
        self._rng = random.Random(self._seed)  # noqa: S311  # deterministic unit tests / training order
        self._train_files = _resolve_glob(cfg.train_shards_glob)
        self._val_files = _resolve_glob(cfg.val_shards_glob)
        self._streams: dict[str, _TokenStream] = {
            "train": _TokenStream(self._train_files),
            "val": _TokenStream(self._val_files),
        }
        self._validation_cache: Tensor | None = None

    def reset_rng(self, seed: int | None = None) -> None:
        if seed is not None:
            self._seed = int(seed)
        self._rng = random.Random(self._seed)  # noqa: S311
        self._streams = {
            "train": _TokenStream(self._train_files),
            "val": _TokenStream(self._val_files),
        }

    def batches(
        self, max_tokens: int | None = None, *, split: str = "train"
    ) -> Iterable[TokenBatch]:
        return ParameterGolfDataModule._BatchIterable(self, split=split, max_tokens=max_tokens)

    def validation_tokens(self) -> Tensor:
        if self._validation_cache is None:
            self._validation_cache = load_parameter_golf_validation_tokens(
                self.cfg.val_shards_glob,
                self.seq_len,
            )
        return self._validation_cache

    def _batch_generator(self, *, split: str, max_tokens: int | None) -> Iterator[TokenBatch]:
        if split not in self._streams:
            raise ValueError(f"Unknown Parameter Golf split: {split}")
        stream = self._streams[split]
        local_tokens = self.batch_size * self.seq_len
        budget = max_tokens
        batch_index = 0
        while True:
            chunk = stream.take(local_tokens + 1)
            input_ids = chunk[:-1].reshape(-1, self.seq_len).to(dtype=torch.long)
            target_ids = chunk[1:].reshape(-1, self.seq_len).to(dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            uids = [f"pg-{split}-{batch_index}-{row}" for row in range(input_ids.size(0))]
            yield TokenBatch(
                input_ids=input_ids,
                attention_mask=attention_mask,
                uids=uids,
                target_ids=target_ids,
            )
            batch_index += 1
            if budget is not None:
                budget -= local_tokens
                if budget <= 0:
                    return
