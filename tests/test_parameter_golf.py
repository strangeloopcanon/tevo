from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import sentencepiece as spm
import torch

import transformer_evolution_llm.trainer as trainer_module
from transformer_evolution_llm import api
from transformer_evolution_llm.candidates import Candidate
from transformer_evolution_llm.dsl import ArchitectureSpec, ParameterGolfConfig
from transformer_evolution_llm.models import EvolutionModel
from transformer_evolution_llm.parameter_golf import (
    ParameterGolfDataModule,
    artifact_size_calibration_table,
    build_sentencepiece_luts,
    estimate_artifact_total_bytes_for_spec,
    estimate_calibrated_artifact_total_bytes_for_spec,
    eval_parameter_golf_val,
    load_parameter_golf_shard,
    measure_quantized_artifact,
    quantize_state_dict_int8,
    resolve_parameter_golf_glob,
    resolve_parameter_golf_path,
)
from transformer_evolution_llm.parameter_golf_export import (
    ParameterGolfExportError,
    build_official_submission_plan,
)
from transformer_evolution_llm.parameter_golf_runtime import (
    preflight_parameter_golf_config,
    rescore_parameter_golf_checkpoint,
    run_parameter_golf_benchmark,
)
from transformer_evolution_llm.trainer import FullWeightTrainer


def _write_pg_shard(path: Path, tokens: list[int]) -> None:
    header = np.zeros(256, dtype="<i4")
    header[0] = 20240520
    header[1] = 1
    header[2] = len(tokens)
    with path.open("wb") as handle:
        header.tofile(handle)
        np.asarray(tokens, dtype="<u2").tofile(handle)


def _build_pg_assets(tmp_path: Path) -> dict[str, object]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text(
        "\n".join(
            [
                "parameter golf loves compact models",
                "shared weights can stretch depth",
                "tiny transformers still learn",
                "openai challenge fineweb tokenizer",
            ]
        )
    )
    model_prefix = tmp_path / "toy_sp"
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        vocab_size=48,
        model_type="bpe",
        character_coverage=1.0,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        hard_vocab_limit=False,
    )
    tokenizer_path = model_prefix.with_suffix(".model")
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))

    train_tokens: list[int] = []
    val_tokens: list[int] = []
    for _ in range(12):
        train_tokens.extend(sp.encode("parameter golf loves compact models", out_type=int))
        train_tokens.extend(sp.encode("shared weights can stretch depth", out_type=int))
    for _ in range(8):
        val_tokens.extend(sp.encode("tiny transformers still learn", out_type=int))
        val_tokens.extend(sp.encode("openai challenge fineweb tokenizer", out_type=int))

    train_path = tmp_path / "pg_train_000.bin"
    val_path = tmp_path / "pg_val_000.bin"
    _write_pg_shard(train_path, train_tokens)
    _write_pg_shard(val_path, val_tokens)
    return {
        "sp": sp,
        "tokenizer_path": tokenizer_path,
        "train_path": train_path,
        "val_path": val_path,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
    }


def _make_pg_spec(assets: dict[str, object]) -> ArchitectureSpec:
    sp = assets["sp"]
    if not isinstance(sp, spm.SentencePieceProcessor):
        raise TypeError("SentencePiece asset is missing.")
    tokenizer_path = assets["tokenizer_path"]
    train_path = assets["train_path"]
    val_path = assets["val_path"]
    if not isinstance(tokenizer_path, Path | str):
        raise TypeError("tokenizer path asset is missing.")
    if not isinstance(train_path, Path | str):
        raise TypeError("train shard asset is missing.")
    if not isinstance(val_path, Path | str):
        raise TypeError("val shard asset is missing.")

    return ArchitectureSpec(
        model={
            "name": "pg-smoke",
            "emb": {"dim": 32, "vocab": sp.vocab_size()},
            "blocks": [
                {
                    "attn": {
                        "kind": "MQA",
                        "heads": 2,
                        "head_dim": 16,
                        "softmax": {"qk_norm": "rms", "softcap": 16.0},
                    },
                    "ffn": {"type": "dense", "hidden": 96, "activation": "swiglu"},
                }
            ],
            "head": {"vocab": sp.vocab_size(), "tie_embeddings": True},
        },
        train={
            "lr": 1e-3,
            "warmup": 0,
            "clip": 1.0,
            "bf16": False,
            "grad_ckpt": False,
            "max_tokens": 32,
            "seed": 0,
        },
        data={
            "tokenizer": "parameter-golf",
            "seq_len": 8,
            "batch_size": 2,
            "workers": 0,
            "eval_tokens": 16,
            "shards": [{"name": "placeholder", "split": "train", "weight": 1.0}],
        },
        parameter_golf=ParameterGolfConfig(
            train_shards_glob=str(train_path),
            val_shards_glob=str(val_path),
            tokenizer_path=str(tokenizer_path),
            artifact_budget_bytes=16_000_000,
            code_bytes=4096,
            track="10min",
        ),
    )


def _make_official_lane_spec(assets: dict[str, object]) -> ArchitectureSpec:
    spec = _make_pg_spec(assets)
    spec.model.name = "pg-official-exportable"
    spec.model.blocks[0].attn.kind = "MQA"
    if spec.model.blocks[0].attn is not None:
        spec.model.blocks[0].attn.kv_groups = spec.model.blocks[0].attn.heads
    if spec.model.blocks[0].ffn is None:
        raise TypeError("missing FFN config")
    spec.model.blocks[0].ffn.activation = "relu_squared"
    spec.train.lr = 0.04
    spec.train.matrix_lr = 0.04
    spec.train.scalar_lr = 0.04
    spec.train.tied_embedding_lr = 0.05
    spec.train.weight_decay = 0.0
    spec.train.clip = 0.0
    spec.train.optimizer.name = "muon"
    spec.train.optimizer.weight_decay = 0.0
    spec.train.optimizer.gradient_transform.mode = "identity"
    spec.train.optimizer.update_filter.mode = "none"
    spec.train.optimizer.muon_momentum = 0.95
    spec.train.optimizer.muon_momentum_warmup_start = 0.85
    spec.train.optimizer.muon_momentum_warmup_steps = 500
    spec.parameter_golf.code_bytes = 50_000
    spec.parameter_golf.max_wallclock_seconds = 600.0
    spec.parameter_golf.val_batch_tokens = 32
    spec.parameter_golf.val_loss_every = 10
    return spec


def _write_official_train_stub(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                '"""official stub"""',
                "from __future__ import annotations",
                "",
                "import os",
                "",
                "class Hyperparameters:",
                '    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", "0"))',
                "",
                'if __name__ == "__main__":',
                "    print(Hyperparameters.train_batch_tokens)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _write_official_patch_stub(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "",
                "import os",
                "from torch import Tensor",
                "",
                "INT8_KEEP_FLOAT_STORE_DTYPE = None",
                "",
                "class Hyperparameters:",
                '    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))',
                "",
                (
                    "def keep_float_tensor("
                    "name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]"
                    ") -> Tensor:"
                ),
                "    if any(pattern in name for pattern in []):",
                "        return t.float().contiguous()",
                "    return t",
                "",
                "def step(args, matrix_params, scale, optimizers, base_model):",
                "    if args.grad_clip_norm > 0:",
                (
                    "        torch.nn.utils.clip_grad_norm_("
                    "base_model.parameters(), args.grad_clip_norm)"
                ),
                "    for opt in optimizers:",
                "        opt.step()",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_parameter_golf_data_module_emits_explicit_targets(tmp_path: Path) -> None:
    assets = _build_pg_assets(tmp_path)
    cfg = ParameterGolfConfig(
        train_shards_glob=str(assets["train_path"]),
        val_shards_glob=str(assets["val_path"]),
        tokenizer_path=str(assets["tokenizer_path"]),
        artifact_budget_bytes=16_000_000,
        code_bytes=4096,
        track="10min",
    )
    module = ParameterGolfDataModule(cfg, seq_len=8, batch_size=2, seed=0)
    batch = next(iter(module.batches(max_tokens=16, split="train")))
    source_tokens = load_parameter_golf_shard(Path(str(assets["train_path"])))
    expected = source_tokens[: 16 + 1].to(dtype=torch.long)
    assert batch.target_ids is not None
    assert torch.equal(batch.input_ids.reshape(-1), expected[:-1])
    assert torch.equal(batch.target_ids.reshape(-1), expected[1:])


def test_sentencepiece_luts_track_leading_space(tmp_path: Path) -> None:
    assets = _build_pg_assets(tmp_path)
    sp = assets["sp"]
    if not isinstance(sp, spm.SentencePieceProcessor):
        raise TypeError("SentencePiece asset is missing.")
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        tokenizer_path=str(assets["tokenizer_path"]),
        vocab_size=sp.vocab_size(),
        device=torch.device("cpu"),
    )

    leading_token = next(i for i in range(sp.vocab_size()) if sp.id_to_piece(i).startswith("▁"))
    normal_token = next(i for i in range(sp.vocab_size()) if not is_boundary_token_lut[i].item())
    piece = sp.id_to_piece(leading_token).removeprefix("▁")

    assert has_leading_space_lut[leading_token].item()
    assert base_bytes_lut[leading_token].item() == len(piece.encode("utf-8"))
    assert (
        base_bytes_lut[leading_token].item()
        + int(
            has_leading_space_lut[leading_token].item()
            and not is_boundary_token_lut[normal_token].item()
        )
        == len(piece.encode("utf-8")) + 1
    )


def test_measure_quantized_artifact_reports_total_bytes() -> None:
    state_dict = {
        "linear.weight": torch.randn(32, 16),
        "linear.bias": torch.randn(32),
    }
    metrics = measure_quantized_artifact(
        state_dict,
        code_bytes=1234,
        artifact_budget_bytes=16_000_000,
    )
    assert metrics["artifact_total_bytes"] == metrics["artifact_zlib_bytes"] + 1234.0
    assert metrics["artifact_payload_bytes"] > 0.0


def test_quantized_artifact_dedupes_tied_weights_and_fp16_passthrough() -> None:
    shared = torch.randn(64, 32)
    state_dict = {
        "embed.weight": shared,
        "lm_head.weight": shared,
    }
    cfg = ParameterGolfConfig(
        train_shards_glob="data/pg/train_*.bin",
        val_shards_glob="data/pg/val_*.bin",
        tokenizer_path="data/pg/sp1024.model",
        tied_embedding_export_dtype="fp16",
    )

    payload, _ = quantize_state_dict_int8(state_dict, parameter_golf=cfg)
    single_metrics = measure_quantized_artifact(
        {"embed.weight": shared},
        code_bytes=0,
        artifact_budget_bytes=16_000_000,
        parameter_golf=cfg,
    )
    aliased_metrics = measure_quantized_artifact(
        state_dict,
        code_bytes=0,
        artifact_budget_bytes=16_000_000,
        parameter_golf=cfg,
    )

    assert payload.get("aliases") == {"lm_head.weight": "embed.weight"}
    assert aliased_metrics["artifact_payload_bytes"] == single_metrics["artifact_payload_bytes"]


def test_quantized_artifact_supports_mixed_i5_i6_export_mode() -> None:
    large = torch.randn(512, 512)
    medium = torch.randn(300, 256)
    state_dict = {
        "large.weight": large,
        "medium.weight": medium,
    }
    cfg = ParameterGolfConfig(
        train_shards_glob="data/pg/train_*.bin",
        val_shards_glob="data/pg/val_*.bin",
        tokenizer_path="data/pg/sp1024.model",
        export_quant_mode="mixed_i5_i6",
    )

    mixed_payload, _ = quantize_state_dict_int8(state_dict, parameter_golf=cfg)
    int8_payload, _ = quantize_state_dict_int8(state_dict, parameter_golf=None)

    assert mixed_payload["qmeta"]["large.weight"]["bits"] == 5
    assert mixed_payload["qmeta"]["medium.weight"]["bits"] == 6
    assert (
        mixed_payload["quantized"]["large.weight"].numel()
        < int8_payload["quantized"]["large.weight"].numel()
    )


def test_parameter_golf_trainer_smoke(tmp_path: Path) -> None:
    assets = _build_pg_assets(tmp_path)
    spec = _make_pg_spec(assets)
    module = ParameterGolfDataModule(
        spec.parameter_golf,
        seq_len=spec.data.seq_len,
        batch_size=spec.data.batch_size,
        seed=0,
    )
    trainer = FullWeightTrainer(
        checkpoint_dir=tmp_path / "checkpoints",
        device="cpu",
        steps=2,
        eval_batches=1,
    )
    metrics, checkpoint = trainer.train(
        Candidate(ident="pg-smoke", spec=spec),
        spec,
        module.batches(max_tokens=spec.train.max_tokens, split="train"),
    )
    assert checkpoint.exists()
    assert metrics["parameter_golf_error"] == 0.0
    assert metrics["val_bpb"] > 0.0
    assert metrics["post_quant_val_bpb"] > 0.0
    assert metrics["artifact_total_bytes"] >= metrics["artifact_code_bytes"]


def test_parameter_golf_trainer_reports_split_time_and_eval_modes(tmp_path: Path) -> None:
    assets = _build_pg_assets(tmp_path / "assets_metrics")
    spec = _make_pg_spec(assets)
    spec.parameter_golf.eval_protocol = "truth_full"
    spec.parameter_golf.report_eval_modes = ["standard", "sliding64"]

    module = ParameterGolfDataModule(
        spec.parameter_golf,
        seq_len=spec.data.seq_len,
        batch_size=spec.data.batch_size,
        seed=0,
    )
    trainer = FullWeightTrainer(
        checkpoint_dir=tmp_path / "checkpoints_metrics",
        device="cpu",
        steps=1,
        eval_batches=1,
    )
    metrics, _ = trainer.train(
        Candidate(ident="pg-metrics", spec=spec),
        spec,
        module.batches(max_tokens=spec.train.max_tokens, split="train"),
    )

    assert metrics["val_bpb"] == metrics["val_bpb_standard"]
    assert metrics["post_quant_val_bpb"] == metrics["post_quant_val_bpb_standard"]
    assert metrics["val_bpb_sliding64"] > 0.0
    assert metrics["post_quant_val_bpb_sliding64"] > 0.0
    assert metrics["train_wallclock_seconds_used"] == metrics["wallclock_seconds_used"]
    assert metrics["post_eval_wallclock_seconds_used"] >= 0.0
    assert metrics["total_job_seconds_used"] >= metrics["train_wallclock_seconds_used"]
    assert metrics["total_job_seconds_used"] >= metrics["post_eval_wallclock_seconds_used"]


def test_parameter_golf_exact_metrics_fall_back_to_cpu_for_mps(tmp_path: Path) -> None:
    assets = _build_pg_assets(tmp_path / "assets_mps_fallback")
    spec = _make_pg_spec(assets)
    trainer = FullWeightTrainer(
        checkpoint_dir=tmp_path / "checkpoints_mps_fallback",
        device="cpu",
        steps=1,
        eval_batches=1,
    )
    trainer.device = torch.device("mps")
    model = EvolutionModel(spec.model)

    metrics = trainer._evaluate_parameter_golf_metrics(
        model,
        spec,
        recurrence_steps={},
    )

    assert metrics["parameter_golf_error"] == 0.0
    assert metrics["val_bpb"] > 0.0
    assert metrics["post_quant_val_bpb"] > 0.0


def test_parameter_golf_metrics_back_off_eval_batch_tokens_on_oom(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assets = _build_pg_assets(tmp_path / "assets_eval_backoff")
    spec = _make_pg_spec(assets)
    trainer = FullWeightTrainer(
        checkpoint_dir=tmp_path / "checkpoints_eval_backoff",
        device="cpu",
        steps=1,
        eval_batches=1,
    )
    model = EvolutionModel(spec.model)
    calls: list[int] = []

    def fake_eval_parameter_golf_val(
        _model: EvolutionModel,
        *,
        batch_tokens: int,
        **_: object,
    ) -> tuple[float, float]:
        calls.append(int(batch_tokens))
        if batch_tokens > int(spec.data.seq_len):
            raise RuntimeError("CUDA out of memory while scoring")
        return 0.5, 1.25

    monkeypatch.setattr(trainer_module, "eval_parameter_golf_val", fake_eval_parameter_golf_val)

    metrics = trainer._evaluate_parameter_golf_metrics(
        model,
        spec,
        recurrence_steps={},
    )

    assert metrics["parameter_golf_error"] == 0.0
    assert metrics["val_bpb"] == pytest.approx(1.25)
    assert metrics["post_quant_val_bpb"] == pytest.approx(1.25)
    assert metrics["parameter_golf_eval_batch_tokens_used"] == float(spec.data.seq_len)
    assert metrics["parameter_golf_post_quant_eval_batch_tokens_used"] == float(spec.data.seq_len)
    assert metrics["parameter_golf_eval_backoff_count"] >= 2.0
    assert max(calls) > int(spec.data.seq_len)


def test_parameter_golf_metrics_record_failure_message(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assets = _build_pg_assets(tmp_path / "assets_eval_failure")
    spec = _make_pg_spec(assets)
    trainer = FullWeightTrainer(
        checkpoint_dir=tmp_path / "checkpoints_eval_failure",
        device="cpu",
        steps=1,
        eval_batches=1,
    )
    model = EvolutionModel(spec.model)

    def fake_eval_parameter_golf_val(
        _model: EvolutionModel,
        **_: object,
    ) -> tuple[float, float]:
        raise ValueError("bad scorer path")

    monkeypatch.setattr(trainer_module, "eval_parameter_golf_val", fake_eval_parameter_golf_val)

    metrics = trainer._evaluate_parameter_golf_metrics(
        model,
        spec,
        recurrence_steps={},
    )

    assert metrics["parameter_golf_error"] == 1.0
    assert trainer.last_parameter_golf_error is not None
    assert "ValueError: bad scorer path" in trainer.last_parameter_golf_error


def test_parameter_golf_rescore_checkpoint_smoke(tmp_path: Path) -> None:
    assets = _build_pg_assets(tmp_path / "assets_rescore")
    spec = _make_pg_spec(assets)
    spec_path = tmp_path / "pg_rescore_spec.yaml"
    api.save_spec(spec, spec_path)

    module = ParameterGolfDataModule(
        spec.parameter_golf,
        seq_len=spec.data.seq_len,
        batch_size=spec.data.batch_size,
        seed=0,
    )
    trainer = FullWeightTrainer(
        checkpoint_dir=tmp_path / "checkpoints_rescore",
        device="cpu",
        steps=2,
        eval_batches=1,
    )
    _, checkpoint = trainer.train(
        Candidate(ident="pg-rescore", spec=spec),
        spec,
        module.batches(max_tokens=spec.train.max_tokens, split="train"),
    )

    summary = rescore_parameter_golf_checkpoint(
        spec_path,
        checkpoint,
        out_path=tmp_path / "rescore_summary.json",
        device="cpu",
        val_batch_tokens=spec.data.seq_len,
        eval_protocol="mid_fidelity",
    )

    assert summary["metrics"]["parameter_golf_error"] == 0.0
    assert summary["metrics"]["val_bpb"] > 0.0
    assert summary["metrics"]["post_quant_val_bpb"] > 0.0
    assert "parameter_golf_error_message" not in summary


def test_parameter_golf_export_writes_workspace(tmp_path: Path) -> None:
    assets = _build_pg_assets(tmp_path)
    spec = _make_official_lane_spec(assets)
    spec_path = tmp_path / "pg_spec.yaml"
    api.save_spec(spec, spec_path)
    official_train_py = _write_official_train_stub(tmp_path / "official_train_gpt.py")

    out_dir = tmp_path / "exported_pg"
    metadata = api.export_parameter_golf_workspace(
        spec_path,
        out_dir,
        official_train_py=official_train_py,
    )
    exported = api.load_spec(out_dir / "parameter_golf_spec.yaml")

    assert (out_dir / "train_gpt.py").exists()
    assert (out_dir / "parameter_golf_export.json").exists()
    assert exported.parameter_golf is not None
    assert exported.parameter_golf.code_bytes == metadata["code_bytes"]
    assert metadata["mode"] == "official"
    assert metadata["exportable"]


def test_parameter_golf_export_rejects_non_official_spec(tmp_path: Path) -> None:
    assets = _build_pg_assets(tmp_path / "assets_export_reject")
    spec = _make_pg_spec(assets)
    spec_path = tmp_path / "pg_spec_reject.yaml"
    api.save_spec(spec, spec_path)
    official_train_py = _write_official_train_stub(tmp_path / "official_train_gpt.py")

    with pytest.raises(ParameterGolfExportError, match="official submission lane"):
        api.export_parameter_golf_workspace(
            spec_path,
            tmp_path / "exported_reject",
            official_train_py=official_train_py,
        )


def test_official_plan_and_export_support_fp16_tied_embedding_patch(tmp_path: Path) -> None:
    assets = _build_pg_assets(tmp_path / "assets_fp16_export")
    spec = _make_official_lane_spec(assets)
    spec.parameter_golf.seed_family = "fp16_tied_embedding"
    spec.parameter_golf.tied_embedding_export_dtype = "fp16"
    spec_path = tmp_path / "pg_fp16_spec.yaml"
    api.save_spec(spec, spec_path)
    official_train_py = _write_official_patch_stub(tmp_path / "official_train_gpt.py")

    plan = build_official_submission_plan(spec, official_train_py=official_train_py)
    assert plan["exportable"]
    assert plan["requires_patch"]
    assert "fp16_tied_embedding_export" in plan["supported_patch_reasons"]

    out_dir = tmp_path / "exported_fp16_pg"
    metadata = api.export_parameter_golf_workspace(
        spec_path,
        out_dir,
        official_train_py=official_train_py,
    )

    rendered = (out_dir / "train_gpt.py").read_text(encoding="utf-8")
    assert metadata["exportable"]
    assert "TIED_EMBED_EXPORT_DTYPE" in rendered
    assert 'name == "tok_emb.weight"' in rendered


def test_parameter_golf_root_env_resolves_relative_paths(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "remote_runs"
    data_dir = root / "parameter_golf" / "fineweb10b_sp1024"
    tok_dir = root / "parameter_golf" / "tokenizers"
    data_dir.mkdir(parents=True)
    tok_dir.mkdir(parents=True)

    assets = _build_pg_assets(tmp_path / "assets")
    train_target = data_dir / "train_000.bin"
    val_target = data_dir / "val_000.bin"
    tok_target = tok_dir / "sp1024.model"
    train_target.write_bytes(Path(str(assets["train_path"])).read_bytes())
    val_target.write_bytes(Path(str(assets["val_path"])).read_bytes())
    tok_target.write_bytes(Path(str(assets["tokenizer_path"])).read_bytes())

    monkeypatch.setenv("TEVO_PARAMETER_GOLF_ROOT", str(root))
    resolved_glob = resolve_parameter_golf_glob("runs/parameter_golf/fineweb10b_sp1024/train_*.bin")
    resolved_tokenizer = resolve_parameter_golf_path("runs/parameter_golf/tokenizers/sp1024.model")

    assert resolved_glob == [train_target]
    assert resolved_tokenizer == tok_target


def test_parameter_golf_default_root_resolves_relative_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "default_remote_runs"
    data_dir = root / "parameter_golf_scout_data"
    tok_dir = root / "parameter_golf" / "tokenizers"
    data_dir.mkdir(parents=True)
    tok_dir.mkdir(parents=True)

    assets = _build_pg_assets(tmp_path / "assets_default_root")
    train_target = data_dir / "train_000000.bin"
    val_target = data_dir / "val_000000.bin"
    tok_target = tok_dir / "sp1024.model"
    train_target.write_bytes(Path(str(assets["train_path"])).read_bytes())
    val_target.write_bytes(Path(str(assets["val_path"])).read_bytes())
    tok_target.write_bytes(Path(str(assets["tokenizer_path"])).read_bytes())

    monkeypatch.delenv("TEVO_PARAMETER_GOLF_ROOT", raising=False)
    monkeypatch.delenv("TEVO_PACKED_ROOT", raising=False)
    monkeypatch.setattr(
        "transformer_evolution_llm.parameter_golf.DEFAULT_PARAMETER_GOLF_ROOTS",
        (root,),
    )

    resolved_glob = resolve_parameter_golf_glob("runs/parameter_golf_scout_data/train_*.bin")
    resolved_tokenizer = resolve_parameter_golf_path("runs/parameter_golf/tokenizers/sp1024.model")

    assert resolved_glob == [train_target]
    assert resolved_tokenizer == tok_target


def test_parameter_golf_preflight_reports_paths_and_size(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "remote_runs"
    data_dir = root / "parameter_golf" / "fineweb10b_sp1024"
    tok_dir = root / "parameter_golf" / "tokenizers"
    data_dir.mkdir(parents=True)
    tok_dir.mkdir(parents=True)

    assets = _build_pg_assets(tmp_path / "assets_preflight")
    (data_dir / "train_000.bin").write_bytes(Path(str(assets["train_path"])).read_bytes())
    (data_dir / "val_000.bin").write_bytes(Path(str(assets["val_path"])).read_bytes())
    (tok_dir / "sp1024.model").write_bytes(Path(str(assets["tokenizer_path"])).read_bytes())

    spec = _make_pg_spec(assets)
    spec.parameter_golf.train_shards_glob = "runs/parameter_golf/fineweb10b_sp1024/train_*.bin"
    spec.parameter_golf.val_shards_glob = "runs/parameter_golf/fineweb10b_sp1024/val_*.bin"
    spec.parameter_golf.tokenizer_path = "runs/parameter_golf/tokenizers/sp1024.model"
    spec_path = tmp_path / "preflight_spec.yaml"
    api.save_spec(spec, spec_path)

    monkeypatch.setenv("TEVO_PARAMETER_GOLF_ROOT", str(root))
    report = preflight_parameter_golf_config(spec_path)

    assert report["train_shard_count"] == 1
    assert report["val_shard_count"] == 1
    assert report["resolved_tokenizer_path"].endswith("sp1024.model")
    assert report["artifact_total_bytes_est"] >= report["artifact_payload_bytes_est"]
    assert report["artifact_payload_bytes_calibrated_est"] > 0
    assert report["artifact_total_bytes_calibrated_est"] > 0
    assert report["train_micro_batch_tokens"] == spec.data.seq_len * spec.data.batch_size
    assert report["grad_accum_steps_est"] == 1
    assert report["eval_protocol"] == "mid_fidelity"
    assert report["report_eval_modes"] == ["standard"]
    assert "official_submission" in report
    assert not report["official_submission"]["exportable"]


def test_build_data_module_for_parameter_golf_spec_uses_pg_loader(tmp_path: Path) -> None:
    assets = _build_pg_assets(tmp_path / "assets_loader")
    spec = _make_pg_spec(assets)

    module = api.build_data_module_for_spec(spec, seed=7)

    assert isinstance(module, ParameterGolfDataModule)
    assert module.batch_size == spec.data.batch_size
    assert module.seq_len == spec.data.seq_len


def test_parameter_golf_benchmark_uses_enough_tokens_for_requested_steps(tmp_path: Path) -> None:
    assets = _build_pg_assets(tmp_path / "assets_budget")
    spec = _make_pg_spec(assets)
    spec_path = tmp_path / "budget_spec.yaml"
    api.save_spec(spec, spec_path)

    summary = run_parameter_golf_benchmark(
        spec_path,
        out_path=tmp_path / "budget_summary.json",
        checkpoint_dir=tmp_path / "budget_checkpoints",
        steps=5,
        eval_batches=1,
        device="cpu",
    )

    assert summary["token_budget"] == 80


def test_parameter_golf_benchmark_respects_target_batch_tokens(tmp_path: Path) -> None:
    assets = _build_pg_assets(tmp_path / "assets_target_budget")
    spec = _make_pg_spec(assets)
    spec.train.batch_tokens = 32
    spec_path = tmp_path / "target_budget_spec.yaml"
    api.save_spec(spec, spec_path)

    summary = run_parameter_golf_benchmark(
        spec_path,
        out_path=tmp_path / "target_budget_summary.json",
        checkpoint_dir=tmp_path / "target_budget_checkpoints",
        steps=5,
        eval_batches=1,
        device="cpu",
    )

    assert summary["preflight"]["grad_accum_steps_est"] == 2
    assert summary["token_budget"] == 160


def test_eval_parameter_golf_val_supports_sliding64(tmp_path: Path) -> None:
    assets = _build_pg_assets(tmp_path / "assets_sliding")
    spec = _make_pg_spec(assets)
    module = ParameterGolfDataModule(spec.parameter_golf, seq_len=8, batch_size=2, seed=0)
    val_tokens = module.validation_tokens()
    model = EvolutionModel(spec.model)
    luts = build_sentencepiece_luts(
        spec.parameter_golf.tokenizer_path, spec.model.head.vocab, torch.device("cpu")
    )

    standard_loss, standard_bpb = eval_parameter_golf_val(
        model,
        seq_len=spec.data.seq_len,
        val_tokens=val_tokens,
        base_bytes_lut=luts[0],
        has_leading_space_lut=luts[1],
        is_boundary_token_lut=luts[2],
        device=torch.device("cpu"),
        batch_tokens=32,
        eval_mode="standard",
        total_eval_tokens=32,
    )
    sliding_loss, sliding_bpb = eval_parameter_golf_val(
        model,
        seq_len=spec.data.seq_len,
        val_tokens=val_tokens,
        base_bytes_lut=luts[0],
        has_leading_space_lut=luts[1],
        is_boundary_token_lut=luts[2],
        device=torch.device("cpu"),
        batch_tokens=32,
        eval_mode="sliding64",
        total_eval_tokens=32,
    )

    assert standard_loss > 0.0
    assert standard_bpb > 0.0
    assert sliding_loss > 0.0
    assert sliding_bpb > 0.0


def test_eval_parameter_golf_val_supports_mps_accumulators(tmp_path: Path) -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")

    assets = _build_pg_assets(tmp_path / "assets_mps")
    spec = _make_pg_spec(assets)
    module = ParameterGolfDataModule(spec.parameter_golf, seq_len=8, batch_size=2, seed=0)
    val_tokens = module.validation_tokens()
    model = EvolutionModel(spec.model).to("mps")
    luts = build_sentencepiece_luts(
        spec.parameter_golf.tokenizer_path,
        spec.model.head.vocab,
        torch.device("mps"),
    )

    loss, bpb = eval_parameter_golf_val(
        model,
        seq_len=spec.data.seq_len,
        val_tokens=val_tokens,
        base_bytes_lut=luts[0],
        has_leading_space_lut=luts[1],
        is_boundary_token_lut=luts[2],
        device=torch.device("mps"),
        batch_tokens=32,
        eval_mode="standard",
        total_eval_tokens=32,
    )

    assert loss > 0.0
    assert bpb > 0.0


def test_artifact_size_calibration_adjusts_estimate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "runs" / "runpod_parameter_golf" / "2026-03-23"
    run_dir.mkdir(parents=True)
    (run_dir / "truth_fake.summary.json").write_text("""
{
  "preflight": {
    "artifact_payload_bytes_est": 2000,
    "artifact_total_bytes_est": 2500,
    "tied_embedding_export_dtype": "int8",
    "export_quant_mode": "int8"
  },
  "metrics": {
    "artifact_payload_bytes": 1000,
    "artifact_total_bytes": 1200
  }
}
""".strip())
    monkeypatch.chdir(tmp_path)
    assets = _build_pg_assets(tmp_path / "assets_calibration")
    spec = _make_pg_spec(assets)

    table = artifact_size_calibration_table()
    raw_payload, raw_total = estimate_artifact_total_bytes_for_spec(spec)
    calibrated_payload, calibrated_total = estimate_calibrated_artifact_total_bytes_for_spec(spec)

    assert table["sample_count"] == 1
    assert calibrated_payload < raw_payload
    assert calibrated_total < raw_total


def test_official_submission_plan_supports_sliding64_eval(tmp_path: Path) -> None:
    assets = _build_pg_assets(tmp_path / "assets_official_sliding")
    spec = _make_official_lane_spec(assets)
    spec.parameter_golf.report_eval_modes = ["standard", "sliding64"]
    official_train_py = _write_official_patch_stub(tmp_path / "train_gpt.py")

    plan = build_official_submission_plan(spec, official_train_py=official_train_py)

    assert "sliding64_eval" in plan["supported_patch_reasons"]
    assert plan["env_overrides"]["EVAL_STRIDE"] == "64"
