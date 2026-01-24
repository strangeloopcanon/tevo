"""Static and dynamic evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .dsl import (
    ArchitectureSpec,
    AssociativeMemoryConfig,
    BlockConfig,
    BranchRouterConfig,
    ChunkMemoryConfig,
    CustomModuleConfig,
    DenseFFNConfig,
    GatedModuleConfig,
    HyperConnectionsConfig,
    KVPolicyConfig,
    LayerScaleConfig,
    LookupMemoryConfig,
    MemoryTokensConfig,
    MoEFFNConfig,
    RetroConfig,
)


def _attn_hidden(block: BlockConfig) -> int:
    if not block.attn:
        return 0
    return block.attn.heads * block.attn.head_dim


def estimate_params(spec: ArchitectureSpec) -> float:
    """Crude parameter count estimator."""
    vocab_value = spec.model.emb.vocab or spec.model.head.vocab
    if vocab_value is None:
        msg = "Vocabulary size must be specified on embedding or head."
        raise ValueError(msg)
    vocab = int(vocab_value)
    d_model = int(spec.model.emb.dim)
    params = float(d_model * vocab)  # embeddings
    for block in spec.model.blocks:
        if block.attn:
            heads = int(block.attn.heads)
            head_dim = int(block.attn.head_dim)
            kv_groups = max(1, int(block.attn.kv_groups or 1))
            kv_heads = max(1, heads // kv_groups)
            q_out = heads * head_dim
            kind = str(getattr(block.attn, "kind", "MHA") or "MHA").upper()
            if kind == "MLA":
                latent = int(getattr(block.attn, "kv_latent_dim", 0) or 0)
                if latent <= 0:
                    latent = kv_heads * head_dim
                params += d_model * q_out  # q_proj
                params += d_model * latent  # kv_down
                params += latent * (2 * kv_heads * head_dim)  # kv_up
                params += q_out * d_model  # out_proj
            else:
                qkv_out = (heads + 2 * kv_heads) * head_dim
                params += d_model * qkv_out  # qkv
                params += q_out * d_model  # out_proj

        if isinstance(block.ffn, DenseFFNConfig):
            hidden = int(block.ffn.hidden)
            act = str(getattr(block.ffn, "activation", "silu") or "silu").lower()
            if act == "swiglu":
                params += 3 * d_model * hidden
            else:
                params += 2 * d_model * hidden
        elif isinstance(block.ffn, MoEFFNConfig):
            hidden = int(block.ffn.hidden)
            n_experts = int(block.ffn.n_experts)
            params += float(d_model * n_experts)  # router
            params += float(n_experts * (3 * d_model * hidden))  # experts (swiglu)
            shared = max(
                int(getattr(block.ffn, "shared", 0) or 0),
                1 if getattr(block.ffn, "shared_expert", False) else 0,
            )
            if shared > 0:
                params += float(3 * d_model * hidden)  # single shared expert module

        if block.ssm:
            inner = max(1, int(getattr(block.ssm, "d_state", d_model) or d_model))
            k = max(1, int(getattr(block.ssm, "d_conv", 1) or 1))
            params += float(d_model * inner)  # in_proj
            params += float(inner * inner * k)  # conv1d (groups=1)
            params += float(inner * d_model)  # out_proj

        for extra in block.extras:
            if isinstance(extra, RetroConfig):
                continue
            if isinstance(extra, GatedModuleConfig):
                params += float(len(extra.targets))
            elif isinstance(extra, CustomModuleConfig):
                inner = int(extra.params.get("dim", d_model))
                params += float(d_model * inner + inner * d_model)
            elif isinstance(extra, AssociativeMemoryConfig):
                inner = int(extra.heads) * int(extra.head_dim)
                params += float(4 * d_model * inner)
            elif isinstance(extra, MemoryTokensConfig):
                inner = int(extra.heads) * int(extra.head_dim)
                params += float(2 * d_model * inner)  # q_proj + out_proj
                params += float(2 * int(extra.tokens) * inner)  # mem_kv
            elif isinstance(extra, ChunkMemoryConfig):
                inner = int(extra.heads) * int(extra.head_dim)
                params += float(4 * d_model * inner)  # q,k,v,o projections
            elif isinstance(extra, LookupMemoryConfig):
                entries = int(extra.entries)
                key_dim = int(getattr(extra, "key_dim", None) or d_model)
                value_dim = int(getattr(extra, "value_dim", None) or d_model)
                params += float(d_model * key_dim)  # q_proj (no bias)
                params += float(entries * key_dim)  # keys
                params += float(entries * value_dim)  # values
                if value_dim != d_model:
                    params += float(value_dim * d_model)  # out_proj (no bias)
            elif isinstance(extra, BranchRouterConfig):
                n_targets = max(1, len(extra.targets))
                router_hidden = getattr(extra, "hidden", None)
                if router_hidden:
                    h = int(router_hidden)
                    params += float(d_model * h + h * n_targets)
                else:
                    params += float(d_model * n_targets)
            elif isinstance(extra, LayerScaleConfig):
                params += float(len(extra.targets) * d_model)
    if not getattr(spec.model.head, "tie_embeddings", True):
        params += float(spec.model.head.vocab * d_model)
    hyper = getattr(spec.model, "hyper", None)
    if isinstance(hyper, HyperConnectionsConfig) and hyper.streams > 1:
        n = int(hyper.streams)
        layers = int(spec.model.n_layers)
        params += float(layers * (n * n + 2 * n) + n)
    return params


def kv_bytes_per_token(spec: ArchitectureSpec) -> float:
    dtype_bytes = 2.0  # fp16/bf16 baseline
    kv_policy = getattr(spec.model, "kv_policy", None)
    if isinstance(kv_policy, KVPolicyConfig):
        if kv_policy.cache == "none":
            return 0.0
        quant = str(getattr(kv_policy, "quant", "none") or "none").lower()
        if quant == "fp8":
            dtype_bytes = 1.0
        elif quant == "int8":
            dtype_bytes = 1.0
        elif quant == "nf4":
            dtype_bytes = 0.5
    total = 0.0
    for block in spec.model.blocks:
        if not block.attn:
            continue
        kind = str(getattr(block.attn, "kind", "MHA") or "MHA").upper()
        if kind == "LINEAR":
            continue
        if (
            kv_policy is not None
            and isinstance(kv_policy, KVPolicyConfig)
            and kv_policy.cache == "latent"
        ):
            latent = int(getattr(kv_policy, "latent_dim", 0) or 0)
            if latent > 0:
                total += 2.0 * float(latent) * dtype_bytes
                continue
        if kind == "MLA":
            latent = int(getattr(block.attn, "kv_latent_dim", 0) or 0)
            if latent > 0:
                total += 2.0 * float(latent) * dtype_bytes
                continue
        kv_groups = int(block.attn.kv_groups or 1)
        kv_heads = max(1, int(block.attn.heads) // max(1, kv_groups))
        # KV cache stores keys + values: 2 tensors per token.
        total += 2.0 * float(kv_heads) * float(block.attn.head_dim) * dtype_bytes
    return float(total)


def throughput_proxy(spec: ArchitectureSpec, seq_len: int) -> float:
    """Rough tokens/second proxy."""
    hidden = sum(_attn_hidden(block) for block in spec.model.blocks)
    denom = max(1, hidden * spec.model.n_layers * seq_len)
    return 1e9 / denom


@dataclass
class StaticCheckResult:
    ok: bool
    metrics: dict[str, float]
    reasons: list[str]


class StaticChecker:
    """Rung 0 filtering without any training."""

    def __init__(
        self,
        max_params: float = 8e9,
        max_kv_bytes: float = 48_000,
        min_throughput: float = 1.0,
    ) -> None:
        self.max_params = max_params
        self.max_kv_bytes = max_kv_bytes
        self.min_throughput = min_throughput

    def run(self, spec: ArchitectureSpec) -> StaticCheckResult:
        params = estimate_params(spec)
        kv = kv_bytes_per_token(spec)
        tps = throughput_proxy(spec, spec.data.seq_len)
        reasons: list[str] = []
        kv_policy = getattr(spec.model, "kv_policy", None)
        if kv_policy is not None:
            cache = str(getattr(kv_policy, "cache", "") or "")
            window = getattr(kv_policy, "window", None)
            latent_dim = getattr(kv_policy, "latent_dim", None)
            if cache in {"window", "ring"} and (window is None or int(window) <= 0):
                reasons.append("kv_policy.cache=window|ring requires kv_policy.window > 0")
            if cache == "latent" and (latent_dim is None or int(latent_dim) <= 0):
                reasons.append("kv_policy.cache=latent requires kv_policy.latent_dim > 0")
        # Sanity bounds for new knobs
        for block in spec.model.blocks:
            if block.attn:
                kind = str(getattr(block.attn, "kind", "MHA") or "MHA").upper()
                if kind == "LINEAR":
                    if getattr(block.attn, "selector", "none") != "none":
                        reasons.append("LINEAR attention does not support selector sparsity")
                    if getattr(block.attn, "sparsity", "none") != "none":
                        reasons.append("LINEAR attention does not support sparsity patterns")
                    if getattr(block.attn, "sw", None) is not None:
                        reasons.append("LINEAR attention does not support sliding_window")
                    if bool(getattr(block.attn, "alibi", False)):
                        reasons.append("LINEAR attention does not support ALiBi")
                if kind == "MLA":
                    latent = getattr(block.attn, "kv_latent_dim", None)
                    if latent is None or int(latent) <= 0:
                        reasons.append("MLA attention requires kv_latent_dim > 0")
                if getattr(block.attn, "selector", "none") != "none":
                    topk = getattr(block.attn, "selector_topk", None)
                    if topk is None or int(topk) <= 0:
                        reasons.append("selector requires selector_topk > 0")
                    elif int(topk) > spec.data.seq_len:
                        reasons.append("selector_topk exceeds seq_len")
                    sel_heads = getattr(block.attn, "selector_heads", None)
                    if sel_heads is None or int(sel_heads) <= 0:
                        reasons.append("selector requires selector_heads > 0")
                    elif int(sel_heads) > int(block.attn.heads):
                        reasons.append("selector_heads cannot exceed heads")
                    sel_dim = getattr(block.attn, "selector_dim", None)
                    if sel_dim is None or int(sel_dim) <= 0:
                        reasons.append("selector requires selector_dim > 0")
                    elif int(sel_dim) > int(block.attn.head_dim):
                        reasons.append("selector_dim cannot exceed head_dim")
            if (
                block.attn
                and block.attn.qk_norm_max is not None
                and not (0.0 < block.attn.qk_norm_max <= 50.0)
            ):
                reasons.append("qk_norm_max outside (0, 50]")
            if block.attn and block.attn.kv_groups is not None:
                if block.attn.kv_groups <= 0:
                    reasons.append("kv_groups must be >=1")
                if block.attn.kv_groups > block.attn.heads:
                    reasons.append("kv_groups cannot exceed heads")
                if (
                    0 < block.attn.kv_groups <= block.attn.heads
                    and block.attn.heads % block.attn.kv_groups != 0
                ):
                    reasons.append("heads must be divisible by kv_groups")
            if (
                block.attn
                and block.attn.block_size is not None
                and block.attn.block_size > spec.data.seq_len
            ):
                reasons.append("block_size exceeds seq_len")
            if block.attn and block.attn.block_stride is not None:
                if block.attn.block_stride <= 0 or block.attn.block_stride > spec.data.seq_len:
                    reasons.append("block_stride must be in (0, seq_len]")
            if block.attn and block.attn.sw is not None and block.attn.sw <= 0:
                reasons.append("sliding_window must be > 0")
            if block.attn and block.attn.sw is not None and block.attn.sw > spec.data.seq_len:
                reasons.append("sliding_window exceeds seq_len")
            if block.attn and block.attn.sparsity == "local_global":
                if block.attn.sw is None or block.attn.sw <= 0:
                    reasons.append("local_global requires positive sliding_window (sw)")
                if (
                    block.attn.global_stride is None
                    or block.attn.global_stride <= 0
                    or block.attn.global_stride > spec.data.seq_len
                ):
                    reasons.append("local_global requires 0 < global_stride <= seq_len")
            if block.attn and block.attn.sparsity == "local_block":
                if block.attn.sw is None or block.attn.sw <= 0:
                    reasons.append("local_block requires sliding_window (sw)")
                if block.attn.block_size is None or block.attn.block_size <= 0:
                    reasons.append("local_block requires block_size > 0")
                if (
                    block.attn.block_stride is None
                    or block.attn.block_stride <= 0
                    or block.attn.block_stride > spec.data.seq_len
                ):
                    reasons.append("local_block requires 0 < block_stride <= seq_len")
            if block.attn and block.attn.dilation is not None and block.attn.dilation <= 0:
                reasons.append("dilation must be > 0 when using dilated sparsity")
            for extra in block.extras:
                if isinstance(extra, MemoryTokensConfig):
                    if extra.tokens > spec.data.seq_len * 4:
                        reasons.append("memory_tokens.tokens unusually large for seq_len")
                elif isinstance(extra, ChunkMemoryConfig):
                    if extra.chunk_size > spec.data.seq_len:
                        reasons.append("chunk_memory.chunk_size exceeds seq_len")
                    if extra.stride is not None and extra.stride > spec.data.seq_len:
                        reasons.append("chunk_memory.stride exceeds seq_len")
                elif isinstance(extra, LookupMemoryConfig):
                    if extra.entries <= 0:
                        reasons.append("lookup_memory.entries must be > 0")
                    if extra.topk <= 0:
                        reasons.append("lookup_memory.topk must be > 0")
                    if extra.topk > extra.entries:
                        reasons.append("lookup_memory.topk cannot exceed entries")
                    if extra.key_dim is not None and extra.key_dim <= 0:
                        reasons.append("lookup_memory.key_dim must be > 0 when set")
                    if extra.value_dim is not None and extra.value_dim <= 0:
                        reasons.append("lookup_memory.value_dim must be > 0 when set")
                    if extra.temperature <= 0.0:
                        reasons.append("lookup_memory.temperature must be > 0")
                    if extra.chunk_size <= 0:
                        reasons.append("lookup_memory.chunk_size must be > 0")
                elif isinstance(extra, BranchRouterConfig):
                    if not extra.targets:
                        reasons.append("branch_router requires non-empty targets")
                    if extra.temperature <= 0.0:
                        reasons.append("branch_router.temperature must be > 0")
                elif isinstance(extra, LayerScaleConfig):
                    if not (0.0 < extra.init <= 1.0):
                        reasons.append("layer_scale.init must be in (0, 1]")
        if params > self.max_params:
            reasons.append(f"params {params/1e9:.2f}B exceeds {self.max_params/1e9:.2f}B limit")
        if kv > self.max_kv_bytes:
            reasons.append(f"kv bytes/token {kv:.0f} > limit {self.max_kv_bytes}")
        if tps < self.min_throughput:
            reasons.append(f"throughput proxy {tps:.2f} below {self.min_throughput}")
        metrics = {
            "params": params,
            "kv_bytes_per_token": kv,
            "throughput_proxy": tps,
        }
        return StaticCheckResult(ok=not reasons, metrics=metrics, reasons=reasons)


def merge_metrics(existing: dict[str, float], updates: dict[str, Any]) -> dict[str, float]:
    merged = dict(existing)
    for key, value in updates.items():
        if isinstance(value, (int, float)):
            merged[key] = float(value)
    return merged
