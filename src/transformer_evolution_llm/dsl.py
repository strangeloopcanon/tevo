"""Typed configuration DSL for evolutionary architecture search."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import ujson as json
import yaml
from pydantic import (
    BaseModel,
    Field,
    RootModel,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)


class EmbeddingConfig(BaseModel):
    """Token embedding definition."""

    dim: int = Field(gt=0)
    vocab: int | None = Field(default=None, gt=0)
    rope: str | None = None
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    init_std: float = Field(default=0.02, gt=0.0)


class AttentionConfig(BaseModel):
    """Attention block configuration."""

    kind: Literal["MHA", "GQA", "MQA", "LINEAR", "MLA"] = "MHA"
    causal: bool = True
    heads: int = Field(gt=0)
    head_dim: int = Field(gt=0, alias="head_dim")
    rope: str | None = None
    rope_theta: float | None = Field(default=None, gt=0.0)
    rope_factor: float | None = Field(default=None, gt=0.0)
    alibi: bool = False
    linear_feature_map: Literal["elu"] = "elu"
    kv_latent_dim: int | None = Field(default=None, gt=0)
    sw: int | None = Field(default=None, alias="sliding_window")
    kv_groups: int | None = Field(default=None, ge=1)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    # Stability knobs (optional, evolvable)
    qk_norm_max: float | None = Field(default=None, gt=0.0)
    # Architectural innovations
    gating_pos: Literal["none", "output", "value"] = "none"
    gating_op: Literal["dense", "diagonal"] = "dense"
    # Sparsity pattern (optional)
    sparsity: Literal["none", "sliding", "block", "local_global", "dilated", "local_block"] = "none"
    block_size: int | None = Field(default=None, gt=0)
    block_stride: int | None = Field(default=None, gt=0)
    # For local_global pattern: reuse sw as local window; add explicit global stride
    global_stride: int | None = Field(default=None, gt=0)
    dilation: int | None = Field(default=None, gt=0)
    # Selector-based sparsity (e.g., DeepSeek-style)
    selector: Literal["none", "dsa"] = "none"
    selector_topk: int | None = Field(default=None, gt=0)
    selector_heads: int | None = Field(default=None, gt=0)
    selector_dim: int | None = Field(default=None, gt=0)
    selector_rope: Literal["none", "partial", "full"] = "none"
    selector_detach: bool = False
    # Higher-level attention knobs (declarative; not fully wired yet)
    stencil: StencilConfig | None = None
    softmax: SoftmaxConfig | None = None
    projection: ProjectionConfig | None = None
    value_glu: bool | None = None

    model_config = {"populate_by_name": True, "extra": "ignore"}

    @model_validator(mode="before")
    @classmethod
    def migrate_gated_field(cls, data: Any) -> Any:
        if isinstance(data, dict) and "gated" in data:
            gated = data.pop("gated")
            if gated and not data.get("gating_pos"):
                data["gating_pos"] = "output"
            data.setdefault("gating_op", "dense")
        return data

    @model_validator(mode="before")
    @classmethod
    def default_kv_groups_for_kind(cls, data: Any) -> Any:
        """Keep `kind` meaningful even when configs omit `kv_groups`."""
        if not isinstance(data, dict):
            return data
        kind = data.get("kind") or "MHA"
        if "kv_groups" in data and data.get("kv_groups") is not None:
            return data
        heads = data.get("heads")
        if not isinstance(heads, int) or heads <= 0:
            return data
        if kind == "MQA":
            data["kv_groups"] = heads
        elif kind == "GQA":
            data["kv_groups"] = max(1, heads // 4)
        else:
            data["kv_groups"] = 1
        return data

    @property
    def hidden_dim(self) -> int:
        return self.heads * self.head_dim

    @model_validator(mode="after")
    def normalize_selector(self) -> AttentionConfig:
        if self.selector != "none":
            if self.selector_topk is None:
                self.selector_topk = 64
            if self.selector_heads is None:
                self.selector_heads = 1
            self.selector_heads = max(1, min(int(self.selector_heads), int(self.heads)))
            if self.selector_dim is None:
                self.selector_dim = min(int(self.head_dim), 32)
            self.selector_dim = max(1, min(int(self.selector_dim), int(self.head_dim)))
        return self

    @model_validator(mode="after")
    def validate_sparsity_config(self) -> AttentionConfig:
        """Validate sparsity pattern configuration with helpful error messages."""
        sparsity = self.sparsity or "none"

        if sparsity == "sliding" and self.sw is not None and self.sw <= 0:
            raise ValueError(
                f"Sliding window attention requires sw > 0, got sw={self.sw}. "
                "Set sw to a positive integer (e.g., sw=128)."
            )

        if sparsity == "local_global":
            if self.sw is None or self.sw <= 0:
                raise ValueError(
                    f"local_global sparsity requires a positive sliding window (sw > 0). "
                    f"Got sw={self.sw}. Set sw to a positive integer (e.g., sw=128)."
                )
            if self.global_stride is None or self.global_stride <= 0:
                raise ValueError(
                    f"local_global sparsity requires global_stride > 0. "
                    f"Got global_stride={self.global_stride}. Set to a positive integer (e.g., 64)."
                )

        if sparsity == "block":
            if self.block_size is None or self.block_size <= 0:
                raise ValueError(
                    f"block sparsity requires block_size > 0. "
                    f"Got block_size={self.block_size}. Set to a positive integer (e.g., 64)."
                )

        if sparsity == "dilated":
            if self.dilation is None or self.dilation <= 0:
                raise ValueError(
                    f"dilated sparsity requires dilation > 0. "
                    f"Got dilation={self.dilation}. Set to a positive integer (e.g., 2)."
                )

        return self


class KVPolicyConfig(BaseModel):
    """Inference KV-cache policy (for memory/latency-aware evolution).

    This is a **declarative** policy used by static metrics and (optionally) future
    inference paths. Training in this repo is full-sequence (no KV cache), so
    this primarily impacts long-context *inference* constraints.
    """

    cache: Literal["full", "window", "ring", "none", "latent"] = "full"
    window: int | None = Field(default=None, gt=0)
    quant: Literal["none", "fp8", "nf4", "int8"] = "none"
    latent_dim: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_policy(self) -> KVPolicyConfig:
        if self.cache in {"window", "ring"} and self.window is None:
            raise ValueError("kv_policy.cache=window|ring requires kv_policy.window")
        if self.cache == "latent" and self.latent_dim is None:
            raise ValueError("kv_policy.cache=latent requires kv_policy.latent_dim")
        return self


class StencilConfig(BaseModel):
    """Macro attention stencil descriptor (declarative).

    This does not currently override the runtime `sparsity` implementation; it
    exists to express richer sparse patterns (ring/hybrid/cross) for future
    wiring and for downstream tooling.
    """

    kind: Literal[
        "full",
        "local",
        "dilated",
        "block",
        "ring",
        "hybrid",
        "sliding",
        "cross",
    ] = "full"
    window: int | None = Field(default=None, gt=0)
    dilation: int | None = Field(default=None, gt=0)
    block: int | None = Field(default=None, gt=0)
    stride: int | None = Field(default=None, gt=0)
    globals: int | None = Field(default=None, gt=0)
    query: str | None = None
    key: str | None = None

    @model_validator(mode="after")
    def validate_stencil(self) -> StencilConfig:
        if self.kind == "ring" and self.block is None:
            raise ValueError("stencil.kind=ring requires stencil.block")
        if self.kind == "cross" and (self.query is None or self.key is None):
            raise ValueError("stencil.kind=cross requires stencil.query and stencil.key")
        return self


class SoftmaxKernelConfig(BaseModel):
    """Kernelized softmax approximation descriptor (declarative)."""

    name: Literal["favor", "gaussian", "laplace"] | None = None
    features: int | None = Field(default=None, gt=0)
    redraw: int | None = Field(default=None, gt=0)
    orthogonal: bool | None = None


class SoftmaxConfig(BaseModel):
    """Softmax/QK normalization policy (declarative)."""

    type: Literal["standard", "kernel", "scaled"] = "standard"
    qk_scale: float | str | None = None
    qk_norm: Literal["none", "rms", "layer"] = "none"
    softcap: float | None = Field(default=None, gt=0.0)
    kernel: SoftmaxKernelConfig | None = None

    @model_validator(mode="after")
    def validate_softmax(self) -> SoftmaxConfig:
        if self.type == "kernel" and (self.kernel is None or self.kernel.features is None):
            raise ValueError("softmax.type=kernel requires softmax.kernel.features")
        return self


class ProjectionConfig(BaseModel):
    """Low-rank projection descriptor (declarative).

    Intended for LoRA/low-rank attention projections in future wiring.
    """

    type: Literal["low_rank", "none"] = "none"
    rank: int | None = Field(default=None, gt=0)
    shared: bool | None = None

    @model_validator(mode="after")
    def validate_projection(self) -> ProjectionConfig:
        if self.type == "low_rank" and self.rank is None:
            raise ValueError("projection.type=low_rank requires projection.rank")
        return self


class MixRouterConfig(BaseModel):
    """Router knobs for macro mix units (declarative)."""

    topk: int | None = Field(default=None, gt=0)
    temp: float | None = Field(default=None, gt=0.0)
    balance: float | None = Field(default=None, ge=0.0)


class MixerConfig(BaseModel):
    """High-level mixer descriptor (declarative).

    This mirrors upstream-style mixer specification (Attention/Retention/SSM/LongConv),
    but does not currently override per-block runtime components.
    """

    kind: Literal["Attention", "Retention", "SSM", "LongConv"]
    heads: int | None = Field(default=None, gt=0)
    groups: int | None = Field(default=None, ge=1)
    head_dim: int | None = Field(default=None, gt=0)
    stencil: StencilConfig | None = None
    softmax: SoftmaxConfig | None = None
    pos: str | None = None
    chunk: int | None = Field(default=None, gt=0)
    mode: Literal["parallel", "recurrent"] | None = None
    d_state: int | None = Field(default=None, gt=0)
    expand: float | None = Field(default=None, gt=0.0)
    kernel_len: int | None = Field(default=None, gt=0)
    projection: ProjectionConfig | None = None
    value_glu: bool | None = None

    @model_validator(mode="after")
    def validate_mixer(self) -> MixerConfig:
        if self.kind == "Attention" and self.mode not in {None, "parallel"}:
            raise ValueError("Attention mixers must use mode=parallel")
        if self.kind in {"Retention", "SSM"} and self.mode == "recurrent":
            return self
        if self.mode == "recurrent" and self.kind not in {"Retention", "SSM"}:
            raise ValueError("Only Retention/SSM mixers may set mode=recurrent")
        if self.groups is not None and self.heads is not None and self.groups > self.heads:
            raise ValueError("mixer.groups cannot exceed mixer.heads")
        return self


class MixUnitConfig(BaseModel):
    """Compose mixers as single / parallel / routed units (declarative)."""

    kind: Literal["single", "par", "route"] = "single"
    mixer: MixerConfig | None = None
    choices: list[MixerConfig] | None = None
    merge: Literal["Add", "WeightedAdd", "Concat"] | None = None
    router: MixRouterConfig | None = None

    @model_validator(mode="after")
    def validate_mix_unit(self) -> MixUnitConfig:
        if self.kind == "single":
            if self.mixer is None:
                raise ValueError("mix_unit.kind=single requires mix_unit.mixer")
        else:
            if not self.choices:
                raise ValueError(f"mix_unit.kind={self.kind} requires mix_unit.choices")
            if self.kind == "route" and self.router is None:
                raise ValueError("mix_unit.kind=route requires mix_unit.router")
            if self.kind == "par" and self.merge is None:
                raise ValueError("mix_unit.kind=par requires mix_unit.merge")
        return self


class ResidualConfig(BaseModel):
    """Residual/topology descriptor (declarative)."""

    kind: Literal["single", "dual", "deepnet"] = "single"
    pre_ln: bool | None = None
    post_ln: bool | None = None
    scale: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def validate_residual(self) -> ResidualConfig:
        if self.kind == "dual":
            if not (self.pre_ln and self.post_ln):
                raise ValueError("residual.kind=dual requires pre_ln and post_ln enabled")
        if self.kind == "deepnet" and self.scale is None:
            raise ValueError("residual.kind=deepnet requires residual.scale")
        return self


class HyperConnectionsConfig(BaseModel):
    """Hyper-residual stream configuration.

    When enabled (streams > 1), the model maintains multiple residual "lanes"
    and learns per-layer mixing weights that generalize standard residual
    connections. The runtime wiring lives in ``models.py``.
    """

    streams: int = Field(default=1, ge=1, le=16)
    diag_bias: float = Field(
        default=4.0,
        ge=0.0,
        description=(
            "Initialization bias for the diagonal of H_res logits "
            "(higher => closer to identity)."
        ),
    )
    noise_std: float = Field(
        default=1e-3,
        ge=0.0,
        description="Stddev of initialization noise added to hyper mixing logits.",
    )
    update_scale: float = Field(default=1.0, ge=0.0)


class HierarchyLevelConfig(BaseModel):
    """One hierarchy level (declarative)."""

    every: int = Field(ge=1)
    downsample: float | None = Field(default=None, gt=0.0)
    up_proj: bool | None = None


class HierarchyConfig(BaseModel):
    """Hierarchical downsample/upsample descriptor (declarative)."""

    levels: list[HierarchyLevelConfig] = Field(default_factory=list)

    @field_validator("levels")
    @classmethod
    def non_empty(cls, value: list[HierarchyLevelConfig]) -> list[HierarchyLevelConfig]:
        if not value:
            raise ValueError("hierarchy.levels must be non-empty")
        return value


class DepthRouterConfig(BaseModel):
    """Dynamic depth routing descriptor (declarative)."""

    kind: Literal["token", "layer", "none"] = "none"
    budget: float | None = Field(default=None, gt=0.0)
    tau: float | None = Field(default=None, gt=0.0)
    min_layers: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_router(self) -> DepthRouterConfig:
        if self.kind != "none" and self.budget is None:
            raise ValueError("depth_router.kind!=none requires depth_router.budget")
        return self


class CondSourceConfig(BaseModel):
    """Conditioning input source (declarative)."""

    kind: Literal["pool_mlp", "segment"] = "pool_mlp"
    hidden: int | None = Field(default=None, gt=0, alias="H")
    segment: int | None = Field(default=None, ge=0)

    model_config = {"populate_by_name": True}

    @model_validator(mode="before")
    @classmethod
    def migrate_kind(cls, data: Any) -> Any:
        if isinstance(data, dict):
            kind = data.get("kind")
            if kind == "pool-mlp":
                data = dict(data)
                data["kind"] = "pool_mlp"
        return data


class CondOpConfig(BaseModel):
    """One conditioning operation (declarative)."""

    where: Literal[
        "pre_mixer",
        "post_mixer",
        "ln",
        "proj_q",
        "proj_v",
        "q",
        "v",
        "token",
    ]
    op: Literal["film", "add", "scale", "lora"]
    share: Literal["global", "per_channel", "per_head"] | None = None
    r: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_op(self) -> CondOpConfig:
        if self.op == "lora" and self.r is None:
            raise ValueError("cond.ops op=lora requires r")
        return self


class CondRegConfig(BaseModel):
    """Conditioning regularizer descriptor (declarative)."""

    kind: Literal["freebits", "none"] = "none"
    kappa: float | None = Field(default=None, gt=0.0)


class CondConfig(BaseModel):
    """Conditioning policy (declarative)."""

    source: CondSourceConfig | None = None
    reg: CondRegConfig | None = None
    ops: list[CondOpConfig] | None = None


class MacroConfig(BaseModel):
    """Optional macro-architecture descriptors (declarative; not fully wired)."""

    mix_unit: MixUnitConfig | None = None
    residual: ResidualConfig | None = None
    hierarchy: HierarchyConfig | None = None
    depth_router: DepthRouterConfig | None = None
    cond: CondConfig | None = None


class DenseFFNConfig(BaseModel):
    """Standard feed-forward block."""

    type: Literal["dense"] = "dense"
    hidden: int = Field(gt=0)
    activation: Literal["silu", "gelu", "relu", "swiglu"] = "swiglu"
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)


class MoEFFNConfig(BaseModel):
    """Mixture-of-Experts feed-forward block."""

    type: Literal["moe"] = "moe"
    hidden: int = Field(gt=0)
    n_experts: int = Field(gt=1)
    k: int = Field(default=2, ge=1)
    capacity_factor: float = Field(default=1.2, gt=0.0)
    balance: float = Field(default=0.05, ge=0.0)
    shared: int = Field(default=0, ge=0)
    router_temperature: float | None = Field(default=None, gt=0.0)
    router_type: Literal["softmax", "sigmoid"] = "softmax"
    router_bias_detached: bool = False
    shared_expert: bool = False
    router_aux_weight: float | None = Field(default=None, ge=0.0)
    router_lb_weight: float | None = Field(default=None, ge=0.0)
    drop_policy: Literal["none", "greedy"] = "none"
    experts: list[MoEExpertConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_moe_topk(self) -> MoEFFNConfig:
        """Ensure k does not exceed n_experts."""
        if self.k > self.n_experts:
            raise ValueError(
                f"MoE top-k ({self.k}) cannot exceed n_experts ({self.n_experts}). "
                f"Set k <= {self.n_experts} or increase n_experts to at least {self.k}."
            )
        return self


class MoEDenseExpertConfig(BaseModel):
    """Dense expert configuration for MoE blocks."""

    type: Literal["dense"] = "dense"
    hidden: int | None = Field(default=None, gt=0)
    activation: Literal["silu", "gelu", "relu", "swiglu"] = "swiglu"
    hops: int = Field(default=1, ge=1)


class MoESSMExpertConfig(BaseModel):
    """SSM expert configuration for MoE blocks."""

    type: Literal["ssm"] = "ssm"
    ssm: SSMConfig
    hops: int = Field(default=1, ge=1)


class MoECustomExpertConfig(BaseModel):
    """Custom expert backed by a plugin or CustomModule."""

    type: Literal["custom"] = "custom"
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


MoEExpertConfig = Annotated[
    MoEDenseExpertConfig | MoESSMExpertConfig | MoECustomExpertConfig,
    Field(discriminator="type"),
]


FfnConfig = Annotated[DenseFFNConfig | MoEFFNConfig, Field(discriminator="type")]


class SSMConfig(BaseModel):
    """State-space model parallel branch."""

    kind: Literal["mamba2", "ssm"] = "mamba2"
    d_state: int = Field(gt=0)
    d_conv: int = Field(gt=0)
    dt_rank: int = Field(gt=0)
    chunk: int = Field(gt=0)
    gate: float = Field(default=0.1, ge=0.0, le=1.0)


class HeadConfig(BaseModel):
    """LM head configuration."""

    tie_embeddings: bool = True
    vocab: int = Field(gt=0)


class RetroConfig(BaseModel):
    """Retro-attention style memory augmentation."""

    type: Literal["retro"] = "retro"
    memory_tokens: int = Field(default=1024, gt=0)
    stride: int = Field(default=64, gt=0)
    aggregator: Literal["mean", "attention", "gate"] = "gate"
    gating_weight: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def coerce_int_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for key in ("memory_tokens", "stride"):
                if key in data and isinstance(data[key], float):
                    data[key] = int(data[key])
        return data


class GatedModuleConfig(BaseModel):
    """Lightweight gate that blends multiple submodules."""

    type: Literal["gated"] = "gated"
    targets: list[str] = Field(default_factory=lambda: ["attn", "ffn"])
    init_weight: float = Field(default=0.1, ge=0.0, le=1.0)
    learnable: bool = True


class CustomModuleConfig(BaseModel):
    """Escape hatch for experimental modules without schema friction."""

    type: Literal["custom"] = "custom"
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class AssociativeMemoryConfig(BaseModel):
    """Causal associative memory (fast-weights / linear-attention style)."""

    type: Literal["assoc_memory"] = "assoc_memory"
    heads: int = Field(gt=0)
    head_dim: int = Field(gt=0)
    feature_map: Literal["elu"] = "elu"
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    gating_weight: float = Field(default=0.1, ge=0.0, le=1.0)


class MemoryTokensConfig(BaseModel):
    """Learnable persistent memory tokens (cross-attention style)."""

    type: Literal["memory_tokens"] = "memory_tokens"
    tokens: int = Field(gt=0)
    heads: int = Field(gt=0)
    head_dim: int = Field(gt=0)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    init_std: float = Field(default=0.02, gt=0.0)
    gating_weight: float = Field(default=0.1, ge=0.0, le=1.0)


class ChunkMemoryConfig(BaseModel):
    """Causal chunk-summary memory (downsampled trailing-window summaries)."""

    type: Literal["chunk_memory"] = "chunk_memory"
    chunk_size: int = Field(gt=0)
    stride: int | None = Field(default=None, gt=0)
    heads: int = Field(gt=0)
    head_dim: int = Field(gt=0)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    gating_weight: float = Field(default=0.1, ge=0.0, le=1.0)


class LookupMemoryConfig(BaseModel):
    """Conditional lookup from a learned key/value table."""

    type: Literal["lookup_memory"] = "lookup_memory"
    entries: int = Field(gt=0)
    topk: int = Field(default=4, gt=0)
    key_dim: int | None = Field(default=None, gt=0)
    value_dim: int | None = Field(default=None, gt=0)
    temperature: float = Field(default=1.0, gt=0.0)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    chunk_size: int = Field(default=1024, gt=0)
    lookup_device: Literal["model", "cpu"] = "model"
    gating_weight: float = Field(default=0.1, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_lookup_topk(self) -> LookupMemoryConfig:
        """Ensure topk does not exceed entries."""
        if self.topk > self.entries:
            raise ValueError(
                f"LookupMemory top-k ({self.topk}) cannot exceed entries ({self.entries}). "
                f"Set topk <= {self.entries} or increase entries to at least {self.topk}."
            )
        return self


class BranchRouterConfig(BaseModel):
    """Learned router that mixes branch outputs inside a block."""

    type: Literal["branch_router"] = "branch_router"
    targets: list[str] = Field(default_factory=lambda: ["attn", "ffn", "ssm", "memory"])
    router_type: Literal["token", "sequence"] = "token"
    hidden: int | None = Field(default=None, gt=0)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    temperature: float = Field(default=1.0, gt=0.0)


class LayerScaleConfig(BaseModel):
    """Per-channel residual scaling (LayerScale/ReZero-like stability knob)."""

    type: Literal["layer_scale"] = "layer_scale"
    targets: list[str] = Field(default_factory=lambda: ["attn", "ffn"])
    init: float = Field(default=1e-5, gt=0.0)
    learnable: bool = True


ExtraModuleConfig = Annotated[
    RetroConfig
    | GatedModuleConfig
    | CustomModuleConfig
    | AssociativeMemoryConfig
    | MemoryTokensConfig
    | ChunkMemoryConfig
    | LookupMemoryConfig
    | BranchRouterConfig
    | LayerScaleConfig,
    Field(discriminator="type"),
]


class BlockConfig(BaseModel):
    """Single transformer block definition."""

    name: str | None = None
    attn: AttentionConfig | None = None
    ffn: FfnConfig | None = None
    ssm: SSMConfig | None = None
    extras: list[ExtraModuleConfig] = Field(default_factory=list)

    def describe(self) -> dict[str, Any]:
        return {
            "attn": self.attn.model_dump(mode="python") if self.attn else None,
            "ffn": self.ffn.model_dump(mode="python") if self.ffn else None,
            "ssm": self.ssm.model_dump(mode="python") if self.ssm else None,
            "extras": [extra.model_dump(mode="python") for extra in self.extras],
        }


class RecurrenceConfig(BaseModel):
    """Wrap a contiguous block range in a recurrent loop."""

    start: int
    end: int
    adapter: Literal["linear", "gated"] = "linear"
    adapter_dim: int | None = None
    concat_prelude: bool = True
    init_state: Literal["zeros", "noise"] = "zeros"
    noise_std: float = Field(default=0.02, ge=0.0)
    train_recurrence: int = Field(default=1, ge=1)
    max_train_recurrence: int = Field(default=4, ge=1)
    curriculum_fraction: float = Field(default=0.25, ge=0.0, le=1.0)
    test_recurrences: list[int] = Field(default_factory=lambda: [1, 2, 4, 8])

    @field_validator("end")
    @classmethod
    def valid_range(cls, value: int, info: ValidationInfo) -> int:
        start = info.data.get("start", 0)
        if value <= start:
            raise ValueError(
                f"Recurrence end ({value}) must be greater than start ({start}). "
                f"Set end > {start} to define a valid block range for recurrence."
            )
        return value

    @model_validator(mode="after")
    def validate_recurrence_params(self) -> RecurrenceConfig:
        """Validate recurrence configuration with helpful error messages."""
        if self.train_recurrence > self.max_train_recurrence:
            raise ValueError(
                f"train_recurrence ({self.train_recurrence}) cannot exceed "
                f"max_train_recurrence ({self.max_train_recurrence}). "
                f"Set train_recurrence <= {self.max_train_recurrence}."
            )
        return self


class TrainSchedule(BaseModel):
    """Training hyper parameters shared by all stages."""

    lr: float = Field(gt=0)
    warmup: int = Field(default=2000, ge=0)
    clip: float = Field(default=1.0, gt=0.0)
    bf16: bool = True
    grad_checkpoint: bool = Field(default=True, alias="grad_ckpt")
    max_tokens: int | None = Field(default=None, ge=0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    seed: int = 0
    # MoE routing auxiliaries (optional)
    router_lb_coeff: float = Field(default=0.0, ge=0.0)
    router_entropy_coeff: float = Field(default=0.0, ge=0.0)
    entropy_threshold: float = Field(default=0.5, ge=0.0)
    entropy_patience: int = Field(default=3, ge=1)
    instability_threshold: float = Field(default=5.0, ge=0.0)
    no_improve_patience: int = Field(default=20, ge=1)
    improvement_tolerance: float = Field(default=1e-3, ge=0.0)
    ppl_stop_threshold: float | None = Field(default=2.5, ge=0.0)
    init_checkpoint: str | None = None
    # Optional synthetic long-context evaluation (passkey probe)
    passkey_eval_steps: int = Field(default=0, ge=0)
    passkey_eval_batches: int = Field(default=8, ge=1)
    passkey_eval_seq_len: int | None = Field(default=None, gt=0)
    passkey_eval_min_distance: int = Field(default=128, ge=0)
    passkey_eval_lr: float | None = Field(default=None, gt=0.0)
    passkey_eval_batch_size: int | None = Field(default=None, gt=0)
    passkey_eval_vocab_limit: int | None = Field(
        default=None, gt=0, description="Optional cap on synthetic passkey token range."
    )
    # Downstream probes: quick capability checks beyond perplexity
    downstream_probes: bool = Field(
        default=False,
        description="If True, run simple code/math/recall probes as part of evaluation.",
    )
    downstream_probe_examples: int = Field(
        default=5,
        ge=1,
        description="Number of examples per downstream probe.",
    )
    # Optional speedrun-style probe: measure time/tokens to hit a target eval loss/ppl.
    speedrun_eval_interval: int = Field(
        default=0,
        ge=0,
        description="If >0, run a periodic eval check every N training steps for speedrun metrics.",
    )
    speedrun_eval_batches: int = Field(
        default=4, ge=1, description="How many eval batches to use per speedrun check."
    )
    speedrun_target_loss: float | None = Field(
        default=None, gt=0.0, description="Target eval loss threshold for the speedrun probe."
    )
    speedrun_target_ppl: float | None = Field(
        default=None, gt=0.0, description="Target eval perplexity threshold for the speedrun probe."
    )
    # Multi-threshold speedrun: track progress toward multiple thresholds and compute AUC
    speedrun_multi_thresholds: list[float] | None = Field(
        default=None,
        description=(
            "List of perplexity thresholds to track (e.g., [2500, 2000, 1500, 1000]). "
            "If provided, computes AUC of tokens-to-threshold curve."
        ),
    )

    model_config = {"populate_by_name": True}
    optimizer: OptimizerConfig = Field(default_factory=lambda: OptimizerConfig())

    @model_validator(mode="after")
    def _validate_speedrun(self) -> TrainSchedule:
        if self.speedrun_eval_interval > 0:
            if self.speedrun_target_loss is None and self.speedrun_target_ppl is None:
                raise ValueError(
                    "speedrun_eval_interval requires speedrun_target_loss or speedrun_target_ppl"
                )
            if self.speedrun_target_loss is not None and self.speedrun_target_ppl is not None:
                raise ValueError("Provide only one of speedrun_target_loss or speedrun_target_ppl")
        return self


class OptimizerConfig(BaseModel):
    """Optimizer selection and hyperparameters (non-evolvable by default)."""

    name: Literal["adamw", "lion"] = "adamw"
    lr: float | None = Field(default=None, gt=0.0)
    betas: tuple[float, float] | None = None
    eps: float | None = Field(default=None, gt=0.0)
    weight_decay: float | None = Field(default=None, ge=0.0)


class DatasetShard(BaseModel):
    """Weighted dataset slice for quick probes."""

    name: str
    split: str = "train"
    weight: float = Field(gt=0.0)
    cache_path: str | None = None
    revision: str | None = None


class DataConfig(BaseModel):
    """Collection of dataset shards."""

    tokenizer: str
    hf_revision: str = "main"
    streaming: bool = Field(
        default=True,
        description="Use streaming datasets (avoids full downloads; requires network).",
    )
    packed: bool = Field(
        default=False,
        description="Use packed-token stream files instead of per-shard text datasets.",
    )
    packed_train_path: str | None = Field(
        default=None, description="Path to packed-token train file (e.g., train.bin)."
    )
    packed_val_path: str | None = Field(
        default=None, description="Path to packed-token val file (e.g., val.bin)."
    )
    packed_split: Literal["train", "val"] | None = Field(
        default=None, description="Override split when reading packed tokens."
    )
    packed_dtype: Literal["uint16", "int32"] = Field(
        default="uint16", description="Packed-token dtype on disk."
    )
    seq_len: int = Field(default=2048, gt=0)
    batch_size: int = Field(default=1, gt=0)
    workers: int = Field(default=2, ge=0)
    shards: list[DatasetShard] = Field(default_factory=list)
    eval_shards: list[DatasetShard] = Field(default_factory=list)
    eval_tokens: int | None = Field(default=None, gt=0)
    healing_shards: list[DatasetShard] = Field(default_factory=list)
    healing_tokens: int | None = Field(default=None, gt=0)

    @field_validator("shards")
    @classmethod
    def non_empty(cls, value: list[DatasetShard], info: ValidationInfo) -> list[DatasetShard]:
        if info.data.get("packed"):
            return value
        if not value:
            raise ValueError("Provide at least one dataset shard.")
        return value

    @model_validator(mode="after")
    def validate_packed_paths(self) -> DataConfig:
        if self.packed and not self.packed_train_path:
            raise ValueError("packed_train_path is required when packed=true.")
        return self

    @property
    def total_weight(self) -> float:
        return sum(shard.weight for shard in self.shards)


class ModelConfig(BaseModel):
    """Full architecture definition."""

    name: str = "candidate"
    emb: EmbeddingConfig
    blocks: list[BlockConfig]
    head: HeadConfig
    norm: Literal["layernorm", "rmsnorm"] = "layernorm"
    hyper: HyperConnectionsConfig | None = None
    kv_policy: KVPolicyConfig | None = None
    macro: MacroConfig | None = None
    recurrences: list[RecurrenceConfig] = Field(default_factory=list)

    @field_validator("blocks")
    @classmethod
    def non_zero_layers(cls, value: list[BlockConfig]) -> list[BlockConfig]:
        if not value:
            raise ValueError("Model requires at least one block.")
        return value

    @property
    def n_layers(self) -> int:
        return len(self.blocks)

    def moe_block_count(self) -> int:
        return sum(1 for block in self.blocks if isinstance(block.ffn, MoEFFNConfig))


class EvolutionConfig(BaseModel):
    """Tunable parameters governing mutation + selection."""

    rung0_thresholds: dict[str, float] = Field(
        default_factory=lambda: {"gate_entropy_min": 1.0, "gate_entropy_max": 3.0}
    )
    rung1_tokens: int = 200_000
    rung2_tokens: int = 1_000_000
    population: int = 12
    topk_keep: float = Field(default=0.33, gt=0.0, le=1.0)
    crossover_prob: float = Field(default=0.2, ge=0.0, le=1.0)
    parent_selection: Literal[
        "weighted", "pareto_uniform", "lexicase", "epsilon_lexicase", "map_elites"
    ] = "weighted"
    # Epsilon-lexicase: tolerance band for near-ties (as fraction of objective range)
    epsilon_lexicase_epsilon: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Tolerance band for epsilon-lexicase selection (fraction of objective range).",
    )
    pareto_objectives: list[str] = Field(
        default_factory=lambda: ["ppl_code", "ppl_math", "long_recall", "throughput", "ram"]
    )
    objectives: dict[str, Literal["max", "min"]] | None = None
    composite_metrics: list[CompositeMetricConfig] = Field(default_factory=list)
    # Optional promotion rung for high-budget candidates (live mode only)
    promotion_prob: float = Field(default=0.0, ge=0.0, le=1.0)
    promotion_min_layers: int = Field(default=0, ge=0)
    promotion_min_moe_blocks: int = Field(default=0, ge=0)
    promotion_steps_multiplier: float = Field(default=1.0, ge=1.0)
    promotion_tokens_multiplier: float = Field(default=1.0, ge=1.0)
    promotion_min_router_entropy: float = Field(default=0.0, ge=0.0)
    promotion_min_recurrence_gain: float = Field(default=0.0)
    promotion_max_instability: float | None = Field(default=None, ge=0.0)
    archive_max_elites: int = Field(default=0, ge=0)
    structural_elite_k: int = Field(default=2, ge=0)
    structural_elite_weights: dict[str, float] | None = None
    adaptive_mutation: bool = False
    # Context-aware mutation: prioritize mutations based on candidate architecture state
    context_aware_mutation: bool = Field(
        default=False,
        description="If True, prioritize mutations based on candidate architecture state.",
    )
    adaptive_mutation_eta: float = Field(default=0.1, gt=0.0, le=1.0)
    adaptive_mutation_min_weight: float = Field(default=0.05, gt=0.0)
    adaptive_mutation_max_weight: float = Field(default=5.0, gt=0.0)
    weight_inheritance: Literal["parent", "init", "scratch"] = "parent"
    # Adaptive rung budgets: promote/demote candidates based on improvement rate
    adaptive_rung_budget: bool = Field(
        default=False,
        description="If True, adjust rung budgets based on improvement rate during training.",
    )
    adaptive_rung_fast_promote_threshold: float = Field(
        default=0.15,
        ge=0.0,
        description="Improvement rate threshold for early promotion to rung2.",
    )
    adaptive_rung_slow_stop_threshold: float = Field(
        default=0.02,
        ge=0.0,
        description="Improvement rate threshold for early stopping (plateau).",
    )
    # Operator invention v1: learn weights for mutation templates and optionally persist/promote
    template_learning: bool = False
    template_learning_eta: float = Field(default=0.2, gt=0.0, le=1.0)
    template_learning_min_weight: float = Field(default=0.05, gt=0.0)
    template_learning_max_weight: float = Field(default=5.0, gt=0.0)
    template_learning_max_templates: int = Field(default=128, ge=0)
    template_learning_save_every: int = Field(default=20, ge=1)
    template_learning_save_path: str = "configs/mutation_templates.yaml"
    template_learning_promote_min_delta: float = Field(default=0.0)

    @model_validator(mode="after")
    def validate_rung_tokens(self) -> EvolutionConfig:
        if self.rung1_tokens > self.rung2_tokens:
            raise ValueError("evolution.rung1_tokens must be <= evolution.rung2_tokens")
        return self


class PriorConfig(BaseModel):
    """Soft priors to inform budgets and a gentle manifold without constraining search."""

    tokens_per_param: float = Field(default=4.0, gt=0.0)
    window_scale: float = Field(default=6.0, gt=0.0)  # for sqrt(seq_len) * scale
    rope_theta_default: float = Field(default=10000.0, gt=0.0)
    prior_weight: float = Field(default=0.0, ge=0.0)  # selection weight; default unused
    compute_penalty_weight: float = Field(default=0.0, ge=0.0)


class ArchitectureSpec(BaseModel):
    """Top-level DSL entity."""

    model: ModelConfig
    train: TrainSchedule
    data: DataConfig
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    priors: PriorConfig = Field(default_factory=PriorConfig)

    def summary(self) -> dict[str, Any]:
        """Return a JSON-friendly summary for logging."""
        hyper = getattr(self.model, "hyper", None)
        hyper_streams = int(getattr(hyper, "streams", 1) or 1) if hyper is not None else 1
        return {
            "name": self.model.name,
            "layers": self.model.n_layers,
            "moe_blocks": self.model.moe_block_count(),
            "seq_len": self.data.seq_len,
            "population": self.evolution.population,
            "custom_modules": sum(len(block.extras) for block in self.model.blocks),
            "hyper_streams": hyper_streams,
        }


class ArchitectureSpecList(RootModel[list[ArchitectureSpec]]):
    """Helper for multi-candidate configs."""


def load_architecture_spec(path: str | Path) -> ArchitectureSpec:
    """Load a spec from YAML or JSON."""
    path = Path(path)
    data: Any
    text = path.read_text()
    if path.suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    try:
        return ArchitectureSpec(**data)
    except ValidationError as exc:
        raise ValueError(f"Invalid config {path}") from exc


def save_architecture_spec(spec: ArchitectureSpec, path: str | Path) -> None:
    """Persist a spec as YAML based on file suffix."""
    path = Path(path)
    if path.suffix in {".yaml", ".yml"}:
        path.write_text(yaml.safe_dump(spec.model_dump(mode="python"), sort_keys=False))
    else:
        path.write_text(json.dumps(spec.model_dump(mode="python"), indent=2))


class CompositeMetricConfig(BaseModel):
    """Define a derived metric based on existing metrics."""

    name: str
    op: Literal["ratio", "product", "weighted_sum"] = "ratio"
    numerator: str | None = None
    denominator: str | None = None
    terms: dict[str, float] = Field(default_factory=dict)
    epsilon: float = 1e-6
