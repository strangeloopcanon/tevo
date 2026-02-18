"""Utilities to generate ablated specs for targeted comparisons."""

from __future__ import annotations

from collections.abc import Callable

from .dsl import ArchitectureSpec, DenseFFNConfig, MoEFFNConfig

AblationFn = Callable[[ArchitectureSpec], ArchitectureSpec]

ABLATIONS = [
    "retro_off",
    "local_global_to_sliding",
    "kv_groups_to_dense",
    "rope_theta_default",
    "norm_swap",
    "remove_ssm",
    "moe_to_dense",
    "mask_off",
    "mask_ratio_one",
    "optimizer_to_adamw",
]


def apply_ablation(spec: ArchitectureSpec, name: str) -> ArchitectureSpec:
    """Return a deep-copied spec with a named ablation applied."""
    child = spec.model_copy(deep=True)
    dim = child.model.emb.dim
    if name == "retro_off":
        for block in child.model.blocks:
            block.extras = [
                extra for extra in block.extras if getattr(extra, "type", None) != "retro"
            ]
    elif name == "local_global_to_sliding":
        for block in child.model.blocks:
            if block.attn and (block.attn.sparsity or "none") == "local_global":
                block.attn.sparsity = "sliding"
                block.attn.global_stride = None
    elif name == "kv_groups_to_dense":
        for block in child.model.blocks:
            if block.attn:
                block.attn.kv_groups = 1
    elif name == "rope_theta_default":
        default_theta = child.priors.rope_theta_default
        for block in child.model.blocks:
            if block.attn and block.attn.rope:
                block.attn.rope_theta = default_theta
    elif name == "norm_swap":
        child.model.norm = "rmsnorm" if child.model.norm == "layernorm" else "layernorm"
    elif name == "remove_ssm":
        for block in child.model.blocks:
            block.ssm = None
    elif name == "moe_to_dense":
        for block in child.model.blocks:
            if isinstance(block.ffn, MoEFFNConfig):
                hidden = block.ffn.hidden or (dim * 4)
                block.ffn = DenseFFNConfig(hidden=hidden, activation="swiglu")
    elif name == "mask_off":
        child.train.optimizer.update_filter.mode = "none"
    elif name == "mask_ratio_one":
        child.train.optimizer.update_filter.keep_ratio = 1.0
    elif name == "optimizer_to_adamw":
        child.train.optimizer.name = "adamw"
        child.train.optimizer.muon_momentum = None
        child.train.optimizer.muon_nesterov = True
        child.train.optimizer.muon_ns_steps = 5
    else:
        raise ValueError(f"Unknown ablation '{name}'")
    return child
