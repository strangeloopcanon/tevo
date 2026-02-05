"""Function-preserving mutation utilities."""

from __future__ import annotations

import copy
import random
from collections.abc import Callable
from typing import Any, Literal

from pydantic import ValidationError

from .dsl import (
    ArchitectureSpec,
    AssociativeMemoryConfig,
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
    RecurrenceConfig,
    RetroConfig,
    SSMConfig,
)
from .template_mutation import apply_template_mutation_with_name

MutationResult = ArchitectureSpec | tuple[str, ArchitectureSpec]
MutationFn = Callable[[ArchitectureSpec, random.Random], MutationResult]


class MutationError(Exception):
    """Raised when a mutation produces an invalid spec."""

    def __init__(self, mutation_name: str, message: str, diff: str | None = None) -> None:
        self.mutation_name = mutation_name
        self.diff = diff
        super().__init__(f"{mutation_name} produced invalid spec: {message}")


def diff_specs(before: ArchitectureSpec, after: dict[str, Any]) -> str:
    """Return a human-readable diff of two specs (before as spec, after as dict).

    Shows only the changed fields to help debug mutation issues.
    """
    before_dict = before.model_dump(mode="python")
    changes: list[str] = []

    def _diff(path: str, old: Any, new: Any) -> None:
        if isinstance(old, dict) and isinstance(new, dict):
            all_keys = set(old.keys()) | set(new.keys())
            for key in sorted(all_keys):
                _diff(f"{path}.{key}" if path else key, old.get(key), new.get(key))
        elif isinstance(old, list) and isinstance(new, list):
            if len(old) != len(new):
                changes.append(f"{path}: list length {len(old)} -> {len(new)}")
            for i, (o, n) in enumerate(zip(old, new, strict=False)):
                _diff(f"{path}[{i}]", o, n)
            # Show added items
            for i in range(len(old), len(new)):
                changes.append(f"{path}[{i}]: (added) {new[i]}")
        elif old != new:
            old_str = str(old)[:50] if old is not None else "None"
            new_str = str(new)[:50] if new is not None else "None"
            changes.append(f"{path}: {old_str} -> {new_str}")

    _diff("", before_dict, after)
    return "\n".join(changes[:20]) if changes else "(no changes detected)"


def validate_mutation_result(
    mutation_name: str, before: ArchitectureSpec, after_dict: dict[str, Any]
) -> ArchitectureSpec:
    """Validate a mutated spec and raise MutationError with helpful info if invalid."""
    try:
        return ArchitectureSpec(**after_dict)
    except ValidationError as e:
        diff = diff_specs(before, after_dict)
        # Extract the most useful error message from Pydantic
        errors = e.errors()
        if errors:
            first_error = errors[0]
            loc = ".".join(str(x) for x in first_error.get("loc", []))
            msg = first_error.get("msg", str(e))
            error_msg = f"At {loc}: {msg}" if loc else msg
        else:
            error_msg = str(e)
        raise MutationError(mutation_name, error_msg, diff) from e


def clone_spec(spec: ArchitectureSpec) -> ArchitectureSpec:
    return spec.model_copy(deep=True)


def dense_to_moe(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    child = clone_spec(spec)
    idx = rng.randrange(len(child.model.blocks))
    block = child.model.blocks[idx]
    if isinstance(block.ffn, MoEFFNConfig):
        return child
    dense = block.ffn
    if not isinstance(dense, DenseFFNConfig):
        msg = "dense_to_moe expects a dense FFN block."
        raise TypeError(msg)
    # Start with a modest expert count so MoE is viable under local-resource caps.
    # Evolution can later scale experts up via tune_experts.
    n_experts = int(rng.choice([8, 12, 16]))
    block.ffn = MoEFFNConfig(
        hidden=dense.hidden,
        n_experts=n_experts,
        k=2,
        balance=0.05,
        shared=1,
    )
    return child


def mutate_topk(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    child = clone_spec(spec)
    moe_blocks = [b for b in child.model.blocks if isinstance(b.ffn, MoEFFNConfig)]
    if not moe_blocks:
        return child
    target = rng.choice(moe_blocks)
    if not isinstance(target.ffn, MoEFFNConfig):
        msg = "mutate_topk requires a MoE FFN."
        raise TypeError(msg)
    target.ffn.k = rng.choice([1, 2, 4])
    target.ffn.capacity_factor = rng.uniform(1.0, 1.5)
    return child


def tune_experts(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    """Jitter MoE expert count and selector top-k to explore capacity/compute tradeoffs."""
    child = clone_spec(spec)
    moe_blocks = [b for b in child.model.blocks if isinstance(b.ffn, MoEFFNConfig)]
    if moe_blocks:
        target = rng.choice(moe_blocks)
        if isinstance(target.ffn, MoEFFNConfig):
            # adjust expert count within a modest range
            candidates = [n for n in [8, 12, 16, 24, 32, 48] if n != target.ffn.n_experts]
            if candidates:
                target.ffn.n_experts = rng.choice(candidates)
            # adjust k bounded by n_experts
            possible_k = [k for k in [1, 2, 4, 8] if k <= target.ffn.n_experts]
            if possible_k:
                target.ffn.k = rng.choice(possible_k)
    # selector top-k tweak on a random attention block
    attn_blocks = [b for b in child.model.blocks if b.attn is not None]
    if attn_blocks:
        b = rng.choice(attn_blocks)
        if b.attn and getattr(b.attn, "selector", "none") != "none":
            b.attn.selector_topk = int(rng.choice([24, 32, 48, 64, 96, 128]))
    return child


def tune_router(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    """Jitter MoE router behaviour (sigmoid/softmax, bias detaching, shared expert)."""
    child = clone_spec(spec)
    moe_blocks = [b for b in child.model.blocks if isinstance(b.ffn, MoEFFNConfig)]
    if not moe_blocks:
        return child
    target = rng.choice(moe_blocks)
    if not isinstance(target.ffn, MoEFFNConfig):
        return child
    target.ffn.router_type = rng.choice(["softmax", "sigmoid"])
    target.ffn.router_bias_detached = rng.choice([True, False])
    target.ffn.shared_expert = rng.choice([True, False])
    target.ffn.k = rng.choice(
        [min(target.ffn.n_experts, k) for k in [2, 4, 8] if k <= target.ffn.n_experts]
        or [target.ffn.k]
    )
    return child


def shift_moe(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    child = clone_spec(spec)
    blocks = child.model.blocks
    moe_idx = [i for i, b in enumerate(blocks) if isinstance(b.ffn, MoEFFNConfig)]
    if not moe_idx:
        return child
    src = rng.choice(moe_idx)
    dst = max(0, min(len(blocks) - 1, src + rng.choice([-2, -1, 1, 2])))
    if src == dst:
        return child
    block = copy.deepcopy(blocks[src])
    del blocks[src]
    blocks.insert(dst, block)
    return child


def make_gqa(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    child = clone_spec(spec)
    idx = rng.randrange(len(child.model.blocks))
    block = child.model.blocks[idx]
    if not block.attn:
        return child
    attn = block.attn
    attn.kind = "GQA"
    attn.kv_groups = max(1, attn.heads // 4)
    return child


def toggle_precision(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    child = clone_spec(spec)
    child.train.bf16 = not child.train.bf16
    return child


def toggle_optimizer(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    child = clone_spec(spec)
    name = str(getattr(child.train.optimizer, "name", "adamw") or "adamw").lower()
    child.train.optimizer.name = "lion" if name == "adamw" else "adamw"
    # Reset optimizer-specific moment knobs; keep lr/wd overrides if already set.
    child.train.optimizer.betas = None
    child.train.optimizer.eps = None
    return child


def tune_optimizer(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    """Jitter optimizer hyperparameters to explore training dynamics."""
    child = clone_spec(spec)
    opt = child.train.optimizer
    name = str(getattr(opt, "name", "adamw") or "adamw").lower()

    base_lr = float(opt.lr if opt.lr is not None else child.train.lr)
    if rng.random() < 0.2:
        opt.lr = None
    else:
        factor = float(rng.choice([0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0]))
        opt.lr = float(max(1e-6, min(base_lr * factor, 5e-3)))

    base_wd = float(opt.weight_decay if opt.weight_decay is not None else child.train.weight_decay)
    if rng.random() < 0.2:
        opt.weight_decay = None
    else:
        if rng.random() < 0.1:
            opt.weight_decay = 0.0
        else:
            factor = float(rng.choice([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]))
            opt.weight_decay = float(max(0.0, min(base_wd * factor, 0.2)))

    if name == "adamw":
        opt.betas = rng.choice(
            [
                None,
                (0.9, 0.95),
                (0.9, 0.98),
                (0.9, 0.99),
                (0.9, 0.999),
                (0.95, 0.999),
            ]
        )
        opt.eps = rng.choice([None, 1e-8, 1e-6, 1e-5])
    else:
        opt.betas = rng.choice([None, (0.9, 0.99), (0.9, 0.98), (0.95, 0.98)])
        opt.eps = None

    return child


def tune_warmup(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    child = clone_spec(spec)
    current = int(getattr(child.train, "warmup", 0) or 0)
    options = [0, 5, 10, 20, 40, 80, 160, 320]
    choices = [v for v in options if v != current]
    child.train.warmup = int(rng.choice(choices or [current]))
    return child


def tune_clip(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    child = clone_spec(spec)
    choices = [0.5, 0.8, 1.0, 1.5, 2.0, 4.0]
    current = float(getattr(child.train, "clip", 1.0) or 1.0)
    # Avoid a no-op when possible.
    options = [v for v in choices if abs(v - current) > 1e-9]
    child.train.clip = float(rng.choice(options or [current]))
    return child


def insert_retro_module(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    child = clone_spec(spec)
    block = rng.choice(child.model.blocks)
    retro = RetroConfig(
        memory_tokens=int(rng.choice([512, 1024, 2048])),
        stride=int(rng.choice([32, 64, 128])),
        aggregator=rng.choice(["mean", "attention", "gate"]),
        gating_weight=rng.uniform(0.1, 0.9),
    )
    block.extras.append(retro)
    return child


def insert_custom_module(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    child = clone_spec(spec)
    block = rng.choice(child.model.blocks)
    module = CustomModuleConfig(
        name=f"exp-{rng.randrange(10_000)}",
        params={
            "dim": rng.choice([256, 512, 1024]),
            "activation": rng.choice(["silu", "gelu", "relu"]),
            "notes": "auto-generated",
        },
    )
    block.extras.append(module)
    return child


def insert_graph_module(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    child = clone_spec(spec)
    block = rng.choice(child.model.blocks)
    module = CustomModuleConfig(
        name="graph_module",
        params={"ops": [{"op": "rmsnorm"}, {"op": "mlp", "hidden_mult": 2.0}]},
    )
    block.extras.append(module)
    return child


def template_mutation(spec: ArchitectureSpec, rng: random.Random) -> tuple[str, ArchitectureSpec]:
    template_name, mutated = apply_template_mutation_with_name(spec, rng)
    safe = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in template_name)
    return f"tpl_{safe}", mutated


def insert_assoc_memory(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    child = clone_spec(spec)
    block = rng.choice(child.model.blocks)
    if any(isinstance(extra, AssociativeMemoryConfig) for extra in block.extras):
        return tune_assoc_memory(child, rng)
    dim = int(child.model.emb.dim)
    head_dim = int(rng.choice([16, 32, 64]))
    heads = int(rng.choice([2, 4, 8]))
    if heads * head_dim > 2 * dim:
        heads = max(1, dim // max(1, head_dim))
    block.extras.append(
        AssociativeMemoryConfig(
            heads=max(1, heads),
            head_dim=head_dim,
            feature_map="elu",
            dropout=rng.choice([0.0, 0.0, 0.1]),
            gating_weight=rng.uniform(0.05, 0.4),
        )
    )
    return child


def tune_assoc_memory(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    child = clone_spec(spec)
    candidates: list[AssociativeMemoryConfig] = []
    for block in child.model.blocks:
        for extra in block.extras:
            if isinstance(extra, AssociativeMemoryConfig):
                candidates.append(extra)
    if not candidates:
        return insert_assoc_memory(child, rng)
    mem = rng.choice(candidates)
    mem.gating_weight = max(0.0, min(1.0, float(mem.gating_weight + rng.uniform(-0.1, 0.1))))
    if rng.random() < 0.5:
        mem.dropout = max(
            0.0, min(0.5, float(getattr(mem, "dropout", 0.0) + rng.uniform(-0.1, 0.1)))
        )
    if rng.random() < 0.4:
        mem.head_dim = int(rng.choice([16, 32, 64]))
    if rng.random() < 0.4:
        mem.heads = int(rng.choice([2, 4, 8]))
    return child


def insert_memory_tokens(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    child = clone_spec(spec)
    block = rng.choice(child.model.blocks)
    existing = next((e for e in block.extras if isinstance(e, MemoryTokensConfig)), None)
    if existing is not None:
        return tune_memory_tokens(child, rng)
    dim = int(child.model.emb.dim)
    head_dim = int(rng.choice([16, 32, 64]))
    heads = int(rng.choice([1, 2, 4, 8]))
    if heads * head_dim > 2 * dim:
        heads = max(1, dim // max(1, head_dim))
    tokens = int(rng.choice([8, 16, 32, 64]))
    block.extras.append(
        MemoryTokensConfig(
            tokens=tokens,
            heads=max(1, heads),
            head_dim=head_dim,
            dropout=rng.choice([0.0, 0.0, 0.1]),
            init_std=0.02,
            gating_weight=rng.uniform(0.05, 0.3),
        )
    )
    return child


def tune_memory_tokens(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    child = clone_spec(spec)
    candidates: list[MemoryTokensConfig] = []
    for block in child.model.blocks:
        for extra in block.extras:
            if isinstance(extra, MemoryTokensConfig):
                candidates.append(extra)
    if not candidates:
        return insert_memory_tokens(child, rng)
    mem = rng.choice(candidates)
    mem.gating_weight = max(0.0, min(1.0, float(mem.gating_weight + rng.uniform(-0.1, 0.1))))
    mem.dropout = max(0.0, min(0.5, float(getattr(mem, "dropout", 0.0) + rng.uniform(-0.1, 0.1))))
    if rng.random() < 0.4:
        mem.tokens = int(rng.choice([4, 8, 16, 32, 64, 128]))
    if rng.random() < 0.3:
        mem.head_dim = int(rng.choice([16, 32, 64]))
    if rng.random() < 0.3:
        mem.heads = int(rng.choice([1, 2, 4, 8]))
    return child


def insert_chunk_memory(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    child = clone_spec(spec)
    block = rng.choice(child.model.blocks)
    if any(isinstance(extra, ChunkMemoryConfig) for extra in block.extras):
        return tune_chunk_memory(child, rng)
    dim = int(child.model.emb.dim)
    seq_len = int(child.data.seq_len)
    chunk_size = int(rng.choice([32, 64, 96, 128, 192, 256]))
    chunk_size = max(8, min(chunk_size, seq_len))
    stride = int(rng.choice([chunk_size, max(1, chunk_size // 2)]))
    head_dim = int(rng.choice([16, 32, 64]))
    heads = int(rng.choice([1, 2, 4, 8]))
    if heads * head_dim > 2 * dim:
        heads = max(1, dim // max(1, head_dim))
    block.extras.append(
        ChunkMemoryConfig(
            chunk_size=chunk_size,
            stride=stride,
            heads=max(1, heads),
            head_dim=head_dim,
            dropout=rng.choice([0.0, 0.0, 0.1]),
            gating_weight=rng.uniform(0.05, 0.3),
        )
    )
    return child


def tune_chunk_memory(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    child = clone_spec(spec)
    candidates: list[ChunkMemoryConfig] = []
    for block in child.model.blocks:
        for extra in block.extras:
            if isinstance(extra, ChunkMemoryConfig):
                candidates.append(extra)
    if not candidates:
        return insert_chunk_memory(child, rng)
    mem = rng.choice(candidates)
    seq_len = int(child.data.seq_len)
    if rng.random() < 0.5:
        chunk_size = int(rng.choice([16, 32, 64, 96, 128, 192, 256]))
        mem.chunk_size = max(8, min(chunk_size, seq_len))
    if rng.random() < 0.5:
        stride = int(rng.choice([mem.chunk_size, max(1, int(mem.chunk_size // 2))]))
        mem.stride = max(1, min(stride, seq_len))
    mem.gating_weight = max(0.0, min(1.0, float(mem.gating_weight + rng.uniform(-0.1, 0.1))))
    mem.dropout = max(0.0, min(0.5, float(getattr(mem, "dropout", 0.0) + rng.uniform(-0.1, 0.1))))
    if rng.random() < 0.3:
        mem.head_dim = int(rng.choice([16, 32, 64]))
    if rng.random() < 0.3:
        mem.heads = int(rng.choice([1, 2, 4, 8]))
    return child


def insert_lookup_memory(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    child = clone_spec(spec)
    block = rng.choice(child.model.blocks)
    if any(isinstance(extra, LookupMemoryConfig) for extra in block.extras):
        return tune_lookup_memory(child, rng)
    dim = int(child.model.emb.dim)
    entries = int(rng.choice([64, 128, 256, 512, 1024]))
    topk = int(rng.choice([2, 4, 8]))
    key_dim = int(rng.choice([max(16, dim // 4), max(16, dim // 2), dim]))
    value_dim = int(rng.choice([max(16, dim // 2), dim]))
    block.extras.append(
        LookupMemoryConfig(
            entries=entries,
            topk=min(topk, entries),
            key_dim=key_dim,
            value_dim=value_dim,
            temperature=rng.uniform(0.7, 1.5),
            dropout=rng.choice([0.0, 0.0, 0.1]),
            chunk_size=int(rng.choice([256, 512, 1024, 2048])),
            lookup_device=rng.choice(["model", "model", "cpu"]),
            gating_weight=rng.uniform(0.05, 0.3),
        )
    )
    return child


def tune_lookup_memory(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    child = clone_spec(spec)
    candidates: list[LookupMemoryConfig] = []
    for block in child.model.blocks:
        for extra in block.extras:
            if isinstance(extra, LookupMemoryConfig):
                candidates.append(extra)
    if not candidates:
        return insert_lookup_memory(child, rng)
    mem = rng.choice(candidates)
    mem.gating_weight = max(0.0, min(1.0, float(mem.gating_weight + rng.uniform(-0.1, 0.1))))
    mem.dropout = max(0.0, min(0.5, float(getattr(mem, "dropout", 0.0) + rng.uniform(-0.1, 0.1))))
    if rng.random() < 0.5:
        mem.temperature = max(0.1, float(getattr(mem, "temperature", 1.0) + rng.uniform(-0.3, 0.3)))
    if rng.random() < 0.4:
        mem.topk = int(rng.choice([1, 2, 4, 8, 16]))
        mem.topk = max(1, min(mem.topk, int(mem.entries)))
    if rng.random() < 0.3:
        mem.entries = int(rng.choice([64, 128, 256, 512, 1024, 2048]))
        mem.entries = max(1, mem.entries)
        mem.topk = max(1, min(int(mem.topk), int(mem.entries)))
    if rng.random() < 0.3:
        mem.chunk_size = int(rng.choice([128, 256, 512, 1024, 2048]))
    if rng.random() < 0.1:
        mem.lookup_device = rng.choice(["model", "cpu"])
    return child


def toggle_branch_router(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    child = clone_spec(spec)
    block = rng.choice(child.model.blocks)
    existing = [extra for extra in block.extras if isinstance(extra, BranchRouterConfig)]
    if existing:
        block.extras = [
            extra for extra in block.extras if not isinstance(extra, BranchRouterConfig)
        ]
        return child
    targets = ["attn", "ffn", "ssm", "memory"]
    rng.shuffle(targets)
    block.extras.append(
        BranchRouterConfig(
            targets=targets,
            router_type=rng.choice(["token", "sequence"]),
            hidden=rng.choice([None, 64, 128, 256]),
            dropout=rng.choice([0.0, 0.0, 0.1]),
            temperature=rng.uniform(0.7, 1.5),
        )
    )
    return child


def tune_branch_router(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    child = clone_spec(spec)
    routers: list[BranchRouterConfig] = []
    for block in child.model.blocks:
        for extra in block.extras:
            if isinstance(extra, BranchRouterConfig):
                routers.append(extra)
    if not routers:
        return toggle_branch_router(child, rng)
    router = rng.choice(routers)
    router.temperature = max(0.1, min(5.0, float(router.temperature) * rng.uniform(0.8, 1.25)))
    router.dropout = max(
        0.0, min(0.5, float(getattr(router, "dropout", 0.0) + rng.uniform(-0.1, 0.1)))
    )
    router.router_type = rng.choice(["token", "sequence"])
    if rng.random() < 0.4:
        router.hidden = rng.choice([None, 32, 64, 128, 256])
    if rng.random() < 0.3:
        targets = list(router.targets or ["attn", "ffn", "ssm", "memory"])
        if rng.random() < 0.5 and "memory" in targets:
            targets.remove("memory")
        elif "memory" not in targets:
            targets.append("memory")
        if targets:
            router.targets = targets
    return child


def insert_layer_scale(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    child = clone_spec(spec)
    block = rng.choice(child.model.blocks)
    if any(isinstance(extra, LayerScaleConfig) for extra in block.extras):
        return tune_layer_scale(child, rng)
    targets = rng.sample(["attn", "ffn", "ssm", "memory"], k=rng.choice([1, 2, 3]))
    init = rng.choice([1e-6, 1e-5, 1e-4, 1e-3])
    block.extras.append(LayerScaleConfig(targets=targets, init=float(init), learnable=True))
    return child


def tune_layer_scale(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    child = clone_spec(spec)
    layerscales: list[LayerScaleConfig] = []
    for block in child.model.blocks:
        for extra in block.extras:
            if isinstance(extra, LayerScaleConfig):
                layerscales.append(extra)
    if not layerscales:
        return insert_layer_scale(child, rng)
    ls = rng.choice(layerscales)
    ls.init = float(max(1e-8, min(0.5, float(ls.init) * rng.uniform(0.5, 2.0))))
    if rng.random() < 0.3:
        ls.learnable = rng.choice([True, False])
    if rng.random() < 0.3:
        targets = list(ls.targets)
        candidate = rng.choice(["attn", "ffn", "ssm", "memory"])
        if candidate in targets and len(targets) > 1:
            targets.remove(candidate)
        elif candidate not in targets:
            targets.append(candidate)
        ls.targets = targets
    return child


def toggle_alibi(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    child = clone_spec(spec)
    blocks = [
        b
        for b in child.model.blocks
        if b.attn is not None and str(getattr(b.attn, "kind", "MHA") or "MHA").upper() != "LINEAR"
    ]
    if not blocks:
        return child
    block = rng.choice(blocks)
    if block.attn is None:
        return child
    block.attn.alibi = not bool(getattr(block.attn, "alibi", False))
    return child


def toggle_linear_attention(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    child = clone_spec(spec)
    blocks = [b for b in child.model.blocks if b.attn is not None]
    if not blocks:
        return child
    block = rng.choice(blocks)
    if block.attn is None:
        return child
    kind = str(getattr(block.attn, "kind", "MHA") or "MHA").upper()
    if kind == "LINEAR":
        block.attn.kind = rng.choice(["MHA", "GQA", "MQA"])
        if block.attn.kind == "MQA":
            block.attn.kv_groups = int(block.attn.heads)
        elif block.attn.kind == "GQA":
            block.attn.kv_groups = max(1, int(block.attn.heads) // 4)
        else:
            block.attn.kv_groups = 1
        return child

    block.attn.kind = "LINEAR"
    block.attn.causal = True
    block.attn.alibi = False
    block.attn.sparsity = "none"
    block.attn.sw = None
    block.attn.block_size = None
    block.attn.block_stride = None
    block.attn.global_stride = None
    block.attn.dilation = None
    block.attn.selector = "none"
    block.attn.selector_topk = None
    block.attn.selector_heads = None
    block.attn.selector_dim = None
    block.attn.selector_rope = "none"
    block.attn.selector_detach = False
    return child


def toggle_mla_attention(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    child = clone_spec(spec)
    blocks = [
        b
        for b in child.model.blocks
        if b.attn is not None and str(getattr(b.attn, "kind", "MHA") or "MHA").upper() != "LINEAR"
    ]
    if not blocks:
        return child
    block = rng.choice(blocks)
    if block.attn is None:
        return child
    kind = str(getattr(block.attn, "kind", "MHA") or "MHA").upper()
    if kind == "MLA":
        block.attn.kind = rng.choice(["MHA", "GQA", "MQA"])
        block.attn.kv_latent_dim = None
        if block.attn.kind == "MQA":
            block.attn.kv_groups = int(block.attn.heads)
        elif block.attn.kind == "GQA":
            block.attn.kv_groups = max(1, int(block.attn.heads) // 4)
        else:
            block.attn.kv_groups = 1
        return child

    block.attn.kind = "MLA"
    kv_groups = max(1, int(block.attn.kv_groups or 1))
    kv_heads = max(1, int(block.attn.heads) // kv_groups)
    full = max(1, kv_heads * int(block.attn.head_dim))
    block.attn.kv_latent_dim = int(rng.choice([max(1, full // 4), max(1, full // 2), full]))
    return child


def toggle_gated_mix(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    child = clone_spec(spec)
    block = rng.choice(child.model.blocks)
    gated = next((extra for extra in block.extras if isinstance(extra, GatedModuleConfig)), None)
    if gated:
        gated.init_weight = 1.0 - gated.init_weight
        gated.targets = list(reversed(gated.targets))
    else:
        block.extras.append(
            GatedModuleConfig(
                targets=rng.sample(["attn", "ffn", "ssm"], k=rng.choice([2, 3])),
                init_weight=rng.uniform(0.05, 0.5),
            )
        )
    return child


def toggle_ssm(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    child = clone_spec(spec)
    idx = rng.randrange(len(child.model.blocks))
    block = child.model.blocks[idx]
    if block.ssm:
        block.ssm = None
    else:
        block.ssm = SSMConfig(kind="mamba2", d_state=16, d_conv=4, dt_rank=8, chunk=128, gate=0.1)
    return child


def tune_kv(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    child = clone_spec(spec)
    # pick a block with attention
    blocks = [b for b in child.model.blocks if b.attn is not None]
    if not blocks:
        return child
    b = rng.choice(blocks)
    if b.attn is None:
        return child
    heads = max(1, int(b.attn.heads))
    choices = [k for k in [1, 2, 4, 8] if k <= heads and heads % k == 0]
    if not choices:
        return child
    b.attn.kv_groups = int(rng.choice(choices))
    return child


def toggle_kv_policy(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    """Toggle an inference KV policy (cache mode + quantization).

    This does not change training compute directly (training is full-sequence),
    but it changes the static KV-memory proxy used for gating/selection.
    """

    child = clone_spec(spec)
    policy = getattr(child.model, "kv_policy", None)
    if policy is None and rng.random() < 0.9:
        child.model.kv_policy = KVPolicyConfig(
            cache="window",
            window=int(rng.choice([1024, 2048, 4096, 8192])),
            quant=rng.choice(["none", "nf4", "fp8", "int8"]),
        )
        return child
    if policy is None:
        child.model.kv_policy = KVPolicyConfig(cache="full", quant=rng.choice(["none", "fp8"]))
        return child

    # Occasionally clear the policy entirely.
    if rng.random() < 0.25:
        child.model.kv_policy = None
        return child

    cache = rng.choice(["full", "window", "ring", "none", "latent"])
    quant = rng.choice(["none", "nf4", "fp8", "int8"])
    if cache in {"window", "ring"}:
        child.model.kv_policy = KVPolicyConfig(
            cache=cache,
            window=int(rng.choice([512, 1024, 2048, 4096, 8192, 16384])),
            quant=quant,
        )
        return child
    if cache == "latent":
        child.model.kv_policy = KVPolicyConfig(
            cache="latent",
            latent_dim=int(rng.choice([64, 96, 128, 192, 256, 384, 512])),
            quant=quant,
        )
        return child
    child.model.kv_policy = KVPolicyConfig(cache=cache, quant=quant)
    return child


def toggle_hyper_connections(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    """Toggle multi-lane hyper-residual streams on/off."""

    child = clone_spec(spec)
    current = getattr(child.model, "hyper", None)
    if isinstance(current, HyperConnectionsConfig) and current.streams > 1:
        child.model.hyper = None
        return child
    streams = int(rng.choice([2, 3, 4]))
    child.model.hyper = HyperConnectionsConfig(
        streams=streams,
        diag_bias=float(rng.choice([2.0, 4.0, 6.0])),
        noise_std=float(rng.choice([0.0, 1e-4, 1e-3])),
        update_scale=float(rng.choice([1.0, float(streams)])),
    )
    return child


def tune_hyper_connections(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    """Jitter hyper stream count and mixing init knobs."""

    child = clone_spec(spec)
    current = getattr(child.model, "hyper", None)
    if not isinstance(current, HyperConnectionsConfig) or current.streams <= 1:
        return toggle_hyper_connections(child, rng)
    streams = int(current.streams)
    if rng.random() < 0.35:
        streams = int(rng.choice([2, 3, 4]))
    diag_bias = float(current.diag_bias)
    if rng.random() < 0.4:
        diag_bias = float(rng.choice([0.0, 2.0, 4.0, 6.0, 8.0]))
    noise_std = float(current.noise_std)
    if rng.random() < 0.4:
        noise_std = float(rng.choice([0.0, 1e-5, 1e-4, 1e-3, 1e-2]))
    update_scale = float(current.update_scale)
    if rng.random() < 0.5:
        update_scale = float(rng.choice([0.25, 0.5, 1.0, 2.0, float(streams)]))
    child.model.hyper = HyperConnectionsConfig(
        streams=streams,
        diag_bias=diag_bias,
        noise_std=noise_std,
        update_scale=update_scale,
    )
    return child


def tune_kv_policy(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    """Jitter KV policy knobs (window size, quant type, latent dim)."""

    child = clone_spec(spec)
    policy = getattr(child.model, "kv_policy", None)
    if policy is None:
        return toggle_kv_policy(child, rng)

    cache = str(getattr(policy, "cache", "full") or "full")
    if cache in {"window", "ring"}:
        current = int(getattr(policy, "window", 4096) or 4096)
        mult = rng.choice([0.5, 0.75, 1.0, 1.25, 1.5])
        policy.window = max(256, min(32768, int(current * mult)))
    elif cache == "latent":
        current = int(getattr(policy, "latent_dim", 256) or 256)
        delta = rng.choice([-64, -32, 0, 32, 64, 128])
        policy.latent_dim = max(32, min(2048, int(current + delta)))

    if rng.random() < 0.6:
        policy.quant = rng.choice(["none", "nf4", "fp8", "int8"])
    child.model.kv_policy = policy
    return child


def toggle_selector(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    """Flip selector-based sparsity on an attention block and retune its knobs."""
    child = clone_spec(spec)
    blocks = [
        b
        for b in child.model.blocks
        if b.attn is not None and str(getattr(b.attn, "kind", "MHA") or "MHA").upper() != "LINEAR"
    ]
    if not blocks:
        return child
    b = rng.choice(blocks)
    if b.attn is None:
        return child
    if getattr(b.attn, "selector", "none") != "none":
        b.attn.selector = "none"
        b.attn.selector_topk = None
        b.attn.selector_heads = None
        b.attn.selector_dim = None
        b.attn.selector_rope = "none"
        b.attn.selector_detach = False
    else:
        b.attn.selector = "dsa"
        b.attn.selector_topk = int(rng.choice([32, 64, 96, 128, 192]))
        b.attn.selector_heads = int(rng.choice([1, 2, 4]))
        b.attn.selector_dim = b.attn.head_dim
        b.attn.selector_rope = rng.choice(["partial", "full", "none"])
        b.attn.selector_detach = rng.choice([True, False])
    return child


def tune_rope(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    child = clone_spec(spec)
    blocks = [b for b in child.model.blocks if b.attn is not None]
    if not blocks:
        return child
    b = rng.choice(blocks)
    if b.attn is None:
        return child
    if rng.random() < 0.1:
        b.attn.rope = None
        b.attn.rope_theta = None
        b.attn.rope_factor = None
        return child

    b.attn.rope = rng.choice(["standard", "linear", "ntk", "yarn"])
    base = float(b.attn.rope_theta or 10000.0)
    jitter = rng.uniform(0.5, 2.0)
    b.attn.rope_theta = max(1000.0, min(200000.0, base * jitter))
    if str(b.attn.rope or "").lower() in {"linear", "ntk", "yarn"}:
        current = float(getattr(b.attn, "rope_factor", None) or 1.0)
        if current <= 0.0:
            current = 1.0
        factor = current * rng.uniform(0.8, 1.25)
        b.attn.rope_factor = max(1.0, min(16.0, factor))
    else:
        b.attn.rope_factor = None
    return child


def add_recurrence(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    child = clone_spec(spec)
    if len(child.model.blocks) < 3:
        return child
    start = rng.randrange(0, len(child.model.blocks) - 1)
    # Allow wider spans so recurrence can stitch distant stages together.
    max_span = max(2, min(len(child.model.blocks), rng.choice([4, 6, 8, len(child.model.blocks)])))
    span = rng.randint(2, max_span)
    end = min(len(child.model.blocks), start + span)
    if end <= start:
        end = min(len(child.model.blocks), start + 1)
    rec = RecurrenceConfig(
        start=start,
        end=end,
        adapter=rng.choice(["linear", "gated"]),
        concat_prelude=rng.choice([True, False]),
        init_state=rng.choice(["zeros", "noise"]),
        noise_std=rng.uniform(0.01, 0.05),
        train_recurrence=rng.choice([1, 2]),
        max_train_recurrence=rng.choice([4, 6, 8]),
        curriculum_fraction=rng.uniform(0.1, 0.4),
        test_recurrences=[1, 2, 4, 8, 16],
    )
    child.model.recurrences.append(rec)
    return child


def tune_recurrence(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    child = clone_spec(spec)
    if not child.model.recurrences:
        return add_recurrence(spec, rng)
    rec = rng.choice(child.model.recurrences)
    rec.max_train_recurrence = max(
        rec.train_recurrence,
        int(rec.max_train_recurrence + rng.choice([-1, 0, 2])),
    )
    rec.curriculum_fraction = max(0.0, min(1.0, rec.curriculum_fraction + rng.uniform(-0.1, 0.1)))
    rec.concat_prelude = rng.choice([True, rec.concat_prelude])
    rec.adapter = rng.choice(["linear", "gated"])
    rec.init_state = rng.choice(["zeros", "noise"])
    return child


def tune_attn_gating(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    child = clone_spec(spec)
    blocks = [b for b in child.model.blocks if b.attn is not None]
    if not blocks:
        return child
    b = rng.choice(blocks)
    if b.attn is None:
        return child

    if b.attn.gating_pos == "none":
        # Enable gating with a random position/op
        b.attn.gating_pos = rng.choice(["output", "value"])
        b.attn.gating_op = rng.choice(["dense", "diagonal"])
    else:
        # 33% chance turn off, 66% chance change params
        if rng.random() < 0.33:
            b.attn.gating_pos = "none"
        else:
            if rng.random() < 0.5:
                b.attn.gating_pos = rng.choice(["output", "value"])
            else:
                b.attn.gating_op = rng.choice(["dense", "diagonal"])
    return child


def tune_attn_shape(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    """Jitter attention heads/head_dim/kv_groups while keeping model dim stable."""
    child = clone_spec(spec)
    blocks = [b for b in child.model.blocks if b.attn is not None]
    if not blocks:
        return child
    b = rng.choice(blocks)
    attn = b.attn
    if attn is None:
        return child
    d_model = child.model.emb.dim
    # pick heads that divide d_model reasonably
    candidate_heads = [h for h in [4, 6, 8, 12, 16] if d_model % h == 0]
    if not candidate_heads:
        return child
    heads = rng.choice(candidate_heads)
    head_dim = d_model // heads
    attn.heads = heads
    attn.head_dim = head_dim
    # kv_groups in [1, heads] preferring divisors
    kv_candidates = [k for k in [1, 2, 4, 8, heads] if k <= heads and heads % k == 0]
    attn.kv_groups = int(rng.choice(kv_candidates))
    return child


def tune_attn_sparsity(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    """Explore sparsity/window settings (local/global strides)."""
    child = clone_spec(spec)
    blocks = [
        b
        for b in child.model.blocks
        if b.attn is not None and str(getattr(b.attn, "kind", "MHA") or "MHA").upper() != "LINEAR"
    ]
    if not blocks:
        return child
    b = rng.choice(blocks)
    attn = b.attn
    if attn is None:
        return child
    sparsity_opts: list[Literal["none", "local_global"]] = ["none", "local_global"]
    attn.sparsity = rng.choice(sparsity_opts)
    if attn.sparsity == "local_global":
        attn.sw = rng.choice([64, 96, 128, 192, 256])
        attn.global_stride = rng.choice([32, 64, 96, 128])
    else:
        attn.sw = None
        attn.global_stride = None
    return child


def tune_ffn_width_activation(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    """Adjust FFN hidden size and activation."""
    child = clone_spec(spec)
    if not child.model.blocks:
        return child
    block = rng.choice(child.model.blocks)
    if block.ffn is None:
        return child
    if not isinstance(block.ffn, DenseFFNConfig):
        return child
    hidden = int(block.ffn.hidden)
    if hidden > 0:
        scale = rng.uniform(0.75, 1.5)
        new_hidden = max(256, min(int(hidden * scale), 8192))
        block.ffn.hidden = new_hidden
    block.ffn.activation = rng.choice(["swiglu", "gelu", "silu", "relu"])
    return child


def tune_router_coeffs(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    """Jitter MoE router temperatures and load-balance coefficients."""
    child = clone_spec(spec)
    moe_blocks = [b for b in child.model.blocks if isinstance(b.ffn, MoEFFNConfig)]
    if not moe_blocks:
        return child
    block = rng.choice(moe_blocks)
    if not isinstance(block.ffn, MoEFFNConfig):
        return child
    block.ffn.router_temperature = rng.choice([None, rng.uniform(0.3, 2.0)])
    block.ffn.router_lb_weight = rng.choice([None, rng.uniform(0.0, 0.1)])
    block.ffn.router_aux_weight = rng.choice([None, rng.uniform(0.0, 0.1)])
    return child


def tune_retro(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    """Adjust retro memory slots/stride and gating."""
    child = clone_spec(spec)
    if not child.model.blocks:
        return child
    block = rng.choice(child.model.blocks)
    # Ensure a retro extra exists
    retro = None
    for extra in block.extras:
        if isinstance(extra, RetroConfig):
            retro = extra
            break
    if retro is None:
        retro = RetroConfig(
            memory_tokens=int(rng.choice([256, 512, 1024])),
            stride=int(rng.choice([32, 64, 128])),
            aggregator=rng.choice(["mean", "attention", "gate"]),
            gating_weight=rng.uniform(0.1, 0.5),
        )
        block.extras.append(retro)
    else:
        retro.memory_tokens = int(rng.choice([256, 512, 768, 1024]))
        retro.stride = int(rng.choice([16, 32, 64, 128]))
        retro.aggregator = rng.choice(["mean", "attention", "gate"])
        retro.gating_weight = rng.uniform(0.1, 0.5)
    return child


def toggle_qk_norm(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    """Toggle QK norm clamp."""
    child = clone_spec(spec)
    blocks = [b for b in child.model.blocks if b.attn is not None]
    if not blocks:
        return child
    b = rng.choice(blocks)
    attn = b.attn
    if attn is None:
        return child
    current = getattr(attn, "qk_norm_max", None)
    if current is None:
        attn.qk_norm_max = rng.choice([0.5, 1.0, 2.0, 4.0])
    else:
        attn.qk_norm_max = None
    return child


def duplicate_block_span(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    """Append a copied span of blocks to increase depth without disturbing existing indices."""
    child = clone_spec(spec)
    blocks = child.model.blocks
    if not blocks:
        return child
    start = rng.randrange(len(blocks))
    span_len = rng.choice([1, 2, 3])
    end = min(len(blocks), start + span_len)
    duplicated = [copy.deepcopy(b) for b in blocks[start:end]]
    blocks.extend(duplicated)
    return child


def shuffle_block_span(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    """Shuffle a local span of blocks to create new phase orderings."""
    child = clone_spec(spec)
    blocks = child.model.blocks
    if len(blocks) < 3:
        return child
    span_len = rng.choice([2, 3, 4])
    if span_len > len(blocks):
        return child
    start = rng.randrange(0, len(blocks) - span_len + 1)
    span = blocks[start : start + span_len]
    rng.shuffle(span)
    blocks[start : start + span_len] = span
    return child


def add_additional_recurrence(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    """Add a second recurrence window to create multi-stage looping."""
    child = clone_spec(spec)
    if len(child.model.blocks) < 2:
        return child
    start = rng.randrange(0, len(child.model.blocks) - 1)
    end = rng.randrange(start + 1, len(child.model.blocks) + 1)
    train_recurrence = int(rng.choice([1, 2, 3]))
    max_recurrence_choices = [value for value in [2, 4, 6] if value >= train_recurrence]
    max_train_recurrence = int(rng.choice(max_recurrence_choices or [train_recurrence]))
    rec = RecurrenceConfig(
        start=start,
        end=end,
        adapter=rng.choice(["linear", "gated"]),
        adapter_dim=None,
        concat_prelude=rng.choice([True, False]),
        init_state=rng.choice(["zeros", "noise"]),
        noise_std=rng.uniform(0.01, 0.05),
        train_recurrence=train_recurrence,
        max_train_recurrence=max_train_recurrence,
        curriculum_fraction=rng.uniform(0.1, 0.4),
        test_recurrences=[1, 2, 4, 8],
    )
    child.model.recurrences.append(rec)
    return child


def add_extra_combo(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    """Attach multiple extras (retro + gated mix) to encourage hybrid blocks."""
    child = clone_spec(spec)
    if not child.model.blocks:
        return child
    block = rng.choice(child.model.blocks)
    existing_types = {type(extra) for extra in block.extras}
    if RetroConfig not in existing_types:
        block.extras.append(
            RetroConfig(
                memory_tokens=int(rng.choice([256, 512, 1024])),
                stride=int(rng.choice([32, 64, 128])),
                aggregator=rng.choice(["mean", "attention", "gate"]),
                gating_weight=rng.uniform(0.1, 0.5),
            )
        )
    if GatedModuleConfig not in existing_types:
        block.extras.append(GatedModuleConfig(init_weight=rng.uniform(0.05, 0.3)))
    return child


def graph_jitter(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    """Apply a handful of neutral edits to increase structural entropy."""
    child = clone_spec(spec)
    jitter_ops = [
        "duplicate_block_span",
        "shuffle_block_span",
        "add_recurrence",
        "add_additional_recurrence",
        "toggle_hyper_connections",
        "tune_hyper_connections",
        "toggle_ssm",
        "insert_retro_module",
        "insert_custom_module",
        "insert_graph_module",
        "insert_assoc_memory",
        "tune_assoc_memory",
        "insert_memory_tokens",
        "tune_memory_tokens",
        "insert_chunk_memory",
        "tune_chunk_memory",
        "insert_lookup_memory",
        "tune_lookup_memory",
        "toggle_branch_router",
        "tune_branch_router",
        "insert_layer_scale",
        "tune_layer_scale",
        "toggle_gated_mix",
        "toggle_alibi",
        "toggle_linear_attention",
        "toggle_mla_attention",
        "tune_attn_gating",
        "tune_kv",
        "toggle_kv_policy",
        "tune_kv_policy",
        "tune_rope",
        "dense_to_moe",
        "mutate_topk",
        "shift_moe",
        "make_gqa",
    ]
    steps = rng.randint(2, 4)
    current = child
    for name in rng.sample(jitter_ops, k=min(steps, len(jitter_ops))):
        fn = REGISTRY.get(name)
        if fn:
            result = fn(current, rng)
            if isinstance(result, tuple):
                current = result[1]
            else:
                current = result
    return current


REGISTRY: dict[str, MutationFn] = {
    "duplicate_block_span": duplicate_block_span,
    "shuffle_block_span": shuffle_block_span,
    "add_additional_recurrence": add_additional_recurrence,
    "add_extra_combo": add_extra_combo,
    "tune_attn_gating": tune_attn_gating,
    "dense_to_moe": dense_to_moe,
    "mutate_topk": mutate_topk,
    "shift_moe": shift_moe,
    "tune_router": tune_router,
    "make_gqa": make_gqa,
    "toggle_precision": toggle_precision,
    "toggle_optimizer": toggle_optimizer,
    "tune_optimizer": tune_optimizer,
    "tune_warmup": tune_warmup,
    "tune_clip": tune_clip,
    "insert_retro_module": insert_retro_module,
    "insert_custom_module": insert_custom_module,
    "insert_graph_module": insert_graph_module,
    "insert_assoc_memory": insert_assoc_memory,
    "tune_assoc_memory": tune_assoc_memory,
    "insert_memory_tokens": insert_memory_tokens,
    "tune_memory_tokens": tune_memory_tokens,
    "insert_chunk_memory": insert_chunk_memory,
    "tune_chunk_memory": tune_chunk_memory,
    "insert_lookup_memory": insert_lookup_memory,
    "tune_lookup_memory": tune_lookup_memory,
    "toggle_branch_router": toggle_branch_router,
    "tune_branch_router": tune_branch_router,
    "insert_layer_scale": insert_layer_scale,
    "tune_layer_scale": tune_layer_scale,
    "toggle_gated_mix": toggle_gated_mix,
    "toggle_ssm": toggle_ssm,
    "toggle_alibi": toggle_alibi,
    "toggle_linear_attention": toggle_linear_attention,
    "toggle_mla_attention": toggle_mla_attention,
    "tune_kv": tune_kv,
    "toggle_kv_policy": toggle_kv_policy,
    "tune_kv_policy": tune_kv_policy,
    "toggle_hyper_connections": toggle_hyper_connections,
    "tune_hyper_connections": tune_hyper_connections,
    "toggle_selector": toggle_selector,
    "tune_rope": tune_rope,
    "tune_attn_shape": tune_attn_shape,
    "tune_attn_sparsity": tune_attn_sparsity,
    "tune_ffn_width_activation": tune_ffn_width_activation,
    "tune_router_coeffs": tune_router_coeffs,
    "tune_retro": tune_retro,
    "toggle_qk_norm": toggle_qk_norm,
    "add_recurrence": add_recurrence,
    "tune_recurrence": tune_recurrence,
    "graph_jitter": graph_jitter,
    "tune_experts": tune_experts,
    "template_mutation": template_mutation,
}


def mutate(
    spec: ArchitectureSpec,
    rng: random.Random | None = None,
    weights: dict[str, float] | None = None,
    steps: int = 1,
    validate: bool = True,
) -> tuple[str, ArchitectureSpec]:
    """Apply one or more registered mutations. If weights provided, sample by weight.

    Args:
        spec: The architecture specification to mutate.
        rng: Random number generator for reproducibility.
        weights: Optional mutation weights for weighted sampling.
        steps: Number of mutations to chain.
        validate: If True, validate the spec after each mutation and raise
            MutationError with a diff if validation fails.

    Returns:
        A tuple of (mutation_label, mutated_spec).

    Raises:
        MutationError: If validate=True and a mutation produces an invalid spec.
    """
    rng = rng or random.Random()  # noqa: S311  # nosec B311 - deterministic enough for search
    names = list(REGISTRY)

    def _pick() -> str:
        if weights:
            w = [max(0.0, float(weights.get(n, 0.0))) for n in names]
            if any(w):
                total = sum(w) or 1.0
                probs = [x / total for x in w]
                return rng.choices(names, weights=probs, k=1)[0]
        return rng.choice(names)

    applied: list[str] = []
    current = spec
    for _ in range(max(1, steps)):
        name = _pick()
        before = current
        result = REGISTRY[name](current, rng)
        if isinstance(result, tuple) and len(result) == 2:
            label, mutated = result
            applied.append(str(label))
        else:
            mutated = result
            label = name
            applied.append(name)

        # Validate the mutated spec if requested
        if validate:
            try:
                # Re-validate by round-tripping through dict
                mutated_dict = mutated.model_dump(mode="python")
                current = validate_mutation_result(label, before, mutated_dict)
            except MutationError:
                # Re-raise with full context
                raise
        else:
            current = mutated

    label = "+".join(applied) if len(applied) > 1 else applied[0]
    return label, current
