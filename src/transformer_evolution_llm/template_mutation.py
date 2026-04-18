"""Template-driven mutation engine for architectural edits."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

import yaml

from .dsl import (
    ArchitectureSpec,
    AssociativeMemoryConfig,
    AttentionConfig,
    BlockConfig,
    BranchRouterConfig,
    ChunkMemoryConfig,
    CustomModuleConfig,
    DenseFFNConfig,
    GatedModuleConfig,
    LayerScaleConfig,
    LookupMemoryConfig,
    MemoryTokensConfig,
    MoEFFNConfig,
    RecurrenceConfig,
    RetroConfig,
    SoftmaxConfig,
    SSMConfig,
)

TEMPLATE_PATH_DEFAULT = Path("configs/mutation_templates.yaml")
TEMPLATE_PATH_ACTIVE = TEMPLATE_PATH_DEFAULT

_TEMPLATE_LEARNING_ENABLED = False
_TEMPLATE_LEARNING_ETA = 0.2
_TEMPLATE_LEARNING_MIN_WEIGHT = 0.05
_TEMPLATE_LEARNING_MAX_WEIGHT = 5.0
_TEMPLATE_LEARNING_MAX_TEMPLATES = 128
_TEMPLATE_LEARNING_SAVE_EVERY = 20
_TEMPLATE_LEARNING_PROMOTE_MIN_DELTA = 0.0
_TEMPLATE_LEARNING_UPDATES = 0
_TEMPLATE_LEARNING_DIRTY = False
_TEMPLATE_WEIGHT_OVERRIDES: dict[str, float] = {}
_TEMPLATE_RECENT: dict[str, MutationTemplate] = {}
_TEMPLATE_POSITIVE: dict[str, int] = {}

AttentionKind = Literal["MHA", "GQA", "MQA", "LINEAR", "MLA"]
SoftmaxNorm = Literal["none", "rms", "layer"]
ActivationName = Literal["silu", "gelu", "relu", "relu_squared", "swiglu"]
OptimizerName = Literal["adamw", "lion", "muon"]
ExportDType = Literal["int8", "fp16"]
GradientTransformMode = Literal[
    "identity",
    "sign",
    "normalize",
    "orthogonalize_2d",
    "sign_orthogonalize_2d",
]
UpdateFilterMode = Literal["none", "bernoulli", "topk"]
UpdateFilterGranularity = Literal["element", "block"]
RecurrenceAdapter = Literal["linear", "gated"]


class TemplateAction(TypedDict, total=False):
    selector: str
    extra_type: str
    params: dict[str, Any]
    block_template: str
    position: str
    new_type: str
    n_experts: int
    k: int
    balance: float
    capacity_factor: float
    # Tuning knobs
    qk_norm_max: float | None
    gating_pos: str | None
    gating_op: str | None
    sw_jitter: int
    temperature: float
    lb_coeff: float
    entropy_coeff: float
    sparsity: str
    block_size: int | None
    block_stride: int | None
    global_stride: int | None
    dilation: int | None
    kind: str | None
    kv_groups: int | None
    kv_latent_dim: int | None
    value_glu: bool | None
    qk_norm: str | None
    qk_scale: float | str | None
    softcap: float | None
    hidden: int | None
    hidden_mult: float | None
    activation: str | None
    optimizer_name: str | None
    lr: float | None
    matrix_lr: float | None
    scalar_lr: float | None
    tied_embedding_lr: float | None
    tied_embedding_export_dtype: str | None
    tied_embed_init_std: float | None
    gradient_transform_mode: str | None
    gradient_transform_ns_steps: int | None
    gradient_transform_eps: float | None
    update_filter_mode: str | None
    update_filter_keep_ratio: float | None
    update_filter_granularity: str | None
    update_filter_block_size: int | None
    update_filter_momentum_blend: float | None
    warmup: int | None
    warmdown_steps: int | None
    muon_momentum_warmup_start: float | None
    muon_momentum_warmup_steps: int | None
    clip: float | None
    tail_blocks: int | None
    start_fraction: float | None
    end_fraction: float | None
    start: int | None
    end: int | None
    adapter: str | None
    concat_prelude: bool | None
    train_recurrence: int | None
    max_train_recurrence: int | None
    test_recurrences: list[int] | None


ActionMap = dict[str, TemplateAction]


def _new_origin_id(rng: random.Random) -> str:
    return f"o{rng.getrandbits(48):012x}"


def _clamp_int(value: int, *, lo: int, hi: int) -> int:
    return max(int(lo), min(int(hi), int(value)))


def _round_hidden(value: float, *, step: int = 128) -> int:
    rounded = int(round(float(value) / float(step)) * step)
    return max(step, rounded)


def _ensure_softmax(attn: AttentionConfig) -> SoftmaxConfig:
    if attn.softmax is None:
        attn.softmax = SoftmaxConfig()
    return attn.softmax


@dataclass
class MutationTemplate:
    name: str
    weight: float
    conditions: dict[str, Any]
    actions: list[ActionMap]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "weight": float(self.weight),
            "conditions": self.conditions,
            "actions": self.actions,
        }


def load_templates() -> list[MutationTemplate]:
    if not TEMPLATE_PATH_ACTIVE.exists():
        return _seed_templates()
    data = yaml.safe_load(TEMPLATE_PATH_ACTIVE.read_text())
    templates = []
    for entry in data.get("templates", []):
        templates.append(
            MutationTemplate(
                name=entry["name"],
                weight=float(entry.get("weight", 1.0)),
                conditions=entry.get("conditions", {}),
                actions=cast(list[ActionMap], entry.get("actions", [])),
            )
        )
    return templates


def template_names() -> list[str]:
    """Return known persisted template names."""
    return [template.name for template in load_templates()]


def configure_template_learning(
    *,
    enabled: bool,
    path: Path = TEMPLATE_PATH_DEFAULT,
    eta: float = 0.2,
    min_weight: float = 0.05,
    max_weight: float = 5.0,
    max_templates: int = 128,
    save_every: int = 20,
    promote_min_delta: float = 0.0,
) -> None:
    global TEMPLATE_PATH_ACTIVE
    global _TEMPLATE_LEARNING_ENABLED
    global _TEMPLATE_LEARNING_ETA
    global _TEMPLATE_LEARNING_MIN_WEIGHT
    global _TEMPLATE_LEARNING_MAX_WEIGHT
    global _TEMPLATE_LEARNING_MAX_TEMPLATES
    global _TEMPLATE_LEARNING_SAVE_EVERY
    global _TEMPLATE_LEARNING_PROMOTE_MIN_DELTA
    global _TEMPLATE_LEARNING_UPDATES
    global _TEMPLATE_LEARNING_DIRTY

    TEMPLATE_PATH_ACTIVE = Path(path)
    _TEMPLATE_LEARNING_ENABLED = bool(enabled)
    _TEMPLATE_LEARNING_ETA = float(eta)
    _TEMPLATE_LEARNING_MIN_WEIGHT = float(min_weight)
    _TEMPLATE_LEARNING_MAX_WEIGHT = float(max_weight)
    _TEMPLATE_LEARNING_MAX_TEMPLATES = int(max_templates)
    _TEMPLATE_LEARNING_SAVE_EVERY = int(save_every)
    _TEMPLATE_LEARNING_PROMOTE_MIN_DELTA = float(promote_min_delta)
    _TEMPLATE_LEARNING_UPDATES = 0
    _TEMPLATE_LEARNING_DIRTY = False
    _TEMPLATE_WEIGHT_OVERRIDES.clear()
    _TEMPLATE_RECENT.clear()
    _TEMPLATE_POSITIVE.clear()

    if _TEMPLATE_LEARNING_ENABLED and not TEMPLATE_PATH_ACTIVE.exists():
        _persist_templates(_seed_templates())


def flush_template_learning() -> None:
    if not _TEMPLATE_LEARNING_ENABLED:
        return
    if not _TEMPLATE_LEARNING_DIRTY:
        return
    merged = _merge_persisted_templates(load_templates())
    _persist_templates(merged)


def record_template_result(template_name: str, delta: float) -> None:
    global _TEMPLATE_LEARNING_UPDATES
    global _TEMPLATE_LEARNING_DIRTY

    if not _TEMPLATE_LEARNING_ENABLED:
        return

    improved = float(delta) > float(_TEMPLATE_LEARNING_PROMOTE_MIN_DELTA)
    templates = load_templates()
    by_name: dict[str, MutationTemplate] = {tpl.name: tpl for tpl in templates}
    tpl = by_name.get(template_name)
    if tpl is None:
        candidate = _TEMPLATE_RECENT.get(template_name)
        if candidate is None or not improved:
            return
        tpl = MutationTemplate(
            name=candidate.name,
            weight=float(candidate.weight),
            conditions=candidate.conditions,
            actions=candidate.actions,
        )
        templates.append(tpl)
        by_name[tpl.name] = tpl

    current = float(_TEMPLATE_WEIGHT_OVERRIDES.get(template_name, tpl.weight))
    eta = max(0.0, min(1.0, float(_TEMPLATE_LEARNING_ETA)))
    if improved:
        updated = current * (1.0 + eta)
        _TEMPLATE_POSITIVE[template_name] = int(_TEMPLATE_POSITIVE.get(template_name, 0)) + 1
    else:
        updated = current * (1.0 - eta)
    updated = max(
        float(_TEMPLATE_LEARNING_MIN_WEIGHT),
        min(float(_TEMPLATE_LEARNING_MAX_WEIGHT), updated),
    )
    _TEMPLATE_WEIGHT_OVERRIDES[template_name] = updated
    by_name[template_name].weight = updated

    templates = _cap_templates(list(by_name.values()))
    _TEMPLATE_LEARNING_UPDATES += 1
    _TEMPLATE_LEARNING_DIRTY = True
    if _TEMPLATE_LEARNING_UPDATES % max(1, int(_TEMPLATE_LEARNING_SAVE_EVERY)) == 0:
        merged = _merge_persisted_templates(templates)
        _persist_templates(merged)
        _TEMPLATE_LEARNING_DIRTY = False


def apply_template_mutation_with_name(
    spec: ArchitectureSpec, rng: random.Random
) -> tuple[str, ArchitectureSpec]:
    base_templates = load_templates()
    dynamic_templates: list[MutationTemplate] = []
    for tpl in base_templates:
        if rng.random() < 0.5:
            dynamic_templates.append(_mutate_template(tpl, rng))
    dynamic_templates.append(_generate_random_template(spec, rng))
    templates = base_templates + dynamic_templates
    eligible = [tpl for tpl in templates if _matches_conditions(spec, tpl.conditions)]
    if not eligible:
        return "none", spec
    weights = [float(_TEMPLATE_WEIGHT_OVERRIDES.get(tpl.name, tpl.weight)) for tpl in eligible]
    template = rng.choices(eligible, weights=weights, k=1)[0]
    new_spec = spec.model_copy(deep=True)
    for action in template.actions:
        _apply_action(new_spec, action, rng)
    _sanitize_topology(new_spec)
    _TEMPLATE_RECENT[template.name] = template
    return template.name, new_spec


def apply_template_mutation_named_with_name(
    spec: ArchitectureSpec,
    rng: random.Random,
    template_name: str,
) -> tuple[str, ArchitectureSpec]:
    """Apply a specific template by name when eligible."""
    selected: MutationTemplate | None = None
    for template in load_templates():
        if template.name != template_name:
            continue
        if not _matches_conditions(spec, template.conditions):
            continue
        selected = template
        break
    if selected is None:
        return "none", spec
    new_spec = spec.model_copy(deep=True)
    for action in selected.actions:
        _apply_action(new_spec, action, rng)
    _sanitize_topology(new_spec)
    _TEMPLATE_RECENT[selected.name] = selected
    return selected.name, new_spec


def apply_template_mutation_named(
    spec: ArchitectureSpec,
    rng: random.Random,
    template_name: str,
) -> ArchitectureSpec:
    """Apply a specific template and return only the mutated spec."""
    _, mutated = apply_template_mutation_named_with_name(spec, rng, template_name)
    return mutated


def apply_template_mutation(spec: ArchitectureSpec, rng: random.Random) -> ArchitectureSpec:
    _, mutated = apply_template_mutation_with_name(spec, rng)
    return mutated


def _merge_persisted_templates(templates: list[MutationTemplate]) -> list[MutationTemplate]:
    merged: dict[str, MutationTemplate] = {tpl.name: tpl for tpl in templates}
    for name, tpl in _TEMPLATE_RECENT.items():
        if name in merged:
            continue
        if int(_TEMPLATE_POSITIVE.get(name, 0)) <= 0:
            continue
        merged[name] = MutationTemplate(
            name=tpl.name,
            weight=float(_TEMPLATE_WEIGHT_OVERRIDES.get(name, tpl.weight)),
            conditions=tpl.conditions,
            actions=tpl.actions,
        )
    return _cap_templates(list(merged.values()))


def _cap_templates(templates: list[MutationTemplate]) -> list[MutationTemplate]:
    cap = int(_TEMPLATE_LEARNING_MAX_TEMPLATES)
    if cap <= 0 or len(templates) <= cap:
        return templates
    protected = {tpl.name for tpl in templates if tpl.name.startswith("seed-")}
    removable = [tpl for tpl in templates if tpl.name not in protected]
    removable.sort(key=lambda tpl: float(getattr(tpl, "weight", 0.0)))
    keep = list(templates)
    drop_n = max(0, len(templates) - cap)
    dropped = 0
    for tpl in removable:
        if dropped >= drop_n:
            break
        keep = [k for k in keep if k.name != tpl.name]
        dropped += 1
    return keep


def _persist_templates(templates: list[MutationTemplate]) -> None:
    payload = {"templates": [tpl.to_dict() for tpl in templates]}
    TEMPLATE_PATH_ACTIVE.parent.mkdir(parents=True, exist_ok=True)
    TEMPLATE_PATH_ACTIVE.write_text(yaml.safe_dump(payload, sort_keys=False))


def _matches_conditions(spec: ArchitectureSpec, conditions: dict[str, Any]) -> bool:
    blocks = spec.model.blocks
    if not conditions:
        return True
    if conditions.get("parameter_golf_only") and spec.parameter_golf is None:
        return False
    if conditions.get("requires_ssm_block"):
        if not any(block.ssm for block in blocks):
            return False
    if conditions.get("requires_dense_ffn"):
        if not any(isinstance(block.ffn, DenseFFNConfig) for block in blocks):
            return False
    if conditions.get("requires_recurrence") and not spec.model.recurrences:
        return False
    if conditions.get("requires_no_recurrence") and spec.model.recurrences:
        return False
    if conditions.get("min_blocks"):
        if len(blocks) < int(conditions["min_blocks"]):
            return False
    return True


def _apply_action(spec: ArchitectureSpec, action: ActionMap, rng: random.Random) -> None:
    if "add_extra" in action:
        _add_extra(spec, action["add_extra"], rng)
    elif "insert_block" in action:
        _insert_block(spec, action["insert_block"], rng)
    elif "replace_ffn" in action:
        _replace_ffn(spec, action["replace_ffn"], rng)
    elif "remove_block" in action:
        _remove_block(spec, action["remove_block"], rng)
    elif "tune_attn" in action:
        _tune_attn(spec, dict(action["tune_attn"]), rng)
    elif "tune_ffn" in action:
        _tune_ffn(spec, dict(action["tune_ffn"]), rng)
    elif "tune_optimizer" in action:
        _tune_optimizer(spec, dict(action["tune_optimizer"]))
    elif "set_recurrence" in action:
        _set_recurrence(spec, dict(action["set_recurrence"]))
    elif "tune_router" in action:
        _tune_router(spec, dict(action["tune_router"]), rng)


def _sanitize_topology(spec: ArchitectureSpec) -> None:
    n_blocks = len(spec.model.blocks)
    if n_blocks <= 0:
        return
    for rec in spec.model.recurrences:
        start = int(max(0, min(n_blocks - 1, int(rec.start))))
        end = int(max(start + 1, min(n_blocks, int(rec.end))))
        rec.start = start
        rec.end = end
    max_idx = max(0, n_blocks - 1)
    for block in spec.model.blocks:
        for extra in block.extras:
            if not isinstance(extra, CustomModuleConfig):
                continue
            params = getattr(extra, "params", None)
            if not isinstance(params, dict):
                continue
            if str(params.get("type", "")).lower() != "feedback":
                continue
            src = params.get("source_block")
            if isinstance(src, (int, float)):
                params["source_block"] = int(max(0, min(max_idx, int(src))))


def _select_block_index(spec: ArchitectureSpec, selector: str, rng: random.Random) -> int | None:
    blocks = spec.model.blocks
    candidates: list[int]
    if selector == "random":
        candidates = list(range(len(blocks)))
    elif selector == "random_moe":
        candidates = [
            idx for idx, block in enumerate(blocks) if isinstance(block.ffn, MoEFFNConfig)
        ]
    elif selector == "random_dense":
        candidates = [
            idx for idx, block in enumerate(blocks) if isinstance(block.ffn, DenseFFNConfig)
        ]
    elif selector == "random_ssm":
        candidates = [idx for idx, block in enumerate(blocks) if block.ssm is not None]
    else:
        candidates = list(range(len(blocks)))
    if not candidates:
        return None
    return rng.choice(candidates)


def _add_extra(spec: ArchitectureSpec, params: TemplateAction, rng: random.Random) -> None:
    idx = _select_block_index(spec, params.get("selector", "random"), rng)
    if idx is None:
        return
    block = spec.model.blocks[idx]
    extra_type = params.get("extra_type", "custom")
    extra_params = params.get("params", {})
    if extra_type == "retro":
        block.extras.append(
            RetroConfig(
                memory_tokens=extra_params.get("memory_tokens", spec.data.seq_len // 4),
                stride=extra_params.get("stride", max(16, spec.data.seq_len // 16)),
                aggregator=extra_params.get("aggregator", "gate"),
                gating_weight=extra_params.get("gating_weight", 0.25),
            )
        )
    elif extra_type == "assoc_memory":
        block.extras.append(
            AssociativeMemoryConfig(
                heads=int(extra_params.get("heads", 4)),
                head_dim=int(extra_params.get("head_dim", 32)),
                feature_map=extra_params.get("feature_map", "elu"),
                dropout=float(extra_params.get("dropout", 0.0)),
                gating_weight=float(extra_params.get("gating_weight", 0.1)),
            )
        )
    elif extra_type == "memory_tokens":
        block.extras.append(
            MemoryTokensConfig(
                tokens=int(extra_params.get("tokens", 16)),
                heads=int(extra_params.get("heads", 2)),
                head_dim=int(extra_params.get("head_dim", 32)),
                dropout=float(extra_params.get("dropout", 0.0)),
                init_std=float(extra_params.get("init_std", 0.02)),
                gating_weight=float(extra_params.get("gating_weight", 0.1)),
            )
        )
    elif extra_type == "chunk_memory":
        block.extras.append(
            ChunkMemoryConfig(
                chunk_size=int(extra_params.get("chunk_size", max(8, spec.data.seq_len // 16))),
                stride=extra_params.get("stride"),
                heads=int(extra_params.get("heads", 2)),
                head_dim=int(extra_params.get("head_dim", 32)),
                dropout=float(extra_params.get("dropout", 0.0)),
                gating_weight=float(extra_params.get("gating_weight", 0.1)),
            )
        )
    elif extra_type == "lookup_memory":
        block.extras.append(
            LookupMemoryConfig(
                entries=int(extra_params.get("entries", 256)),
                topk=int(extra_params.get("topk", 4)),
                key_dim=extra_params.get("key_dim"),
                value_dim=extra_params.get("value_dim"),
                temperature=float(extra_params.get("temperature", 1.0)),
                dropout=float(extra_params.get("dropout", 0.0)),
                chunk_size=int(extra_params.get("chunk_size", 1024)),
                lookup_device=str(extra_params.get("lookup_device", "model") or "model"),
                gating_weight=float(extra_params.get("gating_weight", 0.1)),
            )
        )
    elif extra_type == "gated":
        targets = extra_params.get("targets") or ["attn", "ffn"]
        block.extras.append(
            GatedModuleConfig(
                targets=targets,
                init_weight=float(extra_params.get("init_weight", 0.2)),
                learnable=extra_params.get("learnable", True),
            )
        )
    elif extra_type == "branch_router":
        block.extras.append(
            BranchRouterConfig(
                targets=extra_params.get("targets") or ["attn", "ffn", "ssm", "memory"],
                router_type=extra_params.get("router_type", "token"),
                hidden=extra_params.get("hidden"),
                dropout=float(extra_params.get("dropout", 0.0)),
                temperature=float(extra_params.get("temperature", 1.0)),
            )
        )
    elif extra_type == "layer_scale":
        block.extras.append(
            LayerScaleConfig(
                targets=extra_params.get("targets") or ["attn", "ffn"],
                init=float(extra_params.get("init", 1e-5)),
                learnable=bool(extra_params.get("learnable", True)),
            )
        )
    elif extra_type == "feedback":
        source = rng.randrange(0, idx + 1) if idx > 0 else 0
        block.extras.append(
            CustomModuleConfig(
                name="feedback_gate",
                params={
                    "type": "feedback",
                    "source_block": source,
                    "strength": extra_params.get("strength", 0.1),
                },
            )
        )
    elif extra_type == "graph_module":
        block.extras.append(
            CustomModuleConfig(
                name="graph_module",
                params={"ops": [{"op": "rmsnorm"}, {"op": "mlp", "hidden_mult": 2.0}]},
            )
        )
    else:
        block.extras.append(
            CustomModuleConfig(
                name=extra_params.get("name", "exp"),
                params=extra_params.get("params", {"dim": spec.model.emb.dim}),
            )
        )


def _insert_block(spec: ArchitectureSpec, params: TemplateAction, rng: random.Random) -> None:
    template_name = params.get("block_template", "dense_attn")
    position = params.get("position", "end")
    blocks = spec.model.blocks
    new_block = _build_block(template_name, spec, rng)
    if new_block is None:
        return
    if not getattr(new_block, "origin_id", None):
        new_block.origin_id = _new_origin_id(rng)
    if getattr(new_block, "parent_origin", None) is None:
        new_block.parent_origin = None
    if position == "start":
        blocks.insert(0, new_block)
    elif position == "end":
        blocks.append(new_block)
    elif position == "random":
        blocks.insert(rng.randrange(0, len(blocks) + 1), new_block)
    else:
        blocks.append(new_block)


def _build_block(name: str, spec: ArchitectureSpec, rng: random.Random) -> BlockConfig | None:
    dim = spec.model.emb.dim
    attn = _default_attention(spec)
    ffn: DenseFFNConfig | MoEFFNConfig = DenseFFNConfig(hidden=dim * 4, activation="swiglu")
    extras: list[Any] = []
    ssm = None
    if name == "retro_moe":
        ffn = MoEFFNConfig(
            hidden=dim * 4,
            n_experts=16,
            k=2,
            balance=0.05,
            capacity_factor=1.2,
            shared=1,
        )
        extras.append(
            RetroConfig(
                memory_tokens=min(512, spec.data.seq_len),
                stride=max(32, spec.data.seq_len // 8),
                aggregator="gate",
                gating_weight=0.3,
            )
        )
    elif name == "ssm_dense":
        ssm = SSMConfig(kind="mamba2", d_state=16, d_conv=4, dt_rank=8, chunk=128, gate=0.1)
    elif name == "feedback_dense":
        extras.append(
            CustomModuleConfig(
                name="feedback_gate",
                params={
                    "type": "feedback",
                    "source_block": max(0, len(spec.model.blocks) - 1),
                    "strength": rng.uniform(0.1, 0.3),
                },
            )
        )
        extras.append(
            GatedModuleConfig(
                targets=["attn", "ffn"],
                init_weight=rng.uniform(0.15, 0.35),
                learnable=True,
            )
        )
        extras.append(
            RetroConfig(
                memory_tokens=min(512, spec.data.seq_len),
                stride=max(16, spec.data.seq_len // 16),
                aggregator="gate",
                gating_weight=rng.uniform(0.2, 0.4),
            )
        )
    return BlockConfig(
        origin_id=_new_origin_id(rng),
        parent_origin=None,
        attn=attn,
        ffn=ffn,
        ssm=ssm,
        extras=extras,
    )


def _replace_ffn(spec: ArchitectureSpec, params: TemplateAction, rng: random.Random) -> None:
    idx = _select_block_index(spec, params.get("selector", "random"), rng)
    if idx is None:
        return
    block = spec.model.blocks[idx]
    new_type = params.get("new_type", "dense")
    if new_type == "moe":
        block.ffn = MoEFFNConfig(
            hidden=spec.model.emb.dim * 4,
            n_experts=params.get("n_experts", 8),
            k=params.get("k", 2),
            balance=params.get("balance", 0.05),
            capacity_factor=params.get("capacity_factor", 1.2),
            shared=1,
        )
    else:
        block.ffn = DenseFFNConfig(hidden=spec.model.emb.dim * 4, activation="swiglu")


def _remove_block(spec: ArchitectureSpec, params: TemplateAction, rng: random.Random) -> None:
    if len(spec.model.blocks) <= 1:
        return
    idx = _select_block_index(spec, params.get("selector", "random"), rng)
    if idx is None:
        return
    spec.model.blocks.pop(idx)


def _default_attention(spec: ArchitectureSpec) -> AttentionConfig | None:
    for block in spec.model.blocks:
        if block.attn:
            return block.attn.model_copy(deep=True)
    return None


def _seed_templates() -> list[MutationTemplate]:
    return [
        MutationTemplate(
            name="seed-feedback",
            weight=1.0,
            conditions={"requires_ssm_block": True},
            actions=[
                {
                    "add_extra": {
                        "selector": "random_ssm",
                        "extra_type": "feedback",
                        "params": {"strength": 0.1},
                    }
                }
            ],
        )
    ]


def _mutate_template(template: MutationTemplate, rng: random.Random) -> MutationTemplate:
    mutated = copy.deepcopy(template)
    mutated.name = f"{template.name}-mut-{rng.randrange(10_000)}"
    mutated.weight = max(0.1, template.weight * rng.uniform(0.8, 1.2))
    for action in mutated.actions:
        key = next(iter(action))
        params = action[key]
        if isinstance(params, dict) and "params" in params and isinstance(params["params"], dict):
            for param_key, value in params["params"].items():
                if isinstance(value, (int, float)):
                    params["params"][param_key] = value * rng.uniform(0.8, 1.2)
    return mutated


def _generate_random_template(spec: ArchitectureSpec, rng: random.Random) -> MutationTemplate:
    if spec.parameter_golf is not None:
        return _generate_parameter_golf_template(spec, rng)

    action_type = rng.choice(["add_extra", "insert_block", "replace_ffn"])
    # Occasionally tune stability/router knobs
    if rng.random() < 0.2:
        action_type = rng.choice(["tune_attn", "tune_router"])
    action: ActionMap
    if action_type == "add_extra":
        action = {
            "add_extra": {
                "selector": rng.choice(["random", "random_moe", "random_ssm"]),
                "extra_type": rng.choice(
                    [
                        "gated",
                        "retro",
                        "assoc_memory",
                        "memory_tokens",
                        "chunk_memory",
                        "branch_router",
                        "layer_scale",
                        "feedback",
                        "graph_module",
                    ]
                ),
                "params": {
                    "gating_weight": rng.uniform(0.1, 0.5),
                    "memory_tokens": rng.randint(64, spec.data.seq_len),
                    "tokens": rng.choice([8, 16, 32, 64]),
                    "chunk_size": rng.choice([16, 32, 64, 128, 256]),
                    "heads": rng.choice([2, 4, 8]),
                    "head_dim": rng.choice([16, 32, 64]),
                    "strength": rng.uniform(0.05, 0.3),
                    "temperature": rng.uniform(0.7, 1.5),
                    "init": rng.choice([1e-6, 1e-5, 1e-4]),
                },
            }
        }
    elif action_type == "insert_block":
        action = {
            "insert_block": {
                "position": rng.choice(["start", "end", "random"]),
                "block_template": rng.choice(["retro_moe", "ssm_dense"]),
            }
        }
    else:
        if action_type == "tune_attn":
            action = {
                "tune_attn": {
                    "selector": rng.choice(["random", "random_ssm", "random_dense", "random_moe"]),
                    "qk_norm_max": rng.choice([None, rng.uniform(5.0, 12.0)]),
                    "gating_pos": rng.choice(["none", "output", "value"]),
                    "gating_op": rng.choice(["dense", "diagonal"]),
                    "sw_jitter": rng.choice([-256, -128, -64, -32, 0, 32, 64, 128, 256]),
                    "sparsity": rng.choice(
                        ["none", "sliding", "block", "local_global", "dilated", "local_block"]
                    ),
                    "block_size": rng.choice([None, 32, 64, 128, 256]),
                    "block_stride": rng.choice([None, 16, 32, 64, 128, 256]),
                    "global_stride": rng.choice([None, 16, 32, 64, 128, 256]),
                    "dilation": rng.choice([None, 2, 4, 8, 16]),
                }
            }
        elif action_type == "tune_router":
            action = {
                "tune_router": {
                    "temperature": rng.uniform(0.7, 1.5),
                    "lb_coeff": rng.uniform(0.002, 0.03),
                    "entropy_coeff": rng.uniform(0.0, 0.01),
                }
            }
        else:
            action = {
                "replace_ffn": {
                    "selector": "random",
                    "new_type": rng.choice(["dense", "moe"]),
                    "n_experts": rng.choice([8, 16, 32]),
                    "k": rng.choice([1, 2, 4]),
                    "balance": rng.uniform(0.02, 0.08),
                }
            }
    return MutationTemplate(
        name=f"auto-{rng.randrange(10_000)}",
        weight=1.0,
        conditions={},
        actions=[action],
    )


def _generate_parameter_golf_template(
    spec: ArchitectureSpec, rng: random.Random
) -> MutationTemplate:
    builders = [
        _random_parameter_golf_attn_action,
        _random_parameter_golf_ffn_action,
        _random_parameter_golf_optimizer_action,
        _random_parameter_golf_recurrence_action,
    ]
    first_builder = rng.choice(builders)
    actions: list[ActionMap] = [first_builder(spec, rng)]
    if rng.random() < 0.35:
        remaining = [builder for builder in builders if builder is not first_builder]
        actions.append(rng.choice(remaining)(spec, rng))
    return MutationTemplate(
        name=f"pg-auto-{rng.randrange(10_000)}",
        weight=1.0,
        conditions={"parameter_golf_only": True},
        actions=actions,
    )


def _random_parameter_golf_attn_action(spec: ArchitectureSpec, rng: random.Random) -> ActionMap:
    kind = rng.choice(["GQA", "MQA", "MLA"])
    params: TemplateAction = {
        "selector": rng.choice(["random", "random_dense"]),
        "kind": kind,
        "qk_norm": rng.choice(["none", "rms", "layer"]),
        "softcap": rng.choice([8.0, 12.0, 16.0, 24.0]),
    }
    if kind == "GQA":
        params["kv_groups"] = rng.choice([1, 2, 4])
    else:
        params["kv_latent_dim"] = rng.choice([64, 96, 128, 160])
        params["value_glu"] = rng.choice([True, False])
    return {"tune_attn": params}


def _random_parameter_golf_ffn_action(spec: ArchitectureSpec, rng: random.Random) -> ActionMap:
    return {
        "tune_ffn": {
            "selector": rng.choice(["random", "random_dense"]),
            "hidden_mult": rng.choice([2.0, 2.5, 3.0, 3.5, 4.0]),
            "activation": rng.choice(["swiglu", "gelu", "relu_squared", "silu"]),
        }
    }


def _random_parameter_golf_optimizer_action(
    spec: ArchitectureSpec, rng: random.Random
) -> ActionMap:
    return {
        "tune_optimizer": {
            "optimizer_name": rng.choice(["adamw", "muon"]),
            "lr": rng.choice([7.0e-4, 8.5e-4, 1.0e-3, 1.15e-3]),
            "matrix_lr": rng.choice([0.02, 0.03, 0.04, 0.05]),
            "scalar_lr": rng.choice([0.02, 0.03, 0.04, 0.05]),
            "tied_embedding_lr": rng.choice([0.03, 0.04, 0.05, 0.06]),
            "tied_embedding_export_dtype": rng.choice(["int8", "fp16"]),
            "tied_embed_init_std": rng.choice([0.003, 0.004, 0.005, 0.006, 0.0075]),
            "gradient_transform_mode": rng.choice(
                ["identity", "normalize", "orthogonalize_2d", "sign_orthogonalize_2d"]
            ),
            "gradient_transform_ns_steps": rng.choice([3, 5, 7]),
            "gradient_transform_eps": rng.choice([1.0e-8, 1.0e-7, 1.0e-6]),
            "update_filter_mode": rng.choice(["none", "bernoulli", "topk"]),
            "update_filter_keep_ratio": rng.choice([0.5, 0.65, 0.8, 1.0]),
            "update_filter_granularity": rng.choice(["element", "block"]),
            "update_filter_block_size": rng.choice([64, 128, 256]),
            "update_filter_momentum_blend": rng.choice([0.0, 0.25, 0.5]),
            "warmup": rng.choice([16, 32, 48, 64, 96]),
            "warmdown_steps": rng.choice([0, 200, 600, 1200, 2000]),
            "muon_momentum_warmup_start": rng.choice([0.75, 0.8, 0.85, 0.9]),
            "muon_momentum_warmup_steps": rng.choice([0, 100, 250, 500, 1000]),
            "clip": rng.choice([0.75, 1.0, 1.25]),
        }
    }


def _random_parameter_golf_recurrence_action(
    spec: ArchitectureSpec, rng: random.Random
) -> ActionMap:
    tail_cap = max(2, min(6, len(spec.model.blocks)))
    tail_blocks = rng.choice(list(range(2, tail_cap + 1)))
    train_recurrence = rng.choice([1, 2])
    max_train_recurrence = max(train_recurrence, rng.choice([2, 3, 4]))
    return {
        "set_recurrence": {
            "tail_blocks": tail_blocks,
            "adapter": rng.choice(["linear", "gated"]),
            "concat_prelude": rng.choice([True, False]),
            "train_recurrence": train_recurrence,
            "max_train_recurrence": max_train_recurrence,
            "test_recurrences": [1, 2, 4],
        }
    }


def _tune_attn(spec: ArchitectureSpec, params: dict[str, Any], rng: random.Random) -> None:
    idx = _select_block_index(spec, params.get("selector", "random"), rng)
    if idx is None:
        return
    block = spec.model.blocks[idx]
    if not block.attn:
        return
    kind = params.get("kind")
    if kind is not None:
        resolved_kind = str(kind).upper()
        if resolved_kind in {"MHA", "GQA", "MQA", "LINEAR", "MLA"}:
            block.attn.kind = cast(AttentionKind, resolved_kind)
            if resolved_kind == "MQA":
                block.attn.kv_groups = max(1, int(block.attn.heads))
                block.attn.kv_latent_dim = None
            elif resolved_kind == "GQA":
                kv_groups = params.get("kv_groups")
                if kv_groups is None:
                    kv_groups = max(1, int(block.attn.heads) // 4)
                block.attn.kv_groups = _clamp_int(
                    int(kv_groups), lo=1, hi=max(1, int(block.attn.heads))
                )
                block.attn.kv_latent_dim = None
            elif resolved_kind == "MLA":
                latent_dim = params.get("kv_latent_dim")
                if latent_dim is None:
                    latent_dim = max(int(block.attn.head_dim), int(spec.model.emb.dim) // 4)
                block.attn.kv_latent_dim = max(1, int(latent_dim))
                block.attn.kv_groups = None
            else:
                block.attn.kv_groups = 1
                block.attn.kv_latent_dim = None
    qk_val = params.get("qk_norm_max")
    if qk_val is not None:
        block.attn.qk_norm_max = float(qk_val)
    if "gating_pos" in params and params["gating_pos"] is not None:
        block.attn.gating_pos = params["gating_pos"]
    if "gating_op" in params and params["gating_op"] is not None:
        block.attn.gating_op = params["gating_op"]
    sw_jitter = int(params.get("sw_jitter", 0))
    if sw_jitter != 0:
        current = block.attn.sw or spec.data.seq_len // 8
        block.attn.sw = max(8, min(spec.data.seq_len, int(current + sw_jitter)))
    if "sparsity" in params and block.attn is not None:
        block.attn.sparsity = params["sparsity"]
    if "block_size" in params and params["block_size"] is not None and block.attn is not None:
        block.attn.block_size = int(params["block_size"])
    if "block_stride" in params and params["block_stride"] is not None and block.attn is not None:
        block.attn.block_stride = int(params["block_stride"])
    if "global_stride" in params and params["global_stride"] is not None and block.attn is not None:
        block.attn.global_stride = int(params["global_stride"])
    if "dilation" in params and params["dilation"] is not None and block.attn is not None:
        block.attn.dilation = int(params["dilation"])
    if "kv_groups" in params and params["kv_groups"] is not None:
        block.attn.kv_groups = _clamp_int(
            int(params["kv_groups"]), lo=1, hi=max(1, int(block.attn.heads))
        )
    if "kv_latent_dim" in params and params["kv_latent_dim"] is not None:
        block.attn.kv_latent_dim = max(1, int(params["kv_latent_dim"]))
    if "value_glu" in params and params["value_glu"] is not None:
        block.attn.value_glu = bool(params["value_glu"])
    if any(key in params for key in ("qk_norm", "qk_scale", "softcap")):
        softmax = _ensure_softmax(block.attn)
        if "qk_norm" in params and params["qk_norm"] is not None:
            normalized_qk_norm = str(params["qk_norm"]).lower()
            if normalized_qk_norm in {"none", "rms", "layer"}:
                softmax.qk_norm = cast(SoftmaxNorm, normalized_qk_norm)
        if "qk_scale" in params and params["qk_scale"] is not None:
            softmax.qk_scale = params["qk_scale"]
        if "softcap" in params and params["softcap"] is not None:
            softmax.softcap = float(params["softcap"])


def _tune_ffn(spec: ArchitectureSpec, params: dict[str, Any], rng: random.Random) -> None:
    idx = _select_block_index(spec, params.get("selector", "random"), rng)
    if idx is None:
        return
    block = spec.model.blocks[idx]
    if not isinstance(block.ffn, DenseFFNConfig):
        return
    hidden = params.get("hidden")
    if hidden is None and params.get("hidden_mult") is not None:
        hidden = _round_hidden(float(spec.model.emb.dim) * float(params["hidden_mult"]))
    if hidden is not None:
        block.ffn.hidden = max(128, int(hidden))
    if params.get("activation") is not None:
        normalized_activation = str(params["activation"]).lower()
        if normalized_activation in {"silu", "gelu", "relu", "relu_squared", "swiglu"}:
            block.ffn.activation = cast(ActivationName, normalized_activation)


def _tune_optimizer(spec: ArchitectureSpec, params: dict[str, Any]) -> None:
    optimizer = spec.train.optimizer
    if params.get("optimizer_name") is not None:
        normalized_optimizer = str(params["optimizer_name"]).lower()
        if normalized_optimizer in {"adamw", "lion", "muon"}:
            optimizer.name = cast(OptimizerName, normalized_optimizer)
    if params.get("lr") is not None:
        spec.train.lr = float(params["lr"])
    if params.get("matrix_lr") is not None:
        spec.train.matrix_lr = max(1.0e-9, float(params["matrix_lr"]))
    if params.get("scalar_lr") is not None:
        spec.train.scalar_lr = max(1.0e-9, float(params["scalar_lr"]))
    if params.get("tied_embedding_lr") is not None:
        spec.train.tied_embedding_lr = max(1.0e-9, float(params["tied_embedding_lr"]))
    if params.get("tied_embedding_export_dtype") is not None and spec.parameter_golf is not None:
        normalized_export_dtype = str(params["tied_embedding_export_dtype"]).lower()
        if normalized_export_dtype in {"int8", "fp16"}:
            spec.parameter_golf.tied_embedding_export_dtype = cast(
                ExportDType,
                normalized_export_dtype,
            )
    if params.get("tied_embed_init_std") is not None:
        spec.model.emb.init_std = max(1.0e-9, float(params["tied_embed_init_std"]))
    if params.get("warmup") is not None:
        spec.train.warmup = max(0, int(params["warmup"]))
    if params.get("warmdown_steps") is not None:
        spec.train.warmdown_steps = max(0, int(params["warmdown_steps"]))
    if params.get("clip") is not None:
        spec.train.clip = max(0.0, float(params["clip"]))
    if params.get("muon_momentum_warmup_start") is not None:
        optimizer.muon_momentum_warmup_start = max(
            0.0,
            min(1.0, float(params["muon_momentum_warmup_start"])),
        )
    if params.get("muon_momentum_warmup_steps") is not None:
        optimizer.muon_momentum_warmup_steps = max(0, int(params["muon_momentum_warmup_steps"]))
    if params.get("gradient_transform_mode") is not None:
        normalized_transform_mode = str(params["gradient_transform_mode"]).lower()
        if normalized_transform_mode in {
            "identity",
            "sign",
            "normalize",
            "orthogonalize_2d",
            "sign_orthogonalize_2d",
        }:
            optimizer.gradient_transform.mode = cast(
                GradientTransformMode,
                normalized_transform_mode,
            )
    if params.get("gradient_transform_ns_steps") is not None:
        optimizer.gradient_transform.ns_steps = max(1, int(params["gradient_transform_ns_steps"]))
    if params.get("gradient_transform_eps") is not None:
        optimizer.gradient_transform.eps = max(1.0e-12, float(params["gradient_transform_eps"]))
    if params.get("update_filter_mode") is not None:
        normalized_filter_mode = str(params["update_filter_mode"]).lower()
        if normalized_filter_mode in {"none", "bernoulli", "topk"}:
            optimizer.update_filter.mode = cast(UpdateFilterMode, normalized_filter_mode)
    if params.get("update_filter_keep_ratio") is not None:
        optimizer.update_filter.keep_ratio = max(
            1.0e-3,
            min(1.0, float(params["update_filter_keep_ratio"])),
        )
    if params.get("update_filter_granularity") is not None:
        normalized_granularity = str(params["update_filter_granularity"]).lower()
        if normalized_granularity in {"element", "block"}:
            optimizer.update_filter.granularity = cast(
                UpdateFilterGranularity,
                normalized_granularity,
            )
    if params.get("update_filter_block_size") is not None:
        optimizer.update_filter.block_size = max(1, int(params["update_filter_block_size"]))
    if params.get("update_filter_momentum_blend") is not None:
        optimizer.update_filter.momentum_blend = max(
            0.0,
            min(1.0, float(params["update_filter_momentum_blend"])),
        )


def _set_recurrence(spec: ArchitectureSpec, params: dict[str, Any]) -> None:
    n_blocks = len(spec.model.blocks)
    if n_blocks <= 1:
        return
    recurrence = (
        spec.model.recurrences[0].model_copy(deep=True)
        if spec.model.recurrences
        else RecurrenceConfig(start=max(0, n_blocks - 2), end=n_blocks)
    )

    start = params.get("start")
    end = params.get("end")
    if params.get("tail_blocks") is not None:
        tail = _clamp_int(int(params["tail_blocks"]), lo=1, hi=n_blocks)
        start = max(0, n_blocks - tail)
        end = n_blocks
    if params.get("start_fraction") is not None:
        start = int(float(params["start_fraction"]) * float(n_blocks))
    if params.get("end_fraction") is not None:
        end = int(round(float(params["end_fraction"]) * float(n_blocks)))
    if start is None:
        start = recurrence.start
    if end is None:
        end = recurrence.end
    recurrence.start = _clamp_int(int(start), lo=0, hi=max(0, n_blocks - 1))
    recurrence.end = _clamp_int(int(end), lo=recurrence.start + 1, hi=n_blocks)

    if params.get("adapter") is not None:
        normalized_adapter = str(params["adapter"]).lower()
        if normalized_adapter in {"linear", "gated"}:
            recurrence.adapter = cast(RecurrenceAdapter, normalized_adapter)
    if "concat_prelude" in params and params["concat_prelude"] is not None:
        recurrence.concat_prelude = bool(params["concat_prelude"])
    if params.get("train_recurrence") is not None:
        recurrence.train_recurrence = max(1, int(params["train_recurrence"]))
    if params.get("max_train_recurrence") is not None:
        recurrence.max_train_recurrence = max(1, int(params["max_train_recurrence"]))
    if recurrence.max_train_recurrence < recurrence.train_recurrence:
        recurrence.max_train_recurrence = recurrence.train_recurrence
    if params.get("test_recurrences") is not None:
        recurrence.test_recurrences = [max(1, int(item)) for item in params["test_recurrences"]]

    if spec.model.recurrences:
        spec.model.recurrences[0] = recurrence
        return
    spec.model.recurrences.append(recurrence)


def _tune_router(spec: ArchitectureSpec, params: dict[str, Any], rng: random.Random) -> None:
    # Global training coefficients
    if "lb_coeff" in params:
        spec.train.router_lb_coeff = float(params["lb_coeff"])
    if "entropy_coeff" in params:
        spec.train.router_entropy_coeff = float(params["entropy_coeff"])
    # Per-block MoE temperature
    candidates = [b for b in spec.model.blocks if b.ffn and getattr(b.ffn, "type", "") == "moe"]
    if not candidates:
        return
    block = rng.choice(candidates)
    temp = float(params.get("temperature", 1.0))
    if isinstance(block.ffn, MoEFFNConfig):
        block.ffn.router_temperature = temp
