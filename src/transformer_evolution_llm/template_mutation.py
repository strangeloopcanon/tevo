"""Template-driven mutation engine for architectural edits."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict, cast

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
    RetroConfig,
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


ActionMap = dict[str, TemplateAction]


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
    _TEMPLATE_RECENT[template.name] = template
    return template.name, new_spec


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
    if conditions.get("requires_ssm_block"):
        if not any(block.ssm for block in blocks):
            return False
    if conditions.get("requires_dense_ffn"):
        if not any(isinstance(block.ffn, DenseFFNConfig) for block in blocks):
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
    elif "tune_router" in action:
        _tune_router(spec, dict(action["tune_router"]), rng)


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
    return BlockConfig(attn=attn, ffn=ffn, ssm=ssm, extras=extras)


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


def _tune_attn(spec: ArchitectureSpec, params: dict[str, Any], rng: random.Random) -> None:
    idx = _select_block_index(spec, params.get("selector", "random"), rng)
    if idx is None:
        return
    block = spec.model.blocks[idx]
    if not block.attn:
        return
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
