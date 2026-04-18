"""Architecture scoring, distance metrics, and composite metric helpers.

Extracted from orchestrator.py for readability.
"""

from __future__ import annotations

import math
from collections import Counter

from .attention_patterns import resolve_attention_pattern
from .candidates import ObjectiveDirection
from .dsl import ArchitectureSpec, CompositeMetricConfig

ObjectiveDir = dict[str, ObjectiveDirection]


def default_objectives() -> ObjectiveDir:
    return {
        "ppl_code": "min",
        "ppl_math": "min",
        "long_recall": "max",
        "throughput": "max",
        "ram": "min",
        "layers": "max",
        "effective_depth": "max",
        "physical_depth": "min",
        "moe_blocks": "max",
        "novelty": "max",
        "instability": "min",
        "graph_entropy": "max",
        "passkey_acc": "max",
        "passkey_loss": "min",
        "speedrun_steps_to_target": "min",
        "speedrun_tokens_to_target": "min",
        "speedrun_time_to_target": "min",
        "speedrun_best_eval_loss": "min",
        "val_bpb": "min",
        "post_quant_val_bpb": "min",
        "artifact_total_bytes": "min",
        "artifact_budget_utilization_est": "max",
        "artifact_budget_utilization": "max",
        "artifact_budget_fill_score_est": "max",
        "artifact_budget_fill_score": "max",
        "artifact_budget_edge_score_est": "max",
        "artifact_budget_edge_score": "max",
        "complexity_score": "max",
        "quantization_delta_bpb": "min",
        "official_submission_code_bytes_est": "min",
        "official_submission_patch_bytes_est": "min",
        "official_submission_total_bytes_est": "min",
        "official_submission_unsupported_patch_reason_count": "min",
        "main_track_eligible_est": "max",
        "main_track_eligible_exact": "max",
        "motif_transfer_eligible": "max",
        "motif_anchor_delta_bpb": "max",
    }


def artifact_budget_utilization(total_bytes: float | None, budget_bytes: float | None) -> float:
    """Return how much of the artifact budget a candidate uses."""
    if total_bytes is None or budget_bytes is None:
        return 0.0
    if float(budget_bytes) <= 0.0:
        return 0.0
    return float(total_bytes) / float(budget_bytes)


def artifact_budget_fill_score(total_bytes: float | None, budget_bytes: float | None) -> float:
    """Reward candidates that use most of the budget without drifting far over it."""
    utilization = artifact_budget_utilization(total_bytes, budget_bytes)
    if utilization <= 1.0:
        return utilization
    overflow = utilization - 1.0
    return max(0.0, 1.0 - 4.0 * overflow)


def artifact_budget_edge_score(
    total_bytes: float | None,
    budget_bytes: float | None,
    *,
    target_utilization: float = 0.96,
    under_window: float = 0.35,
    over_window: float = 0.10,
) -> float:
    """Reward candidates that sit near the budget edge without going far over.

    This is stronger than plain utilization: it gives little credit to models
    that are far under-filled, peaks just below the cap, and drops once they
    drift meaningfully over budget.
    """
    utilization = artifact_budget_utilization(total_bytes, budget_bytes)
    if utilization <= 0.0:
        return 0.0
    if utilization <= target_utilization:
        distance = target_utilization - utilization
        return max(0.0, 1.0 - distance / max(under_window, 1.0e-9))
    overflow = utilization - target_utilization
    return max(0.0, 1.0 - overflow / max(over_window, 1.0e-9))


def complexity_score(spec: ArchitectureSpec) -> float:
    """Estimate how structurally rich a model is.

    This is intentionally simple: it rewards depth, wider FFNs, longer context,
    and extra architectural structure without pretending to be a quality metric.
    """
    model_dim = max(1, int(spec.model.emb.dim))
    seq_len = max(1, int(spec.data.seq_len))
    logical_layers = float(spec.model.n_layers)
    physical_layers = float(spec.model.physical_block_count())
    shared_blocks = float(spec.model.shared_block_count())
    moe_blocks = 0.0
    memory_blocks = 0.0
    embedding_ffn_blocks = 0.0
    ssm_blocks = 0.0
    selector_blocks = 0.0
    nonstandard_attn_blocks = 0.0
    sparse_blocks = 0.0
    extras_total = 0.0
    ffn_multiplier_total = 0.0
    ffn_count = 0.0

    for block in spec.model.blocks:
        for ffn in (block.ffn, getattr(block, "ffn_memory", None)):
            if ffn is None:
                continue
            hidden = int(getattr(ffn, "hidden", 0) or 0)
            if hidden > 0:
                ffn_multiplier_total += float(hidden) / float(model_dim)
                ffn_count += 1.0
            if getattr(ffn, "type", "dense") == "moe":
                moe_blocks += 1.0
            if str(getattr(ffn, "input_source", "residual") or "residual") == "embedding":
                embedding_ffn_blocks += 1.0

        if block.ssm is not None:
            ssm_blocks += 1.0

        extras_total += float(len(block.extras))
        if block.extras:
            memory_types = {
                "retro",
                "assoc_memory",
                "memory_tokens",
                "chunk_memory",
                "lookup_memory",
            }
            if any(
                getattr(extra, "type", type(extra).__name__) in memory_types
                for extra in block.extras
            ):
                memory_blocks += 1.0

        attn = block.attn
        if attn is None:
            continue
        kind = str(getattr(attn, "kind", "MHA") or "MHA").upper()
        if kind != "MHA":
            nonstandard_attn_blocks += 1.0
        if str(getattr(attn, "selector", "none") or "none") != "none":
            selector_blocks += 1.0
        pattern = resolve_attention_pattern(attn)
        if pattern.sparsity != "none":
            sparse_blocks += 1.0

    avg_ffn_multiplier = ffn_multiplier_total / ffn_count if ffn_count > 0 else 0.0
    hyper = getattr(spec.model, "hyper", None)
    hyper_streams = int(getattr(hyper, "streams", 1) or 1) if hyper is not None else 1
    recurrence_count = float(len(spec.model.recurrences))
    context_bonus = max(0.0, math.log2(float(seq_len) / 1024.0)) if seq_len > 1024 else 0.0

    score = 0.0
    score += logical_layers
    score += 0.35 * physical_layers
    score += max(0.0, avg_ffn_multiplier - 2.0)
    score += 1.5 * context_bonus
    score += 1.25 * moe_blocks
    score += 1.0 * memory_blocks
    score += 0.75 * recurrence_count
    score += 0.5 * embedding_ffn_blocks
    score += 0.5 * ssm_blocks
    score += 0.4 * selector_blocks
    score += 0.35 * nonstandard_attn_blocks
    score += 0.25 * sparse_blocks
    score += 0.2 * extras_total
    score += 0.2 * shared_blocks
    score += 0.25 * max(0, hyper_streams - 1)
    score += 0.25 * graph_entropy(spec)
    return float(score)


def structural_distance(a: ArchitectureSpec, b: ArchitectureSpec) -> float:
    """Normalised structural distance between two architecture specs."""
    la, lb = a.model.n_layers, b.model.n_layers
    diff: float = float(abs(la - lb))
    for i in range(min(la, lb)):
        ba = a.model.blocks[i]
        bb = b.model.blocks[i]
        ta = getattr(ba.ffn, "type", None) if ba.ffn else None
        tb = getattr(bb.ffn, "type", None) if bb.ffn else None
        if ta != tb:
            diff += 1.0
        sa = getattr(ba.ffn, "input_source", "residual") if ba.ffn else None
        sb = getattr(bb.ffn, "input_source", "residual") if bb.ffn else None
        if sa != sb:
            diff += 0.5
        tma = getattr(getattr(ba, "ffn_memory", None), "type", None)
        tmb = getattr(getattr(bb, "ffn_memory", None), "type", None)
        if tma != tmb:
            diff += 0.75
        sma = getattr(getattr(ba, "ffn_memory", None), "input_source", "residual")
        smb = getattr(getattr(bb, "ffn_memory", None), "input_source", "residual")
        if sma != smb:
            diff += 0.25
        if bool(ba.ssm) != bool(bb.ssm):
            diff += 1.0
        if len(ba.extras) != len(bb.extras):
            diff += 0.5
        if (ba.share_with is None) != (bb.share_with is None):
            diff += 0.5
        elif (
            ba.share_with is not None
            and bb.share_with is not None
            and ba.share_with != bb.share_with
        ):
            diff += 0.25
        if ba.attn and bb.attn:
            pa = resolve_attention_pattern(ba.attn)
            pb = resolve_attention_pattern(bb.attn)
            if (ba.attn.kind or "MHA") != (bb.attn.kind or "MHA"):
                diff += 0.5
            if (ba.attn.kv_groups or ba.attn.heads) != (bb.attn.kv_groups or bb.attn.heads):
                diff += 0.5
            if (ba.attn.rope or None) != (bb.attn.rope or None):
                diff += 0.5
            if bool(getattr(ba.attn, "alibi", False)) != bool(getattr(bb.attn, "alibi", False)):
                diff += 0.25
            if pa.sparsity != pb.sparsity:
                diff += 0.5
            if (pa.sw or None) != (pb.sw or None):
                diff += 0.25
            if (pa.global_stride or None) != (pb.global_stride or None):
                diff += 0.25
            if (pa.block_size or None) != (pb.block_size or None):
                diff += 0.25
            if (pa.block_stride or None) != (pb.block_stride or None):
                diff += 0.25
    # Recurrence differences matter for novelty
    rec_a = [(r.start, r.end, r.adapter, r.concat_prelude) for r in a.model.recurrences]
    rec_b = [(r.start, r.end, r.adapter, r.concat_prelude) for r in b.model.recurrences]
    diff += abs(len(rec_a) - len(rec_b)) * 0.5
    for idx in range(min(len(rec_a), len(rec_b))):
        if rec_a[idx] != rec_b[idx]:
            diff += 0.5
    hyper_a = getattr(a.model, "hyper", None)
    hyper_b = getattr(b.model, "hyper", None)
    streams_a = int(getattr(hyper_a, "streams", 1) or 1) if hyper_a is not None else 1
    streams_b = int(getattr(hyper_b, "streams", 1) or 1) if hyper_b is not None else 1
    if streams_a != streams_b:
        diff += 0.5
    denom = max(1.0, 0.5 * float(la + lb))
    return float(diff) / denom


def prior_distance(spec: ArchitectureSpec) -> float:
    """Gentle distance from a typical architectural manifold.

    Measures how far heads/FFN/windows deviate from common defaults.
    """
    d_model = spec.model.emb.dim
    seq_len = spec.data.seq_len
    scale = spec.priors.window_scale
    rope_default = spec.priors.rope_theta_default
    target_ffn = 4.0 * d_model
    target_hd = 64.0
    target_kv = 2.0
    target_sw = math.sqrt(max(1.0, float(seq_len))) * scale
    dist = 0.0
    count = 1e-6
    for block in spec.model.blocks:
        if block.attn:
            pattern = resolve_attention_pattern(block.attn)
            count += 1.0
            dist += abs(float(block.attn.head_dim) - target_hd) / target_hd
            if block.attn.kv_groups is not None:
                dist += abs(float(block.attn.kv_groups) - target_kv) / target_kv * 0.5
            if pattern.sparsity == "local_global":
                if pattern.sw is not None:
                    dist += abs(float(pattern.sw) - target_sw) / max(1.0, target_sw) * 0.5
                if pattern.global_stride is not None:
                    target_g = max(1.0, math.sqrt(float(seq_len)))
                    dist += abs(float(pattern.global_stride) - target_g) / target_g * 0.25
            if block.attn.rope_theta is not None:
                dist += abs(float(block.attn.rope_theta) - rope_default) / rope_default * 0.25
        if block.ffn is not None and getattr(block.ffn, "type", "dense") == "dense":
            count += 1.0
            dist += abs(float(block.ffn.hidden) - target_ffn) / target_ffn
    return float(dist / count)


def graph_entropy(spec: ArchitectureSpec) -> float:
    """Shannon entropy over the architectural token stream."""
    tokens: list[str] = []
    hyper = getattr(spec.model, "hyper", None)
    streams = int(getattr(hyper, "streams", 1) or 1) if hyper is not None else 1
    if streams > 1:
        tokens.append(f"hyper:{streams}")
    for block in spec.model.blocks:
        if block.attn:
            pattern = resolve_attention_pattern(block.attn)
            tokens.append(f"attn:{block.attn.kind}")
            tokens.append(f"sparsity:{pattern.sparsity}")
            if block.attn.gating_pos and block.attn.gating_pos != "none":
                tokens.append(f"gate:{block.attn.gating_pos}-{block.attn.gating_op or 'dense'}")
        if block.share_with is not None:
            tokens.append("share:block")
        if block.ffn:
            src = getattr(block.ffn, "input_source", "residual") or "residual"
            tokens.append(f"ffn:{getattr(block.ffn, 'type', 'dense')}:{src}")
        ffn_memory = getattr(block, "ffn_memory", None)
        if ffn_memory:
            src = getattr(ffn_memory, "input_source", "residual") or "residual"
            tokens.append(f"ffn_mem:{getattr(ffn_memory, 'type', 'dense')}:{src}")
        if block.ssm:
            tokens.append(f"ssm:{block.ssm.kind}")
        for extra in block.extras:
            tokens.append(f"extra:{getattr(extra, 'type', type(extra).__name__)}")
    for rec in spec.model.recurrences:
        tokens.append(f"rec:{rec.start}-{rec.end}")
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = sum(counts.values())
    probs = [c / total for c in counts.values() if c > 0]
    entropy = -sum(p * math.log(p) for p in probs)
    diversity = len(counts)
    depth_bonus = math.log1p(spec.model.n_layers)
    return float(entropy + 0.05 * diversity + 0.1 * depth_bonus)


def behavioral_descriptor(spec: ArchitectureSpec) -> list[float]:
    """Encode architecture structure as a fixed-size novelty descriptor."""
    layers = max(1, int(spec.model.n_layers))
    moe_blocks = 0
    embed_ffn_blocks = 0
    ssm_blocks = 0
    selector_blocks = 0
    linear_blocks = 0
    mla_blocks = 0
    sparse_blocks = 0
    qk_norm_blocks = 0
    extras_total = 0
    head_dim_sum = 0.0
    kv_groups_sum = 0.0
    attn_blocks = 0
    shared_blocks = 0

    for block in spec.model.blocks:
        if block.share_with is not None:
            shared_blocks += 1
        ffns = [block.ffn, getattr(block, "ffn_memory", None)]
        has_embed_ffn = False
        for ffn in ffns:
            if ffn is None:
                continue
            if getattr(ffn, "type", "dense") == "moe":
                moe_blocks += 1
            if str(getattr(ffn, "input_source", "residual") or "residual") == "embedding":
                has_embed_ffn = True
        if has_embed_ffn:
            embed_ffn_blocks += 1
        if block.ssm is not None:
            ssm_blocks += 1
        extras_total += len(block.extras)
        attn = block.attn
        if attn is None:
            continue
        attn_blocks += 1
        kind = str(getattr(attn, "kind", "MHA") or "MHA").upper()
        if kind == "LINEAR":
            linear_blocks += 1
        if kind == "MLA":
            mla_blocks += 1
        if str(getattr(attn, "selector", "none") or "none") != "none":
            selector_blocks += 1
        if getattr(attn, "qk_norm_max", None) is not None:
            qk_norm_blocks += 1
        pattern = resolve_attention_pattern(attn)
        if pattern.sparsity != "none":
            sparse_blocks += 1
        head_dim_sum += float(getattr(attn, "head_dim", 0) or 0)
        kv_groups_sum += float(getattr(attn, "kv_groups", 1) or 1)

    recurrences = len(spec.model.recurrences)
    attn_den = float(max(1, attn_blocks))
    layers_f = float(layers)
    opt_cfg = spec.train.optimizer
    opt_name = str(getattr(opt_cfg, "name", "adamw") or "adamw").lower()
    opt_family = {"adamw": 0.0, "lion": 1.0, "muon": 2.0}.get(opt_name, -1.0)
    grad_transform = getattr(opt_cfg, "gradient_transform", None)
    grad_mode = str(getattr(grad_transform, "mode", "identity") or "identity").lower()
    grad_mode_value = {
        "identity": 0.0,
        "sign": 1.0,
        "normalize": 2.0,
        "orthogonalize_2d": 3.0,
        "sign_orthogonalize_2d": 4.0,
    }.get(grad_mode, -1.0)
    grad_ns_steps = float(getattr(grad_transform, "ns_steps", 5) or 5)
    grad_eps = float(getattr(grad_transform, "eps", 1e-8) or 1e-8)
    grad_eps_log10 = math.log10(max(1e-12, grad_eps))
    update_filter = getattr(opt_cfg, "update_filter", None)
    mode = str(getattr(update_filter, "mode", "none") or "none").lower()
    granularity = str(getattr(update_filter, "granularity", "element") or "element").lower()
    mask_mode = {"none": 0.0, "bernoulli": 1.0, "topk": 2.0}.get(mode, -1.0)
    mask_granularity = {"element": 0.0, "block": 1.0}.get(granularity, -1.0)
    mask_keep_ratio = float(getattr(update_filter, "keep_ratio", 1.0) or 1.0)
    mask_momentum_blend = float(getattr(update_filter, "momentum_blend", 0.0) or 0.0)
    descriptor = [
        layers_f,
        float(spec.model.physical_block_count()) / layers_f,
        float(shared_blocks) / layers_f,
        float(moe_blocks) / layers_f,
        float(embed_ffn_blocks) / layers_f,
        float(ssm_blocks) / layers_f,
        float(selector_blocks) / layers_f,
        float(linear_blocks) / layers_f,
        float(mla_blocks) / layers_f,
        float(sparse_blocks) / layers_f,
        float(extras_total) / layers_f,
        float(recurrences) / layers_f,
        head_dim_sum / attn_den,
        kv_groups_sum / attn_den,
        graph_entropy(spec),
        opt_family,
        grad_mode_value,
        grad_ns_steps,
        grad_eps_log10,
        mask_mode,
        mask_granularity,
        mask_keep_ratio,
        mask_momentum_blend,
    ]
    return descriptor


def archive_novelty(
    descriptor: list[float],
    archive: list[list[float]],
    *,
    k: int = 15,
) -> float:
    """Compute novelty as average distance to k nearest archive descriptors."""
    if not archive:
        return 0.0
    if not descriptor:
        return 0.0
    k_eff = max(1, min(int(k), len(archive)))
    distances: list[float] = []
    for item in archive:
        dim = min(len(descriptor), len(item))
        if dim <= 0:
            continue
        total = 0.0
        for idx in range(dim):
            delta = float(descriptor[idx]) - float(item[idx])
            total += delta * delta
        distances.append(math.sqrt(total))
    if not distances:
        return 0.0
    distances.sort()
    nearest = distances[:k_eff]
    return float(sum(nearest) / len(nearest))


def compute_composite(comp: CompositeMetricConfig, metrics: dict[str, float]) -> float | None:
    """Evaluate a single composite metric expression against a metrics dict."""
    try:
        if comp.op == "ratio":
            if not comp.numerator or not comp.denominator:
                return None
            num = metrics.get(comp.numerator)
            den = metrics.get(comp.denominator)
            if num is None or den is None:
                return None
            denom = den if abs(den) >= comp.epsilon else comp.epsilon
            return float(num) / float(denom)
        if comp.op == "product":
            if not comp.numerator or not comp.denominator:
                return None
            num = metrics.get(comp.numerator)
            den = metrics.get(comp.denominator)
            if num is None or den is None:
                return None
            return float(num) * float(den)
        if comp.op == "weighted_sum":
            if not comp.terms:
                return None
            total = 0.0
            for name, weight in comp.terms.items():
                val = metrics.get(name)
                if val is None:
                    return None
                total += float(weight) * float(val)
            return total
    except Exception:
        return None
    return None


def default_composites() -> list[CompositeMetricConfig]:
    """Built-in composite metrics applied to every candidate."""
    return [
        CompositeMetricConfig(
            name="ppl_per_long_recall",
            op="ratio",
            numerator="ppl_code",
            denominator="long_recall",
            epsilon=1e-3,
        ),
        CompositeMetricConfig(
            name="ppl_per_param",
            op="ratio",
            numerator="ppl_code",
            denominator="params",
            epsilon=1e-6,
        ),
        CompositeMetricConfig(
            name="ppl_per_throughput",
            op="ratio",
            numerator="ppl_code",
            denominator="throughput",
            epsilon=1e-6,
        ),
    ]


def merge_composites(
    primary: list[CompositeMetricConfig], defaults: list[CompositeMetricConfig]
) -> list[CompositeMetricConfig]:
    """Merge user-specified composites with built-in defaults (no duplicates)."""
    existing = {comp.name for comp in primary}
    merged = list(primary)
    for comp in defaults:
        if comp.name not in existing:
            merged.append(comp)
    return merged
