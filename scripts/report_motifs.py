"""Summarize architectural motifs in a frontier JSON (optionally with lineage).

This is meant for quick "validation" narratives:
- Did evolution rediscover MoE / MLA / memory / sparsity / SSM / recurrence?
- Which frontier entries express each motif?

When a motif shows up in the lineage but not in completed/frontier entries, it
usually means rung0 constraints filtered those candidates out.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


def _as_list(val: Any) -> list[Any]:
    if isinstance(val, list):
        return val
    return []


def _has_motif(spec: dict[str, Any]) -> dict[str, bool]:
    model = spec.get("model", {}) if isinstance(spec, dict) else {}
    blocks = _as_list(model.get("blocks"))
    recurrences = _as_list(model.get("recurrences"))

    has_moe = False
    has_mla = False
    has_selector = False
    has_sparsity = False
    has_memory = False
    has_ssm = False
    has_recurrence = bool(recurrences)
    has_qk_norm = False

    for block in blocks:
        if not isinstance(block, dict):
            continue
        attn = block.get("attn")
        if isinstance(attn, dict):
            kind = str(attn.get("kind") or "MHA").upper()
            if kind == "MLA":
                has_mla = True
            sparsity = str(attn.get("sparsity") or "none").lower()
            if sparsity != "none" or attn.get("sw") is not None:
                has_sparsity = True
            selector = str(attn.get("selector") or "none").lower()
            if selector != "none":
                has_selector = True
            if attn.get("qk_norm_max") is not None:
                has_qk_norm = True

        ffn = block.get("ffn")
        if isinstance(ffn, dict) and str(ffn.get("type") or "dense").lower() == "moe":
            has_moe = True

        if block.get("ssm") is not None:
            has_ssm = True

        for extra in _as_list(block.get("extras")):
            if not isinstance(extra, dict):
                continue
            extra_type = str(extra.get("type") or "").lower()
            if extra_type in {"retro", "assoc_memory", "memory_tokens", "chunk_memory"}:
                has_memory = True

    return {
        "moe": has_moe,
        "mla": has_mla,
        "selector": has_selector,
        "sparsity": has_sparsity,
        "memory": has_memory,
        "ssm": has_ssm,
        "recurrence": has_recurrence,
        "qk_norm": has_qk_norm,
    }


def _count_primitives(spec: dict[str, Any]) -> Counter[str]:
    model = spec.get("model", {}) if isinstance(spec, dict) else {}
    blocks = _as_list(model.get("blocks"))
    counts: Counter[str] = Counter()
    for block in blocks:
        if not isinstance(block, dict):
            continue
        attn = block.get("attn")
        if isinstance(attn, dict):
            kind = str(attn.get("kind") or "MHA").upper()
            counts[f"attn:{kind}"] += 1
            sparsity = str(attn.get("sparsity") or "none").lower()
            if sparsity != "none":
                counts[f"sparsity:{sparsity}"] += 1
            if attn.get("sw") is not None:
                counts["sliding_window"] += 1
            selector = str(attn.get("selector") or "none").lower()
            if selector != "none":
                counts[f"selector:{selector}"] += 1
            if attn.get("qk_norm_max") is not None:
                counts["qk_norm"] += 1
        ffn = block.get("ffn")
        if isinstance(ffn, dict):
            ffn_type = str(ffn.get("type") or "dense").lower()
            counts[f"ffn:{ffn_type}"] += 1
            if ffn_type == "moe":
                try:
                    counts[f"moe_experts:{int(ffn.get('n_experts') or 0)}"] += 1
                except Exception:
                    counts["moe_experts:unknown"] += 1
        if block.get("ssm") is not None:
            counts["ssm"] += 1
        for extra in _as_list(block.get("extras")):
            if not isinstance(extra, dict):
                continue
            extra_type = str(extra.get("type") or "unknown").lower()
            counts[f"extra:{extra_type}"] += 1
    recurrences = _as_list(model.get("recurrences"))
    if recurrences:
        counts["recurrences"] += len(recurrences)
    return counts


def _load_rung0_thresholds(frontier_path: Path) -> dict[str, float] | None:
    """Best-effort rung0 thresholds for interpreting lineage failures.

    Reads `frontier.manifest.json` next to the frontier file, then loads the
    referenced config YAML and extracts evolution.rung0_thresholds.
    """
    manifest_path = frontier_path.with_name(frontier_path.stem + ".manifest.json")
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:
        return None
    if not isinstance(manifest, dict):
        return None
    cfg_path = manifest.get("config")
    if not cfg_path:
        return None
    try:
        cfg_raw = yaml.safe_load(Path(str(cfg_path)).read_text())
    except Exception:
        return None
    if not isinstance(cfg_raw, dict):
        return None
    evo = cfg_raw.get("evolution")
    if not isinstance(evo, dict):
        return None
    thresholds = evo.get("rung0_thresholds")
    if not isinstance(thresholds, dict):
        return None

    parsed: dict[str, float] = {}
    for key in ("max_params", "max_kv_bytes_per_token", "min_throughput_proxy"):
        val = thresholds.get(key)
        if val is None:
            continue
        try:
            parsed[key] = float(val)
        except (TypeError, ValueError):
            continue
    return parsed or None


def _summarize_motifs(entries: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        spec = entry.get("spec", {})
        if not isinstance(spec, dict):
            continue
        for name, present in _has_motif(spec).items():
            if present:
                counts[name] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize frontier architectural motifs.")
    parser.add_argument("frontier", type=Path, help="Path to frontier.json")
    parser.add_argument(
        "--lineage",
        type=Path,
        default=None,
        help="Optional path to frontier_lineage.json (prints attempted vs completed motif coverage).",
    )
    parser.add_argument("--top", type=int, default=10, help="How many example IDs to print")
    args = parser.parse_args()

    entries = json.loads(args.frontier.read_text())
    if not isinstance(entries, list):
        raise SystemExit("frontier JSON must be a list of entries")
    if not entries:
        raise SystemExit("frontier JSON is empty")

    motif_counts: Counter[str] = Counter()
    primitive_counts: Counter[str] = Counter()
    motif_examples: dict[str, list[tuple[float, str]]] = {
        k: [] for k in _has_motif(entries[0].get("spec", {}))
    }

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        cid = str(entry.get("id") or "")
        spec = entry.get("spec", {})
        if not isinstance(spec, dict):
            continue
        metrics = entry.get("metrics", {}) if isinstance(entry.get("metrics"), dict) else {}
        score = float(metrics.get("ppl_per_long_recall", metrics.get("ppl_code", 1e9)))

        motifs = _has_motif(spec)
        for name, present in motifs.items():
            if present:
                motif_counts[name] += 1
                motif_examples[name].append((score, cid))
        primitive_counts.update(_count_primitives(spec))

    order = ["moe", "mla", "selector", "sparsity", "memory", "ssm", "recurrence", "qk_norm"]

    n = len(entries)
    print(f"Frontier entries: {n}")

    if args.lineage is not None:
        payload = json.loads(args.lineage.read_text())
        if isinstance(payload, dict):
            lineage = payload.get("nodes", [])
        else:
            lineage = payload
        if not isinstance(lineage, list):
            raise SystemExit("lineage JSON must be a list of entries")
        completed = [
            row for row in lineage if isinstance(row, dict) and row.get("status") == "completed"
        ]
        motif_attempted = _summarize_motifs([row for row in lineage if isinstance(row, dict)])
        motif_completed = _summarize_motifs(completed)
        print(f"Lineage nodes: {len(lineage)} (completed {len(completed)})")

        print("\nMotifs (attempted / completed / frontier):")
        for name in order:
            attempted = motif_attempted.get(name, 0)
            done = motif_completed.get(name, 0)
            front = motif_counts.get(name, 0)
            print(f"- {name:10s}: {attempted:3d} / {done:3d} / {front:3d}")

        thresholds = _load_rung0_thresholds(args.frontier)
        if thresholds is not None:
            max_params = thresholds.get("max_params")
            max_kv = thresholds.get("max_kv_bytes_per_token")
            min_tps = thresholds.get("min_throughput_proxy")
            if max_params is not None or max_kv is not None or min_tps is not None:
                bits = []
                if max_params is not None:
                    bits.append(f"max_params={max_params:g}")
                if max_kv is not None:
                    bits.append(f"max_kv_bytes_per_token={max_kv:g}")
                if min_tps is not None:
                    bits.append(f"min_throughput_proxy={min_tps:g}")
                if bits:
                    print("\nGate hints (for motifs with 0 completed):")
                    print("rung0=" + ", ".join(bits))

                for name in order:
                    if motif_attempted.get(name, 0) == 0 or motif_completed.get(name, 0) != 0:
                        continue
                    viol_params = viol_kv = viol_tps = 0
                    total = 0
                    for row in lineage:
                        if not isinstance(row, dict):
                            continue
                        spec = row.get("spec", {})
                        if not isinstance(spec, dict):
                            continue
                        if not _has_motif(spec).get(name, False):
                            continue
                        metrics = row.get("metrics", {}) if isinstance(row.get("metrics"), dict) else {}
                        total += 1
                        if max_params is not None and float(metrics.get("params", 0.0) or 0.0) > float(max_params):
                            viol_params += 1
                        if max_kv is not None and float(metrics.get("kv_bytes_per_token", 0.0) or 0.0) > float(max_kv):
                            viol_kv += 1
                        if min_tps is not None and float(metrics.get("throughput_proxy", 0.0) or 0.0) < float(min_tps):
                            viol_tps += 1
                    hints = []
                    if max_params is not None:
                        hints.append(f"params {viol_params}/{total}")
                    if max_kv is not None:
                        hints.append(f"kv {viol_kv}/{total}")
                    if min_tps is not None:
                        hints.append(f"tps {viol_tps}/{total}")
                    if hints:
                        print(f"- {name:10s}: {', '.join(hints)}")

    print("\nMotifs (% of frontier):")
    for name in order:
        count = motif_counts.get(name, 0)
        pct = 100.0 * count / max(1, n)
        print(f"- {name:10s}: {count:3d} ({pct:5.1f}%)")

    print("\nCommon primitives:")
    for key, count in primitive_counts.most_common(25):
        print(f"- {key}: {count}")

    print("\nExample IDs (sorted by ppl_per_long_recall then ppl_code):")
    for name in order:
        examples = sorted(motif_examples.get(name, []))[: args.top]
        if not examples:
            continue
        ids = ", ".join(cid for _score, cid in examples)
        print(f"- {name:10s}: {ids}")


if __name__ == "__main__":
    main()
