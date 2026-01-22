"""Render shareable PNG summaries from run artifacts.

Goal: produce visually comparable "architecture barcodes" from JSON/TXT artifacts
so readers can compare patterns by eye without relying on explicit labeling.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle
except Exception as exc:  # pragma: no cover - optional helper script
    raise SystemExit(
        "scripts/render_outreach_images.py requires matplotlib. "
        "Install it in your environment (e.g. `uv pip install matplotlib`) and rerun."
    ) from exc


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())

def _pretty_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except Exception:
        return str(path)


MOTIF_ORDER: list[tuple[str, str]] = [
    ("Memory", "#2ecc71"),
    ("MoE", "#f39c12"),
    ("SSM", "#9b59b6"),
    ("Recurrence", "#e74c3c"),
    ("Sparsity", "#3498db"),
    ("Selector", "#1abc9c"),
    ("MLA", "#34495e"),
    ("QK-Norm", "#111111"),
]


def _block_has_memory(block: dict[str, Any]) -> bool:
    for extra in _as_list(block.get("extras")):
        extra_dict = _as_dict(extra)
        extra_type = str(extra_dict.get("type") or "").lower()
        if extra_type in {"retro", "assoc_memory", "memory_tokens", "chunk_memory"}:
            return True
    return False


def _per_layer_motifs(spec: dict[str, Any]) -> tuple[list[dict[str, bool]], dict[str, Any]]:
    model = _as_dict(spec.get("model"))
    blocks = _as_list(model.get("blocks"))
    recurrences = _as_list(model.get("recurrences"))
    hyper = model.get("hyper")
    hyper_streams = int(_as_dict(hyper).get("streams") or 1) if hyper is not None else 1

    spans: list[tuple[int, int]] = []
    for rec in recurrences:
        rec_dict = _as_dict(rec)
        try:
            start = int(rec_dict.get("start") or 0)
            end = int(rec_dict.get("end") or 0)
        except Exception:
            continue
        if end > start:
            spans.append((start, end))

    layer_flags: list[dict[str, bool]] = []
    for idx, block_raw in enumerate(blocks):
        block = _as_dict(block_raw)
        attn = _as_dict(block.get("attn"))
        ffn = _as_dict(block.get("ffn"))

        attn_kind = str(attn.get("kind") or "MHA").upper()
        attn_sparsity = str(attn.get("sparsity") or "none").lower()
        selector = str(attn.get("selector") or "none").lower()

        flags: dict[str, bool] = {
            "Memory": _block_has_memory(block),
            "MoE": str(ffn.get("type") or "dense").lower() == "moe",
            "SSM": block.get("ssm") is not None,
            "Recurrence": any(start <= idx < end for start, end in spans),
            "Sparsity": attn_sparsity != "none" or attn.get("sw") is not None,
            "Selector": selector != "none",
            "MLA": attn_kind == "MLA",
            "QK-Norm": attn.get("qk_norm_max") is not None,
        }
        layer_flags.append(flags)

    meta = {
        "layers": len(layer_flags),
        "hyper_streams": hyper_streams,
        "recurrence_spans": spans,
    }
    return layer_flags, meta


def _draw_arch_barcode(ax: plt.Axes, spec: dict[str, Any], *, title: str) -> None:
    layer_flags, meta = _per_layer_motifs(spec)
    layers = max(1, int(meta["layers"]))
    rows = len(MOTIF_ORDER)

    bg = "#f4f6f8"
    off = "#ffffff"

    ax.set_xlim(0, layers)
    ax.set_ylim(0, rows)
    ax.invert_yaxis()
    ax.set_facecolor(bg)

    for r, (motif, color) in enumerate(MOTIF_ORDER):
        for c in range(layers):
            present = False
            if c < len(layer_flags):
                present = bool(layer_flags[c].get(motif, False))
            face = color if present else off
            ax.add_patch(
                Rectangle(
                    (c, r),
                    1,
                    1,
                    facecolor=face,
                    edgecolor=bg,
                    linewidth=1.0,
                )
            )

    ax.set_yticks([r + 0.5 for r in range(rows)], labels=[m for m, _ in MOTIF_ORDER])
    if layers <= 8:
        ax.set_xticks([i + 0.5 for i in range(layers)], labels=[str(i + 1) for i in range(layers)])
    else:
        ticks = [0, 4, 9, 14, 19, 24, 29]
        ticks = [t for t in ticks if t < layers]
        ax.set_xticks([t + 0.5 for t in ticks], labels=[str(t + 1) for t in ticks])
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_title(title, fontsize=11, loc="left", pad=8)

    hyper_streams = int(meta.get("hyper_streams") or 1)
    if hyper_streams > 1:
        ax.text(
            layers,
            -0.4,
            f"hyper streams = {hyper_streams}",
            ha="right",
            va="top",
            fontsize=9,
            color="#555555",
        )

    for spine in ax.spines.values():
        spine.set_visible(False)


def _parse_motifs_txt(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pattern = re.compile(
        r"^\s*-\s*(?P<name>[a-zA-Z0-9_]+)\s*:\s*"
        r"(?P<attempted>\d+)\s*/\s*(?P<completed>\d+)\s*/\s*(?P<frontier>\d+)\s*$"
    )
    for line in path.read_text().splitlines():
        m = pattern.match(line)
        if not m:
            continue
        name = str(m.group("name"))
        attempted = int(m.group("attempted"))
        completed = int(m.group("completed"))
        frontier = int(m.group("frontier"))
        rows.append(
            {
                "motif": name,
                "attempted": attempted,
                "completed": completed,
                "frontier": frontier,
            }
        )
    return rows


def _render_longctx(out_path: Path, *, run_dir: Path) -> None:
    frontier_path = run_dir / "frontier.json"
    motifs_path = run_dir / "motifs.txt"
    manifest_path = run_dir / "frontier.manifest.json"

    frontier = _load_json(frontier_path)
    if not isinstance(frontier, list) or not frontier:
        raise SystemExit(f"{frontier_path} must be a non-empty list")

    def _metric(entry: dict[str, Any], key: str) -> float:
        metrics = _as_dict(entry.get("metrics"))
        try:
            return float(metrics.get(key))
        except Exception:
            return float("inf")

    best_ppl = min(frontier, key=lambda e: _metric(_as_dict(e), "ppl_code"))
    best_pass = min(frontier, key=lambda e: _metric(_as_dict(e), "passkey_loss"))
    best_ppl = _as_dict(best_ppl)
    best_pass = _as_dict(best_pass)

    motifs_rows = _parse_motifs_txt(motifs_path) if motifs_path.exists() else []
    manifest = _as_dict(_load_json(manifest_path)) if manifest_path.exists() else {}

    fig = plt.figure(figsize=(12.5, 8.0), dpi=160)
    gs = fig.add_gridspec(nrows=6, ncols=1, height_ratios=[0.8, 1.6, 1.6, 0.2, 2.3, 0.4])

    ax_title = fig.add_subplot(gs[0])
    ax_title.axis("off")
    ax_title.text(
        0.0,
        0.85,
        "Long-context frontier summary (architecture at a glance)",
        fontsize=16,
        weight="bold",
        ha="left",
        va="top",
    )
    run_line = f"Run: {run_dir.name} · gens {manifest.get('generations','?')} · steps {manifest.get('steps','?')} · eval_batches {manifest.get('eval_batches','?')} · device {manifest.get('device','?')} · seed {manifest.get('seed','?')}"
    cfg_line = f"Config: {Path(str(manifest.get('config') or '')).name or '?'} · Objectives: ppl_code↓, passkey_loss↓"
    ax_title.text(0.0, 0.45, run_line, fontsize=9.5, color="#444444", ha="left", va="top")
    ax_title.text(0.0, 0.18, cfg_line, fontsize=9.5, color="#444444", ha="left", va="top")
    ax_title.text(
        0.0,
        -0.05,
        "Architecture barcode: rows are motifs, columns are layers (filled cell = motif present in that layer).",
        fontsize=9.5,
        color="#1b5e20",
        ha="left",
        va="top",
    )

    ax_best_ppl = fig.add_subplot(gs[1])
    ax_best_pass = fig.add_subplot(gs[2])
    _draw_arch_barcode(
        ax_best_ppl,
        _as_dict(best_ppl.get("spec")),
        title=f"Frontier exemplar A (best ppl_code): {best_ppl.get('id')}",
    )
    _draw_arch_barcode(
        ax_best_pass,
        _as_dict(best_pass.get("spec")),
        title=f"Frontier exemplar B (best passkey_loss): {best_pass.get('id')}",
    )

    ax_gap = fig.add_subplot(gs[3])
    ax_gap.axis("off")

    ax_table = fig.add_subplot(gs[4])
    ax_table.axis("off")

    if motifs_rows:
        total_frontier = len(frontier)
        table_data: list[list[str]] = []
        for row in motifs_rows:
            pct = 0.0
            if total_frontier > 0:
                pct = 100.0 * float(row["frontier"]) / float(total_frontier)
            table_data.append(
                [
                    str(row["motif"]),
                    str(row["attempted"]),
                    str(row["completed"]),
                    str(row["frontier"]),
                    f"{pct:.1f}%",
                ]
            )
        col_labels = ["Motif", "Attempted", "Completed", "Frontier", "Frontier %"]
        table = ax_table.table(
            cellText=table_data,
            colLabels=col_labels,
            cellLoc="center",
            colLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9.5)
        table.scale(1.0, 1.3)

        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor("#e0e0e0")
            if r == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#f4f6f8")
            else:
                motif = table_data[r - 1][0]
                if motif == "memory":
                    cell.set_facecolor("#ecf9f1")
    else:
        ax_table.text(
            0.0,
            0.9,
            "motifs.txt not found; skipped motif coverage table.",
            fontsize=10,
            ha="left",
            va="top",
            color="#aa0000",
        )

    handles = [Patch(facecolor=color, edgecolor="none", label=name) for name, color in MOTIF_ORDER]
    ax_table.legend(
        handles=handles,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        frameon=False,
        fontsize=9,
    )

    ax_footer = fig.add_subplot(gs[5])
    ax_footer.axis("off")
    ax_footer.text(
        0.0,
        0.4,
        f"Sources: {_pretty_path(frontier_path)} and {_pretty_path(motifs_path)}",
        fontsize=8.5,
        color="#666666",
        ha="left",
        va="center",
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _has_motif_global(spec: dict[str, Any], motif: str) -> bool:
    motif_key = motif.lower()
    model = _as_dict(spec.get("model"))
    blocks = _as_list(model.get("blocks"))
    recurrences = _as_list(model.get("recurrences"))
    has_recurrence = bool(recurrences)

    has_moe = False
    has_mla = False
    has_selector = False
    has_sparsity = False
    has_memory = False
    has_ssm = False
    has_qk_norm = False

    for block_raw in blocks:
        block = _as_dict(block_raw)
        attn = _as_dict(block.get("attn"))
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

        ffn = _as_dict(block.get("ffn"))
        if str(ffn.get("type") or "dense").lower() == "moe":
            has_moe = True

        if block.get("ssm") is not None:
            has_ssm = True

        if _block_has_memory(block):
            has_memory = True

    mapping = {
        "moe": has_moe,
        "mla": has_mla,
        "selector": has_selector,
        "sparsity": has_sparsity,
        "memory": has_memory,
        "ssm": has_ssm,
        "recurrence": has_recurrence,
        "qk_norm": has_qk_norm,
    }
    return bool(mapping.get(motif_key, False))


def _render_hybrid(out_path: Path, *, frontier_path: Path, manifest_path: Path | None = None) -> None:
    frontier = _load_json(frontier_path)
    if not isinstance(frontier, list) or not frontier:
        raise SystemExit(f"{frontier_path} must be a non-empty list")

    manifest = _as_dict(_load_json(manifest_path)) if manifest_path and manifest_path.exists() else {}

    def _depth(entry: dict[str, Any]) -> int:
        spec = _as_dict(entry.get("spec"))
        model = _as_dict(spec.get("model"))
        return len(_as_list(model.get("blocks")))

    # Depth stats from specs (more reliable than metrics).
    depths = [_depth(_as_dict(entry)) for entry in frontier]
    depths_sorted = sorted(depths)
    min_depth = depths_sorted[0]
    median_depth = depths_sorted[len(depths_sorted) // 2]
    max_depth = depths_sorted[-1]

    exemplar = None
    for entry in frontier:
        entry_dict = _as_dict(entry)
        if str(entry_dict.get("id") or "") == "xover-119-1d12":
            exemplar = entry_dict
            break
    if exemplar is None:
        exemplar = _as_dict(max(frontier, key=lambda e: _depth(_as_dict(e))))

    motifs = ["memory", "sparsity", "moe", "ssm", "recurrence"]
    counts: dict[str, int] = {}
    for name in motifs:
        counts[name] = sum(
            1
            for entry in frontier
            if _has_motif_global(_as_dict(_as_dict(entry).get("spec")), name)
        )

    fig = plt.figure(figsize=(12.5, 7.2), dpi=160)
    gs = fig.add_gridspec(nrows=5, ncols=1, height_ratios=[0.9, 2.0, 0.25, 2.4, 0.4])

    ax_title = fig.add_subplot(gs[0])
    ax_title.axis("off")
    ax_title.text(
        0.0,
        0.85,
        "Hybrid frontier summary (architecture at a glance)",
        fontsize=16,
        weight="bold",
        ha="left",
        va="top",
    )
    run_line = f"Run: {frontier_path.stem} · gens {manifest.get('generations','?')} · steps {manifest.get('steps','?')} · eval_batches {manifest.get('eval_batches','?')} · device {manifest.get('device','?')} · seed {manifest.get('seed','?')} · parent_selection {manifest.get('parent_selection','?')}"
    cfg_line = f"Config: {Path(str(manifest.get('config') or '')).name or '?'}"
    ax_title.text(0.0, 0.45, run_line, fontsize=9.5, color="#444444", ha="left", va="top")
    ax_title.text(0.0, 0.18, cfg_line, fontsize=9.5, color="#444444", ha="left", va="top")
    ax_title.text(
        0.0,
        -0.05,
        f"Frontier size: {len(frontier)} · Depth: {min_depth}–{max_depth} layers (median {median_depth})",
        fontsize=9.5,
        color="#1b5e20",
        ha="left",
        va="top",
    )

    ax_barcode = fig.add_subplot(gs[1])
    exemplar_depth = _depth(exemplar)
    _draw_arch_barcode(
        ax_barcode,
        _as_dict(exemplar.get("spec")),
        title=f"Frontier exemplar (depth={exemplar_depth}): {exemplar.get('id')}",
    )

    ax_gap = fig.add_subplot(gs[2])
    ax_gap.axis("off")

    ax_bars = fig.add_subplot(gs[3])
    labels = [m.capitalize() for m in motifs]
    values = [counts[m] for m in motifs]
    colors = ["#2ecc71", "#3498db", "#f39c12", "#9b59b6", "#e74c3c"]
    ax_bars.barh(labels, values, color=colors)
    ax_bars.set_xlim(0, len(frontier))
    ax_bars.set_xlabel(f"Frontier entries (of {len(frontier)})")
    ax_bars.set_title("Motif presence across frontier entries", loc="left", fontsize=11, pad=8)
    for idx, v in enumerate(values):
        pct = 100.0 * float(v) / float(len(frontier))
        ax_bars.text(v + 0.5, idx, f"{v}/{len(frontier)} ({pct:.1f}%)", va="center", fontsize=9)
    for spine in ax_bars.spines.values():
        spine.set_visible(False)

    ax_footer = fig.add_subplot(gs[4])
    ax_footer.axis("off")
    ax_footer.text(
        0.0,
        0.4,
        f"Sources: {_pretty_path(frontier_path)}"
        + (f" and {_pretty_path(manifest_path)}" if manifest_path else ""),
        fontsize=8.5,
        color="#666666",
        ha="left",
        va="center",
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _render_frontier_overview(
    out_path: Path,
    *,
    frontier_path: Path,
    manifest_path: Path | None = None,
    title: str,
    exemplar_id: str | None = None,
    exemplar_require: tuple[str, ...] = ("memory", "sparsity", "moe", "ssm"),
    motif_bars: tuple[str, ...] = ("memory", "sparsity", "moe", "ssm", "recurrence"),
) -> None:
    frontier = _load_json(frontier_path)
    if not isinstance(frontier, list) or not frontier:
        raise SystemExit(f"{frontier_path} must be a non-empty list")

    manifest = _as_dict(_load_json(manifest_path)) if manifest_path and manifest_path.exists() else {}

    def _depth(entry: dict[str, Any]) -> int:
        spec = _as_dict(entry.get("spec"))
        model = _as_dict(spec.get("model"))
        return len(_as_list(model.get("blocks")))

    def _metric(entry: dict[str, Any], key: str) -> float | None:
        metrics = _as_dict(entry.get("metrics"))
        try:
            return float(metrics.get(key))
        except Exception:
            return None

    depths = [_depth(_as_dict(entry)) for entry in frontier]
    depths_sorted = sorted(depths)
    min_depth = depths_sorted[0]
    median_depth = depths_sorted[len(depths_sorted) // 2]
    max_depth = depths_sorted[-1]

    exemplar: dict[str, Any] | None = None
    if exemplar_id:
        for entry in frontier:
            entry_dict = _as_dict(entry)
            if str(entry_dict.get("id") or "") == exemplar_id:
                exemplar = entry_dict
                break

    if exemplar is None:
        # Prefer an entry that contains a full hybrid signature (default: memory+sparsity+moe+ssm),
        # breaking ties by ppl_per_long_recall then ppl_code, else fallback to deepest.
        candidates: list[dict[str, Any]] = []
        for entry in frontier:
            entry_dict = _as_dict(entry)
            spec = _as_dict(entry_dict.get("spec"))
            if all(_has_motif_global(spec, name) for name in exemplar_require):
                candidates.append(entry_dict)
        if candidates:
            def _sort_key(e: dict[str, Any]) -> tuple[float, float, int]:
                ppl_per = _metric(e, "ppl_per_long_recall")
                ppl = _metric(e, "ppl_code")
                return (
                    ppl_per if ppl_per is not None else float("inf"),
                    ppl if ppl is not None else float("inf"),
                    _depth(e),
                )
            exemplar = sorted(candidates, key=_sort_key)[0]
        else:
            exemplar = _as_dict(max(frontier, key=lambda e: _depth(_as_dict(e))))

    motifs = list(motif_bars)
    counts: dict[str, int] = {}
    for name in motifs:
        counts[name] = sum(
            1 for entry in frontier if _has_motif_global(_as_dict(_as_dict(entry).get("spec")), name)
        )

    fig = plt.figure(figsize=(12.5, 7.2), dpi=160)
    gs = fig.add_gridspec(nrows=5, ncols=1, height_ratios=[0.9, 2.0, 0.25, 2.4, 0.4])

    ax_title = fig.add_subplot(gs[0])
    ax_title.axis("off")
    ax_title.text(
        0.0,
        0.85,
        title,
        fontsize=16,
        weight="bold",
        ha="left",
        va="top",
    )
    run_line = (
        f"Run: {frontier_path.stem} · gens {manifest.get('generations','?')} · steps {manifest.get('steps','?')} "
        f"· eval_batches {manifest.get('eval_batches','?')} · device {manifest.get('device','?')} "
        f"· seed {manifest.get('seed','?')} · parent_selection {manifest.get('parent_selection','?')}"
    )
    cfg_line = f"Config: {Path(str(manifest.get('config') or '')).name or '?'}"
    ax_title.text(0.0, 0.45, run_line, fontsize=9.5, color="#444444", ha="left", va="top")
    ax_title.text(0.0, 0.18, cfg_line, fontsize=9.5, color="#444444", ha="left", va="top")
    ax_title.text(
        0.0,
        -0.05,
        f"Frontier size: {len(frontier)} · Depth: {min_depth}–{max_depth} layers (median {median_depth})",
        fontsize=9.5,
        color="#1b5e20",
        ha="left",
        va="top",
    )

    ax_barcode = fig.add_subplot(gs[1])
    exemplar_depth = _depth(exemplar)
    _draw_arch_barcode(
        ax_barcode,
        _as_dict(exemplar.get("spec")),
        title=f"Frontier exemplar (depth={exemplar_depth}): {exemplar.get('id')}",
    )

    ax_gap = fig.add_subplot(gs[2])
    ax_gap.axis("off")

    ax_bars = fig.add_subplot(gs[3])
    labels = [m.capitalize() for m in motifs]
    values = [counts[m] for m in motifs]
    colors_by_name = {
        "memory": "#2ecc71",
        "sparsity": "#3498db",
        "moe": "#f39c12",
        "ssm": "#9b59b6",
        "recurrence": "#e74c3c",
        "mla": "#34495e",
        "selector": "#1abc9c",
        "qk_norm": "#111111",
    }
    colors = [colors_by_name.get(m, "#999999") for m in motifs]
    ax_bars.barh(labels, values, color=colors)
    ax_bars.set_xlim(0, len(frontier))
    ax_bars.set_xlabel(f"Frontier entries (of {len(frontier)})")
    ax_bars.set_title("Motif presence across frontier entries", loc="left", fontsize=11, pad=8)
    for idx, v in enumerate(values):
        pct = 100.0 * float(v) / float(len(frontier))
        ax_bars.text(v + 0.5, idx, f"{v}/{len(frontier)} ({pct:.1f}%)", va="center", fontsize=9)
    for spine in ax_bars.spines.values():
        spine.set_visible(False)

    ax_footer = fig.add_subplot(gs[4])
    ax_footer.axis("off")
    ax_footer.text(
        0.0,
        0.4,
        f"Sources: {_pretty_path(frontier_path)}"
        + (f" and {_pretty_path(manifest_path)}" if manifest_path else ""),
        fontsize=8.5,
        color="#666666",
        ha="left",
        va="center",
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render outreach-ready PNG summaries.")
    parser.add_argument(
        "--longctx-run",
        type=Path,
        default=Path("runs/exp_longctx_full_deck_2h_m4_20251217_003818"),
        help="Run directory containing frontier.json + motifs.txt + frontier.manifest.json.",
    )
    parser.add_argument(
        "--hybrid-frontier",
        type=Path,
        default=Path("runs/frontier_phi_entropy_v2.json"),
        help="Path to hybrid run frontier JSON (list).",
    )
    parser.add_argument(
        "--hybrid-manifest",
        type=Path,
        default=Path("runs/frontier_phi_entropy_v2.manifest.json"),
        help="Optional manifest JSON for the hybrid run.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("images"),
        help="Where to write the PNGs.",
    )
    parser.add_argument(
        "--sota-frontier",
        type=Path,
        default=Path("runs/exp_sota_live_2025-12-13_v2/frontier.json"),
        help="Path to a newer multi-objective frontier JSON (list).",
    )
    parser.add_argument(
        "--sota-manifest",
        type=Path,
        default=Path("runs/exp_sota_live_2025-12-13_v2/frontier.manifest.json"),
        help="Optional manifest JSON for the newer multi-objective run.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    _render_longctx(out_dir / "longctx_frontier_summary.png", run_dir=args.longctx_run)
    _render_frontier_overview(
        out_dir / "hybrid_frontier_summary.png",
        frontier_path=args.hybrid_frontier,
        manifest_path=args.hybrid_manifest if args.hybrid_manifest.exists() else None,
        title="Hybrid frontier summary (architecture at a glance)",
        exemplar_id="xover-119-1d12",
    )
    if args.sota_frontier.exists():
        _render_frontier_overview(
            out_dir / "sota_frontier_summary.png",
            frontier_path=args.sota_frontier,
            manifest_path=args.sota_manifest if args.sota_manifest.exists() else None,
            title="Multi-objective frontier summary (architecture at a glance)",
            exemplar_id=None,
        )


if __name__ == "__main__":
    main()
