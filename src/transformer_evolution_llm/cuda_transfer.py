"""Helpers for staging CUDA transfer and handoff flows into autoresearch-family repos."""

from __future__ import annotations

import difflib
import os
import shutil
import subprocess  # nosec B404 - workflow runner executes trusted local benchmark commands.
from enum import StrEnum
from pathlib import Path
from typing import Any

import ujson as json
from pydantic import ValidationError

from .dsl import ArchitectureSpec
from .mlx_transfer import (
    BenchmarkRunResult,
    render_public_transfer_summary_markdown,
    summarize_public_transfer_results,
)
from .train_recipe import (
    TrainRecipe,
    TrainRecipeCompatibilityError,
    TrainRecipeTarget,
    load_train_recipe,
    render_train_recipe_to_path,
    save_train_recipe,
    train_recipe_from_spec,
    train_recipe_projection_applied,
)


class CudaTransferError(ValueError):
    """Base error for the TEVO -> autoresearch CUDA workflow."""


class AutoresearchFlavor(StrEnum):
    """Supported repo flavors that share the CUDA train.py surface."""

    UPSTREAM = "upstream"
    AT_HOME = "at_home"


_AUTORESEARCH_REPO_URLS: dict[AutoresearchFlavor, str] = {
    AutoresearchFlavor.UPSTREAM: "https://github.com/karpathy/autoresearch.git",
    AutoresearchFlavor.AT_HOME: "https://github.com/mutable-state-inc/autoresearch-at-home.git",
}

_HANDOFF_IGNORE_NAMES = shutil.ignore_patterns(
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".DS_Store",
    "run.log",
    "results.tsv",
)


def resolve_autoresearch_repo_url(
    *, flavor: AutoresearchFlavor, repo_url: str | None = None
) -> str:
    """Resolve the effective repo URL from an optional explicit override and a named flavor."""
    explicit = str(repo_url or "").strip()
    if explicit:
        return explicit
    return _AUTORESEARCH_REPO_URLS[flavor]


def infer_autoresearch_flavor(repo_url: str | None) -> AutoresearchFlavor:
    """Best-effort repo flavor inference from a git remote URL."""
    lowered = str(repo_url or "").strip().lower()
    if "mutable-state-inc/autoresearch-at-home" in lowered:
        return AutoresearchFlavor.AT_HOME
    return AutoresearchFlavor.UPSTREAM


def train_recipe_target_for_flavor(flavor: AutoresearchFlavor) -> TrainRecipeTarget:
    """Map a repo flavor onto the matching CUDA render target."""
    if flavor == AutoresearchFlavor.AT_HOME:
        return TrainRecipeTarget.AUTORESEARCH_AT_HOME_CUDA
    return TrainRecipeTarget.AUTORESEARCH_CUDA


def resolve_autoresearch_source_repo(
    *,
    out_dir: str | Path,
    flavor: AutoresearchFlavor = AutoresearchFlavor.UPSTREAM,
    local_repo: str | Path | None = None,
    repo_url: str | None = None,
    repo_ref: str = "master",
    source_dirname: str = "autoresearch_source",
) -> tuple[Path, dict[str, str | None]]:
    """Resolve a local autoresearch checkout or clone a fresh snapshot for rendering."""
    out_dir = Path(out_dir)
    resolved_repo_url = resolve_autoresearch_repo_url(flavor=flavor, repo_url=repo_url)
    if local_repo is not None:
        repo_path = Path(local_repo).resolve()
        _validate_autoresearch_repo_layout(repo_path)
        origin_url = _git_stdout(
            ["git", "-C", str(repo_path), "config", "--get", "remote.origin.url"]
        )
        head_commit = _git_stdout(["git", "-C", str(repo_path), "rev-parse", "HEAD"])
        resolved_flavor = infer_autoresearch_flavor(origin_url or resolved_repo_url)
        return repo_path, {
            "source_kind": "local",
            "repo_url": origin_url or resolved_repo_url or None,
            "repo_ref": head_commit or repo_ref or None,
            "autoresearch_flavor": resolved_flavor.value,
        }

    repo_path = out_dir / source_dirname
    if repo_path.exists():
        shutil.rmtree(repo_path)
    repo_path.parent.mkdir(parents=True, exist_ok=True)
    _clone_repo_ref(repo_url=resolved_repo_url, repo_ref=repo_ref, out_dir=repo_path)
    _validate_autoresearch_repo_layout(repo_path)
    head_commit = _git_stdout(["git", "-C", str(repo_path), "rev-parse", "HEAD"])
    return repo_path, {
        "source_kind": "cloned",
        "repo_url": resolved_repo_url,
        "repo_ref": head_commit or repo_ref,
        "autoresearch_flavor": flavor.value,
    }


def render_public_cuda_variants(
    autoresearch_repo: str | Path,
    recipe_manifest_path: str | Path,
    out_dir: str | Path,
    *,
    target: TrainRecipeTarget = TrainRecipeTarget.AUTORESEARCH_CUDA,
) -> tuple[dict[str, Path], Path]:
    """Render baseline + labeled TEVO train.py variants for CUDA autoresearch."""
    autoresearch_repo = Path(autoresearch_repo)
    out_dir = Path(out_dir)
    train_py_path = autoresearch_repo / "train.py"
    prepare_py_path = autoresearch_repo / "prepare.py"
    if not train_py_path.exists():
        raise CudaTransferError(f"autoresearch train.py not found at {train_py_path}")
    if not prepare_py_path.exists():
        raise CudaTransferError(f"autoresearch prepare.py not found at {prepare_py_path}")

    manifest = json.loads(Path(recipe_manifest_path).read_text())
    selected = manifest.get("selected")
    if not isinstance(selected, list):
        raise CudaTransferError(f"Invalid recipe manifest at {recipe_manifest_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    rendered: dict[str, Path] = {}
    baseline_path = out_dir / "baseline.train.py"
    baseline_path.write_text(train_py_path.read_text())
    rendered["baseline"] = baseline_path

    render_rows: list[dict[str, Any]] = []
    for row in selected:
        if not isinstance(row, dict):
            continue
        label = str(row.get("label") or "").strip()
        recipe_path_raw = row.get("recipe_path")
        if not label or not recipe_path_raw:
            continue
        recipe_path = Path(str(recipe_path_raw))
        out_path = out_dir / f"{label}.train.py"
        recipe = load_train_recipe(recipe_path)
        render_train_recipe_to_path(
            recipe,
            target=target,
            train_py_path=train_py_path,
            out_path=out_path,
        )
        rendered[label] = out_path
        render_rows.append(
            {"label": label, "recipe_path": str(recipe_path), "rendered_train_py": str(out_path)}
        )

    manifest_path = out_dir / "render_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "autoresearch_repo": str(autoresearch_repo.resolve()),
                "source_train_py": str(train_py_path.resolve()),
                "render_target": target.value,
                "rendered": render_rows,
            },
            indent=2,
        )
    )
    return rendered, manifest_path


def prepare_autoresearch_at_home_handoff(
    *,
    frontier_path: str | Path,
    candidate_id: str,
    out_dir: str | Path,
    local_repo: str | Path | None = None,
    repo_url: str | None = None,
    repo_ref: str = "master",
    flavor: AutoresearchFlavor = AutoresearchFlavor.AT_HOME,
    lineage_path: str | Path | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    """Create a single-candidate handoff bundle for autoresearch@home."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_repo, repo_metadata = resolve_autoresearch_source_repo(
        out_dir=out_dir,
        flavor=flavor,
        local_repo=local_repo,
        repo_url=repo_url,
        repo_ref=repo_ref,
        source_dirname="autoresearch_at_home_source",
    )
    recipe = _load_frontier_candidate_recipe(frontier_path, candidate_id)
    render_target = train_recipe_target_for_flavor(flavor)
    projection_applied = train_recipe_projection_applied(recipe, render_target)

    recipe_path = out_dir / "candidate.train_recipe.yaml"
    save_train_recipe(recipe, recipe_path)

    baseline_path = out_dir / "baseline.train.py"
    source_train_py = source_repo / "train.py"
    baseline_path.write_text(source_train_py.read_text())

    candidate_path = out_dir / "candidate.train.py"
    render_train_recipe_to_path(
        recipe,
        target=render_target,
        train_py_path=source_train_py,
        out_path=candidate_path,
    )
    diff_path = write_train_py_diff(baseline_path, candidate_path, out_dir / "candidate.diff")
    staged_repo = _stage_handoff_repo_workspace(
        source_repo=source_repo,
        rendered_train_py=candidate_path,
        out_dir=out_dir / "repo",
    )
    suggested_branch = f"autoresearch/tevo-{_safe_slug(candidate_id)}"
    suggested_description = _suggest_handoff_description(recipe)

    manifest = {
        "candidate_id": candidate_id,
        "frontier_path": str(Path(frontier_path).resolve()),
        "lineage_path": (str(Path(lineage_path).resolve()) if lineage_path is not None else None),
        "recipe_path": str(recipe_path.resolve()),
        "baseline_train_py": str(baseline_path.resolve()),
        "candidate_train_py": str(candidate_path.resolve()),
        "candidate_diff": str(diff_path.resolve()),
        "staged_repo": str(staged_repo.resolve()),
        "source_repo": str(source_repo.resolve()),
        "render_target": render_target.value,
        "projection_applied": projection_applied,
        "suggested_branch": suggested_branch,
        "suggested_experiment_description": suggested_description,
        "candidate_metrics": (dict(recipe.source.metrics) if recipe.source is not None else {}),
        "autoresearch_flavor": repo_metadata.get("autoresearch_flavor"),
        "autoresearch_repo_url": repo_metadata.get("repo_url"),
        "autoresearch_repo_ref": repo_metadata.get("repo_ref"),
        "autoresearch_source_kind": repo_metadata.get("source_kind"),
    }
    manifest_path = out_dir / "handoff_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    summary_path = out_dir / "handoff_summary.md"
    summary_path.write_text(_render_handoff_summary(recipe, manifest))
    return manifest, manifest_path, summary_path


def run_public_cuda_transfer_modal_benchmarks(
    arm_manifest_path: str | Path,
    *,
    repo_url: str,
    repo_ref: str,
    out_dir: str | Path,
    repo_root: str | Path,
    modal_gpu: str = "A100-80GB",
    repeat: int = 3,
    timeout_minutes: int = 10,
    modal_local_out_dir: str | Path | None = None,
) -> tuple[list[BenchmarkRunResult], Path, Path, Path]:
    """Benchmark baseline + seeded CUDA autoresearch arms through Modal."""
    if not repo_url:
        raise CudaTransferError(
            "CUDA Modal benchmarking requires an autoresearch repo URL so Modal can clone it."
        )
    arm_manifest = json.loads(Path(arm_manifest_path).read_text())
    arms = arm_manifest.get("arms")
    if not isinstance(arms, list):
        raise CudaTransferError(f"Invalid arm manifest at {arm_manifest_path}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    modal_out_dir = (
        Path(modal_local_out_dir) if modal_local_out_dir is not None else out_dir / "modal"
    )
    modal_out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        AUTORESEARCH_MODAL_GPU=str(modal_gpu),
        AUTORESEARCH_MODAL_TIMEOUT_S=str(int(timeout_minutes) * 60),
    )
    repo_root = Path(repo_root)
    results: list[BenchmarkRunResult] = []
    for row in arms:
        if not isinstance(row, dict):
            continue
        arm = str(row.get("arm") or "").strip()
        if arm not in {"baseline", "quality", "compute", "balanced"}:
            continue
        train_py_path = Path(str(row.get("variant_source") or row.get("train_py") or ""))
        if not train_py_path.exists():
            raise CudaTransferError(f"Missing rendered train.py for arm {arm}: {train_py_path}")
        run_id = f"{_safe_slug(out_dir.parent.name)}_{arm}"
        command = [
            "modal",
            "run",
            "scripts/modal_run_autoresearch_cuda.py",
            "--train-py-path",
            str(train_py_path),
            "--repo-url",
            repo_url,
            "--repo-ref",
            repo_ref,
            "--repeat",
            str(int(repeat)),
            "--timeout-minutes",
            str(int(timeout_minutes)),
            "--run-id",
            run_id,
            "--download",
            "--local-out-dir",
            str(modal_out_dir),
        ]
        subprocess.run(  # noqa: S603  # nosec B603
            command,
            check=True,
            cwd=repo_root,
            env=env,
        )
        summary_path = modal_out_dir / run_id / "summary.json"
        if not summary_path.exists():
            raise CudaTransferError(
                f"Missing downloaded Modal summary for arm {arm}: {summary_path}"
            )
        payload = json.loads(summary_path.read_text())
        arm_results = payload.get("results")
        if not isinstance(arm_results, list):
            raise CudaTransferError(f"Invalid Modal arm summary at {summary_path}")
        for item in arm_results:
            if not isinstance(item, dict):
                continue
            log_name = str(item.get("log_path") or "")
            results.append(
                BenchmarkRunResult(
                    arm=arm,
                    run_index=int(item.get("run_index") or 0),
                    status=str(item.get("status") or "crash"),
                    val_bpb=_float_or_none(item.get("val_bpb")),
                    peak_memory_gb=_float_or_none(item.get("peak_memory_gb")),
                    log_path=(summary_path.parent / log_name if log_name else summary_path),
                )
            )

    results_path = out_dir / "benchmark_results.json"
    result_rows = _benchmark_rows(results)
    results_path.write_text(json.dumps(result_rows, indent=2))
    tsv_path = out_dir / "benchmark_results.tsv"
    _write_benchmark_tsv(tsv_path, result_rows)

    summary_payload = summarize_public_transfer_results(results)
    summary_path = out_dir / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2))
    markdown_path = out_dir / "benchmark_summary.md"
    markdown_path.write_text(render_public_cuda_transfer_summary_markdown(summary_payload))
    return results, results_path, summary_path, markdown_path


def build_public_cuda_transfer_report(run_root: str | Path) -> tuple[dict[str, Any], str]:
    """Assemble a compact public report for the CUDA transfer run."""
    run_root = Path(run_root)
    prepare_manifest_path = run_root / "transfer_prepare_manifest.json"
    seed_summary_path = run_root / "tevo_seed_summary.json"
    benchmark_summary_path = run_root / "cuda_results" / "benchmark_summary.json"

    prepare_manifest = (
        json.loads(prepare_manifest_path.read_text()) if prepare_manifest_path.exists() else None
    )
    seed_summary = json.loads(seed_summary_path.read_text()) if seed_summary_path.exists() else None
    benchmark_summary = (
        json.loads(benchmark_summary_path.read_text()) if benchmark_summary_path.exists() else None
    )
    report_payload = {
        "prepare_manifest": prepare_manifest,
        "tevo_seed_summary": seed_summary,
        "cuda_benchmark_summary": benchmark_summary,
    }

    lines = ["# Public TEVO -> autoresearch CUDA Report", ""]
    if prepare_manifest is not None:
        discovery_gpu = prepare_manifest.get("modal_gpu") or prepare_manifest.get("device") or "-"
        autoresearch_flavor = prepare_manifest.get("autoresearch_flavor") or "-"
        autoresearch_repo_url = prepare_manifest.get("autoresearch_repo_url") or "-"
        autoresearch_repo_ref = prepare_manifest.get("autoresearch_repo_ref") or "-"
        lines.extend(
            [
                "## Setup",
                "",
                f"- TEVO discovery mode: `{prepare_manifest.get('tevo_mode', '-')}`",
                f"- TEVO discovery GPU: `{discovery_gpu}`",
                f"- CUDA autoresearch flavor: `{autoresearch_flavor}`",
                f"- CUDA autoresearch repo: `{autoresearch_repo_url}`",
                f"- CUDA autoresearch ref: `{autoresearch_repo_ref}`",
                "",
            ]
        )
    if isinstance(seed_summary, dict):
        raw_metrics = seed_summary.get("metrics")
        metrics: dict[str, Any] = raw_metrics if isinstance(raw_metrics, dict) else {}
        lines.extend(
            [
                "## TEVO seed on openwebtext_10m",
                "",
                f"- `ppl_code`: `{_fmt_metric(metrics.get('ppl_code'))}`",
                "- `speedrun_flops_to_target`: "
                f"`{_fmt_metric(metrics.get('speedrun_flops_to_target'))}`",
                "",
            ]
        )
    if isinstance(benchmark_summary, dict):
        lines.extend(
            [
                "## CUDA autoresearch benchmark",
                "",
                render_public_cuda_transfer_summary_markdown(benchmark_summary).rstrip(),
                "",
            ]
        )
    return report_payload, "\n".join(lines).rstrip() + "\n"


def render_public_cuda_transfer_summary_markdown(summary: dict[str, Any]) -> str:
    """Render the CUDA benchmark summary with a backend-accurate title."""
    markdown = render_public_transfer_summary_markdown(summary)
    return markdown.replace(
        "# MLX Transfer Benchmark Summary",
        "# CUDA Transfer Benchmark Summary",
        1,
    )


def write_train_py_diff(
    baseline_train_py: str | Path,
    candidate_train_py: str | Path,
    out_path: str | Path,
) -> Path:
    """Write a unified diff between a baseline and candidate train.py file."""
    baseline_path = Path(baseline_train_py)
    candidate_path = Path(candidate_train_py)
    diff = "".join(
        difflib.unified_diff(
            baseline_path.read_text().splitlines(keepends=True),
            candidate_path.read_text().splitlines(keepends=True),
            fromfile=str(baseline_path),
            tofile=str(candidate_path),
            n=3,
        )
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(diff)
    return out_path


def _clone_repo_ref(*, repo_url: str, repo_ref: str, out_dir: Path) -> None:
    git = _git_binary()
    subprocess.run(  # noqa: S603  # nosec B603
        [git, "clone", "--depth", "1", repo_url, str(out_dir)],
        check=True,
    )
    if not repo_ref:
        return
    checkout = subprocess.run(  # noqa: S603  # nosec B603
        [git, "-C", str(out_dir), "checkout", repo_ref],
        check=False,
        capture_output=True,
        text=True,
    )
    if checkout.returncode == 0:
        return
    subprocess.run(  # noqa: S603  # nosec B603
        [git, "-C", str(out_dir), "fetch", "--depth", "1", "origin", repo_ref],
        check=True,
    )
    subprocess.run(  # noqa: S603  # nosec B603
        [git, "-C", str(out_dir), "checkout", "FETCH_HEAD"],
        check=True,
    )


def _validate_autoresearch_repo_layout(repo_path: Path) -> None:
    train_py_path = repo_path / "train.py"
    prepare_py_path = repo_path / "prepare.py"
    if not train_py_path.exists():
        raise CudaTransferError(f"autoresearch train.py not found at {train_py_path}")
    if not prepare_py_path.exists():
        raise CudaTransferError(f"autoresearch prepare.py not found at {prepare_py_path}")


def _load_frontier_candidate_recipe(frontier_path: str | Path, candidate_id: str) -> TrainRecipe:
    frontier_path = Path(frontier_path)
    payload = json.loads(frontier_path.read_text())
    if not isinstance(payload, list):
        raise CudaTransferError(f"Frontier JSON must be a list: {frontier_path}")
    match = next(
        (
            entry
            for entry in payload
            if isinstance(entry, dict) and str(entry.get("id") or "") == candidate_id
        ),
        None,
    )
    if match is None:
        raise CudaTransferError(f"Candidate {candidate_id!r} not found in {frontier_path}")
    spec_payload = match.get("spec")
    if not isinstance(spec_payload, dict):
        raise CudaTransferError(f"Candidate {candidate_id!r} is missing a spec payload")
    try:
        spec = ArchitectureSpec(**spec_payload)
    except ValidationError as exc:
        raise CudaTransferError(
            f"Candidate {candidate_id!r} has an invalid spec payload: {exc}"
        ) from exc
    try:
        return train_recipe_from_spec(
            spec,
            candidate_id=candidate_id,
            frontier_path=frontier_path,
            metrics=_numeric_metrics(match.get("metrics")),
        )
    except TrainRecipeCompatibilityError as exc:
        raise CudaTransferError(
            f"Candidate {candidate_id!r} is not bridge-compatible: {exc}"
        ) from exc


def _stage_handoff_repo_workspace(
    *,
    source_repo: Path,
    rendered_train_py: Path,
    out_dir: Path,
) -> Path:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(source_repo, out_dir, ignore=_HANDOFF_IGNORE_NAMES)
    source_venv = source_repo / ".venv"
    if source_venv.exists():
        os.symlink(source_venv, out_dir / ".venv")
    shutil.copy2(rendered_train_py, out_dir / "train.py")
    return out_dir


def _suggest_handoff_description(recipe: TrainRecipe) -> str:
    return (
        f"TEVO handoff {recipe.name}: window {recipe.model.window_pattern}, "
        f"n_kv_head {recipe.model.n_kv_head}, "
        f"mlp {recipe.model.mlp_activation}/{recipe.model.mlp_hidden}"
    )


def _render_handoff_summary(recipe: TrainRecipe, manifest: dict[str, Any]) -> str:
    metrics = manifest.get("candidate_metrics") or {}
    metric_lines = [f"- `{key}`: `{_fmt_metric(value)}`" for key, value in sorted(metrics.items())]
    if not metric_lines:
        metric_lines = ["- No numeric TEVO metrics were captured for this candidate."]
    return "\n".join(
        [
            "# TEVO -> autoresearch@home Handoff",
            "",
            "So what: this bundle stages one TEVO-discovered, bridge-compatible candidate as a "
            "runnable `autoresearch@home` workspace. It is a contribution lane into their swarm, "
            "not a distributed TEVO search loop.",
            "",
            "## Candidate",
            "",
            f"- Candidate id: `{manifest['candidate_id']}`",
            f"- Frontier: `{manifest['frontier_path']}`",
            f"- Projection applied: `{manifest['projection_applied']}`",
            f"- Render target: `{manifest['render_target']}`",
            "",
            "## TEVO Metrics",
            "",
            *metric_lines,
            "",
            "## Repo Target",
            "",
            f"- Flavor: `{manifest.get('autoresearch_flavor') or '-'}`",
            f"- Repo URL: `{manifest.get('autoresearch_repo_url') or '-'}`",
            f"- Repo ref: `{manifest.get('autoresearch_repo_ref') or '-'}`",
            f"- Staged repo: `{manifest['staged_repo']}`",
            "",
            "## Suggested Experiment",
            "",
            f"- Description: `{manifest['suggested_experiment_description']}`",
            "",
            "## Next Steps",
            "",
            "```bash",
            f"cd {manifest['staged_repo']}",
            f"git checkout -b {manifest['suggested_branch']}",
            "uv run train.py",
            "```",
            "",
            "If you are running inside `autoresearch@home`, claim this experiment through the "
            "coordinator first and publish the result, insight, and follow-up hypothesis "
            "after the run.",
            "",
        ]
    )


def _git_stdout(command: list[str]) -> str | None:
    command = [
        _git_binary() if idx == 0 and part == "git" else part for idx, part in enumerate(command)
    ]
    completed = subprocess.run(  # noqa: S603  # nosec B603
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    text = str(completed.stdout or "").strip()
    return text or None


def _numeric_metrics(payload: Any) -> dict[str, float]:
    if not isinstance(payload, dict):
        return {}
    metrics: dict[str, float] = {}
    for key, value in payload.items():
        numeric = _float_or_none(value)
        if numeric is not None:
            metrics[str(key)] = numeric
    return metrics


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _benchmark_rows(results: list[BenchmarkRunResult]) -> list[dict[str, Any]]:
    return [
        {
            "arm": item.arm,
            "run_index": item.run_index,
            "val_bpb": item.val_bpb,
            "peak_memory_gb": item.peak_memory_gb,
            "status": item.status,
            "log_path": str(item.log_path),
        }
        for item in results
    ]


def _write_benchmark_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    header = "arm\trun_index\tval_bpb\tpeak_memory_gb\tstatus\tlog_path\n"
    lines = [header]
    for row in rows:
        lines.append(
            "{arm}\t{run_index}\t{val_bpb}\t{peak_memory_gb}\t{status}\t{log_path}\n".format(
                arm=row.get("arm", ""),
                run_index=row.get("run_index", ""),
                val_bpb=_fmt_metric(row.get("val_bpb")),
                peak_memory_gb=_fmt_metric(row.get("peak_memory_gb")),
                status=row.get("status", ""),
                log_path=row.get("log_path", ""),
            )
        )
    path.write_text("".join(lines))


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number) >= 1000:
        return f"{number:.1f}"
    return f"{number:.6f}".rstrip("0").rstrip(".")


def _safe_slug(value: str) -> str:
    slug = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in value).strip("_")
    return slug or "candidate"


def _git_binary() -> str:
    git = shutil.which("git")
    if git is None:
        raise CudaTransferError("git is required for the CUDA transfer workflow.")
    return git
