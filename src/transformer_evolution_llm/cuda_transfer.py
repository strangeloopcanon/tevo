"""Helpers for staging and benchmarking the first TEVO -> autoresearch CUDA transfer run."""

from __future__ import annotations

import os
import shutil
import subprocess  # nosec B404 - workflow runner executes trusted local benchmark commands.
from pathlib import Path
from typing import Any

import ujson as json

from .mlx_transfer import (
    BenchmarkRunResult,
    render_public_transfer_summary_markdown,
    summarize_public_transfer_results,
)
from .train_recipe import TrainRecipeTarget, load_train_recipe, render_train_recipe_to_path


class CudaTransferError(ValueError):
    """Base error for the TEVO -> autoresearch CUDA workflow."""


def resolve_autoresearch_source_repo(
    *,
    out_dir: str | Path,
    local_repo: str | Path | None = None,
    repo_url: str = "https://github.com/karpathy/autoresearch.git",
    repo_ref: str = "master",
) -> tuple[Path, dict[str, str | None]]:
    """Resolve a local autoresearch checkout or clone a fresh snapshot for rendering."""
    out_dir = Path(out_dir)
    if local_repo is not None:
        repo_path = Path(local_repo).resolve()
        if not (repo_path / "train.py").exists():
            raise CudaTransferError(f"autoresearch train.py not found at {repo_path / 'train.py'}")
        origin_url = _git_stdout(
            ["git", "-C", str(repo_path), "config", "--get", "remote.origin.url"]
        )
        head_commit = _git_stdout(["git", "-C", str(repo_path), "rev-parse", "HEAD"])
        return repo_path, {
            "source_kind": "local",
            "repo_url": origin_url or repo_url or None,
            "repo_ref": head_commit or repo_ref or None,
        }

    repo_path = out_dir / "autoresearch_source"
    if repo_path.exists():
        shutil.rmtree(repo_path)
    repo_path.parent.mkdir(parents=True, exist_ok=True)
    _clone_repo_ref(repo_url=repo_url, repo_ref=repo_ref, out_dir=repo_path)
    head_commit = _git_stdout(["git", "-C", str(repo_path), "rev-parse", "HEAD"])
    return repo_path, {
        "source_kind": "cloned",
        "repo_url": repo_url,
        "repo_ref": head_commit or repo_ref,
    }


def render_public_cuda_variants(
    autoresearch_repo: str | Path,
    recipe_manifest_path: str | Path,
    out_dir: str | Path,
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
        if label not in {"quality", "compute", "balanced"} or not recipe_path_raw:
            continue
        recipe_path = Path(str(recipe_path_raw))
        out_path = out_dir / f"{label}.train.py"
        recipe = load_train_recipe(recipe_path)
        render_train_recipe_to_path(
            recipe,
            target=TrainRecipeTarget.AUTORESEARCH_CUDA,
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
                "rendered": render_rows,
            },
            indent=2,
        )
    )
    return rendered, manifest_path


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
        autoresearch_repo_url = prepare_manifest.get("autoresearch_repo_url") or "-"
        autoresearch_repo_ref = prepare_manifest.get("autoresearch_repo_ref") or "-"
        lines.extend(
            [
                "## Setup",
                "",
                f"- TEVO discovery mode: `{prepare_manifest.get('tevo_mode', '-')}`",
                f"- TEVO discovery GPU: `{discovery_gpu}`",
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
