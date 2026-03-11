"""Run a rendered autoresearch CUDA train.py variant on Modal."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]

GPU = os.environ.get("AUTORESEARCH_MODAL_GPU", "A10G")
TIMEOUT_S = int(os.environ.get("AUTORESEARCH_MODAL_TIMEOUT_S", str(60 * 60 * 2)))
PYTORCH_INDEX_URL = os.environ.get("AUTORESEARCH_TORCH_INDEX_URL", "https://download.pytorch.org/whl/cu128")

RUNS_VOLUME_NAME = os.environ.get("AUTORESEARCH_MODAL_RUNS_VOLUME", "autoresearch-runs")
CACHE_VOLUME_NAME = os.environ.get("AUTORESEARCH_MODAL_CACHE_VOLUME", "autoresearch-cache")

RUNS_VOL = modal.Volume.from_name(RUNS_VOLUME_NAME, create_if_missing=True)
CACHE_VOL = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)

_VAL_BPB_RE = re.compile(r"^val_bpb:\s*([-+0-9.eE]+)\s*$", flags=re.MULTILINE)
_PEAK_VRAM_MB_RE = re.compile(r"^peak_vram_mb:\s*([-+0-9.eE]+)\s*$", flags=re.MULTILINE)


def _ignore_repo_path(path: Path) -> bool:
    if path.name.startswith(".env"):
        return True
    if path.name in {".coverage", "htmlcov"}:
        return True
    parts = set(path.parts)
    blocked = {
        ".git",
        ".venv",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "__pycache__",
        "runs",
        "dist",
        "build",
        ".beads",
        ".DS_Store",
    }
    return any(part in blocked for part in parts)


IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch==2.9.1", index_url=PYTORCH_INDEX_URL)
    .pip_install(
        "kernels>=0.11.7",
        "matplotlib>=3.10.8",
        "numpy>=2.2.6",
        "pandas>=2.3.3",
        "pyarrow>=21.0.0",
        "requests>=2.32.0",
        "rustbpe>=0.1.0",
        "tiktoken>=0.11.0",
        "ujson>=5.9",
    )
    .env(
        {
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .add_local_dir(str(REPO_ROOT), remote_path="/repo", ignore=_ignore_repo_path)
)

app = modal.App("transformer-evolution-llm-autoresearch-cuda")


@app.function(
    image=IMAGE,
    gpu=GPU,
    cpu=8,
    memory=32768,
    timeout=TIMEOUT_S,
    volumes={"/runs": RUNS_VOL, "/root/.cache/autoresearch": CACHE_VOL},
)
def run_autoresearch(
    *,
    train_py_text: str,
    repo_url: str,
    repo_ref: str,
    repeat: int,
    run_id: str | None,
    run_timeout_seconds: int,
) -> dict[str, Any]:
    run_name = run_id or f"autoresearch_{int(time.time())}"
    run_root = Path("/runs") / run_name
    run_root.mkdir(parents=True, exist_ok=True)
    repo_dir = Path("/tmp/autoresearch_repo")
    if repo_dir.exists():
        shutil.rmtree(repo_dir)

    subprocess.run(["git", "clone", "--depth", "1", repo_url, str(repo_dir)], check=True)  # noqa: S603
    if repo_ref:
        checkout = subprocess.run(  # noqa: S603
            ["git", "-C", str(repo_dir), "checkout", repo_ref],
            check=False,
            capture_output=True,
            text=True,
        )
        if checkout.returncode != 0:
            subprocess.run(  # noqa: S603
                ["git", "-C", str(repo_dir), "fetch", "--depth", "1", "origin", repo_ref],
                check=True,
            )
            subprocess.run(  # noqa: S603
                ["git", "-C", str(repo_dir), "checkout", "FETCH_HEAD"],
                check=True,
            )
    resolved_ref = subprocess.run(  # noqa: S603
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (repo_dir / "train.py").write_text(train_py_text)

    prepare_log = run_root / "prepare.log"
    prepare = subprocess.run(  # noqa: S603
        ["python", "prepare.py"],
        cwd=repo_dir,
        check=False,
        capture_output=True,
        text=True,
    )
    prepare_output = f"{prepare.stdout}{prepare.stderr}"
    prepare_log.write_text(prepare_output)
    if prepare.returncode != 0:
        summary = {
            "repo_url": repo_url,
            "repo_ref": resolved_ref,
            "prepare_status": "crash",
            "prepare_log_path": str(prepare_log.name),
            "results": [],
        }
        summary_path = run_root / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        RUNS_VOL.commit()
        CACHE_VOL.commit()
        return {
            "run_id": run_name,
            "summary": f"{run_name}/summary.json",
            "logs": [f"{run_name}/{prepare_log.name}"],
        }

    results: list[dict[str, Any]] = []
    downloaded_logs = [f"{run_name}/{prepare_log.name}"]
    for run_index in range(1, int(repeat) + 1):
        log_name = f"train_run{run_index}.log"
        log_path = run_root / log_name
        status = "crash"
        val_bpb: float | None = None
        peak_memory_gb: float | None = None
        try:
            completed = subprocess.run(  # noqa: S603
                ["python", "train.py"],
                cwd=repo_dir,
                check=False,
                capture_output=True,
                text=True,
                timeout=int(run_timeout_seconds),
            )
            output = f"{completed.stdout}{completed.stderr}"
            log_path.write_text(output)
            parsed = _parse_autoresearch_log(output)
            val_bpb = parsed["val_bpb"]
            peak_memory_gb = parsed["peak_memory_gb"]
            status = "ok" if completed.returncode == 0 and val_bpb is not None else "crash"
        except subprocess.TimeoutExpired as exc:
            output = f"{exc.stdout or ''}{exc.stderr or ''}\nTIMEOUT\n"
            log_path.write_text(output)
        results.append(
            {
                "run_index": run_index,
                "status": status,
                "val_bpb": val_bpb,
                "peak_memory_gb": peak_memory_gb,
                "log_path": log_name,
            }
        )
        downloaded_logs.append(f"{run_name}/{log_name}")

    summary = {
        "repo_url": repo_url,
        "repo_ref": resolved_ref,
        "prepare_status": "ok",
        "prepare_log_path": str(prepare_log.name),
        "results": results,
    }
    summary_path = run_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    RUNS_VOL.commit()
    CACHE_VOL.commit()
    return {
        "run_id": run_name,
        "summary": f"{run_name}/summary.json",
        "logs": downloaded_logs,
    }


@app.local_entrypoint()
def main(
    train_py_path: str,
    repo_url: str = "https://github.com/karpathy/autoresearch.git",
    repo_ref: str = "master",
    repeat: int = 3,
    run_id: str | None = None,
    timeout_minutes: int = 10,
    download: bool = False,
    local_out_dir: str = "runs/modal_autoresearch",
) -> None:
    train_py_text = Path(train_py_path).read_text()
    result = run_autoresearch.remote(
        train_py_text=train_py_text,
        repo_url=repo_url,
        repo_ref=repo_ref,
        repeat=repeat,
        run_id=run_id,
        run_timeout_seconds=int(timeout_minutes) * 60,
    )
    print(result)

    if not download:
        return
    out_root = Path(local_out_dir) / str(result["run_id"])
    out_root.mkdir(parents=True, exist_ok=True)
    summary_remote = result.get("summary") or ""
    if summary_remote:
        summary_local = out_root / Path(summary_remote).name
        with summary_local.open("wb") as handle:
            RUNS_VOL.read_file_into_fileobj(summary_remote, handle)
        print(f"downloaded {summary_remote} -> {summary_local}")
    for remote_path in result.get("logs") or []:
        local_path = out_root / Path(str(remote_path)).name
        with local_path.open("wb") as handle:
            RUNS_VOL.read_file_into_fileobj(str(remote_path), handle)
        print(f"downloaded {remote_path} -> {local_path}")


def _parse_autoresearch_log(text: str) -> dict[str, float | None]:
    val_match = _VAL_BPB_RE.search(text)
    peak_match = _PEAK_VRAM_MB_RE.search(text)
    val_bpb = float(val_match.group(1)) if val_match else None
    peak_memory_gb = float(peak_match.group(1)) / 1024.0 if peak_match else None
    return {"val_bpb": val_bpb, "peak_memory_gb": peak_memory_gb}
