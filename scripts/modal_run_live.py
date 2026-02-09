"""Run scripts/run_live.py on Modal GPUs.

This is intentionally minimal: it runs one full evolutionary sweep inside a
single Modal container (so weight inheritance stays local), writes run artifacts
to a persisted Modal Volume, and optionally downloads the frontier/state JSONs
back into the local repo.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]

GPU = os.environ.get("TEVO_MODAL_GPU", "A10G")
TIMEOUT_S = int(os.environ.get("TEVO_MODAL_TIMEOUT_S", str(60 * 60 * 12)))
PYTORCH_INDEX_URL = os.environ.get("TEVO_TORCH_INDEX_URL", "https://download.pytorch.org/whl/cu124")
TORCH_VERSION = os.environ.get("TEVO_TORCH_VERSION", "2.6.0+cu124")

RUNS_VOLUME_NAME = os.environ.get("TEVO_MODAL_RUNS_VOLUME", "tevo-runs")
HF_VOLUME_NAME = os.environ.get("TEVO_MODAL_HF_VOLUME", "tevo-hf-cache")

RUNS_VOL = modal.Volume.from_name(RUNS_VOLUME_NAME, create_if_missing=True)
HF_VOL = modal.Volume.from_name(HF_VOLUME_NAME, create_if_missing=True)


def _ignore_repo_path(path: Path) -> bool:
    # Keep the image small and avoid bundling local artifacts.
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
    .pip_install(f"torch=={TORCH_VERSION}", index_url=PYTORCH_INDEX_URL)
    .pip_install(
        "accelerate>=0.33",
        "datasets>=3.0",
        "einops>=0.7",
        "numpy>=1.26",
        "optimum>=1.21",
        "peft>=0.12",
        "pydantic>=2.8",
        "PyYAML>=6.0",
        "rich>=13.7",
        "sentencepiece>=0.2",
        "safetensors>=0.4",
        "tokenizers>=0.19",
        "tqdm>=4.66",
        "transformers>=4.44",
        "typer[all]>=0.12",
        "ujson>=5.9",
    )
    .env(
        {
            "PYTHONPATH": "/repo/src",
            "HF_HOME": "/hf",
            "HF_DATASETS_CACHE": "/hf/datasets",
            "TRANSFORMERS_CACHE": "/hf/transformers",
            "TEVO_PACKED_ROOT": "/runs",
        }
    )
    .add_local_dir(str(REPO_ROOT), remote_path="/repo", ignore=_ignore_repo_path)
)

app = modal.App("transformer-evolution-llm-live")


@app.function(
    image=IMAGE,
    gpu=GPU,
    cpu=8,
    memory=32768,
    timeout=TIMEOUT_S,
    volumes={"/runs": RUNS_VOL, "/hf": HF_VOL},
)
def run_live(
    *,
    config_path: str,
    generations: int,
    steps: int,
    eval_batches: int,
    seed: int,
    run_id: str | None,
    cleanup_old_checkpoints: bool,
    prune_checkpoints_to_frontier: bool,
    lineage: bool,
    mutation_weight: str = "",
) -> dict[str, str]:
    run_name = run_id or f"modal_{int(time.time())}"
    runs_root = Path("/runs")
    run_root = runs_root / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    out_path = run_root / "frontier.json"
    lineage_path = run_root / "lineage.json"
    default_lineage_path = run_root / "frontier_lineage.json"
    checkpoint_dir = run_root / "checkpoints"

    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = Path("/repo") / cfg_path

    cmd = [
        "python",
        "-u",
        "/repo/scripts/run_live.py",
        str(cfg_path),
        "--device",
        "cuda",
        "--generations",
        str(int(generations)),
        "--steps",
        str(int(steps)),
        "--eval-batches",
        str(int(eval_batches)),
        "--seed",
        str(int(seed)),
        "--out",
        str(out_path),
        "--checkpoint-dir",
        str(checkpoint_dir),
    ]
    if mutation_weight:
        for item in str(mutation_weight).split(","):
            item = str(item).strip()
            if not item:
                continue
            cmd.extend(["--mutation-weight", item])
    if cleanup_old_checkpoints:
        cmd.append("--cleanup-old-checkpoints")
    else:
        cmd.append("--no-cleanup-old-checkpoints")
    if prune_checkpoints_to_frontier:
        cmd.append("--prune-checkpoints-to-frontier")
    if lineage:
        cmd.extend(["--lineage-out", str(lineage_path)])

    started = {
        "timestamp_unix_s": time.time(),
        "cmd": cmd,
        "config_path": str(cfg_path),
        "generations": int(generations),
        "steps": int(steps),
        "eval_batches": int(eval_batches),
        "seed": int(seed),
        "gpu": str(GPU),
        "timeout_s": int(TIMEOUT_S),
        "checkpoint_dir": str(checkpoint_dir),
        "out": str(out_path),
        "lineage_out": str(lineage_path if lineage else default_lineage_path),
        "cleanup_old_checkpoints": bool(cleanup_old_checkpoints),
        "prune_checkpoints_to_frontier": bool(prune_checkpoints_to_frontier),
        "register_template_entries": True,
        "mutation_weight": str(mutation_weight),
    }
    try:
        (run_root / "run_live.started.json").write_text(json.dumps(started, indent=2))
    except Exception:
        pass
    try:
        RUNS_VOL.commit()
    except Exception:
        pass

    log_path = run_root / "run_live.stdout.log"
    returncode = 1
    proc: subprocess.Popen[str] | None = None
    stop = threading.Event()

    def _commit_loop() -> None:
        # Best-effort: persist logs and intermediate artifacts periodically so
        # abrupt termination still leaves evidence behind.
        while not stop.wait(60.0):
            try:
                RUNS_VOL.commit()
            except Exception:
                continue

    committer = threading.Thread(target=_commit_loop, daemon=True)
    committer.start()
    try:
        with log_path.open("w", encoding="utf-8") as log_handle:
            proc = subprocess.Popen(  # noqa: S603
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                try:
                    log_handle.write(line)
                    log_handle.flush()
                except Exception:
                    pass
                try:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                except Exception:
                    pass
            returncode = int(proc.wait())
    finally:
        stop.set()
        try:
            committer.join(timeout=2.0)
        except Exception:
            pass
    try:
        meta = {
            "returncode": returncode,
            "cmd": cmd,
            "timestamp_unix_s": time.time(),
        }
        (run_root / "run_live.subprocess.json").write_text(json.dumps(meta, indent=2))
        if returncode != 0:
            (run_root / "run_live.error.txt").write_text(
                f"subprocess failed (returncode={returncode})\n"
            )
    except Exception:
        pass
    try:
        RUNS_VOL.commit()
        HF_VOL.commit()
    except Exception:
        pass
    resolved_lineage = lineage_path if lineage else default_lineage_path
    return {
        "run_id": run_name,
        "frontier": f"{run_name}/frontier.json",
        "state": f"{run_name}/frontier.state.json",
        "lineage": f"{run_name}/{resolved_lineage.name}",
        "manifest": f"{run_name}/frontier.manifest.json",
        "returncode": str(returncode),
    }


@app.local_entrypoint()
def main(
    config_path: str,
    generations: int = 48,
    steps: int = 360,
    eval_batches: int = 4,
    seed: int = 0,
    run_id: str | None = None,
    download: bool = False,
    local_out_dir: str = "runs/modal",
    cleanup_old_checkpoints: bool = True,
    prune_checkpoints_to_frontier: bool = False,
    lineage: bool = False,
    mutation_weight: str = "",
) -> None:
    result = run_live.remote(
        config_path=config_path,
        generations=generations,
        steps=steps,
        eval_batches=eval_batches,
        seed=seed,
        run_id=run_id,
        cleanup_old_checkpoints=cleanup_old_checkpoints,
        prune_checkpoints_to_frontier=prune_checkpoints_to_frontier,
        lineage=lineage,
        mutation_weight=str(mutation_weight),
    )
    print(result)

    if not download:
        return
    out_root = Path(local_out_dir) / str(result["run_id"])
    out_root.mkdir(parents=True, exist_ok=True)

    for key in ("frontier", "state", "lineage", "manifest"):
        remote_path = result.get(key) or ""
        if not remote_path:
            continue
        local_path = out_root / Path(remote_path).name
        try:
            with local_path.open("wb") as handle:
                RUNS_VOL.read_file_into_fileobj(remote_path, handle)
        except FileNotFoundError:
            local_path.unlink(missing_ok=True)
            print(f"missing {remote_path}; skipping download")
            continue
        print(f"downloaded {remote_path} -> {local_path}")
