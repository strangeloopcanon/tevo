"""Run scripts/run_live.py on Modal GPUs.

This is intentionally minimal: it runs one full evolutionary sweep inside a
single Modal container (so weight inheritance stays local), writes run artifacts
to a persisted Modal Volume, and optionally downloads the frontier/state JSONs
back into the local repo.
"""

from __future__ import annotations

import os
import subprocess
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
    .add_local_dir(str(REPO_ROOT), remote_path="/repo", ignore=_ignore_repo_path)
    .env(
        {
            "PYTHONPATH": "/repo/src",
            "HF_HOME": "/hf",
            "HF_DATASETS_CACHE": "/hf/datasets",
            "TRANSFORMERS_CACHE": "/hf/transformers",
        }
    )
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
) -> dict[str, str]:
    run_name = run_id or f"modal_{int(time.time())}"
    runs_root = Path("/runs")
    run_root = runs_root / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    out_path = run_root / "frontier.json"
    lineage_path = run_root / "lineage.json"
    checkpoint_dir = run_root / "checkpoints"

    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = Path("/repo") / cfg_path

    cmd = [
        "python",
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
    if cleanup_old_checkpoints:
        cmd.append("--cleanup-old-checkpoints")
    else:
        cmd.append("--no-cleanup-old-checkpoints")
    if prune_checkpoints_to_frontier:
        cmd.append("--prune-checkpoints-to-frontier")
    if lineage:
        cmd.extend(["--lineage-out", str(lineage_path)])

    subprocess.run(cmd, check=True)
    RUNS_VOL.commit()
    HF_VOL.commit()
    return {
        "run_id": run_name,
        "frontier": f"{run_name}/frontier.json",
        "state": f"{run_name}/frontier.state.json",
        "lineage": f"{run_name}/lineage.json" if lineage else "",
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
    )
    print(result)

    if not download:
        return
    RUNS_VOL.reload()
    out_root = Path(local_out_dir) / str(result["run_id"])
    out_root.mkdir(parents=True, exist_ok=True)

    for key in ("frontier", "state", "lineage"):
        remote_path = result.get(key) or ""
        if not remote_path:
            continue
        local_path = out_root / Path(remote_path).name
        with local_path.open("wb") as handle:
            RUNS_VOL.read_file_into_fileobj(remote_path, handle)
        print(f"downloaded {remote_path} -> {local_path}")
