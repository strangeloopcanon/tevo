"""Run scripts/run_benchmark.py on Modal GPUs."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]

GPU = os.environ.get("TEVO_MODAL_GPU", "A10G")
TIMEOUT_S = int(os.environ.get("TEVO_MODAL_TIMEOUT_S", str(60 * 60 * 6)))
PYTORCH_INDEX_URL = os.environ.get("TEVO_TORCH_INDEX_URL", "https://download.pytorch.org/whl/cu124")
TORCH_VERSION = os.environ.get("TEVO_TORCH_VERSION", "2.6.0+cu124")

RUNS_VOLUME_NAME = os.environ.get("TEVO_MODAL_RUNS_VOLUME", "tevo-runs")
HF_VOLUME_NAME = os.environ.get("TEVO_MODAL_HF_VOLUME", "tevo-hf-cache")

RUNS_VOL = modal.Volume.from_name(RUNS_VOLUME_NAME, create_if_missing=True)
HF_VOL = modal.Volume.from_name(HF_VOLUME_NAME, create_if_missing=True)


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

app = modal.App("transformer-evolution-llm-benchmark")


@app.function(
    image=IMAGE,
    gpu=GPU,
    cpu=8,
    memory=32768,
    timeout=TIMEOUT_S,
    volumes={"/runs": RUNS_VOL, "/hf": HF_VOL},
)
def run_benchmark(
    *,
    config_path: str,
    steps: int,
    eval_batches: int,
    seed: int,
    run_id: str | None,
    max_tokens: int | None,
) -> dict[str, str]:
    run_name = run_id or f"bench_{int(time.time())}"
    runs_root = Path("/runs/benchmarks")
    run_root = runs_root / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    out_path = run_root / "summary.json"
    history_path = run_root / "history.json"

    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = Path("/repo") / cfg_path

    cmd = [
        "python",
        "/repo/scripts/run_benchmark.py",
        str(cfg_path),
        "--device",
        "cuda",
        "--steps",
        str(int(steps)),
        "--eval-batches",
        str(int(eval_batches)),
        "--seed",
        str(int(seed)),
        "--out",
        str(out_path),
        "--history-out",
        str(history_path),
    ]
    if max_tokens is not None:
        cmd.extend(["--max-tokens", str(int(max_tokens))])

    subprocess.run(cmd, check=True)
    RUNS_VOL.commit()
    HF_VOL.commit()
    return {
        "run_id": run_name,
        "summary": f"benchmarks/{run_name}/summary.json",
        "history": f"benchmarks/{run_name}/history.json",
    }


@app.local_entrypoint()
def main(
    config_path: str,
    steps: int = 240,
    eval_batches: int = 4,
    seed: int = 0,
    run_id: str | None = None,
    max_tokens: int | None = None,
    download: bool = False,
    local_out_dir: str = "runs/modal",
) -> None:
    result = run_benchmark.remote(
        config_path=config_path,
        steps=steps,
        eval_batches=eval_batches,
        seed=seed,
        run_id=run_id,
        max_tokens=max_tokens,
    )
    print(result)

    if not download:
        return
    out_root = Path(local_out_dir) / str(result["run_id"])
    out_root.mkdir(parents=True, exist_ok=True)

    for key in ("summary", "history"):
        remote_path = result.get(key) or ""
        if not remote_path:
            continue
        local_path = out_root / Path(remote_path).name
        with local_path.open("wb") as handle:
            RUNS_VOL.read_file_into_fileobj(remote_path, handle)
        print(f"downloaded {remote_path} -> {local_path}")
