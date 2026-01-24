"""Run scripts/prepare_packed_data.py on Modal (CPU) and write into the runs volume."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]

TIMEOUT_S = int(os.environ.get("TEVO_MODAL_TIMEOUT_S", str(60 * 60 * 6)))

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
    .pip_install(
        "datasets>=3.0",
        "numpy>=1.26",
        "tokenizers>=0.19",
        "tqdm>=4.66",
        "transformers>=4.44",
        "typer[all]>=0.12",
    )
    .env(
        {
            "PYTHONPATH": "/repo/src",
            "HF_HOME": "/hf",
            "HF_DATASETS_CACHE": "/hf/datasets",
            "TRANSFORMERS_CACHE": "/hf/transformers",
            "TOKENIZERS_PARALLELISM": "false",
            "TEVO_PACKED_ROOT": "/runs",
        }
    )
    .add_local_dir(str(REPO_ROOT), remote_path="/repo", ignore=_ignore_repo_path)
)

app = modal.App("transformer-evolution-llm-packed-data")


@app.function(
    image=IMAGE,
    cpu=8,
    memory=32768,
    timeout=TIMEOUT_S,
    volumes={"/runs": RUNS_VOL, "/hf": HF_VOL},
)
def prepare_packed_data(
    *,
    dataset: str,
    out_dir: str,
    dataset_config: str | None,
    tokenizer_name: str,
    hf_revision: str,
    streaming: bool,
    seed: int,
    dtype: str,
    append_eos: bool,
    max_samples: int | None,
    max_train_tokens: int | None,
    max_val_tokens: int | None,
    val_fraction: float,
    run_id: str | None,
) -> dict[str, str]:
    run_name = run_id or f"packed_{int(time.time())}"
    base = Path("/runs/packed")
    target = base / out_dir
    target.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "/repo/scripts/prepare_packed_data.py",
        "--dataset",
        dataset,
        "--out-dir",
        str(target),
        "--tokenizer-name",
        tokenizer_name,
        "--hf-revision",
        hf_revision,
        "--seed",
        str(int(seed)),
        "--dtype",
        dtype,
        "--val-fraction",
        str(float(val_fraction)),
    ]
    if dataset_config:
        cmd.extend(["--dataset-config", dataset_config])
    if streaming:
        cmd.append("--streaming")
    else:
        cmd.append("--no-streaming")
    if append_eos:
        cmd.append("--append-eos")
    else:
        cmd.append("--no-append-eos")
    if max_samples is not None:
        cmd.extend(["--max-samples", str(int(max_samples))])
    if max_train_tokens is not None:
        cmd.extend(["--max-train-tokens", str(int(max_train_tokens))])
    if max_val_tokens is not None:
        cmd.extend(["--max-val-tokens", str(int(max_val_tokens))])

    cmd.append("--hard-exit")
    subprocess.run(cmd, check=True)
    RUNS_VOL.commit()
    HF_VOL.commit()
    return {
        "run_id": run_name,
        "train": f"packed/{out_dir}/train.bin",
        "val": f"packed/{out_dir}/val.bin",
        "metadata": f"packed/{out_dir}/metadata.json",
    }


@app.local_entrypoint()
def main(
    dataset: str = "openwebtext",
    out_dir: str = "openwebtext_10m",
    dataset_config: str | None = None,
    tokenizer_name: str = "gpt2",
    hf_revision: str = "main",
    streaming: bool = True,
    seed: int = 1234,
    dtype: str = "uint16",
    append_eos: bool = True,
    max_samples: int | None = None,
    max_train_tokens: int | None = 10_000_000,
    max_val_tokens: int | None = 1_000_000,
    val_fraction: float = 0.1,
    run_id: str | None = None,
    download_metadata: bool = False,
    local_out_dir: str = "runs/modal",
) -> None:
    result = prepare_packed_data.remote(
        dataset=dataset,
        out_dir=out_dir,
        dataset_config=dataset_config,
        tokenizer_name=tokenizer_name,
        hf_revision=hf_revision,
        streaming=streaming,
        seed=seed,
        dtype=dtype,
        append_eos=append_eos,
        max_samples=max_samples,
        max_train_tokens=max_train_tokens,
        max_val_tokens=max_val_tokens,
        val_fraction=val_fraction,
        run_id=run_id,
    )
    print(result)

    if not download_metadata:
        return
    out_root = Path(local_out_dir) / str(result["run_id"])
    out_root.mkdir(parents=True, exist_ok=True)
    remote_path = result.get("metadata") or ""
    if remote_path:
        local_path = out_root / "metadata.json"
        with local_path.open("wb") as handle:
            RUNS_VOL.read_file_into_fileobj(remote_path, handle)
        print(f"downloaded {remote_path} -> {local_path}")
