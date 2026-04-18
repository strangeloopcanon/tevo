"""Run Parameter Golf benchmark or evolution sweeps on Modal GPUs."""

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
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

GPU = os.environ.get("TEVO_MODAL_GPU", "A10G")
TIMEOUT_S = int(os.environ.get("TEVO_MODAL_TIMEOUT_S", str(60 * 60 * 12)))
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
            "TEVO_PARAMETER_GOLF_ROOT": "/runs",
        }
    )
    .add_local_dir(str(REPO_ROOT), remote_path="/repo", ignore=_ignore_repo_path)
)

app = modal.App("transformer-evolution-llm-parameter-golf")


def _preflight_parameter_golf_config(config_path: str | Path) -> dict[str, object]:
    from transformer_evolution_llm.parameter_golf_runtime import preflight_parameter_golf_config

    return preflight_parameter_golf_config(config_path)


def _commit_loop(stop: threading.Event) -> None:
    while not stop.wait(60.0):
        try:
            RUNS_VOL.commit()
        except Exception as exc:
            print(f"warning: periodic Modal volume commit failed: {exc}", file=sys.stderr)


def _run_with_log(cmd: list[str], log_path: Path) -> int:
    returncode = 1
    stop = threading.Event()
    committer = threading.Thread(target=_commit_loop, args=(stop,), daemon=True)
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
                except Exception as exc:
                    print(f"warning: failed to write run log: {exc}", file=sys.stderr)
                try:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                except Exception as exc:
                    print(f"warning: failed to stream run log: {exc}", file=sys.stderr)
            returncode = int(proc.wait())
    finally:
        stop.set()
        try:
            committer.join(timeout=2.0)
        except Exception as exc:
            print(f"warning: failed to join commit thread: {exc}", file=sys.stderr)
    return returncode


@app.function(
    image=IMAGE,
    gpu=GPU,
    cpu=8,
    memory=32768,
    timeout=TIMEOUT_S,
    volumes={"/runs": RUNS_VOL, "/hf": HF_VOL},
)
def run_parameter_golf(
    *,
    mode: str,
    config_path: str,
    steps: int,
    eval_batches: int,
    generations: int,
    seed: int,
    run_id: str | None,
    max_tokens: int | None,
    cleanup_old_checkpoints: bool,
    prune_checkpoints_to_frontier: bool,
    lineage: bool,
) -> dict[str, str]:
    run_name = run_id or f"pg_{mode}_{int(time.time())}"
    run_root = Path("/runs/parameter_golf_modal") / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = Path("/repo") / cfg_path

    preflight = _preflight_parameter_golf_config(cfg_path)
    preflight_path = run_root / "preflight.json"
    preflight_path.write_text(json.dumps(preflight, indent=2))
    RUNS_VOL.commit()

    checkpoint_dir = run_root / "checkpoints"
    log_path = run_root / "run.log"
    if mode == "benchmark":
        summary_path = run_root / "summary.json"
        cmd = [
            "python",
            "/repo/scripts/run_parameter_golf.py",
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
            str(summary_path),
            "--checkpoint-dir",
            str(checkpoint_dir),
        ]
        if max_tokens is not None:
            cmd.extend(["--max-tokens", str(int(max_tokens))])
    elif mode == "evolution":
        frontier_path = run_root / "frontier.json"
        lineage_path = run_root / "lineage.json"
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
            str(frontier_path),
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
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    started = {
        "mode": mode,
        "config_path": str(cfg_path),
        "seed": int(seed),
        "steps": int(steps),
        "eval_batches": int(eval_batches),
        "generations": int(generations),
        "max_tokens": int(max_tokens) if max_tokens is not None else None,
        "cmd": cmd,
        "gpu": str(GPU),
        "timeout_s": int(TIMEOUT_S),
        "timestamp_unix_s": time.time(),
    }
    (run_root / "run.started.json").write_text(json.dumps(started, indent=2))
    RUNS_VOL.commit()

    returncode = _run_with_log(cmd, log_path)
    (run_root / "run.subprocess.json").write_text(
        json.dumps(
            {"cmd": cmd, "returncode": returncode, "timestamp_unix_s": time.time()}, indent=2
        )
    )
    RUNS_VOL.commit()
    HF_VOL.commit()

    result: dict[str, str] = {
        "run_id": run_name,
        "mode": mode,
        "preflight": f"parameter_golf_modal/{run_name}/preflight.json",
        "log": f"parameter_golf_modal/{run_name}/run.log",
        "returncode": str(returncode),
    }
    if mode == "benchmark":
        result["summary"] = f"parameter_golf_modal/{run_name}/summary.json"
    else:
        result["frontier"] = f"parameter_golf_modal/{run_name}/frontier.json"
        result["state"] = f"parameter_golf_modal/{run_name}/frontier.state.json"
        result["manifest"] = f"parameter_golf_modal/{run_name}/frontier.manifest.json"
        result["lineage"] = f"parameter_golf_modal/{run_name}/lineage.json"
    return result


@app.local_entrypoint()
def main(
    config_path: str,
    mode: str = "benchmark",
    steps: int = 240,
    eval_batches: int = 4,
    generations: int = 24,
    seed: int = 0,
    run_id: str | None = None,
    max_tokens: int | None = None,
    download: bool = False,
    local_out_dir: str = "runs/modal_parameter_golf",
    cleanup_old_checkpoints: bool = True,
    prune_checkpoints_to_frontier: bool = False,
    lineage: bool = False,
) -> None:
    result = run_parameter_golf.remote(
        mode=mode,
        config_path=config_path,
        steps=steps,
        eval_batches=eval_batches,
        generations=generations,
        seed=seed,
        run_id=run_id,
        max_tokens=max_tokens,
        cleanup_old_checkpoints=cleanup_old_checkpoints,
        prune_checkpoints_to_frontier=prune_checkpoints_to_frontier,
        lineage=lineage,
    )
    print(result)

    if not download:
        return
    out_root = Path(local_out_dir) / str(result["run_id"])
    out_root.mkdir(parents=True, exist_ok=True)
    keys = (
        ["preflight", "log", "summary"]
        if mode == "benchmark"
        else [
            "preflight",
            "log",
            "frontier",
            "state",
            "manifest",
            "lineage",
        ]
    )
    for key in keys:
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
