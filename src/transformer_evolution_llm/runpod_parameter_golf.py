"""Runpod helpers for Parameter Golf experiments."""

from __future__ import annotations

import json
import os
import shlex
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib import error, request
from urllib.parse import urlsplit

RUNPOD_API_BASE = "https://rest.runpod.io/v1"
DEFAULT_GPU_TYPE = "NVIDIA H100 80GB HBM3"
DEFAULT_IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
DEFAULT_REMOTE_TEVO_DIR = "/workspace/transformer-evolution-llm"
DEFAULT_REMOTE_PARAMETER_GOLF_DIR = "/workspace/parameter-golf"
DEFAULT_REMOTE_TEVO_RUNS_ROOT = "/workspace/tevo_runs"
DEFAULT_PARAMETER_GOLF_REPO = "https://github.com/openai/parameter-golf.git"
COMMON_SSH_KEYS = (
    Path("~/.runpod/ssh/RunPod-Key-Go"),
    Path("~/.ssh/id_ed25519"),
    Path("~/.ssh/id_rsa"),
    Path("~/.ssh/id_ecdsa"),
)
OFFICIAL_REMOTE_REQUIREMENTS = (
    "numpy",
    "sentencepiece",
    "huggingface-hub",
    "datasets",
    "tqdm",
)


def detect_ssh_private_key() -> Path | None:
    """Return the first common SSH private key path that exists."""
    for candidate in COMMON_SSH_KEYS:
        expanded = candidate.expanduser()
        if expanded.exists():
            return expanded
    return None


def repo_sync_excludes() -> tuple[str, ...]:
    """Return the default ignore set when syncing this repo to a remote pod."""
    return (
        ".git",
        ".venv",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "__pycache__",
        "runs",
        "dist",
        "build",
        ".coverage",
        "htmlcov",
        ".DS_Store",
        ".beads",
        ".codex",
        ".context",
    )


def build_pod_create_payload(
    *,
    name: str,
    gpu_type_id: str = DEFAULT_GPU_TYPE,
    image_name: str = DEFAULT_IMAGE,
    gpu_count: int = 1,
    container_disk_gb: int = 80,
    volume_gb: int = 100,
    min_vcpu_per_gpu: int = 4,
    min_ram_per_gpu: int = 30,
    cloud_type: str = "SECURE",
    ports: Iterable[str] = ("22/tcp",),
    support_public_ip: bool = True,
    volume_mount_path: str = "/workspace",
    env: dict[str, str] | None = None,
    allowed_cuda_versions: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Build a Runpod REST payload for a persistent training pod."""
    payload: dict[str, Any] = {
        "name": str(name),
        "computeType": "GPU",
        "gpuTypeIds": [str(gpu_type_id)],
        "gpuCount": int(gpu_count),
        "imageName": str(image_name),
        "containerDiskInGb": int(container_disk_gb),
        "volumeInGb": int(volume_gb),
        "minVCPUPerGPU": int(min_vcpu_per_gpu),
        "minRAMPerGPU": int(min_ram_per_gpu),
        "cloudType": str(cloud_type),
        "ports": [str(port) for port in ports],
        "supportPublicIp": bool(support_public_ip),
        "volumeMountPath": str(volume_mount_path),
        "interruptible": False,
        "locked": False,
        "globalNetworking": True,
    }
    if env:
        payload["env"] = {str(key): str(value) for key, value in env.items()}
    if allowed_cuda_versions:
        payload["allowedCudaVersions"] = [str(item) for item in allowed_cuda_versions]
    return payload


def make_pod_stub(public_ip: str, ssh_port: int) -> dict[str, Any]:
    """Build a minimal pod-like mapping for SSH and rsync helpers."""
    return {
        "id": "manual",
        "publicIp": str(public_ip),
        "portMappings": {"22": int(ssh_port)},
    }


def _read_api_key(api_key: str | None = None) -> str:
    key = str(api_key or os.environ.get("RUNPOD_API_KEY") or "").strip()
    if not key:
        raise ValueError("Runpod API key is missing. Set RUNPOD_API_KEY in the environment.")
    return key


def _runpod_request_url(path: str) -> str:
    clean_path = str(path).strip()
    if not clean_path.startswith("/"):
        raise ValueError(f"Runpod API path must start with '/': {path!r}")
    url = f"{RUNPOD_API_BASE}{clean_path}"
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.netloc != "rest.runpod.io":
        raise ValueError(f"Unexpected Runpod API origin: {url}")
    return url


def runpod_api_request(
    method: str,
    path: str,
    *,
    api_key: str | None = None,
    payload: dict[str, Any] | None = None,
) -> Any:
    """Call the Runpod REST API using the standard Bearer auth header."""
    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
    req = request.Request(  # noqa: S310 - URL is constrained to the official Runpod API host.
        url=_runpod_request_url(path),
        method=str(method).upper(),
        data=body,
        headers={
            "Authorization": f"Bearer {_read_api_key(api_key)}",
            "Content-Type": "application/json",
        },
    )
    try:
        with request.urlopen(req, timeout=60) as response:  # noqa: S310  # nosec B310
            raw = response.read()
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Runpod API {method} {path} failed: {exc.code} {detail}") from exc
    if not raw:
        return None
    return json.loads(raw.decode("utf-8"))


def create_pod(payload: dict[str, Any], *, api_key: str | None = None) -> dict[str, Any]:
    """Create a new Runpod pod."""
    response = runpod_api_request("POST", "/pods", api_key=api_key, payload=payload)
    if not isinstance(response, dict):
        raise RuntimeError("Unexpected Runpod create_pod response")
    return response


def list_pods(*, api_key: str | None = None) -> list[dict[str, Any]]:
    """List all pods visible to the authenticated user."""
    response = runpod_api_request("GET", "/pods", api_key=api_key)
    if isinstance(response, list):
        return [item for item in response if isinstance(item, dict)]
    if isinstance(response, dict):
        items = response.get("data") or response.get("pods") or response.get("items") or []
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
    raise RuntimeError("Unexpected Runpod list_pods response")


def get_pod(pod_id: str, *, api_key: str | None = None) -> dict[str, Any]:
    """Fetch one pod by id."""
    response = runpod_api_request("GET", f"/pods/{pod_id}", api_key=api_key)
    if not isinstance(response, dict):
        raise RuntimeError("Unexpected Runpod get_pod response")
    return response


def start_pod(pod_id: str, *, api_key: str | None = None) -> Any:
    """Start or resume a stopped pod."""
    return runpod_api_request("POST", f"/pods/{pod_id}/start", api_key=api_key)


def stop_pod(pod_id: str, *, api_key: str | None = None) -> Any:
    """Stop a running pod."""
    return runpod_api_request("POST", f"/pods/{pod_id}/stop", api_key=api_key)


def terminate_pod(pod_id: str, *, api_key: str | None = None) -> Any:
    """Terminate a pod permanently."""
    return runpod_api_request("DELETE", f"/pods/{pod_id}", api_key=api_key)


def wait_for_pod_ssh(
    pod_id: str,
    *,
    api_key: str | None = None,
    timeout_s: int = 900,
    poll_interval_s: int = 10,
) -> dict[str, Any]:
    """Poll until the pod has a public IP and an exposed SSH port."""
    deadline = time.time() + max(1, int(timeout_s))
    while time.time() < deadline:
        pod = get_pod(pod_id, api_key=api_key)
        if pod_ssh_ready(pod):
            return pod
        time.sleep(max(1, int(poll_interval_s)))
    raise TimeoutError(f"Timed out waiting for pod {pod_id} to expose SSH")


def pod_ssh_ready(pod: dict[str, Any]) -> bool:
    """Return whether the pod has enough information for an SSH connection."""
    public_ip = pod.get("publicIp")
    port_mappings = pod.get("portMappings")
    if not public_ip or not isinstance(port_mappings, dict):
        return False
    port = port_mappings.get("22") or port_mappings.get(22)
    return bool(port)


def ssh_address_for_pod(pod: dict[str, Any]) -> tuple[str, int]:
    """Return `(public_ip, ssh_port)` for a pod."""
    if not pod_ssh_ready(pod):
        raise ValueError("Pod does not expose SSH yet.")
    public_ip = str(pod["publicIp"])
    port_mappings = pod.get("portMappings", {})
    port_raw = port_mappings.get("22") or port_mappings.get(22)
    return public_ip, int(port_raw)


def build_ssh_command(
    pod: dict[str, Any],
    *,
    ssh_key_path: str | Path,
    remote_command: str | None = None,
) -> list[str]:
    """Build an SSH command for the given pod."""
    host, port = ssh_address_for_pod(pod)
    cmd = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-i",
        str(Path(ssh_key_path).expanduser()),
        "-p",
        str(port),
        f"root@{host}",
    ]
    if remote_command is not None:
        cmd.append(remote_command)
    return cmd


def build_rsync_command(
    pod: dict[str, Any],
    *,
    ssh_key_path: str | Path,
    local_path: str | Path,
    remote_path: str | Path,
    excludes: Iterable[str] | None = None,
) -> list[str]:
    """Build an rsync command that reuses the pod's SSH address."""
    host, port = ssh_address_for_pod(pod)
    ssh_cmd = (
        "ssh -o StrictHostKeyChecking=accept-new "
        f"-i {shlex.quote(str(Path(ssh_key_path).expanduser()))} "
        f"-p {int(port)}"
    )
    cmd = ["rsync", "-az", "--delete"]
    cmd.extend(["--no-owner", "--no-group", "--omit-dir-times", "--no-perms"])
    for item in excludes or ():
        cmd.extend(["--exclude", str(item)])
    cmd.extend(["-e", ssh_cmd, str(local_path), f"root@{host}:{remote_path}"])
    return cmd


def build_remote_bash_command(script: str) -> str:
    """Wrap a multi-line shell script for remote execution over SSH."""
    return f"bash -lc {shlex.quote(str(script))}"


def format_shell_command(cmd: Iterable[str]) -> str:
    """Render a subprocess command list as a shell-safe string."""
    return " ".join(shlex.quote(str(part)) for part in cmd)


def _shell_join(lines: Iterable[str]) -> str:
    return "\n".join(str(line).rstrip() for line in lines if str(line).strip())


def build_official_parameter_golf_setup_script(
    *,
    repo_dir: str = DEFAULT_REMOTE_PARAMETER_GOLF_DIR,
    repo_url: str = DEFAULT_PARAMETER_GOLF_REPO,
    train_shards: int = 1,
    variant: str = "sp1024",
    install_dependencies: bool = True,
) -> str:
    """Build the remote shell script that clones the official repo and downloads data."""
    parent = str(Path(repo_dir).parent)
    repo_name = Path(repo_dir).name
    lines = [
        "set -euo pipefail",
        f"mkdir -p {shlex.quote(parent)}",
        f"cd {shlex.quote(parent)}",
        (
            "if [ ! -d "
            f"{shlex.quote(repo_name)}"
            " ]; then git clone "
            f"{shlex.quote(repo_url)} {shlex.quote(repo_name)}; fi"
        ),
        f"cd {shlex.quote(repo_name)}",
        "git fetch --all --prune",
        "git checkout main",
    ]
    if install_dependencies:
        packages = " ".join(shlex.quote(item) for item in OFFICIAL_REMOTE_REQUIREMENTS)
        lines.extend(
            [
                "apt update",
                "DEBIAN_FRONTEND=noninteractive apt-get install -y git rsync",
                "python3 -m pip install --upgrade pip",
                f"python3 -m pip install {packages}",
            ]
        )
    lines.append(
        "python3 data/cached_challenge_fineweb.py "
        f"--variant {shlex.quote(str(variant))} --train-shards {int(train_shards)}"
    )
    return _shell_join(lines)


def build_official_parameter_golf_smoke_script(
    *,
    repo_dir: str = DEFAULT_REMOTE_PARAMETER_GOLF_DIR,
    run_id: str = "baseline_sp1024_smoke",
    variant: str = "sp1024",
    nproc_per_node: int = 1,
    iterations: int = 200,
    max_wallclock_seconds: int = 180,
    val_loss_every: int = 0,
) -> str:
    """Build a small official Parameter Golf smoke-run command."""
    return _shell_join(
        [
            "set -euo pipefail",
            f"cd {shlex.quote(repo_dir)}",
            f"RUN_ID={shlex.quote(run_id)} \\",
            f"DATA_PATH=./data/datasets/fineweb10B_{shlex.quote(variant)}/ \\",
            "TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \\",
            "VOCAB_SIZE=1024 \\",
            f"ITERATIONS={int(iterations)} \\",
            f"MAX_WALLCLOCK_SECONDS={int(max_wallclock_seconds)} \\",
            f"VAL_LOSS_EVERY={int(val_loss_every)} \\",
            f"torchrun --standalone --nproc_per_node={int(nproc_per_node)} train_gpt.py",
        ]
    )


def build_tevo_parameter_golf_setup_script(
    *,
    tevo_repo_dir: str = DEFAULT_REMOTE_TEVO_DIR,
    official_repo_dir: str = DEFAULT_REMOTE_PARAMETER_GOLF_DIR,
    tevo_runs_root: str = DEFAULT_REMOTE_TEVO_RUNS_ROOT,
) -> str:
    """Build the remote shell script that prepares TEVO against the official dataset cache."""
    tokenizer_src = Path(official_repo_dir) / "data" / "tokenizers" / "fineweb_1024_bpe.model"
    dataset_src = Path(official_repo_dir) / "data" / "datasets" / "fineweb10B_sp1024"
    tevo_root = Path(tevo_runs_root)
    dataset_dst = tevo_root / "parameter_golf" / "fineweb10b_sp1024"
    tokenizer_dst = tevo_root / "parameter_golf" / "tokenizers" / "sp1024.model"
    return _shell_join(
        [
            "set -euo pipefail",
            f"mkdir -p {shlex.quote(str(Path(tevo_repo_dir).parent))}",
            f"mkdir -p {shlex.quote(str(tokenizer_dst.parent))}",
            f"mkdir -p {shlex.quote(str(dataset_dst.parent))}",
            f"cd {shlex.quote(tevo_repo_dir)}",
            "python3 -m pip install --upgrade pip",
            "python3 -m pip install -e .",
            f"cd {shlex.quote(str(dataset_src))}",
            "for file in fineweb_train_*.bin; do "
            'alias_name="${file#fineweb_}"; '
            'ln -sfn "$file" "$alias_name"; '
            "done",
            "for file in fineweb_val_*.bin; do "
            'alias_name="${file#fineweb_}"; '
            'ln -sfn "$file" "$alias_name"; '
            "done",
            f"cd {shlex.quote(tevo_repo_dir)}",
            f"ln -sfn {shlex.quote(str(dataset_src))} {shlex.quote(str(dataset_dst))}",
            f"ln -sfn {shlex.quote(str(tokenizer_src))} {shlex.quote(str(tokenizer_dst))}",
        ]
    )


def build_tevo_parameter_golf_env(
    *,
    tevo_runs_root: str = DEFAULT_REMOTE_TEVO_RUNS_ROOT,
) -> dict[str, str]:
    """Return the environment variables needed for TEVO to find remote challenge assets."""
    return {
        "TEVO_PARAMETER_GOLF_ROOT": str(tevo_runs_root),
        "TEVO_PACKED_ROOT": str(tevo_runs_root),
        "TOKENIZERS_PARALLELISM": "false",
    }


def _env_prefix(env: dict[str, str]) -> str:
    parts = [f"{key}={shlex.quote(str(value))}" for key, value in env.items()]
    return " ".join(parts)


def build_tevo_parameter_golf_benchmark_script(
    *,
    config_path: str,
    repo_dir: str = DEFAULT_REMOTE_TEVO_DIR,
    tevo_runs_root: str = DEFAULT_REMOTE_TEVO_RUNS_ROOT,
    steps: int = 120,
    eval_batches: int = 2,
    seed: int = 0,
    run_id: str = "tevo_pg_benchmark",
    max_tokens: int | None = None,
) -> str:
    """Build the remote TEVO benchmark command."""
    env = build_tevo_parameter_golf_env(tevo_runs_root=tevo_runs_root)
    cmd = [
        "python3",
        "scripts/run_parameter_golf.py",
        str(config_path),
        "--device",
        "cuda",
        "--steps",
        str(int(steps)),
        "--eval-batches",
        str(int(eval_batches)),
        "--seed",
        str(int(seed)),
        "--out",
        f"runs/{run_id}.summary.json",
        "--checkpoint-dir",
        f"runs/{run_id}.checkpoints",
    ]
    if max_tokens is not None:
        cmd.extend(["--max-tokens", str(int(max_tokens))])
    return _shell_join(
        [
            "set -euo pipefail",
            f"cd {shlex.quote(repo_dir)}",
            f"{_env_prefix(env)} {' '.join(shlex.quote(part) for part in cmd)}",
        ]
    )


def build_tevo_parameter_golf_evolution_script(
    *,
    config_path: str,
    repo_dir: str = DEFAULT_REMOTE_TEVO_DIR,
    tevo_runs_root: str = DEFAULT_REMOTE_TEVO_RUNS_ROOT,
    generations: int = 24,
    steps: int = 120,
    eval_batches: int = 2,
    seed: int = 0,
    run_id: str = "tevo_pg_evolution",
    mutation_steps: int | None = None,
) -> str:
    """Build the remote TEVO evolution command."""
    env = build_tevo_parameter_golf_env(tevo_runs_root=tevo_runs_root)
    cmd = [
        "python3",
        "-u",
        "scripts/run_live.py",
        str(config_path),
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
        f"runs/{run_id}.frontier.json",
        "--checkpoint-dir",
        f"runs/{run_id}.checkpoints",
        "--prune-checkpoints-to-frontier",
    ]
    if mutation_steps is not None:
        cmd.extend(["--mutation-steps", str(int(mutation_steps))])
    return _shell_join(
        [
            "set -euo pipefail",
            f"cd {shlex.quote(repo_dir)}",
            f"{_env_prefix(env)} {' '.join(shlex.quote(part) for part in cmd)}",
        ]
    )


__all__ = [
    "COMMON_SSH_KEYS",
    "DEFAULT_GPU_TYPE",
    "DEFAULT_IMAGE",
    "DEFAULT_PARAMETER_GOLF_REPO",
    "DEFAULT_REMOTE_PARAMETER_GOLF_DIR",
    "DEFAULT_REMOTE_TEVO_DIR",
    "DEFAULT_REMOTE_TEVO_RUNS_ROOT",
    "OFFICIAL_REMOTE_REQUIREMENTS",
    "RUNPOD_API_BASE",
    "build_official_parameter_golf_setup_script",
    "build_official_parameter_golf_smoke_script",
    "build_pod_create_payload",
    "build_remote_bash_command",
    "build_rsync_command",
    "build_ssh_command",
    "build_tevo_parameter_golf_benchmark_script",
    "build_tevo_parameter_golf_env",
    "build_tevo_parameter_golf_evolution_script",
    "build_tevo_parameter_golf_setup_script",
    "create_pod",
    "detect_ssh_private_key",
    "format_shell_command",
    "get_pod",
    "list_pods",
    "make_pod_stub",
    "pod_ssh_ready",
    "repo_sync_excludes",
    "runpod_api_request",
    "ssh_address_for_pod",
    "start_pod",
    "stop_pod",
    "terminate_pod",
    "wait_for_pod_ssh",
]
