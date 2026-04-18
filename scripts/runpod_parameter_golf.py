"""Manage Runpod-based Parameter Golf smoke runs for TEVO."""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from transformer_evolution_llm.api import load_spec  # noqa: E402
from transformer_evolution_llm.parameter_golf import (  # noqa: E402
    estimate_artifact_total_bytes_for_spec,
)
from transformer_evolution_llm.parameter_golf_runtime import (  # noqa: E402
    preflight_parameter_golf_config,
)
from transformer_evolution_llm.runpod_parameter_golf import (  # noqa: E402
    DEFAULT_GPU_TYPE,
    DEFAULT_IMAGE,
    DEFAULT_REMOTE_PARAMETER_GOLF_DIR,
    DEFAULT_REMOTE_TEVO_DIR,
    DEFAULT_REMOTE_TEVO_RUNS_ROOT,
    build_official_parameter_golf_setup_script,
    build_official_parameter_golf_smoke_script,
    build_pod_create_payload,
    build_remote_bash_command,
    build_rsync_command,
    build_ssh_command,
    build_tevo_parameter_golf_benchmark_script,
    build_tevo_parameter_golf_evolution_script,
    build_tevo_parameter_golf_setup_script,
    create_pod,
    detect_ssh_private_key,
    format_shell_command,
    get_pod,
    list_pods,
    make_pod_stub,
    repo_sync_excludes,
    ssh_address_for_pod,
    start_pod,
    stop_pod,
    terminate_pod,
    wait_for_pod_ssh,
)

app = typer.Typer(help="Runpod helpers for Parameter Golf smoke runs.", no_args_is_help=True)
console = Console()

PodIdOption = Annotated[str | None, typer.Option(help="Runpod pod id.")]
HostOption = Annotated[
    str | None,
    typer.Option(help="Public IP address, as an alternative to --pod-id."),
]
PortOption = Annotated[
    int | None,
    typer.Option(help="SSH port, as an alternative to --pod-id."),
]
SSHKeyOption = Annotated[Path | None, typer.Option(help="SSH private key path.")]
DryRunOption = Annotated[
    bool,
    typer.Option(help="Print commands without running them."),
]


def _resolve_pod(
    *,
    pod_id: str | None,
    host: str | None,
    port: int | None,
) -> dict[str, object]:
    if pod_id:
        return get_pod(pod_id)
    if host and port is not None:
        return make_pod_stub(host, port)
    raise typer.BadParameter("Provide either --pod-id or both --host and --port.")


def _resolve_ssh_key(ssh_key_path: Path | None) -> Path:
    if ssh_key_path is not None:
        resolved = ssh_key_path.expanduser().resolve()
        if resolved.exists():
            return resolved
        raise typer.BadParameter(f"SSH key not found: {resolved}")

    detected = detect_ssh_private_key()
    if detected is not None:
        return detected.resolve()
    raise typer.BadParameter(
        "No SSH private key was found automatically. Pass --ssh-key-path explicitly."
    )


def _repo_relative_remote_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError as exc:
        raise typer.BadParameter(
            "Config path must live inside this repo so it can be synced to the remote pod."
        ) from exc


def _local_config_preflight(config_path: Path) -> dict[str, object]:
    try:
        report = preflight_parameter_golf_config(config_path)
    except FileNotFoundError as exc:
        spec = load_spec(config_path)
        payload_bytes_est, total_bytes_est = estimate_artifact_total_bytes_for_spec(spec)
        parameter_golf = spec.parameter_golf
        report = {
            "config_path": str(config_path),
            "track": parameter_golf.track if parameter_golf is not None else None,
            "artifact_payload_bytes_est": int(payload_bytes_est),
            "artifact_total_bytes_est": int(total_bytes_est),
            "artifact_budget_bytes": (
                int(parameter_golf.artifact_budget_bytes) if parameter_golf is not None else None
            ),
            "code_bytes": int(parameter_golf.code_bytes) if parameter_golf is not None else None,
            "local_data_check": "missing",
            "warning": str(exc),
        }
    console.print_json(data=report)
    return report


def _run_local_command(cmd: list[str], *, dry_run: bool) -> None:
    console.print(format_shell_command(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)  # noqa: S603


def _run_remote_script(
    *,
    pod: dict[str, object],
    ssh_key_path: Path,
    script: str,
    dry_run: bool,
) -> None:
    cmd = build_ssh_command(
        pod,
        ssh_key_path=ssh_key_path,
        remote_command=build_remote_bash_command(script),
    )
    _run_local_command(cmd, dry_run=dry_run)


def _ensure_remote_dir(
    *,
    pod: dict[str, object],
    ssh_key_path: Path,
    remote_dir: str,
    dry_run: bool,
) -> None:
    _run_remote_script(
        pod=pod,
        ssh_key_path=ssh_key_path,
        script=f"mkdir -p {shlex.quote(remote_dir)}",
        dry_run=dry_run,
    )


def _display_pods(pods: list[dict[str, object]]) -> None:
    table = Table(title="Runpod Pods")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Status")
    table.add_column("GPU")
    table.add_column("SSH")
    for pod in pods:
        pod_id = str(pod.get("id") or "")
        name = str(pod.get("name") or "")
        status = str(pod.get("desiredStatus") or pod.get("machineStatus") or "")
        gpu = pod.get("gpu") or {}
        if isinstance(gpu, dict):
            gpu_name = str(gpu.get("displayName") or gpu.get("id") or "")
            gpu_count = int(gpu.get("count") or 1)
            gpu_text = f"{gpu_count}x {gpu_name}".strip()
        else:
            gpu_text = ""
        ssh_text = "-"
        try:
            host, port = ssh_address_for_pod(pod)
            ssh_text = f"{host}:{port}"
        except Exception:
            ssh_text = "-"
        table.add_row(pod_id, name, status, gpu_text, ssh_text)
    console.print(table)


@app.command("create-pod")
def create_pod_cmd(
    name: Annotated[str, typer.Option(help="Friendly pod name.")],
    gpu_type: Annotated[str, typer.Option()] = DEFAULT_GPU_TYPE,
    image: Annotated[str, typer.Option()] = DEFAULT_IMAGE,
    gpu_count: Annotated[int, typer.Option(min=1)] = 1,
    container_disk_gb: Annotated[int, typer.Option(min=20)] = 80,
    volume_gb: Annotated[int, typer.Option(min=20)] = 100,
    min_vcpu_per_gpu: Annotated[int, typer.Option(min=1)] = 4,
    min_ram_per_gpu: Annotated[int, typer.Option(min=1)] = 30,
    dry_run: Annotated[
        bool,
        typer.Option(help="Print the payload without calling Runpod."),
    ] = False,
) -> None:
    """Create a new Runpod pod or print the payload for one."""
    payload = build_pod_create_payload(
        name=name,
        gpu_type_id=gpu_type,
        image_name=image,
        gpu_count=gpu_count,
        container_disk_gb=container_disk_gb,
        volume_gb=volume_gb,
        min_vcpu_per_gpu=min_vcpu_per_gpu,
        min_ram_per_gpu=min_ram_per_gpu,
    )
    if dry_run:
        console.print_json(data=payload)
        return
    pod = create_pod(payload)
    console.print_json(data=pod)


@app.command("list-pods")
def list_pods_cmd() -> None:
    """List visible Runpod pods."""
    _display_pods(list_pods())


@app.command("start-pod")
def start_pod_cmd(
    pod_id: Annotated[str, typer.Argument(help="Runpod pod id.")],
) -> None:
    """Start or resume a stopped pod."""
    response = start_pod(pod_id)
    console.print_json(data=response if response is not None else {"ok": True})


@app.command("stop-pod")
def stop_pod_cmd(
    pod_id: Annotated[str, typer.Argument(help="Runpod pod id.")],
) -> None:
    """Stop a running pod."""
    response = stop_pod(pod_id)
    console.print_json(data=response if response is not None else {"ok": True})


@app.command("terminate-pod")
def terminate_pod_cmd(
    pod_id: Annotated[str, typer.Argument(help="Runpod pod id.")],
) -> None:
    """Terminate a pod permanently."""
    response = terminate_pod(pod_id)
    console.print_json(data=response if response is not None else {"ok": True})


@app.command("wait-ready")
def wait_ready_cmd(
    pod_id: Annotated[str, typer.Argument(help="Runpod pod id.")],
    timeout_s: Annotated[int, typer.Option(min=10)] = 900,
    poll_interval_s: Annotated[int, typer.Option(min=1)] = 10,
) -> None:
    """Wait for the pod to expose SSH and print the connection info."""
    pod = wait_for_pod_ssh(pod_id, timeout_s=timeout_s, poll_interval_s=poll_interval_s)
    host, port = ssh_address_for_pod(pod)
    console.print_json(data={"id": pod.get("id"), "host": host, "port": port})


@app.command("ssh")
def ssh_cmd(
    pod_id: PodIdOption = None,
    host: HostOption = None,
    port: PortOption = None,
    ssh_key_path: SSHKeyOption = None,
    dry_run: DryRunOption = False,
    remote_command: Annotated[
        list[str] | None,
        typer.Argument(
            help="Optional remote command to run instead of opening an interactive shell."
        ),
    ] = None,
) -> None:
    """Open an SSH session to the pod or run a single remote command."""
    pod = _resolve_pod(pod_id=pod_id, host=host, port=port)
    key_path = _resolve_ssh_key(ssh_key_path)
    command_text = None
    if remote_command:
        command_text = " ".join(remote_command)
    cmd = build_ssh_command(pod, ssh_key_path=key_path, remote_command=command_text)
    _run_local_command(cmd, dry_run=dry_run)


@app.command("sync-repo")
def sync_repo_cmd(
    pod_id: PodIdOption = None,
    host: HostOption = None,
    port: PortOption = None,
    ssh_key_path: SSHKeyOption = None,
    local_path: Annotated[Path, typer.Option(help="Local repo directory to sync.")] = REPO_ROOT,
    remote_path: Annotated[
        str,
        typer.Option(help="Remote repo destination."),
    ] = DEFAULT_REMOTE_TEVO_DIR,
    dry_run: DryRunOption = False,
) -> None:
    """Sync this TEVO repo to the remote pod with rsync."""
    pod = _resolve_pod(pod_id=pod_id, host=host, port=port)
    key_path = _resolve_ssh_key(ssh_key_path)
    local_root = local_path.expanduser().resolve()
    if not local_root.exists():
        raise typer.BadParameter(f"Local path does not exist: {local_root}")
    _ensure_remote_dir(pod=pod, ssh_key_path=key_path, remote_dir=remote_path, dry_run=dry_run)
    rsync_cmd = build_rsync_command(
        pod,
        ssh_key_path=key_path,
        local_path=f"{local_root}/",
        remote_path=remote_path,
        excludes=repo_sync_excludes(),
    )
    _run_local_command(rsync_cmd, dry_run=dry_run)


@app.command("official-setup")
def official_setup_cmd(
    pod_id: PodIdOption = None,
    host: HostOption = None,
    port: PortOption = None,
    ssh_key_path: SSHKeyOption = None,
    train_shards: Annotated[int, typer.Option(min=0)] = 1,
    variant: Annotated[str, typer.Option()] = "sp1024",
    repo_dir: Annotated[str, typer.Option()] = DEFAULT_REMOTE_PARAMETER_GOLF_DIR,
    install_dependencies: Annotated[bool, typer.Option()] = True,
    dry_run: DryRunOption = False,
) -> None:
    """Clone the official repo remotely and download the challenge dataset subset."""
    pod = _resolve_pod(pod_id=pod_id, host=host, port=port)
    key_path = _resolve_ssh_key(ssh_key_path)
    script = build_official_parameter_golf_setup_script(
        repo_dir=repo_dir,
        train_shards=train_shards,
        variant=variant,
        install_dependencies=install_dependencies,
    )
    _run_remote_script(pod=pod, ssh_key_path=key_path, script=script, dry_run=dry_run)


@app.command("official-smoke")
def official_smoke_cmd(
    pod_id: PodIdOption = None,
    host: HostOption = None,
    port: PortOption = None,
    ssh_key_path: SSHKeyOption = None,
    run_id: Annotated[str, typer.Option()] = "baseline_sp1024_smoke",
    variant: Annotated[str, typer.Option()] = "sp1024",
    nproc_per_node: Annotated[int, typer.Option(min=1)] = 1,
    iterations: Annotated[int, typer.Option(min=1)] = 200,
    max_wallclock_seconds: Annotated[int, typer.Option(min=0)] = 180,
    val_loss_every: Annotated[int, typer.Option(min=0)] = 0,
    repo_dir: Annotated[str, typer.Option()] = DEFAULT_REMOTE_PARAMETER_GOLF_DIR,
    dry_run: DryRunOption = False,
) -> None:
    """Run a small official Parameter Golf baseline smoke test on the pod."""
    pod = _resolve_pod(pod_id=pod_id, host=host, port=port)
    key_path = _resolve_ssh_key(ssh_key_path)
    script = build_official_parameter_golf_smoke_script(
        repo_dir=repo_dir,
        run_id=run_id,
        variant=variant,
        nproc_per_node=nproc_per_node,
        iterations=iterations,
        max_wallclock_seconds=max_wallclock_seconds,
        val_loss_every=val_loss_every,
    )
    _run_remote_script(pod=pod, ssh_key_path=key_path, script=script, dry_run=dry_run)


@app.command("tevo-setup")
def tevo_setup_cmd(
    pod_id: PodIdOption = None,
    host: HostOption = None,
    port: PortOption = None,
    ssh_key_path: SSHKeyOption = None,
    tevo_repo_dir: Annotated[str, typer.Option()] = DEFAULT_REMOTE_TEVO_DIR,
    official_repo_dir: Annotated[str, typer.Option()] = DEFAULT_REMOTE_PARAMETER_GOLF_DIR,
    tevo_runs_root: Annotated[str, typer.Option()] = DEFAULT_REMOTE_TEVO_RUNS_ROOT,
    dry_run: DryRunOption = False,
) -> None:
    """Install TEVO on the pod and connect it to the official challenge data."""
    pod = _resolve_pod(pod_id=pod_id, host=host, port=port)
    key_path = _resolve_ssh_key(ssh_key_path)
    script = build_tevo_parameter_golf_setup_script(
        tevo_repo_dir=tevo_repo_dir,
        official_repo_dir=official_repo_dir,
        tevo_runs_root=tevo_runs_root,
    )
    _run_remote_script(pod=pod, ssh_key_path=key_path, script=script, dry_run=dry_run)


@app.command("tevo-benchmark")
def tevo_benchmark_cmd(
    config_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    pod_id: PodIdOption = None,
    host: HostOption = None,
    port: PortOption = None,
    ssh_key_path: SSHKeyOption = None,
    tevo_repo_dir: Annotated[str, typer.Option()] = DEFAULT_REMOTE_TEVO_DIR,
    tevo_runs_root: Annotated[str, typer.Option()] = DEFAULT_REMOTE_TEVO_RUNS_ROOT,
    steps: Annotated[int, typer.Option(min=1)] = 120,
    eval_batches: Annotated[int, typer.Option(min=1)] = 2,
    seed: Annotated[int, typer.Option()] = 0,
    run_id: Annotated[str | None, typer.Option()] = None,
    max_tokens: Annotated[int | None, typer.Option(min=1)] = None,
    dry_run: DryRunOption = False,
) -> None:
    """Run a single TEVO Parameter Golf benchmark on the pod."""
    pod = _resolve_pod(pod_id=pod_id, host=host, port=port)
    key_path = _resolve_ssh_key(ssh_key_path)
    resolved_config = config_path.resolve()
    _local_config_preflight(resolved_config)
    remote_config_path = _repo_relative_remote_path(resolved_config)
    remote_run_id = run_id or f"{resolved_config.stem}_benchmark"
    script = build_tevo_parameter_golf_benchmark_script(
        config_path=remote_config_path,
        repo_dir=tevo_repo_dir,
        tevo_runs_root=tevo_runs_root,
        steps=steps,
        eval_batches=eval_batches,
        seed=seed,
        run_id=remote_run_id,
        max_tokens=max_tokens,
    )
    _run_remote_script(pod=pod, ssh_key_path=key_path, script=script, dry_run=dry_run)


@app.command("tevo-evolution")
def tevo_evolution_cmd(
    config_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    pod_id: PodIdOption = None,
    host: HostOption = None,
    port: PortOption = None,
    ssh_key_path: SSHKeyOption = None,
    tevo_repo_dir: Annotated[str, typer.Option()] = DEFAULT_REMOTE_TEVO_DIR,
    tevo_runs_root: Annotated[str, typer.Option()] = DEFAULT_REMOTE_TEVO_RUNS_ROOT,
    generations: Annotated[int, typer.Option(min=1)] = 24,
    steps: Annotated[int, typer.Option(min=1)] = 120,
    eval_batches: Annotated[int, typer.Option(min=1)] = 2,
    mutation_steps: Annotated[int | None, typer.Option(min=1)] = None,
    seed: Annotated[int, typer.Option()] = 0,
    run_id: Annotated[str | None, typer.Option()] = None,
    dry_run: DryRunOption = False,
) -> None:
    """Run a TEVO live evolution sweep on the pod."""
    pod = _resolve_pod(pod_id=pod_id, host=host, port=port)
    key_path = _resolve_ssh_key(ssh_key_path)
    resolved_config = config_path.resolve()
    _local_config_preflight(resolved_config)
    remote_config_path = _repo_relative_remote_path(resolved_config)
    remote_run_id = run_id or f"{resolved_config.stem}_evolution"
    script = build_tevo_parameter_golf_evolution_script(
        config_path=remote_config_path,
        repo_dir=tevo_repo_dir,
        tevo_runs_root=tevo_runs_root,
        generations=generations,
        steps=steps,
        eval_batches=eval_batches,
        mutation_steps=mutation_steps,
        seed=seed,
        run_id=remote_run_id,
    )
    _run_remote_script(pod=pod, ssh_key_path=key_path, script=script, dry_run=dry_run)


if __name__ == "__main__":
    app()
