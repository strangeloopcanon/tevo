from __future__ import annotations

from pathlib import Path

from transformer_evolution_llm.runpod_parameter_golf import (
    DEFAULT_REMOTE_TEVO_RUNS_ROOT,
    OFFICIAL_REMOTE_REQUIREMENTS,
    build_official_parameter_golf_setup_script,
    build_official_parameter_golf_smoke_script,
    build_pod_create_payload,
    build_remote_bash_command,
    build_rsync_command,
    build_ssh_command,
    build_tevo_parameter_golf_benchmark_script,
    build_tevo_parameter_golf_evolution_script,
    build_tevo_parameter_golf_setup_script,
    format_shell_command,
    make_pod_stub,
)


def test_build_pod_create_payload_has_expected_defaults() -> None:
    payload = build_pod_create_payload(name="pg-smoke")
    assert payload["name"] == "pg-smoke"
    assert payload["computeType"] == "GPU"
    assert payload["gpuCount"] == 1
    assert payload["supportPublicIp"] is True
    assert payload["ports"] == ["22/tcp"]


def test_build_ssh_and_rsync_commands_use_stub_connection(tmp_path: Path) -> None:
    pod = make_pod_stub("203.0.113.7", 10432)
    ssh_cmd = build_ssh_command(pod, ssh_key_path="~/.ssh/id_ed25519", remote_command="hostname")
    rsync_cmd = build_rsync_command(
        pod,
        ssh_key_path="~/.ssh/id_ed25519",
        local_path=tmp_path,
        remote_path="/workspace/transformer-evolution-llm",
        excludes=("runs", ".git"),
    )

    assert ssh_cmd[:4] == ["ssh", "-o", "StrictHostKeyChecking=accept-new", "-i"]
    assert ssh_cmd[-2] == "root@203.0.113.7"
    assert ssh_cmd[-1] == "hostname"
    assert rsync_cmd[:3] == ["rsync", "-az", "--delete"]
    assert "--no-owner" in rsync_cmd
    assert "--no-group" in rsync_cmd
    assert "root@203.0.113.7:/workspace/transformer-evolution-llm" in rsync_cmd


def test_official_setup_script_installs_packages_and_downloads_subset() -> None:
    script = build_official_parameter_golf_setup_script(train_shards=1, variant="sp1024")
    assert "git clone" in script
    assert "apt-get install -y git rsync" in script
    assert "python3 -m pip install --upgrade pip" in script
    for package in OFFICIAL_REMOTE_REQUIREMENTS:
        assert package in script
    assert "cached_challenge_fineweb.py --variant sp1024 --train-shards 1" in script


def test_official_smoke_script_is_small_by_default() -> None:
    script = build_official_parameter_golf_smoke_script()
    assert "ITERATIONS=200" in script
    assert "MAX_WALLCLOCK_SECONDS=180" in script
    assert "torchrun --standalone --nproc_per_node=1 train_gpt.py" in script


def test_tevo_scripts_include_remote_env_and_outputs() -> None:
    benchmark = build_tevo_parameter_golf_benchmark_script(
        config_path="configs/pg_lane2_shared_depth.yaml",
        run_id="lane2_smoke",
    )
    evolution = build_tevo_parameter_golf_evolution_script(
        config_path="configs/pg_lane2_shared_depth.yaml",
        run_id="lane2_search",
        mutation_steps=2,
    )

    assert f"TEVO_PARAMETER_GOLF_ROOT={DEFAULT_REMOTE_TEVO_RUNS_ROOT}" in benchmark
    assert "runs/lane2_smoke.summary.json" in benchmark
    assert "scripts/run_parameter_golf.py" in benchmark
    assert f"TEVO_PARAMETER_GOLF_ROOT={DEFAULT_REMOTE_TEVO_RUNS_ROOT}" in evolution
    assert "runs/lane2_search.frontier.json" in evolution
    assert "--prune-checkpoints-to-frontier" in evolution
    assert "--mutation-steps 2" in evolution


def test_tevo_setup_script_creates_official_filename_aliases() -> None:
    script = build_tevo_parameter_golf_setup_script()
    assert 'alias_name="${file#fineweb_}"' in script
    assert "fineweb_train_*.bin" in script
    assert "fineweb_val_*.bin" in script


def test_shell_formatters_quote_multiline_scripts() -> None:
    remote = build_remote_bash_command("echo hello\npwd")
    rendered = format_shell_command(["ssh", "root@example", remote])
    assert remote.startswith("bash -lc ")
    assert "echo hello" in remote
    assert "ssh root@example" in rendered
