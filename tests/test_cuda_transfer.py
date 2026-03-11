import shutil
from pathlib import Path
from types import SimpleNamespace

import ujson as json
from typer.testing import CliRunner

from transformer_evolution_llm.cli import app
from transformer_evolution_llm.cuda_transfer import (
    build_public_cuda_transfer_report,
    render_public_cuda_variants,
    resolve_autoresearch_source_repo,
    run_public_cuda_transfer_modal_benchmarks,
)
from transformer_evolution_llm.dsl import ArchitectureSpec
from transformer_evolution_llm.mlx_transfer import (
    export_public_transfer_recipes,
    stage_public_transfer_workspaces,
    write_winning_seed_diff,
)

CUDA_SAMPLE = """
# Model architecture
ASPECT_RATIO = 64
HEAD_DIM = 128
WINDOW_PATTERN = "SSSL"

# Optimization
TOTAL_BATCH_SIZE = 2**19
EMBEDDING_LR = 0.6
UNEMBEDDING_LR = 0.004
MATRIX_LR = 0.04
SCALAR_LR = 0.5
WEIGHT_DECAY = 0.2
ADAM_BETAS = (0.8, 0.95)
WARMUP_RATIO = 0.0
WARMDOWN_RATIO = 0.5
FINAL_LR_FRAC = 0.0

# Model size
DEPTH = 8
DEVICE_BATCH_SIZE = 128


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    return x


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    pass


class CausalSelfAttention(nn.Module):
    def forward(self, q, k, cos, sin):
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        return q


def build_model_config(depth):
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )


config = build_model_config(DEPTH)
"""


def _make_spec(
    *,
    name: str,
    hidden: int,
    kv_groups: int,
    norm: str = "layernorm",
    activation: str = "gelu",
) -> ArchitectureSpec:
    return ArchitectureSpec(
        model={
            "name": name,
            "emb": {"dim": 512, "vocab": 50257},
            "blocks": [
                {
                    "attn": {
                        "kind": "GQA",
                        "heads": 8,
                        "head_dim": 64,
                        "kv_groups": kv_groups,
                        "sparsity": "sliding",
                        "sliding_window": 512,
                    },
                    "ffn": {"type": "dense", "hidden": hidden, "activation": activation},
                },
                {
                    "attn": {
                        "kind": "GQA",
                        "heads": 8,
                        "head_dim": 64,
                        "kv_groups": kv_groups,
                        "sparsity": "none",
                    },
                    "ffn": {"type": "dense", "hidden": hidden, "activation": activation},
                },
            ],
            "head": {"tie_embeddings": True, "vocab": 50257},
            "norm": norm,
        },
        train={
            "lr": 5.0e-4,
            "warmup": 10,
            "clip": 1.0,
            "weight_decay": 0.08,
            "grad_ckpt": False,
        },
        data={
            "tokenizer": "gpt2",
            "seq_len": 1024,
            "batch_size": 4,
            "workers": 0,
            "packed": True,
            "packed_train_path": "runs/packed/openwebtext_10m/train.bin",
            "shards": [],
        },
    )


def _write_frontier(tmp_path: Path) -> Path:
    spec_a = _make_spec(name="quality-shared", hidden=1536, kv_groups=2)
    spec_b = _make_spec(name="quality-shared-again", hidden=1536, kv_groups=2)
    spec_c = _make_spec(name="compute-candidate", hidden=2048, kv_groups=1)
    spec_d = _make_spec(name="balanced-candidate", hidden=1792, kv_groups=2, norm="rmsnorm")
    frontier = tmp_path / "frontier.json"
    frontier.write_text(
        json.dumps(
            [
                {
                    "id": "quality-a",
                    "metrics": {"ppl_code": 1.0, "speedrun_flops_to_target": 20.0},
                    "spec": spec_a.model_dump(mode="python"),
                },
                {
                    "id": "quality-b",
                    "metrics": {"ppl_code": 1.1, "speedrun_flops_to_target": 10.0},
                    "spec": spec_b.model_dump(mode="python"),
                },
                {
                    "id": "compute-a",
                    "metrics": {"ppl_code": 1.4, "speedrun_flops_to_target": 5.0},
                    "spec": spec_c.model_dump(mode="python"),
                },
                {
                    "id": "balanced-a",
                    "metrics": {"ppl_code": 1.2, "speedrun_flops_to_target": 8.0},
                    "spec": spec_d.model_dump(mode="python"),
                },
            ]
        )
    )
    return frontier


def _write_fake_autoresearch_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "autoresearch"
    repo.mkdir()
    (repo / "train.py").write_text(CUDA_SAMPLE)
    (repo / "prepare.py").write_text("MAX_SEQ_LEN = 2048\n")
    return repo


def _run_git(command: list[str], *, cwd: Path) -> None:
    import subprocess

    subprocess.run(command, cwd=cwd, check=True)  # noqa: S603  # nosec B603


def test_resolve_autoresearch_source_repo_uses_git_metadata(tmp_path: Path) -> None:
    repo = _write_fake_autoresearch_repo(tmp_path)

    git = shutil.which("git")
    assert git is not None
    _run_git([git, "init"], cwd=repo)
    _run_git([git, "config", "user.email", "tests@example.com"], cwd=repo)
    _run_git([git, "config", "user.name", "Tests"], cwd=repo)
    _run_git(
        [git, "remote", "add", "origin", "https://github.com/karpathy/autoresearch.git"],
        cwd=repo,
    )
    _run_git([git, "add", "train.py", "prepare.py"], cwd=repo)
    _run_git([git, "commit", "-m", "seed"], cwd=repo)

    resolved, metadata = resolve_autoresearch_source_repo(out_dir=tmp_path / "out", local_repo=repo)
    assert resolved == repo.resolve()
    assert metadata["source_kind"] == "local"
    assert metadata["repo_url"] == "https://github.com/karpathy/autoresearch.git"
    assert metadata["repo_ref"] is not None


def test_render_and_stage_public_cuda_transfer_run(tmp_path: Path) -> None:
    frontier = _write_frontier(tmp_path)
    repo = _write_fake_autoresearch_repo(tmp_path)
    _, selection_manifest = export_public_transfer_recipes(frontier, tmp_path / "recipes")
    rendered, _ = render_public_cuda_variants(repo, selection_manifest, tmp_path / "rendered")
    arm_manifest = stage_public_transfer_workspaces(
        mlx_repo=repo,
        rendered_variants=rendered,
        out_dir=tmp_path / "arms",
    )

    quality_train = Path(json.loads(arm_manifest.read_text())["arms"][1]["train_py"]).read_text()
    assert "TEVO TRAIN RECIPE: CONSTANTS START" in quality_train
    assert "MODEL_DIM = 512" in quality_train
    assert "N_KV_HEAD = 4" in quality_train
    assert 'VALUE_EMBED_MODE = "alternate"' in quality_train
    assert "DEVICE_BATCH_SIZE = 128" in quality_train


def test_run_public_cuda_transfer_modal_benchmarks_with_fake_modal(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frontier = _write_frontier(tmp_path)
    repo = _write_fake_autoresearch_repo(tmp_path)
    _, selection_manifest = export_public_transfer_recipes(frontier, tmp_path / "recipes")
    rendered, _ = render_public_cuda_variants(repo, selection_manifest, tmp_path / "rendered")
    arm_manifest = stage_public_transfer_workspaces(
        mlx_repo=repo,
        rendered_variants=rendered,
        out_dir=tmp_path / "arms",
    )

    scores = {
        "baseline": 2.5,
        "quality": 1.25,
        "compute": 1.5,
        "balanced": 1.35,
    }

    def fake_run(command, check, cwd, env):
        run_id = command[command.index("--run-id") + 1]
        local_out_dir = Path(command[command.index("--local-out-dir") + 1])
        run_dir = local_out_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        arm = run_id.split("_")[-1]
        summary = {
            "results": [
                {
                    "run_index": idx,
                    "status": "ok",
                    "val_bpb": scores[arm],
                    "peak_memory_gb": 12.0,
                    "log_path": f"train_run{idx}.log",
                }
                for idx in range(1, 4)
            ]
        }
        (run_dir / "summary.json").write_text(json.dumps(summary))
        for idx in range(1, 4):
            (run_dir / f"train_run{idx}.log").write_text(
                f"val_bpb:          {scores[arm]:.6f}\npeak_vram_mb:     12288.0\n"
            )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("transformer_evolution_llm.cuda_transfer.subprocess.run", fake_run)
    _, _, summary_path, markdown_path = run_public_cuda_transfer_modal_benchmarks(
        arm_manifest,
        repo_url="https://github.com/karpathy/autoresearch.git",
        repo_ref="deadbeef",
        out_dir=tmp_path / "cuda_results",
        repo_root=Path.cwd(),
        modal_gpu="A10G",
        repeat=3,
        timeout_minutes=10,
        modal_local_out_dir=tmp_path / "modal",
    )
    summary = json.loads(summary_path.read_text())
    assert summary["winner"] == "quality"
    assert "quality" in markdown_path.read_text()

    diff_path = write_winning_seed_diff(arm_manifest, summary_path, tmp_path / "winner.diff")
    assert diff_path.exists()
    assert "TEVO TRAIN RECIPE" in diff_path.read_text()


def test_build_public_cuda_transfer_report_and_cli(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "transfer_prepare_manifest.json").write_text(
        json.dumps(
            {
                "tevo_mode": "modal",
                "modal_gpu": "A10G",
                "autoresearch_repo_url": "https://github.com/karpathy/autoresearch.git",
                "autoresearch_repo_ref": "deadbeef",
            }
        )
    )
    (run_root / "tevo_seed_summary.json").write_text(
        json.dumps({"metrics": {"ppl_code": 1.23, "speedrun_flops_to_target": 4567.0}})
    )
    results_dir = run_root / "cuda_results"
    results_dir.mkdir()
    (results_dir / "benchmark_summary.json").write_text(
        json.dumps(
            {
                "arms": [
                    {
                        "arm": "baseline",
                        "runs": 3,
                        "successful_runs": 3,
                        "median_val_bpb": 2.5,
                        "best_val_bpb": 2.5,
                        "median_peak_memory_gb": 12.0,
                        "all_val_bpb": [2.5, 2.5, 2.5],
                    },
                    {
                        "arm": "quality",
                        "runs": 3,
                        "successful_runs": 3,
                        "median_val_bpb": 1.25,
                        "best_val_bpb": 1.25,
                        "median_peak_memory_gb": 12.0,
                        "all_val_bpb": [1.25, 1.25, 1.25],
                    },
                ],
                "winner": "quality",
                "winner_delta_vs_baseline": -1.25,
            }
        )
    )
    payload, markdown = build_public_cuda_transfer_report(run_root)
    assert payload["tevo_seed_summary"] is not None
    assert "CUDA autoresearch benchmark" in markdown
    assert "https://github.com/karpathy/autoresearch.git" in markdown

    runner = CliRunner()
    out = run_root / "cuda_report.md"
    result = runner.invoke(app, ["cuda-transfer-report", str(run_root), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()
