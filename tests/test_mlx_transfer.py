from pathlib import Path

import ujson as json
from typer.testing import CliRunner

from transformer_evolution_llm.cli import app
from transformer_evolution_llm.dsl import ArchitectureSpec
from transformer_evolution_llm.mlx_transfer import (
    audit_tevo_regions,
    build_public_transfer_report,
    cost_conscious_modal_budget,
    export_public_transfer_recipes,
    render_public_transfer_variants,
    run_public_transfer_benchmarks,
    select_public_transfer_recipes,
    stage_public_transfer_continuation,
    stage_public_transfer_workspaces,
    summarize_continuation_results,
    write_winning_seed_diff,
)

MLX_SAMPLE = """
# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
ASPECT_RATIO = 64
HEAD_DIM = 128
WINDOW_PATTERN = "SSSL"

# v0.1: AdamW only. Muon port is future work.
TOTAL_BATCH_SIZE = 2**16
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
DEPTH = 4
DEVICE_BATCH_SIZE = 16
FINAL_EVAL_BATCH_SIZE = 256
STARTUP_EXCLUDE_STEPS = 1


def norm(x):
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-5)


def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2


def create_additive_causal_mask(seq_len, dtype=mx.float32):
    return seq_len


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def __call__(self, x):
        x = self.c_fc(x)
        x = mx.maximum(x, 0) ** 2
        return self.c_proj(x)


class Block(nn.Module):
    pass


class CausalSelfAttention(nn.Module):
    def __call__(self, q, k):
        q = norm(self.rope(q))
        k = norm(self.rope(k))
        return q + k


model_dim = ((DEPTH * ASPECT_RATIO + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
config = GPTConfig(
    sequence_len=MAX_SEQ_LEN,
    vocab_size=vocab_size,
    n_layer=DEPTH,
    n_head=model_dim // HEAD_DIM,
    n_kv_head=model_dim // HEAD_DIM,
    n_embd=model_dim,
    window_pattern=WINDOW_PATTERN,
)
model = GPT(config)
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


def _write_collapsed_frontier_with_history(tmp_path: Path) -> Path:
    frontier = _write_frontier(tmp_path)
    payload = json.loads(frontier.read_text())
    frontier.write_text(json.dumps([payload[0]]))
    (tmp_path / "frontier.state.json").write_text(json.dumps({"history": payload}))
    return frontier


def _write_fake_mlx_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "autoresearch-mlx"
    repo.mkdir()
    (repo / "train.py").write_text(MLX_SAMPLE)
    (repo / "prepare.py").write_text("def evaluate_bpb():\n    return 0.0\n")
    (repo / "pyproject.toml").write_text(
        "[project]\nname='fake-autoresearch-mlx'\nversion='0.0.0'\n"
    )
    return repo


def test_select_public_transfer_recipes_picks_distinct_labels(tmp_path: Path) -> None:
    frontier = _write_frontier(tmp_path)
    selected = select_public_transfer_recipes(frontier)
    assert [item.label for item in selected] == ["quality", "compute", "balanced"]
    assert [item.candidate_id for item in selected] == ["quality-a", "compute-a", "balanced-a"]
    assert len({item.recipe_key for item in selected}) == 3


def test_select_public_transfer_recipes_falls_back_to_history_when_frontier_collapses(
    tmp_path: Path,
) -> None:
    frontier = _write_collapsed_frontier_with_history(tmp_path)
    selected = select_public_transfer_recipes(frontier)
    assert [item.label for item in selected] == ["quality", "compute", "balanced"]
    assert [item.candidate_id for item in selected] == ["quality-a", "compute-a", "balanced-a"]


def test_cost_conscious_modal_budget_caps_first_pass() -> None:
    assert cost_conscious_modal_budget(8, 240, 8) == {
        "generations": 4,
        "steps": 120,
        "eval_batches": 4,
    }
    assert cost_conscious_modal_budget(2, 60, 2) == {
        "generations": 2,
        "steps": 60,
        "eval_batches": 2,
    }


def test_render_stage_and_benchmark_public_transfer_run(tmp_path: Path) -> None:
    frontier = _write_frontier(tmp_path)
    mlx_repo = _write_fake_mlx_repo(tmp_path)
    _, selection_manifest = export_public_transfer_recipes(frontier, tmp_path / "recipes")
    rendered, _ = render_public_transfer_variants(
        mlx_repo, selection_manifest, tmp_path / "rendered"
    )
    arm_manifest = stage_public_transfer_workspaces(
        mlx_repo=mlx_repo,
        rendered_variants=rendered,
        out_dir=tmp_path / "arms",
    )

    quality_train = Path(json.loads(arm_manifest.read_text())["arms"][1]["train_py"]).read_text()
    assert "TEVO TRAIN RECIPE: CONSTANTS START" in quality_train
    assert 'WINDOW_PATTERN = "SL"' in quality_train

    command = (
        'python3 -c "from pathlib import Path; '
        "arm = Path.cwd().parent.name; "
        "scores = {'baseline': 2.500000, 'quality': 1.250000, "
        "'compute': 1.500000, 'balanced': 1.350000}; "
        "print('---'); "
        "print(f'val_bpb:          {scores[arm]:.6f}'); "
        "print('peak_vram_mb:     1024.0')\""
    )
    _, _, summary_path, markdown_path = run_public_transfer_benchmarks(
        arm_manifest,
        out_dir=tmp_path / "mlx_results",
        repeat=3,
        timeout_seconds=10,
        command=command,
    )
    summary = json.loads(summary_path.read_text())
    assert summary["winner"] == "quality"
    assert "quality" in markdown_path.read_text()

    diff_path = write_winning_seed_diff(arm_manifest, summary_path, tmp_path / "winner.diff")
    assert diff_path.exists()
    assert "TEVO TRAIN RECIPE" in diff_path.read_text()

    continuation_manifest = stage_public_transfer_continuation(
        arm_manifest,
        summary_path,
        tmp_path / "continuation",
        repo_root=Path.cwd(),
    )
    continuation = json.loads(continuation_manifest.read_text())
    prompt_text = Path(continuation["program_path"]).read_text()
    assert "First 12 experiments" in prompt_text
    assert Path(continuation["results_tsv"]).read_text().startswith("commit\tval_bpb")


def test_audit_regions_and_build_public_report(tmp_path: Path) -> None:
    frontier = _write_frontier(tmp_path)
    mlx_repo = _write_fake_mlx_repo(tmp_path)
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "tevo_seed_summary.json").write_text(
        json.dumps({"metrics": {"ppl_code": 1.234, "speedrun_flops_to_target": 5678.0}})
    )

    _, selection_manifest = export_public_transfer_recipes(frontier, run_root / "recipes")
    rendered, _ = render_public_transfer_variants(
        mlx_repo, selection_manifest, run_root / "rendered"
    )
    seeded_text = Path(rendered["quality"]).read_text()

    no_change = audit_tevo_regions(seeded_text, seeded_text + "\n# outside tevo zone\n")
    assert no_change["changed_regions"] == []

    changed = audit_tevo_regions(seeded_text, seeded_text.replace("DEPTH = 2", "DEPTH = 3"))
    assert changed["changed_regions"] == ["CONSTANTS"]

    continuation_dir = run_root / "continuation"
    continuation_dir.mkdir()
    results_tsv = continuation_dir / "results.tsv"
    results_tsv.write_text(
        "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n"
        "abc1234\t1.300000\t1.0\tkeep\tseeded baseline\n"
        "def5678\t1.250000\t1.0\tkeep\tbetter schedule\n"
    )
    continuation_summary = summarize_continuation_results(results_tsv)
    assert continuation_summary["best_val_bpb"] == 1.25

    (run_root / "mlx_results").mkdir()
    (run_root / "mlx_results" / "benchmark_summary.json").write_text(
        json.dumps(
            {
                "arms": [
                    {
                        "arm": "baseline",
                        "runs": 3,
                        "successful_runs": 3,
                        "median_val_bpb": 2.5,
                        "best_val_bpb": 2.5,
                        "median_peak_memory_gb": 1.0,
                        "all_val_bpb": [2.5, 2.5, 2.5],
                    },
                    {
                        "arm": "quality",
                        "runs": 3,
                        "successful_runs": 3,
                        "median_val_bpb": 1.25,
                        "best_val_bpb": 1.25,
                        "median_peak_memory_gb": 1.0,
                        "all_val_bpb": [1.25, 1.25, 1.25],
                    },
                ],
                "winner": "quality",
                "winner_delta_vs_baseline": -1.25,
            }
        )
    )
    report_payload, report_markdown = build_public_transfer_report(run_root)
    assert report_payload["tevo_seed_summary"] is not None
    assert "TEVO seed on openwebtext_10m" in report_markdown
    assert "Seeded continuation trajectory" in report_markdown


def test_cli_audit_and_continuation_summary(tmp_path: Path) -> None:
    seed = tmp_path / "seed.py"
    current = tmp_path / "current.py"
    seed.write_text(
        "# === TEVO TRAIN RECIPE: CONSTANTS START ===\n"
        "DEPTH = 2\n"
        "# === TEVO TRAIN RECIPE: CONSTANTS END ===\n"
    )
    current.write_text(
        "# === TEVO TRAIN RECIPE: CONSTANTS START ===\n"
        "DEPTH = 3\n"
        "# === TEVO TRAIN RECIPE: CONSTANTS END ===\n"
    )
    runner = CliRunner()
    audit_result = runner.invoke(app, ["mlx-transfer-audit", str(seed), str(current)])
    assert audit_result.exit_code == 0, audit_result.output
    assert "CONSTANTS" in audit_result.output

    results_tsv = tmp_path / "results.tsv"
    results_tsv.write_text(
        "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n"
        "abc1234\t1.400000\t1.0\tkeep\tseeded baseline\n"
        "def5678\t1.350000\t1.0\tkeep\tbetter batch\n"
    )
    summary_out = tmp_path / "continuation.md"
    summary_result = runner.invoke(
        app,
        ["mlx-transfer-continuation-summary", str(results_tsv), "--out", str(summary_out)],
    )
    assert summary_result.exit_code == 0, summary_result.output
    assert summary_out.exists()
