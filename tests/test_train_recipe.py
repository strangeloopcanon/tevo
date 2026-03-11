from pathlib import Path

import ujson as json
from typer.testing import CliRunner

from transformer_evolution_llm.cli import app
from transformer_evolution_llm.dsl import ArchitectureSpec, LayerScaleConfig
from transformer_evolution_llm.train_recipe import (
    TrainRecipe,
    TrainRecipeCompatibilityError,
    TrainRecipeModel,
    TrainRecipeSource,
    TrainRecipeTarget,
    apply_train_recipe_to_source,
    shortlist_frontier_train_recipes,
    train_recipe_from_spec,
)

CUDA_SAMPLE = """
# Model architecture
ASPECT_RATIO = 64       # model_dim = depth * ASPECT_RATIO
HEAD_DIM = 128          # target head dimension for attention
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context

# Optimization
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
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
    \"\"\"Returns True if layer should have Value Embedding.\"\"\"
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
    \"\"\"Returns True if layer should have Value Embedding.\"\"\"
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


def _compatible_spec() -> ArchitectureSpec:
    return ArchitectureSpec(
        model={
            "name": "train-recipe-candidate",
            "emb": {"dim": 512, "vocab": 50257},
            "blocks": [
                {
                    "attn": {
                        "kind": "GQA",
                        "heads": 8,
                        "head_dim": 64,
                        "kv_groups": 2,
                        "sparsity": "sliding",
                        "sliding_window": 512,
                    },
                    "ffn": {"type": "dense", "hidden": 1536, "activation": "gelu"},
                },
                {
                    "attn": {
                        "kind": "GQA",
                        "heads": 8,
                        "head_dim": 64,
                        "kv_groups": 2,
                        "sparsity": "none",
                    },
                    "ffn": {"type": "dense", "hidden": 1536, "activation": "gelu"},
                },
            ],
            "head": {"tie_embeddings": True, "vocab": 50257},
            "norm": "layernorm",
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


def test_train_recipe_from_spec_exports_shared_fields() -> None:
    recipe = train_recipe_from_spec(_compatible_spec(), candidate_id="cand-1")
    assert recipe.name == "cand-1"
    assert recipe.model.depth == 2
    assert recipe.model.window_pattern == "SL"
    assert recipe.model.n_head == 8
    assert recipe.model.n_kv_head == 4
    assert recipe.model.norm_kind == "layernorm"
    assert recipe.model.mlp_hidden == 1536
    assert recipe.model.value_embedding_mode == "alternate"
    assert recipe.optimization.device_batch_size is None
    assert recipe.optimization.matrix_lr is None
    assert recipe.optimization.weight_decay is None
    assert recipe.source is not None
    assert recipe.source.notes


def test_train_recipe_from_spec_rejects_non_uniform_blocks() -> None:
    spec = _compatible_spec()
    block = spec.model.blocks[1]
    if block.ffn is None:
        raise AssertionError("expected dense ffn")
    block.ffn.hidden = 2048
    try:
        train_recipe_from_spec(spec)
    except TrainRecipeCompatibilityError as exc:
        assert "share the same dense FFN" in str(exc)
    else:
        raise AssertionError("expected TrainRecipeCompatibilityError")


def test_shortlist_frontier_train_recipes_skips_incompatible_entries(tmp_path: Path) -> None:
    compatible = _compatible_spec()
    incompatible = _compatible_spec()
    incompatible.model.blocks[0].extras = [LayerScaleConfig(init=1e-5, learnable=True)]
    frontier = tmp_path / "frontier.json"
    frontier.write_text(
        json.dumps(
            [
                {
                    "id": "bad-1",
                    "metrics": {"ppl_code": 1.0},
                    "spec": incompatible.model_dump(mode="python"),
                },
                {
                    "id": "good-1",
                    "metrics": {"ppl_code": 3.0},
                    "spec": compatible.model_dump(mode="python"),
                },
                {
                    "id": "good-0",
                    "metrics": {"ppl_code": 2.0},
                    "spec": compatible.model_dump(mode="python"),
                },
            ]
        )
    )
    recipes = shortlist_frontier_train_recipes(frontier, top_k=2, metric="ppl_code")
    assert [recipe.name for recipe in recipes] == ["good-0", "good-1"]


def test_apply_train_recipe_to_cuda_source_is_idempotent() -> None:
    recipe = train_recipe_from_spec(_compatible_spec(), candidate_id="cuda-cand")
    patched = apply_train_recipe_to_source(
        CUDA_SAMPLE,
        recipe,
        TrainRecipeTarget.AUTORESEARCH_CUDA,
    )
    assert "MODEL_DIM = 512" in patched
    assert 'MLP_ACTIVATION = "gelu"' in patched
    assert 'VALUE_EMBED_MODE = "alternate"' in patched
    assert "N_KV_HEAD = 4" in patched
    assert "DEVICE_BATCH_SIZE = 128" in patched
    assert "if USE_QK_NORM:" in patched
    assert "TEVO TRAIN RECIPE: MLP START" in patched
    assert (
        apply_train_recipe_to_source(
            patched,
            recipe,
            TrainRecipeTarget.AUTORESEARCH_CUDA,
        )
        == patched
    )


def test_apply_train_recipe_to_cuda_source_projects_oversized_recipe() -> None:
    recipe = TrainRecipe(
        name="oversized-cuda-cand",
        model=TrainRecipeModel(
            depth=20,
            model_dim=1280,
            head_dim=128,
            n_head=10,
            n_kv_head=10,
            norm_kind="rmsnorm",
            window_pattern="SSSLSSSLSSLLSSSLSSSL",
            mlp_hidden=5120,
            mlp_activation="relu",
            value_embedding_mode="alternate",
            use_qk_norm=False,
        ),
        source=TrainRecipeSource(candidate_id="oversized-cuda-cand"),
    )
    patched = apply_train_recipe_to_source(
        CUDA_SAMPLE,
        recipe,
        TrainRecipeTarget.AUTORESEARCH_CUDA,
    )
    assert "DEPTH = 8" in patched
    assert "MODEL_DIM = 512" in patched
    assert "N_HEAD = 4" in patched
    assert "N_KV_HEAD = 4" in patched
    assert 'WINDOW_PATTERN = "SSSSLSLL"' in patched
    assert "MLP_HIDDEN = 2048" in patched
    assert 'MLP_ACTIVATION = "relu"' in patched


def test_apply_train_recipe_to_mlx_source_is_idempotent() -> None:
    recipe = train_recipe_from_spec(_compatible_spec(), candidate_id="mlx-cand")
    patched = apply_train_recipe_to_source(
        MLX_SAMPLE,
        recipe,
        TrainRecipeTarget.AUTORESEARCH_MLX,
    )
    assert "MODEL_DIM = 512" in patched
    assert 'NORM_KIND = "layernorm"' in patched
    assert 'VALUE_EMBED_MODE = "alternate"' in patched
    assert "DEVICE_BATCH_SIZE = 16" in patched
    assert "model_dim = MODEL_DIM" in patched
    assert "TEVO TRAIN RECIPE: MODEL_CONFIG START" in patched
    assert (
        apply_train_recipe_to_source(
            patched,
            recipe,
            TrainRecipeTarget.AUTORESEARCH_MLX,
        )
        == patched
    )


def test_cli_train_recipe_export_and_render(tmp_path: Path) -> None:
    spec = _compatible_spec()
    frontier = tmp_path / "frontier.json"
    frontier.write_text(
        json.dumps(
            [
                {
                    "id": "good-1",
                    "metrics": {"ppl_code": 2.5},
                    "spec": spec.model_dump(mode="python"),
                }
            ]
        )
    )
    recipe_path = tmp_path / "candidate.train_recipe.yaml"
    runner = CliRunner()
    export_result = runner.invoke(
        app,
        [
            "train-recipe-export",
            str(frontier),
            "--candidate-id",
            "good-1",
            "--out",
            str(recipe_path),
        ],
    )
    assert export_result.exit_code == 0, export_result.output
    assert recipe_path.exists()

    train_py = tmp_path / "train.py"
    train_py.write_text(CUDA_SAMPLE)
    render_result = runner.invoke(
        app,
        [
            "train-recipe-render",
            str(recipe_path),
            "--backend",
            "autoresearch_cuda",
            "--train-py",
            str(train_py),
        ],
    )
    assert render_result.exit_code == 0, render_result.output
    assert "TEVO TRAIN RECIPE: CONSTANTS START" in train_py.read_text()
