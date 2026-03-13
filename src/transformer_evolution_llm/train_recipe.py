"""TrainRecipe bridge for exporting TEVO candidates into autoresearch targets."""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import ujson as json
import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator

from .dsl import ArchitectureSpec, DenseFFNConfig, load_architecture_spec


class TrainRecipeError(ValueError):
    """Base error for train recipe workflows."""


class TrainRecipeCompatibilityError(TrainRecipeError):
    """Raised when a TEVO spec cannot be expressed by the shared recipe surface."""


class TrainRecipeRenderError(TrainRecipeError):
    """Raised when a target train.py file cannot be rendered safely."""


class TrainRecipeTarget(StrEnum):
    """Supported downstream train.py targets."""

    AUTORESEARCH_CUDA = "autoresearch_cuda"
    AUTORESEARCH_AT_HOME_CUDA = "autoresearch_at_home_cuda"
    AUTORESEARCH_MLX = "autoresearch_mlx"


class TrainRecipeModel(BaseModel):
    """Backend-neutral model knobs that are safe to render into train.py templates."""

    depth: int = Field(gt=0)
    model_dim: int = Field(gt=0)
    head_dim: int = Field(gt=0)
    n_head: int = Field(gt=0)
    n_kv_head: int = Field(gt=0)
    norm_kind: Literal["layernorm", "rmsnorm"] = "rmsnorm"
    window_pattern: str = Field(min_length=1)
    mlp_hidden: int = Field(gt=0)
    mlp_activation: Literal["gelu", "relu", "relu_squared", "silu", "swiglu"] = "relu_squared"
    value_embedding_mode: Literal["off", "alternate", "all"] = "alternate"
    use_qk_norm: bool = False

    @model_validator(mode="after")
    def validate_geometry(self) -> TrainRecipeModel:
        if self.model_dim != self.n_head * self.head_dim:
            raise ValueError("model_dim must equal n_head * head_dim")
        if self.n_kv_head > self.n_head:
            raise ValueError("n_kv_head must be <= n_head")
        if self.n_head % self.n_kv_head != 0:
            raise ValueError("n_head must be divisible by n_kv_head")
        pattern = str(self.window_pattern or "").upper()
        if len(pattern) != self.depth:
            raise ValueError("window_pattern must contain exactly one entry per layer")
        if any(ch not in {"S", "L"} for ch in pattern):
            raise ValueError("window_pattern may only contain 'S' and 'L'")
        self.window_pattern = pattern
        return self

    @property
    def kv_groups(self) -> int:
        """Return the TEVO-style kv_groups value for this recipe."""
        return max(1, self.n_head // self.n_kv_head)

    @property
    def mlp_expansion(self) -> float:
        """Return hidden/model_dim for reporting."""
        return float(self.mlp_hidden) / float(self.model_dim)


class TrainRecipeOptimization(BaseModel):
    """Optimization knobs. Missing values fall back to backend defaults at render time."""

    total_batch_size: int | None = Field(default=None, gt=0)
    device_batch_size: int | None = Field(default=None, gt=0)
    final_eval_batch_size: int | None = Field(default=None, gt=0)
    startup_exclude_steps: int | None = Field(default=None, ge=0)
    embedding_lr: float | None = Field(default=None, gt=0.0)
    unembedding_lr: float | None = Field(default=None, gt=0.0)
    matrix_lr: float | None = Field(default=None, gt=0.0)
    scalar_lr: float | None = Field(default=None, gt=0.0)
    weight_decay: float | None = Field(default=None, ge=0.0)
    adam_betas: tuple[float, float] | None = None
    warmup_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    warmdown_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    final_lr_frac: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_betas(self) -> TrainRecipeOptimization:
        if self.adam_betas is None:
            return self
        beta1, beta2 = self.adam_betas
        if not (0.0 < beta1 < 1.0 and 0.0 < beta2 < 1.0):
            raise ValueError("adam_betas must be inside (0, 1)")
        return self


class TrainRecipeSource(BaseModel):
    """Where a recipe came from and what metrics were seen upstream."""

    candidate_id: str | None = None
    frontier_path: str | None = None
    spec_path: str | None = None
    model_name: str | None = None
    seq_len: int | None = Field(default=None, gt=0)
    tokenizer: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class TrainRecipe(BaseModel):
    """Top-level recipe artifact exported from TEVO."""

    name: str
    family: Literal["shared_train_py_v1"] = "shared_train_py_v1"
    model: TrainRecipeModel
    optimization: TrainRecipeOptimization = Field(default_factory=TrainRecipeOptimization)
    source: TrainRecipeSource | None = None


_BACKEND_DEFAULTS: dict[TrainRecipeTarget, dict[str, Any]] = {
    TrainRecipeTarget.AUTORESEARCH_CUDA: {
        "total_batch_size": 2**19,
        "device_batch_size": 128,
        "embedding_lr": 0.6,
        "unembedding_lr": 0.004,
        "matrix_lr": 0.04,
        "scalar_lr": 0.5,
        "weight_decay": 0.2,
        "adam_betas": (0.8, 0.95),
        "warmup_ratio": 0.0,
        "warmdown_ratio": 0.5,
        "final_lr_frac": 0.0,
    },
    TrainRecipeTarget.AUTORESEARCH_AT_HOME_CUDA: {
        "total_batch_size": 2**19,
        "device_batch_size": 128,
        "embedding_lr": 0.6,
        "unembedding_lr": 0.004,
        "matrix_lr": 0.04,
        "scalar_lr": 0.5,
        "weight_decay": 0.2,
        "adam_betas": (0.8, 0.95),
        "warmup_ratio": 0.0,
        "warmdown_ratio": 0.5,
        "final_lr_frac": 0.0,
    },
    TrainRecipeTarget.AUTORESEARCH_MLX: {
        "total_batch_size": 2**16,
        "device_batch_size": 16,
        "final_eval_batch_size": 256,
        "startup_exclude_steps": 1,
        "embedding_lr": 0.6,
        "unembedding_lr": 0.004,
        "matrix_lr": 0.04,
        "scalar_lr": 0.5,
        "weight_decay": 0.2,
        "adam_betas": (0.8, 0.95),
        "warmup_ratio": 0.0,
        "warmdown_ratio": 0.5,
        "final_lr_frac": 0.0,
    },
}

_AUTORESEARCH_CUDA_SAFE_LIMITS: dict[str, int] = {
    "max_depth": 8,
    "max_model_dim": 512,
    "max_mlp_hidden": 2048,
}

_MARKER_FMT = "# === TEVO TRAIN RECIPE: {name} {edge} ==="

_FALLBACK_PATTERNS: dict[TrainRecipeTarget, dict[str, re.Pattern[str]]] = {
    TrainRecipeTarget.AUTORESEARCH_CUDA: {
        "CONSTANTS": re.compile(r"(?ms)^# Model architecture\n.*?^DEVICE_BATCH_SIZE = [^\n]*\n"),
        "NORM": re.compile(r"(?ms)^def norm\(x\):\n.*?(?=^def has_ve\()"),
        "VALUE_EMBED": re.compile(
            r"(?ms)^def has_ve\(layer_idx, n_layer\):\n.*?(?=^def apply_rotary_emb\()"
        ),
        "MLP": re.compile(r"(?ms)^class MLP\(nn\.Module\):\n.*?(?=^class Block\(nn\.Module\):)"),
        "QK_NORM": re.compile(
            r"(?ms)^        q, k = apply_rotary_emb\(q, cos, sin\), "
            r"apply_rotary_emb\(k, cos, sin\)\n"
            r"        q, k = norm\(q\), norm\(k\)\n"
        ),
        "MODEL_CONFIG": re.compile(
            r"(?ms)^def build_model_config\(depth\):\n.*?"
            r"(?=^config = build_model_config\(DEPTH\)\n)"
        ),
    },
    TrainRecipeTarget.AUTORESEARCH_AT_HOME_CUDA: {
        "CONSTANTS": re.compile(r"(?ms)^# Model architecture\n.*?^DEVICE_BATCH_SIZE = [^\n]*\n"),
        "NORM": re.compile(r"(?ms)^def norm\(x\):\n.*?(?=^def has_ve\()"),
        "VALUE_EMBED": re.compile(
            r"(?ms)^def has_ve\(layer_idx, n_layer\):\n.*?(?=^def apply_rotary_emb\()"
        ),
        "MLP": re.compile(r"(?ms)^class MLP\(nn\.Module\):\n.*?(?=^class Block\(nn\.Module\):)"),
        "QK_NORM": re.compile(
            r"(?ms)^        q, k = apply_rotary_emb\(q, cos, sin\), "
            r"apply_rotary_emb\(k, cos, sin\)\n"
            r"        q, k = norm\(q\), norm\(k\)\n"
        ),
        "MODEL_CONFIG": re.compile(
            r"(?ms)^def build_model_config\(depth\):\n.*?"
            r"(?=^config = build_model_config\(DEPTH\)\n)"
        ),
    },
    TrainRecipeTarget.AUTORESEARCH_MLX: {
        "CONSTANTS": re.compile(
            r"(?ms)^# Model architecture\n.*?^STARTUP_EXCLUDE_STEPS = [^\n]*\n"
        ),
        "NORM": re.compile(r"(?ms)^def norm\(x\):\n.*?(?=^def has_ve\()"),
        "VALUE_EMBED": re.compile(
            r"(?ms)^def has_ve\(layer_idx, n_layer\):\n.*?(?=^def create_additive_causal_mask\()"
        ),
        "MLP": re.compile(r"(?ms)^class MLP\(nn\.Module\):\n.*?(?=^class Block\(nn\.Module\):)"),
        "QK_NORM": re.compile(
            r"(?ms)^        q = norm\(self\.rope\(q\)\)\n        k = norm\(self\.rope\(k\)\)\n"
        ),
        "MODEL_CONFIG": re.compile(
            r"(?ms)^model_dim = .*?\nconfig = GPTConfig\(\n.*?^\)\n(?=^model = GPT\(config\)\n)"
        ),
    },
}


def load_train_recipe(path: str | Path) -> TrainRecipe:
    """Load a recipe from YAML or JSON."""
    path = Path(path)
    text = path.read_text()
    if path.suffix in {".yaml", ".yml"}:
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    try:
        return TrainRecipe(**payload)
    except ValidationError as exc:
        raise TrainRecipeError(f"Invalid train recipe at {path}: {exc}") from exc


def save_train_recipe(recipe: TrainRecipe, path: str | Path) -> None:
    """Persist a recipe as YAML or JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = recipe.model_dump(mode="python")
    if path.suffix in {".json"}:
        path.write_text(json.dumps(payload, indent=2))
        return
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def train_recipe_from_spec(
    spec: ArchitectureSpec,
    *,
    candidate_id: str | None = None,
    frontier_path: str | Path | None = None,
    spec_path: str | Path | None = None,
    metrics: dict[str, float] | None = None,
) -> TrainRecipe:
    """Convert a compatible TEVO spec into a backend-neutral train recipe."""
    _validate_recipe_compatible_spec(spec)
    block0 = spec.model.blocks[0]
    if block0.attn is None or not isinstance(block0.ffn, DenseFFNConfig):
        raise TrainRecipeCompatibilityError("Expected a dense attention+ffn block")
    n_head = int(block0.attn.heads)
    n_kv_head = max(1, n_head // max(1, int(block0.attn.kv_groups or 1)))
    source = TrainRecipeSource(
        candidate_id=candidate_id,
        frontier_path=str(frontier_path) if frontier_path is not None else None,
        spec_path=str(spec_path) if spec_path is not None else None,
        model_name=spec.model.name,
        seq_len=int(spec.data.seq_len),
        tokenizer=str(spec.data.tokenizer),
        metrics=dict(metrics or {}),
        notes=[],
    )
    if int(spec.data.seq_len) != 2048:
        source.notes.append(
            "TEVO seq_len differs from downstream prepare.py defaults; "
            "compare relative deltas only."
        )
    source.notes.append(
        "Downstream optimizer and batch-size defaults are preserved unless the recipe "
        "overrides them explicitly."
    )
    return TrainRecipe(
        name=candidate_id or spec.model.name,
        model=TrainRecipeModel(
            depth=spec.model.n_layers,
            model_dim=int(spec.model.emb.dim),
            head_dim=int(block0.attn.head_dim),
            n_head=n_head,
            n_kv_head=n_kv_head,
            norm_kind=str(spec.model.norm or "layernorm"),
            window_pattern=_derive_window_pattern(spec),
            mlp_hidden=int(block0.ffn.hidden),
            mlp_activation=str(block0.ffn.activation or "relu_squared"),
            # Preserve downstream train.py's existing value-embedding recipe unless
            # a future TEVO search lane models it explicitly.
            value_embedding_mode="alternate",
            use_qk_norm=_derive_use_qk_norm(spec),
        ),
        optimization=TrainRecipeOptimization(),
        source=source,
    )


def shortlist_frontier_train_recipes(
    frontier_path: str | Path,
    *,
    top_k: int = 3,
    metric: str = "ppl_code",
    candidate_id: str | None = None,
) -> list[TrainRecipe]:
    """Load a frontier file and export the top compatible train recipes."""
    frontier_path = Path(frontier_path)
    entries = _load_frontier_entries(frontier_path)
    compatible: list[tuple[float, int, TrainRecipe]] = []
    for idx, entry in enumerate(entries):
        entry_id = str(entry.get("id") or "")
        if candidate_id is not None and entry_id != candidate_id:
            continue
        spec_payload = entry.get("spec")
        if not isinstance(spec_payload, dict):
            continue
        try:
            spec = ArchitectureSpec(**spec_payload)
            recipe = train_recipe_from_spec(
                spec,
                candidate_id=entry_id or None,
                frontier_path=frontier_path,
                metrics=_numeric_metrics(entry.get("metrics")),
            )
        except (ValidationError, TrainRecipeCompatibilityError):
            continue
        metric_value = (
            float(recipe.source.metrics.get(metric, float("inf")))
            if recipe.source
            else float("inf")
        )
        compatible.append((metric_value, idx, recipe))
    compatible.sort(key=lambda item: (item[0], item[1], item[2].name))
    return [item[2] for item in compatible[: max(1, int(top_k))]]


def export_train_recipes(
    source_path: str | Path,
    *,
    top_k: int = 1,
    metric: str = "ppl_code",
    candidate_id: str | None = None,
) -> list[TrainRecipe]:
    """Export recipes from either a frontier JSON or a single TEVO spec path."""
    source_path = Path(source_path)
    if _looks_like_frontier(source_path):
        return shortlist_frontier_train_recipes(
            source_path, top_k=top_k, metric=metric, candidate_id=candidate_id
        )
    spec = load_architecture_spec(source_path)
    return [train_recipe_from_spec(spec, spec_path=source_path)]


def render_train_recipe_fragment(recipe: TrainRecipe, target: TrainRecipeTarget) -> str:
    """Render the stable TEVO-owned zones for a downstream backend."""
    prepared = _prepare_train_recipe_for_target(recipe, target)
    regions = _render_regions(prepared, target)
    order = ["CONSTANTS", "NORM", "VALUE_EMBED", "MLP", "QK_NORM", "MODEL_CONFIG"]
    return "\n\n".join(regions[name] for name in order)


def apply_train_recipe_to_source(
    text: str,
    recipe: TrainRecipe,
    target: TrainRecipeTarget,
) -> str:
    """Patch a current autoresearch train.py text with TEVO-owned recipe zones."""
    patterns = _FALLBACK_PATTERNS[target]
    prepared = _prepare_train_recipe_for_target(recipe, target)
    rendered = _render_regions(prepared, target)
    updated = text
    for name in ("CONSTANTS", "NORM", "VALUE_EMBED", "MLP", "QK_NORM", "MODEL_CONFIG"):
        updated = _replace_region(updated, name, rendered[name], patterns[name])
    return updated


def render_train_recipe_to_path(
    recipe: TrainRecipe,
    *,
    target: TrainRecipeTarget,
    train_py_path: str | Path,
    out_path: str | Path | None = None,
) -> Path:
    """Patch a downstream train.py file in place (or to a separate output path)."""
    train_py_path = Path(train_py_path)
    output_path = Path(out_path) if out_path is not None else train_py_path
    patched = apply_train_recipe_to_source(train_py_path.read_text(), recipe, target)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(patched)
    return output_path


def train_recipe_projection_applied(recipe: TrainRecipe, target: TrainRecipeTarget) -> bool:
    """Return whether a target-specific projection will modify the recipe model."""
    return _is_cuda_train_recipe_target(target) and _needs_autoresearch_cuda_projection(
        recipe.model
    )


def _prepare_train_recipe_for_target(recipe: TrainRecipe, target: TrainRecipeTarget) -> TrainRecipe:
    if not _is_cuda_train_recipe_target(target):
        return recipe
    if not _needs_autoresearch_cuda_projection(recipe.model):
        return recipe
    return _project_recipe_for_autoresearch_cuda(recipe)


def _is_cuda_train_recipe_target(target: TrainRecipeTarget) -> bool:
    return target in {
        TrainRecipeTarget.AUTORESEARCH_CUDA,
        TrainRecipeTarget.AUTORESEARCH_AT_HOME_CUDA,
    }


def _looks_like_frontier(path: Path) -> bool:
    if path.suffix != ".json":
        return False
    try:
        payload = json.loads(path.read_text())
    except ValueError:
        return False
    return isinstance(payload, list)


def _load_frontier_entries(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise TrainRecipeError(f"Frontier JSON must be a list: {path}")
    return [entry for entry in payload if isinstance(entry, dict)]


def _numeric_metrics(payload: Any) -> dict[str, float]:
    if not isinstance(payload, dict):
        return {}
    numeric: dict[str, float] = {}
    for key, value in payload.items():
        try:
            numeric[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return numeric


def _validate_recipe_compatible_spec(spec: ArchitectureSpec) -> None:
    if spec.model.hyper is not None:
        raise TrainRecipeCompatibilityError(
            "hyper-connections are outside the train.py bridge surface"
        )
    if spec.model.kv_policy is not None:
        raise TrainRecipeCompatibilityError(
            "kv_policy is not expressed by downstream train.py templates"
        )
    if spec.model.recurrences:
        raise TrainRecipeCompatibilityError("recurrences are outside the train.py bridge surface")
    blocks = spec.model.blocks
    if not blocks:
        raise TrainRecipeCompatibilityError("spec requires at least one block")
    attn_signature: tuple[int, int, int, bool] | None = None
    ffn_signature: tuple[int, str] | None = None
    for idx, block in enumerate(blocks):
        if block.ssm is not None:
            raise TrainRecipeCompatibilityError(
                f"block {idx} includes SSM, which is not bridge-safe"
            )
        if block.extras:
            raise TrainRecipeCompatibilityError(
                f"block {idx} includes extras, which are not bridge-safe"
            )
        if block.ffn_memory is not None:
            raise TrainRecipeCompatibilityError(
                f"block {idx} uses ffn_memory, which current downstream templates do not own"
            )
        if block.attn is None:
            raise TrainRecipeCompatibilityError(f"block {idx} is missing attention")
        if block.attn.kind not in {"MHA", "GQA", "MQA"}:
            raise TrainRecipeCompatibilityError(
                f"block {idx} attention kind {block.attn.kind!r} is not supported by the bridge"
            )
        if block.attn.selector != "none" or block.attn.gating_pos != "none":
            raise TrainRecipeCompatibilityError(
                f"block {idx} uses selector/gating features outside the bridge surface"
            )
        if block.attn.alibi or block.attn.value_glu:
            raise TrainRecipeCompatibilityError(
                f"block {idx} uses ALiBi or value_glu, which current train.py templates do not own"
            )
        if block.attn.kv_latent_dim is not None:
            raise TrainRecipeCompatibilityError(f"block {idx} uses MLA-specific latent KV state")
        if block.ffn is None or not isinstance(block.ffn, DenseFFNConfig):
            raise TrainRecipeCompatibilityError(f"block {idx} must use a dense FFN for the bridge")
        if str(block.ffn.input_source or "residual") != "residual":
            raise TrainRecipeCompatibilityError(
                f"block {idx} reads FFN inputs from embeddings, "
                "which current bridge templates exclude"
            )
        signature = (
            int(block.attn.heads),
            int(block.attn.head_dim),
            int(block.attn.kv_groups or 1),
            _attn_uses_qk_norm(block),
        )
        if attn_signature is None:
            attn_signature = signature
        elif signature != attn_signature:
            raise TrainRecipeCompatibilityError(
                "all blocks must share the same attention geometry and qk-norm setting"
            )
        dense_signature = (int(block.ffn.hidden), str(block.ffn.activation or "relu_squared"))
        if ffn_signature is None:
            ffn_signature = dense_signature
        elif dense_signature != ffn_signature:
            raise TrainRecipeCompatibilityError(
                "all blocks must share the same dense FFN hidden size and activation"
            )
    if attn_signature is None:
        raise TrainRecipeCompatibilityError("could not derive attention geometry")
    if spec.model.emb.dim != attn_signature[0] * attn_signature[1]:
        raise TrainRecipeCompatibilityError("embedding dim must match heads * head_dim")
    _derive_window_pattern(spec)


def _attn_uses_qk_norm(block: Any) -> bool:
    attn = getattr(block, "attn", None)
    if attn is None:
        return False
    if getattr(attn, "qk_norm_max", None) is not None:
        return True
    softmax = getattr(attn, "softmax", None)
    if softmax is None:
        return False
    return str(getattr(softmax, "qk_norm", "none") or "none") != "none"


def _derive_use_qk_norm(spec: ArchitectureSpec) -> bool:
    return _attn_uses_qk_norm(spec.model.blocks[0])


def _derive_window_pattern(spec: ArchitectureSpec) -> str:
    chars: list[str] = []
    half_window = max(1, int(spec.data.seq_len) // 2)
    for idx, block in enumerate(spec.model.blocks):
        attn = block.attn
        if attn is None:
            raise TrainRecipeCompatibilityError(f"block {idx} is missing attention")
        sparsity = str(attn.sparsity or "none")
        if sparsity == "none":
            chars.append("L")
            continue
        if sparsity == "sliding" and int(attn.sw or 0) == half_window:
            chars.append("S")
            continue
        raise TrainRecipeCompatibilityError(
            f"block {idx} uses sparsity={sparsity!r} with sw={attn.sw!r}, which "
            "cannot be rendered as the shared S/L window pattern"
        )
    if chars and chars[-1] != "L":
        raise TrainRecipeCompatibilityError(
            "the final layer must be full-context (L) for downstream templates"
        )
    return "".join(chars)


def _resolve_optimization(recipe: TrainRecipe, target: TrainRecipeTarget) -> dict[str, Any]:
    resolved = dict(_BACKEND_DEFAULTS[target])
    overrides = recipe.optimization.model_dump(mode="python")
    for key, value in overrides.items():
        if value is not None:
            resolved[key] = value
    return resolved


def _needs_autoresearch_cuda_projection(model: TrainRecipeModel) -> bool:
    return (
        model.depth > _AUTORESEARCH_CUDA_SAFE_LIMITS["max_depth"]
        or model.model_dim > _AUTORESEARCH_CUDA_SAFE_LIMITS["max_model_dim"]
        or model.mlp_hidden > _AUTORESEARCH_CUDA_SAFE_LIMITS["max_mlp_hidden"]
    )


def _project_recipe_for_autoresearch_cuda(recipe: TrainRecipe) -> TrainRecipe:
    target_depth = min(recipe.model.depth, _AUTORESEARCH_CUDA_SAFE_LIMITS["max_depth"])
    target_model_dim = min(recipe.model.model_dim, _AUTORESEARCH_CUDA_SAFE_LIMITS["max_model_dim"])
    target_head_dim = _project_head_dim(recipe.model, target_model_dim)
    target_n_head = max(1, target_model_dim // target_head_dim)
    target_n_kv_head = _project_n_kv_head(recipe.model, target_n_head)
    target_mlp_hidden = _project_mlp_hidden(recipe.model, target_model_dim)
    target_window_pattern = _project_window_pattern(recipe.model.window_pattern, target_depth)
    source = (
        recipe.source.model_copy(deep=True) if recipe.source is not None else TrainRecipeSource()
    )
    source.notes.append(
        "Projection: rendered into the autoresearch_cuda safe envelope while preserving "
        "shared motifs (depth/window/activation/kv layout) for downstream validation."
    )
    return recipe.model_copy(
        update={
            "model": recipe.model.model_copy(
                update={
                    "depth": target_depth,
                    "model_dim": target_model_dim,
                    "head_dim": target_head_dim,
                    "n_head": target_n_head,
                    "n_kv_head": target_n_kv_head,
                    "window_pattern": target_window_pattern,
                    "mlp_hidden": target_mlp_hidden,
                }
            ),
            "source": source,
        }
    )


def _project_head_dim(model: TrainRecipeModel, target_model_dim: int) -> int:
    if target_model_dim % model.head_dim == 0:
        return model.head_dim
    safe_head_dim = 128
    if target_model_dim % safe_head_dim == 0:
        return safe_head_dim
    for candidate in (96, 64, 32, 16, 8, 4, 2, 1):
        if target_model_dim % candidate == 0:
            return candidate
    return target_model_dim


def _project_n_kv_head(model: TrainRecipeModel, target_n_head: int) -> int:
    kv_groups = max(1, model.kv_groups)
    desired = max(1, target_n_head // kv_groups)
    divisors = [value for value in range(1, target_n_head + 1) if target_n_head % value == 0]
    return min(divisors, key=lambda value: (abs(value - desired), -value))


def _project_mlp_hidden(model: TrainRecipeModel, target_model_dim: int) -> int:
    capped_expansion = min(max(model.mlp_expansion, 1.0), 4.0)
    raw_hidden = max(target_model_dim, int(round(target_model_dim * capped_expansion)))
    rounded = _round_to_multiple(raw_hidden, 256)
    return min(rounded, _AUTORESEARCH_CUDA_SAFE_LIMITS["max_mlp_hidden"])


def _project_window_pattern(pattern: str, target_depth: int) -> str:
    source = str(pattern or "").upper()
    if len(source) == target_depth:
        return source
    if len(source) < target_depth:
        repeats = (target_depth + len(source) - 1) // len(source)
        expanded = (source * repeats)[:target_depth]
        return expanded[:-1] + "L"

    src_len = len(source)
    projected_chars: list[str] = []
    for idx in range(target_depth):
        start = idx * src_len // target_depth
        end = (idx + 1) * src_len // target_depth
        segment = source[start:end]
        if not segment:
            segment = source[min(src_len - 1, start)]
        s_count = segment.count("S")
        l_count = segment.count("L")
        if l_count > s_count:
            projected_chars.append("L")
        elif s_count > l_count:
            projected_chars.append("S")
        else:
            projected_chars.append("L" if "L" in segment else segment[-1])
    projected_chars[-1] = "L"
    return "".join(projected_chars)


def _round_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 1:
        return max(1, int(value))
    return max(multiple, ((int(value) + multiple - 1) // multiple) * multiple)


def _marker(name: str, edge: Literal["START", "END"]) -> str:
    return _MARKER_FMT.format(name=name, edge=edge)


def _wrap_region(name: str, body: str) -> str:
    return f"{_marker(name, 'START')}\n{body.rstrip()}\n{_marker(name, 'END')}"


def _replace_region(text: str, name: str, replacement: str, fallback: re.Pattern[str]) -> str:
    start_marker = _marker(name, "START")
    end_marker = _marker(name, "END")
    if start_marker in text and end_marker in text:
        marker_pattern = re.compile(
            re.escape(start_marker) + r".*?" + re.escape(end_marker),
            flags=re.DOTALL,
        )
        updated, count = marker_pattern.subn(replacement, text, count=1)
        if count == 1:
            return updated
    updated, count = fallback.subn(replacement + "\n", text, count=1)
    if count != 1:
        raise TrainRecipeRenderError(
            f"Could not find the {name.lower()} region in the target train.py for safe rendering."
        )
    return updated


def _render_regions(recipe: TrainRecipe, target: TrainRecipeTarget) -> dict[str, str]:
    resolved = _resolve_optimization(recipe, target)
    return {
        "CONSTANTS": _wrap_region("CONSTANTS", _render_constants(recipe, target, resolved)),
        "NORM": _wrap_region("NORM", _render_norm(target)),
        "VALUE_EMBED": _wrap_region("VALUE_EMBED", _render_has_value_embedding()),
        "MLP": _wrap_region("MLP", _render_mlp(target)),
        "QK_NORM": _wrap_region("QK_NORM", _render_qk_norm(target)),
        "MODEL_CONFIG": _wrap_region("MODEL_CONFIG", _render_model_config(target)),
    }


def _render_constants(
    recipe: TrainRecipe,
    target: TrainRecipeTarget,
    resolved: dict[str, Any],
) -> str:
    source_id = recipe.source.candidate_id if recipe.source is not None else None
    lines = [
        f"# Generated by TEVO TrainRecipe bridge for {target.value}",
        f"# Recipe: {recipe.name}",
    ]
    if source_id:
        lines.append(f"# Source candidate: {source_id}")
    lines.extend(
        [
            "# Model architecture",
            f"DEPTH = {recipe.model.depth}",
            f"MODEL_DIM = {recipe.model.model_dim}",
            f"N_HEAD = {recipe.model.n_head}",
            f"N_KV_HEAD = {recipe.model.n_kv_head}",
            f"HEAD_DIM = {recipe.model.head_dim}",
            f'WINDOW_PATTERN = "{recipe.model.window_pattern}"',
            f'NORM_KIND = "{recipe.model.norm_kind}"',
            f"MLP_HIDDEN = {recipe.model.mlp_hidden}",
            f'MLP_ACTIVATION = "{recipe.model.mlp_activation}"',
            f'VALUE_EMBED_MODE = "{recipe.model.value_embedding_mode}"',
            f"USE_QK_NORM = {recipe.model.use_qk_norm}",
            "",
            "# Optimization",
            f"TOTAL_BATCH_SIZE = {resolved['total_batch_size']}",
            f"EMBEDDING_LR = {resolved['embedding_lr']}",
            f"UNEMBEDDING_LR = {resolved['unembedding_lr']}",
            f"MATRIX_LR = {resolved['matrix_lr']}",
            f"SCALAR_LR = {resolved['scalar_lr']}",
            f"WEIGHT_DECAY = {resolved['weight_decay']}",
            f"ADAM_BETAS = ({resolved['adam_betas'][0]}, {resolved['adam_betas'][1]})",
            f"WARMUP_RATIO = {resolved['warmup_ratio']}",
            f"WARMDOWN_RATIO = {resolved['warmdown_ratio']}",
            f"FINAL_LR_FRAC = {resolved['final_lr_frac']}",
            "",
            "# Model size",
            f"DEVICE_BATCH_SIZE = {resolved['device_batch_size']}",
        ]
    )
    if target == TrainRecipeTarget.AUTORESEARCH_MLX:
        lines.extend(
            [
                f"FINAL_EVAL_BATCH_SIZE = {resolved['final_eval_batch_size']}",
                f"STARTUP_EXCLUDE_STEPS = {resolved['startup_exclude_steps']}",
            ]
        )
    return "\n".join(lines)


def _render_norm(target: TrainRecipeTarget) -> str:
    if _is_cuda_train_recipe_target(target):
        return """def norm(x):
    if NORM_KIND == "layernorm":
        return F.layer_norm(x, (x.size(-1),))
    return F.rms_norm(x, (x.size(-1),))
"""
    return """def norm(x):
    if NORM_KIND == "layernorm":
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.mean((x - mean) * (x - mean), axis=-1, keepdims=True)
        return (x - mean) * mx.rsqrt(var + 1e-5)
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-5)
"""


def _render_has_value_embedding() -> str:
    return """def has_ve(layer_idx, n_layer):
    if VALUE_EMBED_MODE == "off":
        return False
    if VALUE_EMBED_MODE == "all":
        return True
    return layer_idx % 2 == (n_layer - 1) % 2
"""


def _render_mlp(target: TrainRecipeTarget) -> str:
    if _is_cuda_train_recipe_target(target):
        return """def apply_mlp_activation(x):
    if MLP_ACTIVATION == "gelu":
        return F.gelu(x)
    if MLP_ACTIVATION == "relu":
        return F.relu(x)
    if MLP_ACTIVATION == "relu_squared":
        return F.relu(x).square()
    if MLP_ACTIVATION == "silu":
        return F.silu(x)
    if MLP_ACTIVATION == "swiglu":
        value, gate = x.chunk(2, dim=-1)
        return value * F.silu(gate)
    raise ValueError(f"Unsupported MLP_ACTIVATION: {MLP_ACTIVATION}")


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        inner = MLP_HIDDEN * 2 if MLP_ACTIVATION == "swiglu" else MLP_HIDDEN
        self.c_fc = nn.Linear(config.n_embd, inner, bias=False)
        self.c_proj = nn.Linear(MLP_HIDDEN, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = apply_mlp_activation(x)
        x = self.c_proj(x)
        return x
"""
    return """def apply_mlp_activation(x):
    if MLP_ACTIVATION == "gelu":
        return 0.5 * x * (1.0 + mx.erf(x / math.sqrt(2.0)))
    if MLP_ACTIVATION == "relu":
        return mx.maximum(x, 0)
    if MLP_ACTIVATION == "relu_squared":
        return mx.maximum(x, 0) ** 2
    if MLP_ACTIVATION == "silu":
        return x * mx.sigmoid(x)
    if MLP_ACTIVATION == "swiglu":
        value, gate = mx.split(x, 2, axis=-1)
        return value * mx.sigmoid(gate)
    raise ValueError(f"Unsupported MLP_ACTIVATION: {MLP_ACTIVATION}")


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        inner = MLP_HIDDEN * 2 if MLP_ACTIVATION == "swiglu" else MLP_HIDDEN
        self.c_fc = nn.Linear(config.n_embd, inner, bias=False)
        self.c_proj = nn.Linear(MLP_HIDDEN, config.n_embd, bias=False)

    def __call__(self, x):
        x = self.c_fc(x)
        x = apply_mlp_activation(x)
        return self.c_proj(x)
"""


def _render_qk_norm(target: TrainRecipeTarget) -> str:
    if _is_cuda_train_recipe_target(target):
        return """        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        if USE_QK_NORM:
            q, k = norm(q), norm(k)
"""
    return """        q = self.rope(q)
        k = self.rope(k)
        if USE_QK_NORM:
            q = norm(q)
            k = norm(k)
"""


def _render_model_config(target: TrainRecipeTarget) -> str:
    if _is_cuda_train_recipe_target(target):
        return """def build_model_config(depth):
    assert depth == DEPTH
    assert MODEL_DIM == N_HEAD * HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=N_HEAD,
        n_kv_head=N_KV_HEAD,
        n_embd=MODEL_DIM,
        window_pattern=WINDOW_PATTERN,
    )
"""
    return """model_dim = MODEL_DIM
assert MODEL_DIM == N_HEAD * HEAD_DIM
config = GPTConfig(
    sequence_len=MAX_SEQ_LEN,
    vocab_size=vocab_size,
    n_layer=DEPTH,
    n_head=N_HEAD,
    n_kv_head=N_KV_HEAD,
    n_embd=MODEL_DIM,
    window_pattern=WINDOW_PATTERN,
)
"""
