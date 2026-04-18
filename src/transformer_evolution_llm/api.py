"""Public API for downstream modules."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch
import ujson as json

from .data import DataModule
from .dsl import ArchitectureSpec, load_architecture_spec, save_architecture_spec
from .parameter_golf import ParameterGolfDataModule
from .parameter_golf_export import (
    build_official_submission_plan,
    export_parameter_golf_workspace,
)
from .parameter_golf_runtime import preflight_parameter_golf_config, run_parameter_golf_benchmark
from .parameter_golf_seeded import transfer_parameter_golf_motif
from .train_recipe import (
    TrainRecipe,
    TrainRecipeTarget,
    export_train_recipes,
    load_train_recipe,
    render_train_recipe_to_path,
    save_train_recipe,
)

__all__ = [
    "ArchitectureSpec",
    "TrainRecipe",
    "TrainRecipeTarget",
    "load_spec",
    "save_spec",
    "load_recipe",
    "save_recipe",
    "run_evolution",
    "export_frontier_seed",
    "export_train_recipe",
    "export_parameter_golf_workspace",
    "build_official_submission_plan",
    "preflight_parameter_golf_config",
    "build_data_module_for_spec",
    "render_train_recipe",
    "prune_checkpoints",
    "convert_checkpoints",
    "run_parameter_golf_benchmark",
    "transfer_parameter_golf_motif",
]


def load_spec(path: str | Path) -> ArchitectureSpec:
    """Read an architecture spec from disk."""
    return load_architecture_spec(path)


def save_spec(spec: ArchitectureSpec, path: str | Path) -> None:
    """Persist an architecture spec to disk."""
    save_architecture_spec(spec, path)


def load_recipe(path: str | Path) -> TrainRecipe:
    """Read a train recipe from disk."""
    return load_train_recipe(path)


def save_recipe(recipe: TrainRecipe, path: str | Path) -> None:
    """Persist a train recipe to disk."""
    save_train_recipe(recipe, path)


def build_data_module_for_spec(
    spec: ArchitectureSpec,
    *,
    seed: int | None = None,
) -> DataModule | ParameterGolfDataModule:
    """Construct the right training data module for a spec.

    Parameter Golf specs use the challenge shard reader instead of the generic
    Hugging Face-backed data module.
    """
    seed_val = int(seed if seed is not None else getattr(spec.train, "seed", 0) or 0)
    if spec.parameter_golf is not None:
        return ParameterGolfDataModule(
            spec.parameter_golf,
            seq_len=spec.data.seq_len,
            batch_size=spec.data.batch_size,
            seed=seed_val,
        )
    return DataModule(spec.data, seed=seed_val)


def run_evolution(
    config_path: str | Path,
    generations: int,
    mode: str = "simulate",
    seed: int = 0,
    out_path: str | Path = "runs/frontier.json",
) -> None:
    """Entry point used by the CLI to run a full search."""
    from .orchestrator import EvolutionRunner

    spec = load_spec(config_path)
    runner = EvolutionRunner(
        base_spec=spec,
        evolution_cfg=spec.evolution,
        mode=mode,
        seed=seed,
    )
    runner.run(generations)
    runner.save_frontier(Path(out_path))


def export_frontier_seed(
    frontier_path: str | Path,
    candidate_id: str,
    out_path: str | Path,
    checkpoint_dir: str | Path = "runs/checkpoints",
) -> None:
    """Export a frontier candidate as a reusable seed config.

    The exported spec includes a train.init_checkpoint field pointing to the
    expected checkpoint for the selected candidate so future live runs can
    start from learned weights instead of reinitializing.
    """
    frontier_path = Path(frontier_path)
    if not frontier_path.exists():
        msg = f"Frontier file not found: {frontier_path}"
        raise FileNotFoundError(msg)
    data: list[dict[str, Any]] = json.loads(frontier_path.read_text())
    match = next((row for row in data if row.get("id") == candidate_id), None)
    if match is None:
        msg = f"Candidate {candidate_id} not found in {frontier_path}"
        raise ValueError(msg)
    spec_data = match.get("spec")
    if not isinstance(spec_data, dict):
        msg = f"Frontier entry for {candidate_id} missing spec payload"
        raise ValueError(msg)
    spec = ArchitectureSpec(**spec_data)
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_path = checkpoint_dir_path / f"{candidate_id}.pt"
    if not checkpoint_path.exists():
        msg = f"Checkpoint for {candidate_id} not found at {checkpoint_path}"
        raise FileNotFoundError(msg)
    spec.train.init_checkpoint = str(checkpoint_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_architecture_spec(spec, out_path)


def export_train_recipe(
    source_path: str | Path,
    out_path: str | Path,
    candidate_id: str | None = None,
    top_k: int = 1,
    metric: str = "ppl_code",
) -> list[Path]:
    """Export one or more train recipes from a spec or frontier file."""
    recipes = export_train_recipes(
        source_path=source_path,
        candidate_id=candidate_id,
        top_k=top_k,
        metric=metric,
    )
    out_path = Path(out_path)
    written: list[Path] = []
    if len(recipes) == 1 and out_path.suffix in {".yaml", ".yml", ".json"}:
        save_train_recipe(recipes[0], out_path)
        return [out_path]
    out_path.mkdir(parents=True, exist_ok=True)
    for recipe in recipes:
        safe_name = (
            "".join(
                ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in str(recipe.name)
            ).strip("_")
            or "recipe"
        )
        recipe_path = out_path / f"{safe_name}.train_recipe.yaml"
        save_train_recipe(recipe, recipe_path)
        written.append(recipe_path)
    return written


def render_train_recipe(
    recipe_path: str | Path,
    target: TrainRecipeTarget,
    train_py_path: str | Path,
    out_path: str | Path | None = None,
) -> Path:
    """Render a train recipe into a downstream train.py target."""
    recipe = load_train_recipe(recipe_path)
    return render_train_recipe_to_path(
        recipe,
        target=target,
        train_py_path=train_py_path,
        out_path=out_path,
    )


def _ids_from_state(state_path: Path) -> set[str]:
    """Collect candidate ids referenced by a saved state file."""
    state: dict[str, Any] = json.loads(state_path.read_text())
    keep: set[str] = set()
    for key in ("frontier", "pool", "parents"):
        items = state.get(key, [])
        if isinstance(items, list):
            for item in items:
                if isinstance(item, str):
                    keep.add(item)
    history = state.get("history", [])
    if isinstance(history, list):
        for item in history:
            if isinstance(item, dict):
                cid = item.get("id")
                if cid:
                    keep.add(cid)
            elif isinstance(item, str):
                keep.add(item)
    return keep


def _ids_from_frontier(frontier_path: Path) -> set[str]:
    """Collect candidate ids referenced by a frontier file."""
    data: list[dict[str, Any]] = json.loads(frontier_path.read_text())
    keep: set[str] = set()
    for row in data:
        cid = row.get("id")
        if cid:
            keep.add(cid)
    return keep


def prune_checkpoints(
    checkpoint_dir: str | Path,
    frontier_path: str | Path | None = None,
    state_path: str | Path | None = None,
    dry_run: bool = False,
) -> tuple[list[Path], list[Path]]:
    """Remove checkpoints not referenced by a frontier/state.

    Returns (kept, removed) lists of paths.
    """
    if frontier_path is None and state_path is None:
        msg = "Provide at least one of frontier_path or state_path"
        raise ValueError(msg)
    checkpoint_dir_path = Path(checkpoint_dir)
    if not checkpoint_dir_path.exists():
        msg = f"Checkpoint directory not found: {checkpoint_dir_path}"
        raise FileNotFoundError(msg)

    keep_ids: set[str] = set()
    if frontier_path is not None:
        frontier_path = Path(frontier_path)
        if not frontier_path.exists():
            msg = f"Frontier file not found: {frontier_path}"
            raise FileNotFoundError(msg)
        keep_ids.update(_ids_from_frontier(frontier_path))
    if state_path is not None:
        state_path = Path(state_path)
        if not state_path.exists():
            msg = f"State file not found: {state_path}"
            raise FileNotFoundError(msg)
        keep_ids.update(_ids_from_state(state_path))

    kept: list[Path] = []
    removed: list[Path] = []
    for path in checkpoint_dir_path.glob("*.pt"):
        cid = path.stem
        if cid in keep_ids:
            kept.append(path)
        else:
            removed.append(path)
            if not dry_run:
                path.unlink(missing_ok=True)
    return kept, removed


def _parse_checkpoint_dtype(dtype: str) -> torch.dtype:
    key = (dtype or "fp16").lower()
    if key in {"fp16", "float16"}:
        return torch.float16
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported checkpoint dtype: {dtype}")


def convert_checkpoints(
    checkpoint_dir: str | Path,
    dtype: str = "fp16",
    dry_run: bool = False,
) -> tuple[list[Path], int, int]:
    """Convert checkpoint tensors to a smaller dtype (in-place).

    Returns (paths, total_before_bytes, total_after_bytes).
    """
    checkpoint_dir_path = Path(checkpoint_dir)
    if not checkpoint_dir_path.exists():
        msg = f"Checkpoint directory not found: {checkpoint_dir_path}"
        raise FileNotFoundError(msg)
    torch_dtype = _parse_checkpoint_dtype(dtype)

    paths = sorted(checkpoint_dir_path.glob("*.pt"))
    total_before = 0
    total_after = 0
    processed: list[Path] = []
    for path in paths:
        try:
            before = path.stat().st_size
        except OSError:
            before = 0
        total_before += before
        if dry_run:
            total_after += before
            processed.append(path)
            continue

        state = torch.load(
            path, map_location="cpu"
        )  # nosec B614 - loading trusted local checkpoints
        if not isinstance(state, dict):
            msg = f"Checkpoint {path} did not contain a state_dict mapping"
            raise ValueError(msg)
        converted: dict[str, Any] = {}
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                tensor = value.detach().to(device="cpu")
                if tensor.is_floating_point():
                    tensor = tensor.to(dtype=torch_dtype)
                converted[key] = tensor
            else:
                converted[key] = value

        with tempfile.NamedTemporaryFile(
            delete=False,
            dir=checkpoint_dir_path,
            prefix=f"{path.stem}.",
            suffix=".tmp.pt",
        ) as tmp:
            tmp_path = Path(tmp.name)
        try:
            torch.save(converted, tmp_path)
            os.replace(tmp_path, path)
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            total_after += path.stat().st_size
        except OSError:
            total_after += 0
        processed.append(path)

    return processed, total_before, total_after
