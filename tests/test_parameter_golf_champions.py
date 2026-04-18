from __future__ import annotations

import json
from pathlib import Path

from transformer_evolution_llm.api import load_spec
from transformer_evolution_llm.parameter_golf_champions import (
    build_champion_state,
    load_champion_seed_manifest,
    load_frontier_entry,
    normalize_seed_spec,
)


def _write_frontier(path: Path, spec_payload: dict, *, ident: str, score: float) -> None:
    payload = [
        {
            "id": ident,
            "status": "completed",
            "metrics": {
                "post_quant_val_bpb": score,
                "artifact_total_bytes": 12345.0,
                "throughput": 100.0,
            },
            "metadata": {"source": ident},
            "spec": spec_payload,
        }
    ]
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_load_frontier_entry_defaults_to_best(tmp_path: Path) -> None:
    spec = load_spec("configs/pg_combo_ffn_input_practical.yaml")
    frontier_path = tmp_path / "frontier.json"
    payload = [
        {
            "id": "worse",
            "status": "completed",
            "metrics": {"post_quant_val_bpb": 3.0, "artifact_total_bytes": 2.0},
            "spec": spec.model_dump(mode="python"),
        },
        {
            "id": "better",
            "status": "completed",
            "metrics": {"post_quant_val_bpb": 2.5, "artifact_total_bytes": 3.0},
            "spec": spec.model_dump(mode="python"),
        },
    ]
    frontier_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    entry = load_frontier_entry(frontier_path)

    assert entry["id"] == "better"


def test_normalize_seed_spec_uses_base_runtime_settings() -> None:
    base = load_spec("configs/pg_combo_champions_medium.yaml")
    source = load_spec("configs/pg_combo_ten_layer_broad_winner.yaml")

    normalized = normalize_seed_spec(base, source, source_name="ten-layer")

    assert normalized.model.n_layers == source.model.n_layers
    assert normalized.train.weight_decay == source.train.weight_decay
    assert normalized.parameter_golf is not None
    assert normalized.parameter_golf.train_shards_glob == base.parameter_golf.train_shards_glob
    assert normalized.parameter_golf.seed_family == "champion::ten-layer"
    assert normalized.evolution.population == base.evolution.population


def test_build_champion_state_writes_multiple_seed_pool(tmp_path: Path) -> None:
    base = load_spec("configs/pg_combo_champions_medium.yaml")
    spec_a = base.model_copy(deep=True)
    spec_a.model.name = "seed-a"
    spec_b = load_spec("configs/pg_combo_ten_layer_broad_winner.yaml")

    frontier_a = tmp_path / "frontier_a.json"
    frontier_b = tmp_path / "frontier_b.json"
    _write_frontier(
        frontier_a,
        spec_a.model_dump(mode="python"),
        ident="seed-a-candidate",
        score=2.8,
    )
    _write_frontier(
        frontier_b,
        spec_b.model_dump(mode="python"),
        ident="seed-b-candidate",
        score=2.7,
    )

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {"name": "seed-a", "frontier_path": str(frontier_a)},
                {"name": "seed-b", "frontier_path": str(frontier_b)},
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    state_path = tmp_path / "champions.state.json"
    build_champion_state(
        base_config_path="configs/pg_combo_champions_medium.yaml",
        seed_manifest_path=manifest,
        state_out=state_path,
        checkpoint_dir=tmp_path / "checkpoints",
        seed=0,
    )

    payload = json.loads(state_path.read_text())
    pool = payload["pool"]
    names = [entry["metadata"]["champion_seed_name"] for entry in pool]
    assert names == ["seed-a", "seed-b"]
    assert pool[0]["spec"]["parameter_golf"]["train_shards_glob"] == base.parameter_golf.train_shards_glob
    assert payload["frontier"] == [entry["id"] for entry in pool]
    manifest_seeds = load_champion_seed_manifest(manifest)
    assert [seed.name for seed in manifest_seeds] == ["seed-a", "seed-b"]
