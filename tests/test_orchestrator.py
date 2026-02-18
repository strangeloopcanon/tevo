from pathlib import Path

from transformer_evolution_llm.api import save_spec
from transformer_evolution_llm.candidates import Candidate
from transformer_evolution_llm.dsl import CustomModuleConfig
from transformer_evolution_llm.orchestrator import EvolutionRunner


def test_simulated_run_advances_frontier(tmp_path: Path, tiny_spec):
    cfg_path = tmp_path / "spec.yaml"
    save_spec(tiny_spec, cfg_path)
    runner = EvolutionRunner(
        base_spec=tiny_spec,
        evolution_cfg=tiny_spec.evolution,
        mode="simulate",
        seed=123,
    )
    results = runner.run(generations=2)
    assert results, "expected at least one evaluated candidate"
    assert runner.frontier.entries, "frontier should not be empty"
    out_path = tmp_path / "frontier.json"
    runner.save_frontier(out_path)
    assert out_path.exists()


def test_map_elites_parent_selection_builds_archive(tiny_spec):
    spec = tiny_spec.model_copy(deep=True)
    spec.evolution.parent_selection = "map_elites"
    runner = EvolutionRunner(
        base_spec=spec,
        evolution_cfg=spec.evolution,
        mode="simulate",
        seed=123,
    )
    _ = runner.run(generations=3)
    assert runner.archive, "expected map-elites strategy to populate an archive"


def test_static_checker_reads_resource_thresholds(tiny_spec):
    spec = tiny_spec.model_copy(deep=True)
    spec.evolution.rung0_thresholds = {
        "max_params": 123.0,
        "max_kv_bytes_per_token": 456.0,
        "min_throughput_proxy": 7.0,
    }
    runner = EvolutionRunner(
        base_spec=spec,
        evolution_cfg=spec.evolution,
        mode="simulate",
        seed=123,
    )
    assert runner.checker.max_params == 123.0
    assert runner.checker.max_kv_bytes == 456.0
    assert runner.checker.min_throughput == 7.0


def test_state_roundtrip_preserves_min_objective_signs(tmp_path: Path, tiny_spec):
    runner = EvolutionRunner(
        base_spec=tiny_spec,
        evolution_cfg=tiny_spec.evolution,
        mode="simulate",
        seed=123,
    )
    runner.run(generations=1)
    state_path = tmp_path / "runner.state.json"
    runner.save_state(state_path)

    restored = EvolutionRunner.load_state(state_path, mode="simulate")
    for metric, direction in runner.objective_dir.items():
        if direction != "min":
            continue
        assert restored.score_weights[metric] == runner.score_weights[metric]


def test_checkpoint_gc_keeps_init_checkpoint(tmp_path: Path, tiny_spec):
    runner = EvolutionRunner(
        base_spec=tiny_spec,
        evolution_cfg=tiny_spec.evolution,
        mode="simulate",
        seed=123,
    )
    runner.checkpoint_dir = tmp_path
    init_ckpt = tmp_path / "init.pt"
    orphan_ckpt = tmp_path / "orphan.pt"
    init_ckpt.write_text("init")
    orphan_ckpt.write_text("orphan")
    runner._init_checkpoint = init_ckpt

    runner._garbage_collect_checkpoints()

    assert init_ckpt.exists()
    assert not orphan_ckpt.exists()


def test_gate_schedule_overrides_thresholds(tiny_spec):
    spec = tiny_spec.model_copy(deep=True)
    spec.evolution.rung0_thresholds = {"min_layers": 1.0}
    spec.evolution.gate_schedule = [
        {"generation": 0, "thresholds": {"min_layers": 1.0}},
        {"generation": 2, "thresholds": {"min_layers": 3.0}},
    ]
    runner = EvolutionRunner(
        base_spec=spec,
        evolution_cfg=spec.evolution,
        mode="simulate",
        seed=123,
    )
    runner._set_generation(0)
    assert runner._active_thresholds()["min_layers"] == 1.0
    runner._set_generation(2)
    assert runner._active_thresholds()["min_layers"] == 3.0


def test_novelty_archive_updates_metrics(tiny_spec):
    runner = EvolutionRunner(
        base_spec=tiny_spec,
        evolution_cfg=tiny_spec.evolution,
        mode="simulate",
        seed=123,
    )
    cand1 = Candidate(ident="cand-1", spec=tiny_spec.model_copy(deep=True))
    runner._evaluate_candidate(cand1)
    assert "novelty" in cand1.metrics
    assert "parent_novelty" in cand1.metrics
    assert runner._novelty_archive


def test_lineage_payload_contains_nodes_and_archive(tmp_path: Path, tiny_spec):
    runner = EvolutionRunner(
        base_spec=tiny_spec,
        evolution_cfg=tiny_spec.evolution,
        mode="simulate",
        seed=123,
    )
    runner.run(generations=1)
    out = tmp_path / "lineage.json"
    runner.save_lineage(out)
    import ujson as json

    payload = json.loads(out.read_text())
    assert isinstance(payload, dict)
    assert isinstance(payload.get("nodes"), list)
    assert isinstance(payload.get("novelty_archive"), list)


def test_map_elites_complexity_banding_splits_archive_keys(tiny_spec) -> None:
    spec = tiny_spec.model_copy(deep=True)
    spec.evolution.parent_selection = "map_elites"
    spec.evolution.map_elites_complexity_band = True
    spec.evolution.complexity_band_width = 1.0
    runner = EvolutionRunner(
        base_spec=spec,
        evolution_cfg=spec.evolution,
        mode="simulate",
        seed=123,
    )
    base = tiny_spec.model_copy(deep=True)
    richer = tiny_spec.model_copy(deep=True)
    richer.model.blocks[0].extras.extend(
        [CustomModuleConfig(type="custom", name=f"cm{i}") for i in range(6)]
    )
    cand_a = Candidate(ident="cand-a", spec=base, metrics={"ppl_code": 100.0, "throughput": 1.0})
    cand_b = Candidate(ident="cand-b", spec=richer, metrics={"ppl_code": 99.0, "throughput": 1.0})
    runner._update_archive(cand_a)
    runner._update_archive(cand_b)
    assert len(runner.archive) >= 2
    assert all("_C" in key for key in runner.archive)


def test_mutation_allowlist_restricts_sampling(tiny_spec) -> None:
    spec = tiny_spec.model_copy(deep=True)
    spec.evolution.crossover_prob = 0.0
    spec.evolution.mutation_allowlist = ["mix_optimizer_recipe"]
    spec.evolution.mutation_weights = {"mix_optimizer_recipe": 2.0}
    runner = EvolutionRunner(
        base_spec=spec,
        evolution_cfg=spec.evolution,
        mode="simulate",
        seed=123,
    )
    runner.run(generations=2)
    assert runner.mutation_weights == {"mix_optimizer_recipe": 2.0}
    assert runner._mutation_allowlist == ["mix_optimizer_recipe"]
    mutated = [cand for cand in runner._history if cand.parent is not None]
    assert mutated
    for cand in mutated:
        assert cand.mutation_trace == ["mix_optimizer_recipe"]
