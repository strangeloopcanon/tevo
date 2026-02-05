from pathlib import Path

from transformer_evolution_llm.api import save_spec
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
