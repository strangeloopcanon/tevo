from pathlib import Path

from transformer_evolution_llm.api import save_spec
from transformer_evolution_llm.candidates import Candidate
from transformer_evolution_llm.dsl import CustomModuleConfig, ParameterGolfConfig
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


def test_parameter_golf_candidates_report_official_submission_metrics(tiny_spec) -> None:
    spec = tiny_spec.model_copy(deep=True)
    spec.parameter_golf = ParameterGolfConfig(
        train_shards_glob="data/pg/train_*.bin",
        val_shards_glob="data/pg/val_*.bin",
        tokenizer_path="data/pg/sp1024.model",
        artifact_budget_bytes=16_000_000,
        code_bytes=50_000,
        track="10min",
        seed_family="exact_official_baseline",
        lane_kind="exportable",
        exportable_family="exact_official_baseline",
    )
    runner = EvolutionRunner(
        base_spec=spec,
        evolution_cfg=spec.evolution,
        mode="simulate",
        seed=123,
    )
    candidate = Candidate(ident="cand-official", spec=spec.model_copy(deep=True))
    runner._evaluate_candidate(candidate)
    assert "official_submission_exportable" in candidate.metrics
    assert "official_submission_total_bytes_est" in candidate.metrics
    assert "main_track_eligible_est" in candidate.metrics
    assert candidate.metrics["artifact_budget_utilization_est"] > 0.0
    assert candidate.metrics["artifact_budget_fill_score_est"] > 0.0
    assert candidate.metrics["artifact_budget_edge_score_est"] >= 0.0
    assert candidate.metrics["complexity_score"] > 0.0
    assert candidate.metadata["seed_family"] == "exact_official_baseline"
    assert candidate.metadata["lane_kind"] == "exportable"
    assert "motif_signature" in candidate.metadata


def test_generic_min_thresholds_can_require_richer_candidates(tiny_spec) -> None:
    spec = tiny_spec.model_copy(deep=True)
    spec.parameter_golf = ParameterGolfConfig(
        train_shards_glob="data/pg/train_*.bin",
        val_shards_glob="data/pg/val_*.bin",
        tokenizer_path="data/pg/sp1024.model",
        artifact_budget_bytes=16_000_000,
        code_bytes=50_000,
        track="10min",
        seed_family="growth-lane",
        lane_kind="incubator",
    )
    spec.evolution.rung0_thresholds = {"min_complexity_score": 99.0}
    runner = EvolutionRunner(
        base_spec=spec,
        evolution_cfg=spec.evolution,
        mode="simulate",
        seed=123,
    )
    candidate = Candidate(ident="cand-growth", spec=spec.model_copy(deep=True))
    runner._evaluate_candidate(candidate)
    assert candidate.status == "failed"


def test_edge_score_prefers_near_budget_candidates(tiny_spec) -> None:
    spec = tiny_spec.model_copy(deep=True)
    spec.parameter_golf = ParameterGolfConfig(
        train_shards_glob="data/pg/train_*.bin",
        val_shards_glob="data/pg/val_*.bin",
        tokenizer_path="data/pg/sp1024.model",
        artifact_budget_bytes=16_000_000,
        code_bytes=50_000,
        track="10min",
        seed_family="edge-lane",
        lane_kind="incubator",
    )
    runner = EvolutionRunner(
        base_spec=spec,
        evolution_cfg=spec.evolution,
        mode="simulate",
        seed=123,
    )

    small = Candidate(ident="small", spec=spec.model_copy(deep=True))
    near_edge = Candidate(ident="near-edge", spec=spec.model_copy(deep=True))
    runner._evaluate_candidate(small)
    runner._evaluate_candidate(near_edge)

    small.metrics["artifact_total_bytes"] = 8_000_000.0
    near_edge.metrics["artifact_total_bytes"] = 15_400_000.0
    runner._annotate_official_submission_metrics(small)
    runner._annotate_official_submission_metrics(near_edge)

    assert (
        near_edge.metrics["artifact_budget_edge_score"]
        > small.metrics["artifact_budget_edge_score"]
    )


def test_edge_score_respects_parameter_golf_target_window(tiny_spec) -> None:
    spec = tiny_spec.model_copy(deep=True)
    spec.parameter_golf = ParameterGolfConfig(
        train_shards_glob="data/pg/train_*.bin",
        val_shards_glob="data/pg/val_*.bin",
        tokenizer_path="data/pg/sp1024.model",
        artifact_budget_bytes=16_000_000,
        code_bytes=50_000,
        track="10min",
        seed_family="edge-lane-tight",
        lane_kind="incubator",
        target_budget_utilization=0.99,
        target_budget_under_window=0.05,
        target_budget_over_window=0.02,
    )
    runner = EvolutionRunner(
        base_spec=spec,
        evolution_cfg=spec.evolution,
        mode="simulate",
        seed=123,
    )

    below_target = Candidate(ident="below-target", spec=spec.model_copy(deep=True))
    at_target = Candidate(ident="at-target", spec=spec.model_copy(deep=True))
    runner._evaluate_candidate(below_target)
    runner._evaluate_candidate(at_target)

    below_target.metrics["artifact_total_bytes"] = 15_400_000.0
    at_target.metrics["artifact_total_bytes"] = 15_840_000.0
    runner._annotate_official_submission_metrics(below_target)
    runner._annotate_official_submission_metrics(at_target)

    assert (
        at_target.metrics["artifact_budget_edge_score"]
        > below_target.metrics["artifact_budget_edge_score"]
    )


def test_template_learning_bootstraps_seed_file(tmp_path: Path, tiny_spec) -> None:
    spec = tiny_spec.model_copy(deep=True)
    seed_path = tmp_path / "seed_templates.yaml"
    save_path = tmp_path / "learned" / "templates.yaml"
    seed_payload = (
        "templates:\n  - name: seed-one\n    weight: 1.0\n    conditions: {}\n    actions: []\n"
    )
    seed_path.write_text(seed_payload)
    spec.evolution.template_learning = True
    spec.evolution.template_learning_seed_path = str(seed_path)
    spec.evolution.template_learning_save_path = str(save_path)

    EvolutionRunner(
        base_spec=spec,
        evolution_cfg=spec.evolution,
        mode="simulate",
        seed=123,
    )

    assert save_path.exists()
    assert save_path.read_text() == seed_payload


def test_incubator_lane_never_marks_candidate_main_track_eligible(tiny_spec) -> None:
    spec = tiny_spec.model_copy(deep=True)
    spec.parameter_golf = ParameterGolfConfig(
        train_shards_glob="data/pg/train_*.bin",
        val_shards_glob="data/pg/val_*.bin",
        tokenizer_path="data/pg/sp1024.model",
        artifact_budget_bytes=16_000_000,
        code_bytes=50_000,
        track="10min",
        seed_family="incubator_keep90",
        lane_kind="incubator",
        incubator_anchor_post_quant_val_bpb=3.65,
    )
    runner = EvolutionRunner(
        base_spec=spec,
        evolution_cfg=spec.evolution,
        mode="simulate",
        seed=123,
    )
    candidate = Candidate(ident="cand-incubator", spec=spec.model_copy(deep=True))
    runner._evaluate_candidate(candidate)
    assert candidate.metrics["main_track_eligible_est"] == 0.0
    assert candidate.metadata["lane_kind"] == "incubator"


def test_incubator_rediscovery_counts_across_seed_families(tiny_spec) -> None:
    spec = tiny_spec.model_copy(deep=True)
    spec.parameter_golf = ParameterGolfConfig(
        train_shards_glob="data/pg/train_*.bin",
        val_shards_glob="data/pg/val_*.bin",
        tokenizer_path="data/pg/sp1024.model",
        artifact_budget_bytes=16_000_000,
        code_bytes=50_000,
        track="10min",
        seed_family="incubator_keep90",
        lane_kind="incubator",
        incubator_anchor_post_quant_val_bpb=3.65,
    )
    runner = EvolutionRunner(
        base_spec=spec,
        evolution_cfg=spec.evolution,
        mode="simulate",
        seed=123,
    )
    candidate = Candidate(ident="cand-incubator", spec=spec.model_copy(deep=True))
    candidate.metrics["post_quant_val_bpb"] = 3.65
    runner._annotate_seed_lane_metadata(candidate)
    signature = candidate.metadata["motif_signature"]
    runner._motif_families[signature] = {"exact_official_baseline"}

    runner._annotate_incubator_metrics(candidate)

    assert candidate.metrics["motif_appearance_count"] == 2.0
    assert candidate.metrics["motif_transfer_eligible"] == 1.0
    assert candidate.metadata["motif_promotion"]["reason"] == "rediscovered"
