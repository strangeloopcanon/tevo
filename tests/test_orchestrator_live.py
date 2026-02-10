import torch

from transformer_evolution_llm.candidates import Candidate
from transformer_evolution_llm.data import TokenBatch
from transformer_evolution_llm.models import EvolutionModel
from transformer_evolution_llm.orchestrator import EvolutionRunner


class DummyTrainer:
    def __init__(self):
        self.calls = 0

    def train(self, candidate, spec, batch_iter, seed_state_path=None):
        self.calls += 1
        return (
            {
                "ppl_code": 1.5,
                "ppl_math": 1.6,
                "throughput": 10.0,
                "params": 123,
                "ram": 0.01,
                "long_recall": 0.1,
            },
            spec.model.name and spec.model.name,  # dummy path placeholder handled by monkeypatch
        )


class DummyDataModule:
    def __init__(self, cfg):
        self.cfg = cfg

    def batches(self, max_tokens=None):
        for _ in range(4):
            ids = torch.randint(0, self.cfg.seq_len, (1, self.cfg.seq_len))
            yield TokenBatch(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                uids=["dummy"],
            )


class TrackingDataModule:
    def __init__(self, cfg):
        self.cfg = cfg
        self.max_tokens_calls: list[int | None] = []

    def reset_rng(self, seed):
        _ = seed

    def batches(self, max_tokens=None):
        self.max_tokens_calls.append(max_tokens)
        items: list[TokenBatch] = []
        for _ in range(2):
            ids = torch.randint(0, self.cfg.seq_len, (1, self.cfg.seq_len))
            items.append(
                TokenBatch(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    uids=["dummy"],
                )
            )
        return items


def test_live_runner_uses_trainer(monkeypatch, tiny_spec, tmp_path):
    monkeypatch.setattr("transformer_evolution_llm.orchestrator.DataModule", DummyDataModule)
    runner = EvolutionRunner(tiny_spec, tiny_spec.evolution, mode="live", seed=0)
    trainer = DummyTrainer()

    def fake_train(candidate, spec, batch_iter, seed_state_path=None):
        trainer.calls += 1
        ckpt = tmp_path / f"{candidate.ident}.pt"
        ckpt.write_text("checkpoint")
        return (
            {
                "ppl_code": 1.2,
                "ppl_math": 1.3,
                "throughput": 20.0,
                "params": 10,
                "ram": 0.001,
                "long_recall": 0.2,
            },
            ckpt,
        )

    runner.trainer = trainer
    runner.trainer.train = fake_train  # type: ignore[assignment]
    results = runner.run(generations=1)
    assert results
    assert trainer.calls >= 2  # seed + new candidate


def test_spawn_candidate_crossover(monkeypatch, tiny_spec, tmp_path):
    monkeypatch.setattr("transformer_evolution_llm.orchestrator.DataModule", DummyDataModule)
    runner = EvolutionRunner(tiny_spec, tiny_spec.evolution, mode="live", seed=0)
    runner.cfg.crossover_prob = 1.0
    runner.trainer = DummyTrainer()
    state_path = tmp_path / "parent.pt"
    torch.save(EvolutionModel(tiny_spec.model).state_dict(), state_path)
    spec_a = tiny_spec.model_copy(deep=True)
    spec_b = tiny_spec.model_copy(deep=True)
    # Ensure splice_blocks can create genuinely novel children.
    spec_a.model.blocks = [spec_a.model.blocks[0], spec_a.model.blocks[0]]
    spec_b.model.blocks = [spec_b.model.blocks[0], spec_b.model.blocks[0]]
    spec_a.model.blocks[0].attn.kind = "GQA"
    spec_a.model.blocks[1].attn.kind = "GQA"
    spec_b.model.blocks[0].attn.kind = "MHA"
    spec_b.model.blocks[1].attn.kind = "MHA"
    parent_candidate = Candidate(
        ident="parent-1",
        spec=spec_a,
        checkpoint=state_path,
        status="completed",
        metrics={"ppl_code": 1.0, "throughput": 1.0},
    )
    parent_candidate_2 = Candidate(
        ident="parent-2",
        spec=spec_b,
        checkpoint=state_path,
        status="completed",
        metrics={"ppl_code": 1.1, "throughput": 1.0},
    )
    runner.pool = [parent_candidate, parent_candidate_2]
    child = runner._spawn_candidate()
    assert child.seed_state_path is not None
    assert child.crossover_report is not None
    assert child.mutation_trace == ["xover::aligned"]


def test_spawn_candidate_resamples_noop_mutations(monkeypatch, tiny_spec):
    runner = EvolutionRunner(tiny_spec, tiny_spec.evolution, mode="simulate", seed=0)
    parent_candidate = Candidate(
        ident="parent-1",
        spec=tiny_spec.model_copy(deep=True),
        status="completed",
        metrics={"ppl_code": 1.0, "throughput": 1.0},
    )
    runner.pool = [parent_candidate]

    calls = {"n": 0}

    def fake_mutate_with_trace(spec, rng, weights, steps=1, validate=True):
        calls["n"] += 1
        if calls["n"] == 1:
            # First attempt is a no-op clone.
            return ("noop", spec.model_copy(deep=True), ["noop"])
        # Second attempt changes the spec.
        updated = spec.model_copy(deep=True)
        updated.train.lr = float(updated.train.lr) * 1.01
        return ("lr_jitter", updated, ["lr_jitter"])

    monkeypatch.setattr(
        "transformer_evolution_llm.orchestrator.mutate_with_trace",
        fake_mutate_with_trace,
    )
    child = runner._spawn_candidate()
    assert calls["n"] >= 2
    assert float(child.spec.train.lr) != float(parent_candidate.spec.train.lr)
    assert child.mutation_trace == ["lr_jitter"]


def test_live_runner_single_rung_uses_full_steps(monkeypatch, tiny_spec, tmp_path):
    monkeypatch.setattr("transformer_evolution_llm.orchestrator.DataModule", DummyDataModule)
    runner = EvolutionRunner(tiny_spec, tiny_spec.evolution, mode="live", seed=0)
    runner.cfg.rung1_tokens = 1024
    runner.cfg.rung2_tokens = 1024  # no continuation rung
    trainer = DummyTrainer()
    trainer.steps = 50

    def fake_train(candidate, spec, batch_iter, seed_state_path=None):
        assert trainer.steps == 50
        ckpt = tmp_path / f"{candidate.ident}.pt"
        ckpt.write_text("checkpoint")
        return (
            {
                "ppl_code": 1.2,
                "ppl_math": 1.3,
                "throughput": 20.0,
                "params": 10,
                "ram": 0.001,
                "long_recall": 0.2,
            },
            ckpt,
        )

    runner.trainer = trainer
    runner.trainer.train = fake_train  # type: ignore[assignment]
    results = runner.run(generations=1)
    assert results


def test_live_runner_rejects_low_entropy_candidates(monkeypatch, tiny_spec, tmp_path):
    monkeypatch.setattr("transformer_evolution_llm.orchestrator.DataModule", DummyDataModule)
    runner = EvolutionRunner(tiny_spec, tiny_spec.evolution, mode="live", seed=0)

    def fake_train(candidate, spec, batch_iter, seed_state_path=None):
        stop_code = 4.0 if candidate.ident.startswith("seed") else 2.0
        ckpt = tmp_path / f"{candidate.ident}.pt"
        ckpt.write_text("checkpoint")
        return (
            {
                "ppl_code": 1.2,
                "ppl_math": 1.3,
                "throughput": 20.0,
                "params": 10,
                "ram": 0.001,
                "long_recall": 0.2,
                "stop_reason_code": stop_code,
            },
            ckpt,
        )

    trainer = DummyTrainer()
    runner.trainer = trainer
    runner.trainer.train = fake_train  # type: ignore[assignment]

    results = runner.run(generations=1)
    assert len(results) == 1, "expected only the seed to be completed"
    assert all(
        cand.metrics.get("stop_reason_code") != 2.0 for cand in runner.frontier.entries
    ), "low-entropy candidates should not enter the frontier"


def test_live_runner_rechecks_objectives_after_promotion(monkeypatch, tiny_spec, tmp_path):
    monkeypatch.setattr("transformer_evolution_llm.orchestrator.DataModule", DummyDataModule)
    runner = EvolutionRunner(tiny_spec, tiny_spec.evolution, mode="live", seed=0)
    runner.cfg.rung1_tokens = 256
    runner.cfg.rung2_tokens = 512
    trainer = DummyTrainer()
    trainer.steps = 16
    calls = {"n": 0}

    def fake_train(candidate, spec, batch_iter, seed_state_path=None):
        _ = (spec, batch_iter, seed_state_path)
        calls["n"] += 1
        ckpt = tmp_path / f"{candidate.ident}-{calls['n']}.pt"
        ckpt.write_text("checkpoint")
        metrics = {
            "ppl_code": 1.2,
            "ppl_math": 1.3,
            "throughput": 20.0,
            "params": 10,
            "ram": 0.001,
            "long_recall": 0.2,
        }
        if calls["n"] == 3:
            metrics["stop_reason_code"] = 2.0
        return metrics, ckpt

    runner.trainer = trainer
    runner.trainer.train = fake_train  # type: ignore[assignment]
    monkeypatch.setattr(runner, "_should_promote", lambda *_args, **_kwargs: True)

    candidate = Candidate(ident="cand-promo", spec=tiny_spec.model_copy(deep=True))
    runner._evaluate_candidate(candidate)

    assert calls["n"] == 3
    assert candidate.status == "failed"
    assert candidate not in runner.frontier.entries
    assert candidate.checkpoint is None


def test_live_runner_adaptive_rung_clamps_to_budget(monkeypatch, tiny_spec, tmp_path):
    monkeypatch.setattr("transformer_evolution_llm.orchestrator.DataModule", DummyDataModule)
    monkeypatch.setattr(
        "transformer_evolution_llm.orchestrator.estimate_params", lambda _spec: 100.0
    )
    runner = EvolutionRunner(tiny_spec, tiny_spec.evolution, mode="live", seed=0)
    runner.data_module = TrackingDataModule(tiny_spec.data)
    runner.cfg.rung1_tokens = 300
    runner.cfg.rung2_tokens = 400
    runner.cfg.adaptive_rung_budget = True
    runner.cfg.adaptive_rung_fast_promote_threshold = 0.1
    runner.cfg.adaptive_rung_slow_stop_threshold = 0.0
    trainer = DummyTrainer()
    trainer.steps = 16
    calls = {"n": 0}

    def fake_train(candidate, spec, batch_iter, seed_state_path=None):
        _ = (spec, batch_iter, seed_state_path)
        calls["n"] += 1
        ckpt = tmp_path / f"{candidate.ident}-{calls['n']}.pt"
        ckpt.write_text("checkpoint")
        if calls["n"] == 1:
            return (
                {
                    "ppl_code": 1.2,
                    "ppl_math": 1.3,
                    "throughput": 20.0,
                    "params": 10,
                    "ram": 0.001,
                    "long_recall": 0.2,
                    "improvement_rate": 0.8,
                },
                ckpt,
            )
        return (
            {
                "ppl_code": 1.2,
                "ppl_math": 1.3,
                "throughput": 20.0,
                "params": 10,
                "ram": 0.001,
                "long_recall": 0.2,
            },
            ckpt,
        )

    runner.trainer = trainer
    runner.trainer.train = fake_train  # type: ignore[assignment]
    captured: dict[str, int] = {}

    def capture_should_promote(_candidate, tokens_budget, used_tokens):
        captured["tokens_budget"] = int(tokens_budget)
        captured["used_tokens"] = int(used_tokens)
        return False

    monkeypatch.setattr(runner, "_should_promote", capture_should_promote)
    candidate_spec = tiny_spec.model_copy(deep=True)
    candidate_spec.priors.tokens_per_param = 4.0
    candidate = Candidate(ident="cand-adaptive", spec=candidate_spec)
    runner._evaluate_candidate(candidate)

    assert candidate.status == "completed"
    assert captured["tokens_budget"] == 400
    assert captured["used_tokens"] <= captured["tokens_budget"]
    assert runner.data_module.max_tokens_calls[:2] == [300, 100]
