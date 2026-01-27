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
    parent_candidate = Candidate(ident="parent-1", spec=tiny_spec, checkpoint=state_path)
    parent_candidate_2 = Candidate(ident="parent-2", spec=tiny_spec, checkpoint=state_path)
    runner.pool = [parent_candidate, parent_candidate_2]
    child = runner._spawn_candidate()
    assert child.seed_state_path is not None


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
