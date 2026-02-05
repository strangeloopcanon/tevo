import math
from pathlib import Path

import torch

from transformer_evolution_llm.candidates import Candidate
from transformer_evolution_llm.data import TokenBatch
from transformer_evolution_llm.models import EvolutionModel
from transformer_evolution_llm.trainer import FullWeightTrainer


def synthetic_batches(vocab: int, seq_len: int, steps: int):
    for _ in range(steps):
        ids = torch.randint(0, vocab, (2, seq_len))
        yield TokenBatch(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            uids=["synthetic"],
        )


def test_full_weight_trainer_runs(tmp_path: Path, tiny_spec) -> None:
    trainer = FullWeightTrainer(checkpoint_dir=tmp_path, steps=2, eval_batches=1, device="cpu")
    candidate = Candidate(ident="cand-1", spec=tiny_spec)
    metrics, ckpt = trainer.train(
        candidate,
        tiny_spec,
        synthetic_batches(tiny_spec.model.head.vocab, tiny_spec.data.seq_len, steps=4),
    )
    assert "ppl_code" in metrics
    assert "ppl_train" in metrics
    assert "ppl_eval" in metrics
    # Router metrics should always be present for tooling, even if no MoE blocks exist.
    assert "router_entropy" in metrics
    assert "router_lb" in metrics
    assert "router_load_max" in metrics
    assert "router_load_min" in metrics
    assert "long_recall" in metrics
    assert "long_recall_proxy" in metrics
    assert "stop_reason_code" in metrics
    assert "nan_seen" in metrics
    assert "loss_spike" in metrics
    assert ckpt.exists()


def test_full_weight_trainer_passkey_probe(tmp_path: Path, tiny_spec) -> None:
    spec = tiny_spec.model_copy(deep=True)
    spec.train.passkey_eval_steps = 1
    spec.train.passkey_eval_batches = 1
    spec.train.passkey_eval_seq_len = 32
    spec.train.passkey_eval_min_distance = 0
    spec.train.passkey_eval_batch_size = 2

    trainer = FullWeightTrainer(checkpoint_dir=tmp_path, steps=1, eval_batches=1, device="cpu")
    candidate = Candidate(ident="cand-2", spec=spec)
    metrics, _ = trainer.train(
        candidate,
        spec,
        synthetic_batches(spec.model.head.vocab, spec.data.seq_len, steps=3),
    )

    assert "passkey_acc" in metrics
    assert "passkey_loss" in metrics
    assert 0.0 <= float(metrics["passkey_acc"]) <= 1.0
    assert float(metrics["passkey_loss"]) >= 0.0
    assert float(metrics["long_recall"]) == float(metrics["passkey_acc"])


def test_full_weight_trainer_speedrun_metrics(tmp_path: Path, tiny_spec) -> None:
    spec = tiny_spec.model_copy(deep=True)
    spec.train.speedrun_eval_interval = 10_000
    spec.train.speedrun_eval_batches = 1

    trainer = FullWeightTrainer(checkpoint_dir=tmp_path, steps=2, eval_batches=1, device="cpu")
    candidate = Candidate(ident="cand-speedrun", spec=spec)
    metrics, _ = trainer.train(
        candidate,
        spec,
        synthetic_batches(spec.model.head.vocab, spec.data.seq_len, steps=4),
    )

    for key in (
        "speedrun_reached",
        "speedrun_steps_to_target",
        "speedrun_tokens_to_target",
        "speedrun_time_to_target",
        "speedrun_best_eval_loss",
        "speedrun_end_eval_loss",
        "speedrun_loss_auc",
        "speedrun_loss_gap",
        "speedrun_score",
        "speedrun_time_score",
        "speedrun_error",
    ):
        assert key in metrics
        assert math.isfinite(float(metrics[key]))


def test_evaluate_perplexity_empty_iter_returns_penalty(tmp_path: Path, tiny_spec) -> None:
    trainer = FullWeightTrainer(checkpoint_dir=tmp_path, steps=1, eval_batches=1, device="cpu")
    model = EvolutionModel(tiny_spec.model)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    ppl = trainer._evaluate_perplexity(model, tiny_spec, iter(()), criterion)
    assert ppl >= 1e9


def test_speedrun_multi_thresholds_accept_strings(tmp_path: Path, tiny_spec, monkeypatch) -> None:
    spec = tiny_spec.model_copy(deep=True)
    spec.train.speedrun_eval_interval = 1
    spec.train.speedrun_eval_batches = 1
    spec.train.speedrun_multi_thresholds = ["3.5", "2.5", "bad"]
    trainer = FullWeightTrainer(checkpoint_dir=tmp_path, steps=1, eval_batches=1, device="cpu")
    candidate = Candidate(ident="cand-speedrun-thresholds", spec=spec)

    class DummyEvalModule:
        def __init__(self, vocab: int, seq_len: int):
            self.vocab = vocab
            self.seq_len = seq_len

        def reset_rng(self, seed: int) -> None:
            _ = seed

        def batches(self, max_tokens: int | None = None):
            _ = max_tokens
            ids = torch.randint(0, self.vocab, (2, self.seq_len))
            yield TokenBatch(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                uids=["eval"],
            )

    eval_module = DummyEvalModule(spec.model.head.vocab, spec.data.seq_len)
    monkeypatch.setattr(
        trainer,
        "_get_eval_module",
        lambda _spec, eval_batches=None: (
            eval_module,
            spec.data.seq_len * spec.data.batch_size * int(eval_batches or 1),
        ),
    )

    metrics, _ = trainer.train(
        candidate,
        spec,
        synthetic_batches(spec.model.head.vocab, spec.data.seq_len, steps=3),
    )

    assert "speedrun_auc" in metrics
    assert "speedrun_reached_3" in metrics
    assert "speedrun_reached_2" in metrics
