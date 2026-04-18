import math
from pathlib import Path

import torch

from transformer_evolution_llm.candidates import Candidate
from transformer_evolution_llm.data import TokenBatch
from transformer_evolution_llm.dsl import ParameterGolfConfig
from transformer_evolution_llm.models import EvolutionModel
from transformer_evolution_llm.optimizers import build_optimizer
from transformer_evolution_llm.trainer import FullWeightTrainer
import transformer_evolution_llm.trainer as trainer_module


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
    assert "opt_mask_keep_observed_avg" in metrics
    assert "opt_mask_keep_observed_last" in metrics
    assert "opt_grad_transform_applied_avg" in metrics
    assert "opt_grad_transform_applied_last" in metrics
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


def test_full_weight_trainer_reports_gradient_accumulation_metrics(
    tmp_path: Path, tiny_spec
) -> None:
    spec = tiny_spec.model_copy(deep=True)
    spec.train.batch_tokens = int(spec.data.seq_len) * int(spec.data.batch_size) * 2

    trainer = FullWeightTrainer(checkpoint_dir=tmp_path, steps=2, eval_batches=1, device="cpu")
    candidate = Candidate(ident="cand-accum", spec=spec)
    metrics, _ = trainer.train(
        candidate,
        spec,
        synthetic_batches(spec.model.head.vocab, spec.data.seq_len, steps=6),
    )

    assert metrics["grad_accum_steps"] == 2.0
    assert metrics["train_micro_batch_tokens"] == float(spec.data.seq_len * spec.data.batch_size)
    assert metrics["train_effective_batch_tokens"] == float(
        spec.data.seq_len * spec.data.batch_size * 2
    )


def test_full_weight_trainer_allows_zero_clip(tmp_path: Path, tiny_spec) -> None:
    spec = tiny_spec.model_copy(deep=True)
    spec.train.clip = 0.0

    trainer = FullWeightTrainer(checkpoint_dir=tmp_path, steps=2, eval_batches=1, device="cpu")
    candidate = Candidate(ident="cand-zero-clip", spec=spec)
    metrics, _ = trainer.train(
        candidate,
        spec,
        synthetic_batches(spec.model.head.vocab, spec.data.seq_len, steps=4),
    )

    assert "max_grad_norm" in metrics
    assert math.isfinite(float(metrics["max_grad_norm"]))


def test_muon_momentum_warmup_schedule_adjusts_group_momentum(tmp_path: Path, tiny_spec) -> None:
    spec = tiny_spec.model_copy(deep=True)
    spec.train.optimizer.name = "muon"
    spec.train.optimizer.muon_momentum = 0.95
    spec.train.optimizer.muon_momentum_warmup_start = 0.8
    spec.train.optimizer.muon_momentum_warmup_steps = 10

    trainer = FullWeightTrainer(checkpoint_dir=tmp_path, steps=1, eval_batches=1, device="cpu")
    model = EvolutionModel(spec.model)
    built = build_optimizer(model, spec.train)
    base_lrs = [float(group.get("lr", 0.0)) for group in built.param_groups]
    base_momentums = [
        float(group.get("momentum")) if group.get("momentum") is not None else None
        for group in built.param_groups
    ]

    trainer._apply_optimizer_schedule(
        built,
        base_lrs,
        base_momentums,
        spec,
        step_idx=0,
        total_steps=10,
        elapsed_s=0.0,
    )

    momentums = [float(group["momentum"]) for group in built.param_groups if "momentum" in group]
    assert momentums
    assert all(abs(momentum - 0.8) < 1e-9 for momentum in momentums)


def test_parameter_golf_wallclock_budget_caps_training_only(
    tmp_path: Path,
    tiny_spec,
    monkeypatch,
) -> None:
    spec = tiny_spec.model_copy(deep=True)
    spec.parameter_golf = ParameterGolfConfig(
        train_shards_glob="data/pg/train_*.bin",
        val_shards_glob="data/pg/val_*.bin",
        tokenizer_path="data/pg/sp1024.model",
        max_wallclock_seconds=5e-5,
        eval_protocol="mid_fidelity",
    )

    ticks = {"value": 0.0}

    def fake_perf_counter() -> float:
        ticks["value"] += 1e-6
        return ticks["value"]

    monkeypatch.setattr(trainer_module.time, "perf_counter", fake_perf_counter)

    trainer = FullWeightTrainer(checkpoint_dir=tmp_path, steps=10, eval_batches=1, device="cpu")
    monkeypatch.setattr(
        trainer,
        "_predict_elapsed_seconds",
        lambda _durations, fallback_seconds=0.0: max(float(fallback_seconds), 3e-5),
    )

    candidate = Candidate(ident="cand-pg-wallclock", spec=spec)
    metrics, _ = trainer.train(
        candidate,
        spec,
        synthetic_batches(spec.model.head.vocab, spec.data.seq_len, steps=40),
    )

    assert metrics["wallclock_stop_triggered"] == 1.0
    assert metrics["train_wallclock_seconds_used"] <= metrics["wallclock_seconds_budget"] + 1e-12
    assert metrics["total_job_seconds_used"] >= metrics["train_wallclock_seconds_used"]
