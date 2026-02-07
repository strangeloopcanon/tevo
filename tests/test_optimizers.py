import torch

from transformer_evolution_llm.dsl import OptimizerConfig, TrainSchedule
from transformer_evolution_llm.optimizers import build_optimizer


def _muon_schedule() -> TrainSchedule:
    return TrainSchedule(
        lr=1e-3,
        warmup=0,
        clip=1.0,
        bf16=False,
        grad_ckpt=False,
        max_tokens=16,
        optimizer=OptimizerConfig(
            name="muon",
            lr=1e-2,
            betas=(0.9, 0.98),
            eps=1e-8,
            weight_decay=0.0,
            muon_momentum=0.9,
            muon_nesterov=True,
            muon_ns_steps=2,
        ),
    )


def test_muon_uses_adamw_fallback_for_non_matrix_params() -> None:
    vec = torch.nn.Parameter(torch.tensor([1.0, -1.0], dtype=torch.float32))
    mat = torch.nn.Parameter(torch.randn(4, 4, dtype=torch.float32))
    optimizer = build_optimizer([vec, mat], _muon_schedule())

    loss = (vec.square().sum() + mat.square().sum()) * 0.5
    loss.backward()
    optimizer.step()

    vec_state = optimizer.state[vec]
    assert "exp_avg" in vec_state
    assert "exp_avg_sq" in vec_state
    assert "momentum_buffer" not in vec_state
    assert torch.isfinite(vec).all()


def test_muon_keeps_matrix_newton_schulz_state() -> None:
    mat = torch.nn.Parameter(torch.randn(4, 4, dtype=torch.float32))
    optimizer = build_optimizer([mat], _muon_schedule())

    loss = mat.square().sum() * 0.5
    loss.backward()
    optimizer.step()

    mat_state = optimizer.state[mat]
    assert "momentum_buffer" in mat_state
