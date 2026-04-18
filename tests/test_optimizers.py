from types import SimpleNamespace

import torch

from transformer_evolution_llm import optimizers as optimizers_mod
from transformer_evolution_llm.dsl import (
    ArchitectureSpec,
    GradientTransformConfig,
    OptimizerConfig,
    TrainSchedule,
    UpdateFilterConfig,
)
from transformer_evolution_llm.models import EvolutionModel
from transformer_evolution_llm.optimizers import (
    _topk_mask,
    apply_gradient_transform_,
    apply_update_filter_,
    build_optimizer,
)


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


def _adamw_schedule_with_filter(
    *,
    mode: str,
    keep_ratio: float,
    granularity: str = "element",
    block_size: int = 128,
    momentum_blend: float = 0.0,
    rescale_kept: bool = True,
    grad_mode: str = "identity",
    grad_ns_steps: int = 5,
    grad_eps: float = 1e-8,
) -> TrainSchedule:
    return TrainSchedule(
        lr=1e-3,
        warmup=0,
        clip=1.0,
        bf16=False,
        grad_ckpt=False,
        max_tokens=16,
        optimizer=OptimizerConfig(
            name="adamw",
            gradient_transform=GradientTransformConfig(
                mode=grad_mode,
                ns_steps=grad_ns_steps,
                eps=grad_eps,
            ),
            update_filter=UpdateFilterConfig(
                mode=mode,
                keep_ratio=keep_ratio,
                granularity=granularity,
                block_size=block_size,
                momentum_blend=momentum_blend,
                rescale_kept=rescale_kept,
            ),
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


def test_apply_update_filter_none_keeps_all() -> None:
    param = torch.nn.Parameter(torch.ones(10, dtype=torch.float32))
    optimizer = build_optimizer([param], _adamw_schedule_with_filter(mode="none", keep_ratio=1.0))
    grad = torch.ones_like(param)
    param.grad = grad.clone()

    keep = apply_update_filter_(optimizer, _adamw_schedule_with_filter(mode="none", keep_ratio=1.0))
    assert keep == 1.0
    assert torch.equal(param.grad, grad)


def test_apply_update_filter_element_topk_masks_expected_count() -> None:
    param = torch.nn.Parameter(torch.arange(10, dtype=torch.float32))
    schedule = _adamw_schedule_with_filter(mode="topk", keep_ratio=0.2, rescale_kept=False)
    optimizer = build_optimizer([param], schedule)
    param.grad = torch.arange(10, dtype=torch.float32)

    keep = apply_update_filter_(optimizer, schedule)
    non_zero = int((param.grad != 0).sum().item())
    assert keep == 0.2
    assert non_zero == 2


def test_apply_update_filter_block_bernoulli_applies_blockwise_mask() -> None:
    torch.manual_seed(0)
    param = torch.nn.Parameter(torch.ones(16, dtype=torch.float32))
    schedule = _adamw_schedule_with_filter(
        mode="bernoulli",
        keep_ratio=0.5,
        granularity="block",
        block_size=4,
        rescale_kept=False,
    )
    optimizer = build_optimizer([param], schedule)
    param.grad = torch.ones_like(param)

    keep = apply_update_filter_(optimizer, schedule)
    kept = int((param.grad != 0).sum().item())
    assert 0.0 <= keep <= 1.0
    assert kept % 4 == 0


def test_apply_gradient_transform_sign_changes_grad_values() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0, -2.0, 0.25], dtype=torch.float32))
    schedule = _adamw_schedule_with_filter(mode="none", keep_ratio=1.0, grad_mode="sign")
    optimizer = build_optimizer([param], schedule)
    param.grad = torch.tensor([0.5, -0.25, 4.0], dtype=torch.float32)

    applied = apply_gradient_transform_(optimizer, schedule)
    assert applied == 1.0
    assert torch.equal(param.grad, torch.tensor([1.0, -1.0, 1.0], dtype=torch.float32))


def test_apply_gradient_transform_orthogonalize_2d_only_applies_on_matrices() -> None:
    mat = torch.nn.Parameter(torch.randn(4, 4, dtype=torch.float32))
    vec = torch.nn.Parameter(torch.randn(8, dtype=torch.float32))
    schedule = _adamw_schedule_with_filter(
        mode="none",
        keep_ratio=1.0,
        grad_mode="orthogonalize_2d",
        grad_ns_steps=2,
    )
    optimizer = build_optimizer([mat, vec], schedule)
    mat.grad = torch.randn_like(mat)
    vec.grad = torch.randn_like(vec)

    applied = apply_gradient_transform_(optimizer, schedule)
    assert 0.0 < applied < 1.0


def test_topk_mask_clamps_out_of_bounds_indices(monkeypatch) -> None:
    values = torch.arange(0, 10, dtype=torch.float32)

    def _fake_topk(_input, k, largest=True, sorted=False):
        _ = (k, largest, sorted)
        # Simulate backend returning one invalid index == numel.
        return SimpleNamespace(
            indices=torch.tensor([0, int(values.numel())], dtype=torch.long),
        )

    monkeypatch.setattr(optimizers_mod.torch, "topk", _fake_topk)
    mask = _topk_mask(values, keep_ratio=0.2)
    assert mask.shape == values.shape
    assert mask.dtype == torch.bool
    # Clamp turns the invalid index into the last valid index.
    assert bool(mask[0].item())
    assert bool(mask[-1].item())


def test_build_optimizer_splits_tied_embedding_lr_group() -> None:
    spec = ArchitectureSpec(
        model={
            "name": "tiny-tied",
            "emb": {"dim": 8, "vocab": 16},
            "blocks": [
                {
                    "attn": {"kind": "MQA", "heads": 1, "head_dim": 8},
                    "ffn": {"type": "dense", "hidden": 16, "activation": "gelu"},
                }
            ],
            "head": {"vocab": 16, "tie_embeddings": True},
        },
        train={
            "lr": 1e-3,
            "tied_embedding_lr": 5e-2,
            "warmup": 0,
            "clip": 1.0,
            "bf16": False,
            "grad_ckpt": False,
            "max_tokens": 16,
        },
        data={
            "tokenizer": "gpt2",
            "seq_len": 4,
            "batch_size": 1,
            "workers": 0,
            "shards": [{"name": "dummy", "split": "train", "weight": 1.0}],
        },
    )
    model = EvolutionModel(spec.model)

    optimizer = build_optimizer(model, spec.train)

    assert len(optimizer.param_groups) == 2
    group_lrs = sorted(float(group["lr"]) for group in optimizer.param_groups)
    assert group_lrs == [1e-3, 5e-2]
    tied_group = next(group for group in optimizer.param_groups if float(group["lr"]) == 5e-2)
    assert tied_group["params"] == [model.embed.weight]


def test_build_optimizer_splits_official_style_parameter_groups() -> None:
    spec = ArchitectureSpec(
        model={
            "name": "tiny-untied",
            "emb": {"dim": 8, "vocab": 16},
            "blocks": [
                {
                    "attn": {"kind": "GQA", "heads": 2, "head_dim": 4, "kv_groups": 1},
                    "ffn": {"type": "dense", "hidden": 16, "activation": "relu_squared"},
                }
            ],
            "head": {"vocab": 16, "tie_embeddings": False},
        },
        train={
            "lr": 1e-3,
            "embed_lr": 2e-3,
            "head_lr": 3e-3,
            "matrix_lr": 4e-3,
            "scalar_lr": 5e-3,
            "warmup": 0,
            "clip": 1.0,
            "bf16": False,
            "grad_ckpt": False,
            "max_tokens": 16,
            "optimizer": {"name": "muon"},
        },
        data={
            "tokenizer": "gpt2",
            "seq_len": 4,
            "batch_size": 1,
            "workers": 0,
            "shards": [{"name": "dummy", "split": "train", "weight": 1.0}],
        },
    )
    model = EvolutionModel(spec.model)

    optimizer = build_optimizer(model, spec.train)

    group_lrs = {
        str(group.get("name", "default")): float(group["lr"]) for group in optimizer.param_groups
    }
    assert group_lrs["embedding"] == 2e-3
    assert group_lrs["untied_head"] == 3e-3
    assert group_lrs["matrix"] == 4e-3
    assert group_lrs["scalar_vector"] == 5e-3
