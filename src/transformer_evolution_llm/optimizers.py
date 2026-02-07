"""Optimizer registry (AdamW, Lion, Muon).

Optimizer configuration lives in the DSL. Evolution can optionally mutate it via
registered mutations.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import cast, overload

import torch
from torch import Tensor
from torch.optim import AdamW, Optimizer

from .dsl import OptimizerConfig, TrainSchedule


class Lion(Optimizer):
    """Minimal Lion optimizer (Chen et al., 2023).

    Note: This is a simple implementation suitable for small experiments.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError("Invalid learning rate for Lion: lr must be > 0")
        if not (0.0 <= weight_decay):
            raise ValueError("Invalid weight_decay for Lion: must be >= 0")
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @overload
    def step(self, closure: None = ...) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None):
        loss: float | None = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr: float = group["lr"]
            beta1, _beta2 = group["betas"]
            weight_decay: float = group.get("weight_decay", 0.0)
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                m = state["exp_avg"]
                # Decoupled weight decay
                if weight_decay != 0.0:
                    p.add_(p, alpha=-lr * weight_decay)
                # First-moment update
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                p.add_(m.sign(), alpha=-lr)
        return loss


def _newton_schulz_orthogonalize(
    g: Tensor,
    ns_steps: int = 5,
    ns_coeffs: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
    eps: float = 1e-7,
) -> Tensor:
    """Newton-Schulz iteration to approximate the zeroth power (sign) of a matrix.

    Orthogonalizes the gradient so the update has uniform singular values,
    which provides learning-rate transfer across weight shapes.
    """
    a, b, c = ns_coeffs
    # Ensure tall orientation: (rows >= cols)
    transposed = False
    if g.shape[0] < g.shape[1]:
        g = g.T
        transposed = True

    # Normalize to unit spectral norm (approximate)
    g_f = g.to(dtype=torch.float32)
    norm = g_f.norm() + eps
    g_f = g_f / norm

    x = g_f
    for _ in range(ns_steps):
        xtx = x.T @ x
        x = a * x + b * (x @ xtx) + c * (x @ (xtx @ xtx))

    if transposed:
        x = x.T
    return cast(Tensor, x.to(dtype=g.dtype))


class Muon(Optimizer):
    """Muon optimizer: Newton-Schulz orthogonalization on momentum updates.

    Applies orthogonalized updates to 2D weight matrices. Non-2D parameters
    (biases, norms, embeddings) use an AdamW-style fallback update so mixed
    parameter shapes are handled safely in a single optimizer.

    Reference: Keller Jordan, "Muon: An optimizer for hidden layers"
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        ns_coeffs: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
        eps: float = 1e-7,
        fallback_beta2: float = 0.99,
        fallback_eps: float = 1e-8,
    ) -> None:
        if lr <= 0.0:
            raise ValueError("Invalid learning rate for Muon: lr must be > 0")
        if not (0.0 <= momentum < 1.0):
            raise ValueError("Invalid momentum for Muon: must be in [0, 1)")
        if not (0.0 <= fallback_beta2 < 1.0):
            raise ValueError("Invalid fallback_beta2 for Muon: must be in [0, 1)")
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "nesterov": nesterov,
            "weight_decay": weight_decay,
            "ns_steps": ns_steps,
            "ns_coeffs": ns_coeffs,
            "eps": eps,
            "fallback_beta2": fallback_beta2,
            "fallback_eps": fallback_eps,
        }
        super().__init__(params, defaults)

    @overload
    def step(self, closure: None = ...) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None):
        loss: float | None = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            momentum: float = group["momentum"]
            nesterov: bool = group["nesterov"]
            weight_decay: float = group.get("weight_decay", 0.0)
            ns_steps: int = group.get("ns_steps", 5)
            ns_coeffs: tuple[float, float, float] = group.get(
                "ns_coeffs", (3.4445, -4.7750, 2.0315)
            )
            eps: float = group.get("eps", 1e-7)
            fallback_beta2: float = group.get("fallback_beta2", 0.99)
            fallback_eps: float = group.get("fallback_eps", 1e-8)

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                # Muon only applies to 2D weight matrices
                if p.ndim != 2:
                    # Fallback: AdamW-style update for non-matrix params.
                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step"] = int(state["step"]) + 1
                    step = int(state["step"])
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg.mul_(momentum).add_(grad, alpha=1.0 - momentum)
                    exp_avg_sq.mul_(fallback_beta2).addcmul_(grad, grad, value=1.0 - fallback_beta2)
                    if weight_decay != 0.0:
                        p.add_(p, alpha=-lr * weight_decay)
                    bias_c1 = 1.0 - momentum**step
                    bias_c2 = 1.0 - fallback_beta2**step
                    denom = (
                        exp_avg_sq.sqrt().div_(math.sqrt(max(bias_c2, 1e-16))).add_(fallback_eps)
                    )
                    step_size = lr / max(bias_c1, 1e-16)
                    p.addcdiv_(exp_avg, denom, value=-step_size)
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]

                # Momentum update
                buf.mul_(momentum).add_(grad)

                # Nesterov lookahead
                if nesterov:
                    update = grad + momentum * buf
                else:
                    update = buf

                # Newton-Schulz orthogonalization
                ortho = _newton_schulz_orthogonalize(update, ns_steps, ns_coeffs, eps)

                # Decoupled weight decay
                if weight_decay != 0.0:
                    p.add_(p, alpha=-lr * weight_decay)

                # LR adjustment for rectangular matrices (Keller's original)
                rows, cols = p.shape
                scale = math.sqrt(max(1, rows / cols))

                p.add_(ortho, alpha=-lr * scale)
        return loss


def build_optimizer(params: Iterable[torch.nn.Parameter], schedule: TrainSchedule) -> Optimizer:
    cfg: OptimizerConfig = getattr(schedule, "optimizer", OptimizerConfig())
    name = (cfg.name or "adamw").lower()
    # Effective hparams: optimizer overrides or fall back to TrainSchedule
    lr = float(cfg.lr if cfg.lr is not None else schedule.lr)
    weight_decay = float(
        cfg.weight_decay if cfg.weight_decay is not None else schedule.weight_decay
    )
    if name == "adamw":
        betas = cfg.betas if cfg.betas is not None else (0.9, 0.999)
        eps = cfg.eps if cfg.eps is not None else 1e-8
        return AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    if name == "lion":
        betas = cfg.betas if cfg.betas is not None else (0.9, 0.99)
        return Lion(params, lr=lr, betas=betas, weight_decay=weight_decay)
    if name == "muon":
        momentum = float(cfg.muon_momentum if cfg.muon_momentum is not None else 0.95)
        nesterov = bool(cfg.muon_nesterov)
        ns_steps = int(cfg.muon_ns_steps) if cfg.muon_ns_steps else 5
        ns_eps = float(cfg.eps if cfg.eps is not None else 1e-7)
        fallback_eps = float(cfg.eps if cfg.eps is not None else 1e-8)
        fallback_beta2 = (
            float(cfg.betas[1]) if cfg.betas is not None and len(cfg.betas) >= 2 else 0.99
        )
        return Muon(
            params,
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            eps=ns_eps,
            fallback_beta2=fallback_beta2,
            fallback_eps=fallback_eps,
        )
    # Fallback
    raise ValueError(f"Unsupported optimizer: {name}")
