"""Optimizer registry (AdamW built-in + optional Lion).

Optimizer configuration lives in the DSL. Evolution can optionally mutate it via
registered mutations.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import overload

import torch
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
    # Fallback
    raise ValueError(f"Unsupported optimizer: {name}")
