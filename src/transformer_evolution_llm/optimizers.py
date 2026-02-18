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


def _momentum_proxy_tensor(state: dict[object, object], grad: Tensor) -> Tensor | None:
    for key in ("momentum_buffer", "exp_avg"):
        value = state.get(key)
        if isinstance(value, torch.Tensor) and value.shape == grad.shape:
            return value
    return None


def _topk_mask(values: Tensor, keep_ratio: float) -> Tensor:
    flat = values.reshape(-1)
    numel = int(flat.numel())
    if numel <= 0 or keep_ratio >= 1.0:
        return torch.ones_like(values, dtype=torch.bool)
    k = max(1, min(numel, int(round(float(keep_ratio) * numel))))
    if k >= numel:
        return torch.ones_like(values, dtype=torch.bool)
    keep_idx = torch.topk(flat, k=k, largest=True, sorted=False).indices
    # Guard against occasional backend index glitches (observed on MPS) where
    # topk may include `numel` as an index, which is out of bounds.
    if keep_idx.numel() > 0:
        keep_idx = keep_idx.to(dtype=torch.long).clamp(0, numel - 1)
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask[keep_idx] = True
    return mask.reshape_as(values)


def _block_mask(values: Tensor, *, mode: str, keep_ratio: float, block_size: int) -> Tensor:
    flat = values.reshape(-1)
    numel = int(flat.numel())
    if numel <= 0:
        return torch.ones_like(values, dtype=torch.bool)
    size = max(1, int(block_size))
    n_blocks = (numel + size - 1) // size
    if n_blocks <= 1 or keep_ratio >= 1.0:
        return torch.ones_like(values, dtype=torch.bool)
    if mode == "bernoulli":
        block_keep = torch.rand(n_blocks, device=flat.device) < float(keep_ratio)
    else:
        block_scores = torch.zeros(n_blocks, dtype=torch.float32, device=flat.device)
        for idx in range(n_blocks):
            start = idx * size
            end = min(start + size, numel)
            block_scores[idx] = flat[start:end].mean()
        block_keep = _topk_mask(block_scores, keep_ratio).reshape(-1)
    expanded = block_keep.repeat_interleave(size)[:numel]
    return expanded.reshape_as(values)


@torch.no_grad()
def apply_gradient_transform_(optimizer: Optimizer, schedule: TrainSchedule) -> float:
    """Apply configured gradient transforms in-place before optimizer.step.

    Returns the observed fraction of gradient elements transformed.
    """
    cfg: OptimizerConfig = getattr(schedule, "optimizer", OptimizerConfig())
    transform_cfg = getattr(cfg, "gradient_transform", None)
    if transform_cfg is None:
        return 0.0

    mode = str(getattr(transform_cfg, "mode", "identity") or "identity").lower()
    if mode == "identity":
        return 0.0

    ns_steps = int(getattr(transform_cfg, "ns_steps", 5) or 5)
    eps = float(getattr(transform_cfg, "eps", 1e-8) or 1e-8)
    ns_steps = max(1, ns_steps)
    eps = max(1e-12, eps)

    total_elements = 0
    transformed_elements = 0

    for group in optimizer.param_groups:
        for param in group["params"]:
            grad = param.grad
            if grad is None:
                continue

            count = int(grad.numel())
            if count <= 0:
                continue
            total_elements += count

            if mode == "sign":
                grad.copy_(grad.sign())
                transformed_elements += count
                continue

            if mode == "normalize":
                norm = float(grad.norm().item())
                if math.isfinite(norm) and norm > 0.0:
                    grad.div_(norm + eps)
                transformed_elements += count
                continue

            if mode == "orthogonalize_2d":
                if grad.ndim == 2:
                    transformed = _newton_schulz_orthogonalize(grad, ns_steps=ns_steps, eps=eps)
                    if torch.isfinite(transformed).all():
                        grad.copy_(transformed)
                    transformed_elements += count
                continue

            if mode == "sign_orthogonalize_2d":
                if grad.ndim == 2:
                    transformed = _newton_schulz_orthogonalize(grad, ns_steps=ns_steps, eps=eps)
                    if torch.isfinite(transformed).all():
                        grad.copy_(transformed.sign())
                    else:
                        grad.copy_(grad.sign())
                else:
                    grad.copy_(grad.sign())
                transformed_elements += count
                continue

    if total_elements <= 0:
        return 0.0
    return float(transformed_elements) / float(total_elements)


@torch.no_grad()
def apply_update_filter_(optimizer: Optimizer, schedule: TrainSchedule) -> float:
    """Mask gradients before optimizer.step according to the configured policy.

    Returns the observed keep fraction across all gradient elements.
    """
    cfg: OptimizerConfig = getattr(schedule, "optimizer", OptimizerConfig())
    filter_cfg = getattr(cfg, "update_filter", None)
    if filter_cfg is None:
        return 1.0

    mode = str(getattr(filter_cfg, "mode", "none") or "none").lower()
    keep_ratio = float(getattr(filter_cfg, "keep_ratio", 1.0) or 1.0)
    if mode == "none" or keep_ratio >= 1.0:
        return 1.0

    granularity = str(getattr(filter_cfg, "granularity", "element") or "element").lower()
    block_size = int(getattr(filter_cfg, "block_size", 128) or 128)
    momentum_blend = float(getattr(filter_cfg, "momentum_blend", 0.0) or 0.0)
    rescale_kept = bool(getattr(filter_cfg, "rescale_kept", True))
    momentum_blend = max(0.0, min(1.0, momentum_blend))
    keep_ratio = max(0.0, min(1.0, keep_ratio))

    total_elements = 0
    kept_elements = 0

    for group in optimizer.param_groups:
        for param in group["params"]:
            grad = param.grad
            if grad is None:
                continue
            grad_score = grad.detach().abs().to(dtype=torch.float32)
            if momentum_blend > 0.0:
                proxy = _momentum_proxy_tensor(optimizer.state[param], grad)
                if proxy is not None:
                    proxy_score = proxy.detach().abs().to(dtype=torch.float32)
                    grad_score = (1.0 - momentum_blend) * grad_score + momentum_blend * proxy_score

            if granularity == "block":
                mask = _block_mask(
                    grad_score,
                    mode=mode,
                    keep_ratio=keep_ratio,
                    block_size=block_size,
                )
            elif mode == "bernoulli":
                mask = torch.rand_like(grad_score, dtype=torch.float32) < keep_ratio
            else:
                mask = _topk_mask(grad_score, keep_ratio)

            count = int(mask.numel())
            keep = int(mask.sum().item())
            total_elements += count
            kept_elements += keep

            if keep == count:
                continue
            if keep <= 0:
                grad.zero_()
                continue
            keep_frac = float(keep) / float(max(1, count))
            grad.mul_(mask.to(dtype=grad.dtype))
            if rescale_kept and keep_frac > 0.0:
                grad.div_(keep_frac)

    if total_elements <= 0:
        return 1.0
    return float(kept_elements) / float(total_elements)


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
