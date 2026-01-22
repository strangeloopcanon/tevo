"""Registration hooks for plugin-style architectural components.

This module provides a minimal, opt-in registry so external packages can
attach custom blocks (e.g., attention variants, adapters) without
editing the core model files. Registration happens via simple Python
imports and does not affect behavior unless a matching plugin name is
present in the config.
"""

from __future__ import annotations

from typing import Protocol

from .dsl import CustomModuleConfig


class ComponentBuilder(Protocol):
    """Callable that builds a module for a custom component."""

    def __call__(self, cfg: CustomModuleConfig, dim: int): ...


_REGISTRY: dict[str, ComponentBuilder] = {}


def register_component(name: str, builder: ComponentBuilder) -> None:
    """Register a custom architectural component builder.

    Parameters
    ----------
    name:
        Identifier used in ``CustomModuleConfig.name``.
    builder:
        Callable that takes the component config and model dimension and
        returns an ``nn.Module`` instance.
    """
    _REGISTRY[name] = builder


def get_component(name: str) -> ComponentBuilder | None:
    """Return the registered builder for ``name``, if any."""
    return _REGISTRY.get(name)


def list_components() -> list[str]:
    """Return the list of registered component names."""
    return sorted(_REGISTRY)


def _register_builtin_components() -> None:
    def _build_graph_module(cfg: CustomModuleConfig, dim: int):
        import torch
        from torch import nn

        class _RMSNorm(nn.Module):
            def __init__(self, d: int, eps: float = 1e-6) -> None:
                super().__init__()
                self.eps = float(eps)
                self.weight = nn.Parameter(torch.ones(d, dtype=torch.float32))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x_f = x.to(dtype=torch.float32)
                rms = x_f.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
                y = x_f / rms * self.weight.view(1, 1, -1)
                return y.to(dtype=x.dtype)

        class _ScalarGate(nn.Module):
            def __init__(self, init: float = 0.1) -> None:
                super().__init__()
                init = max(1e-6, min(1.0 - 1e-6, float(init)))
                logit = torch.log(torch.tensor(init / (1.0 - init), dtype=torch.float32))
                self.logit = nn.Parameter(logit)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                gate = torch.sigmoid(self.logit).to(dtype=x.dtype, device=x.device)
                return x * gate

        ops = cfg.params.get("ops", [])
        if not isinstance(ops, list) or not ops:
            return nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

        modules: list[nn.Module] = []
        for entry in ops:
            if not isinstance(entry, dict):
                continue
            op = str(entry.get("op") or "").lower()
            if op in {"rmsnorm", "norm"}:
                modules.append(_RMSNorm(dim, eps=float(entry.get("eps", 1e-6))))
            elif op in {"mlp"}:
                mult = float(entry.get("hidden_mult", 2.0))
                hidden = max(1, int(round(dim * mult)))
                modules.append(
                    nn.Sequential(
                        nn.Linear(dim, hidden),
                        nn.SiLU(),
                        nn.Linear(hidden, dim),
                    )
                )
            elif op in {"gate"}:
                modules.append(_ScalarGate(init=float(entry.get("init", 0.1))))
            elif op in {"identity"}:
                modules.append(nn.Identity())

        if not modules:
            return nn.Identity()
        return nn.Sequential(*modules)

    register_component("graph_module", _build_graph_module)


_register_builtin_components()
