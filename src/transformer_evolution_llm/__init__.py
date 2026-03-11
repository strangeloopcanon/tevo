"""
TEVO
====

Utilities for describing, mutating, and evaluating transformer architectures
on resource-constrained hardware.
"""

from importlib.metadata import PackageNotFoundError, version


def get_version() -> str:
    """Return the installed package version or '0.0.0' when unavailable."""
    for dist_name in ("tevo", "transformer-evolution-llm", "transformer_evolution_llm"):
        try:
            return version(dist_name)
        except PackageNotFoundError:
            continue
    return "0.0.0"


__all__ = ["get_version"]
