#!/usr/bin/env python
"""Gate sensitivity analysis: sweep over structural gates to understand their effects.

This script runs short evolution sweeps with different gate configurations
and reports how the frontier composition changes.

Usage:
    python scripts/sweep_gates.py configs/base_config.yaml --generations 30 --device mps

Example output:
    | min_layers | min_moe_blocks | frontier_size | avg_layers | avg_moe | avg_ppl |
    |------------|----------------|---------------|------------|---------|---------|
    | 4          | 0              | 8             | 5.2        | 1.3     | 145.2   |
    | 4          | 2              | 6             | 6.1        | 3.2     | 138.7   |
    | 8          | 0              | 5             | 9.3        | 0.8     | 152.1   |
    ...
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformer_evolution_llm.dsl import ArchitectureSpec, load_architecture_spec
from transformer_evolution_llm.orchestrator import EvolutionRunner


def run_sweep(
    base_config_path: str,
    generations: int = 30,
    device: str = "cpu",
    steps: int = 50,
    eval_batches: int = 2,
    seed: int = 0,
    output_dir: str | None = None,
) -> list[dict[str, Any]]:
    """Run gate sensitivity sweep over structural thresholds.

    Args:
        base_config_path: Path to base YAML config.
        generations: Number of generations per sweep.
        device: Device to run on.
        steps: Training steps per candidate.
        eval_batches: Eval batches per candidate.
        seed: Random seed.
        output_dir: Optional directory to save results.

    Returns:
        List of sweep results, one per configuration.
    """
    base_spec = load_architecture_spec(base_config_path)

    # Define gate sweep ranges
    min_layers_range = [0, 4, 6, 8]
    min_moe_blocks_range = [0, 1, 2, 3]
    min_selector_blocks_range = [0, 1, 2]

    results: list[dict[str, Any]] = []

    print(f"Running gate sensitivity sweep with {generations} generations per config...")
    print(f"Device: {device}, Steps: {steps}, Seed: {seed}")
    print("-" * 80)

    total_configs = len(min_layers_range) * len(min_moe_blocks_range) * len(min_selector_blocks_range)
    config_idx = 0

    for min_layers, min_moe, min_selector in product(
        min_layers_range, min_moe_blocks_range, min_selector_blocks_range
    ):
        config_idx += 1
        print(f"\n[{config_idx}/{total_configs}] Running: min_layers={min_layers}, "
              f"min_moe_blocks={min_moe}, min_selector_blocks={min_selector}")

        # Create modified spec with updated thresholds
        spec_dict = base_spec.model_dump(mode="python")

        # Update rung0 thresholds
        if "evolution" not in spec_dict:
            spec_dict["evolution"] = {}
        if "rung0_thresholds" not in spec_dict["evolution"]:
            spec_dict["evolution"]["rung0_thresholds"] = {}

        spec_dict["evolution"]["rung0_thresholds"]["min_layers"] = float(min_layers)
        spec_dict["evolution"]["rung0_thresholds"]["min_moe_blocks"] = float(min_moe)
        spec_dict["evolution"]["rung0_thresholds"]["min_selector_blocks"] = float(min_selector)

        try:
            spec = ArchitectureSpec(**spec_dict)
        except Exception as e:
            print(f"  [ERROR] Invalid config: {e}")
            continue

        # Run evolution
        try:
            runner = EvolutionRunner(
                base_spec=spec,
                evolution_cfg=spec.evolution,
                mode="simulate",  # Use simulate mode for speed
                seed=seed,
            )
            survivors = runner.run(generations)

            # Collect frontier statistics
            frontier_entries = runner.frontier.entries
            frontier_size = len(frontier_entries)

            if frontier_size > 0:
                avg_layers = sum(
                    c.metrics.get("layers", c.spec.model.n_layers) for c in frontier_entries
                ) / frontier_size
                avg_moe = sum(
                    c.metrics.get("moe_blocks", 0) for c in frontier_entries
                ) / frontier_size
                avg_selector = sum(
                    c.metrics.get("selector_blocks", 0) for c in frontier_entries
                ) / frontier_size
                avg_memory = sum(
                    c.metrics.get("memory_blocks", 0) for c in frontier_entries
                ) / frontier_size
                avg_ppl = sum(
                    c.metrics.get("ppl_code", 1e9) for c in frontier_entries
                ) / frontier_size
            else:
                avg_layers = avg_moe = avg_selector = avg_memory = 0.0
                avg_ppl = float("inf")

            result = {
                "min_layers": min_layers,
                "min_moe_blocks": min_moe,
                "min_selector_blocks": min_selector,
                "frontier_size": frontier_size,
                "avg_layers": round(avg_layers, 2),
                "avg_moe_blocks": round(avg_moe, 2),
                "avg_selector_blocks": round(avg_selector, 2),
                "avg_memory_blocks": round(avg_memory, 2),
                "avg_ppl_code": round(avg_ppl, 2) if avg_ppl < 1e8 else "inf",
                "total_survivors": len(survivors),
            }
            results.append(result)

            print(f"  Frontier: {frontier_size} entries, avg_layers={avg_layers:.1f}, "
                  f"avg_moe={avg_moe:.1f}, avg_ppl={avg_ppl:.1f}")

        except Exception as e:
            print(f"  [ERROR] Evolution failed: {e}")
            results.append({
                "min_layers": min_layers,
                "min_moe_blocks": min_moe,
                "min_selector_blocks": min_selector,
                "error": str(e),
            })

    # Print summary table
    print("\n" + "=" * 80)
    print("GATE SENSITIVITY ANALYSIS RESULTS")
    print("=" * 80)
    print(f"{'min_layers':>10} | {'min_moe':>8} | {'min_sel':>8} | {'frontier':>8} | "
          f"{'avg_layers':>10} | {'avg_moe':>8} | {'avg_ppl':>10}")
    print("-" * 80)

    for r in results:
        if "error" in r:
            print(f"{r['min_layers']:>10} | {r['min_moe_blocks']:>8} | "
                  f"{r['min_selector_blocks']:>8} | ERROR: {r['error'][:30]}")
        else:
            print(f"{r['min_layers']:>10} | {r['min_moe_blocks']:>8} | "
                  f"{r['min_selector_blocks']:>8} | {r['frontier_size']:>8} | "
                  f"{r['avg_layers']:>10} | {r['avg_moe_blocks']:>8} | "
                  f"{r['avg_ppl_code']:>10}")

    # Save results if output dir specified
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = out_path / f"gate_sweep_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump({
                "config": base_config_path,
                "generations": generations,
                "device": device,
                "steps": steps,
                "seed": seed,
                "results": results,
            }, f, indent=2)
        print(f"\nResults saved to: {results_file}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run gate sensitivity analysis over structural thresholds."
    )
    parser.add_argument("config", help="Path to base YAML config")
    parser.add_argument("--generations", type=int, default=30, help="Generations per config")
    parser.add_argument("--device", default="cpu", help="Device (cpu, cuda, mps)")
    parser.add_argument("--steps", type=int, default=50, help="Training steps per candidate")
    parser.add_argument("--eval-batches", type=int, default=2, help="Eval batches")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--output-dir", help="Directory to save results")

    args = parser.parse_args()

    run_sweep(
        base_config_path=args.config,
        generations=args.generations,
        device=args.device,
        steps=args.steps,
        eval_batches=args.eval_batches,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
