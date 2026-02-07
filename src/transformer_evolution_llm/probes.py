"""Capability probes: needle-in-haystack variants and downstream task probes."""

from __future__ import annotations

import torch
from torch import nn

from .dsl import ArchitectureSpec


def run_multi_needle_probes(
    model: nn.Module,
    spec: ArchitectureSpec,
    device: torch.device,
    probe_types: list[str] | None = None,
) -> dict[str, float]:
    """Run multiple needle-in-haystack probe variants.

    Probe types:
    - "first": Retrieve value placed at the first position
    - "middle": Retrieve value placed in the middle
    - "last": Retrieve value placed near the end (but before query)
    - "pattern": Find a repeated pattern in the context

    Returns:
        Dictionary of probe_type -> accuracy for each probe.
    """
    if probe_types is None:
        probe_types = ["first", "middle", "last", "pattern"]

    results: dict[str, float] = {}
    vocab = int(spec.model.head.vocab)
    seq_len = int(spec.data.seq_len)
    batch_size = max(1, int(spec.data.batch_size))

    if vocab < 10 or seq_len < 16:
        return {f"needle_{pt}_acc": 0.0 for pt in probe_types}

    seed_val = int(getattr(spec.train, "seed", 0) or 0)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed_val + 42)

    query_marker = vocab - 2
    noise_vocab = max(1, vocab - 3)
    eval_batches = 4

    was_training = bool(getattr(model, "training", False))

    try:
        model.eval()
        with torch.no_grad():
            for probe_type in probe_types:
                correct = 0
                total = 0

                for _ in range(eval_batches):
                    ids = torch.randint(
                        0, noise_vocab, (batch_size, seq_len), generator=generator, dtype=torch.long
                    )
                    target = torch.randint(
                        0, noise_vocab, (batch_size,), generator=generator, dtype=torch.long
                    )

                    if probe_type == "first":
                        # Place target at position 0
                        ids[:, 0] = target
                    elif probe_type == "middle":
                        # Place target in the middle
                        mid_pos = seq_len // 2
                        ids[:, mid_pos] = target
                    elif probe_type == "last":
                        # Place target near the end (3 positions before query)
                        ids[:, -4] = target
                    elif probe_type == "pattern":
                        # Repeat target 3 times in early context
                        pattern_positions = [1, 3, 5]
                        for pos in pattern_positions:
                            if pos < seq_len - 2:
                                ids[:, pos] = target
                    else:
                        continue

                    # Add query marker and target at end
                    ids[:, -2] = int(query_marker)
                    ids[:, -1] = target

                    input_ids = ids.to(device)
                    logits = model(input_ids)

                    if logits.size(1) < 2:
                        continue

                    # Predict from position -2 (after query marker)
                    pred = logits[:, -2, :noise_vocab].argmax(dim=-1)
                    correct += int((pred == target.to(device)).sum().item())
                    total += int(target.numel())

                acc = float(correct) / float(max(1, total))
                results[f"needle_{probe_type}_acc"] = acc

    finally:
        if was_training:
            model.train()

    # Compute aggregate needle score
    accs = [v for k, v in results.items() if k.endswith("_acc")]
    results["needle_avg_acc"] = sum(accs) / len(accs) if accs else 0.0

    return results


def run_downstream_probes(
    model: nn.Module,
    spec: ArchitectureSpec,
    device: torch.device,
    n_examples: int = 5,
) -> dict[str, float]:
    """Run simple downstream probes to assess capability beyond perplexity.

    Probes:
    - code_probe: Simple code completion (predict closing bracket/token)
    - math_probe: Basic arithmetic pattern completion
    - recall_probe: Factual pattern recall (given context, predict answer)

    These are synthetic tasks designed to run quickly and provide a rough
    signal about model capability without requiring external datasets.

    Returns:
        Dictionary with probe accuracies.
    """
    results: dict[str, float] = {}
    vocab = int(spec.model.head.vocab)
    seq_len = min(256, int(spec.data.seq_len))
    batch_size = 1

    if vocab < 50:
        return {"code_probe_acc": 0.0, "math_probe_acc": 0.0, "recall_probe_acc": 0.0}

    seed_val = int(getattr(spec.train, "seed", 0) or 0)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed_val + 123)

    was_training = bool(getattr(model, "training", False))

    try:
        model.eval()

        # Probe 1: Code completion (bracket matching)
        # Pattern: context tokens + open bracket -> predict close bracket
        code_correct = 0
        code_total = 0
        # Use simple token patterns for synthetic code
        open_token = min(40, vocab - 1)  # Represents '('
        close_token = min(41, vocab - 1)  # Represents ')'

        with torch.no_grad():
            for _ in range(n_examples):
                # Create pattern: [noise] [open] [noise] -> predict [close]
                ids = torch.randint(0, min(100, vocab), (batch_size, seq_len), dtype=torch.long)
                ids[:, seq_len // 2] = open_token
                ids[:, -1] = close_token

                input_ids = ids[:, :-1].to(device)
                target = int(close_token)

                logits = model(input_ids)
                if logits.size(1) > 0:
                    pred = logits[:, -1, :].argmax(dim=-1).item()
                    if pred == target:
                        code_correct += 1
                    code_total += 1

        results["code_probe_acc"] = float(code_correct) / max(1, code_total)

        # Probe 2: Math pattern (arithmetic sequence)
        # Pattern: A + B = [answer] where answer = A + B (mod vocab)
        math_correct = 0
        math_total = 0
        plus_token = min(43, vocab - 1)  # '+'
        eq_token = min(61, vocab - 1)  # '='

        with torch.no_grad():
            for _ in range(n_examples):
                a = int(torch.randint(1, min(50, vocab // 4), (1,), generator=generator).item())
                b = int(torch.randint(1, min(50, vocab // 4), (1,), generator=generator).item())
                answer = int((a + b) % max(1, vocab // 2))

                # Create: [noise] A + B = [answer]
                ids = torch.randint(0, min(100, vocab), (batch_size, seq_len), dtype=torch.long)
                ids[:, -5] = a
                ids[:, -4] = plus_token
                ids[:, -3] = b
                ids[:, -2] = eq_token
                ids[:, -1] = answer

                input_ids = ids[:, :-1].to(device)
                target = int(answer)

                logits = model(input_ids)
                if logits.size(1) > 0:
                    pred = logits[:, -1, :].argmax(dim=-1).item()
                    if pred == target:
                        math_correct += 1
                    math_total += 1

        results["math_probe_acc"] = float(math_correct) / max(1, math_total)

        # Probe 3: Recall (repeat a value from earlier in context)
        # Pattern: [marker] [value] ... [query_marker] -> [value]
        recall_correct = 0
        recall_total = 0
        marker_token = min(vocab - 3, max(0, vocab - 3))
        query_token = min(vocab - 2, max(0, vocab - 2))

        with torch.no_grad():
            for _ in range(n_examples):
                value = int(torch.randint(0, min(100, vocab), (1,), generator=generator).item())

                # Create: [marker] [value] [noise] [query] -> [value]
                ids = torch.randint(0, min(100, vocab), (batch_size, seq_len), dtype=torch.long)
                ids[:, 0] = marker_token
                ids[:, 1] = value
                ids[:, -2] = query_token
                ids[:, -1] = value

                input_ids = ids[:, :-1].to(device)
                target = int(value)

                logits = model(input_ids)
                if logits.size(1) > 0:
                    pred = logits[:, -1, :].argmax(dim=-1).item()
                    if pred == target:
                        recall_correct += 1
                    recall_total += 1

        results["recall_probe_acc"] = float(recall_correct) / max(1, recall_total)

        # Compute aggregate
        accs = [
            results.get("code_probe_acc", 0),
            results.get("math_probe_acc", 0),
            results.get("recall_probe_acc", 0),
        ]
        results["downstream_avg_acc"] = sum(accs) / len(accs)

    except Exception:
        results = {
            "code_probe_acc": 0.0,
            "math_probe_acc": 0.0,
            "recall_probe_acc": 0.0,
            "downstream_avg_acc": 0.0,
            "downstream_error": 1.0,
        }
    finally:
        if was_training:
            model.train()

    return results


def _estimate_long_recall(spec: ArchitectureSpec) -> float:
    layers = max(1, spec.model.n_layers)
    memory_blocks = sum(
        1
        for block in spec.model.blocks
        for extra in block.extras
        if getattr(extra, "type", None)
        in {"retro", "assoc_memory", "memory_tokens", "chunk_memory", "lookup_memory"}
    )
    ssm_blocks = sum(1 for block in spec.model.blocks if block.ssm is not None)
    rec_spans = len(spec.model.recurrences)
    extra_types = {
        getattr(extra, "type", type(extra).__name__)
        for block in spec.model.blocks
        for extra in block.extras
    }
    density = (memory_blocks + 0.5 * ssm_blocks + 0.5 * rec_spans) / layers
    diversity_bonus = 0.1 * len(extra_types)
    return float(min(2.0, density + diversity_bonus))
