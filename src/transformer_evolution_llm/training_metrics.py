"""Training metrics: speedrun AUC, learning curve fitting, router entropy."""

from __future__ import annotations

import math

import torch
from torch import nn


def compute_speedrun_auc(thresholds: list[float], tokens_to_threshold: list[float | None]) -> float:
    """Compute area under the tokens-to-threshold curve.

    Uses trapezoidal integration over thresholds where we reached the target.
    Lower AUC means faster training (reached thresholds with fewer tokens).

    Args:
        thresholds: Perplexity thresholds (should be sorted descending).
        tokens_to_threshold: Tokens to reach each threshold (None if not reached).

    Returns:
        AUC value. If fewer than 2 thresholds were reached, returns infinity.
    """
    # Filter to thresholds we actually reached
    valid_pairs = [
        (t, tok)
        for t, tok in zip(thresholds, tokens_to_threshold, strict=False)
        if tok is not None and math.isfinite(tok)
    ]

    if len(valid_pairs) < 2:
        return float("inf")

    # Sort by threshold (descending) for proper integration
    valid_pairs.sort(key=lambda x: x[0], reverse=True)

    # Trapezoidal integration
    auc = 0.0
    for i in range(1, len(valid_pairs)):
        t_prev, tok_prev = valid_pairs[i - 1]
        t_curr, tok_curr = valid_pairs[i]
        dt = abs(t_prev - t_curr)
        avg_tok = (tok_prev + tok_curr) / 2
        auc += dt * avg_tok

    return auc


def fit_learning_curve(
    steps: list[int], losses: list[float], target_steps: int | None = None
) -> dict[str, float]:
    """Fit a power-law learning curve to predict long-term performance.

    Uses the form: loss(t) = a * t^(-b) + c
    where a, b, c are fitted parameters.

    Args:
        steps: Training step indices.
        losses: Corresponding loss values.
        target_steps: Optional target step count for extrapolation.

    Returns:
        Dictionary with:
        - a, b, c: Fitted parameters
        - r_squared: Fit quality (0-1)
        - extrapolated_loss: Predicted loss at target_steps (if provided)
        - extrapolated_ppl: Predicted perplexity at target_steps (if provided)
        - fit_error: 1.0 if fitting failed, 0.0 otherwise
    """
    result: dict[str, float] = {
        "learning_curve_a": 0.0,
        "learning_curve_b": 0.0,
        "learning_curve_c": 0.0,
        "learning_curve_r_squared": 0.0,
        "fit_error": 0.0,
    }

    if len(steps) < 3 or len(losses) < 3:
        result["fit_error"] = 1.0
        return result

    # Filter out invalid values
    valid_pairs = [
        (step, loss)
        for step, loss in zip(steps, losses, strict=False)
        if step > 0 and math.isfinite(loss) and loss > 0
    ]
    if len(valid_pairs) < 3:
        result["fit_error"] = 1.0
        return result

    steps_arr = [p[0] for p in valid_pairs]
    losses_arr = [p[1] for p in valid_pairs]

    try:
        # Simple power-law fit using log-linear regression
        # log(loss - c) = log(a) - b * log(t)
        # We estimate c as the minimum observed loss (floor estimate)
        c_est = min(losses_arr) * 0.9  # Slightly below minimum

        # Shift losses
        shifted = [max(loss - c_est, 1e-9) for loss in losses_arr]

        # Log-transform
        log_t = [math.log(s) for s in steps_arr]
        log_loss = [math.log(loss) for loss in shifted]

        # Linear regression: log_l = log_a - b * log_t
        n = len(log_t)
        sum_x = sum(log_t)
        sum_y = sum(log_loss)
        sum_xy = sum(x * y for x, y in zip(log_t, log_loss, strict=False))
        sum_x2 = sum(x * x for x in log_t)

        denom = n * sum_x2 - sum_x * sum_x
        if abs(denom) < 1e-12:
            result["fit_error"] = 1.0
            return result

        b = -(n * sum_xy - sum_x * sum_y) / denom
        log_a = (sum_y + b * sum_x) / n
        a = math.exp(log_a)
        c = c_est

        # Compute R-squared
        mean_y = sum_y / n
        ss_tot = sum((y - mean_y) ** 2 for y in log_loss)
        predicted = [log_a - b * x for x in log_t]
        ss_res = sum((y - p) ** 2 for y, p in zip(log_loss, predicted, strict=False))
        r_squared = 1.0 - (ss_res / max(ss_tot, 1e-12)) if ss_tot > 0 else 0.0

        result["learning_curve_a"] = float(a)
        result["learning_curve_b"] = float(b)
        result["learning_curve_c"] = float(c)
        result["learning_curve_r_squared"] = max(0.0, min(1.0, float(r_squared)))

        # Extrapolate to target if provided
        if target_steps is not None and target_steps > 0 and b > 0:
            extrapolated = a * (target_steps ** (-b)) + c
            result["extrapolated_loss"] = float(extrapolated)
            result["extrapolated_ppl"] = float(math.exp(extrapolated))

    except (ValueError, OverflowError, ZeroDivisionError):
        result["fit_error"] = 1.0

    return result


def _average_router_entropy(model: nn.Module) -> float | None:
    with torch.no_grad():
        entropies = []
        for mod in model.modules():
            if hasattr(mod, "last_entropy"):
                last = getattr(mod, "last_entropy", None)
                if isinstance(last, torch.Tensor):
                    entropies.append(float(last.item()))
    if not entropies:
        return None
    return sum(entropies) / len(entropies)
