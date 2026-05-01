"""
Competition metric: extract answer from model output and score against ground truth.
Mirrors the official NVIDIA Nemotron evaluation logic.
"""

import re
import math


def extract_answer(text: str) -> str | None:
    # Priority 1: last \boxed{...} in the response
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        return boxed[-1].strip()

    # Priority 2: "the answer is <value>" pattern
    m = re.search(r"(?i)the answer is[:\s]+([^\n.,]+)", text)
    if m:
        return m.group(1).strip()

    # Priority 3: last standalone number
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if nums:
        return nums[-1]

    return None


def _is_numeric(s: str) -> bool:
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def score_prediction(prediction: str | None, ground_truth: str, tol: float = 1e-6) -> bool:
    if prediction is None:
        return False

    pred = prediction.strip()
    truth = ground_truth.strip()

    # Exact string match
    if pred == truth:
        return True

    # Numeric match within relative tolerance
    if _is_numeric(pred) and _is_numeric(truth):
        p, t = float(pred), float(truth)
        if t == 0:
            return abs(p) < tol
        return abs(p - t) / abs(t) < tol

    return False


def evaluate(predictions: list[str | None], ground_truths: list[str]) -> dict:
    assert len(predictions) == len(ground_truths)
    results = [score_prediction(p, g) for p, g in zip(predictions, ground_truths)]
    return {
        "accuracy": sum(results) / len(results),
        "correct": sum(results),
        "total": len(results),
        "per_sample": results,
    }
