"""
Evaluation Metrics for ICU Deterioration Prediction

Implements AUROC, AUPRC, calibration metrics, and threshold selection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class EvaluationMetrics:
    """Complete evaluation metrics."""
    auroc: float
    auprc: float
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1: float
    brier_score: float
    optimal_threshold: float
    n_samples: int
    n_positive: int


def compute_auroc(
    y_true: list[int] | np.ndarray,
    y_score: list[float] | np.ndarray,
) -> Optional[float]:
    """
    Compute Area Under ROC Curve.

    Args:
        y_true: Binary labels (0 or 1)
        y_score: Predicted probabilities

    Returns:
        AUROC score or None if undefined
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    n = len(y_true)
    if n == 0:
        return None

    n_pos = y_true.sum()
    n_neg = n - n_pos

    if n_pos == 0 or n_neg == 0:
        return None

    # Sort by score descending
    sorted_indices = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[sorted_indices]
    y_score_sorted = y_score[sorted_indices]

    # Compute ROC curve points
    tp = 0
    fp = 0
    prev_score = None
    points = [(0.0, 0.0)]

    for i, (score, label) in enumerate(zip(y_score_sorted, y_true_sorted)):
        if prev_score is not None and score != prev_score:
            fpr = fp / n_neg
            tpr = tp / n_pos
            points.append((fpr, tpr))

        if label == 1:
            tp += 1
        else:
            fp += 1
        prev_score = score

    points.append((fp / n_neg, tp / n_pos))
    points.append((1.0, 1.0))

    # Compute area using trapezoidal rule
    area = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        area += (x1 - x0) * (y0 + y1) / 2.0

    return area


def compute_auprc(
    y_true: list[int] | np.ndarray,
    y_score: list[float] | np.ndarray,
) -> Optional[float]:
    """
    Compute Area Under Precision-Recall Curve (Average Precision).

    Args:
        y_true: Binary labels
        y_score: Predicted probabilities

    Returns:
        AUPRC score or None if undefined
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    n = len(y_true)
    if n == 0:
        return None

    n_pos = y_true.sum()
    if n_pos == 0:
        return None

    # Sort by score descending
    sorted_indices = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[sorted_indices]

    # Compute average precision
    tp = 0
    fp = 0
    ap = 0.0

    for label in y_true_sorted:
        if label == 1:
            tp += 1
            ap += tp / (tp + fp)
        else:
            fp += 1

    return ap / n_pos


def compute_brier_score(
    y_true: list[int] | np.ndarray,
    y_prob: list[float] | np.ndarray,
) -> Optional[float]:
    """
    Compute Brier Score (mean squared error of probabilities).

    Lower is better. Range: [0, 1].

    Args:
        y_true: Binary labels
        y_prob: Predicted probabilities

    Returns:
        Brier score
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    if len(y_true) == 0:
        return None

    return float(np.mean((y_prob - y_true) ** 2))


def compute_calibration(
    y_true: list[int] | np.ndarray,
    y_prob: list[float] | np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute calibration curve data for reliability diagram.

    Args:
        y_true: Binary labels
        y_prob: Predicted probabilities
        n_bins: Number of calibration bins

    Returns:
        (mean_predicted, fraction_positive, bin_counts, ece)
        - mean_predicted: Mean predicted probability per bin
        - fraction_positive: Actual positive fraction per bin
        - bin_counts: Number of samples per bin
        - ece: Expected Calibration Error
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    # Bin edges
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins[1:-1])

    mean_predicted = np.zeros(n_bins)
    fraction_positive = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_counts[i] = mask.sum()
            mean_predicted[i] = y_prob[mask].mean()
            fraction_positive[i] = y_true[mask].mean()

    # Expected Calibration Error
    # ECE = sum(|acc(b) - conf(b)| * n(b)) / n
    ece = 0.0
    total = len(y_true)
    for i in range(n_bins):
        if bin_counts[i] > 0:
            ece += np.abs(fraction_positive[i] - mean_predicted[i]) * bin_counts[i]
    ece /= total

    return mean_predicted, fraction_positive, bin_counts, ece


def find_optimal_threshold(
    y_true: list[int] | np.ndarray,
    y_prob: list[float] | np.ndarray,
    metric: str = "f1",
) -> tuple[float, float]:
    """
    Find optimal threshold based on specified metric.

    Args:
        y_true: Binary labels
        y_prob: Predicted probabilities
        metric: Metric to optimize ("f1", "youden", "precision", "recall")

    Returns:
        (optimal_threshold, metric_value)
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    best_threshold = 0.5
    best_score = -1.0

    for k in range(1, 200):
        threshold = k / 200.0
        y_pred = (y_prob >= threshold).astype(int)

        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()

        if metric == "f1":
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        elif metric == "youden":
            # Youden's J = Sensitivity + Specificity - 1
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            score = sensitivity + specificity - 1

        elif metric == "precision":
            score = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        elif metric == "recall":
            score = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score


def compute_metrics_at_threshold(
    y_true: list[int] | np.ndarray,
    y_prob: list[float] | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute classification metrics at a specific threshold.

    Args:
        y_true: Binary labels
        y_prob: Predicted probabilities
        threshold: Classification threshold

    Returns:
        Dict with accuracy, precision, recall, specificity, f1
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
    }


def compute_metrics(
    y_true: list[int] | np.ndarray,
    y_prob: list[float] | np.ndarray,
    threshold: Optional[float] = None,
) -> EvaluationMetrics:
    """
    Compute all evaluation metrics.

    Args:
        y_true: Binary labels
        y_prob: Predicted probabilities
        threshold: Classification threshold (if None, finds optimal)

    Returns:
        EvaluationMetrics dataclass
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    # Find optimal threshold if not provided
    if threshold is None:
        threshold, _ = find_optimal_threshold(y_true, y_prob, metric="f1")

    # Compute all metrics
    auroc = compute_auroc(y_true, y_prob)
    auprc = compute_auprc(y_true, y_prob)
    brier = compute_brier_score(y_true, y_prob)

    threshold_metrics = compute_metrics_at_threshold(y_true, y_prob, threshold)

    return EvaluationMetrics(
        auroc=auroc or 0.0,
        auprc=auprc or 0.0,
        accuracy=threshold_metrics["accuracy"],
        precision=threshold_metrics["precision"],
        recall=threshold_metrics["recall"],
        specificity=threshold_metrics["specificity"],
        f1=threshold_metrics["f1"],
        brier_score=brier or 0.0,
        optimal_threshold=threshold,
        n_samples=len(y_true),
        n_positive=int(y_true.sum()),
    )


def evaluate_subgroups(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    subgroup_labels: np.ndarray,
    subgroup_names: list[str],
) -> dict[str, EvaluationMetrics]:
    """
    Evaluate model across different subgroups.

    Args:
        y_true: Binary labels
        y_prob: Predicted probabilities
        subgroup_labels: Integer array indicating subgroup membership
        subgroup_names: Names for each subgroup

    Returns:
        Dict mapping subgroup name to metrics
    """
    results = {}

    for i, name in enumerate(subgroup_names):
        mask = subgroup_labels == i
        if mask.sum() >= 10:  # Minimum samples for evaluation
            metrics = compute_metrics(y_true[mask], y_prob[mask])
            results[name] = metrics

    return results
