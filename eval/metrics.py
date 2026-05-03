"""
eval/metrics.py — Accuracy metrics for FigureVault benchmarking

Functions:
  • compute_rmse(y_true, y_pred) → float
  • compute_r2(y_true, y_pred) → float
  • match_series(ground_truth, predicted) → list of (gt_series, pred_series) pairs
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Square Error between true and predicted values.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values.
    y_pred : np.ndarray
        Predicted values (must be same length as y_true).

    Returns
    -------
    float
        RMSE value. Returns 0.0 if arrays are empty.
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        return 0.0
    # Normalise to [0,1] range to make RMSE comparable across scales
    y_range = y_true.max() - y_true.min()
    if y_range == 0:
        return 0.0
    y_true_norm = (y_true - y_true.min()) / y_range
    y_pred_norm = (y_pred - y_true.min()) / y_range
    min_len = min(len(y_true_norm), len(y_pred_norm))
    return float(np.sqrt(np.mean((y_true_norm[:min_len] - y_pred_norm[:min_len]) ** 2)))


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the R² (coefficient of determination) score.

    Parameters
    ----------
    y_true : np.ndarray
    y_pred : np.ndarray

    Returns
    -------
    float
        R² score in range (-∞, 1]. Returns 0.0 if insufficient data.
    """
    if len(y_true) < 2:
        return 0.0
    min_len = min(len(y_true), len(y_pred))
    yt = y_true[:min_len]
    yp = y_pred[:min_len]
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1 - ss_res / ss_tot)


def match_series(
    ground_truth: list[dict[str, Any]],
    predicted: list[dict[str, Any]],
) -> list[tuple[dict, dict]]:
    """Match predicted data series to ground truth series by name similarity.

    Uses a simple name-matching strategy: exact match first, then
    fuzzy positional fallback.

    Parameters
    ----------
    ground_truth : list[dict]
        List of ground-truth series dicts (each has a ``series_name`` key).
    predicted : list[dict]
        List of predicted series dicts.

    Returns
    -------
    list of (gt_series, pred_series) tuples
    """
    if not ground_truth or not predicted:
        return []

    matched: list[tuple[dict, dict]] = []
    unmatched_pred = list(predicted)

    for gt in ground_truth:
        gt_name = (gt.get("series_name") or "").lower().strip()
        best_match: dict | None = None
        best_idx = -1

        # 1. Exact name match
        for i, pred in enumerate(unmatched_pred):
            pred_name = (pred.get("series_name") or "").lower().strip()
            if gt_name and pred_name and gt_name == pred_name:
                best_match = pred
                best_idx = i
                break

        # 2. Partial name match
        if best_match is None and gt_name:
            for i, pred in enumerate(unmatched_pred):
                pred_name = (pred.get("series_name") or "").lower().strip()
                if gt_name in pred_name or pred_name in gt_name:
                    best_match = pred
                    best_idx = i
                    break

        # 3. Positional fallback
        if best_match is None and unmatched_pred:
            best_match = unmatched_pred[0]
            best_idx = 0

        if best_match is not None:
            matched.append((gt, best_match))
            unmatched_pred.pop(best_idx)

    return matched


def compute_point_precision_recall(
    gt_points: list[dict],
    pred_points: list[dict],
    tolerance: float = 0.05,
) -> tuple[float, float]:
    """Compute precision and recall for point-level digitisation accuracy.

    A predicted point is "correct" if it falls within ``tolerance`` of any
    ground truth point (using normalised Euclidean distance).

    Parameters
    ----------
    gt_points : list of {"x": float, "y": float}
    pred_points : list of {"x": float, "y": float}
    tolerance : float
        Normalised distance threshold (0.05 = 5% of data range).

    Returns
    -------
    tuple (precision, recall) both in [0, 1].
    """
    if not gt_points or not pred_points:
        return 0.0, 0.0

    gt_arr = np.array([[p["x"], p["y"]] for p in gt_points], dtype=float)
    pred_arr = np.array([[p["x"], p["y"]] for p in pred_points], dtype=float)

    # Normalise to [0, 1] × [0, 1]
    for arr in (gt_arr, pred_arr):
        for dim in range(2):
            col = arr[:, dim]
            span = col.max() - col.min()
            if span > 0:
                arr[:, dim] = (col - col.min()) / span

    # True positives from prediction perspective
    tp_pred = sum(
        1 for pp in pred_arr
        if np.any(np.linalg.norm(gt_arr - pp, axis=1) <= tolerance)
    )
    # True positives from ground-truth perspective
    tp_gt = sum(
        1 for gp in gt_arr
        if np.any(np.linalg.norm(pred_arr - gp, axis=1) <= tolerance)
    )

    precision = tp_pred / len(pred_arr) if pred_arr.size else 0.0
    recall = tp_gt / len(gt_arr) if gt_arr.size else 0.0
    return float(precision), float(recall)
