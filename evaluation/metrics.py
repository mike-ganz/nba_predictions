"""Evaluation metrics for probabilistic forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_log_loss(predicted_probs: np.ndarray, outcomes: np.ndarray) -> float:
    probs = np.clip(predicted_probs, 1e-12, 1.0)
    idx = np.arange(len(probs))
    return float(-np.log(probs[idx, outcomes]).mean())


def compute_crps(cdf: np.ndarray, outcomes: np.ndarray) -> float:
    errors = []
    for i, outcome in enumerate(outcomes):
        indicator = np.zeros_like(cdf[i])
        indicator[outcome:] = 1
        errors.append(np.mean((cdf[i] - indicator) ** 2))
    return float(np.mean(errors))


def pit_histogram(cdf: np.ndarray, outcomes: np.ndarray, bins: int = 10) -> np.ndarray:
    pit_values = [cdf[i, outcome] for i, outcome in enumerate(outcomes)]
    hist, _ = np.histogram(pit_values, bins=bins, range=(0, 1))
    return hist


def mean_absolute_error_margin(margin_pmf: np.ndarray, outcomes: np.ndarray, margin_range: tuple[int, int]) -> float:
    low, high = margin_range
    idx = np.arange(low, high + 1)
    expected_margin = margin_pmf @ idx
    error = np.abs(expected_margin - outcomes)
    return float(np.mean(error))


def brier_score(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean((probabilities - outcomes) ** 2))


def bucket_metrics(margins: np.ndarray, spreads: np.ndarray, totals: np.ndarray, metric_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    spread_bins = pd.cut(spreads, bins=[-20, -10, -5, 0, 5, 10, 20], include_lowest=True)
    total_bins = pd.cut(totals, bins=[150, 200, 210, 220, 230, 240, 260])
    spread_df = pd.DataFrame({"spread_bin": spread_bins, "metric": metric_values})
    total_df = pd.DataFrame({"total_bin": total_bins, "metric": metric_values})
    spread_summary = spread_df.groupby("spread_bin")['metric'].mean().reset_index()
    total_summary = total_df.groupby("total_bin")['metric'].mean().reset_index()
    return spread_summary, total_summary


__all__ = [
    "compute_log_loss",
    "compute_crps",
    "pit_histogram",
    "mean_absolute_error_margin",
    "brier_score",
    "bucket_metrics",
]

