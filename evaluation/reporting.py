"""Reporting utilities for evaluation outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def save_pit_histogram(pit_values: Iterable[float], path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(pit_values, bins=10, stat="probability", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("PIT Value")
    ax.set_ylabel("Frequency")
    ax.set_ylim(0, 0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_reliability_curve(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=df, x="pred_bin_mid", y="actual_rate", marker="o", ax=ax, label="Actual")
    sns.lineplot(data=df, x="pred_bin_mid", y="pred_rate", ax=ax, label="Predicted")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    ax.set_title("Home Win Probability Reliability")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Actual Win Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_bucket_bar(df: pd.DataFrame, bucket_col: str, metric: str, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df, x=bucket_col, y=metric, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(bucket_col.replace("_", " ").title())
    ax.set_ylabel(metric.replace("_", " ").title())
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_summary_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)


__all__ = [
    "save_pit_histogram",
    "save_reliability_curve",
    "save_bucket_bar",
    "save_summary_table",
]
