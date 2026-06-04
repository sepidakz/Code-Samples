"""Matplotlib visualisation helpers for recommendation outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_similarity_heatmap(
    similarity_matrix: np.ndarray,
    labels: list[str],
    output_path: str | Path,
) -> Path:
    """Save a pairwise similarity heatmap as a PNG file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(similarity_matrix, aspect="auto")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    axis.set_xticks(range(len(labels)))
    axis.set_yticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.set_yticklabels(labels)
    axis.set_title("Cosine Similarity Between Campaign Descriptions")

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def save_recommendation_bar_chart(
    recommendations: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Save a horizontal bar chart for final recommendation scores."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ordered = recommendations.sort_values("final_score", ascending=True)

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.barh(ordered["title"], ordered["final_score"])
    axis.set_xlabel("Final score")
    axis.set_title("Top Campaign Recommendations")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
