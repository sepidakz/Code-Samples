"""Mathematical scoring functions for the PredictPro recommender."""

from __future__ import annotations

import numpy as np
import pandas as pd

from configuration import MOOD_FACTORS, SCENARIO_WEIGHTS


def min_max_normalize(values: np.ndarray) -> np.ndarray:
    """Scale numeric values to the interval [0, 1].

    If all values are equal, the function returns zeros to avoid division by
    zero. This behaviour is explicit, deterministic, and easy to test.
    """
    values = np.asarray(values, dtype=float)
    minimum = values.min()
    maximum = values.max()

    if np.isclose(maximum, minimum):
        return np.zeros_like(values, dtype=float)

    return (values - minimum) / (maximum - minimum)


def get_scenario_weights(scenario: str) -> dict[str, float]:
    """Return metric weights for a scenario, falling back to equal weights."""
    normalized_scenario = scenario.strip().lower()

    if normalized_scenario in SCENARIO_WEIGHTS:
        return SCENARIO_WEIGHTS[normalized_scenario]

    return {
        "empathy_index": 1.0 / 3.0,
        "planetary_impact": 1.0 / 3.0,
        "ethical_resonance": 1.0 / 3.0,
    }


def get_mood_factor(mood: str) -> float:
    """Return the mood amplification factor, falling back to neutral."""
    return MOOD_FACTORS.get(mood.strip().lower(), 1.0)


def compute_speculative_score(
    dataframe: pd.DataFrame,
    scenario: str,
    mood: str,
) -> pd.Series:
    """Compute a weighted speculative score for each campaign.

    The score combines three interpretable dimensions:

    speculative_score =
        mood_factor * (
            empathy_index * w_empathy
            + planetary_impact * w_planetary
            + ethical_resonance * w_ethical
        )

    Args:
        dataframe: Clean campaign DataFrame.
        scenario: User-selected scenario.
        mood: User-selected mood.

    Returns:
        A pandas Series with one score per campaign.
    """
    weights = get_scenario_weights(scenario)
    mood_factor = get_mood_factor(mood)

    score = mood_factor * (
        dataframe["empathy_index"] * weights["empathy_index"]
        + dataframe["planetary_impact"] * weights["planetary_impact"]
        + dataframe["ethical_resonance"] * weights["ethical_resonance"]
    )

    return score.clip(lower=0.0, upper=1.2)


def combine_scores(
    semantic_similarity: np.ndarray,
    speculative_scores: pd.Series,
    semantic_weight: float = 0.50,
) -> np.ndarray:
    """Combine semantic similarity and speculative scores into final ranking.

    Args:
        semantic_similarity: Raw cosine similarity values.
        speculative_scores: Weighted scenario and mood scores.
        semantic_weight: Weight assigned to semantic similarity. The remaining
            weight is assigned to the speculative score.

    Returns:
        Final score as a NumPy array.
    """
    if not 0.0 <= semantic_weight <= 1.0:
        raise ValueError("semantic_weight must be between 0 and 1.")

        normalized_similarity = min_max_normalize(semantic_similarity)
    normalized_speculative = min_max_normalize(
        speculative_scores.to_numpy(dtype=float)
    )
    speculative_weight = 1.0 - semantic_weight

    return (
        semantic_weight * normalized_similarity
        + speculative_weight * normalized_speculative
    )
