"""Configuration values for the PredictPro recommender.

The values below make the scoring logic explicit and easy to discuss during
the intake interview. Scenario weights always sum to 1.0.
"""

from __future__ import annotations

SCENARIO_WEIGHTS: dict[str, dict[str, float]] = {
    "climate emergency": {
        "empathy_index": 0.20,
        "planetary_impact": 0.60,
        "ethical_resonance": 0.20,
    },
    "synthetic reality": {
        "empathy_index": 0.50,
        "planetary_impact": 0.10,
        "ethical_resonance": 0.40,
    },
    "decentralized data economy": {
        "empathy_index": 0.30,
        "planetary_impact": 0.20,
        "ethical_resonance": 0.50,
    },
    "bio-digital consciousness": {
        "empathy_index": 0.45,
        "planetary_impact": 0.15,
        "ethical_resonance": 0.40,
    },
    "ai governance collapse": {
        "empathy_index": 0.25,
        "planetary_impact": 0.20,
        "ethical_resonance": 0.55,
    },
}

MOOD_FACTORS: dict[str, float] = {
    "hopeful": 1.00,
    "curious": 0.90,
    "concerned": 1.10,
    "defiant": 1.20,
}

REQUIRED_COLUMNS: set[str] = {
    "title",
    "description",
    "scenario",
    "mood",
    "empathy_index",
    "planetary_impact",
    "ethical_resonance",
}
