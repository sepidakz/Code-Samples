"""Data loading and preprocessing utilities for campaign records."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from configuration import REQUIRED_COLUMNS


COLUMN_ALIASES = {
    "Title": "title",
    "Description": "description",
    "Scenario": "scenario",
    "Mood": "mood",
    "EmpathyIndex": "empathy_index",
    "PlanetaryImpact": "planetary_impact",
    "EthicalResonance": "ethical_resonance",
    "ai_compliance": "ethical_resonance",
}


def load_campaign_data(csv_path: str | Path) -> pd.DataFrame:
    """Load campaign data from CSV and return a cleaned DataFrame.

    Args:
        csv_path: Path to a CSV file with campaign records.

    Returns:
        A validated and cleaned pandas DataFrame.

    Raises:
        FileNotFoundError: If the CSV path does not exist.
        ValueError: If required columns are missing.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    dataframe = pd.read_csv(path)
    dataframe = clean_campaign_data(dataframe)
    validate_campaign_data(dataframe)
    return dataframe


def clean_campaign_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Standardise columns, text fields, and numeric metric ranges.

    The original thesis dataset used some speculative metric names. For this
    admission-ready version, the columns are normalised to snake_case.
    """
    cleaned = dataframe.copy()
    cleaned = cleaned.rename(columns=COLUMN_ALIASES)

    cleaned.columns = [column.strip().lower() for column in cleaned.columns]

    text_columns = ["title", "description", "scenario", "mood"]
    for column in text_columns:
        if column in cleaned.columns:
            cleaned[column] = (
                cleaned[column]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
            )

    if "description" in cleaned.columns:
        cleaned["description"] = cleaned["description"].replace(
            {"": "no description provided"}
        )

    cleaned = cleaned.drop_duplicates(subset=["title", "description"]).reset_index(
        drop=True
    )

    metric_columns = ["empathy_index", "planetary_impact", "ethical_resonance"]
    for column in metric_columns:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    present_metrics = [col for col in metric_columns if col in cleaned.columns]
    if present_metrics:
        cleaned[present_metrics] = cleaned[present_metrics].fillna(
            cleaned[present_metrics].median(numeric_only=True)
        )

    # Accept both 0-1 and 0-100 input scales. Convert 0-100 values to 0-1.
    for column in present_metrics:
        if cleaned[column].max() > 1.0:
            cleaned[column] = cleaned[column] / 100.0

    return cleaned


def validate_campaign_data(dataframe: pd.DataFrame) -> None:
    """Validate that the DataFrame has all columns required by the model."""
    missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    if dataframe.empty:
        raise ValueError("Dataset is empty after cleaning.")

    if dataframe["description"].str.len().sum() == 0:
        raise ValueError("Descriptions are empty, TF-IDF cannot be fitted.")
