"""Command-line entry point for the PredictPro recommender sample."""

from __future__ import annotations

import argparse
from pathlib import Path

from data_loader import load_campaign_data
from recommender import CampaignRecommender
from similarity import demo_manual_cosine_similarity
from visualization import save_recommendation_bar_chart, save_similarity_heatmap


DEFAULT_QUERY = (
    "transparent campaign concept that supports ethical decision-making, "
    "environmental awareness, and user consent"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the PredictPro content-based recommender."
    )
    parser.add_argument(
        "--data",
        default="../data/PredictPro_Campaign_Dataset.csv",
        help="Path to the campaign dataset CSV.",
    )
    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help="Text query used to rank campaign concepts.",
    )
    parser.add_argument(
        "--scenario",
        default="Climate Emergency",
        help="Scenario used for speculative scoring.",
    )
    parser.add_argument(
        "--mood",
        default="Concerned",
        help="Mood used for score amplification.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of recommendations to return.",
    )
    parser.add_argument(
        "--semantic-weight",
        type=float,
        default=0.50,
        help="Weight assigned to cosine similarity, between 0 and 1.",
    )
    parser.add_argument(
        "--manual-similarity",
        action="store_true",
        help="Use the from-scratch NumPy cosine similarity implementation.",
    )
    parser.add_argument(
        "--output-dir",
        default="../outputs",
        help="Directory where charts and results are saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = Path(__file__).parent / data_path

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    campaigns = load_campaign_data(data_path)
    recommender = CampaignRecommender().fit(campaigns)

    if args.manual_similarity:
        recommendations = recommender.recommend_with_manual_similarity(
            query=args.query,
            scenario=args.scenario,
            mood=args.mood,
            top_n=args.top_n,
            semantic_weight=args.semantic_weight,
        )
    else:
        recommendations = recommender.recommend(
            query=args.query,
            scenario=args.scenario,
            mood=args.mood,
            top_n=args.top_n,
            semantic_weight=args.semantic_weight,
        )

    results_path = output_dir / "recommendations.csv"
    recommendations.to_csv(results_path, index=False)

    save_similarity_heatmap(
        recommender.similarity_matrix(),
        labels=campaigns["title"].tolist(),
        output_path=output_dir / "similarity_heatmap.png",
    )
    save_recommendation_bar_chart(
        recommendations,
        output_path=output_dir / "recommendation_scores.png",
    )

    display_columns = [
        "title",
        "semantic_similarity",
        "speculative_score",
        "final_score",
    ]

    print("\nManual cosine similarity demo:")
    print(f"cosine([1, 2, 0], [2, 1, 1]) = {demo_manual_cosine_similarity():.4f}")

    print("\nTop recommendations:")
    print(recommendations[display_columns].to_string(index=False))
    print(f"\nSaved results to: {results_path}")


if __name__ == "__main__":
    main()
