"""Content-based recommendation algorithm for PredictPro.

The algorithm uses TF-IDF vectorisation and cosine similarity to compare a
query concept with existing campaign descriptions. A second scoring layer
adjusts the ranking using scenario and mood-based speculative metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from scoring import combine_scores, compute_speculative_score
from similarity import cosine_similarity_matrix_numpy


class CampaignRecommender:
    """Content-based campaign recommender using TF-IDF and cosine similarity."""

    def __init__(self, ngram_range: tuple[int, int] = (1, 2)) -> None:
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=ngram_range,
            min_df=1,
        )
        self._campaigns: pd.DataFrame | None = None
        self._tfidf_matrix = None

    def fit(self, campaigns: pd.DataFrame) -> "CampaignRecommender":
        """Fit the TF-IDF model on campaign descriptions."""
        self._campaigns = campaigns.copy()
        self._tfidf_matrix = self.vectorizer.fit_transform(
            self._campaigns["description"]
        )
        return self

    def similarity_matrix(self) -> np.ndarray:
        """Return pairwise cosine similarity between all campaigns."""
        self._check_is_fitted()
        return cosine_similarity(self._tfidf_matrix, self._tfidf_matrix)

    def recommend(
        self,
        query: str,
        scenario: str,
        mood: str,
        top_n: int = 5,
        semantic_weight: float = 0.50,
    ) -> pd.DataFrame:
        """Rank campaigns using semantic and speculative scores.

        Args:
            query: Text describing the target campaign concept.
            scenario: Future scenario used for speculative weighting.
            mood: Mood factor used to amplify or soften scoring.
            top_n: Number of recommendations to return.
            semantic_weight: Weight assigned to cosine similarity.

        Returns:
            DataFrame sorted by final_score in descending order.
        """
        self._check_is_fitted()

        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        query_vector = self.vectorizer.transform([query.strip().lower()])
        semantic_similarity = cosine_similarity(
            query_vector,
            self._tfidf_matrix,
        ).flatten()

        speculative_score = compute_speculative_score(
            self._campaigns,
            scenario=scenario,
            mood=mood,
        )

        final_score = combine_scores(
            semantic_similarity=semantic_similarity,
            speculative_scores=speculative_score,
            semantic_weight=semantic_weight,
        )

        ranked = self._campaigns.copy()
        ranked["semantic_similarity"] = semantic_similarity
        ranked["speculative_score"] = speculative_score
        ranked["final_score"] = final_score

        return (
            ranked.sort_values("final_score", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )


    def recommend_with_manual_similarity(
        self,
        query: str,
        scenario: str,
        mood: str,
        top_n: int = 5,
        semantic_weight: float = 0.50,
    ) -> pd.DataFrame:
        """Rank campaigns using the from-scratch NumPy cosine function.

        This method is included to show the underlying linear algebra behind
        cosine similarity: dot product divided by the product of vector norms.
        """
        self._check_is_fitted()

        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        query_vector = self.vectorizer.transform([query.strip().lower()]).toarray()[0]
        tfidf_array = self._tfidf_matrix.toarray()
        semantic_similarity = cosine_similarity_matrix_numpy(
            query_vector=query_vector,
            matrix=tfidf_array,
        )

        speculative_score = compute_speculative_score(
            self._campaigns,
            scenario=scenario,
            mood=mood,
        )

        final_score = combine_scores(
            semantic_similarity=semantic_similarity,
            speculative_scores=speculative_score,
            semantic_weight=semantic_weight,
        )

        ranked = self._campaigns.copy()
        ranked["semantic_similarity"] = semantic_similarity
        ranked["speculative_score"] = speculative_score
        ranked["final_score"] = final_score

        return (
            ranked.sort_values("final_score", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

    def _check_is_fitted(self) -> None:
        if self._campaigns is None or self._tfidf_matrix is None:
            raise RuntimeError("The recommender must be fitted before use.")
