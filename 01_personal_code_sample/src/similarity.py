"""Vector similarity functions implemented with NumPy.

Scikit-learn is used in the main recommender for production-style TF-IDF and
cosine similarity. This file includes a small from-scratch implementation of
cosine similarity to show the underlying linear algebra explicitly.
"""

from __future__ import annotations

import numpy as np


def cosine_similarity_numpy(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors using NumPy.

    cosine_similarity(a, b) = dot(a, b) / (norm(a) * norm(b))
    """
    a = np.asarray(vector_a, dtype=float)
    b = np.asarray(vector_b, dtype=float)

    if a.shape != b.shape:
        raise ValueError("Both vectors must have the same shape.")

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if np.isclose(norm_a, 0.0) or np.isclose(norm_b, 0.0):
        raise ValueError("Cosine similarity is undefined for a zero vector.")

    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_similarity_matrix_numpy(query_vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between one query vector and all matrix rows."""
    query = np.asarray(query_vector, dtype=float)
    vectors = np.asarray(matrix, dtype=float)

    if query.ndim != 1:
        raise ValueError("query_vector must be one-dimensional.")
    if vectors.ndim != 2:
        raise ValueError("matrix must be two-dimensional.")
    if vectors.shape[1] != query.shape[0]:
        raise ValueError("query_vector length must match matrix column count.")

    query_norm = np.linalg.norm(query)
    vector_norms = np.linalg.norm(vectors, axis=1)

    if np.isclose(query_norm, 0.0):
        raise ValueError("Cosine similarity is undefined for a zero query vector.")
    if np.any(np.isclose(vector_norms, 0.0)):
        raise ValueError("Cosine similarity is undefined for zero matrix rows.")

    dot_products = vectors @ query
    return dot_products / (vector_norms * query_norm)


def demo_manual_cosine_similarity() -> float:
    """Return a deterministic example for quick inspection."""
    vector_a = np.array([1.0, 2.0, 0.0])
    vector_b = np.array([2.0, 1.0, 1.0])
    return cosine_similarity_numpy(vector_a, vector_b)
