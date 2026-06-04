# Code Explanation, PredictPro Personal Programming Sample

## Purpose

This code sample is based on the PredictPro prototype from my HBO Master thesis. It demonstrates a content-based recommendation pipeline using Python and common data science libraries.

## Algorithmic approach

1. The campaign dataset is loaded from CSV.
2. Text fields are cleaned and normalised.
3. Campaign descriptions are transformed into TF-IDF vectors.
4. A user query is transformed using the same vectoriser.
5. Cosine similarity compares the query vector with each campaign vector.
6. Scenario and mood variables are converted into weighted speculative scores.
7. Semantic similarity and speculative scores are combined into one final ranking.

## Mathematical concepts used

- Vector representation of text through TF-IDF.
- Dot product and vector norms in the manual NumPy cosine similarity function.
- Cosine similarity between vectors.
- Weighted sum for multi-criteria scoring.
- Min-max normalisation.
- Basic matrix-style operations through NumPy and Scikit-learn.

## Files

- `src/recommender.py`: main recommendation algorithm.
- `src/scoring.py`: normalisation and weighted scoring logic.
- `src/similarity.py`: manual cosine similarity implementation.
- `src/data_loader.py`: data preprocessing and validation.
- `src/main.py`: command-line runner.
- `src/visualization.py`: visualisation helpers.

## Why this is relevant to the master

The code is not a computer vision project, but it shows my current foundation in Python-based scientific programming, data preprocessing, vectorisation, similarity measurement, linear algebra concepts, and explainable algorithmic scoring. These are skills I want to deepen and extend toward image data analysis, machine learning, and computer vision during the master.
