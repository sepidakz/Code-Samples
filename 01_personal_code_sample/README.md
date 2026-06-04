# PredictPro Personal Code Sample

This folder contains my personal Python code sample for the NHL Stenden Master Computer Vision & Data Science admission procedure.

The project is based on my HBO Master thesis prototype, **PredictPro**, a speculative AI foresight concept for campaign decision-making. For the admission code sample, I refactored the strongest algorithmic part into a modular Python project.

## What the code demonstrates

- Loading and validating CSV data with `pandas`
- Preprocessing text and numeric metrics
- TF-IDF vectorisation with `scikit-learn`
- Cosine similarity for content-based recommendation
- From-scratch cosine similarity using `numpy`
- Weighted scoring using scenario and mood variables
- Min-max normalisation using `numpy`
- Chart generation using `matplotlib`
- Modular Python files with type hints and docstrings

## Main files to review

Please review these files as the strongest evidence of algorithmic and scientific programming work:

- `src/recommender.py`, TF-IDF vectorisation, cosine similarity, and ranking algorithm
- `src/scoring.py`, mathematical scoring, normalisation, and weighted score combination
- `src/similarity.py`, manual NumPy cosine similarity using dot product and vector norms
- `src/data_loader.py`, CSV loading, cleaning, validation, and preprocessing
- `src/main.py`, command-line runner that executes the full pipeline

## How to run

From this folder:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows

pip install -r ../requirements.txt
cd src
python main.py
```

To run the same pipeline with the manual NumPy cosine similarity implementation:

```bash
python main.py --manual-similarity
```

## Note on scope

This is not a computer vision project yet. It is submitted as a personal programming code example that demonstrates Python, data processing, vectorisation, similarity scoring, and modular algorithmic thinking. My goal in the master is to extend this technical foundation toward image data analysis, machine learning, and computer vision.
