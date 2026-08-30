# Code Samples

Sepideh Zamani — Utrecht, Netherlands

A Python package refactored out of exploratory research notebooks, kept as a sample of how I
structure code once an idea has stopped being a prototype.

The companion repository, [`data-science-work-samples`](https://github.com/sepidakz/data-science-work-samples),
holds the analysis side of my work — cross-validated modelling, data quality auditing, and
reporting of negative results. This repository is the engineering side of the same progression:
what the exploratory work looks like after it has been pulled apart and rebuilt.

---

## `01_personal_code_sample` — PredictPro ranking pipeline

The source was a set of Jupyter notebooks from my Master's thesis prototype, PredictPro, built
with Monks as the client partner. The notebooks were exploratory: linear, stateful, and dependent
on execution order. The strongest algorithmic component was extracted and rebuilt as a modular
package.

**What the refactor changed**

| Notebook original | This package |
|---|---|
| Single linear flow, order-dependent | Six modules with separated responsibilities |
| Inline constants scattered through cells | Centralised in `configuration.py` |
| No input validation | Explicit schema checks with actionable error messages |
| Implicit assumptions | Type hints and docstrings throughout |
| Not runnable outside the notebook | Command-line entry point, reproducible from a clean checkout |

**What the code demonstrates**

- Loading, cleaning and validating CSV input, including column aliasing, deduplication, median imputation and automatic detection of 0–1 versus 0–100 metric scales
- TF-IDF vectorisation and cosine similarity with scikit-learn
- A from-scratch NumPy cosine similarity implementation — dot product over the product of vector norms — included to make the underlying linear algebra explicit rather than delegated
- Weighted multi-criteria scoring with min-max normalisation
- Guard clauses, fail-fast validation, and errors that say what went wrong
- Modular structure with type hints and docstrings on every public function

## What this is not

Worth stating plainly, since the file names could suggest otherwise:

- **This is not machine learning.** TF-IDF combined with cosine similarity and a weighted sum is information retrieval and arithmetic. Nothing here is trained, and there is no train/test split.
- **There is no evaluation, and there cannot be.** The underlying thesis dataset is speculative by design — the metrics describe imagined futures rather than measured outcomes, so no ground truth exists to score against.
- **The dataset is small.** It is a demonstration fixture, not a corpus.

For modelling work with cross-validation, baselines, permutation testing and evaluation, see the
companion repository linked above.

## Structure

```
01_personal_code_sample/
├── data/            # demonstration dataset
├── outputs/         # generated charts and rankings
└── src/
    ├── configuration.py   # weights, factors, required columns
    ├── data_loader.py     # loading, cleaning, validation
    ├── recommender.py     # TF-IDF vectorisation and ranking
    ├── scoring.py         # normalisation and weighted combination
    ├── similarity.py      # NumPy cosine similarity from scratch
    ├── visualization.py   # chart helpers
    └── main.py            # command-line runner
```

## Running it

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cd 01_personal_code_sample/src
python main.py
```

To run the pipeline using the from-scratch NumPy similarity function instead of scikit-learn's:

```bash
python main.py --manual-similarity
```

## Background

PredictPro was a speculative design project: a tool for reasoning about the ethical and
environmental consequences of marketing campaigns rather than optimising their performance. The
client brief asked for metrics that do not exist — empathy, planetary impact, ethical resonance —
so the interesting engineering problem was making invented, weighted, subjective inputs behave
transparently and reproducibly. That constraint is what shaped the scoring and normalisation
logic here.
