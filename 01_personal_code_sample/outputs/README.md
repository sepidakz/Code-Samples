# Data

`PredictPro_Campaign_Dataset.csv` — a small demonstration fixture of five
speculative campaign records, each with a text description and three
invented metrics (empathy index, planetary impact, ethical resonance).

The metrics are fictional by design: the source thesis project modelled
imagined future scenarios rather than measured outcomes. The file exists
to exercise the loading, validation and scoring pipeline, not to support
analysis.

Metric values are stored on a 0–100 scale; `data_loader.py` detects this
and converts to 0–1 automatically.
