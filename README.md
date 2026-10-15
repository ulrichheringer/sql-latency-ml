# SQL Query Latency Estimation

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)  [![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)  [![MIT License](https://img.shields.io/badge/License-MIT-2ea44f?style=for-the-badge)](LICENSE)  ![Synthetic Data](https://img.shields.io/badge/Data-Synthetic-6b7280?style=for-the-badge)

Estimates SQL query runtime from query structure and estimated row counts.

## Data

Deterministic synthetic data is generated locally in `pipeline.py`. No private data or external download is required.

## Install and run

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python cli.py evaluate
python cli.py train --model model.joblib
python cli.py predict --model model.joblib
```

## Evaluation results

With seed 42, a random holdout of 450 synthetic queries had an MAE of 17.1671 ms, an RMSE of 21.8282 ms, and an R² of 0.9773. The generated latency follows a known formula based on estimated rows and query structure, so the result does not represent a real database engine. If no row estimate is supplied in the input or SQL comment, the parser uses 1,000.

## Limitations

The generated runtimes are not database measurements. Real use would require query plans and execution timings from the target database, evaluated on queries that were not used for training.
