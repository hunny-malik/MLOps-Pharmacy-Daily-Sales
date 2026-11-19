# SCSE3040 MLOps Mini Project --- Group G179

## Problem Details
- **Group ID**: G179
- **Problem ID**: MP108
- **Problem Name**: Pharmacy Daily Sales (Bhopal)
- **Batch**: 47
- **Faculty**: Dr. Richa Sharma

## Group Members
| Enrolment No. | Name |
|---|---|
| S24CSEU1858 | AmitSinghVikram |
| S24CSEU0755 | HUNNYMALIK |
| S24CSEU1833 | SashankGautam |
| S24CSEU1836 | VISHU. |

---

## Project Overview
The Bhopal Chemists' Association predicts daily medicine unit sales (`units_sold`) for pharmacies in Bhopal during flu season. This repository implements an end-to-end MLOps pipeline covering:
- **Version Control & DVC**: Tracking data artifacts (`train.csv`, `later.csv`) with DVC while keeping raw data out of git.
- **Data Quality & Problem Detection**: Systematic detection of 7 data anomalies in `train.csv`.
- **Model Tracking & Registry**: MLflow tracking with SQLite backend (`sqlite:///mlflow.db`), logging parameters, metrics (MAE), and model registration with `@champion` alias.
- **Prediction Service**: FastAPI REST service with schema validation, pydantic field constraints, and health endpoint.
- **Containerization & Deployment**: Docker container (`mp108-api:1.0`), Kubernetes deployment (`mp108-api`), and service (`mp108-svc`).
- **CI/CD Pipeline**: GitHub Actions running automated test suites on push.
- **Drift Monitoring & Retraining**: Population Stability Index (PSI), group-wise error analysis, retraining logic, and holdout predictions.

---

## Repository Structure
```
.
├── .dvc/                             # DVC configuration and tracking
├── .github/workflows/ci.yml          # GitHub Actions CI pipeline
├── app/                              # FastAPI application
│   ├── __init__.py
│   └── main.py                       # REST API (POST /predict, GET /health)
├── data/
│   └── raw/
│       ├── .gitignore                # Ignores train.csv and later.csv in git
│       ├── train.csv.dvc             # DVC pointer for train.csv
│       ├── later.csv.dvc             # DVC pointer for later.csv
│       └── holdout_features.csv      # September holdout features
├── k8s/                              # Kubernetes manifests
│   ├── deployment.yaml
│   └── service.yaml
├── models/                           # Serialized models
│   └── model.joblib
├── predictions/                      # Holdout predictions
│   └── holdout_predictions.csv
├── reports/
│   └── findings.yaml                 # Stage 1-4 project findings
├── src/
│   └── mp108/                        # Source package
│       ├── __init__.py
│       ├── data_checks.py            # Code detecting the 7 data problems
│       ├── data_cleaner.py           # Data cleaning pipeline
│       └── utils.py                  # Config loader and utilities
├── tests/                            # Automated test suite
│   ├── __init__.py
│   └── test_data_checks.py
├── .dockerignore                     # Docker build exclusions
├── .gitignore                        # Git ignore patterns
├── Dockerfile                        # Container definition
├── params.yaml                       # Project settings read via yaml.safe_load
├── README.md                         # Project documentation
└── requirements.txt                  # Pinned dependencies
```

---

## Stage 1: Data Problem Findings Summary (`train.csv`)
The 7 data problem checks executed via `src/mp108/data_checks.py` identified:
1. **Leaking Column**: `false` (No target-leaking feature present).
2. **Unit Mixup**: `true` (`distance_to_hospital_km` for `Old Town` was recorded in metres instead of kilometres; `factor: 1000`).
3. **Sentinel Value**: `true` (`nearby_clinics` has `60` instances of sentinel value `9999`).
4. **Duplicate Rows**: `true` (`26` exact duplicate rows).
5. **Stuck Sensor**: `true` (`flu_cases_reported` was stuck at `134` from `2026-03-15` to `2026-04-04`).
6. **Target Typos**: `false` (No 10x typographical outliers in `units_sold`).
7. **Spelling Variants**: `false` (All categorical levels for `shop_type` and `locality` are consistent with `later.csv`).

---

## Quick Start

### 1. Environment Setup
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Pull Data via DVC
```bash
dvc pull
```

### 3. Run Data Checks
```bash
python -m src.mp108.data_checks
```

### 4. Run Pytest Tests
```bash
pytest
```
