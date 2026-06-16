# CAPSDAC 2.0 Level 3 MLE Forecasting Platform

This repository is an **interview-facing Level 3 Machine Learning Engineer package** for a CAPSDAC-style preschool enrollment forecasting platform.

It builds on the Level 2 ML-enabled data platform and adds deeper MLE components:

- 3-month-ahead forecasting for every preschool site
- enrollment demand, capacity risk, and staffing need outputs
- leakage-safe feature engineering
- expanding-window time-series validation
- multi-model comparison
- local MLflow-style experiment tracking
- champion/challenger model registry
- drift monitoring
- retraining decision policy
- FastAPI serving endpoints
- Streamlit monitoring dashboard
- final MLE interview documentation

This public version uses **synthetic, de-identified sample data only**. It is not an official CAPSDAC production system and does not contain confidential child-level records.

## Interview positioning

**Best-fit roles**

- Machine Learning Engineer
- ML Platform Engineer
- Senior Data Engineer with production ML ownership
- Applied Forecasting Engineer
- Data Science Engineer

**One-line story**

> I extended a governed CAPSDAC-style data platform into a Level 3 MLE forecasting system that predicts 3-month-ahead preschool enrollment, compares candidate models through expanding-window validation, tracks experiments, promotes a champion model, monitors drift, generates capacity and staffing signals, and serves forecasts through API and dashboard layers.

## V3 roadmap delivered

| Version | Focus | Delivered in this package |
|---|---|---|
| V3.1 | Forecasting depth | 3-month-ahead site enrollment, capacity risk, staffing estimate |
| V3.2 | MLflow / model registry | Local MLflow-style run artifacts and champion/challenger registry |
| V3.3 | Retraining pipeline | Monthly retraining policy and promote/reject decision file |
| V3.4 | API + monitoring dashboard | FastAPI endpoints and Streamlit dashboard |
| V3.5 | Final interview documentation | MLE guide, architecture notes, model card, run summary |

## Core modules

```text
forecasting/src/capsdac_ml/
├── data_contracts.py          # snapshot validation
├── feature_engineering.py     # V3 feature store and 3-month target
├── model_selection.py         # candidate models and expanding-window CV
├── monitoring.py              # PSI drift diagnostics
├── experiment_tracking.py     # local MLflow-style run tracking
├── model_registry.py          # champion/challenger registry
├── retraining.py              # retraining decision policy
├── forecasting.py             # site-level forecast output
└── contribution_analysis.py   # vendor/statewide aggregation
```

## Candidate models

The runnable public package uses scikit-learn models so it is easy to install and run:

- Median baseline
- Linear Regression
- Ridge Regression
- Random Forest
- Gradient Boosting
- Extra Trees, used as a fast tree-ensemble challenger

Optional production extensions documented but not required by default:

- XGBoost
- LightGBM
- Prophet
- LSTM / Temporal Fusion Transformer

Production extensions are documented for XGBoost, LightGBM, Prophet, and LSTM. They are intentionally not required by default because those packages can create heavy native dependency issues during interviews.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make validate
```

Equivalent commands without Make:

```bash
export PYTHONPATH=$PWD/forecasting
pytest
python forecasting/scripts/run_capsdac_pipeline.py
python forecasting/scripts/generate_visualization_report.py
```

## Expected outputs

```text
data/processed/monthly_enrollment_features.csv
outputs/metrics/model_metrics.json
outputs/metrics/model_leaderboard.csv
outputs/metrics/time_series_cv_results.json
outputs/reports/data_validation_report.json
outputs/reports/drift_report.json
outputs/reports/run_summary.md
outputs/experiments/runs.csv
outputs/retraining/retraining_decision.json
outputs/forecasts/site_forecast.csv
outputs/forecasts/vendor_forecast.csv
outputs/forecasts/statewide_forecast.csv
models/registry/champion_model.joblib
models/registry/latest_challenger_model.joblib
models/registry/model_registry.json
```

## API demo

```bash
uvicorn serving.api.main:app --reload
```

Useful endpoints:

```text
/health
/forecasts/statewide
/forecasts/sites
/forecasts/vendors/top
/metrics/model
/metrics/leaderboard
/monitoring/drift
/monitoring/retraining
/registry/champion
```

## Dashboard demo

```bash
streamlit run dashboards/streamlit_app.py
```

## How this answers MLE interview questions

### What is the forecasting problem?

The model predicts **3-month-ahead enrollment for each preschool site**. The downstream outputs support aggregate planning: program demand, capacity risk, staffing need, and statewide/vendor-level forecast review.

### How did you avoid leakage?

Lag and rolling features are shifted before aggregation. The target is shifted forward by 3 months at the same site. Validation uses expanding-window splits where training months always occur before validation months.

### Why multiple models?

The package compares simple baselines, linear models, tree ensembles, and boosting-style models. The champion is selected by validation RMSE, not by preference.

### Why not require XGBoost, LightGBM, Prophet, and LSTM by default?

For an interview repo, reproducibility matters. The default package runs with scikit-learn only. In production, I would add XGBoost and LightGBM as optional candidates and use MLflow/Vertex AI to compare runtime, accuracy, stability, and explainability. Prophet and LSTM would be evaluated only if the data volume and seasonality justify them.

### What happens when drift is detected?

The pipeline creates a PSI-based drift report and a retraining decision file. In production, moderate/high drift would trigger investigation, challenger training, stakeholder review, and model promotion only if the challenger meets acceptance criteria.

## Responsible use

This repository is for portfolio demonstration and aggregate planning only. Production use would require approved data access, privacy review, data governance approval, official data quality thresholds, CI/CD secrets, model monitoring thresholds, stakeholder sign-off, and formal deployment controls.
