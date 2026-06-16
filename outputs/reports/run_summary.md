# CAPSDAC 2.0 Level 3 MLE Forecast Run Summary

## Data validation

- Input rows: 216
- Sites: 6
- Vendors: 4
- Snapshot range: 2023-01 to 2025-12
- Missing required columns: []
- Duplicate site-month records: 0
- Negative enrollment rows: 0

## Forecasting problem

- Forecast target: 3-month-ahead preschool site enrollment
- Business outputs: monthly enrollment, program demand, site capacity risk, staffing need estimate
- Validation strategy: expanding-window monthly validation

## Model selection

- Selected model: linear_regression
- MAE: 0.308
- RMSE: 0.378
- MAPE: 0.348%
- R²: 0.999

## Leaderboard snapshot

- linear_regression: RMSE 0.378, MAE 0.308, MAPE 0.348%
- ridge_regression: RMSE 0.566, MAE 0.488, MAPE 0.543%
- ridge_regression: RMSE 0.952, MAE 0.777, MAPE 0.861%
- ridge_regression: RMSE 1.877, MAE 1.544, MAPE 1.711%
- gradient_boosting: RMSE 3.653, MAE 3.115, MAPE 3.496%
- gradient_boosting: RMSE 3.819, MAE 3.256, MAPE 3.594%
- gradient_boosting: RMSE 4.000, MAE 3.363, MAPE 3.703%

## Local MLflow-style experiment tracking

- Tracking mode: local file-based experiment artifacts
- Runs table: outputs/experiments/runs.csv
- Registry decision: promoted
- Champion model: linear_regression
- Champion artifact: models/registry/champion_model.joblib

## Drift monitoring and retraining

- Drift method: population_stability_index
- Overall drift status: high
- Retraining triggered: True
- Retraining reasons: Data drift status is high.

## Forecast outputs

- Statewide forecast rows: 1
- Vendor forecast rows: 4
- Top vendor forecast: BrightStart Education with 208 predicted enrollments

## Interview note

This report is generated from synthetic public data. The V3 package demonstrates the production pattern: leakage-safe forecasting features, model comparison, time-series validation, local MLflow-style tracking, champion/challenger registry logic, monitoring, retraining policy, API serving, and stakeholder-ready documentation. In production, the same workflow would be connected to official CAPSDAC thresholds, MLflow or Vertex AI Model Registry, automated schedulers, access controls, and stakeholder sign-off.
