from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    metrics = json.loads((ROOT / "outputs/metrics/model_metrics.json").read_text())
    validation = json.loads((ROOT / "outputs/reports/data_validation_report.json").read_text())
    drift = json.loads((ROOT / "outputs/reports/drift_report.json").read_text())
    registry = json.loads((ROOT / "models/registry/model_registry.json").read_text())
    retraining = json.loads((ROOT / "outputs/retraining/retraining_decision.json").read_text())
    statewide = pd.read_csv(ROOT / "outputs/forecasts/statewide_forecast.csv")
    vendors = pd.read_csv(ROOT / "outputs/forecasts/vendor_forecast.csv")
    leaderboard = pd.read_csv(ROOT / "outputs/metrics/model_leaderboard.csv")

    selected_model = metrics.get("selected_model_name", "unknown")
    avg = metrics.get("time_series_cv_avg_metrics", {})
    top_rows = "\n".join(
        f"- {row.model_name}: RMSE {row.rmse:.3f}, MAE {row.mae:.3f}, MAPE {row.mape:.3f}%"
        for row in leaderboard.head(7).itertuples()
    )

    report = f"""# CAPSDAC 2.0 Level 3 MLE Forecast Run Summary

## Data validation

- Input rows: {validation['row_count']}
- Sites: {validation['site_count']}
- Vendors: {validation['vendor_count']}
- Snapshot range: {validation['snapshot_month_min']} to {validation['snapshot_month_max']}
- Missing required columns: {validation['missing_required_columns']}
- Duplicate site-month records: {validation['duplicate_grain_count']}
- Negative enrollment rows: {validation['negative_enrollment_count']}

## Forecasting problem

- Forecast target: 3-month-ahead preschool site enrollment
- Business outputs: monthly enrollment, program demand, site capacity risk, staffing need estimate
- Validation strategy: expanding-window monthly validation

## Model selection

- Selected model: {selected_model}
- MAE: {avg.get('mae', 0):.3f}
- RMSE: {avg.get('rmse', 0):.3f}
- MAPE: {avg.get('mape', 0):.3f}%
- R²: {avg.get('r2', 0):.3f}

## Leaderboard snapshot

{top_rows}

## Local MLflow-style experiment tracking

- Tracking mode: local file-based experiment artifacts
- Runs table: outputs/experiments/runs.csv
- Registry decision: {registry.get('decision')}
- Champion model: {registry.get('champion_model_name')}
- Champion artifact: {registry.get('champion_model_uri')}

## Drift monitoring and retraining

- Drift method: {drift.get('method', 'population_stability_index')}
- Overall drift status: {drift.get('overall_status')}
- Retraining triggered: {retraining.get('triggered')}
- Retraining reasons: {', '.join(retraining.get('trigger_reasons', []))}

## Forecast outputs

- Statewide forecast rows: {len(statewide)}
- Vendor forecast rows: {len(vendors)}
- Top vendor forecast: {vendors.iloc[0]['VendorName']} with {int(vendors.iloc[0]['PredictedEnrollment'])} predicted enrollments

## Interview note

This report is generated from synthetic public data. The V3 package demonstrates the production pattern: leakage-safe forecasting features, model comparison, time-series validation, local MLflow-style tracking, champion/challenger registry logic, monitoring, retraining policy, API serving, and stakeholder-ready documentation. In production, the same workflow would be connected to official CAPSDAC thresholds, MLflow or Vertex AI Model Registry, automated schedulers, access controls, and stakeholder sign-off.
"""
    output_path = ROOT / "outputs/reports/run_summary.md"
    output_path.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
