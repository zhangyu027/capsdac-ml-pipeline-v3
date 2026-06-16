import pandas as pd

from src.capsdac_ml.feature_engineering import build_monthly_features
from src.capsdac_ml.model_selection import expanding_window_splits, run_model_selection
from src.capsdac_ml.monitoring import drift_report


def test_expanding_window_splits_and_model_selection_run():
    raw = pd.read_csv("data/raw/capsdac_child_enrollment_sample.csv", dtype={"VendorNumber": str, "PreschoolCDSCode": str})
    features = build_monthly_features(raw)
    splits = expanding_window_splits(features, min_train_months=12)
    assert len(splits) > 0
    model, report, leaderboard = run_model_selection(features, max_param_sets_per_model=1)
    assert model is not None
    assert "selected_model" in report
    assert not leaderboard.empty
    assert {"model_name", "mae", "rmse", "mape", "r2"}.issubset(leaderboard.columns)


def test_drift_report_has_status():
    raw = pd.read_csv("data/raw/capsdac_child_enrollment_sample.csv", dtype={"VendorNumber": str, "PreschoolCDSCode": str})
    features = build_monthly_features(raw)
    report = drift_report(features)
    assert report["overall_status"] in {"stable", "moderate", "high"}
    assert len(report["features"]) > 0
