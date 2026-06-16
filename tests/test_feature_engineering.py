import pandas as pd
from src.capsdac_ml.feature_engineering import FEATURES, TARGET_COL, build_monthly_features


def test_build_monthly_features_has_v3_lags_and_target():
    months = pd.date_range("2020-01-01", periods=20, freq="MS")
    df = pd.DataFrame({
        "SnapshotMonth": [m.strftime("%Y-%m") for m in months],
        "VendorNumber": [1] * len(months),
        "VendorName": ["Vendor"] * len(months),
        "PreschoolCDSCode": ["A"] * len(months),
        "SiteName": ["Site"] * len(months),
        "County": ["Orange"] * len(months),
        "EnrollmentCount": list(range(50, 50 + len(months))),
        "FundingType": ["CSPP"] * len(months),
        "DataSource": ["test"] * len(months),
    })
    out = build_monthly_features(df)
    assert all(c in out.columns for c in FEATURES)
    assert TARGET_COL in out.columns
    assert "capacity_utilization_lag1" in out.columns
    assert len(out) > 0
