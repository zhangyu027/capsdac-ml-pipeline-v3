from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import RegressorMixin

from .feature_engineering import FEATURES, FORECAST_HORIZON_MONTHS, TARGET_COL


def generate_site_forecast(model: RegressorMixin, feature_df: pd.DataFrame, horizon_months: int = FORECAST_HORIZON_MONTHS) -> pd.DataFrame:
    """Generate site-level horizon forecasts from each site's latest feature row."""
    latest = feature_df.sort_values(["PreschoolCDSCode", "MonthDate"]).groupby("PreschoolCDSCode", as_index=False).tail(1).copy()
    latest["ForecastMonth"] = (latest["MonthDate"] + pd.DateOffset(months=horizon_months)).dt.strftime("%Y-%m")
    latest["ForecastHorizonMonths"] = horizon_months
    latest["PredictedEnrollment"] = model.predict(latest[FEATURES]).round().clip(min=0).astype(int)
    latest["CapacityRiskFlag"] = latest["PredictedEnrollment"] > latest["site_capacity"] * 0.95
    latest["StaffingNeedEstimate"] = (latest["PredictedEnrollment"] / 12).round().clip(lower=1).astype(int)
    cols = [
        "ForecastMonth", "ForecastHorizonMonths", "VendorNumber", "VendorName", "PreschoolCDSCode", "SiteName",
        "County", "FundingType", "site_capacity", "PredictedEnrollment", "CapacityRiskFlag", "StaffingNeedEstimate",
    ]
    return latest[cols].sort_values(["ForecastMonth", "VendorNumber", "PreschoolCDSCode"]).reset_index(drop=True)


def save_model(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
