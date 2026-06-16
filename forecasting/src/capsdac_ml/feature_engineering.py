from __future__ import annotations

import numpy as np
import pandas as pd

# Interview-facing feature set for V3.  All target-dependent features are shifted
# before aggregation to avoid using the current/future month to predict itself.
FEATURES = [
    "lag_1",
    "lag_3",
    "lag_6",
    "lag_12",
    "rolling_3",
    "rolling_6",
    "rolling_12",
    "month_sin",
    "month_cos",
    "trend_index",
    "site_capacity",
    "capacity_utilization_lag1",
    "attendance_rate_lag1",
    "teacher_student_ratio_lag1",
    "county_population_index",
    "county_unemployment_proxy",
]

TARGET_COL = "TargetEnrollmentH3"
FORECAST_HORIZON_MONTHS = 3


def _stable_site_capacity(site_code: str, base_enrollment: float) -> int:
    """Deterministic synthetic capacity used only for public demo data."""
    code_tail = int(str(site_code)[-2:]) if str(site_code)[-2:].isdigit() else 10
    buffer = 18 + code_tail % 12
    return int(max(base_enrollment + buffer, base_enrollment * 1.18))


def add_synthetic_operational_context(df: pd.DataFrame) -> pd.DataFrame:
    """Add V3 operational/demographic context when public sample data lacks it.

    Production CAPSDAC would source these from official site capacity, staffing,
    attendance, and county context tables. The public repo creates deterministic
    synthetic values so the ML workflow is reproducible without confidential data.
    """
    out = df.copy()
    out["MonthDate"] = pd.to_datetime(out["SnapshotMonth"] + "-01")
    site_avg = out.groupby("PreschoolCDSCode")["EnrollmentCount"].transform("mean")

    if "SiteCapacity" not in out.columns:
        out["SiteCapacity"] = [
            _stable_site_capacity(site, avg) for site, avg in zip(out["PreschoolCDSCode"], site_avg)
        ]
    if "AttendanceRate" not in out.columns:
        month = out["MonthDate"].dt.month
        county_adjustment = out["County"].astype(str).map(lambda c: (sum(map(ord, c)) % 5) / 100)
        out["AttendanceRate"] = (0.89 + 0.03 * np.sin(2 * np.pi * month / 12) + county_adjustment).clip(0.82, 0.97)
    if "TeacherCount" not in out.columns:
        out["TeacherCount"] = np.maximum(1, np.ceil(out["EnrollmentCount"] / 12)).astype(int)
    if "CountyPopulationIndex" not in out.columns:
        county_rank = out["County"].astype(str).rank(method="dense").astype(int)
        out["CountyPopulationIndex"] = 100 + county_rank * 3 + (out["MonthDate"].dt.year - out["MonthDate"].dt.year.min()) * 1.5
    if "CountyUnemploymentProxy" not in out.columns:
        out["CountyUnemploymentProxy"] = 4.0 + (out["MonthDate"].dt.month % 6) * 0.15
    return out


def build_monthly_features(df: pd.DataFrame, horizon_months: int = FORECAST_HORIZON_MONTHS) -> pd.DataFrame:
    """Build leakage-safe site-month features for multi-step enrollment forecasting.

    The target is enrollment `horizon_months` ahead at the same preschool site.
    """
    out = add_synthetic_operational_context(df)
    out = out.sort_values(["PreschoolCDSCode", "MonthDate"]).reset_index(drop=True)

    site_group = out.groupby("PreschoolCDSCode", group_keys=False)
    for lag in [1, 3, 6, 12]:
        out[f"lag_{lag}"] = site_group["EnrollmentCount"].shift(lag)

    for window in [3, 6, 12]:
        out[f"rolling_{window}"] = site_group["EnrollmentCount"].transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=window).mean()
        )

    out["month"] = out["MonthDate"].dt.month
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out["trend_index"] = (out["MonthDate"].dt.year - out["MonthDate"].dt.year.min()) * 12 + out["month"]

    out["site_capacity"] = out["SiteCapacity"].astype(float)
    out["capacity_utilization"] = out["EnrollmentCount"] / out["site_capacity"].replace(0, np.nan)
    out["teacher_student_ratio"] = out["TeacherCount"] / out["EnrollmentCount"].replace(0, np.nan)
    out["capacity_utilization_lag1"] = site_group["capacity_utilization"].shift(1)
    out["attendance_rate_lag1"] = site_group["AttendanceRate"].shift(1)
    out["teacher_student_ratio_lag1"] = site_group["teacher_student_ratio"].shift(1)
    out["county_population_index"] = out["CountyPopulationIndex"].astype(float)
    out["county_unemployment_proxy"] = out["CountyUnemploymentProxy"].astype(float)

    out[TARGET_COL] = site_group["EnrollmentCount"].shift(-horizon_months)
    out["TargetMonth"] = (out["MonthDate"] + pd.DateOffset(months=horizon_months)).dt.strftime("%Y-%m")

    required = FEATURES + [TARGET_COL]
    return out.dropna(subset=required).reset_index(drop=True)
