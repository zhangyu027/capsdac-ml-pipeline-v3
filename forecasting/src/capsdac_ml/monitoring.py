from __future__ import annotations

from typing import Dict, Any
import numpy as np
import pandas as pd

from .feature_engineering import FEATURES


def population_stability_index(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    expected = pd.to_numeric(expected, errors="coerce").dropna()
    actual = pd.to_numeric(actual, errors="coerce").dropna()
    if expected.empty or actual.empty:
        return 0.0
    quantiles = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(quantiles) < 3:
        return 0.0
    exp_counts, _ = np.histogram(expected, bins=quantiles)
    act_counts, _ = np.histogram(actual, bins=quantiles)
    exp_pct = np.maximum(exp_counts / max(exp_counts.sum(), 1), 1e-6)
    act_pct = np.maximum(act_counts / max(act_counts.sum(), 1), 1e-6)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def drift_report(feature_df: pd.DataFrame, baseline_end: str = "2024-12-01", current_start: str = "2025-07-01") -> Dict[str, Any]:
    baseline = feature_df[feature_df["MonthDate"] <= pd.Timestamp(baseline_end)]
    current = feature_df[feature_df["MonthDate"] >= pd.Timestamp(current_start)]
    rows = []
    for feature in FEATURES + ["EnrollmentCount"]:
        psi = population_stability_index(baseline[feature], current[feature])
        rows.append(
            {
                "feature": feature,
                "psi": psi,
                "status": "high" if psi >= 0.25 else "moderate" if psi >= 0.10 else "stable",
                "baseline_mean": float(baseline[feature].mean()),
                "current_mean": float(current[feature].mean()),
            }
        )
    return {
        "baseline_end": baseline_end,
        "current_start": current_start,
        "method": "population_stability_index",
        "thresholds": {"stable": "<0.10", "moderate": "0.10-0.25", "high": ">=0.25"},
        "features": rows,
        "overall_status": "high" if any(r["status"] == "high" for r in rows) else "moderate" if any(r["status"] == "moderate" for r in rows) else "stable",
    }
