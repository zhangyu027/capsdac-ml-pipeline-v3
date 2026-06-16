from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title="CAPSDAC Level 3 MLE Forecast Platform", layout="wide")
st.title("CAPSDAC Level 3 MLE Forecast Platform")
st.caption("Synthetic public demo: 3-month-ahead enrollment, capacity risk, staffing need, model registry, and monitoring")

metrics_path = ROOT / "outputs/metrics/model_metrics.json"
if not metrics_path.exists():
    st.warning("Run `make run` before opening the dashboard.")
    st.stop()

metrics = json.loads(metrics_path.read_text())
avg = metrics["time_series_cv_avg_metrics"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Selected Model", metrics["selected_model_name"])
c2.metric("RMSE", f"{avg['rmse']:.3f}")
c3.metric("MAE", f"{avg['mae']:.3f}")
c4.metric("MAPE", f"{avg['mape']:.3f}%")

st.subheader("Model Leaderboard")
st.dataframe(pd.read_csv(ROOT / "outputs/metrics/model_leaderboard.csv"), use_container_width=True)

st.subheader("Site Forecasts")
st.dataframe(pd.read_csv(ROOT / "outputs/forecasts/site_forecast.csv"), use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Drift Monitoring")
    st.json(json.loads((ROOT / "outputs/reports/drift_report.json").read_text()))
with col2:
    st.subheader("Champion Registry")
    st.json(json.loads((ROOT / "models/registry/model_registry.json").read_text()))

st.subheader("Retraining Decision")
st.json(json.loads((ROOT / "outputs/retraining/retraining_decision.json").read_text()))
