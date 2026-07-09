"""Evaluation helpers for overall and steering-region metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REGION_ORDER = ["Sharp left", "Left", "Straight", "Right", "Sharp right"]


def assign_steering_region(steering: float) -> str:
    if steering < -0.30:
        return "Sharp left"
    if steering < -0.05:
        return "Left"
    if steering <= 0.05:
        return "Straight"
    if steering <= 0.30:
        return "Right"
    return "Sharp right"


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "Prediction bias": float(np.mean(y_pred - y_true)),
    }


def regional_metrics(y_true, y_pred) -> pd.DataFrame:
    df = pd.DataFrame({"actual": np.asarray(y_true), "predicted": np.asarray(y_pred).reshape(-1)})
    df["region"] = df["actual"].apply(assign_steering_region)
    rows = []
    for region in REGION_ORDER:
        subset = df[df["region"] == region]
        if len(subset) == 0:
            continue
        rows.append({
            "Steering region": region,
            "Samples": len(subset),
            "MAE": mean_absolute_error(subset["actual"], subset["predicted"]),
            "RMSE": np.sqrt(mean_squared_error(subset["actual"], subset["predicted"])),
            "Mean actual": subset["actual"].mean(),
            "Mean predicted": subset["predicted"].mean(),
        })
    return pd.DataFrame(rows)
