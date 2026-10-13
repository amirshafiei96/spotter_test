#!/usr/bin/env python3
"""Train and use a posted-rate regression model for shipment data.

Examples
--------
Train and backtest:
    python posted_rate_predictor.py train --data train-test.csv --out-dir model_output

Predict new shipments:
    python posted_rate_predictor.py predict --model model_output/posted_rate_model.cbm \
        --input new_shipments.csv --output predictions.csv

The input CSV should contain the original predictor columns from train-test.csv.
`posted_rate` is the target column and is never used as a prediction feature.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGET = "posted_rate"
REQUIRED_FEATURES = [
    "load_id", "pickup", "delivery", "pickup_lat", "pickup_lon",
    "delivery_lat", "delivery_lon", "distance", "equipment", "weight",
    "date", "market_index", "quote_signal",
]
CATEGORICAL_FEATURES = ["pickup", "delivery", "equipment", "route"]
MODEL_PARAMS = {
    "depth": 8,
    "learning_rate": 0.05,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "l2_leaf_reg": 5,
    "random_seed": 42,
    "thread_count": 8,
    "random_strength": 0.3,
    "allow_writing_files": False,
    "verbose": False,
}


def make_features(raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Build features from original shipment columns; return X and parsed dates."""
    missing = [c for c in REQUIRED_FEATURES if c not in raw.columns]
    if missing:
        raise ValueError(
            "Input CSV is missing required columns: " + ", ".join(missing)
        )

    X = raw[REQUIRED_FEATURES].copy()
    dates = pd.to_datetime(X["date"], errors="coerce")
    if dates.isna().all():
        raise ValueError("The 'date' column could not be parsed as dates.")

    # The ID is treated as a numeric sequence rather than a high-cardinality category.
    X["load_number"] = pd.to_numeric(
        X["load_id"].astype(str).str.extract(r"(\d+)$", expand=False),
        errors="coerce",
    )
    X["date_ordinal"] = dates.map(
        lambda value: value.toordinal() if pd.notna(value) else np.nan
    )
    X["date_year"] = dates.dt.year
    X["date_month"] = dates.dt.month
    X["date_day"] = dates.dt.day
    X["date_dayofweek"] = dates.dt.dayofweek
    X["date_dayofyear"] = dates.dt.dayofyear
    X["date_weekofyear"] = dates.dt.isocalendar().week.astype(float)
    X["is_weekend"] = (dates.dt.dayofweek >= 5).astype(float)
    X["route"] = X["pickup"].astype(str) + "__TO__" + X["delivery"].astype(str)

    X = X.drop(columns=["load_id", "date"])
    for column in CATEGORICAL_FEATURES:
        X[column] = X[column].where(X[column].notna(), "__MISSING__").astype(str)
    for column in X.columns:
        if column not in CATEGORICAL_FEATURES:
            X[column] = pd.to_numeric(X[column], errors="coerce")
    return X, dates


def regression_metrics(actual: pd.Series, predicted: np.ndarray) -> dict:
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    absolute_error = np.abs(actual_array - predicted_array)
    denominator = np.maximum(np.abs(actual_array), 1e-9)
    return {
        "mae": float(mean_absolute_error(actual_array, predicted_array)),
        "rmse": float(mean_squared_error(actual_array, predicted_array) ** 0.5),
        "r2": float(r2_score(actual_array, predicted_array)),
        "mape_percent": float(np.mean(absolute_error / denominator) * 100.0),
        "wape_percent": float(absolute_error.sum() / max(np.abs(actual_array).sum(), 1e-9) * 100.0),
        "median_absolute_error": float(np.median(absolute_error)),
        "p95_absolute_error": float(np.quantile(absolute_error, 0.95)),
    }


def fit_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    iterations: int,
    X_valid: pd.DataFrame | None = None,
    y_valid: pd.Series | None = None,
    early_stopping_rounds: int | None = None,
) -> CatBoostRegressor:
    model = CatBoostRegressor(iterations=iterations, **MODEL_PARAMS)
    fit_kwargs = {"cat_features": CATEGORICAL_FEATURES, "verbose": False}
    if X_valid is not None and y_valid is not None:
        fit_kwargs["eval_set"] = (X_valid, y_valid)
        if early_stopping_rounds is not None:
            fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
    model.fit(X_train, y_train, **fit_kwargs)
    return model


def train(args: argparse.Namespace) -> None:
    data_path = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(data_path)
    if TARGET not in data.columns:
        raise ValueError(f"Training CSV must include target column '{TARGET}'.")
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    X, dates = make_features(data)
    valid_rows = data[TARGET].notna() & dates.notna()
    if not valid_rows.all():
        dropped = int((~valid_rows).sum())
        print(f"Dropping {dropped} row(s) with missing target or invalid date.")
        X = X.loc[valid_rows].reset_index(drop=True)
        dates = dates.loc[valid_rows].reset_index(drop=True)
        data = data.loc[valid_rows].reset_index(drop=True)
        y = data[TARGET].astype(float)
    else:
        y = data[TARGET].astype(float).reset_index(drop=True)
        X = X.reset_index(drop=True)
        dates = dates.reset_index(drop=True)

    unique_dates = np.array(sorted(dates.dropna().unique()))
    if len(unique_dates) < 5:
        raise ValueError("At least five unique dates are needed for a chronological holdout.")
    cutoff = pd.Timestamp(unique_dates[int(len(unique_dates) * 0.80)])
    train_mask = dates < cutoff
    valid_mask = dates >= cutoff
    if train_mask.sum() == 0 or valid_mask.sum() == 0:
        raise ValueError("Could not create non-empty chronological train/validation sets.")

    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_valid, y_valid = X.loc[valid_mask], y.loc[valid_mask]
    print(f"Rows: {len(data):,}; train: {len(X_train):,}; validation: {len(X_valid):,}")
    print(f"Chronological validation starts {cutoff.date()} and ends {dates.loc[valid_mask].max().date()}.")

    validation_model = fit_catboost(
        X_train, y_train, iterations=1500, X_valid=X_valid, y_valid=y_valid,
        early_stopping_rounds=100,
    )
    best_iteration_count = max(1, int(validation_model.best_iteration_) + 1)
    valid_predictions = validation_model.predict(X_valid)
    model_metrics = regression_metrics(y_valid, valid_predictions)

    # Transparent simple baseline: training-set median rate per distance unit.
    source_distances = pd.to_numeric(data.loc[train_mask, "distance"], errors="coerce")
    target_rates = y_train
    rate_per_distance = (target_rates / source_distances.replace(0, np.nan)).median()
    baseline_predictions = (
        pd.to_numeric(data.loc[valid_mask, "distance"], errors="coerce").to_numpy()
        * float(rate_per_distance)
    )
    baseline_metrics = regression_metrics(y_valid, baseline_predictions)

    holdout = data.loc[valid_mask, ["load_id", "date", "pickup", "delivery", "distance", TARGET]].copy()
    holdout["predicted_posted_rate"] = valid_predictions
    holdout["absolute_error"] = np.abs(holdout[TARGET].to_numpy() - valid_predictions)
    holdout["absolute_percentage_error_percent"] = (
        holdout["absolute_error"].to_numpy()
        / np.maximum(np.abs(holdout[TARGET].to_numpy()), 1e-9) * 100.0
    )
    holdout.to_csv(out_dir / "holdout_predictions.csv", index=False, float_format="%.4f")

    importance = pd.DataFrame({
        "feature": X.columns,
        "importance_percent": validation_model.feature_importances_,
    }).sort_values("importance_percent", ascending=False)
    importance.to_csv(out_dir / "feature_importance.csv", index=False, float_format="%.6f")

    # Once validation chooses the iteration count, retrain on all available labelled rows.
    final_model = fit_catboost(X, y, iterations=best_iteration_count)
    model_path = out_dir / "posted_rate_model.cbm"
    final_model.save_model(str(model_path))

    mae_improvement = (
        (baseline_metrics["mae"] - model_metrics["mae"]) / baseline_metrics["mae"] * 100.0
        if baseline_metrics["mae"] else None
    )
    report = {
        "target": TARGET,
        "training_rows": int(len(data)),
        "feature_count": int(X.shape[1]),
        "categorical_features": CATEGORICAL_FEATURES,
        "validation_method": "chronological holdout: earliest 80% of unique dates for fitting; latest 20% for validation",
        "validation_start_date": str(cutoff.date()),
        "validation_end_date": str(dates.max().date()),
        "train_rows_for_validation": int(train_mask.sum()),
        "validation_rows": int(valid_mask.sum()),
        "chosen_iterations": best_iteration_count,
        "validation_metrics_catboost": model_metrics,
        "validation_metrics_distance_only_baseline": baseline_metrics,
        "mae_improvement_vs_distance_only_baseline_percent": float(mae_improvement),
        "deployment_model_trained_on": "all labelled rows in the input training file",
        "deployment_model_path": model_path.name,
        "notes": [
            "posted_rate is the target and is excluded from all input features.",
            "Weight and market_index can contain missing values; CatBoost handles numeric missing values.",
            "The dataset covers historical dates only. Performance may change if future market conditions differ.",
            "Rare unusually high rates contribute disproportionately to RMSE; review high-value shipments separately.",
        ],
    }
    with (out_dir / "validation_report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("\nCatBoost chronological validation metrics:")
    for key, value in model_metrics.items():
        print(f"  {key}: {value:.4f}")
    print("\nDistance-only baseline metrics:")
    for key, value in baseline_metrics.items():
        print(f"  {key}: {value:.4f}")
    print(f"\nChosen iterations: {best_iteration_count}")
    print(f"Saved full-data model: {model_path}")
    print(f"Saved report, feature importance, and holdout predictions in: {out_dir}")


def predict(args: argparse.Namespace) -> None:
    model_path = Path(args.model)
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = CatBoostRegressor()
    model.load_model(str(model_path))

    raw = pd.read_csv(input_path)
    X, _ = make_features(raw)
    expected = list(model.feature_names_)
    missing = [c for c in expected if c not in X.columns]
    extra = [c for c in X.columns if c not in expected]
    if missing:
        raise ValueError("Feature engineering did not produce expected columns: " + ", ".join(missing))
    if extra:
        X = X.drop(columns=extra)
    X = X[expected]
    predictions = model.predict(X)

    # Return only the requested identifier and predicted rate.
    output = pd.DataFrame({
        "load_id": raw["load_id"],
        "predicted_rate": np.round(predictions, 2),
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Predicted {len(output):,} row(s). Saved only load_id and predicted_rate: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or apply a posted-rate prediction model.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Backtest and train using labelled historical rows.")
    train_parser.add_argument("--data", required=True, help="CSV with features and posted_rate target.")
    train_parser.add_argument("--out-dir", default="model_output", help="Directory to save model and reports.")
    train_parser.set_defaults(func=train)

    predict_parser = subparsers.add_parser("predict", help="Predict posted_rate for new rows.")
    predict_parser.add_argument("--model", required=True, help="Path to posted_rate_model.cbm.")
    predict_parser.add_argument("--input", required=True, help="New shipment CSV; posted_rate is not required.")
    predict_parser.add_argument("--output", default="predictions.csv", help="Output CSV path.")
    predict_parser.set_defaults(func=predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
