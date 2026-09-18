"""Trains the delay-risk regression model on the REAL per-station delay
history (railblock.ml.historical_delay) -- Feature 1 of the 5-features
change note: padding a train's occupied interval in compute_availability()
by a predicted delay margin, instead of trusting its timetabled time
exactly.

TARGET: delay_minutes at a given station (how late the train typically
is by the time it reaches that station), not a ratio -- unlike the
adaptive-allocation model's demand_efficiency_pct, there's no equivalent
"demanded" denominator to express delay relative to.

FEATURES: train_type, station_code, day_of_week. Deliberately NOT raw
train_no as the primary signal (225 distinct values would let the model
just memorize individual train identities instead of learning
generalizable punctuality patterns, and would predict nothing useful for
any real corridor train outside this exact 225 the day this was trained).

Same TWO-quantile-regressor pattern as train_utilization_model.py
(GradientBoostingRegressor, loss="quantile"), for the same reason: a
train scheduled to depart at exactly its predicted MEDIAN delay would
still run over its buffer about half the time. p70 is what
predict_delay.py actually uses for the real margin; p50 is informational
only.

Prints held-out test-set MAE, R2, and quantile coverage plainly, same
honesty convention as the utilization model's training script -- this is
the one other part of the project that involves real model fitting, so
it gets a real, unembellished evaluation rather than an assumed "it
works".
"""

from __future__ import annotations

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from railblock.ml.historical_delay import FEATURES, TARGET, load_training_data
from railblock.paths import DELAY_MODEL_JOBLIB

CATEGORICAL_FEATURES = ["train_type", "station_code"]

QUANTILES = {"p50": 0.5, "p70": 0.7}


def _build_pipeline(alpha: float) -> Pipeline:
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)],
        remainder="passthrough",
    )
    model = GradientBoostingRegressor(loss="quantile", alpha=alpha, n_estimators=200, max_depth=3, random_state=13)
    return Pipeline([("pre", pre), ("model", model)])


def train_and_save(out_path=None, test_size: float = 0.2, seed: int = 13) -> dict:
    out_path = out_path or DELAY_MODEL_JOBLIB

    df = load_training_data()
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    models: dict[str, Pipeline] = {}
    metrics: dict[str, dict] = {}
    for name, alpha in QUANTILES.items():
        pipe = _build_pipeline(alpha)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        metrics[name] = {
            "mae": float(mean_absolute_error(y_test, preds)),
            "r2": float(r2_score(y_test, preds)),
            "coverage": float((y_test <= preds).mean()),  # target: ~alpha
        }
        models[name] = pipe

    joblib.dump(
        {"models": models, "features": FEATURES, "target": TARGET, "trained_rows": len(df), "quantiles": QUANTILES},
        out_path,
    )
    return metrics


if __name__ == "__main__":
    metrics = train_and_save()
    df = load_training_data()
    print(f"trained on {len(df)} real per-station delay observations")
    print(f"saved to {DELAY_MODEL_JOBLIB}")
    print()
    for name, m in metrics.items():
        target_q = QUANTILES[name]
        print(f"  {name} (target quantile={target_q}):  MAE={m['mae']:.1f}min  R2={m['r2']:.3f}  coverage={m['coverage']:.1%} (target ~{target_q:.0%})")
