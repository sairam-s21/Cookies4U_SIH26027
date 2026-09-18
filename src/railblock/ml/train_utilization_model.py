"""Trains the adaptive-allocation regression model on the historical
block-utilization dataset (railblock.ml.historical_utilization).

TARGET: demand_efficiency_pct = actual_utilized_hours / demanded_block_hours * 100
-- deliberately NOT utilization_pct (= actual/SANCTIONED) that the
historical dataset itself stores. The live scheduler, when deciding
whether to try a reduced allocation for a task that doesn't fit, only
ever knows that task's DEMANDED hours -- there is no separate "sanctioned"
figure for a task that hasn't been through COA's grant process yet (that
process is exactly what this feature is trying to shortcut for the
tasks that would otherwise just sit in the waiting list). So the model
must be trained to predict relative to what's actually available at
inference time. demand_efficiency_pct = sanction_ratio * utilization_pct
algebraically, so it still carries both real, real-anchored effects
(how much COA typically grants, and how much of the grant gets used).

FEATURES: department, defect_type, criticality, demanded_block_hours.
No section_id -- this project has no real or reasoned basis for a
section-level effect on utilization, unlike the other three.

TWO quantile regressors (GradientBoostingRegressor, loss="quantile"),
not one plain point-estimate regressor:
  - p50 (median) -- informational only, "typical" historical efficiency.
  - p70 -- the one railblock.ml.predict_allocation actually uses for its
    "safe" recommendation. Reasoning: allocating a task at its predicted
    MEDIAN historical need would, by definition, run short about half the
    time -- not "safe" at all. The p70 prediction means roughly 70% of
    similar historical tasks needed AT MOST that fraction of their
    demanded time -- a real safety margin, chosen deliberately over p80
    after comparing p50/p65/p70/p75/p80 side by side in conversation: p80
    was well-calibrated but barely ever recommended a reduction at all
    (the real data's right-skew means its 80th percentile commonly sits
    at or above 100%), while p70 offers a materially larger reduction on
    non-overrun-prone defect types and still leaves real margin, with R2
    that was actually BETTER than p80's (the middle of this distribution
    turned out easier for the model to explain than either tail). See
    predict_allocation.py's docstring and PROGRESS.md for the full
    comparison that drove this choice.

Prints held-out test-set MAE, R2, and (for the quantile models)
*coverage* -- for a well-calibrated p70 model, ~70% of real held-out
rows should fall at or below the prediction. This is reported plainly
rather than just asserting the model "works" -- consistent with this
project's rule against fabricated accuracy claims for anything that
doesn't actually involve training (this is the one part of the project
that does, so it gets a real, honest evaluation).
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

from railblock.paths import HISTORICAL_UTILIZATION_CSV, UTILIZATION_MODEL_JOBLIB

FEATURES = ["department", "defect_type", "criticality", "demanded_block_hours"]
CATEGORICAL_FEATURES = ["department", "defect_type", "criticality"]
TARGET = "demand_efficiency_pct"

# alpha = which quantile of the target distribution this model predicts.
QUANTILES = {"p50": 0.5, "p70": 0.7}

# A demand_efficiency_pct above this is almost certainly a rare, extreme
# real overrun (see historical_utilization.py's own UTILIZATION_PCT_CLIP)
# rather than a representative case worth letting the quantile fit chase
# -- capped so a handful of tail rows don't distort the fitted quantiles.
TARGET_CLIP_MAX = 400.0


def _build_pipeline(alpha: float) -> Pipeline:
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)],
        remainder="passthrough",
    )
    model = GradientBoostingRegressor(loss="quantile", alpha=alpha, n_estimators=200, max_depth=3, random_state=13)
    return Pipeline([("pre", pre), ("model", model)])


def train_and_save(csv_path=None, out_path=None, test_size: float = 0.2, seed: int = 13) -> dict:
    csv_path = csv_path or HISTORICAL_UTILIZATION_CSV
    out_path = out_path or UTILIZATION_MODEL_JOBLIB

    df = pd.read_csv(csv_path)
    df = df[df["demanded_block_hours"] > 0].copy()
    df[TARGET] = (df["actual_utilized_hours"] / df["demanded_block_hours"] * 100).clip(upper=TARGET_CLIP_MAX)

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
    print(f"trained on {HISTORICAL_UTILIZATION_CSV}")
    print(f"saved to {UTILIZATION_MODEL_JOBLIB}")
    print()
    for name, m in metrics.items():
        target_q = QUANTILES[name]
        print(f"  {name} (target quantile={target_q}):  MAE={m['mae']:.1f}pp  R2={m['r2']:.3f}  coverage={m['coverage']:.1%} (target ~{target_q:.0%})")
