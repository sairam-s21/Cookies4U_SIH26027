"""Session 26, at explicit user request: the delay-risk regression --
REAL per-station delay history (railblock.ml.historical_delay), trained
model (railblock.ml.train_delay_model), and its inference (railblock.ml.
predict_delay). Uses the REAL trained model artifact -- no mocks,
consistent with this project's testing style elsewhere (see
test_adaptive_allocation.py).
"""

import pandas as pd
import pytest

from railblock.ml.predict_delay import (
    MAX_MARGIN_MINUTES,
    is_available,
    predict_delay_margin,
    predict_delay_margin_batch,
)
from railblock.paths import DELAY_MODEL_JOBLIB

pytestmark = pytest.mark.skipif(not DELAY_MODEL_JOBLIB.exists(), reason="delay-risk model not trained yet")


def test_model_is_available():
    assert is_available()


def test_margin_is_never_negative_even_for_an_early_running_train_type():
    # Shatabdi services historically run early at some stations (real
    # data showed CBE at -9.2 min average) -- the margin must still floor
    # at 0, never shrink a train's occupied interval.
    r = predict_delay_margin("Shtb", "CBE", day_of_week=4)
    assert r is not None
    assert r["margin_minutes"] >= 0.0


def test_margin_never_exceeds_the_hard_cap():
    for train_type in ("Shtb", "SF", "Pass", "Exp", "GR"):
        for station in ("MAS", "CBE", "SA", "ED"):
            r = predict_delay_margin(train_type, station, day_of_week=2)
            assert r is not None
            assert 0.0 <= r["margin_minutes"] <= MAX_MARGIN_MINUTES


def test_unrecognised_inputs_never_raise_and_return_none_or_a_valid_result():
    # OneHotEncoder(handle_unknown="ignore") means an unseen category
    # still gets a real prediction (all-zero encoding for that column),
    # not an exception -- confirm that holds rather than assume it.
    r = predict_delay_margin("NotARealType", "ZZZ", day_of_week=0)
    assert r is None or 0.0 <= r["margin_minutes"] <= MAX_MARGIN_MINUTES


def test_batch_prediction_matches_per_row_prediction_exactly():
    rows = pd.DataFrame([
        {"train_type": "Shtb", "station_code": "MAS", "day_of_week": 0},
        {"train_type": "SF", "station_code": "SA", "day_of_week": 2},
        {"train_type": "Pass", "station_code": "ED", "day_of_week": 5},
    ])
    batch = predict_delay_margin_batch(rows)
    assert batch is not None
    for idx, row in rows.iterrows():
        per_row = predict_delay_margin(row["train_type"], row["station_code"], row["day_of_week"])
        assert per_row is not None
        assert batch.loc[idx, "margin_minutes"] == pytest.approx(per_row["margin_minutes"])
        assert batch.loc[idx, "p70_delay_minutes"] == pytest.approx(per_row["p70_delay_minutes"])


def test_batch_prediction_on_empty_input_returns_none():
    assert predict_delay_margin_batch(pd.DataFrame(columns=["train_type", "station_code", "day_of_week"])) is None


def test_day_of_week_actually_changes_the_prediction_for_at_least_one_case():
    # Not asserting a specific direction (real weekday-vs-weekend delay
    # patterns vary by train/station) -- just that day_of_week is a real,
    # used feature, not silently ignored by the pipeline.
    values = {
        predict_delay_margin("SF", "SA", day_of_week=d)["p70_delay_minutes"]
        for d in range(7)
    }
    assert len(values) > 1
