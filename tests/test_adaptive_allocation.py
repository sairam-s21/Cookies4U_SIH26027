"""Session 14: the adaptive-allocation regression tier that runs on
every ranked task before CP-SAT ever sees the batch. Uses the REAL
trained model (data/derived/ml/block_utilization_regressor.joblib) --
no mocks -- consistent with this project's testing style elsewhere.
"""

import pandas as pd
import pytest

from railblock.ml.predict_allocation import is_available, predict_safe_allocation, predict_safe_allocation_batch
from railblock.paths import UTILIZATION_MODEL_JOBLIB
from railblock.scheduling.adaptive_allocation import apply_adaptive_allocation

pytestmark = pytest.mark.skipif(not UTILIZATION_MODEL_JOBLIB.exists(), reason="regression model not trained yet")


def _task(task_id, department, defect_type, priority, hours):
    return {
        "task_id": task_id,
        "department": department,
        "defect_type": defect_type,
        "requester_priority": priority,
        "estimated_block_hours": hours,
    }


def test_model_is_available():
    assert is_available()


def test_empty_input_returns_empty_output():
    out = apply_adaptive_allocation(pd.DataFrame(columns=["task_id", "department", "defect_type", "requester_priority", "estimated_block_hours"]))
    assert out.empty


def test_every_row_gets_demanded_block_hours_preserved():
    df = pd.DataFrame([
        _task("T1", "Engineering", "Rail fracture", "Critical", 4.0),
        _task("T2", "Traction", "OHE contact wire wear", "Critical", 4.0),
    ])
    out = apply_adaptive_allocation(df)
    assert (out["demanded_block_hours"] == df["estimated_block_hours"]).all()


def test_traction_overrun_prone_defect_never_trimmed():
    """Traction's real historical data shows an overrun tendency (median
    utilization >100%, see historical_utilization.py) -- the trained
    model should have learned this and never recommend a reduction for
    it, matching the "solver supremacy never grants extra, but also
    shouldn't remove real needed margin" framing."""
    df = pd.DataFrame([_task("T1", "Traction", "OHE contact wire wear", "Critical", 4.0)])
    out = apply_adaptive_allocation(df)
    row = out.iloc[0]
    assert row["adaptive_allocation_applied"] == False  # noqa: E712
    assert row["estimated_block_hours"] == 4.0


def test_predictable_engineering_defect_gets_trimmed():
    """Ballast deficiency is tagged as machine-based/predictable in
    historical_utilization.py's DEFECT_TIER (narrower spread, real
    tamping-productivity citation) -- should reliably get a real
    reduction at a large-enough demanded duration."""
    df = pd.DataFrame([_task("T1", "Engineering", "Ballast deficiency", "Routine", 4.0)])
    out = apply_adaptive_allocation(df)
    row = out.iloc[0]
    assert row["adaptive_allocation_applied"] == True  # noqa: E712
    assert row["estimated_block_hours"] < row["demanded_block_hours"]
    assert row["estimated_block_hours"] > 0


def test_reduction_never_exceeds_original_demand():
    df = pd.DataFrame([
        _task("T1", "Engineering", "Rail fracture", "Moderate", 3.5),
        _task("T2", "Signalling", "Signal lamp failure", "Routine", 1.0),
        _task("T3", "Signalling", "Interlocking relay fault", "Critical", 5.0),
    ])
    out = apply_adaptive_allocation(df)
    assert (out["estimated_block_hours"] <= out["demanded_block_hours"]).all()
    assert (out["estimated_block_hours"] > 0).all()


def test_batch_prediction_matches_per_row_prediction_exactly():
    """Session 25, at explicit user request for a large scheduling-time
    reduction: predict_safe_allocation_batch() does the same real math
    as calling predict_safe_allocation() once per row -- one sklearn
    .predict() call instead of many, never a different answer."""
    df = pd.DataFrame([
        _task("T1", "Engineering", "Rail fracture", "Critical", 4.0),
        _task("T2", "Traction", "OHE contact wire wear", "Critical", 4.0),
        _task("T3", "Engineering", "Ballast deficiency", "Routine", 4.0),
        _task("T4", "Signalling", "Signal lamp failure", "Moderate", 1.0),
    ])
    batch = predict_safe_allocation_batch(df)
    assert batch is not None
    for _, row in df.iterrows():
        per_row = predict_safe_allocation(row["department"], row["defect_type"], row["requester_priority"], row["estimated_block_hours"])
        batch_row = batch.loc[df.index[df["task_id"] == row["task_id"]][0]]
        assert per_row is not None
        assert batch_row["safe_allocation_hours"] == pytest.approx(per_row["safe_allocation_hours"])
        assert batch_row["safe_fraction"] == pytest.approx(per_row["safe_fraction"])
        assert batch_row["p70_pct"] == pytest.approx(per_row["p70_pct"])


def test_mixed_batch_only_flags_rows_actually_changed():
    df = pd.DataFrame([
        _task("T1", "Traction", "OHE contact wire wear", "Critical", 4.0),  # should NOT change
        _task("T2", "Engineering", "Ballast deficiency", "Routine", 4.0),  # should change
    ])
    out = apply_adaptive_allocation(df)
    unchanged = out[~out["adaptive_allocation_applied"]]
    changed = out[out["adaptive_allocation_applied"]]
    assert set(unchanged["task_id"]) == {"T1"}
    assert set(changed["task_id"]) == {"T2"}
    assert (unchanged["estimated_block_hours"] == unchanged["demanded_block_hours"]).all()
