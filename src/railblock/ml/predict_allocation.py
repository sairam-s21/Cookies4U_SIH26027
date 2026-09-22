"""Inference for the adaptive-allocation regression (see
train_utilization_model.py for what it was trained on and why).

Wired into the live scheduling solve via
railblock.scheduling.adaptive_allocation.apply_adaptive_allocation, which
runs this on EVERY ranked task right after Whittle-index ranking and
BEFORE CP-SAT ever sees the batch (at explicit user direction) -- not as
a fallback tried only after a task's full duration has already failed to
fit. This module itself never assigns a window or acts on its own
initiative either way -- it only ever returns a number for a caller to
use.

WHY p70, not p50 (median) or the raw model mean: a task allocated at
exactly its predicted MEDIAN historical need would, by definition, run
short about half the time -- not a "safe" fallback at all. The p70
prediction means roughly 70% of similar historical tasks needed AT MOST
that fraction of their demanded time, which is a real safety margin --
chosen deliberately over the stricter p80 after comparing p50/p65/p70/
p75/p80 side by side (p80 was well-calibrated but barely ever
recommended a reduction at all, since the real data's right-skew means
its 80th percentile commonly sits at or above 100%; p70 offers a
materially larger reduction while still leaving real margin, and its R2
was actually better than p80's). Not an arbitrary default -- an earlier
framing that would have used the pooled median was rejected for the
same "not actually safe" reason.

SAFETY GUARDRAILS (explicit, matching this project's existing "solver
supremacy" / graceful-fallback pattern used everywhere else -- e.g.
railblock.integrations.railradar, railblock.scheduling.critical_task_escalation):
  - Never recommends MORE than the task's own full demanded duration.
    clip to <=100% of demanded -- a department/defect type the model has
    learned tends to overrun (e.g. Traction) will naturally get a p70
    prediction at or above 100%, which this clips to "no reduction
    offered", never "grant extra time".
  - Never recommends below MIN_SAFE_ALLOCATION_FRACTION of demanded -- a
    hard engineering floor, not a data-derived number, so a wild or
    degenerate prediction can never produce an unusably small allocation.
  - Every call is wrapped so it can never raise into a caller. A missing
    or corrupt model file, an unrecognised defect_type/department, or any
    other failure returns None -- the caller MUST already have a working
    fallback (the task's own unmodified demanded duration) and use it
    whenever this returns None, exactly like every other optional real-
    data enrichment in this codebase.
"""

from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from railblock.paths import UTILIZATION_MODEL_JOBLIB

MIN_SAFE_ALLOCATION_FRACTION = 0.10


@lru_cache(maxsize=1)
def _load_model() -> dict | None:
    if not UTILIZATION_MODEL_JOBLIB.exists():
        return None
    try:
        return joblib.load(UTILIZATION_MODEL_JOBLIB)
    except Exception:
        return None


def is_available() -> bool:
    return _load_model() is not None


def predict_safe_allocation(
    department: str,
    defect_type: str,
    criticality: str,
    demanded_hours: float,
) -> dict | None:
    """Returns a dict with the recommended reduced allocation, or None if
    the model isn't available or the prediction failed for any reason --
    callers MUST fall back to `demanded_hours` unchanged in that case,
    never treat None as zero or as an error to surface to the end user.

    Only calls the p70 model, not p50 -- at explicit user request, since
    this runs once per task in `apply_adaptive_allocation`'s loop and the
    p50 ("typical") prediction was never actually consumed by anything;
    it only ever fed an informational field nothing read. Halves the
    real per-task model-call cost with no behavior change.

    Return shape: {
        "safe_allocation_hours": float,  # what to actually try scheduling
        "safe_fraction": float,          # safe_allocation_hours / demanded_hours
        "p70_pct": float,                # raw model output, for display
    }
    """
    bundle = _load_model()
    if bundle is None or demanded_hours is None or demanded_hours <= 0:
        return None
    try:
        X = pd.DataFrame(
            [{
                "department": department,
                "defect_type": defect_type,
                "criticality": criticality,
                "demanded_block_hours": float(demanded_hours),
            }]
        )
        p70 = float(bundle["models"]["p70"].predict(X)[0])
    except Exception:
        return None

    safe_fraction = max(MIN_SAFE_ALLOCATION_FRACTION, min(1.0, p70 / 100.0))

    return {
        "safe_allocation_hours": round(demanded_hours * safe_fraction, 2),
        "safe_fraction": round(safe_fraction, 3),
        "p70_pct": round(p70, 1),
    }


def predict_safe_allocation_batch(rows: pd.DataFrame) -> pd.DataFrame | None:
    """Batched version of predict_safe_allocation, for
    apply_adaptive_allocation's per-request loop over every ranked
    task -- ONE call into the sklearn model instead of one per task.
    Profiling showed the per-row version was most of that step's cost
    (each single-row call was rebuilding a fresh 1-row DataFrame and
    re-entering the model's prediction pipeline from scratch, 70+ times
    per request). Identical math to the per-row version, just
    vectorized -- not an approximation.

    `rows` needs real department/defect_type/requester_priority/
    estimated_block_hours columns (the same shape every caller already
    has post-ranking). Returns a DataFrame aligned to a SUBSET of `rows`'
    index (only the rows with a real positive demand -- matching
    predict_safe_allocation's own "demanded_hours <= 0 -> skip" rule)
    with safe_allocation_hours/safe_fraction/p70_pct columns, or None if
    the model isn't loaded or the batch predict failed for ANY reason at
    all. Deliberately all-or-nothing on failure (never a partial result)
    -- the caller falls back to the row-by-row predict_safe_allocation()
    in that case, preserving this project's existing per-row error
    isolation: one malformed row silently degrading every other real
    task's prediction in a shared batch call would be a real regression,
    not just a missed optimization."""
    bundle = _load_model()
    if bundle is None:
        return None
    valid = rows[rows["estimated_block_hours"].notna() & (rows["estimated_block_hours"] > 0)]
    if valid.empty:
        return valid[[]].assign(safe_allocation_hours=[], safe_fraction=[], p70_pct=[])
    try:
        X = pd.DataFrame(
            {
                "department": valid["department"],
                "defect_type": valid["defect_type"],
                "criticality": valid["requester_priority"],
                "demanded_block_hours": valid["estimated_block_hours"].astype(float),
            }
        )
        p70 = bundle["models"]["p70"].predict(X)
    except Exception:
        return None

    demanded = valid["estimated_block_hours"].astype(float).to_numpy()
    safe_fraction = np.clip(p70 / 100.0, MIN_SAFE_ALLOCATION_FRACTION, 1.0)
    return pd.DataFrame(
        {
            "safe_allocation_hours": np.round(demanded * safe_fraction, 2),
            "safe_fraction": np.round(safe_fraction, 3),
            "p70_pct": np.round(p70, 1),
        },
        index=valid.index,
    )


if __name__ == "__main__":
    cases = [
        ("Engineering", "Rail fracture", "Critical", 4.0),
        ("Engineering", "Ballast deficiency", "Routine", 4.0),
        ("Signalling", "Signal lamp failure", "Moderate", 1.0),
        ("Signalling", "Interlocking relay fault", "Critical", 4.0),
        ("Traction", "OHE contact wire wear", "Critical", 4.0),
        ("Traction", "Earthing/bonding fault", "Routine", 1.5),
    ]
    print(f"{'department':<12} {'defect_type':<28} {'demanded':>8}  {'safe_hrs':>8}  {'safe_%':>7}  {'p70_%':>6}")
    for dep, defect, crit, hrs in cases:
        r = predict_safe_allocation(dep, defect, crit, hrs)
        if r is None:
            print(f"{dep:<12} {defect:<28} -- model unavailable --")
            continue
        print(f"{dep:<12} {defect:<28} {hrs:>8.2f}  {r['safe_allocation_hours']:>8.2f}  {r['safe_fraction']*100:>6.1f}%  {r['p70_pct']:>6.1f}")
