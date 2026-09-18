"""Inference for the delay-risk regression (see train_delay_model.py for
what it was trained on and why).

Meant to be wired into railblock.availability.corridor_availability.
compute_availability(): instead of trusting a train's timetabled
end-minute exactly, pad it by this module's predicted delay margin
before subtracting it from a section's free time -- Feature 1 of the
5-features change note. This module itself never touches availability
data directly; it only ever returns a number of minutes for a caller to
add.

WHY p70, not p50 or the raw model mean: the same reasoning as
predict_allocation.py's choice, applied to the opposite side of the
scheduling problem. A margin sized at the predicted MEDIAN delay would,
by definition, under-cover about half of real historical runs -- not a
safety margin at all. p70 means roughly 70% of comparable historical
observations were at or below this many minutes late.

HONEST CALIBRATION NOTE (train_delay_model.py's real held-out numbers):
coverage is well-calibrated (p70 covers ~69.9% of real held-out rows,
p50 ~50.6%) -- the property that actually matters for sizing a margin.
R2 is modest (p70 R2=0.088) -- train_type/station/day_of_week explain
real but limited variance in any one specific day's delay, which is
expected for genuinely noisy day-to-day operational data. Disclosed
plainly rather than implied to be more precise than it is, matching this
project's rule against fabricated accuracy claims.

SAFETY GUARDRAILS (same pattern as predict_allocation.py):
  - Never a negative margin. A predicted delay can be negative (a train
    that historically tends to run EARLY at a given station), but a
    margin padding a window forward can't meaningfully be negative --
    clipped to 0, never used to shrink a train's occupied interval.
  - Capped at MAX_MARGIN_MINUTES=25 minutes, at explicit user request
    after checking this against real data: among every real corridor-
    station-specific (train_type, station, day_of_week) combination the
    model actually produces, the highest real prediction is 26.1
    minutes (Express trains, several corridor stations) -- the cap sits
    just under that, not as an arbitrary round number. An EARLIER cap of
    60 minutes was never actually wrong, but was never actually
    triggered by a real corridor prediction either -- it only ever got
    hit by far-flung NON-corridor stations (e.g. Rajahmundry, Cuttack --
    real stops on these same trains' full real routes, tracked because
    etrain.info's history covers a whole journey, not just the corridor
    segment) that this project will never actually query.
  - Every call is wrapped so it can never raise into a caller. A missing
    model, an unrecognised train_type/station, or any other failure
    returns None -- the caller MUST already have a working fallback (the
    train's own unmodified timetabled time) and use it whenever this
    returns None.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from railblock.paths import DELAY_MODEL_JOBLIB

MAX_MARGIN_MINUTES = 25.0


@lru_cache(maxsize=1)
def _load_model() -> dict | None:
    if not DELAY_MODEL_JOBLIB.exists():
        return None
    try:
        import joblib
        return joblib.load(DELAY_MODEL_JOBLIB)
    except Exception:
        return None


def is_available() -> bool:
    return _load_model() is not None


def predict_delay_margin(train_type: str, station_code: str, day_of_week: int) -> dict | None:
    """Returns a dict with the recommended padding margin, or None if
    the model isn't available or the prediction failed for any reason --
    callers MUST fall back to the train's own unmodified timetabled time
    in that case, never treat None as "no delay expected".

    `day_of_week`: 0=Monday..6=Sunday, matching this project's existing
    convention (service_frequency.py).

    Return shape: {
        "margin_minutes": float,   # what to actually add, clipped >=0
        "p70_delay_minutes": float,  # raw model output, for display
    }
    """
    bundle = _load_model()
    if bundle is None:
        return None
    try:
        X = pd.DataFrame([{"train_type": train_type, "station_code": station_code, "day_of_week": int(day_of_week)}])
        p70 = float(bundle["models"]["p70"].predict(X)[0])
    except Exception:
        return None

    margin = max(0.0, min(MAX_MARGIN_MINUTES, p70))
    return {"margin_minutes": round(margin, 1), "p70_delay_minutes": round(p70, 1)}


def predict_delay_margin_batch(rows: pd.DataFrame) -> pd.DataFrame | None:
    """Batched version of predict_delay_margin, for a caller padding
    many (train, station, day) occupancy rows in one pass -- same
    all-or-nothing-on-failure contract as predict_allocation.py's
    predict_safe_allocation_batch (falls back to the per-row function on
    any failure, never a silently partial result).

    `rows` needs train_type, station_code, day_of_week columns. Returns
    a DataFrame aligned to `rows`' index with margin_minutes/
    p70_delay_minutes columns, or None if the model isn't loaded or the
    batch predict failed for any reason."""
    bundle = _load_model()
    if bundle is None or rows.empty:
        return None
    try:
        X = pd.DataFrame({
            "train_type": rows["train_type"],
            "station_code": rows["station_code"],
            "day_of_week": rows["day_of_week"].astype(int),
        })
        p70 = bundle["models"]["p70"].predict(X)
    except Exception:
        return None

    margin = np.clip(p70, 0.0, MAX_MARGIN_MINUTES)
    return pd.DataFrame(
        {"margin_minutes": np.round(margin, 1), "p70_delay_minutes": np.round(p70, 1)},
        index=rows.index,
    )


if __name__ == "__main__":
    cases = [
        ("Shtb", "MAS", 0),
        ("Shtb", "CBE", 4),
        ("SF", "SA", 2),
        ("Pass", "ED", 5),
        ("Exp", "JTJ", 6),
    ]
    print(f"{'train_type':<10} {'station':<8} {'day':>3}  {'margin_min':>10}  {'p70_min':>8}")
    for tt, station, dow in cases:
        r = predict_delay_margin(tt, station, dow)
        if r is None:
            print(f"{tt:<10} {station:<8} {dow:>3}  -- model unavailable --")
            continue
        print(f"{tt:<10} {station:<8} {dow:>3}  {r['margin_minutes']:>10.1f}  {r['p70_delay_minutes']:>8.1f}")
