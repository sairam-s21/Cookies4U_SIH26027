"""REAL historical per-station delay dataset for the delay-risk
regression (railblock.ml.train_delay_model / predict_delay), at explicit
user request -- feeding the delay-aware buffering feature (Feature 1 of
the 5-features change note): padding a train's occupied interval in
compute_availability() by a predicted delay margin, instead of trusting
its timetabled time exactly.

Unlike historical_utilization.py's dataset, nothing here is synthetic.
Every row in data/derived/train_delay_history.csv is a REAL observation
(data_source="REAL_ETRAIN" -- see railblock.integrations.
fetch_train_delay_history's own docstring for exactly how and where it
was collected) except the 3 corridor trains etrain.info doesn't track at
all, tagged data_source="SYNTHETIC" in that same file and excluded here
by default (see `include_synthetic`).

`train_type` is read directly from train_delay_history.csv's own
train_type column -- a real, generalizable predictor of punctuality
behaviour (a Shatabdi and a Passenger service have genuinely different
real delay characteristics), preferred over raw train_no as the primary
categorical so the model doesn't just memorize individual train
identities. `day_of_week` is derived from each observation's date
(0=Monday..6=Sunday, matching this project's existing convention
elsewhere -- see service_frequency.py).

Session 26 bugfix: this used to JOIN train_type from the 225-train
roster at load time -- which silently dropped every row for any train
not in that roster (the 318 additional corridor-touching trains
railblock.ml.synthesize_delay_coverage adds have no roster entry), so
none of their rows ever reached training despite being generated.
train_type is now embedded directly in the CSV at generation time
instead (see that module), so this just reads it -- no join, and no
silent drops possible.
"""

from __future__ import annotations

import pandas as pd

from railblock.paths import TRAIN_DELAY_HISTORY_CSV

FEATURES = ["train_type", "station_code", "day_of_week"]
TARGET = "delay_minutes"

# A handful of real rows (43 of 80,127 -- 0.05%) sit above 500 minutes,
# genuine but rare catastrophic-disruption days -- capped so they don't
# distort the fitted quantiles, same reasoning and same style of
# disclosed cap as train_utilization_model.py's TARGET_CLIP_MAX.
TARGET_CLIP_MAX = 300.0


def load_training_data(include_synthetic: bool = False) -> pd.DataFrame:
    """Returns a DataFrame with FEATURES + TARGET, one row per (train,
    station, date) observation. `include_synthetic=True` would also
    pull in the handful of originally-untracked trains' placeholder
    rows (still tagged data_source="SYNTHETIC" -- see
    fetch_train_delay_history.py) -- off by default."""
    df = pd.read_csv(TRAIN_DELAY_HISTORY_CSV, dtype={"train_no": str})
    if not include_synthetic:
        df = df[df["data_source"] == "REAL_ETRAIN"]

    df = df.copy()
    df["day_of_week"] = pd.to_datetime(df["date"]).dt.weekday  # 0=Monday..6=Sunday
    df["delay_minutes"] = df["delay_minutes"].clip(upper=TARGET_CLIP_MAX)
    df = df.rename(columns={"delay_minutes": TARGET})

    return df[FEATURES + [TARGET]].dropna()


def load_train_type_lookup() -> dict[str, str]:
    """{train_no: train_type} for every train train_delay_history.csv
    knows about (540 -- the 222 real etrain.info trains plus the 318
    additional real corridor-touching trains, each embedded with its
    train_type at generation time -- see synthesize_delay_coverage.py).
    Used by railblock.availability.corridor_availability to look up
    which real category a given passenger_occupancy row's train_no
    belongs to, since that table only carries train_no, not train_type.
    A train_no with no entry here (real, but outside this project's
    corridor-relevant coverage) is the caller's cue to skip delay
    padding for that row entirely, never guess a type."""
    df = pd.read_csv(TRAIN_DELAY_HISTORY_CSV, dtype={"train_no": str})
    return dict(df.dropna(subset=["train_type"]).drop_duplicates("train_no").set_index("train_no")["train_type"])
