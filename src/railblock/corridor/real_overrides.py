"""Merges real, RailRadar-confirmed per-station schedule data (see
railblock.integrations.fetch_real_train_data) into the 2017 static
timetable, for exactly the trains that fetch has validated against this
corridor's own real stations (see that script's identity-validation --
necessary because Indian Railways reassigns/renumbers train numbers over
the years, so an old train number's data cannot be trusted without
confirming it still refers to the same physical train; see
train_positions.py's TRUSTED_LIVE_TRAIN_NOS for the same issue with
train 11028).

A train with no entry in data/derived/real_train_details_2026.csv is
completely untouched here: same 2017 rows, same statistical weekday
assumption (service_frequency.assign_weekdays). This is purely
additive/replacing, never destructive to trains with no real answer.
"""

from __future__ import annotations

import pandas as pd

from railblock.paths import REAL_TRAIN_DETAILS_CSV


def apply_real_train_overrides(timetable_df: pd.DataFrame) -> pd.DataFrame:
    """Replace every row for a train present in REAL_TRAIN_DETAILS_CSV
    with that file's real rows; every other train's 2017 rows pass
    through unchanged. Returns timetable_df unchanged if the real dataset
    hasn't been fetched yet (see fetch_real_train_data.py)."""
    if not REAL_TRAIN_DETAILS_CSV.exists():
        return timetable_df

    real_df = pd.read_csv(REAL_TRAIN_DETAILS_CSV, dtype=str)
    if real_df.empty:
        return timetable_df

    real_df["SEQ"] = real_df["SEQ"].astype(int)
    real_df["Distance"] = pd.to_numeric(real_df["Distance"], errors="coerce").fillna(0).round().astype(int)

    real_train_nos = set(real_df["Train No"])
    remaining = timetable_df[~timetable_df["Train No"].isin(real_train_nos)]
    return pd.concat([remaining, real_df], ignore_index=True)
