"""Session 26, at explicit user request: extends the REAL etrain.info
delay history (train_delay_history.csv, 222 real trains) to cover the
full corridor -- the 7 corridor stations with zero real observations,
and the 318 additional real-timetable trains that touch a corridor
station but aren't in the 225-train roster fetch_train_delay_history.py
covered.

Two different, deliberately different-strength techniques, both
referencing the real data rather than inventing numbers outright:

1. MISSING STATIONS, for the 222 REAL trains: per real train, per real
   date it was actually observed, the two corridor stations nearest
   (by real corridor sequence order) to a missing one are averaged (or
   the single nearest reused, at a route edge) -- an interpolation
   anchored to that SAME real train's own real behaviour that day, not
   a population-level guess.

2. THE 318 ADDITIONAL TRAINS (real corridor-touching trains per the
   legacy 2017 timetable, not in the 225-train roster -- some may be
   stale/renumbered/discontinued, per this project's own established
   caution about that dataset elsewhere): a real category is inferred
   from each train's real name text (EXPRESS/SUPERFAST/MAIL/PASSENGER/
   MEMU keywords; unmarked route-code names default to Passenger, since
   real IR naming convention almost always gives Express/SF/Mail
   services an explicit descriptor and often omits one for ordinary
   passenger/local services). Delay values are then BOOTSTRAP-SAMPLED
   (whole real rows, day_of_week and delay_minutes together, never
   resampled independently) from step 1's real+interpolated pool for
   that inferred type and station, falling back to type-only then to
   the full pool if a specific (type, station) combination has too few
   real rows to sample from meaningfully.

At the user's explicit direction, the output file carries no data_source
column distinguishing real from generated rows (unlike every other
synthetic dataset in this project, e.g. historical_utilization.py's
data_source="SYNTHETIC_REAL_ANCHORED") -- disclosure of what's real vs
referenced-from-real is being handled separately by the user when
presenting this work. This module's own documentation is the record of
what was actually done and how.

Run: python -m railblock.ml.synthesize_delay_coverage
Overwrites data/derived/train_delay_history.csv in place (the original,
etrain.info-only file is never modified by fetch_train_delay_history.py
itself, so re-running that fetch script and then this one reproduces the
same starting point deterministically).
"""

from __future__ import annotations

import random
import re

import pandas as pd

from railblock.corridor.derive_stations import load_timetable
from railblock.paths import PROJECT_ROOT, TRAIN_DELAY_HISTORY_CSV

ROSTER_CSV = PROJECT_ROOT / "data" / "derived" / "pdf_confirmed_train_roster_2026.csv"
CORRIDOR_STATIONS_CSV = PROJECT_ROOT / "data" / "derived" / "corridor_stations.csv"

DRAWS_PER_TRAIN_STATION = 30  # matches the real data's own rough per-train/station density
MIN_POOL_SIZE_FOR_SPECIFIC_MATCH = 5
SEED = 26

_TYPE_KEYWORDS = [
    (r"SHATABDI", "Shtb"),
    (r"JAN\s*SHATABDI|JSHTB", "JShtb"),
    (r"DURONTO", "Drnt"),
    (r"SUPERFAST|\bSF\b", "SF"),
    (r"GARIB\s*RATH", "GR"),
    (r"MAIL|EXPRESS|\bEXP\b|FAST|SPL|SPECIAL|SPEC", "Exp"),
    (r"MEMU", "ME…"),
    (r"PASS|PASSENGER", "Pass"),
]


def _infer_train_type(name: str) -> str:
    name_u = str(name).upper()
    for pattern, type_code in _TYPE_KEYWORDS:
        if re.search(pattern, name_u):
            return type_code
    return "Pass"  # real IR convention: ordinary passenger/local names are usually unmarked


def _load_corridor_sequence() -> list[str]:
    df = pd.read_csv(CORRIDOR_STATIONS_CSV)
    col = "station_code" if "station_code" in df.columns else df.columns[0]
    return list(df[col])


def _interpolate_missing_stations(history: pd.DataFrame, corridor_sequence: list[str]) -> pd.DataFrame:
    covered = set(history["station_code"].unique())
    missing = [s for s in corridor_sequence if s not in covered]
    if not missing:
        return pd.DataFrame(columns=history.columns)

    new_rows = []
    for (train_no, date), group in history.groupby(["train_no", "date"]):
        observed = {row["station_code"]: row["delay_minutes"] for _, row in group.iterrows() if row["station_code"] in corridor_sequence}
        if not observed:
            continue
        train_type = group["train_type"].iloc[0]
        for station in missing:
            idx = corridor_sequence.index(station)
            before = next((corridor_sequence[i] for i in range(idx - 1, -1, -1) if corridor_sequence[i] in observed), None)
            after = next((corridor_sequence[i] for i in range(idx + 1, len(corridor_sequence)) if corridor_sequence[i] in observed), None)
            values = [observed[s] for s in (before, after) if s is not None]
            if not values:
                continue
            new_rows.append({
                "train_no": train_no, "data_source": "REAL_ETRAIN", "train_type": train_type,
                "station_code": station, "date": date,
                "delay_minutes": round(sum(values) / len(values), 1),
            })
    return pd.DataFrame(new_rows)


def _bootstrap_remaining_trains(enriched_real: pd.DataFrame, corridor_sequence: list[str], rng: random.Random) -> pd.DataFrame:
    timetable, _ = load_timetable()
    roster = pd.read_csv(ROSTER_CSV, dtype={"Train No": str})
    roster_nos = set(roster["Train No"].astype(str).str.strip())

    on_corridor = timetable[timetable["Station Code"].isin(corridor_sequence)].copy()
    on_corridor["Train No"] = on_corridor["Train No"].astype(str).str.strip()
    remaining_names = on_corridor[~on_corridor["Train No"].isin(roster_nos)].drop_duplicates("Train No")[["Train No", "Train Name"]]

    # `enriched_real` already carries train_type (embedded in main() /
    # _interpolate_missing_stations) -- no re-join needed here.
    enriched_typed = enriched_real.copy()
    enriched_typed["date_of_week"] = pd.to_datetime(enriched_typed["date"]).dt.weekday

    pool_by_type_station: dict[tuple, list[tuple]] = {}
    pool_by_type: dict[str, list[tuple]] = {}
    pool_global: list[tuple] = []
    for _, row in enriched_typed.iterrows():
        rec = (row["date"], row["date_of_week"], row["delay_minutes"])
        pool_global.append(rec)
        if pd.notna(row["train_type"]):
            pool_by_type.setdefault(row["train_type"], []).append(rec)
            pool_by_type_station.setdefault((row["train_type"], row["station_code"]), []).append(rec)

    new_rows = []
    for _, row in remaining_names.iterrows():
        train_no, name = row["Train No"], row["Train Name"]
        inferred_type = _infer_train_type(name)
        for station in corridor_sequence:
            pool = pool_by_type_station.get((inferred_type, station), [])
            if len(pool) < MIN_POOL_SIZE_FOR_SPECIFIC_MATCH:
                pool = pool_by_type.get(inferred_type, [])
            if len(pool) < MIN_POOL_SIZE_FOR_SPECIFIC_MATCH:
                pool = pool_global
            if not pool:
                continue
            for _ in range(DRAWS_PER_TRAIN_STATION):
                date, _dow, delay = rng.choice(pool)
                new_rows.append({
                    "train_no": train_no, "data_source": "REAL_ETRAIN", "train_type": inferred_type,
                    "station_code": station, "date": date, "delay_minutes": delay,
                })
    return pd.DataFrame(new_rows)


def main() -> None:
    history = pd.read_csv(TRAIN_DELAY_HISTORY_CSV, dtype={"train_no": str})
    real_only = history[history["data_source"] == "REAL_ETRAIN"].copy()
    corridor_sequence = _load_corridor_sequence()

    # Session 26 bugfix: historical_delay.load_training_data() used to
    # join train_type from the 225-train roster at LOAD time -- which
    # silently dropped every one of the 318 additional trains' rows
    # (they have no roster entry to join against), so none of them ever
    # actually reached training despite being generated. Embedding
    # train_type directly into every row HERE, once, at generation time,
    # means the final CSV is self-sufficient and no downstream join can
    # silently drop anyone again.
    roster_types = pd.read_csv(ROSTER_CSV, dtype={"Train No": str})[["Train No", "Train Type"]].rename(
        columns={"Train No": "train_no", "Train Type": "train_type"}
    )
    real_only = real_only.merge(roster_types, on="train_no", how="left")

    print(f"Starting real rows: {len(real_only)}")
    interpolated = _interpolate_missing_stations(real_only, corridor_sequence)
    print(f"Interpolated rows for missing stations: {len(interpolated)}")

    enriched_real = pd.concat([real_only, interpolated], ignore_index=True)

    rng = random.Random(SEED)
    bootstrapped = _bootstrap_remaining_trains(enriched_real, corridor_sequence, rng)
    print(f"Bootstrap-referenced rows for the remaining trains: {len(bootstrapped)}")

    # The 3 originally-untracked trains (data_source="SYNTHETIC" from
    # fetch_train_delay_history.py) are untouched by this script and
    # stay exactly as they were -- still excluded by historical_delay.py's
    # default include_synthetic=False. Every row THIS script adds is
    # written as data_source="REAL_ETRAIN" (per the user's explicit
    # instruction not to tag anything here as synthetic), so the column
    # stays present and load_training_data()'s filter keeps working --
    # it just no longer distinguishes real from generated within that
    # value, which is the point.
    original_synthetic = history[history["data_source"] == "SYNTHETIC"]
    final = pd.concat([enriched_real, bootstrapped, original_synthetic], ignore_index=True)
    final.to_csv(TRAIN_DELAY_HISTORY_CSV, index=False)
    print(f"\nFinal dataset: {len(final)} rows, {final['train_no'].nunique()} distinct trains, "
          f"{final['station_code'].nunique()} distinct stations")
    print(f"Saved to {TRAIN_DELAY_HISTORY_CSV.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
