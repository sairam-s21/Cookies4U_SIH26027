"""Session 13: ONE-TIME script -- rebuilds real per-station schedule data
for up to 500 corridor trains from RailRadar, replacing the 2017 CSV's
stale timing for exactly the trains this fetch confirms, and merging real
running-days into the same real_running_days.json cache that
service_frequency.py's assign_weekdays() already checks.

Candidate list (built once, free, from local data only -- see the git
history for how data/derived/train_fetch_candidates.json was generated):
225 corridor trains confirmed by the Jan-2026 Southern Railway timetable-
changes PDF the user supplied (real, current, but that PDF's own running-
days column turned out to be a rendering artifact, not real data -- see
that investigation), plus 275 more of the corridor's remaining trains
picked by transit frequency (most likely to actually be seen on the map).

Each candidate costs exactly ONE RailRadar call (`/trains/{no}/live`,
which also feeds the live-tracking feature -- see railradar.py's
get_train_static_profile). A train's real route is checked against this
corridor's own station codes before being trusted at all -- this is the
same identity-validation that the train-11028 bug taught us is necessary
(the 2017 dataset's train numbers are not guaranteed to still mean the
same real train 9 years later). A train that fails validation is left
completely alone: no CSV row, no running-days entry, exactly its current
2017-CSV + statistical-assumption behaviour.

Resumable: progress is written to data/derived/train_fetch_progress.json
after EVERY single train, not batched at the end, so an interrupted run
(network blip, Ctrl+C, a real quota/rate-limit stop) never loses already-
confirmed trains or re-spends a call on them -- just run this again and it
picks up where it left off. Run:

    python -m railblock.integrations.fetch_real_train_data

Regenerates data/derived/real_train_details_2026.csv (same 12-column
shape as Train_details_22122017.csv, so it can be merged into the
timetable the same way -- see corridor/real_overrides.py) and updates
data/derived/real_running_days.json every run, from whatever is currently
in the progress cache -- safe to run partway through and still get a
usable (if incomplete) dataset out of it.
"""

from __future__ import annotations

import csv
import json
import time

from railblock.corridor.fine_stations import load_fine_corridor_stations
from railblock.integrations.railradar import get_train_static_profile, is_configured
from railblock.paths import (
    REAL_TRAIN_DETAILS_CSV,
    TRAIN_FETCH_CANDIDATES_JSON,
    TRAIN_FETCH_PROGRESS_JSON,
)

CSV_COLUMNS = [
    "Train No", "Train Name", "SEQ", "Station Code", "Station Name",
    "Arrival time", "Departure Time", "Distance",
    "Source Station", "Source Station Name",
    "Destination Station", "Destination Station Name",
]


def _minute_to_hhmmss(minute: float | None) -> str | None:
    if minute is None:
        return None
    wrapped = round((minute % 1440) * 60) / 60  # tolerate tiny float drift
    total_seconds = round(wrapped * 60)
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _load_progress() -> dict:
    if TRAIN_FETCH_PROGRESS_JSON.exists():
        with open(TRAIN_FETCH_PROGRESS_JSON) as f:
            return json.load(f)
    return {}


def _save_progress(progress: dict) -> None:
    with open(TRAIN_FETCH_PROGRESS_JSON, "w") as f:
        json.dump(progress, f, indent=2)


def _validate_and_normalize(profile: dict, corridor_codes: set[str]) -> dict | None:
    """A train is only trusted if its REAL route (per RailRadar, today)
    actually overlaps this corridor's own stations, in at least two
    places -- one shared station proves nothing (could be a junction any
    train passes through); two proves the train genuinely traverses part
    of this corridor. Returns a JSON-safe normalized profile, or None."""
    corridor_stops = [
        s for s in profile["stops"]
        if s["station_code"] in corridor_codes
        and (s["scheduled_arrival_minute"] is not None or s["scheduled_departure_minute"] is not None)
    ]
    if len(corridor_stops) < 2:
        return None
    corridor_stops.sort(key=lambda s: s["sequence"] if s["sequence"] is not None else 0)
    return {
        "train_no": profile["train_no"],
        "train_name": profile["train_name"],
        "run_days": sorted(profile["run_days"]) if profile["run_days"] else None,
        "stops": corridor_stops,
    }


def main() -> None:
    if not is_configured():
        print("RAILRADAR_API_KEY not set (see .env) -- nothing to do.")
        return
    if not TRAIN_FETCH_CANDIDATES_JSON.exists():
        print(f"{TRAIN_FETCH_CANDIDATES_JSON} not found -- build the candidate list first.")
        return

    with open(TRAIN_FETCH_CANDIDATES_JSON) as f:
        candidates_data = json.load(f)
    # Session 13: scoped down to the 225 PDF-confirmed trains only, per
    # explicit user decision -- the 275 additional by-frequency candidates
    # (candidates_data["extra_by_frequency"]) are deliberately excluded now.
    candidates = candidates_data["pdf_matched"]

    stations = load_fine_corridor_stations()
    corridor_codes = set(stations["station_code"])
    code_to_name = dict(zip(stations["station_code"], stations["station_name"]))

    progress = _load_progress()
    remaining = [t for t in candidates if t not in progress]
    print(f"{len(candidates)} candidates total, {len(progress)} already attempted, {len(remaining)} left to fetch.")

    confirmed_before = sum(1 for v in progress.values() if v.get("status") == "valid")

    for i, train_no in enumerate(remaining, 1):
        profile = get_train_static_profile(train_no)
        if profile is None:
            print(f"  [{i}/{len(remaining)}] {train_no}: fetch failed (network/rate-limit) -- will retry next run, stopping here.")
            break
        normalized = _validate_and_normalize(profile, corridor_codes)
        if normalized is None:
            progress[train_no] = {"status": "invalid_route"}
            print(f"  [{i}/{len(remaining)}] {train_no}: real route does not match this corridor -- discarded (like 11028).")
        else:
            progress[train_no] = {"status": "valid", "profile": normalized}
            print(f"  [{i}/{len(remaining)}] {train_no}: confirmed -- {normalized['train_name']}")
        _save_progress(progress)
        time.sleep(0.15)

    confirmed_after = sum(1 for v in progress.values() if v.get("status") == "valid")
    invalid = sum(1 for v in progress.values() if v.get("status") == "invalid_route")
    print(f"\nProgress: {confirmed_after} confirmed real corridor trains ({confirmed_after - confirmed_before} new this run), "
          f"{invalid} discarded (route mismatch), {len(candidates) - len(progress)} not yet attempted.")

    _write_outputs(progress, code_to_name)


def _write_outputs(progress: dict, code_to_name: dict) -> None:
    running_days_path = TRAIN_FETCH_PROGRESS_JSON.parent / "real_running_days.json"
    real_running_days = {}
    if running_days_path.exists():
        with open(running_days_path) as f:
            real_running_days = json.load(f)

    rows = []
    for train_no, entry in progress.items():
        if entry.get("status") != "valid":
            continue
        p = entry["profile"]
        if p["run_days"]:
            real_running_days[train_no] = p["run_days"]

        stops = p["stops"]
        source_code, dest_code = stops[0]["station_code"], stops[-1]["station_code"]
        for j, stop in enumerate(stops):
            arrival = stop["scheduled_arrival_minute"]
            departure = stop["scheduled_departure_minute"]
            if arrival is None:
                arrival = departure
            if departure is None:
                departure = arrival
            rows.append({
                "Train No": train_no,
                "Train Name": p["train_name"] or "",
                "SEQ": j + 1,
                "Station Code": stop["station_code"],
                "Station Name": code_to_name.get(stop["station_code"], ""),
                "Arrival time": _minute_to_hhmmss(arrival),
                "Departure Time": _minute_to_hhmmss(departure),
                "Distance": stop["distance_km"] if stop["distance_km"] is not None else "",
                "Source Station": source_code,
                "Source Station Name": code_to_name.get(source_code, ""),
                "Destination Station": dest_code,
                "Destination Station Name": code_to_name.get(dest_code, ""),
            })

    with open(REAL_TRAIN_DETAILS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {REAL_TRAIN_DETAILS_CSV} ({len(rows)} rows, {sum(1 for e in progress.values() if e.get('status') == 'valid')} trains).")

    with open(running_days_path, "w") as f:
        json.dump(real_running_days, f, indent=2)
    print(f"Wrote {running_days_path} ({len(real_running_days)} trains with confirmed real running days).")


if __name__ == "__main__":
    main()
