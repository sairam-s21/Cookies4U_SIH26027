"""Session 9: real train positions at a given date+time, for the
Dashboard map's moving-train marker.

Reuses `passenger_occupancy` (railblock.availability.corridor_availability
.build_passenger_occupancy) exactly as-is -- every row already says which
section a real train transits, when (start/end minute), and which
weekdays it runs (railblock.availability.service_frequency). This module
does not add or invent anything: it just filters those real rows to "is
this train inside this section at this exact minute, on this weekday" and
reports how far through the section it is (0.0 = just entered, 1.0 =
about to leave), for positioning a marker along the section's real
geographic line.
"""

from __future__ import annotations

from datetime import date as Date

import pandas as pd

MINUTES_PER_DAY = 24 * 60


def trains_at(passenger_occupancy: pd.DataFrame, the_date: Date, minute_of_day: float) -> list[dict]:
    """Every real train transit that is physically inside a corridor
    section at `minute_of_day` on `the_date`, per the real timetable +
    the assigned weekly running pattern (service_frequency.py)."""
    if passenger_occupancy.empty:
        return []
    weekday = the_date.weekday()

    out = []
    for _, row in passenger_occupancy.iterrows():
        wd_set = row.get("weekdays")
        if wd_set is not None and weekday not in wd_set:
            continue
        s, e = row["start_minute"], row["end_minute"]
        # normal same-day transit
        if s <= minute_of_day < e:
            progress = (minute_of_day - s) / (e - s) if e > s else 0.0
            out.append(_row(row, progress, s, e))
        # overnight spillover: this transit started yesterday, still running into today
        elif e > MINUTES_PER_DAY and (minute_of_day < e - MINUTES_PER_DAY):
            adj_s, adj_e = s - MINUTES_PER_DAY, e - MINUTES_PER_DAY
            if adj_s <= minute_of_day < adj_e:
                progress = (minute_of_day - adj_s) / (adj_e - adj_s) if adj_e > adj_s else 0.0
                out.append(_row(row, progress, adj_s, adj_e))
    return out


def enrich_many_with_live_status(trains: list[dict]) -> None:
    """Runs enrich_with_live_status for every train CONCURRENTLY (each
    lookup is an independent, short-timeout HTTP call, and httpx's
    blocking call releases the GIL while it waits on the network -- see
    railblock.scheduling.schedule_options for the same real-parallelism
    technique) so N trains cost roughly one lookup's wall-clock time, not
    N of them summed. Mutates each dict in place; never raises."""
    from concurrent.futures import ThreadPoolExecutor

    candidates = [t for t in trains if t["train_no"] in TRUSTED_LIVE_TRAIN_NOS]
    if not candidates:
        return
    with ThreadPoolExecutor(max_workers=min(16, len(candidates))) as pool:
        list(pool.map(enrich_with_live_status, candidates))


# Session 13 bug fix: the project's static timetable (Train_details_
# 22122017.csv) is from 2017. Indian Railways reassigns/renumbers train
# numbers over the years, so a live lookup by an old train_no can return
# a COMPLETELY different real train today (confirmed: train 11028 in our
# 2017 dataset sits near Chennai/Arakkonam, but RailRadar's live tracker
# reports today's 11028 as the MAS-CSMT Mumbai Express near Pune/
# Solapur -- both correct for their own data source, just about two
# different physical trains 9 years apart). distance_from_origin_km is
# only meaningful on this corridor's own km scale for a train whose REAL
# route actually follows MAS-CBE end to end -- which we can only trust
# for the one train this corridor's geometry was itself derived from
# (12243, and its RailRadar-confirmed return working 12244; see
# fetch_real_route_geometry.py). Every other train_no stays
# schedule-computed only, which also cuts live API calls from up to
# dozens per request down to at most 2, cached 60s.
TRUSTED_LIVE_TRAIN_NOS = frozenset({"12243", "12244"})


def enrich_with_live_status(train: dict) -> None:
    """Session 12/13: mutates `train` in place, overlaying a real
    RailRadar live-status lookup if one succeeds. RailRadar's live
    endpoint has no direct lat/lon (confirmed against a real response --
    see railradar.get_live_status's docstring); it gives a real
    distance-from-origin km figure instead, which the CALLER (e.g.
    Dashboard.jsx's trainMarkers) interpolates along the same real
    major-station geography already used for schedule-computed positions
    -- so `distance_from_origin_km`, not lat/lon, is what actually makes
    a train "live" here. Never raises, never removes the existing
    schedule-computed fields -- a failed/unavailable lookup leaves
    `train` exactly as it was (source stays "computed"). Only ever
    attempted for TRUSTED_LIVE_TRAIN_NOS -- see that constant's comment."""
    if train["train_no"] not in TRUSTED_LIVE_TRAIN_NOS:
        return
    from railblock.integrations.railradar import get_live_status  # local import: optional dependency

    live = get_live_status(train["train_no"])
    if live and "distance_from_origin_km" in live:
        train["distance_from_origin_km"] = live["distance_from_origin_km"]
        train["delay_minutes"] = live.get("delay_minutes")
        train["current_halt"] = live.get("current_halt")
        train["live_status"] = live.get("live_status")
        train["source"] = "live"


def _row(row, progress: float, start_minute: float, end_minute: float) -> dict:
    return {
        "train_no": str(row["train_no"]),
        "train_name": row.get("train_name", ""),
        "section_id": row["section_id"],
        "progress": round(float(progress), 4),
        "start_minute": round(float(start_minute), 2),
        "end_minute": round(float(end_minute), 2),
    }
