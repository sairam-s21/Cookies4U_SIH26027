"""Real geographic coordinates for the corridor's major stations, for an
OpenStreetMap-based map.

`datasets/india_railway_stations.csv` has real latitude/longitude for
every station by station_code. Only the major stations used in the
coarse corridor derivation (railblock.corridor.derive_stations) are
looked up here -- the other fine-grained points
(railblock.corridor.fine_stations) are signal cabins/halts with no
public lat/long. All codes below are verified present in the dataset.

The corridor scope is MAS-JTJ: the first 9 stations of the original
19-station MAS-CBE list (everything from TPT/CBF/CBE onward dropped).
"""

from __future__ import annotations

import json
from functools import lru_cache

import pandas as pd

from railblock.paths import DERIVED_DIR, STATIONS_CSV

REAL_ROUTE_GEOMETRY_JSON = DERIVED_DIR / "real_route_geometry.json"

MAJOR_STATION_CODES = [
    "MAS", "AJJ", "WJR", "MCN", "KPD", "GYM", "AB", "VN", "JTJ",
]


@lru_cache(maxsize=1)
def load_major_station_geo() -> dict[str, dict]:
    """{station_code: {lat, lon, station_name}} for the major corridor
    stations (MAS-JTJ), from the real dataset. Cached for the process
    lifetime -- this is small, static reference data."""
    df = pd.read_csv(STATIONS_CSV)
    df = df[df["station_code"].isin(MAJOR_STATION_CODES)]
    out = {}
    for _, row in df.iterrows():
        out[row["station_code"]] = {
            "lat": float(row["latitude"]),
            "lon": float(row["longitude"]),
            "station_name": row["station_name"],
        }
    return out


def interpolate_all_station_geo(fine_stations: pd.DataFrame) -> dict[str, dict]:
    """Positions every fine-grained station, not just the 9 majors with
    real public lat/lon, so maintenance highlighting on the map can be
    drawn at the actual sub-section granularity instead of spanning an
    entire major-to-major stretch (e.g. the whole 36km AJJ-WJR gap) when
    the maintenance actually only affects one small sub-section within
    it. Without this, any section not bounded by two majors (most of
    them -- there are 56 real fine sections between the 9 majors)
    couldn't be drawn correctly at all.

    Returns {station_code: {lat, lon, station_name, geo_source}} for
    EVERY station in `fine_stations` (all 57, not just the 9 majors):
    - the 9 majors keep their real dataset lat/lon (geo_source="real").
    - every other station gets a position linearly interpolated between
      its two bracketing majors, by real distance_km fraction along the
      straight line between them (geo_source="interpolated_between_majors")
      -- not a survey-accurate curve position (this corridor has no public
      lat/lon for these points at all, real or otherwise), but a stated,
      honest approximation that at least places each real sub-section at
      its own real proportional distance, rather than collapsing dozens of
      distinct real sections onto one oversized major-to-major line.
    """
    majors = load_major_station_geo()
    ordered = fine_stations.sort_values("distance_km").reset_index(drop=True)
    major_rows = ordered[ordered["station_code"].isin(majors)].reset_index(drop=True)

    out: dict[str, dict] = {}
    for code, g in majors.items():
        out[code] = {**g, "geo_source": "real"}

    for _, row in ordered.iterrows():
        code = row["station_code"]
        if code in out:
            continue
        km = row["distance_km"]
        # bracketing majors: the last major at/before this km, and the
        # first major at/after it (a fine station always lies between two
        # majors, or coincides with one, since majors are themselves a
        # subset of the fine list).
        before = major_rows[major_rows["distance_km"] <= km]
        after = major_rows[major_rows["distance_km"] >= km]
        if before.empty or after.empty:
            continue  # shouldn't happen given majors bracket the whole corridor, but never guess a position
        a = before.iloc[-1]
        b = after.iloc[0]
        a_geo, b_geo = majors[a["station_code"]], majors[b["station_code"]]
        span = b["distance_km"] - a["distance_km"]
        frac = 0.0 if span <= 0 else (km - a["distance_km"]) / span
        out[code] = {
            "lat": a_geo["lat"] + (b_geo["lat"] - a_geo["lat"]) * frac,
            "lon": a_geo["lon"] + (b_geo["lon"] - a_geo["lon"]) * frac,
            "station_name": row["station_name"],
            "geo_source": "interpolated_between_majors",
        }
    return out


@lru_cache(maxsize=1)
def load_real_route_geometry() -> list[list[float]] | None:
    """Real track-curve points ([lat, lon] pairs) fetched from RailRadar
    (see railblock.integrations.fetch_real_route_geometry) -- None if
    that one-time script hasn't been run yet, in which case the map
    falls back to straight lines between major stations."""
    if not REAL_ROUTE_GEOMETRY_JSON.exists():
        return None
    with open(REAL_ROUTE_GEOMETRY_JSON) as f:
        return json.load(f)
