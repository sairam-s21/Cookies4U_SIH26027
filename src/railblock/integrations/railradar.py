"""Session 12: optional real-time enrichment from the RailRadar API
(https://railradar.in/docs), added at the user's explicit request and
using a key they provided (read from .env via railblock.config -- never
committed, never sent to the frontend).

Honesty/robustness rules, deliberately strict because this is the
project's first real external network dependency with a real quota:
  - EVERY call is wrapped so it can never raise into a caller. Any
    failure (network error, timeout, non-200, malformed JSON, missing
    key) returns None -- callers MUST already have a real, working
    fallback (the existing schedule-computed position / assumed running
    days), and always use it when this returns None.
  - Live status is cached in-process for LIVE_CACHE_TTL_S so rapid
    repeated UI interaction (e.g. dragging a time slider) does not
    multiply real API calls -- the free tier is 1,000 requests/month.
  - This module was written without the ability to make a single live
    call against the real API from the sandbox this was authored in
    (the sandbox's network resets the TLS connection to railradar.in
    entirely -- confirmed with curl and Python's own HTTP client, while
    other HTTPS hosts worked fine, so this is an environment egress
    restriction, not a key or service problem). The request format
    below matches RailRadar's own published docs exactly; the response
    parsing is defensive (multiple candidate field names tried) because
    it could not be confirmed against a real response before this code
    was written. Verify against a real response the first time this is
    run outside that sandbox and adjust field names here if needed.
"""

from __future__ import annotations

import time

import httpx

from railblock.config import RAILRADAR_API_KEY

BASE_URL = "https://api.railradar.in/v1"
# Deliberately short: this is called once per visible train for a live UI
# toggle, potentially dozens of times per request (see
# railblock.availability.train_positions.enrich_many_with_live_status,
# which also runs these concurrently) -- a slow/unreachable API must fail
# fast, not make the page hang.
REQUEST_TIMEOUT_S = httpx.Timeout(connect=2.0, read=2.0, write=2.0, pool=2.0)
LIVE_CACHE_TTL_S = 60.0

_live_cache: dict[str, tuple[float, dict | None]] = {}


def is_configured() -> bool:
    return bool(RAILRADAR_API_KEY)


def _get(path: str) -> dict | None:
    if not RAILRADAR_API_KEY:
        return None
    try:
        resp = httpx.get(
            f"{BASE_URL}{path}",
            headers={"Authorization": f"Bearer {RAILRADAR_API_KEY}"},
            timeout=REQUEST_TIMEOUT_S,
        )
        resp.raise_for_status()
        body = resp.json()
    except Exception:
        return None
    if not isinstance(body, dict) or not body.get("success"):
        return None
    data = body.get("data")
    return data if isinstance(data, dict) else None


def get_live_status(train_no: str) -> dict | None:
    """Real-time status for one train, cached for LIVE_CACHE_TTL_S.

    CONFIRMED against a real response (2026-09-05, train 12243) -- there
    is no direct lat/lon field at all (the earlier guess assuming a
    "lat,lon" position string was wrong and has been removed). What the
    real response DOES give: `currentLocation.distanceFromOriginKm`, a
    real cumulative-distance-along-route figure on the same km scale as
    this corridor (their `train.distance` reads 493.2 km vs this
    project's own real 495 km MAS-CBE distance) -- so instead of a raw
    lat/lon, this returns that real km value and lets the SAME
    major-station interpolation already used for schedule-computed
    positions place it on the map (see
    railblock.availability.train_positions.enrich_with_live_status and
    frontend Dashboard.jsx's trainMarkers), rather than needing a second,
    separate positioning method.

    Returns {train_no, train_name, live_status, delay_minutes,
    current_halt, distance_from_origin_km} or None. `live_status` is
    RailRadar's own lifecycle string (e.g. "not-started", "at-station",
    "running" -- not this project's other "status" fields, kept as a
    distinct key to avoid confusion)."""
    now = time.time()
    cached = _live_cache.get(train_no)
    if cached and now - cached[0] < LIVE_CACHE_TTL_S:
        return cached[1]

    data = _get(f"/trains/{train_no}/live")
    result = None
    if data:
        loc = data.get("currentLocation") if isinstance(data.get("currentLocation"), dict) else {}
        result = {
            "train_no": str(data.get("trainNumber", train_no)),
            "train_name": data.get("trainName"),
            "live_status": data.get("status"),
            "delay_minutes": data.get("delayMinutes"),
            "current_halt": loc.get("stationName") or loc.get("stationCode"),
        }
        km = loc.get("distanceFromOriginKm")
        if isinstance(km, (int, float)):
            result["distance_from_origin_km"] = float(km)

    _live_cache[train_no] = (now, result)
    return result


def get_route_geometry(train_no: str) -> list[tuple[float, float]] | None:
    """Real track-curve geometry for one train's route, from RailRadar's
    "Train Route Geometry (GIS)" endpoint (GET /trains/{no}/route) --
    used one-time (see railblock.integrations.fetch_real_route_geometry)
    to replace the corridor map's straight-line approximation between
    major stations with the REAL physical track shape. Returns a list of
    (lat, lon) points, or None.

    CONFIRMED against a real response (2026-09-05, train 12243): the
    shape is {"success":true,"data":{"trainNumber":..., "format":"geojson",
    "geojson":{"type":"Feature","geometry":{"type":"LineString",
    "coordinates":[[lon,lat],...]}}}}. The unwrap loop below still walks
    through a couple of alternate wrapper keys defensively (in case a
    different train_no or a future API version nests it differently),
    but "geojson" -> "geometry" -> "coordinates" is the real, verified
    path, not a guess.
    """
    data = _get(f"/trains/{train_no}/route")
    if not data:
        return None

    node = data
    for key in ("geojson", "geometry", "route"):
        candidate = node.get(key) if isinstance(node, dict) else None
        if isinstance(candidate, dict):
            node = candidate
    if isinstance(node, dict) and node.get("type") == "Feature" and isinstance(node.get("geometry"), dict):
        node = node["geometry"]

    coords = node.get("coordinates") if isinstance(node, dict) else None
    if coords is None:
        for key in ("route", "points", "path"):
            if isinstance(data.get(key), list):
                coords = data[key]
                break
    if not isinstance(coords, list) or not coords:
        return None

    points = []
    for c in coords:
        try:
            if isinstance(c, dict):
                lat, lon = float(c["lat"]), float(c["lon"])
            else:
                # GeoJSON order is [lon, lat]
                lon, lat = float(c[0]), float(c[1])
            points.append((lat, lon))
        except (KeyError, IndexError, TypeError, ValueError):
            continue
    return points or None


_DAY_ABBR = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}


def _iso_to_minute_of_day(iso_ts) -> float | None:
    if not isinstance(iso_ts, str) or not iso_ts:
        return None
    try:
        from datetime import datetime

        dt = datetime.fromisoformat(iso_ts)
        return dt.hour * 60 + dt.minute + dt.second / 60
    except ValueError:
        return None


def get_train_static_profile(train_no: str) -> dict | None:
    """Session 13: one-time, per-train real profile for rebuilding this
    project's train dataset from RailRadar instead of the 2017 CSV -- see
    railblock.integrations.fetch_real_train_data (the one-time script that
    calls this). Deliberately a SEPARATE, un-cached function from
    get_live_status(): the fields extracted here (run days, per-station
    schedule) are day-invariant real facts meant to be fetched once and
    persisted, not live moment-in-time state.

    An earlier attempt at real running-days data guessed a bare
    `GET /trains/{no}` endpoint (see git history) that never once returned
    a successful response (429 "quota exceeded" even when the account's
    real remaining balance contradicted that, then a connection reset --
    consistent with it not being a real endpoint at all, not a rate
    limit). The real answer was sitting in the ALREADY-CONFIRMED-WORKING
    /live endpoint the whole time: CONFIRMED against a real response
    (2026-09-06, train 12243) at `data.train.runDays`, a list of lowercase
    3-letter weekday abbreviations (e.g. ["mon","wed","thu","fri","sat",
    "sun"] -- 6 days, skipping Tuesday). `data.route[]` gives each real
    stop's station code, sequence, distance-from-origin km, and
    scheduledArrival/scheduledDeparture as full ISO-8601 timestamps (IST).

    Returns None on any failure. Otherwise: {train_no, train_name,
    train_type, source_code, destination_code, total_distance_km,
    duration_minutes, run_days (frozenset[int] 0=Mon..6=Sun, or None if
    RailRadar didn't provide it), stops: [{station_code, sequence,
    distance_km, scheduled_arrival_minute, scheduled_departure_minute}]}.
    """
    data = _get(f"/trains/{train_no}/live")
    if not data:
        return None

    train = data.get("train") if isinstance(data.get("train"), dict) else {}

    run_days = None
    run_days_raw = train.get("runDays")
    if isinstance(run_days_raw, list):
        days = {
            _DAY_ABBR[d.strip().lower()]
            for d in run_days_raw
            if isinstance(d, str) and d.strip().lower() in _DAY_ABBR
        }
        if days:
            run_days = frozenset(days)

    stops = []
    for entry in data.get("route") or []:
        if not isinstance(entry, dict):
            continue
        code = entry.get("stationCode")
        if not code:
            continue
        stops.append(
            {
                "station_code": code,
                "sequence": entry.get("sequence"),
                "distance_km": entry.get("distance"),
                "scheduled_arrival_minute": _iso_to_minute_of_day(entry.get("scheduledArrival")),
                "scheduled_departure_minute": _iso_to_minute_of_day(entry.get("scheduledDeparture")),
            }
        )

    source = train.get("source") if isinstance(train.get("source"), dict) else {}
    destination = train.get("destination") if isinstance(train.get("destination"), dict) else {}

    return {
        "train_no": str(data.get("trainNumber", train_no)),
        "train_name": data.get("trainName"),
        "train_type": train.get("type"),
        "source_code": source.get("code"),
        "destination_code": destination.get("code"),
        "total_distance_km": train.get("distance"),
        "duration_minutes": train.get("duration"),
        "run_days": run_days,
        "stops": stops,
    }
