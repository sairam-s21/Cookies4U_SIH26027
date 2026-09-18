"""Layer 1 -- Corridor Availability Computation.

    Availability(section, date) = Section Time (24h) - (Passenger occupancy UNION Goods occupancy)

Session 26, at explicit user request (Feature 1 of the 5-features change
note): each passenger occurrence's real occupancy now runs a bit past
its timetabled end-minute, padded by the real delay-risk model's
predicted margin for that train's category/section/weekday (see
_delay_margin_minutes below) -- so a train's own real tendency to run
late is reflected in what actually counts as "occupied" on a section,
not just its timetabled duration. Falls back to the unmodified
timetabled time whenever the model, or a real category for that
specific train, isn't available -- never a hard dependency.

Passenger occupancy is derived from the REAL timetable
(Train_details_22122017.csv): for every train that stops at two or more of
the 19 corridor stations, the time it spends transiting between consecutive
corridor stops is attributed to every block section that transit passes
through, interpolated proportionally by distance when the train skips an
intermediate corridor halt (e.g. an express that stops at WJR then KPD
without stopping at MCN still physically occupies both the WJR-MCN and
MCN-KPD sections while passing through).

The timetable has no day-of-week/run-frequency column. Session 1/2 treated
every train as running literally every day (a documented simplification)
and Session 3 showed this collapses free time to near-zero on busy
sections. Session 3's Option 1 fix: railblock.availability.service_frequency
assigns each train a realistic weekly running pattern (a documented
statistical assumption, not a recovered per-train fact -- see that
module's docstring), and build_passenger_occupancy now attaches a
`weekdays` column recording which days each transit leg actually applies
to. compute_availability is weekday-aware: a transit only occupies a
section on the dates its train actually runs.

Goods occupancy comes from the synthetic generator
(railblock.synthetic.goods_forecast) and is already date-specific.

This module is a real, callable function -- not a static table: call
compute_availability(section_id, the_date, passenger_occupancy,
goods_occupancy) and get back actual free/occupied minute-windows for that
section on that date.
"""

from __future__ import annotations

import re
from datetime import date as Date, timedelta
from functools import lru_cache

import pandas as pd

from railblock.availability.service_frequency import ALL_WEEKDAYS, assign_weekdays
from railblock.ml.historical_delay import load_train_type_lookup
from railblock.ml.predict_delay import predict_delay_margin

MINUTES_PER_DAY = 24 * 60


@lru_cache(maxsize=1)
def _cached_train_type_lookup() -> dict[str, str]:
    """Session 26, at explicit user request: loaded once per process,
    not once per (section, date) call -- this dict covers 540 trains
    and doesn't change mid-run, so re-reading train_delay_history.csv on
    every single call would be pure waste (same reasoning as this
    module's existing per-request `cache` parameter, just at process
    scope instead of per-request scope). Returns {} if the file is
    missing entirely (e.g. a fresh checkout that hasn't run the delay-
    history fetch yet) -- every caller already treats "no type found"
    as the graceful no-padding case, so an empty dict degrades safely."""
    try:
        return load_train_type_lookup()
    except Exception:
        return {}


def _delay_margin_minutes(train_no, section_id: str, weekday: int) -> float:
    """Real delay-risk padding (Feature 1 of the 5-features change
    note), at explicit user request: instead of trusting a train's
    timetabled end-minute exactly, pad it by the delay-risk model's
    predicted margin for this train's real category, at this section's
    downstream station, on this real weekday -- so a chronically-late
    train doesn't quietly make a maintenance block's neighbour look
    more available than it actually is.

    `section_id` is always the FIXED ascending-km "FROM-TO" pair (see
    build_passenger_occupancy) regardless of which direction a given
    train is actually travelling -- per-row direction isn't preserved
    in passenger_occupancy, so this uses the section's second-named
    station as a pragmatic downstream proxy rather than the exact real
    direction-aware one. A known, disclosed simplification: the delay
    model's own accuracy (R2 ~0.09, per train_delay_model.py) already
    means this margin is a rough real signal, not a precise one, so an
    occasional wrong-direction station lookup is a second-order error
    on top of a real one, not a new correctness class of its own.

    Returns 0.0 (never negative, never raises) whenever the train's
    type isn't known, the model isn't trained yet, or anything else
    goes wrong -- the caller's fallback is simply the timetabled time,
    unmodified, exactly as if this function didn't exist."""
    train_type = _cached_train_type_lookup().get(str(train_no))
    if train_type is None:
        return 0.0
    to_code = section_id.split("-", 1)[-1]
    return _cached_margin_minutes(train_type, to_code, weekday)


@lru_cache(maxsize=100_000)
def _cached_margin_minutes(train_type: str, station_code: str, weekday: int) -> float:
    """Session 26 perf fix, after a real measured regression: the model's
    real input space is exactly (train_type, station_code, weekday) --
    a small, bounded, pure-function domain (540 types x 24 corridor
    stations x 7 days is at most ~90,000 combinations, and real usage
    hits a small fraction of that) -- yet the un-cached version called a
    fresh single-row sklearn .predict() (measured ~3ms) for every single
    passenger-occupancy ROW, in every _passenger_intervals() call, in
    every compute_availability() call. The full test suite went from
    ~3.5 minutes to 15+ minutes and still climbing before this was
    caught. Caching at exactly this granularity (not per train_no, which
    would miss the fact that many different real trains share the same
    type) turns every REPEAT of a (type, station, weekday) combination
    -- the overwhelming majority of real calls, since the same trains
    run the same corridor sections on the same handful of weekdays over
    and over -- into a dict lookup instead of a model call."""
    margin = predict_delay_margin(train_type, station_code, weekday)
    return margin["margin_minutes"] if margin is not None else 0.0

_TIME_RE = re.compile(r"^(\d{1,2}):(\d{2}):(\d{2})$")


def _time_to_minutes(t) -> float | None:
    if not isinstance(t, str):
        return None
    m = _TIME_RE.match(t.strip())
    if not m:
        return None
    h, mi, _s = (int(x) for x in m.groups())
    return float((h * 60 + mi) % MINUTES_PER_DAY)


def build_passenger_occupancy(
    timetable_df: pd.DataFrame, corridor_stations: pd.DataFrame
) -> pd.DataFrame:
    """Derive real passenger-train occupancy intervals per block section.

    Returns one row per (train, section) transit leg: section_id, train_no,
    start_minute (0-1439), end_minute (start_minute + duration; may exceed
    1439 to represent the transit spilling past midnight), source, weekdays
    (a frozenset of 0=Mon..6=Sun -- see service_frequency.py -- the days
    this particular train is modeled as actually running).
    """
    corridor_codes = set(corridor_stations["station_code"])
    code_to_km = dict(zip(corridor_stations["station_code"], corridor_stations["distance_km"]))
    ordered = corridor_stations.sort_values("sequence_order").reset_index(drop=True)

    # ascending-km section boundaries, matching build_block_sections' section_id convention
    section_bounds = [
        (
            ordered.iloc[i]["station_code"],
            ordered.iloc[i + 1]["station_code"],
            float(ordered.iloc[i]["distance_km"]),
            float(ordered.iloc[i + 1]["distance_km"]),
        )
        for i in range(len(ordered) - 1)
    ]

    df = timetable_df[timetable_df["Station Code"].isin(corridor_codes)]
    if df.empty:
        return pd.DataFrame(columns=["section_id", "train_no", "train_name", "start_minute", "end_minute", "source", "weekdays"])

    eps = 1e-6
    rows = []
    for train_no, g in df.groupby("Train No"):
        g = g.sort_values("SEQ")
        if len(g) < 2:
            continue
        train_name = g["Train Name"].iloc[0] if "Train Name" in g.columns else ""
        weekdays = assign_weekdays(train_no, train_name)
        stops = g.to_dict("records")
        for i in range(len(stops) - 1):
            s_from, s_to = stops[i], stops[i + 1]
            km_from = code_to_km[s_from["Station Code"]]
            km_to = code_to_km[s_to["Station Code"]]
            if km_from == km_to:
                continue

            depart = _time_to_minutes(s_from["Departure Time"])
            arrive = _time_to_minutes(s_to["Arrival time"])
            if depart is None or arrive is None:
                continue
            total_duration = arrive - depart
            if total_duration <= 0:
                total_duration += MINUTES_PER_DAY  # crossed midnight

            lo_km, hi_km = sorted([km_from, km_to])
            total_km = hi_km - lo_km
            direction = 1 if km_to > km_from else -1

            spanned = [
                sec
                for sec in section_bounds
                if sec[2] >= lo_km - eps and sec[3] <= hi_km + eps
            ]
            if direction < 0:
                spanned = list(reversed(spanned))

            cum_minute = depart
            for from_code, to_code, sec_from_km, sec_to_km in spanned:
                sec_km = abs(sec_to_km - sec_from_km)
                duration = total_duration * (sec_km / total_km) if total_km > eps else total_duration
                start_minute = cum_minute % MINUTES_PER_DAY
                end_minute = start_minute + duration
                rows.append(
                    {
                        "section_id": f"{from_code}-{to_code}",
                        "train_no": train_no,
                        "train_name": train_name,
                        "start_minute": start_minute,
                        "end_minute": end_minute,
                        "source": "real_timetable",
                        "weekdays": weekdays,
                    }
                )
                cum_minute += duration

    return pd.DataFrame(rows)


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted((float(s), float(e)) for s, e in intervals if e > s)
    if not intervals:
        return []
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [tuple(x) for x in merged]


def _passenger_intervals(
    passenger_occupancy: pd.DataFrame, section_id: str, the_date: Date | None = None
) -> list[tuple[float, float]]:
    """A transit only occupies a section on the dates its own train
    actually runs (see service_frequency.py), so this needs `the_date` to
    check each row's `weekdays` set. If passenger_occupancy has no
    `weekdays` column (e.g. hand-built test fixtures) or `the_date` is
    omitted, every row is treated as applying every day -- the old,
    unconditional behaviour -- so existing callers/tests are unaffected.

    A transit that spills past midnight also blocks the start of TODAY via
    YESTERDAY's occurrence -- fold that wraparound in explicitly, checking
    yesterday's weekday (a train can run yesterday but not today, or vice
    versa; the two occurrences are independent).

    Session 26, at explicit user request: each occurrence's end_minute
    is padded by the real delay-risk model's predicted margin for that
    train/section/weekday (see _delay_margin_minutes) BEFORE being added
    to `intervals` -- so a chronically-late train's real lateness is
    reflected in what counts as "occupied", not just its timetabled
    duration. Today's occurrence and yesterday's spillover occurrence
    get their OWN margins (different real weekdays can have genuinely
    different real delay patterns for the same train) -- computed only
    once `the_date` is known, since a margin needs a real weekday to
    look up; the weekday-agnostic (`the_date is None`) path is
    unaffected, exactly as it was before this model existed.
    """
    rows = passenger_occupancy[passenger_occupancy["section_id"] == section_id]
    has_weekdays = "weekdays" in passenger_occupancy.columns and the_date is not None
    if has_weekdays:
        today_wd = the_date.weekday()
        yesterday_wd = (today_wd - 1) % 7

    intervals = []
    for _, row in rows.iterrows():
        s, e = row["start_minute"], row["end_minute"]
        if has_weekdays:
            wd_set = row["weekdays"]
            today_applies = today_wd in wd_set
            yesterday_applies = yesterday_wd in wd_set
        else:
            today_applies = True
            yesterday_applies = True
        if today_applies:
            margin = _delay_margin_minutes(row["train_no"], section_id, today_wd) if has_weekdays else 0.0
            intervals.append((s, e + margin))
        if e > MINUTES_PER_DAY and yesterday_applies:
            margin = _delay_margin_minutes(row["train_no"], section_id, yesterday_wd) if has_weekdays else 0.0
            intervals.append((0.0, e - MINUTES_PER_DAY + margin))
    return intervals


def _goods_intervals(goods_occupancy: pd.DataFrame, section_id: str, date_str: str, prev_date_str: str) -> list[tuple[float, float]]:
    """Goods occupancy is date-specific; a movement starting late on the
    previous day can still spill into today's first few minutes."""
    intervals = []
    today = goods_occupancy[
        (goods_occupancy["section_id"] == section_id) & (goods_occupancy["date"] == date_str)
    ]
    for _, row in today.iterrows():
        intervals.append((row["start_minute"], row["end_minute"]))

    yesterday = goods_occupancy[
        (goods_occupancy["section_id"] == section_id) & (goods_occupancy["date"] == prev_date_str)
    ]
    for _, row in yesterday.iterrows():
        if row["end_minute"] > MINUTES_PER_DAY:
            intervals.append((0.0, row["end_minute"] - MINUTES_PER_DAY))

    return intervals


def compute_availability(
    section_id: str,
    the_date: Date,
    passenger_occupancy: pd.DataFrame,
    goods_occupancy: pd.DataFrame,
    cache: dict | None = None,
) -> dict:
    """Return free/occupied minute-windows for one section on one date.

    Both occupancy DataFrames use start_minute/end_minute (end may exceed
    1439 for overnight spillover), as produced by build_passenger_occupancy
    and railblock.synthetic.goods_forecast.generate_goods_forecast.

    Session 25, at explicit user request (real profiling showed this
    exact call, over every (section, date) pair, was being repeated up to
    3x per single scheduling request -- once for Whittle-index ranking's
    congestion calc, once for the critical-tasks-first Step A solve's own
    internal capacity computation, once for Step D's -- with byte-
    identical inputs and therefore byte-identical results every time,
    since passenger_occupancy/goods_occupancy never change mid-request):
    `cache` is an OPTIONAL dict the caller creates fresh per request and
    passes to every call in that request (see api/app.py) -- purely a
    memoization keyed on (section_id, date), never changing what's
    computed or returned, only how many times. Omit it (the default) for
    the exact old behavior -- always recomputed, safe for any one-off or
    cross-request caller that can't guarantee a stable occupancy pair.
    """
    date_str = the_date.isoformat() if hasattr(the_date, "isoformat") else str(the_date)
    if cache is not None:
        cache_key = (section_id, date_str)
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
    if hasattr(the_date, "isoformat"):
        prev_date_str = (the_date - timedelta(days=1)).isoformat()
    else:
        prev_date_str = ""

    intervals = _passenger_intervals(passenger_occupancy, section_id, the_date if hasattr(the_date, "weekday") else None)
    intervals += _goods_intervals(goods_occupancy, section_id, date_str, prev_date_str)

    merged = _merge_intervals(intervals)
    clipped = [(max(0.0, s), min(MINUTES_PER_DAY, e)) for s, e in merged if e > 0 and s < MINUTES_PER_DAY]

    free = []
    cursor = 0.0
    for s, e in clipped:
        if s > cursor:
            free.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < MINUTES_PER_DAY:
        free.append((cursor, MINUTES_PER_DAY))

    occupied_minutes = sum(e - s for s, e in clipped)
    free_minutes = MINUTES_PER_DAY - occupied_minutes

    result = {
        "section_id": section_id,
        "date": date_str,
        "occupied_intervals": clipped,
        "free_intervals": free,
        "free_minutes": round(free_minutes, 2),
        "occupied_minutes": round(occupied_minutes, 2),
        "total_minutes": MINUTES_PER_DAY,
    }
    if cache is not None:
        cache[(section_id, date_str)] = result
    return result
