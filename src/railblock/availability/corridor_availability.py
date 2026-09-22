"""Layer 1 -- Corridor Availability Computation.

    Availability(section, date) = Section Time (24h) - (Passenger occupancy UNION Goods occupancy)

Each passenger occurrence's occupancy runs a bit past its timetabled
end-minute, padded by the delay-risk model's predicted margin for that
train's category/section/weekday (see _delay_margin_minutes below) -- so
a train's own tendency to run late is reflected in what actually counts
as "occupied" on a section, not just its timetabled duration. Falls back
to the unmodified timetabled time whenever the model, or a category for
that specific train, isn't available -- never a hard dependency.

Passenger occupancy is derived from the REAL timetable
(Train_details_22122017.csv): for every train that stops at two or more of
the 19 corridor stations, the time it spends transiting between consecutive
corridor stops is attributed to every block section that transit passes
through, interpolated proportionally by distance when the train skips an
intermediate corridor halt (e.g. an express that stops at WJR then KPD
without stopping at MCN still physically occupies both the WJR-MCN and
MCN-KPD sections while passing through).

The timetable has no day-of-week/run-frequency column. Treating every
train as running literally every day collapses free time to near-zero on
busy sections, so railblock.availability.service_frequency instead
assigns each train a realistic weekly running pattern (a documented
statistical assumption, not a recovered per-train fact -- see that
module's docstring), and build_passenger_occupancy attaches a
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
import weakref
from datetime import date as Date, timedelta
from functools import lru_cache

import pandas as pd

from railblock.availability.service_frequency import ALL_WEEKDAYS, assign_weekdays
from railblock.ml.historical_delay import load_train_type_lookup
from railblock.ml.predict_delay import predict_delay_margin, predict_delay_margin_batch

MINUTES_PER_DAY = 24 * 60


@lru_cache(maxsize=1)
def _cached_train_type_lookup() -> dict[str, str]:
    """Loaded once per process, not once per (section, date) call -- this
    dict covers 540 trains and doesn't change mid-run, so re-reading
    train_delay_history.csv on every single call would be pure waste
    (same reasoning as this module's per-request `cache` parameter, just
    at process scope instead of per-request scope). Returns {} if the
    file is missing entirely (e.g. a fresh checkout that hasn't run the
    delay-history fetch yet) -- every caller already treats "no type
    found" as the graceful no-padding case, so an empty dict degrades
    safely."""
    try:
        return load_train_type_lookup()
    except Exception:
        return {}


def _delay_margin_minutes(train_no, section_id: str, weekday: int) -> float:
    """Instead of trusting a train's timetabled end-minute exactly, pads
    it by the delay-risk model's predicted margin for this train's
    category, at this section's downstream station, on this weekday --
    so a chronically-late train doesn't quietly make a maintenance
    block's neighbour look more available than it actually is.

    `section_id` is always the FIXED ascending-km "FROM-TO" pair (see
    build_passenger_occupancy) regardless of which direction a given
    train is actually travelling -- per-row direction isn't preserved
    in passenger_occupancy, so this uses the section's second-named
    station as a pragmatic downstream proxy rather than the exact
    direction-aware one. A known, disclosed simplification: the delay
    model's own accuracy (R2 ~0.09, per train_delay_model.py) already
    means this margin is a rough signal, not a precise one, so an
    occasional wrong-direction station lookup is a second-order error
    on top of an existing one, not a new correctness class of its own.

    Returns 0.0 (never negative, never raises) whenever the train's
    type isn't known, the model isn't trained yet, or anything else
    goes wrong -- the caller's fallback is simply the timetabled time,
    unmodified, exactly as if this function didn't exist."""
    train_type = _cached_train_type_lookup().get(str(train_no))
    if train_type is None:
        return 0.0
    to_code = section_id.split("-", 1)[-1]
    return _cached_margin_minutes(train_type, to_code, weekday)


# The model's input space is exactly (train_type, station_code, weekday)
# -- a small, bounded, pure-function domain (540 types x 24 corridor
# stations x 7 days is at most ~90,000 combinations, and real usage hits
# a small fraction of that -- 2,863 for this corridor's real timetable,
# confirmed by direct enumeration). A plain dict, not @lru_cache: a
# per-call cache only ever helps a REPEAT of a combination already seen
# during THIS process's life, so the very first request after every cold
# start (Render's free tier spins down after 15 minutes idle) still pays
# for every single miss -- and a fresh single-row sklearn .predict()
# (measured ~8ms once real pipeline/joblib overhead is included, not the
# bare model's own compute) run individually for all 2,863 of them
# measured 23+ real seconds, the dominant cost in a ~31s /requests/
# whittle-rank call. A dict lets prewarm_delay_margin_cache below fill in
# every combination this corridor's real timetable will ever need with
# ONE batched predict() call instead (measured 0.5s for the same 2,863
# rows -- batching amortizes the fixed per-call pipeline/joblib overhead
# across however many rows are predicted at once, so predicting them all
# together is not just parallel-per-row but genuinely cheaper in total).
_margin_cache: dict[tuple[str, str, int], float] = {}


def prewarm_delay_margin_cache(passenger_occupancy: pd.DataFrame) -> None:
    """Called once, by api.store.get_corridor_context() right after it
    builds `passenger_occupancy` -- that data is a process-lifetime
    singleton (see get_corridor_context's own docstring), so the full set
    of (train_type, station_code, weekday) combinations this corridor's
    real timetable could ever need is fixed for the life of the process
    too. Enumerating and batch-predicting all of them ONCE here, instead
    of leaving each one to be discovered (and predicted individually) the
    first time some request happens to need it, moves this cost out of
    every user-facing request entirely -- see _margin_cache's own
    comment for the real measured numbers this replaces."""
    type_lookup = load_train_type_lookup()
    if not type_lookup or passenger_occupancy.empty:
        return
    triples: set[tuple[str, str, int]] = set()
    for row in passenger_occupancy.itertuples(index=False):
        train_type = type_lookup.get(str(row.train_no))
        if train_type is None:
            continue
        to_code = row.section_id.split("-", 1)[-1]
        for weekday in range(7):
            triples.add((train_type, to_code, weekday))
    triples -= set(_margin_cache)
    if not triples:
        return
    ordered = sorted(triples)
    rows = pd.DataFrame(ordered, columns=["train_type", "station_code", "day_of_week"])
    result = predict_delay_margin_batch(rows)
    if result is None:
        return  # falls back to _delay_margin_minutes' own per-row path below
    for (train_type, station_code, weekday), margin in zip(ordered, result["margin_minutes"]):
        _margin_cache[(train_type, station_code, weekday)] = float(margin)


def _cached_margin_minutes(train_type: str, station_code: str, weekday: int) -> float:
    key = (train_type, station_code, weekday)
    cached = _margin_cache.get(key)
    if cached is not None:
        return cached
    # Cache miss: a combination prewarm_delay_margin_cache didn't cover
    # (e.g. a train/section added to the real timetable after this
    # process started, or the prewarm's own batch predict failing) --
    # falls back to a single-row predict so correctness never depends on
    # the prewarm having succeeded, only speed does.
    margin = predict_delay_margin(train_type, station_code, weekday)
    value = margin["margin_minutes"] if margin is not None else 0.0
    _margin_cache[key] = value
    return value

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


def _grouped_by_section(passenger_occupancy: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """passenger_occupancy[passenger_occupancy["section_id"] == section_id]
    is a full-table boolean-mask scan -- fine once, but compute_availability
    calls _passenger_intervals once per (section, date), so a real caller
    ranking/scheduling against a 7-day horizon repeats the SAME section's
    scan 7 times over, for an identical result every time (the filter
    doesn't depend on the date at all). Grouped once per real
    passenger_occupancy OBJECT, turning every one of those repeats into
    an O(1) dict lookup instead.

    Keyed by id(), with a weakref.finalize callback to evict that exact
    entry the moment THIS specific object is garbage collected -- a bare
    id()-keyed dict looked correct in isolation but broke for real under
    the full test suite: CPython can and does reuse a garbage-collected
    object's id() for an unrelated LATER object, so a short-lived test
    fixture DataFrame could silently inherit a completely different,
    stale grouping computed for whatever DataFrame previously died at
    that same address. A plain WeakKeyDictionary can't be used instead --
    it needs the key itself to be hashable, and DataFrame deliberately
    isn't (it's mutable) -- but a DataFrame IS weakly-referenceable, and
    that's all a finalize callback needs: it fires exactly once, exactly
    when this object's refcount hits zero, well before Python could ever
    hand its address to something new."""
    key = id(passenger_occupancy)
    cached = _grouped_by_section_cache.get(key)
    if cached is not None:
        return cached
    grouped = {sec: g for sec, g in passenger_occupancy.groupby("section_id")}
    _grouped_by_section_cache[key] = grouped
    weakref.finalize(passenger_occupancy, _grouped_by_section_cache.pop, key, None)
    return grouped


_grouped_by_section_cache: dict[int, dict[str, pd.DataFrame]] = {}


def _passenger_intervals(
    passenger_occupancy: pd.DataFrame, section_id: str, the_date: Date | None = None
) -> list[tuple[float, float]]:
    """A transit only occupies a section on the dates its own train
    actually runs (see service_frequency.py), so this needs `the_date` to
    check each row's `weekdays` set. If passenger_occupancy has no
    `weekdays` column (e.g. hand-built test fixtures) or `the_date` is
    omitted, every row is treated as applying every day, so callers that
    don't track weekdays still get sensible results.

    A transit that spills past midnight also blocks the start of TODAY via
    YESTERDAY's occurrence -- fold that wraparound in explicitly, checking
    yesterday's weekday (a train can run yesterday but not today, or vice
    versa; the two occurrences are independent).

    Each occurrence's end_minute is padded by the delay-risk model's
    predicted margin for that train/section/weekday (see
    _delay_margin_minutes) BEFORE being added to `intervals` -- so a
    chronically-late train's lateness is reflected in what counts as
    "occupied", not just its timetabled duration. Today's occurrence and
    yesterday's spillover occurrence get their OWN margins (different
    weekdays can have genuinely different delay patterns for the same
    train) -- computed only once `the_date` is known, since a margin
    needs a weekday to look up; the weekday-agnostic (`the_date is
    None`) path skips margin padding entirely.
    """
    rows = _grouped_by_section(passenger_occupancy).get(section_id)
    has_weekdays = "weekdays" in passenger_occupancy.columns and the_date is not None
    if has_weekdays:
        today_wd = the_date.weekday()
        yesterday_wd = (today_wd - 1) % 7

    intervals = []
    if rows is None:
        return intervals
    # itertuples(), not iterrows(): iterrows() builds a full pandas Series
    # per row (real, measured overhead -- see prewarm_delay_margin_cache's
    # sibling comment for this same class of fix), which this loop's own
    # profiling showed as the single largest remaining cost once the
    # delay-margin model calls were batched. Plain namedtuples are enough
    # here since every field is read by name, never assigned back.
    for row in rows.itertuples(index=False):
        s, e = row.start_minute, row.end_minute
        if has_weekdays:
            wd_set = row.weekdays
            today_applies = today_wd in wd_set
            yesterday_applies = yesterday_wd in wd_set
        else:
            today_applies = True
            yesterday_applies = True
        if today_applies:
            margin = _delay_margin_minutes(row.train_no, section_id, today_wd) if has_weekdays else 0.0
            intervals.append((s, e + margin))
        if e > MINUTES_PER_DAY and yesterday_applies:
            margin = _delay_margin_minutes(row.train_no, section_id, yesterday_wd) if has_weekdays else 0.0
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

    This exact call, over every (section, date) pair, can be repeated up
    to 3x per single scheduling request -- once for Whittle-index
    ranking's congestion calc, once for the critical-tasks-first Step A
    solve's own internal capacity computation, once for Step D's -- with
    byte-identical inputs and therefore byte-identical results every
    time, since passenger_occupancy/goods_occupancy never change
    mid-request. `cache` is an OPTIONAL dict the caller creates fresh
    per request and passes to every call in that request (see
    api/app.py) -- purely a memoization keyed on (section_id, date),
    never changing what's computed or returned, only how many times.
    Omit it (the default) to always recompute -- safe for any one-off or
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
