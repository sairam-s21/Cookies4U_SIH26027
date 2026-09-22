"""SYNTHETIC, fixed, never-regenerated dataset of already-granted historical
maintenance blocks.

Distinct from tasks.csv/maintenance_tasks.py's live waiting-list generator
in one important way: every row here already went through approval in the
past (this is not something users submit or the scheduler ever touches) --
it exists purely so the app has real-looking granted-block HISTORY to show
on day one, without waiting for a live demo session to accumulate its own.

Each row carries a real (date, start_minute, end_minute) GRANT window.
Whether a row counts as "completed", "currently active", or "approved but
upcoming" is NEVER stored here -- it's computed live by comparing that
window against the real current time (see railblock.api.app's
_classify_granted_history), so the same fixed file stays honestly correct
no matter when it's read, without ever being edited.

Design rationale for the current bucket layout: an earlier, narrower file
(30 always-completed rows + 20 spread across roughly a week around its
generation day) meant "Currently Active Blocks" and "Upcoming Blocks" on
the Dashboard went empty once that week passed. The dataset is instead
built around a much wider, SIH-evaluation-length horizon and two buckets:
  - PAST_BUCKET (50 rows): dated from PAST_RANGE_DAYS_BEFORE days before
    GENERATION_ANCHOR through the day before it -- always "completed",
    real variety for the Completed History page.
  - FUTURE_BUCKET (TASKS_PER_DAY x n_future_days rows): dated from
    GENERATION_ANCHOR through FUTURE_RANGE_END, PLUS a
    FUTURE_BUFFER_DAYS_PAST_END tail past that, so "5 upcoming" (GET
    /blocks/upcoming, the next 5 not-yet-started windows) never runs dry
    checking ON Nov 30 itself -- the buffer exists purely for that top-5
    view's own tail edge case, not to change the deck's own "through
    November" scope.

    An earlier design guaranteed a strict "3 simultaneously active,
    always" floor by stretching every task's window to 19-23 hours
    regardless of its defect type's own real duration. That worked, but
    became a visible, confusing discrepancy once a task's own details
    became inspectable: a user could see a task's real 1-hour-ish
    estimated_block_hours sitting next to a ~20-hour-wide calendar bar.
    The dataset now uses real, natural durations (DEFECT_DURATION_HOURS,
    maintenance_tasks.py -- the same per-defect-type triangular
    distribution the PAST bucket and the live waiting-list generator both
    already use). TASKS_PER_DAY (3) still gives every day multiple real
    tasks -- "each day just has active tasks" is still true, and "active
    blocks" (a real window covering the exact current moment) still shows
    up regularly -- but there is deliberately NO strict guarantee anymore
    that some block is active at every possible instant; that guarantee
    is what forced the unrealistic stretching in the first place, and
    realism was the priority.

    An earlier version of this module also placed every window at a
    PURELY random minute of the day, with no check against real train
    occupancy at all -- unlike every live-scheduled task, which Layer 1
    (railblock.availability.corridor_availability) only ever offers a
    window inside genuinely FREE time for. That let a user clicking into
    a granted-history block on a busy section (e.g. MAS-BBQ) see real
    trains visibly passing THROUGH the middle of a "maintenance" block --
    exactly the double-booking this whole system exists to prevent, just
    in a display-only dataset instead of a live one. Every row's window
    is now placed inside a REAL free interval for its own (section_id,
    date) -- the same compute_availability() Layer 1 already uses for
    live scheduling -- picking the largest free interval that day and
    clipping the task's natural duration down to fit it on the (real,
    occasional) days a section is too saturated for its full natural
    duration, rather than ever ignoring real train occupancy.

Run directly to (re)generate GRANTED_HISTORY_XLSX:
    python -m railblock.synthetic.granted_history
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from railblock.availability.corridor_availability import build_passenger_occupancy, compute_availability
from railblock.corridor.derive_stations import load_timetable
from railblock.corridor.fine_stations import load_fine_corridor_stations
from railblock.corridor.real_overrides import apply_real_train_overrides
from railblock.corridor.sections import build_block_sections
from railblock.paths import GRANTED_HISTORY_XLSX
from railblock.synthetic.goods_forecast import generate_goods_forecast
from railblock.synthetic.maintenance_tasks import (
    DEFECT_TYPES,
    DEPARTMENTS,
    SOURCE_SYSTEM,
    approval_path_for,
    generate_maintenance_tasks,
)

N_PAST = 50         # always "completed" -- Completed History variety
TASKS_PER_DAY = 3   # real, independently-timed tasks per future day -- "each day just has active tasks"

# task_id carries no distinguishing "GH-" prefix, so this module's own
# {source_system}-{n} numbering would otherwise collide with POST
# /demo/seed's raw generate_maintenance_tasks() numbering (also
# unprefixed, also starting at 1). Offsetting this module's own numbering
# well clear of /demo/seed's largest scenario (DEMAND_SCENARIOS'
# "stress_test", 72 tasks) keeps the two id spaces disjoint without
# reintroducing a visible prefix.
ID_OFFSET = 10000

PAST_RANGE_DAYS_BEFORE = 31  # "one month before" the generation anchor
FUTURE_RANGE_END_MONTH_DAY = (11, 30)  # "through November" -- SIH evaluation window
FUTURE_BUFFER_DAYS_PAST_END = 7  # tail past Nov 30 so "5 upcoming" never runs dry checking ON Nov 30 itself

# Every other timestamp in this corridor (real train timetables, block
# windows) is implicitly IST wall-clock time, but this deployment's host
# OS clock runs UTC -- a naive `datetime.now()`/`date.today()` would
# silently compare/generate against the wrong "now" (off by IST's fixed
# +5:30, no DST in India so this offset is always exact). Every
# real-current-time comparison in this module goes through `now_ist()`
# instead.
#
# `now_ist()` is public (not prefixed) because api/app.py's GET /now also
# uses it: the frontend trusting the VIEWER's own machine clock for
# "today" (Schedule.jsx, RecommendedScheduling.jsx, CorridorMapPage.jsx)
# can disagree with the server's clock by more than a day (e.g. the
# Monthly Schedule's "Today" button landing on the wrong month), so every
# real-current-time UI default comes from the server via that endpoint
# instead of the browser's own `new Date()`.
IST = ZoneInfo("Asia/Kolkata")


def now_ist() -> datetime:
    return datetime.now(IST).replace(tzinfo=None)


def _future_range_end(as_of: date) -> date:
    """November of `as_of`'s year -- rolls to next year if generation
    somehow happens after that date (never true in practice, but a static
    Nov-30-this-year constant would be silently wrong for a late-year
    regeneration)."""
    month, day = FUTURE_RANGE_END_MONTH_DAY
    end = date(as_of.year, month, day)
    return end if end >= as_of else date(as_of.year + 1, month, day)


def _place_in_free_time(
    rng: np.random.Generator,
    section_id: str,
    grant_date: date,
    duration_min: float,
    passenger_occupancy: pd.DataFrame,
    goods_occupancy: pd.DataFrame,
    avail_cache: dict,
) -> tuple[int, int]:
    """Real placement, not a blind random minute: picks the LARGEST real
    free interval compute_availability() (Layer 1's own real availability
    function) reports for this exact (section_id, date), and a random
    start within it. Clips `duration_min` down to that interval's own
    size on the (real, occasional) days a section is too saturated for
    the task's full natural duration -- never places a window bigger
    than the real free time actually available that day, and never
    overlaps a real train's passage the way a purely random minute
    could. Falls back to a nominal 15-minute window at midnight only in
    the practically-impossible case of zero free time at all that day.

    free_intervals' own bounds are real floats (actual train
    arrival/departure times), but this function must return whole
    minutes -- naively int()-truncating a fractional start (e.g. 465.88
    -> 465) rounds DOWN across the boundary, landing the block 0.88
    minutes into the train occupancy that free interval was supposed to
    exclude (measured against real data: 178 of 296 rows overlapped a
    real train this way before intervals were snapped). Every free
    interval is snapped to whole minutes FIRST -- ceil() the start,
    floor() the end -- so the integer interval used for placement is
    always a true SUBSET of the real free time, never spilling a
    fraction of a minute across either edge."""
    avail = compute_availability(section_id, grant_date, passenger_occupancy, goods_occupancy, cache=avail_cache)
    free = avail["free_intervals"]
    int_free = [(math.ceil(s), math.floor(e)) for s, e in free]
    int_free = [(s, e) for s, e in int_free if e > s]
    if not int_free:
        return 0, 15

    best_s, best_e = max(int_free, key=lambda iv: iv[1] - iv[0])
    interval_len = best_e - best_s
    actual_duration = min(int(duration_min), interval_len)
    actual_duration = max(15, actual_duration)
    actual_duration = min(actual_duration, interval_len)  # the 15-min floor itself might exceed a genuinely tiny gap

    slack = interval_len - actual_duration
    start_minute = best_s + (int(rng.integers(0, slack + 1)) if slack > 0 else 0)
    end_minute = min(start_minute + actual_duration, best_e, 1439)
    return start_minute, end_minute


def generate_granted_history(as_of: date | None = None, seed: int = 20260917) -> pd.DataFrame:
    """Builds the dataset: N_PAST completed rows plus TASKS_PER_DAY rows
    for every day from `as_of` through the buffered end of the future
    range. `as_of` anchors both buckets (defaults to real today, IST) --
    pass a fixed value only for a reproducible test, never for the real
    generated file (that one should anchor to the real day it's generated
    on)."""
    if as_of is None:
        as_of = now_ist().date()
    rng = np.random.default_rng(seed)

    stations = load_fine_corridor_stations()
    sections = build_block_sections(stations)

    # Real passenger occupancy (the exact same source live scheduling
    # uses -- a weekday-pattern timetable, not date-specific, so one
    # build covers every date below) and real goods occupancy across
    # this file's WHOLE real date span, so _place_in_free_time can check
    # every row's placement against genuine train/freight occupancy.
    timetable, _ = load_timetable()
    timetable = apply_real_train_overrides(timetable)
    passenger_occupancy = build_passenger_occupancy(timetable, stations)

    past_start = as_of - timedelta(days=PAST_RANGE_DAYS_BEFORE)
    future_end = _future_range_end(as_of) + timedelta(days=FUTURE_BUFFER_DAYS_PAST_END)
    future_days = [as_of + timedelta(days=d) for d in range((future_end - as_of).days + 1)]
    n_future_tasks = len(future_days) * TASKS_PER_DAY
    n_tasks = N_PAST + n_future_tasks

    total_days = (future_end - past_start).days + 1
    goods_occupancy = generate_goods_forecast(sections, past_start, n_days=total_days, seed=seed)
    avail_cache: dict = {}

    # generate_maintenance_tasks gives realistic department/section/
    # defect_type/priority/hours/approval_path -- reused as-is rather than
    # re-deriving the same real-anchored logic a second time. as_of_date
    # here only affects due_date/days_overdue math inside that function,
    # not the grant date/time this module assigns below.
    base = generate_maintenance_tasks(sections, n_tasks=n_tasks, as_of_date=as_of, seed=seed)
    base = base.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # shuffle so past/future buckets aren't correlated with department order

    rows: list[dict] = []

    # ---------------------------------------------------------------
    # Past bucket: rows 0..N_PAST-1, one month before `as_of` through
    # yesterday. These only need to read as "completed", never "active",
    # but still get a REAL, train-occupancy-aware placement -- a
    # "granted" block in the past should look exactly as real as a live
    # one would have.
    # ---------------------------------------------------------------
    for i in range(N_PAST):
        task = base.iloc[i]
        days_ago = int(rng.integers(1, PAST_RANGE_DAYS_BEFORE + 1))
        grant_date = as_of - timedelta(days=days_ago)
        duration_min = max(15, round(task["estimated_block_hours"] * 60))
        start_minute, end_minute = _place_in_free_time(
            rng, task["section_id"], grant_date, duration_min, passenger_occupancy, goods_occupancy, avail_cache
        )
        rows.append(_row(task, i, grant_date, start_minute, end_minute))

    # ---------------------------------------------------------------
    # Future bucket: rows N_PAST..n_tasks-1, `as_of` through the
    # buffered future_end inclusive. Every day gets TASKS_PER_DAY real
    # tasks, each using its OWN real estimated_block_hours (the same
    # per-defect-type triangular duration generate_maintenance_tasks
    # already assigned it -- no artificial stretching), placed in real
    # free time exactly like the past bucket above.
    # ---------------------------------------------------------------
    task_i = N_PAST
    for grant_date in future_days:
        for _task_of_day in range(TASKS_PER_DAY):
            task = base.iloc[task_i]
            duration_min = max(15, round(task["estimated_block_hours"] * 60))
            start_minute, end_minute = _place_in_free_time(
                rng, task["section_id"], grant_date, duration_min, passenger_occupancy, goods_occupancy, avail_cache
            )
            rows.append(_row(task, task_i, grant_date, start_minute, end_minute))
            task_i += 1

    return pd.DataFrame(rows)


def _row(task: pd.Series, i: int, grant_date: date, start_minute: int, end_minute: int) -> dict:
    # `task` (generate_maintenance_tasks()'s own row) already carries
    # splittable/raised_date/due_date/days_overdue, real and already
    # computed -- all four must be passed through here or a
    # granted-history block's details modal shows "-" for them even
    # though the underlying data exists.
    return {
        "task_id": f"{task['source_system']}-{i + ID_OFFSET + 1:05d}",
        "department": task["department"],
        "section_id": task["section_id"],
        "defect_type": task["defect_type"],
        "requester_priority": task["requester_priority"],
        "estimated_block_hours": task["estimated_block_hours"],
        "raised_date": task["raised_date"],
        "due_date": task["due_date"],
        "days_overdue": task["days_overdue"],
        "splittable": bool(task["splittable"]),
        "date": grant_date.isoformat(),
        "start_minute": start_minute,
        "end_minute": end_minute,
        "option": "whole",
        "negotiated_exception": False,
        "approval_path": task["approval_path"],
        "on_time": grant_date.isoformat() <= task["due_date"],
        "data_source": "GRANTED_HISTORY_SYNTHETIC",
    }


def demo_active_row_task_id(now: datetime, seed: int = 20260917) -> str:
    """The task_id demo_active_row would use for `now`'s calendar day --
    exposed standalone so api/app.py can check whether TODAY's specific
    device is already a live request (see emergency_resolve) without
    duplicating this same day-seeded department pick, or risking it
    drifting out of sync with demo_active_row's own."""
    rng = np.random.default_rng(seed + now.toordinal())
    department = str(rng.choice(DEPARTMENTS))
    return f"{SOURCE_SYSTEM[department]}-90001"


def demo_active_row(sections: pd.DataFrame, now: datetime, seed: int = 20260917, already_used: bool = False) -> dict | None:
    """One additional real-looking row, computed FRESH on every call
    (never baked into the static xlsx `load_granted_history()` reads --
    that's @lru_cache'd, so a row placed there would freeze its timing at
    whatever moment first loaded it in this process). It is ALWAYS
    classified "active" (start before `now`, end at least 60 minutes
    after it) no matter when the Dashboard/Weekly-Monthly Schedule/
    Corridor Map are actually loaded -- giving every one of them
    something genuinely live to show, since all three already read from
    the same shared granted-history data. Named with the same
    `{source_system}-#####` convention every other row here uses (a
    number well past this file's own real range, so it can never collide
    with one), so it's not visually distinguishable as synthetic.

    `already_used`: True once this device's task_id has been promoted
    into a real request this session (see emergency_resolve) -- it's a
    genuine, real task sitting in the Waiting List now, not something
    still "currently active" to keep showing (get_store().get_request(...)
    already returns it directly before this function is ever reached).
    Returns None rather than substituting a different device: this row
    exists to guarantee ONE real, honest "something is active right now"
    demo fact, not an inexhaustible supply -- silently swapping in a
    fresh synthetic replacement every time the last one gets used, at the
    exact same "right now" placement, reads as new fake tasks conjured
    out of thin air the moment an old one is dismissed (confirmed live:
    that's exactly the impression it gave). Once used, the corridor's own
    real data is what the Dashboard/emergency search should reflect --
    nothing manufactured to paper over it looking quieter than before.

    Deliberately skips _place_in_free_time's train-occupancy-aware
    placement, since that needs passenger/goods occupancy rebuilt fresh
    -- a cost this function accepts because it's called on every
    request, for one purely cosmetic row, not the actual scored
    dataset."""
    if already_used:
        return None
    rng = np.random.default_rng(seed + now.toordinal())  # a fresh pick each real calendar day, stable within it
    department = str(rng.choice(DEPARTMENTS))
    section_id = str(rng.choice(sections["section_id"].tolist()))
    defect_type = str(rng.choice(DEFECT_TYPES[department]))

    today = now.date()
    day_start = datetime.combine(today, datetime.min.time())
    now_minute = int((now - day_start).total_seconds() // 60)
    start_minute = max(0, now_minute - 30)
    end_minute = min(1439, now_minute + 90)
    duration_hours = round((end_minute - start_minute) / 60, 2)

    return {
        "task_id": f"{SOURCE_SYSTEM[department]}-90001",
        "department": department,
        "section_id": section_id,
        "defect_type": defect_type,
        "requester_priority": "Critical",
        "estimated_block_hours": duration_hours,
        "raised_date": (today - timedelta(days=3)).isoformat(),
        "due_date": (today + timedelta(days=10)).isoformat(),
        "days_overdue": 0,
        "splittable": True,
        "date": today.isoformat(),
        "start_minute": start_minute,
        "end_minute": end_minute,
        "option": "whole",
        "negotiated_exception": False,
        "approval_path": approval_path_for(duration_hours),
        "on_time": True,
        "data_source": "GRANTED_HISTORY_SYNTHETIC",
    }


@lru_cache(maxsize=1)
def load_granted_history() -> pd.DataFrame:
    """Reads the fixed xlsx file once per process (small, static, never
    modified) -- an empty DataFrame if it hasn't been generated yet
    (`python -m railblock.synthetic.granted_history`), never an error, so
    every consumer degrades gracefully rather than crashing an endpoint."""
    if not GRANTED_HISTORY_XLSX.exists():
        return pd.DataFrame()
    return pd.read_excel(GRANTED_HISTORY_XLSX, engine="openpyxl")


def _window(row: pd.Series) -> tuple[datetime, datetime]:
    day = date.fromisoformat(row["date"])
    day_start = datetime.combine(day, datetime.min.time())
    return day_start + timedelta(minutes=int(row["start_minute"])), day_start + timedelta(minutes=int(row["end_minute"]))


def classify_blocks_by_time(df: pd.DataFrame, now: datetime | None = None) -> dict[str, pd.DataFrame]:
    """General-purpose block-window classifier, not specific to this
    module's own dataset -- also used in api/app.py on live store.approved
    rows, since those have the exact same (date, start_minute, end_minute)
    shape. Splits `df` into {"completed", "active", "upcoming"} by
    comparing each row's real grant window against `now` (real current
    IST time by default -- see `now_ist()`, this deployment's OS clock
    is UTC but every window here is implicitly IST wall-clock time) --
    computed fresh on every call, never cached or stored, so a fixed file
    (or a live-but-unchanging approved row) always classifies correctly
    regardless of when it's read.

    A row missing start_minute/end_minute (e.g. a negotiated-exception
    row -- "not modeled", per BlockDetailsModal.jsx) has no real window to
    compare, so it's excluded from all three buckets rather than guessed
    into one -- callers that need the full unfiltered set should use the
    original `df`, not this classification, for that purpose."""
    if now is None:
        now = now_ist()
    required = {"start_minute", "end_minute", "date"}
    if df.empty or not required.issubset(df.columns):
        # Not just "no rows" -- a DataFrame built from a real but older/
        # differently-shaped set of dicts (e.g. store.approved rows from
        # before this project tracked start_minute/end_minute) may have
        # zero rows AND be missing these columns entirely, which a plain
        # column lookup would KeyError on rather than just finding nothing.
        empty = df.iloc[0:0]
        return {"completed": empty, "active": empty, "upcoming": empty}

    timeable = df[df["start_minute"].notna() & df["end_minute"].notna() & df["date"].notna()]
    if timeable.empty:
        empty = df.iloc[0:0]
        return {"completed": empty, "active": empty, "upcoming": empty}

    starts, ends = [], []
    for _, row in timeable.iterrows():
        s, e = _window(row)
        starts.append(s)
        ends.append(e)
    tagged = timeable.copy()
    tagged["_start_dt"] = starts
    tagged["_end_dt"] = ends

    completed = tagged[tagged["_end_dt"] < now].drop(columns=["_start_dt", "_end_dt"])
    active = tagged[(tagged["_start_dt"] <= now) & (now <= tagged["_end_dt"])].drop(columns=["_start_dt", "_end_dt"])
    upcoming = tagged[tagged["_start_dt"] > now].drop(columns=["_start_dt", "_end_dt"])
    return {"completed": completed, "active": active, "upcoming": upcoming}


def main() -> None:
    df = generate_granted_history()
    df.to_excel(GRANTED_HISTORY_XLSX, index=False, engine="openpyxl")
    n_future = len(df) - N_PAST
    print(f"Wrote {len(df)} granted-history rows -> {GRANTED_HISTORY_XLSX}")
    print(f"  past bucket (always completed): {N_PAST}")
    print(f"  future bucket ({TASKS_PER_DAY} real tasks/day through the evaluation window + buffer): {n_future}")


if __name__ == "__main__":
    main()
