"""Session 30, at explicit user request: Emergency Handling.

Train cancellation/rescheduling is out of scope for this project (see
orchestrator.py's own Session 30 note) -- an emergency here never touches
a real train. Instead it is an instantly-placed, UNAPPROVED maintenance
block (e.g. a rail fracture) that occupies real time on a section right
now -- "a demo emergency situation has to be done as fast as possible",
so it skips the normal request/approve pipeline entirely. What DOES need
human approval is rescheduling whatever already-scheduled maintenance
blocks it displaces -- via the exact same scheduling machinery every
other task already uses (a single CP-SAT solve, allowed to combine with
a different department already occupying a section's window), never a
special "3-day loop" or any other bespoke strategy.

Every call to `create_demo_emergency` makes a brand-new emergency -- no
dedup, no reuse of a previous one, however many times the demo button is
clicked.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date as Date, datetime, timedelta

import numpy as np
import pandas as pd

from railblock.prioritization.whittle import rank_tasks
from railblock.scheduling.capacity import compute_daily_window_capacity
from railblock.scheduling.combining import (
    approved_rows_to_window_rows,
    combine_into_occupied_windows,
    subtract_consumed_capacity,
)
from railblock.scheduling.orchestrator import solve_schedule_with_options
from railblock.synthetic.goods_forecast import generate_goods_forecast

EMERGENCY_MIN_HOURS = 1.5
EMERGENCY_MAX_HOURS = 4.0

# Defect types where, left unfixed, could pose immediate danger to a
# running train -- spans all 3 departments, reusing the exact same real
# taxonomy every other task already uses (maintenance_tasks.DEFECT_TYPES)
# rather than inventing a new one.
EMERGENCY_DEFECT_TYPES = {
    "Engineering": ["Rail fracture", "Bridge/girder deterioration"],
    "Signalling": ["Interlocking relay fault", "Track circuit failure"],
    "Traction": ["Traction substation fault", "Feeder cable fault"],
}


def build_emergency_segments(start_dt: datetime, duration_hours: float) -> list[dict]:
    """Splits [start_dt, start_dt + duration_hours) at every midnight
    boundary into per-day (date, start_minute, end_minute) segments --
    the same convention every other block in this system already follows
    (granted_history.py's placement never crosses midnight in one row
    either), rather than inventing a window that runs past minute 1439."""
    remaining_minutes = round(duration_hours * 60)
    cursor = start_dt.replace(second=0, microsecond=0)
    segments = []
    while remaining_minutes > 0:
        day_start_minute = cursor.hour * 60 + cursor.minute
        minutes_left_today = 1440 - day_start_minute
        take = min(remaining_minutes, minutes_left_today)
        segments.append({
            "date": cursor.date().isoformat(),
            "start_minute": day_start_minute,
            "end_minute": day_start_minute + take,
        })
        remaining_minutes -= take
        cursor = datetime.combine(cursor.date() + timedelta(days=1), datetime.min.time())
    return segments


def _row_start_datetime(row: dict) -> datetime:
    day = Date.fromisoformat(row["date"])
    return datetime.combine(day, datetime.min.time()) + timedelta(minutes=int(row["start_minute"]))


def create_demo_emergency(sections: pd.DataFrame, existing_rows: list[dict], now: datetime, rng: np.random.Generator, task_id: str) -> dict:
    """Builds one brand-new emergency: random department + defect type
    (from EMERGENCY_DEFECT_TYPES) and a duration drawn uniformly from
    [EMERGENCY_MIN_HOURS, EMERGENCY_MAX_HOURS].

    Session 30, at explicit user request, after a real reported gap:
    "always create an emergency situation such that one or more blocks
    are affected" -- picking a purely random window starting exactly at
    `now` left this to chance (this corridor's real schedule is sparse
    enough that "something happens to be running in this exact instant"
    often just isn't true).

    Session 39, at explicit user request ("always ... at least affects
    one of the already scheduled tasks"), after confirming live that
    Session 30's original "today's calendar date only" search left a
    real gap once EMERGENCY_MAX_HOURS was tightened way down from 10h to
    4h (a much shorter emergency duration means "today, whatever's left"
    is a much smaller net -- e.g. late in the day, once every block
    scheduled for the rest of today has already finished, this found
    nothing at all, even in a session with a real, densely-scheduled
    70-task batch, confirmed live against this corridor's own real
    schedule): the search horizon is now [now, now + EMERGENCY_MAX_HOURS)
    -- the widest an emergency's real reach could ever be, not an
    arbitrary calendar-day cutoff -- correctly spanning a midnight
    boundary when `now` is late enough that it needs to. Still anchors
    the emergency's own start to the chosen candidate's own real start
    (clamped to never be earlier than `now`, same as before) rather than
    always starting exactly at `now`: overlap is then guaranteed by
    construction regardless of which specific duration gets drawn below
    (the segment necessarily starts exactly when the candidate's own
    window does, and any two windows sharing a start point with nonzero
    length overlap) -- using `now` itself as the anchor point would only
    guarantee overlap for a duration long enough to reach that specific
    candidate, which isn't otherwise true for every draw in [1.5, 4]
    hours. Never reintroduces Session 30's original bug (anchoring to a
    task potentially DAYS away) since the search horizon is bounded by
    this emergency's own real maximum duration, not "whatever's still
    technically upcoming." Still falls back to a random section with no
    guaranteed overlap only if truly nothing on the whole corridor is
    scheduled anywhere within that reachable window -- now a much
    tighter, much rarer gap than before."""
    duration_hours = float(rng.uniform(EMERGENCY_MIN_HOURS, EMERGENCY_MAX_HOURS))

    reachable_until = now + timedelta(hours=EMERGENCY_MAX_HOURS)
    candidates = [
        r for r in existing_rows
        if r.get("start_minute") is not None
        and _row_start_datetime(r) < reachable_until
        and _row_end_datetime(r) > now
    ]
    if candidates:
        target = min(candidates, key=_row_start_datetime)
        start_dt = max(_row_start_datetime(target), now)
        section_id = target["section_id"]
    else:
        start_dt = now
        section_id = rng.choice(sections["section_id"].tolist())

    segments = build_emergency_segments(start_dt, duration_hours)

    department = rng.choice(list(EMERGENCY_DEFECT_TYPES.keys()))
    defect_type = rng.choice(EMERGENCY_DEFECT_TYPES[department])

    return {
        "task_id": task_id,
        "section_id": str(section_id),
        "department": str(department),
        "defect_type": str(defect_type),
        "estimated_block_hours": round(duration_hours, 2),
        "created_at": now.isoformat(),
        "segments": segments,
        "is_emergency": True,
        "status": "proposed",
    }


def find_affected_rows(emergency: dict, existing_rows: list[dict], now: datetime) -> list[dict]:
    """Every existing task (any section -- an incident in one place can
    cause trains to be stopped/rerouted elsewhere, so a block scheduled
    anywhere during the same real time window is a genuine candidate, not
    just ones on the emergency's own section) with at least one session
    whose (date, start_minute, end_minute) overlaps any of the
    emergency's segments. Excludes rows already completed relative to
    `now`, and any row with no modeled start/end (nothing to compare).

    Deduped by task_id -- `existing_rows` is at SESSION granularity (see
    api/app.py's _expanded_schedule_rows: a split task contributes one
    row per real session), so a split task with two or more sessions
    that both happen to overlap must still only be offered ONCE for
    reschedule. A real reported bug: without this, the SAME task_id fed
    `rank_tasks`/`expand_splittable_tasks` as two+ separate rows, which
    silently corrupted the CP-SAT part-expansion (two unrelated "part 1
    of N" rows collapsing onto the identical part-level id)."""
    affected = []
    seen_task_ids: set[str] = set()
    for row in existing_rows:
        if row.get("start_minute") is None or row.get("end_minute") is None:
            continue
        if row["task_id"] in seen_task_ids:
            continue
        for seg in emergency["segments"]:
            if row["date"] != seg["date"]:
                continue
            if row["start_minute"] < seg["end_minute"] and row["end_minute"] > seg["start_minute"]:
                if _row_end_datetime(row) > now:
                    affected.append(row)
                    seen_task_ids.add(row["task_id"])
                break
    return affected


def _row_end_datetime(row: dict) -> datetime:
    day = Date.fromisoformat(row["date"])
    return datetime.combine(day, datetime.min.time()) + timedelta(minutes=int(row["end_minute"]))


@dataclass
class RescheduleProposal:
    task_id: str
    old: dict
    proposed: dict | None  # {date, start_minute, end_minute, window_index, combined} or None if no slot found


def _clip_capacity_to_now(capacity: pd.DataFrame, now: datetime) -> pd.DataFrame:
    """Session 30, at explicit user request, after a real reported bug:
    "the affected blocks must be rescheduled afterwards only" -- without
    this, `now`'s own real calendar date still offered its FULL day of
    candidate windows, including whatever's already earlier than the
    actual current moment (a real window from, say, 09:00-11:00 is
    still a legitimate real free interval today even at 14:00 -- Layer 1
    has no reason to know a displaced task's re-placement is being
    decided AFTER that time already passed). Drops any window on
    `now`'s date that has already fully ended, and clips one straddling
    `now` to start exactly at `now` -- so a reschedule proposal can
    never land earlier today than the real moment it was computed at."""
    if capacity.empty:
        return capacity
    today_iso = now.date().isoformat()
    now_minute = now.hour * 60 + now.minute
    capacity = capacity.copy()
    is_today = capacity["date"] == today_iso
    capacity = capacity[~(is_today & (capacity["end_minute"] <= now_minute))].copy()
    is_today = capacity["date"] == today_iso
    spans_now = is_today & (capacity["start_minute"] < now_minute) & (capacity["end_minute"] > now_minute)
    if spans_now.any():
        capacity["start_minute"] = capacity["start_minute"].astype(float)
        capacity["window_minutes"] = capacity["window_minutes"].astype(float)
        capacity.loc[spans_now, "start_minute"] = float(now_minute)
        capacity.loc[spans_now, "window_minutes"] = capacity.loc[spans_now, "end_minute"] - float(now_minute)
    return capacity


def _goods_occupancy_with_emergency(
    sections: pd.DataFrame, emergency: dict, start_date: Date, n_days: int
) -> pd.DataFrame:
    """The real synthetic goods forecast for [start_date, start_date+n_days),
    plus the emergency's own segments injected as fixed occupied
    intervals -- the exact same "goods occupancy is fixed, non-
    negotiable time" mechanism every other real freight movement already
    relies on, so the solve can never re-propose a slot inside the
    emergency's own footprint, no special-casing needed in Layer 1/3."""
    goods_occupancy = generate_goods_forecast(
        sections, start_date, n_days=n_days, seed=int(start_date.strftime("%Y%m%d"))
    )
    # Every segment is included unconditionally, regardless of `n_days`
    # -- one outside the current horizon is simply never looked at by
    # compute_daily_window_capacity/rank_tasks (both only ever iterate
    # `range(n_days)` from `start_date`), so there's no need to filter
    # it out here.
    emergency_rows = pd.DataFrame([
        {
            "freight_id": emergency["task_id"], "section_id": emergency["section_id"],
            "date": seg["date"], "start_minute": seg["start_minute"], "end_minute": seg["end_minute"],
            "duration_minutes": seg["end_minute"] - seg["start_minute"], "data_source": "EMERGENCY",
        }
        for seg in emergency["segments"]
    ])
    return pd.concat([goods_occupancy, emergency_rows], ignore_index=True)


def _next_horizon(n_days: int) -> int:
    """+1 per step through the first week (matches the literal "increment
    the horizon" ask for the common case, where a slot is usually found
    within the first few days), then doubles -- reaching max_n_days=120
    in ~11 steps total instead of 120, which matters because EVERY step
    re-runs a real CP-SAT solve (see propose_reschedule's own wall-clock
    budget note)."""
    return n_days + 1 if n_days < 7 else n_days * 2


def propose_reschedule(
    affected_tasks: list[dict],
    sections: pd.DataFrame,
    passenger_occupancy: pd.DataFrame,
    emergency: dict,
    other_occupied_rows: list[dict],
    start_date: Date,
    now: datetime,
    time_limit_s: float = 5.0,
    max_n_days: int = 120,
    max_wall_clock_s: float = 120.0,
) -> list[RescheduleProposal]:
    """Reschedules every affected task via the exact same machinery a
    normal request goes through -- ranked by real urgency/impact/
    congestion (rank_tasks), a SINGLE CP-SAT solve (not the 3-option
    generate_schedule_options -- one option is enough here), allowed to
    combine with a different department already occupying a section's
    window (`other_occupied_rows` -- the live schedule MINUS the affected
    tasks themselves, so their own old slot is never treated as "already
    occupied, immovable"). `affected_tasks`: each dict needs task_id,
    department, section_id, requester_priority, estimated_block_hours,
    defect_type, raised_date, due_date, days_overdue, date, start_minute,
    end_minute (its OLD placement) -- the caller (api/app.py) is
    responsible for enriching a bare schedule row with these via the same
    lookup GET /requests/{task_id} already does. `now`: the real moment
    this reschedule is being decided at -- see _clip_capacity_to_now.

    Session 30, at explicit user request, after a real reported gap:
    "how can two tasks have no alternative slot found? there is no limit
    for the horizon, first keep horizon to 1 day, if some tasks are not
    scheduled, then increment the horizon ... repeat till all tasks are
    scheduled." A single fixed n_days=7 attempt could genuinely run out
    of real room for every affected task even though a real, valid slot
    existed further out (due dates in this corridor's real data reach
    weeks out -- see RecommendedScheduling.jsx's own Session 30 note on
    exactly this). Starts at a 1-day horizon and widens (see
    `_next_horizon`), re-solving the WHOLE affected batch together each
    time (never just the still-unplaced remainder -- a task placed at a
    smaller horizon must stay visible as a genuine combining candidate
    for one that only finds room at a larger horizon), stopping the
    moment every task has a real proposed slot.

    Two safety bounds, not the literal "no limit" asked for -- this runs
    synchronously inside one live HTTP request, so an unconditional loop
    is a real hang risk, not just a slow-but-eventually-correct one:
      - `max_n_days` (120, comfortably past this corridor's longest real
        due-date window of ~60 days): stop widening past this, however
        many tasks are still unplaced.
      - `max_wall_clock_s` (120s -- a real reported bug: an earlier
        version with time_limit_s=15 per attempt and a literal +1
        widening step took over two minutes and had to be killed; a
        FIRST fix at 25s then turned out to cut the search off before it
        genuinely found a real slot for two small, ordinary, splittable
        tasks that DID resolve given enough time -- confirmed directly by
        re-running just those two in isolation with a wider budget).
        Checked BETWEEN attempts (an individual CP-SAT solve's own
        `time_limit_s` already bounds any SINGLE attempt) -- whatever's
        still unplaced when this is hit is reported as "no alternative
        slot found" rather than making the request hang indefinitely."""
    if not affected_tasks:
        return []

    # Session 34, at explicit user request, after a real reported gap
    # ("it takes around 2 mins to load emergency situation"): confirmed
    # live -- a single real /emergency/create call against a populated
    # session took 2m25s, and the CP-SAT solve itself is already bounded
    # (time_limit_s=5.0 per attempt) -- the actual cost is compute_
    # availability(), called by both rank_tasks and compute_daily_window_
    # capacity, with NO cache at all, re-run from scratch at EVERY
    # widening step (1,2,3,4,5,6,7,14,28,56,112 -- up to 11 attempts).
    # Two separate redundant-computation costs were found, both fixed
    # the same way -- generate/compute ONCE at the largest horizon this
    # call could ever reach, then reuse across every widening attempt,
    # rather than redoing days 0..n_days-1 from scratch every time:
    #
    # 1. compute_availability's own cache is keyed purely on (section_id,
    #    date) (see corridor_availability.py's Session 25 note), and its
    #    result for a given (section, date) can never change between
    #    attempts (same passenger_occupancy, same goods_occupancy content
    #    for that date -- see point 2). `availability_cache` is created
    #    ONCE here and shared across every attempt below (not just within
    #    one, the way /schedule/options already shares one across
    #    ranking+capacity for a SINGLE solve).
    # 2. `_goods_occupancy_with_emergency` -- a pure-Python nested loop
    #    (generate_goods_forecast) over every section x day, drawing 2-7
    #    freight movements each via RNG calls, genuinely expensive at
    #    n_days=56/112 -- was being called fresh at EVERY widening step,
    #    redoing every already-generated day's work each time. Its own
    #    RNG stream is consumed strictly in day-then-section order, so
    #    day D's rows are byte-identical regardless of how many MORE days
    #    get generated after it -- generating it ONCE at `max_n_days`
    #    (comfortably covering every n_days this loop can ever reach) and
    #    reusing that same frame avoids ever redoing that slow generation.
    #
    #    Measured live, this alone was NOT enough, and at first even made
    #    an early (small-n_days) attempt slightly SLOWER: _goods_intervals/
    #    _passenger_intervals (corridor_availability.py) filter with a
    #    plain pandas boolean mask over however big a frame they're given
    #    -- O(rows in the frame), not O(rows that actually match) -- so
    #    handing every attempt the FULL max_n_days-sized frame made even
    #    a 1-day attempt's first-ever (uncached) lookups scan a ~17x
    #    bigger haystack than before. Sliced back down to exactly this
    #    attempt's own [start_date, start_date+n_days) below -- a single
    #    vectorized date comparison, not the slow nested Python loop --
    #    keeps every attempt's own filtering cost proportional to what it
    #    actually needs, while still generating the underlying data only
    #    once.
    # A THIRD, much bigger cost was found the same way, live: compute_
    # availability's own per-call work (_passenger_intervals iterates
    # every real passenger row on a section via .iterrows(), plus a real
    # delay-margin model lookup per row -- see corridor_availability.py's
    # own Session 26 note on how expensive that model call is) was being
    # paid for all 56 corridor sections at every widening step, even
    # though a displaced task can only ever be RE-placed on the exact
    # section it was already on -- model.py's own solve only ever builds
    # assign[] variables from `windows_by_section[task's own section_id]`,
    # never any other section, and combine_into_occupied_windows' own
    # candidate search requires `ranked_tasks["section_id"] ==
    # w["section_id"]` too -- so a window on a section none of the
    # affected tasks occupy can never be combined into either. Measured
    # live: at n_days=112, first-time-computing all 56 sections' worth of
    # NEWLY-added days took 48.89s by itself; restricting to only the
    # affected tasks' own (typically far fewer than 56) unique sections
    # cuts that proportionally, with no lost combining opportunity.
    relevant_sections = sections[sections["section_id"].isin({t["section_id"] for t in affected_tasks})]

    availability_cache: dict = {}
    goods_occupancy_full = _goods_occupancy_with_emergency(relevant_sections, emergency, start_date, max_n_days)

    start_time = time.monotonic()
    n_days = 1
    while True:
        horizon_end = (start_date + timedelta(days=n_days)).isoformat()
        goods_occupancy = goods_occupancy_full[goods_occupancy_full["date"] < horizon_end]
        proposals = _propose_reschedule_at_horizon(
            affected_tasks, relevant_sections, passenger_occupancy, goods_occupancy, other_occupied_rows,
            start_date, now, n_days, time_limit_s, availability_cache,
        )
        if all(p.proposed is not None for p in proposals):
            return proposals
        if n_days >= max_n_days or (time.monotonic() - start_time) >= max_wall_clock_s:
            return proposals
        n_days = _next_horizon(n_days)


def _propose_reschedule_at_horizon(
    affected_tasks: list[dict],
    sections: pd.DataFrame,
    passenger_occupancy: pd.DataFrame,
    goods_occupancy: pd.DataFrame,
    other_occupied_rows: list[dict],
    start_date: Date,
    now: datetime,
    n_days: int,
    time_limit_s: float,
    availability_cache: dict,
) -> list[RescheduleProposal]:
    """One single-horizon attempt -- see propose_reschedule for the
    widening-horizon loop around this. `availability_cache`: shared across
    every widening attempt in the SAME propose_reschedule call (and
    between ranking and capacity within this one attempt) -- see
    propose_reschedule's own Session 34 note for why that's safe."""
    old_by_id = {t["task_id"]: {"date": t["date"], "start_minute": t["start_minute"], "end_minute": t["end_minute"]} for t in affected_tasks}
    task_df = pd.DataFrame(affected_tasks)

    ranked = rank_tasks(
        task_df, sections, passenger_occupancy, goods_occupancy, start_date, n_days=n_days,
        availability_cache=availability_cache,
    )
    capacity = compute_daily_window_capacity(
        sections, passenger_occupancy, goods_occupancy, start_date, n_days,
        availability_cache=availability_cache,
    )
    capacity = _clip_capacity_to_now(capacity, now)

    occupied_windows = approved_rows_to_window_rows(other_occupied_rows, capacity)
    remaining, forced_full_rows, updated_windows = combine_into_occupied_windows(occupied_windows, ranked, capacity)
    capacity = subtract_consumed_capacity(capacity, updated_windows)

    def _shape_placement(sessions: list[tuple], section_id: str, window_lookup: dict) -> dict:
        # A task the solve had to split (rare here -- affected tasks are
        # usually a single displaced block, but splitting is always on
        # the table now that every task is splittable) gets every
        # session preserved, not just the first -- losing the rest would
        # silently under-report what actually got scheduled.
        #
        # Session 30 fix, after a real reported bug: whether a session
        # is genuinely combined is NOT the same as the schedule row's
        # own `option` field being "combined" -- that value is only ever
        # set by the pre-CP-SAT forcing passes (force_combinable_
        # placements/combine_into_occupied_windows). When the core
        # CP-SAT solve organically places two different-department tasks
        # into the SAME real window on its own (its normal, intended
        # behavior -- see model.py's COORDINATION_BONUS), the resulting
        # schedule rows still say option="whole"/"split", never
        # "combined", even though they physically share the window.
        # RecommendedScheduling.jsx's own buildBlocksBySection already
        # gets this right by checking the WINDOWS table's own `combined`
        # flag instead -- same fix, here.
        d, w, s, e = sessions[0]
        combined = bool(window_lookup.get((section_id, d, w), False))
        return {
            "date": d, "start_minute": s, "end_minute": e, "window_index": w,
            "combined": combined, "n_parts": len(sessions),
            "sessions": [{"date": ds, "window_index": wi, "start_minute": si, "end_minute": ei} for ds, wi, si, ei in sessions],
        }

    section_by_id = {t["task_id"]: t["section_id"] for t in affected_tasks}
    placed_by_id: dict[str, dict] = {}
    for row in forced_full_rows:
        sessions = row.get("sessions") or []
        if sessions:
            # A forced placement is, by construction, always genuinely
            # combined -- force_combinable_placements/combine_into_
            # occupied_windows only ever fire when a real shared window
            # with a different department was found.
            window_lookup = {(section_by_id[row["task_id"]], sessions[0][0], sessions[0][1]): True}
            placed_by_id[row["task_id"]] = _shape_placement(sessions, section_by_id[row["task_id"]], window_lookup)

    if not remaining.empty:
        result = solve_schedule_with_options(
            remaining, sections, passenger_occupancy, goods_occupancy, start_date,
            n_days=n_days, time_limit_s=time_limit_s, capacity=capacity,
        )
        window_lookup = {}
        if not result.windows.empty:
            for _, w in result.windows.iterrows():
                window_lookup[(w["section_id"], w["date"], w["window_index"])] = bool(w["combined"])
        if not result.schedule.empty:
            for _, row in result.schedule.iterrows():
                sessions = row.get("sessions") or [(row["date"], None, None, None)]
                placed_by_id[row["task_id"]] = _shape_placement(sessions, section_by_id[row["task_id"]], window_lookup)

    proposals = []
    for t in affected_tasks:
        task_id = t["task_id"]
        proposals.append(RescheduleProposal(task_id=task_id, old=old_by_id[task_id], proposed=placed_by_id.get(task_id)))
    return proposals
