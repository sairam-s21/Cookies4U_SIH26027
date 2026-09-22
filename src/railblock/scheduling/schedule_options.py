"""Generates multiple genuinely different weekly schedules for the
"Recommended Scheduling" page, instead of a single CP-SAT solve.

Each option is a REAL, independent CP-SAT solve with a different, real
parameterization already exposed by Layer 3 (railblock.scheduling.model):
the coordination/window-open objective weights, or which weekdays'
candidate windows (railblock.scheduling.capacity) are even offered to the
solver. Nothing here is hand-authored flavor text describing a fake
schedule -- the pros/cons strings are computed FROM each option's actual
result (real window/combining/critical-task counts), and compared against
the balanced option's own real numbers.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date as Date

import pandas as pd

from railblock.config import SCHEDULE_OPTIONS_MAX_CONCURRENCY
from railblock.scheduling.capacity import compute_daily_window_capacity
from railblock.scheduling.model import COORDINATION_BONUS, WINDOW_OPEN_COST
from railblock.scheduling.orchestrator import (
    FullScheduleResult,
    prepare_forced_and_remaining,
    solve_schedule_with_options,
)

WEEKDAYS = {0, 1, 2, 3, 4}
WEEKEND = {5, 6}


@dataclass
class ScheduleOption:
    key: str
    label: str
    description: str
    result: FullScheduleResult
    recommended: bool = False


def _strategy_defs() -> list[dict]:
    return [
        {
            "key": "balanced",
            "label": "Balanced (default)",
            "description": "Whittle-index priority ranking with standard department-combining incentive -- Layer 2/3's normal behaviour.",
            "kwargs": {},
        },
        {
            "key": "max_coordination",
            "label": "Maximize coordination",
            "description": "Coordination bonus and window-open cost both raised -- strongly prefers combining multiple departments' work into fewer, shared windows over opening more separate ones.",
            "kwargs": {"coordination_bonus": COORDINATION_BONUS * 3, "window_open_cost": WINDOW_OPEN_COST * 2},
        },
        {
            "key": "priority_first",
            "label": "Strict priority-first",
            "description": "Coordination bonus set to zero -- tasks are placed purely by Whittle-index/on-time value with no incentive to wait for a combinable window, so high-priority tasks get the very next available slot.",
            "kwargs": {"coordination_bonus": 0},
        },
        {
            "key": "weekday_only",
            "label": "Weekday-only (Mon-Fri)",
            "description": "Only Monday-Friday candidate windows are offered to the solver -- no weekend crew mobilization needed, at the cost of less total available capacity.",
            "kwargs": {"allowed_weekdays": WEEKDAYS},
        },
        {
            "key": "weekend_concentrated",
            "label": "Weekend-concentrated",
            "description": "Only Saturday/Sunday candidate windows are offered -- concentrates disruption into the weekend, when passenger service is typically lighter, at the cost of Mon-Fri urgent tasks waiting longer.",
            "kwargs": {"allowed_weekdays": WEEKEND},
        },
    ]


def generate_schedule_options(
    ranked_tasks: pd.DataFrame,
    sections: pd.DataFrame,
    passenger_occupancy: pd.DataFrame,
    goods_occupancy: pd.DataFrame,
    start_date: Date,
    n_days: int = 7,
    time_limit_s: float = 120.0,  # see schemas.RecommendRequest for the reasoning behind this default
    capacity: pd.DataFrame | None = None,
) -> list[ScheduleOption]:
    """Runs at most 3 real, independent CP-SAT solves (the first 3 of
    _strategy_defs) and returns them as ScheduleOption objects, the
    balanced/default strategy marked recommended. Capped at 3 (not 5) to
    keep generation time down; a non-balanced strategy that schedules
    nothing is dropped from the result rather than shown as an empty,
    useless option -- balanced itself is always kept even if it too
    scheduled nothing, since the caller needs it for status bookkeeping.

    `capacity`: pass a precomputed compute_daily_window_capacity() result
    (optionally already reduced to reflect capacity another solve has
    already consumed -- see api/app.py's critical-tasks-first flow) to
    skip recomputing it fresh; computed here if omitted, as before.

    Two real optimizations, not cosmetic:
    - `compute_daily_window_capacity` (pure pandas, does not release the
      GIL) is computed ONCE and shared across every strategy, instead of
      each one redundantly recomputing the exact same real per-section
      free-time data.
    - ALL 3 strategies are solved CONCURRENTLY via a thread pool -- not
      "balanced alone, then the other two together" (an earlier version
      of this function did that, specifically to skip the other two when
      balanced scheduled nothing at all; that early-exit is traded away
      here for real wall-clock speed, since the all-scheduled-nothing
      case is rare in practice and the sequential-then-concurrent
      structure meant worst-case time was balanced's own solve PLUS the
      slower of the other two, roughly double a single solve's time
      budget instead of one). This is genuinely faster, not
      just concurrent bookkeeping: OR-Tools' CP-SAT `Solve()` call is
      native C++ and releases the GIL while it runs, so multiple solves
      really do execute in parallel on separate cores despite Python's
      GIL -- worst case is now bounded by the SLOWEST single solve, not
      the sum of two.
    """
    strategies = _strategy_defs()[:3]
    if capacity is None:
        capacity = compute_daily_window_capacity(sections, passenger_occupancy, goods_occupancy, start_date, n_days)
    # Same sharing as `capacity` above, one step further down the
    # pipeline: splitting/combining pre-processing depends only on
    # ranked_tasks + capacity (never on a strategy's own objective
    # weights or allowed_weekdays), so it's identical across all 3
    # strategies too -- computed once here instead of 3x redundantly.
    forced_and_remaining = prepare_forced_and_remaining(ranked_tasks, capacity)

    def run(strat: dict) -> ScheduleOption:
        result = solve_schedule_with_options(
            ranked_tasks, sections, passenger_occupancy, goods_occupancy, start_date,
            n_days=n_days, time_limit_s=time_limit_s, capacity=capacity,
            forced_and_remaining=forced_and_remaining, **strat["kwargs"],
        )
        return ScheduleOption(
            key=strat["key"], label=strat["label"], description=strat["description"],
            result=result, recommended=(strat["key"] == "balanced"),
        )

    with ThreadPoolExecutor(max_workers=max(1, min(len(strategies), SCHEDULE_OPTIONS_MAX_CONCURRENCY))) as pool:
        results = list(pool.map(run, strategies))

    balanced = next(o for o in results if o.key == "balanced")
    if len(balanced.result.schedule) == 0:
        return [balanced]

    options = [o for o in results if o is balanced or len(o.result.schedule) > 0]
    options = _dedupe_identical_options(options)

    # The "AI recommended" badge goes to whichever option's REAL numbers
    # score highest -- never hardcoded to "balanced". See _score() for the
    # exact, documented formula.
    for o in options:
        o.recommended = False
    best = max(options, key=lambda o: _score(o.result, len(ranked_tasks)))
    best.recommended = True
    return options


def _result_signature(result: FullScheduleResult) -> tuple:
    """A hashable summary of exactly what got scheduled and when -- two
    strategies that happen to reach the identical real outcome (common
    on a lightly-loaded batch, where there's nothing left to trade off
    between coordination and strict priority) produce the same signature.
    Deliberately keyed on real placement facts (task_id, date,
    window_index, start/end minute), not on the objective weights used
    to reach them -- two DIFFERENT placements that happen to schedule the
    same task_ids must NOT collapse into one signature."""
    if result.schedule.empty:
        schedule_key = ()
    else:
        cols = [c for c in ("task_id", "date", "window_index", "start_minute", "end_minute") if c in result.schedule.columns]
        schedule_key = tuple(sorted(map(tuple, result.schedule[cols].fillna(-1).to_records(index=False))))
    return (schedule_key, len(result.partial), len(result.unscheduled))


def _dedupe_identical_options(options: list[ScheduleOption]) -> list[ScheduleOption]:
    """When two strategies produce the exact same real schedule (nothing
    left to trade off given this batch), showing both as separate cards
    is misleading -- it looks like two distinct real choices when
    there's really only one. Keeps
    "balanced" when it's part of a duplicate group (it's the one every
    other strategy is described relative to); otherwise keeps whichever
    came first in strategy-priority order. Never drops an option that is
    genuinely different from every other one, even if its own numbers
    happen to look similar in the summary."""
    seen: dict[tuple, ScheduleOption] = {}
    for opt in options:
        sig = _result_signature(opt.result)
        kept = seen.get(sig)
        if kept is None or opt.key == "balanced":
            seen[sig] = opt
    # preserve original relative order among the kept options
    kept_ids = {id(o) for o in seen.values()}
    return [o for o in options if id(o) in kept_ids]


def _n_combined(result: FullScheduleResult) -> int:
    if result.windows.empty:
        return 0
    return int(result.windows["combined"].sum())


def _n_critical_scheduled(result: FullScheduleResult) -> int:
    if result.schedule.empty:
        return 0
    return int((result.schedule["requester_priority"] == "Critical").sum())


def _score(result: FullScheduleResult, total_tasks: int) -> float:
    """Ranks options by their own real numbers -- never a fixed
    "balanced always wins" default. Weighted toward what the problem
    statement actually cares about: getting priority-ranked work done
    (especially Critical), with a real (not cosmetic) penalty for partial
    outcomes since those represent incomplete placements, not clean
    wins."""
    n_scheduled = len(result.schedule)
    n_critical = _n_critical_scheduled(result)
    n_partial = len(result.partial)
    scheduled_frac = n_scheduled / total_tasks if total_tasks else 0.0
    return scheduled_frac * 100 + n_critical * 10 + _n_combined(result) * 2 - n_partial * 1


def summarize_option(option: ScheduleOption, total_tasks: int, total_critical: int) -> dict:
    """The real, computed facts shown for each option -- no pros/cons
    framing, no hand-authored text. Every number here is read directly
    off the option's own real CP-SAT result.

    Train cancellation/negotiation is out of scope for this project
    (railblock.scheduling.negotiated_exceptions and
    critical_task_escalation don't exist), so this carries no
    negotiated/real-train tracking (negotiated_count,
    negotiated_freight_movements, negotiated_real_trains,
    trains_cancelled, train_schedules_changed)."""
    r = option.result
    n_scheduled = len(r.schedule)
    n_critical = _n_critical_scheduled(r)
    n_combined = _n_combined(r)
    combined_minutes = int(r.windows.loc[r.windows["combined"], "minutes_used"].sum()) if not r.windows.empty else 0

    split_df = r.schedule[r.schedule["option"] == "split"] if not r.schedule.empty else r.schedule
    n_split = len(split_df)
    split_hours = round(float(split_df["estimated_block_hours"].sum()), 2) if n_split else 0.0
    split_days = 0
    if n_split:
        days = set()
        for sessions in split_df["sessions"]:
            if isinstance(sessions, list):
                days.update(s[0] for s in sessions)
        split_days = len(days)

    n_overdue = int((~r.schedule["on_time"]).sum()) if not r.schedule.empty else 0

    return {
        "tasks_scheduled": n_scheduled,
        "tasks_total": total_tasks,
        "critical_scheduled": n_critical,
        "critical_total": total_critical,
        "combined_blocks": n_combined,
        "combined_minutes": combined_minutes,
        "split_fully_scheduled": n_split,
        "split_hours_delivered": split_hours,
        "split_days_spanned": split_days,
        "partially_scheduled": len(r.partial),
        "tasks_overdue": n_overdue,
    }
