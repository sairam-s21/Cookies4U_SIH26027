"""Orchestrates Options 1-2 around Layer 3's core CP-SAT solve, in the
required priority order:

  1. (Option 1, always applied upstream) `passenger_occupancy` passed in
     here already reflects realistic day-of-week running patterns -- see
     railblock.availability.service_frequency -- so the core solve below
     already benefits from Option 1 without this module doing anything
     extra for it.
  2. (Option 2, automatic) tasks are pre-expanded via
     railblock.scheduling.splitting.expand_splittable_tasks BEFORE the
     core CP-SAT solve, so splitting is tried as part of the same solve.

Train cancellation/rescheduling is out of scope for this project, so
there is no "Option 3" last-resort pass that shifts or cancels a real
train/goods movement for a Critical task still unscheduled after Options
1-2, and no critical-tasks-first two-solve architecture or
human-confirmed escalation step. Critical is just a higher whittle_index
weight (railblock.prioritization.whittle), scheduled in the exact same
single CP-SAT solve as Moderate/Routine -- a task that doesn't fit
whole/split simply stays unscheduled, same as any other priority tier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as Date

import pandas as pd

from railblock.scheduling.capacity import compute_daily_window_capacity
from railblock.scheduling.combining import (
    combined_window_rows,
    find_combinable_pairs,
    force_combinable_placements,
    partner_minutes_by_task,
)
from railblock.scheduling.model import solve_schedule
from railblock.scheduling.splitting import collapse_split_results, expand_splittable_tasks


def _collapse_window_task_ids(windows: pd.DataFrame) -> pd.DataFrame:
    """`windows` comes straight from the core CP-SAT solve over
    `expand_splittable_tasks`'s EXPANDED, part-level task set (see
    splitting.py's `_part_task_id`, "{task_id}__part{i}of{n}"), so a
    combined window's `task_ids` list can contain a split-session id like
    "TDMS-00019__part2of2" that never exists as a real request task_id --
    the frontend's combined-block details modal (GET /requests/{task_id}
    per id) would 404 on it, leaving blank details for the whole window if
    every id in it happened to be a split session. This maps every id in
    `task_ids` back to its real parent task_id (the part before "__part")
    and dedupes, since a window can combine two sessions of the SAME split
    task with a different task -- that task must only be counted/shown
    once, not twice."""
    if windows.empty or "task_ids" not in windows.columns:
        return windows
    windows = windows.copy()
    windows["task_ids"] = windows["task_ids"].apply(
        lambda ids: list(dict.fromkeys(tid.split("__part")[0] for tid in ids)) if isinstance(ids, list) else ids
    )
    return windows


@dataclass
class FullScheduleResult:
    schedule: pd.DataFrame       # one row per fully-completed task: option in {"whole","split","combined"}
    partial: pd.DataFrame        # split tasks with some but not all sessions placed
    unscheduled: pd.DataFrame    # nothing worked for this task
    windows: pd.DataFrame        # core CP-SAT solve's opened windows -- task_ids already collapsed back to real parent task_ids, see _collapse_window_task_ids
    status: str
    solve_time_s: float
    objective_value: float
    option_counts: dict = field(default_factory=dict)


def prepare_forced_and_remaining(
    ranked_tasks: pd.DataFrame,
    capacity: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The pandas-heavy pre-processing between capacity and the actual
    CP-SAT solve: identify combining candidates, expand splittable tasks
    against real window sizes, then deterministically force-place any
    combination that's genuinely feasible right now (see the inline
    comments below for the full reasoning). Depends only on `ranked_tasks`
    and `capacity` -- NOT on any strategy-specific objective weight or
    `allowed_weekdays` (those only affect the CP-SAT solve itself) -- so
    schedule_options.py's 3 strategies, which share the same
    ranked_tasks/capacity, can call this ONCE and reuse the result instead
    of each redundantly repeating this same pure-Python/pandas work. This
    redundancy is real: on a CPU-constrained host, 3x this step is a large
    share of why /schedule/options stays slow even after CP-SAT's own
    per-strategy concurrency is turned down (see config.py's
    SCHEDULE_OPTIONS_MAX_CONCURRENCY)."""
    combinable_pairs = find_combinable_pairs(ranked_tasks)
    combinable_partner_minutes = partner_minutes_by_task(ranked_tasks, combinable_pairs)
    expanded_tasks = expand_splittable_tasks(ranked_tasks, capacity, combinable_partner_minutes)
    return force_combinable_placements(expanded_tasks, capacity, combinable_pairs)


def solve_schedule_with_options(
    ranked_tasks: pd.DataFrame,
    sections: pd.DataFrame,
    passenger_occupancy: pd.DataFrame,
    goods_occupancy: pd.DataFrame,
    start_date: Date,
    n_days: int = 7,
    time_limit_s: float = 30.0,
    weekly_targets: dict[str, float] | None = None,
    coordination_bonus: float | None = None,
    window_open_cost: float | None = None,
    on_time_bonus: float | None = None,
    allowed_weekdays: set[int] | None = None,
    capacity: pd.DataFrame | None = None,
    forced_and_remaining: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None,
) -> FullScheduleResult:
    """`capacity`: pass a precomputed compute_daily_window_capacity() result
    to skip recomputing it -- e.g. schedule_options.py's multiple
    strategies all share the SAME real per-section free-time data (only
    the objective weights/allowed_weekdays differ), so computing it once
    and reusing it across strategies avoids redundant, pure-Python/pandas
    work that would otherwise dominate wall-clock time when the strategies
    are run concurrently (CP-SAT's own C++ solve releases the GIL; this
    pandas-heavy step does not, so it's the part that must be shared
    rather than parallelized).

    `forced_and_remaining`: same idea, one step further down the pipeline
    -- pass a precomputed prepare_forced_and_remaining(ranked_tasks,
    capacity) result to skip recomputing splitting/combining too."""
    if capacity is None:
        capacity = compute_daily_window_capacity(sections, passenger_occupancy, goods_occupancy, start_date, n_days)
    # Before splitting decides part sizes, identify genuine
    # department-combining candidates -- same section, overlapping
    # [raised_date, due_date] windows, different departments (see
    # combining.py) -- so a split part can be aimed at a SPECIFIC real
    # partner's duration, not just any real window.
    #
    # CP-SAT's own COORDINATION_BONUS incentive alone doesn't reliably
    # realize an obvious combining opportunity: a time-limited, 8-worker
    # parallel search doesn't guarantee finding every locally obvious win
    # across a whole horizon. So this deterministically pre-assigns any
    # combination that's genuinely feasible right now -- same section,
    # real shared window, each task's own hours fitting it, both within
    # their own due dates -- before CP-SAT ever runs. What's placed here
    # is removed from CP-SAT's pool entirely (capacity subtracted too), so
    # it can never be silently missed OR undone.
    if forced_and_remaining is None:
        forced_and_remaining = prepare_forced_and_remaining(ranked_tasks, capacity)
    forced_schedule, remaining_expanded, capacity_after_force = forced_and_remaining

    weight_kwargs = {}
    if coordination_bonus is not None:
        weight_kwargs["coordination_bonus"] = coordination_bonus
    if window_open_cost is not None:
        weight_kwargs["window_open_cost"] = window_open_cost
    if on_time_bonus is not None:
        weight_kwargs["on_time_bonus"] = on_time_bonus

    core_result = solve_schedule(
        remaining_expanded, sections, passenger_occupancy, goods_occupancy, start_date, n_days, time_limit_s,
        capacity=capacity_after_force, weekly_targets=weekly_targets, allowed_weekdays=allowed_weekdays, **weight_kwargs,
    )

    merged_part_schedule = (
        pd.concat([core_result.schedule, forced_schedule], ignore_index=True)
        if not forced_schedule.empty else core_result.schedule
    )
    full_df, partial_df, fully_unscheduled_df = collapse_split_results(
        merged_part_schedule, core_result.unscheduled, ranked_tasks
    )
    windows_df = _collapse_window_task_ids(core_result.windows)
    if not forced_schedule.empty:
        # forced_schedule's task_id is still the PART-level id (e.g.
        # "TDMS-REQ-00202__part1of6"), which would 404 against GET
        # /requests and show "No details found" in the frontend's
        # combined-block modal -- same collapsing fix as above.
        windows_df = pd.concat(
            [windows_df, _collapse_window_task_ids(combined_window_rows(forced_schedule, capacity))],
            ignore_index=True,
        )

    option_counts = {
        "whole": int((full_df["option"] == "whole").sum()) if not full_df.empty else 0,
        "split": int((full_df["option"] == "split").sum()) if not full_df.empty else 0,
        "partial": int(len(partial_df)),
        "unscheduled": int(len(fully_unscheduled_df)),
    }

    return FullScheduleResult(
        schedule=full_df,
        partial=partial_df,
        unscheduled=fully_unscheduled_df,
        windows=windows_df,
        status=core_result.status,
        solve_time_s=core_result.solve_time_s,
        objective_value=core_result.objective_value,
        option_counts=option_counts,
    )
