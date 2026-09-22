"""Due-date-aware rolling horizon + backlog carryover.

Completion must be measured against each task's own `due_date` (Critical
3-7 days, Moderate 15-30, Routine 45-90), not a FIXED test-window
boundary (e.g. 7 or 28 days). A Routine task not yet scheduled by day 7
has not failed anything -- it may have 80+ days left. This module runs
Layer 4's existing weekly solve repeatedly across a horizon long enough to
cover the longest real SLA (90 days), and classifies each task's TRUE
outcome against its own due_date:
  - "on_time": fully scheduled (all parts, for split tasks) on or before
    its own due_date.
  - "late": fully scheduled, but after its own due_date.
  - "unscheduled": never fully scheduled within the whole run.

No carryover/rolling/leftover-pool logic exists elsewhere in
scheduling/* -- every other weekly/monthly call operates on a single,
self-contained task batch with no memory of previous weeks. Implemented
here: a task not fully scheduled in week N remains a candidate in week
N+1's pool, re-ranked with its `days_overdue` recomputed against week
N+1's own date (not frozen at the horizon's start) so an aging task's
urgency score correctly grows as its due date approaches/passes, per
Layer 2's existing formula -- this achieves the desired "recalculated as
appropriate" urgency behaviour by feeding Layer 2's unmodified formula a
fresher `days_overdue` each week, not by changing the formula itself.

Deliberately does NOT modify Layer 3's Option 1-3 mechanics, Layer 4's
per-week solve, or the corridor model -- each week's solve is an
unmodified call to solve_schedule_with_options; this module only adds the
week-to-week bookkeeping loop and due-date classification around it.

Known simplification: a task "partial"ly completed in one week (some but
not all of its split sessions placed) is re-queued as a FRESH whole task
for the following week rather than carrying forward only its remaining
hours -- preserving partial progress across week boundaries would mean
extending Option 2's splitting/tracking mechanics to be horizon-aware,
which this module does not attempt. Its `first_partial_week` is recorded
for visibility; its actual outcome is whichever later week it appears in
that week's `result.schedule` (fully done), or "unscheduled" if that
never happens within the horizon.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date, timedelta

import pandas as pd

from railblock.prioritization.whittle import rank_tasks
from railblock.scheduling.orchestrator import solve_schedule_with_options
from railblock.synthetic.goods_forecast import generate_goods_forecast

DAYS_PER_WEEK = 7


@dataclass
class RollingHorizonResult:
    outcomes: pd.DataFrame            # one row per original task: status, completed_date, first_partial_week, weeks_pending
    weekly_option_counts: list[dict]  # per-week solver option_counts, for visibility/debugging
    n_weeks: int
    horizon_days: int


def _recompute_days_overdue(tasks: pd.DataFrame, as_of: Date) -> pd.Series:
    due = pd.to_datetime(tasks["due_date"])
    return (pd.Timestamp(as_of) - due).dt.days.clip(lower=0)


def run_rolling_horizon(
    tasks: pd.DataFrame,
    sections: pd.DataFrame,
    passenger_occupancy: pd.DataFrame,
    horizon_days: int,
    start_date: Date,
    time_limit_s: float = 120.0,  # see schemas.RecommendRequest for the reasoning behind this default. Only affects the offline/precompute path, not a live-clicked request.
    goods_seed: int | None = None,
    goods_occupancy_override: pd.DataFrame | None = None,
) -> RollingHorizonResult:
    """Run `tasks` (generated ONCE, at start_date -- not regenerated per
    week, per the brief: this models one batch's full realistic lifetime)
    week-by-week across [start_date, start_date + horizon_days), carrying
    every not-yet-fully-scheduled task forward, and classify each task's
    true on_time/late/unscheduled outcome against its own due_date.

    `goods_occupancy_override`: if given, used as-is for every week instead
    of generating a fresh synthetic forecast per week -- a testability
    hook so a scenario's capacity can be fully deterministic (e.g. an
    empty DataFrame removes goods-side randomness entirely, isolating
    carryover behaviour from goods-forecast noise). Production callers
    should leave this None.
    """
    n_weeks = -(-horizon_days // DAYS_PER_WEEK)  # ceil division

    outcomes: dict[str, dict] = {
        row["task_id"]: {
            "task_id": row["task_id"],
            "department": row["department"],
            "requester_priority": row["requester_priority"],
            "section_id": row["section_id"],
            "due_date": row["due_date"],
            # tasks[...]'s own days_overdue is computed by generate_maintenance_tasks
            # at as_of_date=start_date -- i.e. exactly "at generation", before this
            # loop ever recomputes it per week. A large fraction of the generator's
            # output can be already-overdue the moment it's created (raise_offset up
            # to 120 days back vs a due window as short as 3-90 days) -- this flag lets
            # "genuinely late due to scheduling" be told apart from "was pre-existing
            # backlog no scheduling speed could have made on-time", a distinction that
            # matters a lot for correctly interpreting the outcome breakdown.
            "was_already_overdue_at_generation": bool(row["days_overdue"] > 0),
            "status": "unscheduled",
            "completed_date": None,
            "first_partial_week": None,
            "weeks_pending": 0,
        }
        for _, row in tasks.iterrows()
    }

    weekly_option_counts = []

    for week_idx in range(n_weeks):
        week_start = start_date + timedelta(days=week_idx * DAYS_PER_WEEK)
        pending_ids = [tid for tid, o in outcomes.items() if o["status"] == "unscheduled"]
        if not pending_ids:
            break

        week_tasks = tasks[tasks["task_id"].isin(pending_ids)].copy()
        # re-recompute urgency input for THIS week's date, not the horizon's
        # start -- an aging task must not stay frozen at its original
        # (possibly stale, possibly zero) days_overdue as weeks pass.
        week_tasks["days_overdue"] = _recompute_days_overdue(week_tasks, week_start)

        if goods_occupancy_override is not None:
            goods_occupancy = goods_occupancy_override
        else:
            goods_occupancy = generate_goods_forecast(sections, week_start, n_days=DAYS_PER_WEEK, seed=goods_seed)
        ranked = rank_tasks(
            week_tasks, sections, passenger_occupancy, goods_occupancy, week_start, n_days=DAYS_PER_WEEK
        )
        result = solve_schedule_with_options(
            ranked, sections, passenger_occupancy, goods_occupancy, week_start,
            n_days=DAYS_PER_WEEK, time_limit_s=time_limit_s,
        )
        weekly_option_counts.append(
            {"week": week_idx + 1, "week_start": week_start.isoformat(), **result.option_counts}
        )

        if not result.schedule.empty:
            for _, row in result.schedule.iterrows():
                tid = row["task_id"]
                due = pd.Timestamp(outcomes[tid]["due_date"])
                completed = pd.Timestamp(row["date"])
                outcomes[tid]["status"] = "on_time" if completed <= due else "late"
                outcomes[tid]["completed_date"] = row["date"]

        if not result.partial.empty:
            for _, row in result.partial.iterrows():
                tid = row["task_id"]
                if outcomes[tid]["first_partial_week"] is None:
                    outcomes[tid]["first_partial_week"] = week_idx + 1
                # remains "unscheduled" for carryover -- see module docstring

        for tid in pending_ids:
            if outcomes[tid]["status"] == "unscheduled":
                outcomes[tid]["weeks_pending"] += 1

    outcomes_df = pd.DataFrame(outcomes.values())
    return RollingHorizonResult(
        outcomes=outcomes_df, weekly_option_counts=weekly_option_counts, n_weeks=n_weeks, horizon_days=horizon_days
    )
