"""Layer 4 -- Weekly & Monthly Plan Generator.

Two cadences over the same underlying data (docs/MASTER_PROMPT_SIH_26027.md
Section 4):
  - generate_weekly_plan: full-detail, day/window-level, operational,
    directly executable -- Options 1-3 (day-of-week occupancy, automatic
    splitting, bounded negotiated exceptions) are all active, since this
    is what actually gets handed to the departments to execute.
  - generate_monthly_plan: coarser, per-section per-week hour allocation
    TARGETS over a 4-week horizon (railblock.scheduling.monthly) --
    tactical, not meant to be executed directly.

Handoff: generate_weekly_plan_with_monthly_handoff runs the monthly plan
first, reads off `week_number`'s per-section targets, and passes them into
the weekly plan's underlying CP-SAT solve as a SOFT cap (model.py's
weekly_targets/MEETS_TARGET_BONUS) -- rewarded for hitting a section's
target, never hard-blocked from doing more or less if the operational,
window-level picture calls for it.
"""

from __future__ import annotations

from datetime import date as Date, timedelta

import pandas as pd

from railblock.scheduling.monthly import (
    DAYS_PER_WEEK,
    WEEKS_PER_MONTH,
    MonthlyPlanResult,
    solve_monthly_plan,
)
from railblock.scheduling.orchestrator import FullScheduleResult, solve_schedule_with_options


def generate_weekly_plan(
    ranked_tasks: pd.DataFrame,
    sections: pd.DataFrame,
    passenger_occupancy: pd.DataFrame,
    goods_occupancy: pd.DataFrame,
    start_date: Date,
    time_limit_s: float = 30.0,
    weekly_targets: dict[str, float] | None = None,
) -> FullScheduleResult:
    """The operational, directly-executable plan for a single 7-day week."""
    return solve_schedule_with_options(
        ranked_tasks, sections, passenger_occupancy, goods_occupancy, start_date,
        n_days=DAYS_PER_WEEK, time_limit_s=time_limit_s, weekly_targets=weekly_targets,
    )


def generate_monthly_plan(
    ranked_tasks: pd.DataFrame,
    sections: pd.DataFrame,
    passenger_occupancy: pd.DataFrame,
    goods_occupancy: pd.DataFrame,
    start_date: Date,
    n_weeks: int = WEEKS_PER_MONTH,
    time_limit_s: float = 30.0,
) -> MonthlyPlanResult:
    """The tactical plan: per-section weekly-hour allocation targets over
    a 4-week horizon starting `start_date`."""
    return solve_monthly_plan(
        ranked_tasks, sections, passenger_occupancy, goods_occupancy, start_date, n_weeks, time_limit_s
    )


def generate_weekly_plan_with_monthly_handoff(
    ranked_tasks: pd.DataFrame,
    sections: pd.DataFrame,
    passenger_occupancy: pd.DataFrame,
    goods_occupancy: pd.DataFrame,
    monthly_start_date: Date,
    week_number: int = 1,
    n_weeks: int = WEEKS_PER_MONTH,
    time_limit_s: float = 30.0,
) -> tuple[MonthlyPlanResult, FullScheduleResult]:
    """Run the monthly plan over [monthly_start_date, +n_weeks), then
    generate `week_number`'s (1-indexed) detailed weekly plan with that
    week's monthly-derived per-section targets as a soft cap. Returns
    (monthly_result, weekly_result) so both the tactical targets and the
    operational outcome are visible together.
    """
    monthly_result = generate_monthly_plan(
        ranked_tasks, sections, passenger_occupancy, goods_occupancy, monthly_start_date, n_weeks, time_limit_s
    )

    targets = monthly_result.weekly_targets
    week_targets = targets[targets["week_number"] == week_number] if not targets.empty else targets
    weekly_targets_dict = dict(zip(week_targets["section_id"], week_targets["target_minutes"]))

    week_start_date = monthly_start_date + timedelta(days=(week_number - 1) * DAYS_PER_WEEK)
    weekly_result = generate_weekly_plan(
        ranked_tasks, sections, passenger_occupancy, goods_occupancy, week_start_date, time_limit_s,
        weekly_targets=weekly_targets_dict,
    )
    return monthly_result, weekly_result
