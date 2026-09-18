"""Layer 4b -- Monthly plan.

Runs the same underlying task-to-resource assignment idea as Layer 3, but
at coarser (per-section, per-ISO-week) granularity over a 4-week horizon,
producing per-section weekly-hour allocation TARGETS rather than a
day/window-level schedule -- the "tactical" counterpart to the weekly
plan's "operational, directly executable" detail (see
docs/MASTER_PROMPT_SIH_26027.md Section 4).

Capacity per (section, week) here is that week's TOTAL free minutes
(summed across its 7 days via Layer 1's compute_availability directly) --
deliberately coarser than Layer 3's day/window-fragmentation-aware
capacity: at monthly-tactical granularity we're allocating a rough weekly
hour budget per section, not committing to exact windows. Department
combination and on-time-by-window detail are intentionally NOT modelled
here -- that level of detail belongs to the weekly run.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date, timedelta

import pandas as pd
from ortools.sat.python import cp_model

from railblock.availability.corridor_availability import compute_availability

WEEKS_PER_MONTH = 4
DAYS_PER_WEEK = 7
PRIORITY_VALUE_SCALE = 10


@dataclass
class MonthlyPlanResult:
    weekly_targets: pd.DataFrame   # section_id, week_number, target_hours, target_minutes
    assigned_tasks: pd.DataFrame   # task_id, section_id, week_number, estimated_block_hours
    unscheduled: pd.DataFrame
    status: str
    objective_value: float


def compute_weekly_capacity_budget(
    sections: pd.DataFrame,
    passenger_occupancy: pd.DataFrame,
    goods_occupancy: pd.DataFrame,
    start_date: Date,
    n_weeks: int = WEEKS_PER_MONTH,
) -> pd.DataFrame:
    """One row per (section_id, week_number): free_minutes = total free
    time summed across that week's 7 days (coarse budget, not window-level)."""
    rows = []
    for section_id in sections["section_id"]:
        for week in range(n_weeks):
            total_free = 0.0
            for day in range(DAYS_PER_WEEK):
                the_date = start_date + timedelta(days=week * DAYS_PER_WEEK + day)
                avail = compute_availability(section_id, the_date, passenger_occupancy, goods_occupancy)
                total_free += avail["free_minutes"]
            rows.append({"section_id": section_id, "week_number": week + 1, "free_minutes": total_free})
    return pd.DataFrame(rows)


def solve_monthly_plan(
    ranked_tasks: pd.DataFrame,
    sections: pd.DataFrame,
    passenger_occupancy: pd.DataFrame,
    goods_occupancy: pd.DataFrame,
    start_date: Date,
    n_weeks: int = WEEKS_PER_MONTH,
    time_limit_s: float = 30.0,
) -> MonthlyPlanResult:
    """Assign each task to at most one ISO week (not a specific day/window)
    subject to that week's coarse per-section capacity budget. Returns
    per-(section, week) target hours -- the sum of hours assigned there --
    for handoff to the weekly plan as a soft capacity target.
    """
    budget = compute_weekly_capacity_budget(sections, passenger_occupancy, goods_occupancy, start_date, n_weeks)
    budget_lookup = {(r["section_id"], r["week_number"]): r["free_minutes"] for _, r in budget.iterrows()}

    tasks = ranked_tasks.reset_index(drop=True)
    hours_min = (tasks["estimated_block_hours"] * 60).round().astype(int)
    priority_value = (tasks["whittle_index"] * PRIORITY_VALUE_SCALE).round().astype(int).clip(lower=1)
    weeks = list(range(1, n_weeks + 1))

    model = cp_model.CpModel()
    assign = {(t, w): model.NewBoolVar(f"assign[{t},{w}]") for t in tasks.index for w in weeks}
    scheduled = {t: model.NewBoolVar(f"scheduled[{t}]") for t in tasks.index}

    for t in tasks.index:
        vars_t = [assign[(t, w)] for w in weeks]
        model.Add(sum(vars_t) <= 1)
        model.Add(sum(vars_t) == scheduled[t])

    for s in sections["section_id"]:
        idxs = tasks.index[tasks["section_id"] == s].tolist()
        for w in weeks:
            cap_min = int(round(budget_lookup.get((s, w), 0.0)))
            usage = sum(int(hours_min[t]) * assign[(t, w)] for t in idxs)
            model.Add(usage <= cap_min)

    model.Maximize(sum(int(priority_value[t]) * scheduled[t] for t in tasks.index))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)
    status_name = solver.StatusName(status)
    solved_ok = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    assigned_rows = []
    if solved_ok:
        for t in tasks.index:
            for w in weeks:
                if solver.Value(assign[(t, w)]):
                    assigned_rows.append(
                        {
                            "task_id": tasks.at[t, "task_id"],
                            "section_id": tasks.at[t, "section_id"],
                            "week_number": w,
                            "estimated_block_hours": tasks.at[t, "estimated_block_hours"],
                        }
                    )
    assigned_df = pd.DataFrame(assigned_rows)
    scheduled_ids = set(assigned_df["task_id"]) if not assigned_df.empty else set()
    unscheduled_df = tasks[~tasks["task_id"].isin(scheduled_ids)].copy()

    if not assigned_df.empty:
        target = assigned_df.groupby(["section_id", "week_number"])["estimated_block_hours"].sum().reset_index()
        target = target.rename(columns={"estimated_block_hours": "target_hours"})
        target["target_minutes"] = target["target_hours"] * 60
    else:
        target = pd.DataFrame(columns=["section_id", "week_number", "target_hours", "target_minutes"])

    objective_value = solver.ObjectiveValue() if solved_ok else float("nan")
    return MonthlyPlanResult(
        weekly_targets=target,
        assigned_tasks=assigned_df,
        unscheduled=unscheduled_df,
        status=status_name,
        objective_value=objective_value,
    )
