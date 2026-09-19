"""Layer 3 -- Scheduling Optimizer (Google OR-Tools CP-SAT).

Adapted from:
  - Tomas Lidén's CISMR model: only the maintenance-WINDOW sub-problem's
    variable structure is reused (window choice, work-start indicator).
    His train-scheduling and crew-scheduling variables are NOT
    implemented -- explicitly out of scope per
    docs/MASTER_PROMPT_SIH_26027.md Section 2 ("trains are a fixed input,
    not a decision variable") and Section 3 ("maintenance-window
    sub-problem only").
  - Pour et al. (2019): their crew-synchronisation mechanism is
    reinterpreted here as a department block-COMBINATION incentive --
    when multiple departments compete for the same section/window, the
    objective rewards granting them a single shared window rather than
    separate ones. This directly targets the problem statement's named
    pain point, "poor coordination[reducing] asset availability".

A section/day can offer several disjoint free intervals (see
railblock.scheduling.capacity's "window fragmentation" note), so windows
are indexed by (section, date, window_index) -- window_index 0 is that
day's largest gap, 1 the next largest, etc, capped at
capacity.MAX_WINDOWS_PER_DAY. This is Lidén's window-CHOICE concept taken
literally: the model actually chooses among several available windows, not
just whether to use "the" one window a day offers.

Decision variables (w = (section, date, window_index)):
  window_open[w]            bool  Lidén-style window CHOICE: is this
                                  specific candidate window actually
                                  granted as a maintenance block.
  assign[t, w]               bool  Lidén-style work-START indicator: does
                                  task t's work start in window w (only
                                  defined for w's whose section matches
                                  t's section). Each task is assigned to at
                                  most one window across the whole horizon.
  dept_present[w, dept]      bool  reified OR: does department `dept` have
                                  >=1 assigned task in window w.
  combined_window[w]         bool  reified: do >=2 distinct departments
                                  share window w (the Pour et al.-style
                                  synchronisation signal).

Hard constraints:
  - assign[t, w] <= window_open[w]                    -- no work without an open window
  - PER DEPARTMENT dep: sum(hours[t]*assign[t,w] for t in dep assignable to w)
        <= window_minutes[w] * window_open[w]           -- never exceeds Layer 1's real capacity.
        Applied per department rather than pooled across every task in the
        window: a real joint disconnection (SR 3.51.6) lets different
        departments' crews work concurrently within the same granted block,
        so two departments sharing a window only need it as long as the
        busier of the two, not the sum of both -- one department's OWN
        tasks in the same window still stack sequentially (one crew can't
        be in two places).
  - sum_w assign[t, w] <= 1                             -- a task is scheduled at most once

Objective (maximize):
    sum_t  priority_value[t] * scheduled[t]          -- Layer 2's ranked output: get more
                                                          high-Whittle-index tasks done
  + ON_TIME_BONUS      * sum_t on_time[t]             -- prefer finishing before due_date
                                                          when there is a genuine choice
  + COORDINATION_BONUS * sum_w combined_window[w]        -- reward multi-department combination
  - WINDOW_OPEN_COST   * sum_w window_open[w]             -- a small nudge against opening
                                                             more separate block grants than
                                                             necessary when a schedule is
                                                             otherwise tied, not a real
                                                             disruption cost -- see Session 26
                                                             below, it is deliberately kept
                                                             too low to ever block a task from
                                                             being scheduled.

All weight constants below are illustrative, tunable, and documented as
such -- not fitted to any real IR cost data (none exists publicly, per
Session 1's dataset findings).

Session 26, at explicit user request ("our goal is to schedule all the
tasks that are coming"): WINDOW_OPEN_COST was originally 20, high enough
that a Routine-priority task (priority_value ~12-14 at PRIORITY_VALUE_
SCALE=10) could never justify opening its own window even in a
completely empty section -- a real, deliberate "minimize disruption
events" framing from the problem statement, but never something the user
asked for at that specific weight, and in direct tension with getting
every raised task scheduled. Dropped to 1 (priority_value is clipped to a
minimum of 1, so even the lowest-ranked task now breaks even opening its
own window rather than losing outright) -- COORDINATION_BONUS is
unchanged and still real money on the table, so combining departments
into a shared window remains clearly favoured whenever it's genuinely
possible, this just stops it being the ONLY way a low-priority task can
ever get scheduled at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date, timedelta

import pandas as pd
from ortools.sat.python import cp_model

from railblock.config import CPSAT_SEARCH_WORKERS
from railblock.scheduling.capacity import compute_daily_window_capacity
from railblock.scheduling.disconnection_procedure import REQUIRED_SIGNOFF_ROLES_JOINT_SCHEDULE
from railblock.synthetic.maintenance_tasks import (
    SM_DIRECT_APPROVAL_MAX_HOURS,
    DEPARTMENTS,
    approval_path_for,
)

ON_TIME_BONUS = 5
COORDINATION_BONUS = 50
WINDOW_OPEN_COST = 1
PRIORITY_VALUE_SCALE = 10  # whittle_index resolution kept when rounded to an integer objective weight
MEETS_TARGET_BONUS = 30  # Layer 4 handoff: reward hitting a section's monthly-derived weekly target


@dataclass
class ScheduleResult:
    schedule: pd.DataFrame       # one row per scheduled task
    unscheduled: pd.DataFrame    # ranked tasks that didn't fit this horizon
    windows: pd.DataFrame        # one row per opened (section, date, window_index) window
    status: str
    solve_time_s: float
    objective_value: float


def solve_schedule(
    ranked_tasks: pd.DataFrame,
    sections: pd.DataFrame,
    passenger_occupancy: pd.DataFrame,
    goods_occupancy: pd.DataFrame,
    start_date: Date,
    n_days: int = 7,
    time_limit_s: float = 30.0,
    capacity: pd.DataFrame | None = None,
    weekly_targets: dict[str, float] | None = None,
    coordination_bonus: float = COORDINATION_BONUS,
    window_open_cost: float = WINDOW_OPEN_COST,
    on_time_bonus: float = ON_TIME_BONUS,
    allowed_weekdays: set[int] | None = None,
) -> ScheduleResult:
    """Assign `ranked_tasks` (Layer 2's output, needs whittle_index) into
    block windows over [start_date, start_date + n_days) subject to Layer
    1's real per-section capacity and the multi-department synchronisation
    constraint. Returns a ScheduleResult with a feasible (never
    capacity-violating) assignment -- not every task will necessarily fit;
    unscheduled tasks are returned too, never silently dropped.

    `capacity`: pass a precomputed compute_daily_window_capacity() result
    to avoid recomputing it (callers like solve_schedule_with_options
    already need it for Option 2's splitting decision); computed here if
    omitted.

    `weekly_targets`: Layer 4 handoff -- {section_id: target_minutes} for
    THIS n_days horizon, as derived by railblock.scheduling.monthly's
    solve_monthly_plan. A SOFT cap: sections named here get an objective
    bonus (MEETS_TARGET_BONUS) for reaching their target total scheduled
    minutes this run, nudging the weekly solver towards the monthly plan's
    allocation without ever hard-blocking it from scheduling more (or
    less) if the operational, day/window-level picture calls for it.
    """
    if ranked_tasks.empty:
        # Reachable in practice: the critical-task escalation workflow
        # (critical_task_escalation.py) can resolve every single pending
        # task via escalation, leaving nothing for the solver at all.
        # `ranked_tasks[...]` below assumes real columns exist -- an
        # empty-but-column-less DataFrame (the common shape for "nothing
        # left") has none, so this must short-circuit before touching it.
        return ScheduleResult(
            schedule=pd.DataFrame(), unscheduled=pd.DataFrame(), windows=pd.DataFrame(),
            status="OPTIMAL", solve_time_s=0.0, objective_value=0.0,
        )
    if capacity is None:
        capacity = compute_daily_window_capacity(sections, passenger_occupancy, goods_occupancy, start_date, n_days)
    if allowed_weekdays is not None:
        # Session 9: a real, principled constraint restricting which of Layer 1's
        # actual candidate windows are even offered to the solver (e.g. Sat/Sun
        # only, or Mon-Fri only) -- used by generate_schedule_options() to produce
        # genuinely different schedules, not a cosmetic filter on the output.
        capacity = capacity[
            pd.to_datetime(capacity["date"]).dt.weekday.isin(allowed_weekdays)
        ]
    windows_by_section: dict[str, list[tuple[str, int]]] = {s: [] for s in sections["section_id"]}
    window_minutes: dict[tuple[str, str, int], float] = {}
    window_span: dict[tuple[str, str, int], tuple[float, float]] = {}
    for _, row in capacity.iterrows():
        key = (row["section_id"], row["date"], int(row["window_index"]))
        window_minutes[key] = row["window_minutes"]
        window_span[key] = (row.get("start_minute"), row.get("end_minute"))
        windows_by_section[row["section_id"]].append((row["date"], int(row["window_index"])))

    all_windows = list(window_minutes.keys())  # (section_id, date, window_index)

    tasks = ranked_tasks.reset_index(drop=True)
    task_hours_min = (tasks["estimated_block_hours"] * 60).round().astype(int)
    priority_value = (tasks["whittle_index"] * PRIORITY_VALUE_SCALE).round().astype(int).clip(lower=1)
    due_dates = pd.to_datetime(tasks["due_date"])

    model = cp_model.CpModel()

    window_open = {w: model.NewBoolVar(f"window_open{w}") for w in all_windows}
    assign = {}
    for t in tasks.index:
        section_id = tasks.at[t, "section_id"]
        for (d, k) in windows_by_section[section_id]:
            assign[(t, section_id, d, k)] = model.NewBoolVar(f"assign[{t},{section_id},{d},{k}]")
    scheduled = {t: model.NewBoolVar(f"scheduled[{t}]") for t in tasks.index}
    on_time = {t: model.NewBoolVar(f"on_time[{t}]") for t in tasks.index}

    for t in tasks.index:
        section_id = tasks.at[t, "section_id"]
        my_windows = windows_by_section[section_id]
        assign_vars = [assign[(t, section_id, d, k)] for (d, k) in my_windows]
        if assign_vars:
            model.Add(sum(assign_vars) <= 1)
            model.Add(sum(assign_vars) == scheduled[t])
        else:
            model.Add(scheduled[t] == 0)  # no usable window exists for this section at all

        for (d, k) in my_windows:
            model.Add(assign[(t, section_id, d, k)] <= window_open[(section_id, d, k)])

        due = due_dates[t]
        on_time_vars = [assign[(t, section_id, d, k)] for (d, k) in my_windows if pd.Timestamp(d) <= due]
        if on_time_vars:
            model.Add(sum(on_time_vars) >= on_time[t])
        else:
            model.Add(on_time[t] == 0)
        model.Add(on_time[t] <= scheduled[t])

    # Capacity per DEPARTMENT, not pooled across all tasks in a window: a
    # real joint disconnection (SR 3.51.6) lets multiple departments' own
    # crews work CONCURRENTLY within the same granted block -- one
    # department's own tasks still stack sequentially (one crew can't be
    # in two places), but a different department sharing the same window
    # doesn't add to its duration, it runs alongside it. So the window
    # only needs to be as long as its busiest single department, not the
    # sum of every task assigned to it -- enforced by requiring EACH
    # department present to independently fit within window_minutes,
    # which is exactly max(dept totals) without needing a max() variable.
    for s in sections["section_id"]:
        for (d, k) in windows_by_section[s]:
            w = (s, d, k)
            for dep in DEPARTMENTS:
                dept_tasks = tasks.index[(tasks["section_id"] == s) & (tasks["department"] == dep)].tolist()
                if not dept_tasks:
                    continue
                dept_usage = sum(int(task_hours_min[t]) * assign[(t, s, d, k)] for t in dept_tasks)
                model.Add(dept_usage <= int(round(window_minutes[w])) * window_open[w])

    dept_present = {}
    combined_window = {}
    for s in sections["section_id"]:
        tasks_by_dept = {
            dep: tasks.index[(tasks["section_id"] == s) & (tasks["department"] == dep)].tolist()
            for dep in DEPARTMENTS
        }
        for (d, k) in windows_by_section[s]:
            w = (s, d, k)
            present_vars = []
            for dep in DEPARTMENTS:
                var = model.NewBoolVar(f"dept_present[{w},{dep}]")
                dept_assigns = [assign[(t, s, d, k)] for t in tasks_by_dept[dep]]
                if dept_assigns:
                    model.Add(sum(dept_assigns) >= 1).OnlyEnforceIf(var)
                    model.Add(sum(dept_assigns) == 0).OnlyEnforceIf(var.Not())
                else:
                    model.Add(var == 0)
                dept_present[(w, dep)] = var
                present_vars.append(var)

            combo = model.NewBoolVar(f"combined_window{w}")
            model.Add(sum(present_vars) >= 2).OnlyEnforceIf(combo)
            model.Add(sum(present_vars) <= 1).OnlyEnforceIf(combo.Not())
            combined_window[w] = combo

    meets_target = {}
    if weekly_targets:
        for s, target_minutes in weekly_targets.items():
            if s not in windows_by_section or not windows_by_section[s]:
                continue
            tasks_in_section = tasks.index[tasks["section_id"] == s].tolist()
            section_total_minutes = sum(
                int(task_hours_min[t]) * assign[(t, s, d, k)]
                for t in tasks_in_section
                for (d, k) in windows_by_section[s]
            )
            var = model.NewBoolVar(f"meets_target[{s}]")
            target_min_int = int(round(target_minutes))
            model.Add(section_total_minutes >= target_min_int).OnlyEnforceIf(var)
            model.Add(section_total_minutes < target_min_int).OnlyEnforceIf(var.Not())
            meets_target[s] = var

    objective_terms = []
    for t in tasks.index:
        objective_terms.append(int(priority_value[t]) * scheduled[t])
        objective_terms.append(int(on_time_bonus) * on_time[t])
    for w in all_windows:
        objective_terms.append(int(coordination_bonus) * combined_window[w])
        objective_terms.append(-int(window_open_cost) * window_open[w])
    for var in meets_target.values():
        objective_terms.append(MEETS_TARGET_BONUS * var)
    model.Maximize(sum(objective_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_search_workers = CPSAT_SEARCH_WORKERS
    # Session 22, at explicit user request, after a real bug traced to
    # this: with 8 parallel search workers and no fixed seed, CP-SAT's
    # internal worker-race timing is genuinely nondeterministic, so
    # identical inputs could return a DIFFERENT (equally OPTIMAL, since
    # several distinct task selections can tie on total objective value)
    # solution from one call to the next. This only ever bit the
    # critical-tasks-first flow's two-call design (POST /schedule/options
    # solves Critical tasks once to find who needs escalation; POST
    # /schedule/escalate deliberately re-solves the SAME Critical tasks
    # fresh rather than persisting state -- see api/app.py's
    # _finalize_schedule docstring) -- if that second solve happened to
    # schedule a different SET of Critical tasks than the first, a
    # decision collected for one round's specific unscheduled task could
    # land on a task the second round already scheduled by itself,
    # while the second round's own (different) unscheduled task never
    # got a decision applied to it at all. A fixed seed makes identical
    # inputs always return the identical solution -- no scheduling
    # QUALITY change, only reproducibility.
    solver.parameters.random_seed = 42
    status = solver.Solve(model)
    status_name = solver.StatusName(status)
    solve_time_s = solver.WallTime()
    solved_ok = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    schedule_rows = []
    if solved_ok:
        for t in tasks.index:
            if solver.Value(scheduled[t]):
                section_id = tasks.at[t, "section_id"]
                (d, k) = next(
                    (d, k) for (d, k) in windows_by_section[section_id]
                    if solver.Value(assign[(t, section_id, d, k)])
                )
                w_start, w_end = window_span.get((section_id, d, k), (None, None))
                schedule_rows.append(
                    {
                        "task_id": tasks.at[t, "task_id"],
                        "parent_task_id": tasks.at[t, "parent_task_id"] if "parent_task_id" in tasks.columns else tasks.at[t, "task_id"],
                        "n_parts": tasks.at[t, "n_parts"] if "n_parts" in tasks.columns else 1,
                        "part_index": tasks.at[t, "part_index"] if "part_index" in tasks.columns else 1,
                        "department": tasks.at[t, "department"],
                        "section_id": section_id,
                        "date": d,
                        "window_index": k,
                        "start_minute": w_start,
                        "end_minute": w_end,
                        "estimated_block_hours": tasks.at[t, "estimated_block_hours"],
                        "requester_priority": tasks.at[t, "requester_priority"],
                        "whittle_index": tasks.at[t, "whittle_index"],
                        "on_time": bool(solver.Value(on_time[t])),
                        "approval_path": tasks.at[t, "approval_path"] if "approval_path" in tasks.columns else approval_path_for(tasks.at[t, "estimated_block_hours"]),
                        "completion_verified": False,  # SR 3.51.6: reconnection requires a Track Fit Certificate + joint SI/SM testing -- never implied by the window merely ending
                    }
                )
    schedule_df = pd.DataFrame(schedule_rows)
    scheduled_ids = set(schedule_df["task_id"]) if not schedule_df.empty else set()
    unscheduled_df = tasks[~tasks["task_id"].isin(scheduled_ids)].copy()

    window_rows = []
    if solved_ok:
        for w in all_windows:
            if solver.Value(window_open[w]):
                s, d, k = w
                depts = [dep for dep in DEPARTMENTS if solver.Value(dept_present[(w, dep)])]
                assigned_here = [
                    t for t in tasks.index
                    if tasks.at[t, "section_id"] == s and solver.Value(assign[(t, s, d, k)])
                ]
                # Matches the per-department capacity constraint above: the
                # window's real occupied time is the BUSIEST single
                # department's own total (concurrent departments don't add),
                # not the sum of every task regardless of department.
                minutes_used_by_dept = {}
                for t in assigned_here:
                    dep = tasks.at[t, "department"]
                    minutes_used_by_dept[dep] = minutes_used_by_dept.get(dep, 0) + int(task_hours_min[t])
                minutes_used = max(minutes_used_by_dept.values(), default=0)
                is_combined = bool(solver.Value(combined_window[w]))
                # SR 3.51.6: a combined block that includes any task needing more than
                # an hour's disconnection requires a joint schedule signed by all four
                # roles, not just SM approval -- see disconnection_procedure.py.
                needs_joint_signoff = is_combined and any(
                    tasks.at[t, "estimated_block_hours"] > SM_DIRECT_APPROVAL_MAX_HOURS for t in assigned_here
                )
                w_start, w_end = window_span.get(w, (None, None))
                window_rows.append(
                    {
                        "section_id": s,
                        "date": d,
                        "window_index": k,
                        "start_minute": w_start,
                        "end_minute": w_end,
                        "minutes_used": int(minutes_used),
                        "minutes_capacity": int(round(window_minutes[w])),
                        "departments": ",".join(depts),
                        "task_ids": [tasks.at[t, "task_id"] for t in assigned_here],
                        "combined": is_combined,
                        "required_signoffs": REQUIRED_SIGNOFF_ROLES_JOINT_SCHEDULE if needs_joint_signoff else [],
                    }
                )
    windows_df = pd.DataFrame(window_rows)

    objective_value = solver.ObjectiveValue() if solved_ok else float("nan")

    return ScheduleResult(
        schedule=schedule_df,
        unscheduled=unscheduled_df,
        windows=windows_df,
        status=status_name,
        solve_time_s=solve_time_s,
        objective_value=objective_value,
    )
