from datetime import date

import pandas as pd
import pytest

from railblock.scheduling.monthly import compute_weekly_capacity_budget, solve_monthly_plan
from railblock.scheduling.layer4 import (
    generate_monthly_plan,
    generate_weekly_plan,
    generate_weekly_plan_with_monthly_handoff,
)


def _sections():
    return pd.DataFrame(
        [
            {"section_id": "A-B", "length_km": 20.0},
            {"section_id": "C-D", "length_km": 20.0},
        ]
    )


def _pax(rows=()):
    cols = ["section_id", "train_no", "start_minute", "end_minute", "source"]
    return pd.DataFrame(list(rows), columns=cols)


def _goods(rows=()):
    cols = ["section_id", "date", "start_minute", "end_minute"]
    return pd.DataFrame(list(rows), columns=cols)


def _task(task_id, section_id, hours, whittle_index=100.0, priority="Moderate", splittable=False, due="2026-12-31"):
    return {
        "task_id": task_id,
        "department": "Engineering",
        "section_id": section_id,
        "estimated_block_hours": hours,
        "requester_priority": priority,
        "due_date": due,
        "whittle_index": whittle_index,
        "splittable": splittable,
    }


MONDAY = date(2026, 9, 7)


def test_weekly_capacity_budget_sums_free_minutes_across_seven_days():
    pax = _pax([{"section_id": "A-B", "train_no": "1", "start_minute": 0, "end_minute": 700, "source": "real_timetable"}])
    budget = compute_weekly_capacity_budget(_sections(), pax, _goods(), MONDAY, n_weeks=1)
    ab = budget[budget["section_id"] == "A-B"].iloc[0]
    assert ab["free_minutes"] == pytest.approx((1440 - 700) * 7)


def test_monthly_plan_respects_weekly_capacity():
    sections = _sections()
    pax = _pax([{"section_id": "A-B", "train_no": "1", "start_minute": 0, "end_minute": 1400, "source": "real_timetable"}])  # 40 min/day = 280 min/week free
    tasks = pd.DataFrame([_task(f"T{i}", "A-B", hours=1.0) for i in range(10)])  # 10x60min tasks, way more than 280min/week x4 weeks fits some but not all

    result = solve_monthly_plan(tasks, sections, pax, _goods(), MONDAY, n_weeks=4, time_limit_s=15)
    assert result.status in ("OPTIMAL", "FEASIBLE")
    for _, row in result.weekly_targets.iterrows():
        cap = 280.0  # 40*7
        assert row["target_minutes"] <= cap + 1e-6


def test_generate_weekly_plan_is_callable_directly():
    sections = _sections()
    tasks = pd.DataFrame([_task("T1", "A-B", hours=1.0, priority="Critical")])
    result = generate_weekly_plan(tasks, sections, _pax([]), _goods(), MONDAY, time_limit_s=10)
    assert result.status in ("OPTIMAL", "FEASIBLE")
    assert len(result.schedule) == 1


def test_generate_monthly_plan_is_callable_directly():
    sections = _sections()
    tasks = pd.DataFrame([_task("T1", "A-B", hours=1.0)])
    result = generate_monthly_plan(tasks, sections, _pax([]), _goods(), MONDAY, n_weeks=4, time_limit_s=10)
    assert result.status in ("OPTIMAL", "FEASIBLE")


def test_monthly_handoff_wiring_runs_end_to_end_and_is_internally_consistent():
    sections = _sections()
    pax = _pax([])
    monthly_tasks = pd.DataFrame(
        [_task(f"CD{i}", "C-D", hours=2.0, whittle_index=500.0) for i in range(20)]
        + [_task("AB1", "A-B", hours=1.0, whittle_index=10.0)]
    )

    monthly_result, weekly_result = generate_weekly_plan_with_monthly_handoff(
        monthly_tasks, sections, pax, _goods(), MONDAY, week_number=1, n_weeks=4, time_limit_s=15
    )
    assert monthly_result.status in ("OPTIMAL", "FEASIBLE")
    assert weekly_result.status in ("OPTIMAL", "FEASIBLE")
    # every assigned task lands in exactly one of the 4 weeks, never more
    if not monthly_result.assigned_tasks.empty:
        assert monthly_result.assigned_tasks["task_id"].is_unique


def test_weekly_target_soft_cap_actually_changes_the_solver_choice():
    """Direct, deterministic proof that a weekly_target changes what the
    weekly CP-SAT solve chooses to schedule: a small task whose own
    priority value is less than the cost of opening a second window is
    left unscheduled on its own merits (opening a window just for it is a
    net loss) -- but once hitting a section's monthly-derived target is
    itself worth more than that opening cost, the solver schedules it to
    claim the bonus. This is model.py's MEETS_TARGET_BONUS mechanism,
    exercised directly rather than through the monthly plan's own
    (arbitrary, when capacity is ample) week assignment.

    Session 26, at explicit user request ("schedule all the tasks that are
    coming"): WINDOW_OPEN_COST's production default dropped from 20 to 1,
    specifically so a low-priority task is no longer locked out of its own
    window -- which is exactly the lever this test needs to demonstrate
    the OPPOSITE thing (a task genuinely not worth scheduling without the
    target's bonus). window_open_cost=20 is pinned explicitly here, not
    via the module default, purely to isolate and exercise MEETS_TARGET_
    BONUS's mechanism on its own terms; it says nothing about what the
    real scheduler now uses.
    """
    from railblock.scheduling.model import solve_schedule

    sections = _sections()
    # day 1: a 100-minute window; day 2: a separate 100-minute window (own WINDOW_OPEN_COST)
    pax = _pax(
        [
            {"section_id": "A-B", "train_no": "1", "start_minute": 100, "end_minute": 1440, "source": "real_timetable"},
        ]
    )
    tasks = pd.DataFrame(
        [
            _task("BIG", "A-B", hours=90 / 60, whittle_index=50.0),   # priority_value=500, fits day 1 alone
            _task("SMALL", "A-B", hours=20 / 60, whittle_index=1.0),  # priority_value=10 < the pinned window_open_cost=20
        ]
    )

    without_target = solve_schedule(tasks, sections, pax, _goods(), MONDAY, n_days=2, time_limit_s=10, window_open_cost=20)
    scheduled_without = set(without_target.schedule["task_id"]) if not without_target.schedule.empty else set()
    assert scheduled_without == {"BIG"}  # SMALL isn't worth its own window on priority value alone

    with_target = solve_schedule(
        tasks, sections, pax, _goods(), MONDAY, n_days=2, time_limit_s=10, window_open_cost=20,
        weekly_targets={"A-B": 110.0},  # only reachable if BOTH tasks' minutes are scheduled
    )
    scheduled_with = set(with_target.schedule["task_id"]) if not with_target.schedule.empty else set()
    assert scheduled_with == {"BIG", "SMALL"}  # the target's bonus tips SMALL into being worth scheduling


def test_more_horizon_helps_now_that_options_1_and_2_relax_the_bottleneck():
    """Session 3's finding was that extending n_days didn't help because
    eligibility (not day-count) was the bottleneck under the old "every
    train every day, no splitting" model. With Option 1 (realistic
    weekdays) and Option 2 (splitting) active, a longer horizon should now
    let MORE total task-hours be scheduled, since splittable tasks can use
    additional nights/sessions across more days.
    """
    sections = _sections()
    # a section whose free capacity is real but modest per day
    pax = _pax([{"section_id": "A-B", "train_no": "1", "start_minute": 0, "end_minute": 1380, "source": "real_timetable"}])  # 60 min/day free
    tasks = pd.DataFrame(
        [_task(f"T{i}", "A-B", hours=1.0, splittable=True, priority="Critical", whittle_index=100.0 - i) for i in range(20)]
    )

    from railblock.scheduling.orchestrator import solve_schedule_with_options

    short = solve_schedule_with_options(tasks, sections, pax, _goods(), MONDAY, n_days=2, time_limit_s=15)
    long = solve_schedule_with_options(tasks, sections, pax, _goods(), MONDAY, n_days=10, time_limit_s=15)

    n_done_short = short.option_counts["whole"] + short.option_counts["split"]
    n_done_long = long.option_counts["whole"] + long.option_counts["split"]
    assert n_done_long >= n_done_short
    assert n_done_long > n_done_short  # strictly more with 5x the horizon and identical daily capacity
