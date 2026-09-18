from datetime import date, timedelta

import pandas as pd
import pytest

from railblock.scheduling.rolling_horizon import DAYS_PER_WEEK, run_rolling_horizon


def _sections():
    return pd.DataFrame([{"section_id": "A-B", "length_km": 20.0}])


def _pax_one_window_per_week():
    """Section A-B is fully occupied every day EXCEPT Monday, and even
    Monday only offers a single ~150-minute gap -- exactly enough for ONE
    120-minute task per week, never two. This recurs identically every
    week (weekday-based, not date-range-based), so the scarcity is the
    same in week 1 and week 2 -- carryover, not a capacity change, is
    what has to explain a week-2 success.
    """
    cols = ["section_id", "train_no", "start_minute", "end_minute", "source", "weekdays"]
    return pd.DataFrame(
        [
            {  # Tue-Sun: fully occupied
                "section_id": "A-B", "train_no": "1", "start_minute": 0, "end_minute": 1440,
                "source": "real_timetable", "weekdays": frozenset({1, 2, 3, 4, 5, 6}),
            },
            {  # Monday: occupied except the last 150 minutes
                "section_id": "A-B", "train_no": "2", "start_minute": 0, "end_minute": 1290,
                "source": "real_timetable", "weekdays": frozenset({0}),
            },
        ],
        columns=cols,
    )


def _empty_goods():
    cols = ["section_id", "date", "start_minute", "end_minute"]
    return pd.DataFrame(columns=cols)


def _task(task_id, priority, due_date, hours=2.0, department="Engineering", defect_type="Ballast deficiency"):
    return {
        "task_id": task_id,
        "department": department,
        "source_system": "TMS",
        "section_id": "A-B",
        "defect_type": defect_type,
        "requester_priority": priority,
        "raised_date": due_date.isoformat(),
        "due_date": due_date.isoformat(),
        "days_overdue": 0,
        "estimated_block_hours": hours,
        "splittable": False,
        "data_source": "SYNTHETIC",
    }


def _next_monday(d: date) -> date:
    return d + timedelta(days=(7 - d.weekday()) % 7)


def test_carryover_a_task_that_loses_week_1_succeeds_in_week_2_not_lost_forever():
    sections = _sections()
    pax = _pax_one_window_per_week()
    start = _next_monday(date(2026, 9, 7))  # ensure the window-bearing weekday lines up predictably

    # both need the corridor's ONLY 150-minute weekly window (120 min each -- 150 fits one, not two)
    tasks = pd.DataFrame(
        [
            _task("HIGH", "Critical", due_date=start + timedelta(days=3)),   # should win week 1
            _task("LOW", "Routine", due_date=start + timedelta(days=60)),    # should lose week 1, win week 2
        ]
    )

    result = run_rolling_horizon(
        tasks, sections, pax, horizon_days=2 * DAYS_PER_WEEK, start_date=start,
        time_limit_s=15, goods_occupancy_override=_empty_goods(),
    )

    assert result.n_weeks == 2
    outcomes = result.outcomes.set_index("task_id")

    assert outcomes.loc["HIGH", "status"] in ("on_time", "late")  # scheduled in week 1
    assert outcomes.loc["HIGH", "weeks_pending"] == 0  # never had to wait

    # the actual carryover proof: LOW was NOT permanently lost after failing week 1
    assert outcomes.loc["LOW", "status"] in ("on_time", "late")
    assert outcomes.loc["LOW", "status"] != "unscheduled"
    assert outcomes.loc["LOW", "weeks_pending"] >= 1  # it did have to wait at least one week
    assert outcomes.loc["LOW", "completed_date"] is not None
    # LOW's due date is 60 days out and it can only have succeeded in week 2 (day 7-13) at the latest -> on_time
    assert outcomes.loc["LOW", "status"] == "on_time"

    assert len(result.weekly_option_counts) == 2


def test_without_enough_horizon_a_perpetually_losing_task_stays_unscheduled_not_silently_dropped():
    """Sanity check on the other side: a task that NEVER wins any week's
    single slot (because a fresh higher-priority competitor appears each
    week) is reported as unscheduled with weeks_pending tracking every
    week it waited -- not silently disappearing from the outcomes table.
    """
    sections = _sections()
    pax = _pax_one_window_per_week()
    start = _next_monday(date(2026, 9, 7))

    tasks = pd.DataFrame(
        [
            _task("ALWAYS_LOSES", "Routine", due_date=start + timedelta(days=90), hours=2.0),
            _task("BLOCKER", "Critical", due_date=start + timedelta(days=3), hours=2.0),
        ]
    )
    result = run_rolling_horizon(
        tasks, sections, pax, horizon_days=1 * DAYS_PER_WEEK, start_date=start,
        time_limit_s=15, goods_occupancy_override=_empty_goods(),
    )
    assert "ALWAYS_LOSES" in set(result.outcomes["task_id"])  # still present in the table, not dropped
    outcomes = result.outcomes.set_index("task_id")
    assert outcomes.loc["ALWAYS_LOSES", "status"] == "unscheduled"
    assert outcomes.loc["ALWAYS_LOSES", "weeks_pending"] == 1
