from datetime import date

import pandas as pd
import pytest

from railblock.scheduling.capacity import compute_daily_window_capacity
from railblock.scheduling.model import solve_schedule


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


def _task(task_id, section_id, hours, priority="Critical", department="Engineering", due="2026-12-31", whittle=100.0):
    return {
        "task_id": task_id,
        "department": department,
        "section_id": section_id,
        "estimated_block_hours": hours,
        "requester_priority": priority,
        "due_date": due,
        "whittle_index": whittle,
    }


# --------------------------------------------------------------- capacity

def test_daily_window_capacity_matches_availability_free_minutes():
    sections = _sections()
    pax = _pax([{"section_id": "A-B", "train_no": "1", "start_minute": 0, "end_minute": 700, "source": "real_timetable"}])
    cap = compute_daily_window_capacity(sections, pax, _goods(), date(2026, 9, 7), n_days=1)
    ab = cap[cap["section_id"] == "A-B"].iloc[0]
    assert ab["window_minutes"] == pytest.approx(1440 - 700)
    cd = cap[cap["section_id"] == "C-D"].iloc[0]
    assert cd["window_minutes"] == pytest.approx(1440)


# ----------------------------------------------------------- feasibility

def test_solver_never_double_books_a_section():
    sections = _sections()
    # A-B has only 100 minutes free per day
    pax = _pax([{"section_id": "A-B", "train_no": "1", "start_minute": 0, "end_minute": 1340, "source": "real_timetable"}])
    tasks = pd.DataFrame(
        [_task(f"T{i}", "A-B", hours=1.5, whittle=100.0 - i) for i in range(10)]
    )
    result = solve_schedule(tasks, sections, pax, _goods(), date(2026, 9, 7), n_days=5, time_limit_s=15)
    assert result.status in ("OPTIMAL", "FEASIBLE")
    for _, w in result.windows.iterrows():
        assert w["minutes_used"] <= w["minutes_capacity"]


def test_solver_respects_capacity_and_leaves_excess_unscheduled():
    sections = _sections()
    # Only 60 minutes/day free on A-B, 20 tasks each needing 2 hours -- can't all fit in 1 day
    pax = _pax([{"section_id": "A-B", "train_no": "1", "start_minute": 0, "end_minute": 1380, "source": "real_timetable"}])
    tasks = pd.DataFrame(
        [_task(f"T{i}", "A-B", hours=2.0, whittle=1000.0 - i) for i in range(20)]
    )
    result = solve_schedule(tasks, sections, pax, _goods(), date(2026, 9, 7), n_days=1, time_limit_s=15)
    assert result.status in ("OPTIMAL", "FEASIBLE")
    assert len(result.schedule) == 0  # 60 min capacity can't fit a single 2h task
    assert len(result.unscheduled) == 20


def test_solver_prefers_higher_whittle_index_tasks_when_capacity_is_scarce():
    sections = _sections()
    pax = _pax([{"section_id": "A-B", "train_no": "1", "start_minute": 0, "end_minute": 1320, "source": "real_timetable"}])  # 120 min free/day
    tasks = pd.DataFrame(
        [
            _task("LOW", "A-B", hours=2.0, whittle=10.0),
            _task("HIGH", "A-B", hours=2.0, whittle=500.0),
        ]
    )
    result = solve_schedule(tasks, sections, pax, _goods(), date(2026, 9, 7), n_days=1, time_limit_s=10)
    scheduled_ids = set(result.schedule["task_id"]) if not result.schedule.empty else set()
    assert "HIGH" in scheduled_ids
    assert "LOW" not in scheduled_ids  # only one 2h task fits in a single 2h (120 min) window


# --------------------------------------------------- department combination

def test_combinable_cross_department_tasks_share_one_window():
    sections = _sections()
    tasks = pd.DataFrame(
        [
            _task("ENG", "A-B", hours=2.0, department="Engineering"),
            _task("SIG", "A-B", hours=2.0, department="Signalling"),
        ]
    )
    result = solve_schedule(tasks, sections, _pax([]), _goods(), date(2026, 9, 7), n_days=3, time_limit_s=10)
    assert len(result.schedule) == 2
    dates_used = result.schedule["date"].unique()
    assert len(dates_used) == 1  # both landed on the same day
    window = result.windows[(result.windows["section_id"] == "A-B") & (result.windows["date"] == dates_used[0])]
    assert window.iloc[0]["combined"]
    assert set(window.iloc[0]["departments"].split(",")) == {"Engineering", "Signalling"}


def test_non_combinable_departments_still_scheduled_without_conflict_when_capacity_allows():
    sections = _sections()
    # Plenty of capacity across 5 days -- both tasks should get scheduled,
    # combined or not, without ever exceeding capacity.
    tasks = pd.DataFrame(
        [
            _task("ENG", "A-B", hours=3.0, department="Engineering"),
            _task("TRAC", "A-B", hours=3.0, department="Traction"),
        ]
    )
    result = solve_schedule(tasks, sections, _pax([]), _goods(), date(2026, 9, 7), n_days=5, time_limit_s=10)
    assert len(result.schedule) == 2
    for _, w in result.windows.iterrows():
        assert w["minutes_used"] <= w["minutes_capacity"]
