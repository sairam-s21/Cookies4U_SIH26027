from datetime import date

import pandas as pd
import pytest

from railblock.scheduling.disconnection_procedure import REQUIRED_SIGNOFF_ROLES_JOINT_SCHEDULE
from railblock.scheduling.model import solve_schedule
from railblock.synthetic.maintenance_tasks import (
    SM_DIRECT_APPROVAL_MAX_HOURS,
    approval_path_for,
    generate_maintenance_tasks,
)


def _sections():
    return pd.DataFrame([{"section_id": "A-B", "length_km": 20.0}])


def _pax(rows=()):
    cols = ["section_id", "train_no", "start_minute", "end_minute", "source"]
    return pd.DataFrame(list(rows), columns=cols)


def _goods(rows=()):
    cols = ["section_id", "date", "start_minute", "end_minute"]
    return pd.DataFrame(list(rows), columns=cols)


def _task(task_id, hours, department="Engineering", whittle_index=100.0, due="2026-12-31"):
    return {
        "task_id": task_id,
        "department": department,
        "section_id": "A-B",
        "estimated_block_hours": hours,
        "requester_priority": "Critical",
        "due_date": due,
        "whittle_index": whittle_index,
        "approval_path": approval_path_for(hours),
    }


MONDAY = date(2026, 9, 7)


# ------------------------------------------------------------ approval_path

def test_approval_path_boundary_at_one_hour():
    assert approval_path_for(1.0) == "sm_direct"
    assert approval_path_for(0.5) == "sm_direct"
    assert approval_path_for(1.01) == "joint_schedule"
    assert approval_path_for(6.0) == "joint_schedule"


def test_generated_tasks_carry_correct_approval_path():
    sections = pd.DataFrame([{"section_id": "A-B", "length_km": 20.0}])
    tasks = generate_maintenance_tasks(sections, n_tasks=300, seed=3)
    for _, row in tasks.iterrows():
        expected = "sm_direct" if row["estimated_block_hours"] <= SM_DIRECT_APPROVAL_MAX_HOURS else "joint_schedule"
        assert row["approval_path"] == expected


# --------------------------------------------------- scheduled task fields

def test_scheduled_tasks_carry_approval_path_and_unverified_completion():
    sections = _sections()
    tasks = pd.DataFrame([_task("BIG", 2.0), _task("SMALL", 0.5, whittle_index=50.0)])
    result = solve_schedule(tasks, sections, _pax([]), _goods(), MONDAY, n_days=3, time_limit_s=10)
    assert not result.schedule.empty
    assert set(result.schedule["approval_path"]) <= {"sm_direct", "joint_schedule"}
    assert (result.schedule["completion_verified"] == False).all()  # noqa: E712
    big_row = result.schedule[result.schedule["task_id"] == "BIG"]
    if not big_row.empty:
        assert big_row.iloc[0]["approval_path"] == "joint_schedule"
    small_row = result.schedule[result.schedule["task_id"] == "SMALL"]
    if not small_row.empty:
        assert small_row.iloc[0]["approval_path"] == "sm_direct"


# ------------------------------------------------------- sign-off tagging

def test_combined_window_with_over_one_hour_task_requires_joint_signoff():
    sections = _sections()
    tasks = pd.DataFrame(
        [
            _task("ENG", 1.5, department="Engineering"),   # >1h -> needs joint schedule
            _task("SIG", 0.5, department="Signalling"),
        ]
    )
    result = solve_schedule(tasks, sections, _pax([]), _goods(), MONDAY, n_days=3, time_limit_s=10)
    combined = result.windows[result.windows["combined"]]
    assert not combined.empty
    assert combined.iloc[0]["required_signoffs"] == REQUIRED_SIGNOFF_ROLES_JOINT_SCHEDULE


def test_combined_window_with_only_short_tasks_needs_no_signoff():
    sections = _sections()
    tasks = pd.DataFrame(
        [
            _task("ENG", 0.5, department="Engineering"),
            _task("SIG", 0.5, department="Signalling"),
        ]
    )
    result = solve_schedule(tasks, sections, _pax([]), _goods(), MONDAY, n_days=3, time_limit_s=10)
    combined = result.windows[result.windows["combined"]]
    assert not combined.empty
    assert combined.iloc[0]["required_signoffs"] == []


def test_non_combined_window_never_gets_signoffs():
    sections = _sections()
    tasks = pd.DataFrame([_task("SOLO", 1.5)])
    result = solve_schedule(tasks, sections, _pax([]), _goods(), MONDAY, n_days=3, time_limit_s=10)
    assert not result.windows.empty
    assert (~result.windows["combined"]).all()
    assert (result.windows["required_signoffs"].apply(len) == 0).all()
