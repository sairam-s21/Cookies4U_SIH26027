import pandas as pd
import pytest

from railblock.scheduling.splitting import (
    MAX_SPLIT_SESSIONS,
    collapse_split_results,
    expand_splittable_tasks,
)


def _capacity(rows):
    return pd.DataFrame(rows, columns=["section_id", "date", "window_index", "window_minutes"])


def _task(task_id, section_id, hours, splittable, whittle_index=100.0):
    return {
        "task_id": task_id,
        "department": "Engineering",
        "section_id": section_id,
        "estimated_block_hours": hours,
        "requester_priority": "Critical",
        "due_date": "2026-12-31",
        "whittle_index": whittle_index,
        "splittable": splittable,
    }


def test_task_that_fits_whole_is_not_split():
    tasks = pd.DataFrame([_task("T1", "A-B", 1.5, splittable=True)])
    cap = _capacity([{"section_id": "A-B", "date": "2026-09-07", "window_index": 0, "window_minutes": 100}])
    expanded = expand_splittable_tasks(tasks, cap)
    assert len(expanded) == 1
    assert expanded.iloc[0]["n_parts"] == 1
    assert expanded.iloc[0]["task_id"] == "T1"


def test_splittable_task_that_doesnt_fit_is_split_and_hours_sum_preserved():
    tasks = pd.DataFrame([_task("T1", "A-B", 3.0, splittable=True)])  # 180 min
    cap = _capacity([{"section_id": "A-B", "date": "2026-09-07", "window_index": 0, "window_minutes": 100}])  # max window 100 min
    expanded = expand_splittable_tasks(tasks, cap)
    assert len(expanded) == 2  # 180/2=90 <= 100, so 2 parts suffice
    assert all(expanded["parent_task_id"] == "T1")
    assert expanded["n_parts"].tolist() == [2, 2]
    total_minutes = (expanded["estimated_block_hours"] * 60).sum()
    assert total_minutes == pytest.approx(180.0)


def test_split_caps_at_max_sessions():
    # Sized dynamically off MAX_SPLIT_SESSIONS itself (not a fixed
    # literal) so this stays a genuine "hits the cap, last part absorbs
    # the real remainder" case regardless of the constant's own value --
    # a real reported bug: a hardcoded 6.0h task silently stopped testing
    # this once MAX_SPLIT_SESSIONS moved to 10 (the "stop early once
    # nothing meaningful is left" logic legitimately produced fewer,
    # correctly-sized parts instead of hitting the cap at all).
    window_minutes = 50
    required_minutes = (MAX_SPLIT_SESSIONS - 1) * window_minutes + 150
    tasks = pd.DataFrame([_task("T1", "A-B", required_minutes / 60.0, splittable=True)])
    cap = _capacity([{"section_id": "A-B", "date": "2026-09-07", "window_index": 0, "window_minutes": window_minutes}])  # tiny window
    expanded = expand_splittable_tasks(tasks, cap)
    assert len(expanded) == MAX_SPLIT_SESSIONS
    total_minutes = (expanded["estimated_block_hours"] * 60).sum()
    assert total_minutes == pytest.approx(required_minutes)


def test_falls_back_to_any_real_window_when_none_fall_within_due_date():
    """A real reported bug: due_date is NOT a hard scheduling constraint
    in model.py (only on_time tagging) -- but if the search horizon's
    real windows don't happen to overlap [raised_date, due_date] at all
    (e.g. the caller's chosen start_date is well after this task's own
    due_date), sizing fell all the way through to the "no real window
    data" blind equal-split, even though real window data DID exist for
    this section elsewhere in the horizon -- on a section with unusually
    tiny real windows, that produced part sizes bigger than any real
    window, guaranteeing total failure regardless of due_date. Must fall
    back to this section's real window sizes from anywhere in the
    horizon instead of going blind."""
    tasks = pd.DataFrame([{
        "task_id": "T1", "department": "Engineering", "section_id": "A-B",
        "estimated_block_hours": 1.0, "requester_priority": "Critical", "whittle_index": 100.0,
        "splittable": True, "raised_date": "2026-09-06", "due_date": "2026-09-29",
    }])
    # every real window is well AFTER this task's own due_date
    cap = _capacity([
        {"section_id": "A-B", "date": "2026-10-10", "window_index": 0, "window_minutes": 20},
        {"section_id": "A-B", "date": "2026-10-11", "window_index": 0, "window_minutes": 20},
    ])
    expanded = expand_splittable_tasks(tasks, cap)
    # a blind equal-split of 60 min across 2 parts gives 30-min parts --
    # bigger than any real 20-min window. Sizing against the real
    # (fallback) windows instead must keep every part at or under 20.
    assert (expanded["estimated_block_hours"] * 60 <= 20 + 0.01).all()
    total_minutes = (expanded["estimated_block_hours"] * 60).sum()
    assert total_minutes == pytest.approx(60.0)


def test_real_window_sizing_never_leaves_a_zero_minute_phantom_part():
    # Session 26 bug fix, after a real reported gap: greedily sizing
    # each part against the largest remaining real window can exhaust
    # the task's total duration before all n_parts slots are used --
    # e.g. a 168.6-minute task split into parts sized against a section
    # with many diverse window sizes (46, 40, 34, 27, 22, 20, 18, 17,
    # 16 minutes) already sums past 168.6 after 5 parts, so the old
    # code's fixed "always emit n_parts rows" produced a 6th part worth
    # exactly 0 minutes -- real work for nobody, and a size nothing can
    # meaningfully combine or be scheduled against.
    tasks = pd.DataFrame([_task("T1", "A-B", 168.6 / 60.0, splittable=True)])
    cap = _capacity([
        {"section_id": "A-B", "date": "2026-09-07", "window_index": i, "window_minutes": m}
        for i, m in enumerate([46, 40, 34, 27, 22, 20, 18, 17, 16])
    ])
    expanded = expand_splittable_tasks(tasks, cap)

    assert (expanded["estimated_block_hours"] * 60.0 > 0.01).all()
    total_minutes = (expanded["estimated_block_hours"] * 60).sum()
    assert total_minutes == pytest.approx(168.6)
    # n_parts on every emitted row must match the ACTUAL row count, or
    # collapse_split_results would wrongly report a fully-scheduled task
    # as only "partial" (n_done < the old, inflated n_parts target).
    assert (expanded["n_parts"] == len(expanded)).all()


def test_non_splittable_task_is_never_split_even_if_it_would_help():
    tasks = pd.DataFrame([_task("T1", "A-B", 3.0, splittable=False)])
    cap = _capacity([{"section_id": "A-B", "date": "2026-09-07", "window_index": 0, "window_minutes": 100}])
    expanded = expand_splittable_tasks(tasks, cap)
    assert len(expanded) == 1
    assert expanded.iloc[0]["n_parts"] == 1
    assert expanded.iloc[0]["task_id"] == "T1"


def test_no_capacity_at_all_for_section_does_not_crash():
    tasks = pd.DataFrame([_task("T1", "Z-Z", 2.0, splittable=True)])
    cap = _capacity([])  # section not present at all
    expanded = expand_splittable_tasks(tasks, cap)
    assert len(expanded) == MAX_SPLIT_SESSIONS


def test_collapse_buckets_full_partial_and_unscheduled_correctly():
    original = pd.DataFrame(
        [
            _task("FULL1", "A-B", 1.0, splittable=False) | {"n_parts": 1},
            _task("SPLIT_FULL", "A-B", 3.0, splittable=True) | {"n_parts": 2},
            _task("SPLIT_PARTIAL", "A-B", 3.0, splittable=True) | {"n_parts": 2},
            _task("NONE", "A-B", 1.0, splittable=False) | {"n_parts": 1},
        ]
    )
    part_schedule = pd.DataFrame(
        [
            {"task_id": "FULL1", "parent_task_id": "FULL1", "n_parts": 1, "date": "2026-09-07", "window_index": 0, "on_time": True},
            {"task_id": "SPLIT_FULL__part1of2", "parent_task_id": "SPLIT_FULL", "n_parts": 2, "date": "2026-09-07", "window_index": 0, "on_time": True},
            {"task_id": "SPLIT_FULL__part2of2", "parent_task_id": "SPLIT_FULL", "n_parts": 2, "date": "2026-09-08", "window_index": 0, "on_time": True},
            {"task_id": "SPLIT_PARTIAL__part1of2", "parent_task_id": "SPLIT_PARTIAL", "n_parts": 2, "date": "2026-09-07", "window_index": 1, "on_time": True},
        ]
    )
    part_unscheduled = pd.DataFrame(columns=part_schedule.columns)

    full_df, partial_df, unscheduled_df = collapse_split_results(part_schedule, part_unscheduled, original)

    assert set(full_df["task_id"]) == {"FULL1", "SPLIT_FULL"}
    assert full_df.set_index("task_id").loc["FULL1", "option"] == "whole"
    assert full_df.set_index("task_id").loc["SPLIT_FULL", "option"] == "split"

    assert set(partial_df["task_id"]) == {"SPLIT_PARTIAL"}
    assert partial_df.iloc[0]["n_parts_scheduled"] == 1
    assert partial_df.iloc[0]["n_parts"] == 2

    assert set(unscheduled_df["task_id"]) == {"NONE"}
