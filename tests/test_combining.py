"""Session 26, at explicit user request: the pre-CP-SAT combining-
candidate detection -- same section, overlapping [raised_date, due_date]
windows, different departments."""

import pandas as pd

from railblock.scheduling.combining import (
    approved_rows_to_window_rows,
    combine_into_occupied_windows,
    combined_window_rows,
    find_combinable_pairs,
    force_combinable_placements,
    partner_minutes_by_task,
)


def _task(task_id, section_id, department, raised, due, hours=2.0):
    return {
        "task_id": task_id,
        "section_id": section_id,
        "department": department,
        "raised_date": raised,
        "due_date": due,
        "estimated_block_hours": hours,
    }


def _expanded(task_id, parent_task_id, section_id, department, raised, due, hours,
              n_parts=1, part_index=1, priority="Routine", whittle=1.0, approval="sm_direct"):
    return {
        "task_id": task_id,
        "parent_task_id": parent_task_id,
        "n_parts": n_parts,
        "part_index": part_index,
        "department": department,
        "section_id": section_id,
        "raised_date": raised,
        "due_date": due,
        "estimated_block_hours": hours,
        "requester_priority": priority,
        "whittle_index": whittle,
        "approval_path": approval,
    }


def _cap(rows):
    cols = ["section_id", "date", "window_index", "window_minutes", "start_minute", "end_minute"]
    return pd.DataFrame(rows, columns=cols)


def test_same_section_different_department_overlapping_dates_are_combinable():
    tasks = pd.DataFrame([
        _task("T1", "A-B", "Engineering", "2026-09-01", "2026-09-20"),
        _task("T2", "A-B", "Traction", "2026-09-10", "2026-09-30"),
    ])
    pairs = find_combinable_pairs(tasks)
    assert pairs == {"T1": ["T2"], "T2": ["T1"]}


def test_same_department_is_never_combinable_even_if_everything_else_matches():
    tasks = pd.DataFrame([
        _task("T1", "A-B", "Engineering", "2026-09-01", "2026-09-20"),
        _task("T2", "A-B", "Engineering", "2026-09-10", "2026-09-30"),
    ])
    assert find_combinable_pairs(tasks) == {}


def test_different_section_is_never_combinable():
    tasks = pd.DataFrame([
        _task("T1", "A-B", "Engineering", "2026-09-01", "2026-09-20"),
        _task("T2", "C-D", "Traction", "2026-09-01", "2026-09-20"),
    ])
    assert find_combinable_pairs(tasks) == {}


def test_non_overlapping_due_date_windows_are_never_combinable():
    tasks = pd.DataFrame([
        _task("T1", "A-B", "Engineering", "2026-09-01", "2026-09-05"),
        _task("T2", "A-B", "Traction", "2026-09-10", "2026-09-20"),
    ])
    assert find_combinable_pairs(tasks) == {}


def test_touching_windows_count_as_overlapping():
    # T1's window ends exactly the day T2's begins -- one real shared day.
    tasks = pd.DataFrame([
        _task("T1", "A-B", "Engineering", "2026-09-01", "2026-09-10"),
        _task("T2", "A-B", "Traction", "2026-09-10", "2026-09-20"),
    ])
    assert find_combinable_pairs(tasks) == {"T1": ["T2"], "T2": ["T1"]}


def test_a_task_can_have_multiple_combinable_partners():
    tasks = pd.DataFrame([
        _task("T1", "A-B", "Engineering", "2026-09-01", "2026-09-20"),
        _task("T2", "A-B", "Traction", "2026-09-05", "2026-09-15"),
        # T3's window (09-16 to 09-25) overlaps T1's but not T2's (T2 ends 09-15).
        _task("T3", "A-B", "Signalling", "2026-09-16", "2026-09-25"),
    ])
    pairs = find_combinable_pairs(tasks)
    assert set(pairs["T1"]) == {"T2", "T3"}
    assert pairs["T2"] == ["T1"]
    assert pairs["T3"] == ["T1"]


def test_partner_minutes_converts_hours_to_minutes():
    tasks = pd.DataFrame([
        _task("T1", "A-B", "Engineering", "2026-09-01", "2026-09-20", hours=1.5),
        _task("T2", "A-B", "Traction", "2026-09-05", "2026-09-15", hours=2.0),
    ])
    pairs = find_combinable_pairs(tasks)
    minutes = partner_minutes_by_task(tasks, pairs)
    assert minutes["T1"] == [120.0]  # T2's 2.0h
    assert minutes["T2"] == [90.0]  # T1's 1.5h


def test_no_pairs_means_no_partner_minutes():
    assert partner_minutes_by_task(pd.DataFrame(), {}) == {}


# -------------------------- force_combinable_placements --------------------------

def test_two_whole_tasks_that_both_fit_a_real_window_are_forced_together():
    expanded = pd.DataFrame([
        _expanded("ENG1", "ENG1", "A-B", "Engineering", "2026-09-01", "2026-09-20", hours=1.43),
        _expanded("TRAC1", "TRAC1", "A-B", "Traction", "2026-09-01", "2026-09-20", hours=1.0),
    ])
    capacity = _cap([
        {"section_id": "A-B", "date": "2026-09-17", "window_index": 0, "window_minutes": 120, "start_minute": 52, "end_minute": 172},
    ])
    pairs = find_combinable_pairs(expanded)
    combined, remaining, capacity_after = force_combinable_placements(expanded, capacity, pairs)

    assert set(combined["task_id"]) == {"ENG1", "TRAC1"}
    assert remaining.empty
    assert (combined["date"] == "2026-09-17").all()
    assert (combined["window_index"] == 0).all()
    # per-department "max not sum": window's remaining capacity drops by
    # the BUSIER department's minutes (85.8), not the combined total.
    assert capacity_after.iloc[0]["window_minutes"] == 120 - 85.8


def test_no_real_window_big_enough_for_both_forces_nothing():
    expanded = pd.DataFrame([
        _expanded("ENG1", "ENG1", "A-B", "Engineering", "2026-09-01", "2026-09-20", hours=1.43),
        _expanded("TRAC1", "TRAC1", "A-B", "Traction", "2026-09-01", "2026-09-20", hours=5.0),  # 300 min, too big
    ])
    capacity = _cap([
        {"section_id": "A-B", "date": "2026-09-17", "window_index": 0, "window_minutes": 120, "start_minute": 52, "end_minute": 172},
    ])
    pairs = find_combinable_pairs(expanded)
    combined, remaining, capacity_after = force_combinable_placements(expanded, capacity, pairs)

    assert combined.empty
    assert set(remaining["task_id"]) == {"ENG1", "TRAC1"}
    assert capacity_after.iloc[0]["window_minutes"] == 120


def test_window_outside_the_overlapping_due_date_range_is_never_used():
    expanded = pd.DataFrame([
        _expanded("ENG1", "ENG1", "A-B", "Engineering", "2026-09-01", "2026-09-10", hours=1.0),
        _expanded("TRAC1", "TRAC1", "A-B", "Traction", "2026-09-01", "2026-09-10", hours=1.0),
    ])
    # the only real window is AFTER both tasks' due date -- must not be used
    capacity = _cap([
        {"section_id": "A-B", "date": "2026-09-17", "window_index": 0, "window_minutes": 120, "start_minute": 52, "end_minute": 172},
    ])
    pairs = find_combinable_pairs(expanded)
    combined, remaining, capacity_after = force_combinable_placements(expanded, capacity, pairs)

    assert combined.empty
    assert set(remaining["task_id"]) == {"ENG1", "TRAC1"}


def test_at_most_one_task_per_department_ever_shares_a_window():
    # Session 26 rewrite, at explicit user request: a real window showed
    # 6 tasks "combined" under the old algorithm -- a joint disconnection
    # only ever makes sense as one crew per department. Two Engineering
    # tasks here (ENG1, ENG2) can NEVER both combine with the same
    # Traction task, even though ENG2 is tiny and would easily fit the
    # window's remaining room capacity-wise -- the cap is on department
    # COUNT (at most one Engineering task per window), not room.
    expanded = pd.DataFrame([
        _expanded("ENG1", "ENG1", "A-B", "Engineering", "2026-09-01", "2026-09-20", hours=1.43),
        _expanded("TRAC1", "TRAC1", "A-B", "Traction", "2026-09-01", "2026-09-20", hours=1.0),
        _expanded("ENG2", "ENG2", "A-B", "Engineering", "2026-09-01", "2026-09-20", hours=0.1),  # tiny -- easily fits, still excluded
    ])
    capacity = _cap([
        {"section_id": "A-B", "date": "2026-09-17", "window_index": 0, "window_minutes": 120, "start_minute": 52, "end_minute": 172},
    ])
    combined, remaining, capacity_after = force_combinable_placements(expanded, capacity)

    # Only one Engineering task can pair with Traction's one task --
    # Traction has no second task to pair ENG2 with, regardless of room.
    assert len(combined) == 2
    assert set(combined["department"]) == {"Engineering", "Traction"}
    assert set(remaining["task_id"]) == {"ENG2"}


def test_a_combined_block_never_exceeds_three_tasks():
    # The exact reported bug: with 3 departments each having several
    # tasks in one section, a window must never end up with more than
    # one task per department (so at most 3 total), no matter how many
    # tasks exist overall.
    expanded = pd.DataFrame(
        [_expanded(f"ENG{i}", f"ENG{i}", "A-B", "Engineering", "2026-09-01", "2026-09-20", hours=0.3) for i in range(1, 4)]
        + [_expanded(f"SIG{i}", f"SIG{i}", "A-B", "Signalling", "2026-09-01", "2026-09-20", hours=0.3) for i in range(1, 4)]
        + [_expanded(f"TRAC{i}", f"TRAC{i}", "A-B", "Traction", "2026-09-01", "2026-09-20", hours=0.3) for i in range(1, 4)]
    )
    capacity = _cap([
        {"section_id": "A-B", "date": "2026-09-17", "window_index": 0, "window_minutes": 120, "start_minute": 52, "end_minute": 172},
    ])
    combined, remaining, capacity_after = force_combinable_placements(expanded, capacity)

    if not combined.empty:
        counts = combined.groupby(["date", "window_index"]).size()
        assert (counts <= 3).all()
        dept_counts = combined.groupby(["date", "window_index"])["department"].apply(lambda s: s.duplicated().any())
        assert not dept_counts.any()  # never two tasks from the same department in one window


def test_three_departments_scarcest_bounds_triples_then_leftover_pairs():
    # Signalling=1 (scarcest), Engineering=2, Traction=3. Exactly one
    # 3-way combo should form (bounded by Signalling's single task);
    # after that, Engineering's 1 remaining task pairs with one of
    # Traction's 2 remaining tasks; Traction's last task has nothing
    # left to pair with and stays unforced.
    expanded = pd.DataFrame(
        [_expanded("SIG1", "SIG1", "A-B", "Signalling", "2026-09-01", "2026-09-20", hours=0.3)]
        + [_expanded(f"ENG{i}", f"ENG{i}", "A-B", "Engineering", "2026-09-01", "2026-09-20", hours=0.3) for i in range(1, 3)]
        + [_expanded(f"TRAC{i}", f"TRAC{i}", "A-B", "Traction", "2026-09-01", "2026-09-20", hours=0.3) for i in range(1, 4)]
    )
    capacity = _cap([
        {"section_id": "A-B", "date": d, "window_index": 0, "window_minutes": 120, "start_minute": 52, "end_minute": 172}
        for d in ("2026-09-16", "2026-09-17", "2026-09-18")
    ])
    combined, remaining, capacity_after = force_combinable_placements(expanded, capacity)

    assert len(combined) == 5  # one triple (3) + one pair (2)
    assert len(remaining) == 1  # exactly one Traction task left over
    assert remaining.iloc[0]["department"] == "Traction"


def test_scarcest_is_ranked_by_total_time_not_task_count_user_example():
    # The user's own exact example, at explicit request: "if total time
    # of engineering tasks is 5 hrs, total time of signaling tasks is 3
    # hrs, and total time of traction tasks is 2 hrs ... combine all 3
    # departments for 2 hrs, engineering and signaling departments for
    # remaining 1 hr and remaining 2 hrs of engineering department
    # should take place separately." Built from discrete 1h parts (this
    # function receives already-expanded parts, so continuous "hours"
    # become N one-hour rows) so the pooled, keep-draining cascade is
    # actually exercised across multiple rounds -- not just a single
    # fixed pairing, which is exactly the bug this rewrite fixes (the
    # scarce resource is a department's TOTAL remaining minutes, not how
    # many task rows happen to carry them).
    expanded = pd.DataFrame(
        [_expanded(f"TRAC{i}", f"TRAC{i}", "A-B", "Traction", "2026-09-01", "2026-09-20", hours=1.0) for i in range(1, 3)]
        + [_expanded(f"SIG{i}", f"SIG{i}", "A-B", "Signalling", "2026-09-01", "2026-09-20", hours=1.0) for i in range(1, 4)]
        + [_expanded(f"ENG{i}", f"ENG{i}", "A-B", "Engineering", "2026-09-01", "2026-09-20", hours=1.0) for i in range(1, 6)]
    )
    capacity = _cap([
        {"section_id": "A-B", "date": d, "window_index": 0, "window_minutes": 60, "start_minute": 0, "end_minute": 60}
        for d in ("2026-09-16", "2026-09-17", "2026-09-18", "2026-09-19")
    ])
    combined, remaining, capacity_after = force_combinable_placements(expanded, capacity)

    minutes_by_dept = combined.groupby("department")["estimated_block_hours"].sum() * 60.0
    assert minutes_by_dept.get("Traction", 0) == 120  # Traction's whole 2h -- it was the real bottleneck
    assert minutes_by_dept.get("Signalling", 0) == 180  # Signalling's whole 3h
    assert minutes_by_dept.get("Engineering", 0) == 180  # 2h (triple) + 1h (pair) of Engineering's 5h

    # Traction and Signalling are fully drained; Engineering's leftover
    # 2h (its two least-urgent tasks) is never forced -- normal solve.
    assert remaining.empty is False
    assert set(remaining["department"]) == {"Engineering"}
    assert remaining["estimated_block_hours"].sum() == 2.0

    # never more than one task per department in any single shared window
    dept_dupes = combined.groupby(["date", "window_index"])["department"].apply(lambda s: s.duplicated().any())
    assert not dept_dupes.any()


def test_empty_pairs_or_empty_tasks_is_a_safe_no_op():
    expanded = pd.DataFrame([_expanded("ENG1", "ENG1", "A-B", "Engineering", "2026-09-01", "2026-09-20", hours=1.0)])
    capacity = _cap([{"section_id": "A-B", "date": "2026-09-17", "window_index": 0, "window_minutes": 120, "start_minute": 52, "end_minute": 172}])
    combined, remaining, capacity_after = force_combinable_placements(expanded, capacity, {})
    assert combined.empty
    assert len(remaining) == 1
    assert capacity_after.iloc[0]["window_minutes"] == 120


def test_combined_window_rows_marks_combined_true_and_lists_both_departments():
    expanded = pd.DataFrame([
        _expanded("ENG1", "ENG1", "A-B", "Engineering", "2026-09-01", "2026-09-20", hours=1.43),
        _expanded("TRAC1", "TRAC1", "A-B", "Traction", "2026-09-01", "2026-09-20", hours=1.0),
    ])
    capacity = _cap([
        {"section_id": "A-B", "date": "2026-09-17", "window_index": 0, "window_minutes": 120, "start_minute": 52, "end_minute": 172},
    ])
    pairs = find_combinable_pairs(expanded)
    combined, _, _ = force_combinable_placements(expanded, capacity, pairs)
    windows = combined_window_rows(combined, capacity)

    assert len(windows) == 1
    row = windows.iloc[0]
    assert bool(row["combined"]) is True
    assert row["departments"] == "Engineering,Traction"
    assert set(row["task_ids"]) == {"ENG1", "TRAC1"}
    assert row["minutes_used"] == 86  # busiest department (Engineering, 85.8 rounded)
    assert row["minutes_capacity"] == 120


def test_combined_window_rows_collapses_split_part_ids_to_parent_task_id():
    # Session 26 bug fix, after a real reported "No details found" bug:
    # a split task's part rows carry a PART-level task_id (e.g.
    # "ENG1__part1of2"), which never exists as a real request task_id --
    # the frontend's per-id GET /requests lookup 404s on it. task_ids
    # must report the real parent_task_id instead.
    combined = pd.DataFrame([
        _expanded("ENG1__part1of2", "ENG1", "A-B", "Engineering", "2026-09-01", "2026-09-20", hours=0.5),
        _expanded("TRAC1", "TRAC1", "A-B", "Traction", "2026-09-01", "2026-09-20", hours=1.0),
    ])
    combined["date"] = "2026-09-17"
    combined["window_index"] = 0
    combined["start_minute"] = 52
    combined["end_minute"] = 172
    combined["on_time"] = True
    combined["completion_verified"] = False
    capacity = _cap([
        {"section_id": "A-B", "date": "2026-09-17", "window_index": 0, "window_minutes": 120, "start_minute": 52, "end_minute": 172},
    ])
    windows = combined_window_rows(combined, capacity)

    assert set(windows.iloc[0]["task_ids"]) == {"ENG1", "TRAC1"}


# -------------------- approved_rows_to_window_rows --------------------

def _approved(task_id, section_id, department, hours, date, window_index, start_minute=0, end_minute=60):
    return {
        "task_id": task_id,
        "department": department,
        "section_id": section_id,
        "estimated_block_hours": hours,
        "option": "whole",
        "sessions": [(date, window_index, start_minute, end_minute)],
        "date": date,
        "on_time": True,
        "negotiated_exception": False,
        "approval_path": "sm_direct",
        "completion_verified": False,
    }


def _approved_negotiated(task_id, section_id, department, hours, date):
    return {
        "task_id": task_id,
        "department": department,
        "section_id": section_id,
        "estimated_block_hours": hours,
        "option": "negotiated",
        "date": date,
        "on_time": True,
        "negotiated_exception": True,
        "approval_path": "sm_direct",
        "completion_verified": False,
    }


def test_approved_rows_to_window_rows_builds_one_row_per_window_key():
    approved = [_approved("ENG1", "A-B", "Engineering", 1.0, "2026-09-17", 0, start_minute=52, end_minute=112)]
    capacity = _cap([{"section_id": "A-B", "date": "2026-09-17", "window_index": 0, "window_minutes": 120, "start_minute": 52, "end_minute": 172}])
    windows = approved_rows_to_window_rows(approved, capacity)

    assert len(windows) == 1
    row = windows.iloc[0]
    assert row["departments"] == "Engineering"
    assert row["task_ids"] == ["ENG1"]
    assert row["minutes_used"] == 60
    assert row["minutes_capacity"] == 120
    assert bool(row["combined"]) is False


def test_approved_rows_to_window_rows_skips_negotiated_exceptions():
    # Session 26: a negotiated exception (a shifted real train, not a
    # discrete window) never carries a window_index at all -- there is
    # no shared slot for anyone else to be offered here.
    approved = [_approved_negotiated("ENG1", "A-B", "Engineering", 1.0, "2026-09-17")]
    capacity = _cap([{"section_id": "A-B", "date": "2026-09-17", "window_index": 0, "window_minutes": 120, "start_minute": 52, "end_minute": 172}])
    windows = approved_rows_to_window_rows(approved, capacity)
    assert windows.empty


def test_approved_rows_to_window_rows_skips_keys_outside_current_capacity():
    # Approved on a date this run's own capacity table doesn't even offer
    # -- nothing to double-book, so it's dropped, not an error.
    approved = [_approved("ENG1", "A-B", "Engineering", 1.0, "2026-08-01", 0)]
    capacity = _cap([{"section_id": "A-B", "date": "2026-09-17", "window_index": 0, "window_minutes": 120, "start_minute": 52, "end_minute": 172}])
    windows = approved_rows_to_window_rows(approved, capacity)
    assert windows.empty


# -------------------- combine_into_occupied_windows --------------------

def _ranked(task_id, section_id, department, raised, due, hours, priority="Routine", whittle=1.0, approval="sm_direct"):
    return {
        "task_id": task_id,
        "section_id": section_id,
        "department": department,
        "raised_date": raised,
        "due_date": due,
        "estimated_block_hours": hours,
        "requester_priority": priority,
        "whittle_index": whittle,
        "approval_path": approval,
    }


def test_combine_into_occupied_windows_adds_a_new_department_into_spare_room():
    # The user's own reported gap: an Engineering task already approved
    # in an EARLIER scheduling run occupies a real window -- a NEW
    # Traction task, in THIS run, should still be offered that same
    # window's spare "max not sum" room, not just whatever's left open
    # elsewhere.
    cap_rows = [{"section_id": "A-B", "date": "2026-09-17", "window_index": 0, "window_minutes": 120, "start_minute": 52, "end_minute": 172}]
    capacity = _cap(cap_rows)
    occupied = approved_rows_to_window_rows(
        [_approved("ENG1", "A-B", "Engineering", 1.0, "2026-09-17", 0, start_minute=52, end_minute=172)], capacity
    )
    ranked = pd.DataFrame([_ranked("TRAC1", "A-B", "Traction", "2026-09-01", "2026-09-20", 1.5)])

    remaining, forced_rows, updated = combine_into_occupied_windows(occupied, ranked, capacity)

    assert remaining.empty
    assert len(forced_rows) == 1
    assert forced_rows[0]["task_id"] == "TRAC1"
    assert forced_rows[0]["option"] == "combined"
    assert forced_rows[0]["sessions"] == [("2026-09-17", 0, 52, 172)]
    row = updated.iloc[0]
    assert row["departments"] == "Engineering,Traction"
    assert set(row["task_ids"]) == {"ENG1", "TRAC1"}
    assert bool(row["combined"]) is True
    assert row["minutes_used"] == 90  # Traction's 1.5h (90) > Engineering's 60 -- the new running max


def test_combine_into_occupied_windows_never_exceeds_three_departments():
    occupied = pd.DataFrame([{
        "section_id": "A-B", "date": "2026-09-17", "window_index": 0,
        "start_minute": 0, "end_minute": 120, "minutes_used": 60,
        "minutes_capacity": 120, "departments": "Engineering,Signalling",
        "task_ids": ["ENG1", "SIG1"], "combined": True, "required_signoffs": [],
    }])
    capacity = _cap([{"section_id": "A-B", "date": "2026-09-17", "window_index": 0, "window_minutes": 120, "start_minute": 0, "end_minute": 120}])
    ranked = pd.DataFrame([
        _ranked("TRAC1", "A-B", "Traction", "2026-09-01", "2026-09-20", 1.0),
        _ranked("TRAC2", "A-B", "Traction", "2026-09-01", "2026-09-20", 1.0),  # would also fit, but the window is full at 3
    ])

    remaining, forced_rows, updated = combine_into_occupied_windows(occupied, ranked, capacity)

    assert len(forced_rows) == 1
    assert set(remaining["task_id"]) == {"TRAC2"}
    row = updated.iloc[0]
    assert len(row["task_ids"]) == 3
    assert row["departments"] == "Engineering,Signalling,Traction"


def test_combine_into_occupied_windows_respects_due_date():
    cap_rows = [{"section_id": "A-B", "date": "2026-09-17", "window_index": 0, "window_minutes": 120, "start_minute": 0, "end_minute": 60}]
    capacity = _cap(cap_rows)
    occupied = approved_rows_to_window_rows([_approved("ENG1", "A-B", "Engineering", 1.0, "2026-09-17", 0)], capacity)
    ranked = pd.DataFrame([_ranked("TRAC1", "A-B", "Traction", "2026-09-20", "2026-09-25", 1.0)])  # not raised yet on 09-17

    remaining, forced_rows, updated = combine_into_occupied_windows(occupied, ranked, capacity)

    assert forced_rows == []
    assert len(remaining) == 1


def test_combine_into_occupied_windows_never_forces_a_task_too_big_for_the_window():
    cap_rows = [{"section_id": "A-B", "date": "2026-09-17", "window_index": 0, "window_minutes": 60, "start_minute": 0, "end_minute": 60}]
    capacity = _cap(cap_rows)
    occupied = approved_rows_to_window_rows([_approved("ENG1", "A-B", "Engineering", 1.0, "2026-09-17", 0)], capacity)
    ranked = pd.DataFrame([_ranked("TRAC1", "A-B", "Traction", "2026-09-01", "2026-09-20", 5.0)])  # 300 min, way too big

    remaining, forced_rows, updated = combine_into_occupied_windows(occupied, ranked, capacity)

    assert forced_rows == []
    assert len(remaining) == 1


def test_combine_into_occupied_windows_is_a_safe_no_op_on_empty_inputs():
    empty = pd.DataFrame()
    assert combine_into_occupied_windows(empty, empty, empty)[0].empty
    ranked = pd.DataFrame([_ranked("TRAC1", "A-B", "Traction", "2026-09-01", "2026-09-20", 1.0)])
    remaining, forced_rows, updated = combine_into_occupied_windows(empty, ranked, empty)
    assert len(remaining) == 1
    assert forced_rows == []
