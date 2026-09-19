from datetime import date as Date, datetime

import numpy as np
import pandas as pd
import pytest

from railblock.api.store import get_corridor_context
from railblock.paths import TRAIN_DETAILS_CSV
from railblock.scheduling.emergency import (
    EMERGENCY_DEFECT_TYPES,
    EMERGENCY_MAX_HOURS,
    EMERGENCY_MIN_HOURS,
    _clip_capacity_to_now,
    build_emergency_segments,
    create_demo_emergency,
    find_affected_rows,
    propose_reschedule,
)

needs_real_dataset = pytest.mark.skipif(not TRAIN_DETAILS_CSV.exists(), reason="real dataset not present")


@pytest.fixture
def toy_sections():
    return pd.DataFrame(
        [
            {"section_id": "MAS-AJJ", "length_km": 68.0},
            {"section_id": "AJJ-WJR", "length_km": 36.0},
            {"section_id": "BQI-SA", "length_km": 43.0},
        ]
    )


def _row(task_id, section_id, date, start_minute, end_minute):
    return {"task_id": task_id, "section_id": section_id, "date": date, "start_minute": start_minute, "end_minute": end_minute}


# ---------------------------------------------------------- segment splitting

def test_build_emergency_segments_within_one_day():
    segments = build_emergency_segments(datetime(2026, 9, 17, 10, 0), 4.0)
    assert segments == [{"date": "2026-09-17", "start_minute": 600, "end_minute": 840}]


def test_build_emergency_segments_crosses_midnight():
    # 22:00 + 4h = 02:00 the next day -- real minutes 1320-1440 today, 0-120 tomorrow.
    segments = build_emergency_segments(datetime(2026, 9, 17, 22, 0), 4.0)
    assert segments == [
        {"date": "2026-09-17", "start_minute": 1320, "end_minute": 1440},
        {"date": "2026-09-18", "start_minute": 0, "end_minute": 120},
    ]


def test_build_emergency_segments_total_duration_is_preserved():
    for start_hour in (0, 6, 18, 23):
        for hours in (EMERGENCY_MIN_HOURS, 7.5, EMERGENCY_MAX_HOURS):
            segments = build_emergency_segments(datetime(2026, 9, 17, start_hour, 0), hours)
            total = sum(s["end_minute"] - s["start_minute"] for s in segments)
            assert total == round(hours * 60)
            # every segment's window is a real, non-degenerate one within a single day
            for s in segments:
                assert 0 <= s["start_minute"] < s["end_minute"] <= 1440


# -------------------------------------------------------------- affected rows

def test_find_affected_rows_matches_overlap_regardless_of_section():
    emergency = {"segments": [{"date": "2026-09-17", "start_minute": 600, "end_minute": 900}]}
    now = datetime(2026, 9, 17, 5, 0)
    existing = [
        _row("SAME-SECTION-OVERLAP", "A-B", "2026-09-17", 700, 800),  # overlaps
        _row("OTHER-SECTION-OVERLAP", "C-D", "2026-09-17", 850, 950),  # overlaps a different section
        _row("NO-OVERLAP-SAME-DAY", "A-B", "2026-09-17", 0, 599),  # ends before the emergency starts
        _row("WRONG-DATE", "A-B", "2026-09-18", 700, 800),  # right minutes, wrong date
    ]
    affected_ids = {r["task_id"] for r in find_affected_rows(emergency, existing, now)}
    assert affected_ids == {"SAME-SECTION-OVERLAP", "OTHER-SECTION-OVERLAP"}


def test_find_affected_rows_excludes_already_completed():
    emergency = {"segments": [{"date": "2026-09-17", "start_minute": 600, "end_minute": 900}]}
    now = datetime(2026, 9, 17, 20, 0)  # after this row's own end
    existing = [_row("ALREADY-DONE", "A-B", "2026-09-17", 700, 800)]
    assert find_affected_rows(emergency, existing, now) == []


def test_find_affected_rows_dedupes_a_split_tasks_multiple_overlapping_sessions():
    # A real reported bug: existing_rows is at SESSION granularity, so a
    # split task with two sessions that both overlap the emergency must
    # still only appear ONCE -- otherwise the same task_id feeds
    # rank_tasks/expand_splittable_tasks as two separate rows, silently
    # corrupting the part-level CP-SAT expansion downstream.
    emergency = {"segments": [{"date": "2026-09-17", "start_minute": 600, "end_minute": 900}]}
    now = datetime(2026, 9, 17, 5, 0)
    existing = [
        _row("SPLIT-TASK", "A-B", "2026-09-17", 650, 700),
        _row("SPLIT-TASK", "A-B", "2026-09-17", 750, 800),
    ]
    affected = find_affected_rows(emergency, existing, now)
    assert len(affected) == 1
    assert affected[0]["task_id"] == "SPLIT-TASK"


def test_find_affected_rows_ignores_rows_with_no_modeled_window():
    emergency = {"segments": [{"date": "2026-09-17", "start_minute": 600, "end_minute": 900}]}
    now = datetime(2026, 9, 17, 5, 0)
    existing = [_row("NEGOTIATED-STYLE", "A-B", "2026-09-17", None, None)]
    assert find_affected_rows(emergency, existing, now) == []


# ----------------------------------------------------------- create emergency

def test_create_demo_emergency_fields_are_valid(toy_sections):
    rng = np.random.default_rng(1)
    now = datetime(2026, 9, 17, 10, 0)
    emergency = create_demo_emergency(toy_sections, [], now, rng, "EMRG-00001")

    assert emergency["task_id"] == "EMRG-00001"
    assert emergency["section_id"] in set(toy_sections["section_id"])
    assert emergency["department"] in EMERGENCY_DEFECT_TYPES
    assert emergency["defect_type"] in EMERGENCY_DEFECT_TYPES[emergency["department"]]
    assert EMERGENCY_MIN_HOURS <= emergency["estimated_block_hours"] <= EMERGENCY_MAX_HOURS
    assert emergency["segments"]
    assert emergency["is_emergency"] is True
    assert emergency["status"] == "proposed"


def test_create_demo_emergency_always_overlaps_a_real_upcoming_task(toy_sections):
    # "always create an emergency situation such that one or more blocks
    # are affected" -- create_demo_emergency now anchors its window to a
    # real still-upcoming task's own start time (the soonest one), not a
    # blind "starts exactly now" draw left to chance. Verified here the
    # way it actually matters: find_affected_rows must genuinely catch
    # it, for every seed (the RNG only controls duration/department/
    # defect_type now, never whether a real conflict exists). Timed
    # within EMERGENCY_MAX_HOURS (4h) of `now` -- Session 39 tightened
    # the search horizon to this emergency's own real reachable range.
    now = datetime(2026, 9, 17, 6, 0)
    existing = [_row("SOON-TASK", "AJJ-WJR", "2026-09-17", 500, 550)]
    for seed in range(8):
        rng = np.random.default_rng(seed)
        emergency = create_demo_emergency(toy_sections, existing, now, rng, f"EMRG-{seed:05d}")
        assert emergency["section_id"] == "AJJ-WJR"
        affected_ids = {r["task_id"] for r in find_affected_rows(emergency, existing, now)}
        assert affected_ids == {"SOON-TASK"}


def test_create_demo_emergency_never_anchors_before_now(toy_sections):
    # A real task that's already IN PROGRESS (started before `now`, ends
    # after it) must never push the emergency's own reported start
    # earlier than the real moment the button was clicked.
    now = datetime(2026, 9, 17, 10, 0)
    existing = [_row("IN-PROGRESS", "AJJ-WJR", "2026-09-17", 500, 700)]
    rng = np.random.default_rng(3)
    emergency = create_demo_emergency(toy_sections, existing, now, rng, "EMRG-00001")
    first_seg = emergency["segments"][0]
    assert first_seg["date"] == "2026-09-17"
    assert first_seg["start_minute"] == 600  # now (10:00) -- not the task's own earlier 500


def test_create_demo_emergency_picks_the_soonest_of_several_upcoming_tasks(toy_sections):
    # Both candidates within EMERGENCY_MAX_HOURS (4h) of `now` (minute
    # 360) -- Session 39 tightened the search horizon to this
    # emergency's own real reachable range, see the function's docstring.
    now = datetime(2026, 9, 17, 6, 0)
    existing = [
        _row("LATER-TASK", "BQI-SA", "2026-09-17", 550, 590),
        _row("SOONEST-TASK", "MAS-AJJ", "2026-09-17", 450, 500),
    ]
    rng = np.random.default_rng(2)
    emergency = create_demo_emergency(toy_sections, existing, now, rng, "EMRG-00001")
    assert emergency["section_id"] == "MAS-AJJ"
    assert emergency["segments"][0]["start_minute"] == 450


def test_create_demo_emergency_never_anchors_to_a_future_day(toy_sections):
    # A real reported bug: with nothing left scheduled TODAY but
    # something scheduled tomorrow, the emergency anchored to tomorrow
    # instead -- "an emergency reported now" showing up as tomorrow's
    # incident. Must always stay on `now`'s own real date, ignoring any
    # real task on a later day entirely (even though it's technically
    # "still upcoming").
    now = datetime(2026, 9, 17, 20, 0)  # late in the day, nothing left today
    existing = [_row("TOMORROW-TASK", "AJJ-WJR", "2026-09-18", 100, 150)]
    rng = np.random.default_rng(5)
    emergency = create_demo_emergency(toy_sections, existing, now, rng, "EMRG-00001")
    assert emergency["segments"][0]["date"] == "2026-09-17"
    assert emergency["segments"][0]["start_minute"] == 1200  # falls back to `now` itself


def test_create_demo_emergency_reaches_past_midnight_when_todays_schedule_is_empty(toy_sections):
    # Session 39: the real gap this closes -- confirmed live against this
    # corridor's own schedule that once EMERGENCY_MAX_HOURS dropped to 4h,
    # late in the day (nothing left scheduled for the REST of today) used
    # to fall through to the no-guarantee random-section path even
    # though something was genuinely about to start just after midnight,
    # well within this emergency's own real reach. The search horizon is
    # now real elapsed time, not calendar-day boundaries, so it correctly
    # spans midnight here.
    now = datetime(2026, 9, 17, 23, 0)  # nothing left today; EMERGENCY_MAX_HOURS reaches to 03:00 tomorrow
    existing = [_row("JUST-AFTER-MIDNIGHT", "AJJ-WJR", "2026-09-18", 90, 140)]  # 01:30-02:20 tomorrow
    rng = np.random.default_rng(9)
    emergency = create_demo_emergency(toy_sections, existing, now, rng, "EMRG-00001")
    assert emergency["section_id"] == "AJJ-WJR"
    affected_ids = {r["task_id"] for r in find_affected_rows(emergency, existing, now)}
    assert affected_ids == {"JUST-AFTER-MIDNIGHT"}


def test_create_demo_emergency_is_fresh_every_call(toy_sections):
    # "each time a user clicks 'create emergency situation' it should
    # create a new emergency situation" -- verified here at the pure
    # function level: two calls with different task_ids never collapse
    # into the same emergency, even with the same rng state re-seeded.
    now = datetime(2026, 9, 17, 10, 0)
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    e1 = create_demo_emergency(toy_sections, [], now, rng1, "EMRG-00001")
    e2 = create_demo_emergency(toy_sections, [], now, rng2, "EMRG-00002")
    assert e1["task_id"] != e2["task_id"]


# ------------------------------------------------------------ reschedule

def _cap(rows):
    cols = ["section_id", "date", "window_index", "window_minutes", "start_minute", "end_minute"]
    return pd.DataFrame(rows, columns=cols)


def test_clip_capacity_to_now_drops_windows_that_already_ended():
    capacity = _cap([{"section_id": "A-B", "date": "2026-09-17", "window_index": 0, "window_minutes": 60, "start_minute": 400, "end_minute": 460}])
    clipped = _clip_capacity_to_now(capacity, datetime(2026, 9, 17, 8, 0))  # 08:00 = minute 480
    assert clipped.empty


def test_clip_capacity_to_now_trims_a_window_spanning_the_current_moment():
    # "the affected blocks must be rescheduled afterwards only" -- a real
    # window from 07:00-09:00 must not still offer 07:00-08:00 (already
    # passed) once it's really 08:00.
    capacity = _cap([{"section_id": "A-B", "date": "2026-09-17", "window_index": 0, "window_minutes": 120, "start_minute": 420, "end_minute": 540}])
    clipped = _clip_capacity_to_now(capacity, datetime(2026, 9, 17, 8, 0))  # minute 480
    assert len(clipped) == 1
    row = clipped.iloc[0]
    assert row["start_minute"] == 480
    assert row["end_minute"] == 540
    assert row["window_minutes"] == 60


def test_clip_capacity_to_now_leaves_other_dates_untouched():
    capacity = _cap([
        {"section_id": "A-B", "date": "2026-09-17", "window_index": 0, "window_minutes": 60, "start_minute": 0, "end_minute": 60},
        {"section_id": "A-B", "date": "2026-09-18", "window_index": 0, "window_minutes": 60, "start_minute": 0, "end_minute": 60},
    ])
    clipped = _clip_capacity_to_now(capacity, datetime(2026, 9, 17, 8, 0))
    # today's already-ended window is dropped; tomorrow's is untouched
    assert len(clipped) == 1
    assert clipped.iloc[0]["date"] == "2026-09-18"
    assert clipped.iloc[0]["start_minute"] == 0


@needs_real_dataset
def test_propose_reschedule_marks_a_genuinely_shared_window_as_combined():
    """A real reported bug: two different-department affected tasks that
    the CP-SAT solve organically placed into the exact same real window
    (its normal, intended combining behavior -- see model.py's
    COORDINATION_BONUS) both came back with combined=False, because the
    old code checked the schedule row's own `option` field (only ever
    "combined" for the separate PRE-solve forcing passes) instead of the
    real windows table's own `combined` flag -- the same thing
    RecommendedScheduling.jsx's buildBlocksBySection already gets right."""
    ctx = get_corridor_context()
    section_id = ctx.sections["section_id"].iloc[0]
    affected = [
        {
            "task_id": "A1", "section_id": section_id, "department": "Engineering",
            "requester_priority": "Moderate", "estimated_block_hours": 1.0,
            "defect_type": "Fishplate/joint looseness", "raised_date": "2026-09-10", "due_date": "2026-10-10",
            "days_overdue": 0, "splittable": True, "approval_path": "sm_direct",
            "date": "2026-09-18", "start_minute": 500, "end_minute": 560,
        },
        {
            "task_id": "A2", "section_id": section_id, "department": "Traction",
            "requester_priority": "Moderate", "estimated_block_hours": 1.0,
            "defect_type": "Jumper wire fault", "raised_date": "2026-09-10", "due_date": "2026-10-10",
            "days_overdue": 0, "splittable": True, "approval_path": "sm_direct",
            "date": "2026-09-18", "start_minute": 600, "end_minute": 660,
        },
    ]
    fake_emergency = {"task_id": "EMRG-TEST", "section_id": section_id, "segments": []}
    proposals = propose_reschedule(
        affected, ctx.sections, ctx.passenger_occupancy, fake_emergency, [], Date(2026, 9, 18), datetime(2026, 9, 18, 0, 0),
        time_limit_s=15,
    )
    by_id = {p.task_id: p for p in proposals}
    assert by_id["A1"].proposed is not None
    assert by_id["A2"].proposed is not None
    # they must genuinely share the same real window to assert "combined"
    assert (by_id["A1"].proposed["date"], by_id["A1"].proposed["window_index"]) == (
        by_id["A2"].proposed["date"], by_id["A2"].proposed["window_index"],
    )
    assert by_id["A1"].proposed["combined"] is True
    assert by_id["A2"].proposed["combined"] is True


@needs_real_dataset
def test_propose_reschedule_widens_horizon_until_a_slot_is_found():
    """"how can two tasks have no alternative slot found? ... first keep
    horizon to 1 day, if some tasks are not scheduled, then increment the
    horizon ... repeat till all tasks are scheduled." Blocks the ENTIRE
    first day on this section via the emergency's own segment (a real,
    deterministic way to force day 1 to have zero free capacity there,
    without needing to fight the real corridor's own occupancy) --
    the widening loop must reach a later day to find a real slot."""
    ctx = get_corridor_context()
    section_id = ctx.sections["section_id"].iloc[0]
    fake_emergency = {
        "task_id": "EMRG-TEST-WIDE", "section_id": section_id,
        "segments": [{"date": "2026-09-18", "start_minute": 0, "end_minute": 1440}],
    }
    affected = [{
        "task_id": "A1", "section_id": section_id, "department": "Engineering",
        "requester_priority": "Moderate", "estimated_block_hours": 1.0,
        "defect_type": "Fishplate/joint looseness", "raised_date": "2026-09-10", "due_date": "2026-10-10",
        "days_overdue": 0, "splittable": True, "approval_path": "sm_direct",
        "date": "2026-09-18", "start_minute": 500, "end_minute": 560,
    }]
    proposals = propose_reschedule(
        affected, ctx.sections, ctx.passenger_occupancy, fake_emergency, [], Date(2026, 9, 18), datetime(2026, 9, 18, 0, 0),
        time_limit_s=15,
    )
    assert proposals[0].proposed is not None
    assert proposals[0].proposed["date"] != "2026-09-18"  # couldn't be today -- had to widen


@needs_real_dataset
def test_propose_reschedule_gives_up_at_max_n_days_without_hanging():
    """A task that's genuinely unplaceable within the given cap must
    still return proposed=None (not loop forever) -- the safety bound
    behind "no limit for the horizon"."""
    ctx = get_corridor_context()
    section_id = ctx.sections["section_id"].iloc[0]
    fake_emergency = {
        "task_id": "EMRG-TEST-CAP", "section_id": section_id,
        "segments": [
            {"date": "2026-09-18", "start_minute": 0, "end_minute": 1440},
            {"date": "2026-09-19", "start_minute": 0, "end_minute": 1440},
            {"date": "2026-09-20", "start_minute": 0, "end_minute": 1440},
        ],
    }
    affected = [{
        "task_id": "A1", "section_id": section_id, "department": "Engineering",
        "requester_priority": "Moderate", "estimated_block_hours": 1.0,
        "defect_type": "Fishplate/joint looseness", "raised_date": "2026-09-10", "due_date": "2026-10-10",
        "days_overdue": 0, "splittable": True, "approval_path": "sm_direct",
        "date": "2026-09-18", "start_minute": 500, "end_minute": 560,
    }]
    proposals = propose_reschedule(
        affected, ctx.sections, ctx.passenger_occupancy, fake_emergency, [], Date(2026, 9, 18), datetime(2026, 9, 18, 0, 0),
        time_limit_s=5, max_n_days=2,
    )
    assert proposals[0].proposed is None
