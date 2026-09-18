import pandas as pd

from railblock.scheduling.orchestrator import _collapse_window_task_ids


def test_collapse_window_task_ids_strips_split_session_suffix():
    """Session 19: a combined window's task_ids came straight from the
    core solve over expand_splittable_tasks's EXPANDED task set, so it
    could contain a split-session id like "TDMS-00019__part2of2" that
    never exists as a real request task_id -- the frontend's combined-
    block details modal 404s on it. Must be mapped back to the real
    parent task_id."""
    windows = pd.DataFrame([
        {"section_id": "A-B", "date": "2026-09-07", "window_index": 0, "combined": True,
         "task_ids": ["TDMS-00019__part2of2", "SMMS-00017"]},
    ])
    out = _collapse_window_task_ids(windows)
    assert out.iloc[0]["task_ids"] == ["TDMS-00019", "SMMS-00017"]


def test_collapse_window_task_ids_dedupes_same_parent_two_sessions():
    """A window can combine two DIFFERENT sessions of the SAME split
    task with a different task -- that task must only be counted once."""
    windows = pd.DataFrame([
        {"section_id": "A-B", "date": "2026-09-07", "window_index": 0, "combined": True,
         "task_ids": ["TDMS-00019__part1of2", "TDMS-00019__part2of2", "SMMS-00017"]},
    ])
    out = _collapse_window_task_ids(windows)
    assert out.iloc[0]["task_ids"] == ["TDMS-00019", "SMMS-00017"]


def test_collapse_window_task_ids_leaves_non_split_ids_untouched():
    windows = pd.DataFrame([
        {"section_id": "A-B", "date": "2026-09-07", "window_index": 0, "combined": True,
         "task_ids": ["TDMS-00001", "SMMS-00002"]},
    ])
    out = _collapse_window_task_ids(windows)
    assert out.iloc[0]["task_ids"] == ["TDMS-00001", "SMMS-00002"]


def test_collapse_window_task_ids_handles_empty_dataframe():
    assert _collapse_window_task_ids(pd.DataFrame()).empty


def test_collapse_window_task_ids_handles_missing_column():
    windows = pd.DataFrame([{"section_id": "A-B"}])
    out = _collapse_window_task_ids(windows)
    assert "task_ids" not in out.columns
