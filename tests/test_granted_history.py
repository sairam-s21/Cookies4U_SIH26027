from datetime import datetime

import pandas as pd
import pytest

from railblock.synthetic.granted_history import classify_blocks_by_time, now_ist as real_now_ist


def _row(date, start_minute, end_minute):
    return {"task_id": "X", "section_id": "MAS-BBQ", "department": "Engineering", "date": date,
            "start_minute": start_minute, "end_minute": end_minute}


def test_classification_uses_real_ist_time_not_naive_server_local():
    """Session 18: this deployment's host OS clock runs UTC, but every
    block window in this project is implicitly IST wall-clock time -- a
    row whose window has already ended in IST (even if not yet ended in
    naive UTC 'now') must classify as completed, not active."""
    now_ist = real_now_ist()
    # a window that ended 10 minutes ago in real IST time
    ended_minute = now_ist.hour * 60 + now_ist.minute - 10
    df = pd.DataFrame([_row(now_ist.date().isoformat(), max(0, ended_minute - 60), max(0, ended_minute))])
    result = classify_blocks_by_time(df)
    assert len(result["completed"]) == 1
    assert len(result["active"]) == 0


def test_classification_active_window_contains_real_ist_now():
    now_ist = real_now_ist()
    minute = now_ist.hour * 60 + now_ist.minute
    df = pd.DataFrame([_row(now_ist.date().isoformat(), max(0, minute - 30), min(1439, minute + 30))])
    result = classify_blocks_by_time(df)
    assert len(result["active"]) == 1


def test_classify_blocks_by_time_handles_missing_columns_without_crashing():
    # e.g. live store.approved rows from before this project tracked
    # start_minute/end_minute -- a real crash caught in Session 17.
    df = pd.DataFrame([{"task_id": "X"}])
    result = classify_blocks_by_time(df)
    assert len(result["completed"]) == 0
    assert len(result["active"]) == 0
    assert len(result["upcoming"]) == 0


def test_classify_blocks_by_time_handles_empty_dataframe():
    result = classify_blocks_by_time(pd.DataFrame())
    assert all(len(v) == 0 for v in result.values())


def test_explicit_now_argument_still_respected():
    fixed_now = datetime(2026, 1, 1, 12, 0)
    df = pd.DataFrame([_row("2026-01-01", 600, 660)])  # 10:00-11:00, before fixed_now
    result = classify_blocks_by_time(df, now=fixed_now)
    assert len(result["completed"]) == 1
