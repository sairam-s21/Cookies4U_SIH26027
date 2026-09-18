"""Session 13: PostgreSQL-backed BlockRequestStore persists correctly and
is visible across separate connections to the same database (the actual
point of a real server-backed store, replacing Session 8's SQLite file).

Every test gets a freshly-truncated `railblock_test` database (see
_test_store()) so tests never touch real demo data and never leak state
between each other, the same guarantee SQLite's ":memory:" gave for free.

Session 32, at explicit user request: every mutable-state method now takes
a `session_id` -- these tests use a single fixed SESSION for most cases
(unrelated to what's under test) plus a dedicated pair (SESSION_A/
SESSION_B) for the isolation tests that are actually about session_id
itself.
"""

from datetime import date, timedelta

import pandas as pd

from railblock.api.store import BlockRequestStore, DEFAULT_SESSION_ID, _test_database_url
from railblock.scheduling.orchestrator import FullScheduleResult
from railblock.synthetic.granted_history import now_ist

SESSION = DEFAULT_SESSION_ID
SESSION_A = "session-a"
SESSION_B = "session-b"

SAMPLE_REQUEST = {
    "task_id": "SMMS-REQ-00001",
    "department": "Signalling",
    "section_id": "MAS-BBQ",
    "defect_type": "Signal lamp failure",
    "requester_priority": "Critical",
    "estimated_block_hours": 0.75,
    "due_date": "2026-12-31",
    "days_overdue": 0,
    "splittable": False,
}


def _test_store() -> BlockRequestStore:
    store = BlockRequestStore(dsn=_test_database_url())
    store.reset()
    return store


def test_add_and_get_request():
    store = _test_store()
    store.add_request(SAMPLE_REQUEST, SESSION)
    row = store.get_request("SMMS-REQ-00001", SESSION)
    assert row["status"] == "pending"
    assert row["department"] == "Signalling"


def test_pending_and_all_requests_df():
    store = _test_store()
    store.add_request(SAMPLE_REQUEST, SESSION)
    other = dict(SAMPLE_REQUEST, task_id="SMMS-REQ-00002")
    store.add_request(other, SESSION)
    store.update_request_status("SMMS-REQ-00002", "approved", SESSION)

    assert len(store.all_requests_df(SESSION)) == 2
    pending = store.pending_requests_df(SESSION)
    assert len(pending) == 1
    assert pending.iloc[0]["task_id"] == "SMMS-REQ-00001"


def test_non_approved_requests_df_includes_every_unapproved_status():
    store = _test_store()
    for i, status in enumerate(["pending", "recommended_full", "recommended_partial", "recommended_unscheduled", "approved"]):
        row = dict(SAMPLE_REQUEST, task_id=f"SMMS-REQ-{i:05d}")
        store.add_request(row, SESSION)
        store.update_request_status(row["task_id"], status, SESSION)

    non_approved = store.non_approved_requests_df(SESSION)
    assert len(non_approved) == 4
    assert "approved" not in set(non_approved["status"])
    assert {"pending", "recommended_full", "recommended_partial", "recommended_unscheduled"} == set(non_approved["status"])


def test_update_request_status_persists():
    store = _test_store()
    store.add_request(SAMPLE_REQUEST, SESSION)
    store.update_request_status("SMMS-REQ-00001", "recommended_full", SESSION)
    assert store.get_request("SMMS-REQ-00001", SESSION)["status"] == "recommended_full"


def test_approved_roundtrip():
    store = _test_store()
    store.append_approved([dict(SAMPLE_REQUEST, task_id="SMMS-REQ-00001")], SESSION)
    approved = store.get_approved(SESSION)
    assert len(approved) == 1
    assert approved[0]["task_id"] == "SMMS-REQ-00001"


def test_recommendation_roundtrip():
    store = _test_store()
    schedule = pd.DataFrame([{"task_id": "SMMS-REQ-00001", "date": "2026-09-07", "option": "whole"}])
    empty = pd.DataFrame()
    result = FullScheduleResult(
        schedule=schedule, partial=empty, unscheduled=empty, windows=empty,
        status="OPTIMAL", solve_time_s=1.2, objective_value=5.0, option_counts={"whole": 1},
    )
    tasks_df = pd.DataFrame([SAMPLE_REQUEST])
    store.set_last_recommendation(result, tasks_df, {"start_date": "2026-09-07", "n_days": 7}, empty, SESSION)

    assert store.get_last_recommendation(SESSION).status == "OPTIMAL"
    assert store.get_last_recommendation(SESSION).schedule.iloc[0]["task_id"] == "SMMS-REQ-00001"
    assert store.get_last_recommendation_params(SESSION)["n_days"] == 7
    assert len(store.get_last_recommendation_tasks(SESSION)) == 1


def test_reset_clears_everything():
    store = _test_store()
    store.add_request(SAMPLE_REQUEST, SESSION)
    store.append_approved([SAMPLE_REQUEST], SESSION)
    store.reset()
    assert store.all_requests_df(SESSION).empty
    assert store.get_approved(SESSION) == []
    assert store.get_last_recommendation(SESSION) is None


def test_state_visible_across_separate_connections():
    """The real point of a server-backed store over the old in-process
    SQLite file: two independent connections to the same database see the
    same committed data -- e.g. the API process and a separate admin/
    inspection script, or two API workers."""
    store1 = _test_store()
    store1.add_request(SAMPLE_REQUEST, SESSION)

    store2 = BlockRequestStore(dsn=_test_database_url())
    row = store2.get_request("SMMS-REQ-00001", SESSION)
    assert row is not None
    assert row["department"] == "Signalling"


# ------------------------------------------ per-visitor session isolation

def test_two_sessions_never_see_each_others_requests():
    store = _test_store()
    store.add_request(SAMPLE_REQUEST, SESSION_A)
    assert store.get_request("SMMS-REQ-00001", SESSION_B) is None
    assert len(store.all_requests_df(SESSION_B)) == 0
    assert store.get_request("SMMS-REQ-00001", SESSION_A) is not None


def test_two_sessions_can_reuse_the_same_task_id_independently():
    # The whole point of copying the shared demo template into each
    # visitor's own rows: the SAME task_id must be able to exist,
    # independently, in both sessions at once.
    store = _test_store()
    store.add_request(SAMPLE_REQUEST, SESSION_A)
    store.add_request(SAMPLE_REQUEST, SESSION_B)
    store.update_request_status("SMMS-REQ-00001", "approved", SESSION_A)

    assert store.get_request("SMMS-REQ-00001", SESSION_A)["status"] == "approved"
    assert store.get_request("SMMS-REQ-00001", SESSION_B)["status"] == "pending"


def test_two_sessions_never_see_each_others_approved_rows():
    store = _test_store()
    store.append_approved([SAMPLE_REQUEST], SESSION_A)
    assert store.get_approved(SESSION_A) == [SAMPLE_REQUEST]
    assert store.get_approved(SESSION_B) == []


def test_two_sessions_never_see_each_others_emergencies():
    store = _test_store()
    store.add_emergency({"task_id": "EMRG-00001", "department": "Engineering"}, SESSION_A)
    assert store.get_emergencies(SESSION_A) != []
    assert store.get_emergencies(SESSION_B) == []
    # each session's own emergency numbering starts fresh, independently
    assert store.next_emergency_id(SESSION_A) == "EMRG-00002"
    assert store.next_emergency_id(SESSION_B) == "EMRG-00001"


def test_reset_session_tasks_only_clears_that_session():
    store = _test_store()
    store.add_request(SAMPLE_REQUEST, SESSION_A)
    store.add_request(SAMPLE_REQUEST, SESSION_B)
    store.append_approved([SAMPLE_REQUEST], SESSION_A)

    store.reset_session_tasks(SESSION_A)

    assert store.all_requests_df(SESSION_A).empty
    assert store.get_approved(SESSION_A) == []
    assert len(store.all_requests_df(SESSION_B)) == 1


def test_reset_session_emergencies_only_clears_that_session():
    store = _test_store()
    store.add_emergency({"task_id": "EMRG-00001", "department": "Engineering"}, SESSION_A)
    store.add_emergency({"task_id": "EMRG-00001", "department": "Engineering"}, SESSION_B)
    store.set_emergency_reassignment("SMMS-REQ-00001", {"date": "2026-09-07"}, SESSION_A)

    store.reset_session_emergencies(SESSION_A)

    assert store.get_emergencies(SESSION_A) == []
    assert store.get_emergency_reassignments(SESSION_A) == {}
    assert len(store.get_emergencies(SESSION_B)) == 1


# ------------------------------------------------- shared demo task template

def test_activate_demo_batch_copies_templates_into_the_sessions_own_requests():
    store = _test_store()
    store.add_template_task(dict(SAMPLE_REQUEST, status="pending"))

    activated = store.activate_demo_batch(SESSION_A)

    assert activated == 1
    row = store.get_request("SMMS-REQ-00001", SESSION_A)
    assert row is not None
    assert row["status"] == "pending"
    # the shared template pool itself is untouched, and not visible to any
    # OTHER session that hasn't activated it
    assert store.get_request("SMMS-REQ-00001", SESSION_B) is None


def test_activate_demo_batch_is_idempotent():
    store = _test_store()
    store.add_template_task(dict(SAMPLE_REQUEST, status="pending"))
    store.activate_demo_batch(SESSION_A)
    store.update_request_status("SMMS-REQ-00001", "approved", SESSION_A)

    # re-activating (e.g. a stray double click) must not error, and resets
    # this task back to pending rather than leaving it stuck approved
    store.activate_demo_batch(SESSION_A)
    assert store.get_request("SMMS-REQ-00001", SESSION_A)["status"] == "pending"


def test_activate_demo_batch_with_no_templates_is_a_safe_no_op():
    store = _test_store()
    assert store.activate_demo_batch(SESSION_A) == 0
    assert store.all_requests_df(SESSION_A).empty


def test_activate_demo_batch_anchors_dates_to_real_today_not_the_templates_own_frozen_values():
    # Session 35, at explicit user request, after a real reported gap:
    # the template's own raised_date/due_date (baked in once, at
    # whatever real date tasks.csv was imported on) must NEVER be used
    # as-is -- an evaluator opening the hosted link weeks later would
    # otherwise see everything already overdue on arrival.
    store = _test_store()
    stale_template = dict(SAMPLE_REQUEST, raised_date="2020-01-01", due_date="2020-01-15", days_overdue=999)
    store.add_template_task(stale_template)

    store.activate_demo_batch(SESSION_A)
    row = store.get_request("SMMS-REQ-00001", SESSION_A)

    today = now_ist().date()
    raised = date.fromisoformat(row["raised_date"])
    due = date.fromisoformat(row["due_date"])
    assert raised != date(2020, 1, 1)
    assert today - timedelta(days=5) <= raised <= today
    assert due >= today
    assert row["days_overdue"] == 0
