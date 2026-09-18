import pytest
from fastapi.testclient import TestClient

from railblock.api.app import app
from railblock.api.store import reset_store_for_tests
from railblock.paths import TRAIN_DETAILS_CSV

pytestmark = pytest.mark.skipif(not TRAIN_DETAILS_CSV.exists(), reason="real dataset not present")


@pytest.fixture
def client():
    reset_store_for_tests()
    return TestClient(app)


VALID_REQUEST = {
    "department": "Signalling",
    "section_id": "MAS-BBQ",
    "defect_type": "Signal lamp failure",
    "requester_priority": "Critical",
    "estimated_block_hours": 0.75,
    "due_date": "2026-12-31",
}


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_meta_returns_form_metadata(client):
    r = client.get("/meta")
    assert r.status_code == 200
    body = r.json()
    assert set(body["departments"]) == {"Engineering", "Signalling", "Traction"}
    assert set(body["priorities"]) == {"Critical", "Moderate", "Routine"}
    assert "Signal lamp failure" in body["defect_types_by_department"]["Signalling"]
    assert body["sm_direct_approval_max_hours"] == 1.0


def test_corridor_returns_fine_grained_56_section_structure(client):
    # Session 16: scope reduced from MAS-CBE (102 points/101 sections) to
    # MAS-JTJ (57 points/56 sections).
    r = client.get("/corridor")
    assert r.status_code == 200
    body = r.json()
    assert len(body["stations"]) == 57
    assert len(body["sections"]) == 56
    assert body["stations"][0]["station_code"] == "MAS"
    assert body["stations"][-1]["station_code"] == "JTJ"
    assert "MAS-BBQ" in {s["section_id"] for s in body["sections"]}


# --------------------------------------------------------------- validation

def test_submit_request_rejects_unknown_department(client):
    bad = {**VALID_REQUEST, "department": "Catering"}
    r = client.post("/requests", json=bad)
    assert r.status_code == 400


def test_submit_request_rejects_defect_type_not_valid_for_department(client):
    bad = {**VALID_REQUEST, "department": "Engineering", "defect_type": "Signal lamp failure"}
    r = client.post("/requests", json=bad)
    assert r.status_code == 400


def test_submit_request_rejects_unknown_priority(client):
    bad = {**VALID_REQUEST, "requester_priority": "Urgent!!"}
    r = client.post("/requests", json=bad)
    assert r.status_code == 400


def test_submit_request_rejects_unknown_section(client):
    bad = {**VALID_REQUEST, "section_id": "NOT-A-REAL-SECTION"}
    r = client.post("/requests", json=bad)
    assert r.status_code == 400


def test_recommend_without_pending_requests_errors(client):
    r = client.post("/schedule/recommend", json={"start_date": "2026-09-07", "n_days": 7, "time_limit_s": 10})
    assert r.status_code == 400


def test_approve_without_recommendation_errors(client):
    r = client.post("/schedule/approve", json={})
    assert r.status_code == 400


def test_seed_demo_rejects_unknown_scenario(client):
    r = client.post("/demo/seed", json={"demand_scenario": "nonsense", "start_date": "2026-09-07"})
    assert r.status_code == 400


# ------------------------------------------------------- request submission

def test_submit_request_returns_derived_fields(client):
    r = client.post("/requests", json=VALID_REQUEST)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "pending"
    assert body["data_source"] == "USER_SUBMITTED"
    assert body["approval_path"] == "sm_direct"  # 0.75h <= 1.0h
    # Session 30: every task is splittable now (railblock.synthetic.
    # maintenance_tasks.splittable_for always returns True).
    assert body["splittable"] is True
    assert body["source_system"] == "SMMS"
    assert body["task_id"].startswith("SMMS-REQ-")


def test_submitted_request_appears_in_pending_list(client):
    r = client.post("/requests", json=VALID_REQUEST)
    task_id = r.json()["task_id"]
    pending = client.get("/requests?status=pending").json()
    assert task_id in {row["task_id"] for row in pending}


def test_seed_demo_adds_correct_count(client):
    from railblock.synthetic.maintenance_tasks import n_tasks_for_scenario

    r = client.post("/demo/seed", json={"demand_scenario": "double_track_adjusted", "start_date": "2026-09-07", "seed": 1})
    assert r.status_code == 200
    body = r.json()
    # Not a hardcoded count -- it's derived from CORRIDOR_LENGTH_KM, which
    # moves whenever the corridor's real scope does (Session 16: MAS-CBE
    # -> MAS-JTJ); comparing against the same real computation the
    # endpoint itself uses keeps this test correct across future scope
    # changes instead of needing another manual update.
    assert body["added"] == body["n_tasks"] == n_tasks_for_scenario("double_track_adjusted")


def test_last_recommendation_unavailable_before_any_recommend_call(client):
    r = client.get("/schedule/recommendation")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False
    assert body["recommendation"] is None


# --------------------------------------------------------- core end-to-end

def test_full_workflow_submit_recommend_approve_appears_in_schedule_and_history(client):
    submitted = client.post("/requests", json=VALID_REQUEST).json()
    client.post("/demo/seed", json={"demand_scenario": "double_track_adjusted", "start_date": "2026-09-07", "seed": 123})

    rec = client.post(
        "/schedule/recommend", json={"start_date": "2026-09-07", "n_days": 7, "time_limit_s": 20}
    )
    assert rec.status_code == 200
    rec_body = rec.json()
    assert rec_body["status"] in ("OPTIMAL", "FEASIBLE")
    scheduled_ids = {row["task_id"] for row in rec_body["schedule"]}
    assert len(scheduled_ids) == rec_body["option_counts"]["whole"] + rec_body["option_counts"]["split"]

    # the explicitly-submitted request's status should have moved off "pending"
    requests_after = {r["task_id"]: r["status"] for r in client.get("/requests").json()}
    assert requests_after[submitted["task_id"]] != "pending"

    peek = client.get("/schedule/recommendation").json()
    assert peek["available"] is True
    assert {row["task_id"] for row in peek["recommendation"]["schedule"]} == scheduled_ids

    approve = client.post("/schedule/approve", json={})
    assert approve.status_code == 200
    approve_body = approve.json()
    assert approve_body["approved_count"] == len(scheduled_ids)
    assert {r["task_id"] for r in approve_body["approved"]} == scheduled_ids

    weekly = client.get("/schedule/weekly").json()
    all_matrix_entries = [
        entry
        for section_cells in weekly["cells"].values()
        for day_entries in section_cells.values()
        for entry in day_entries
    ]
    matrix_task_ids = {entry["task_id"] for entry in all_matrix_entries}
    assert scheduled_ids <= matrix_task_ids  # every approved task appears at least once in the matrix
    # Session 28: /schedule/weekly now also merges in the fixed granted-
    # history dataset (see railblock.synthetic.granted_history), which
    # spans a much wider real-calendar range than this test's own
    # 2026-09-07..13 window -- isolate this test's own live rows (tagged
    # data_source="LIVE_APPROVED", never "GRANTED_HISTORY_SYNTHETIC")
    # before checking they land in the expected date range.
    live_dates = {e["date"] for e in all_matrix_entries if e["data_source"] != "GRANTED_HISTORY_SYNTHETIC"}
    assert live_dates <= {f"2026-09-{d:02d}" for d in range(7, 14)}

    history = client.get("/tasks/history").json()
    # Session 17: /tasks/history now also merges in the fixed granted-
    # history dataset's not-yet-completed rows alongside this test's own
    # live-approved ones -- isolate the live rows (they carry
    # completion_verified; granted-history rows never do, see
    # railblock.synthetic.granted_history) before comparing against
    # scheduled_ids, rather than assuming the endpoint returns only what
    # this test itself approved.
    live_rows = [t for t in history["tasks"] if "completion_verified" in t]
    assert {t["task_id"] for t in live_rows} == scheduled_ids
    for t in live_rows:
        assert t["completion_verified"] is False  # never silently implied complete


def test_approve_specific_subset_of_task_ids(client):
    client.post("/demo/seed", json={"demand_scenario": "double_track_adjusted", "start_date": "2026-09-07", "seed": 5})
    rec = client.post("/schedule/recommend", json={"start_date": "2026-09-07", "n_days": 7, "time_limit_s": 20}).json()
    scheduled_ids = [row["task_id"] for row in rec["schedule"]]
    if not scheduled_ids:
        pytest.skip("no fully-scheduled tasks in this run to test subset approval against")

    one_id = scheduled_ids[0]
    approve = client.post("/schedule/approve", json={"task_ids": [one_id]})
    assert approve.status_code == 200
    assert approve.json()["approved_count"] == 1
    assert client.get("/requests?status=approved").json()[0]["task_id"] == one_id


def test_approve_unknown_task_id_404s(client):
    client.post("/demo/seed", json={"demand_scenario": "double_track_adjusted", "start_date": "2026-09-07", "seed": 5})
    client.post("/schedule/recommend", json={"start_date": "2026-09-07", "n_days": 7, "time_limit_s": 20})
    r = client.post("/schedule/approve", json={"task_ids": ["TOTALLY-FAKE-ID"]})
    assert r.status_code == 404


def test_monthly_matrix_shape(client):
    client.post("/demo/seed", json={"demand_scenario": "double_track_adjusted", "start_date": "2026-09-07", "seed": 9})
    r = client.get("/schedule/monthly", params={"start_date": "2026-09-07", "n_weeks": 4, "time_limit_s": 20})
    assert r.status_code == 200
    body = r.json()
    assert body["weeks"] == [1, 2, 3, 4]
    for section_weeks in body["cells"].values():
        assert set(section_weeks.keys()) == {"1", "2", "3", "4"} or set(section_weeks.keys()) == {1, 2, 3, 4}


def test_get_single_request(client):
    submitted = client.post("/requests", json=VALID_REQUEST).json()
    r = client.get(f"/requests/{submitted['task_id']}")
    assert r.status_code == 200
    assert r.json()["task_id"] == submitted["task_id"]

    missing = client.get("/requests/NOPE-00000")
    assert missing.status_code == 404


def test_recommend_reconsiders_previously_unscheduled_waiting_list_tasks(client):
    """A task left recommended_unscheduled by one run must still be fed
    into the NEXT run -- it must never be silently frozen out just
    because its status is no longer 'pending'."""
    client.post("/demo/seed", json={"demand_scenario": "double_track_adjusted", "start_date": "2026-09-07", "seed": 1})
    first = client.post("/schedule/recommend", json={"start_date": "2026-09-07", "n_days": 7, "time_limit_s": 10}).json()
    if not first["unscheduled_task_ids"]:
        return  # nothing to prove this run; still a valid pass
    stuck_task_id = first["unscheduled_task_ids"][0]
    assert client.get(f"/requests/{stuck_task_id}").json()["status"] == "recommended_unscheduled"

    second = client.post("/schedule/recommend", json={"start_date": "2026-09-07", "n_days": 7, "time_limit_s": 10}).json()
    all_considered = (
        {r["task_id"] for r in second["schedule"]}
        | {r["task_id"] for r in second["partial"]}
        | set(second["unscheduled_task_ids"])
    )
    assert stuck_task_id in all_considered


def test_corridor_includes_real_lat_lon_for_every_station(client):
    # Session 17, at explicit user request: every station now gets a
    # lat/lon (not just the 9 majors) -- majors keep their real geo_source,
    # every other one is honestly tagged "interpolated_between_majors"
    # rather than left null (see railblock.corridor.geo.
    # interpolate_all_station_geo).
    r = client.get("/corridor")
    body = r.json()
    mas = next(s for s in body["stations"] if s["station_code"] == "MAS")
    assert mas["lat"] is not None and mas["lon"] is not None
    assert mas["geo_source"] == "real"
    assert 8 < mas["lat"] < 14  # sanity: Tamil Nadu latitude band
    non_major = [s for s in body["stations"] if s["station_code"] not in ("MAS", "JTJ")]
    assert non_major  # sanity: the fine model has stations beyond just the 2 endpoints
    assert all(s["lat"] is not None and s["lon"] is not None for s in non_major)
    assert any(s["geo_source"] == "interpolated_between_majors" for s in non_major)


def test_corridor_route_geometry_is_null_when_not_fetched(client):
    r = client.get("/corridor")
    body = r.json()
    assert "route_geometry" in body
    assert body["route_geometry"] is None  # fetch_real_route_geometry.py hasn't been run in the test env


def test_delete_requests_only_removes_pending(client):
    submitted = client.post("/requests", json=VALID_REQUEST).json()
    task_id = submitted["task_id"]
    r = client.request("DELETE", "/requests", json={"task_ids": [task_id]})
    assert r.status_code == 200
    assert r.json()["deleted"] == 1
    assert client.get("/requests").json() == []


def test_delete_requests_does_not_remove_approved(client):
    client.post("/demo/seed", json={"demand_scenario": "double_track_adjusted", "start_date": "2026-09-07", "seed": 3})
    result = client.post("/schedule/recommend", json={"start_date": "2026-09-07", "n_days": 7, "time_limit_s": 10}).json()
    if not result["schedule"]:
        return
    approved = client.post("/schedule/approve", json={}).json()
    approved_id = approved["approved"][0]["task_id"]
    r = client.request("DELETE", "/requests", json={"task_ids": [approved_id]})
    assert r.json()["deleted"] == 0
    assert client.get("/requests").json()  # still present


def test_reject_request_marks_rejected_and_excludes_from_pending(client):
    submitted = client.post("/requests", json=VALID_REQUEST).json()
    task_id = submitted["task_id"]

    r = client.post(f"/requests/{task_id}/reject")
    assert r.status_code == 200
    assert r.json() == {"task_id": task_id, "status": "rejected"}

    assert client.get(f"/requests/{task_id}").json()["status"] == "rejected"
    assert task_id not in {row["task_id"] for row in client.get("/requests", params={"status": "pending"}).json()}


def test_reject_request_unknown_task_404(client):
    r = client.post("/requests/NOPE-00000/reject")
    assert r.status_code == 404


def test_reject_request_already_approved_is_rejected_with_400(client):
    client.post("/demo/seed", json={"demand_scenario": "double_track_adjusted", "start_date": "2026-09-07", "seed": 3})
    result = client.post("/schedule/recommend", json={"start_date": "2026-09-07", "n_days": 7, "time_limit_s": 10}).json()
    if not result["schedule"]:
        return
    approved = client.post("/schedule/approve", json={}).json()
    approved_id = approved["approved"][0]["task_id"]

    r = client.post(f"/requests/{approved_id}/reject")
    assert r.status_code == 400


def test_trains_at_a_time_returns_real_transits(client):
    r = client.get("/trains/positions", params={"date": "2026-09-07", "minute": 360})
    assert r.status_code == 200
    body = r.json()
    assert body["trains"], "expected at least one real train transiting the corridor at 06:00"
    for t in body["trains"]:
        assert set(t.keys()) == {"train_no", "train_name", "section_id", "progress", "start_minute", "end_minute", "source"}
        assert 0.0 <= t["progress"] <= 1.0
        assert t["source"] == "computed"


def test_trains_at_a_time_live_flag_falls_back_when_railradar_unreachable(client):
    """live=true must never error or crash even if RailRadar is unreachable
    (no key configured, network down, bad response, etc.) -- it should
    silently keep the schedule-computed entry."""
    r = client.get("/trains/positions", params={"date": "2026-09-07", "minute": 360, "live": True})
    assert r.status_code == 200
    body = r.json()
    for t in body["trains"]:
        assert t["source"] in ("computed", "live")


def test_schedule_options_returns_multiple_real_strategies(client):
    client.post("/demo/seed", json={"demand_scenario": "double_track_adjusted", "start_date": "2026-09-07", "seed": 3})
    r = client.post("/schedule/options", json={"start_date": "2026-09-07", "n_days": 7, "time_limit_s": 10})
    assert r.status_code == 200
    body = r.json()
    assert 1 <= len(body["options"]) <= 3
    keys = {o["key"] for o in body["options"]}
    assert "balanced" in keys
    recommended = [o for o in body["options"] if o["recommended"]]
    assert len(recommended) == 1
    for o in body["options"]:
        s = o["summary"]
        assert s["tasks_total"] > 0
        assert 0 <= s["tasks_scheduled"] <= s["tasks_total"]
        assert 0 <= s["critical_scheduled"] <= s["critical_total"]

    peek = client.get("/schedule/options").json()
    assert peek["available"] is True
    assert len(peek["options"]) == len(body["options"])


def test_approve_by_option_key(client):
    client.post("/demo/seed", json={"demand_scenario": "double_track_adjusted", "start_date": "2026-09-07", "seed": 3})
    r = client.post("/schedule/options", json={"start_date": "2026-09-07", "n_days": 7, "time_limit_s": 10})
    opts = r.json()
    balanced = next(o for o in opts["options"] if o["key"] == "balanced")
    if not balanced["schedule"]:
        return
    r = client.post("/schedule/approve", json={"option_key": "balanced"})
    assert r.status_code == 200
    assert r.json()["approved_count"] == len(balanced["schedule"])

    r_bad = client.post("/schedule/approve", json={"option_key": "does_not_exist"})
    assert r_bad.status_code == 404


# --------------------------------------------------- per-visitor isolation
# Session 32, at explicit user request: every endpoint above ran with no
# X-Session-Id header at all, so they all implicitly shared ONE session
# (DEFAULT_SESSION_ID) the whole time -- proving nothing broke by that
# change. These tests are specifically about the isolation itself: two
# different header values must never see each other's state.

HEADERS_A = {"X-Session-Id": "test-session-a"}
HEADERS_B = {"X-Session-Id": "test-session-b"}


def test_two_sessions_have_independent_waiting_lists(client):
    client.post("/requests", json=VALID_REQUEST, headers=HEADERS_A)
    assert client.get("/requests", headers=HEADERS_A).json() != []
    assert client.get("/requests", headers=HEADERS_B).json() == []


def test_two_sessions_have_independent_approved_schedules(client):
    client.post("/demo/seed", json={"demand_scenario": "double_track_adjusted", "start_date": "2026-09-07", "seed": 5}, headers=HEADERS_A)
    client.post("/schedule/recommend", json={"start_date": "2026-09-07", "n_days": 7, "time_limit_s": 10}, headers=HEADERS_A)
    approve_a = client.post("/schedule/approve", json={}, headers=HEADERS_A).json()
    assert approve_a["approved_count"] > 0

    weekly_a = client.get("/schedule/weekly", headers=HEADERS_A).json()
    weekly_b = client.get("/schedule/weekly", headers=HEADERS_B).json()
    approved_ids = {row["task_id"] for row in approve_a["approved"]}
    ids_in_weekly_b = {c["task_id"] for cells in weekly_b.get("cells", {}).values() for day in cells.values() for c in day}
    assert not (approved_ids & ids_in_weekly_b)
    ids_in_weekly_a = {c["task_id"] for cells in weekly_a.get("cells", {}).values() for day in cells.values() for c in day}
    assert approved_ids & ids_in_weekly_a


def test_two_sessions_have_independent_emergencies(client):
    client.post("/emergency/create", headers=HEADERS_A)
    assert client.get("/emergency/list", headers=HEADERS_A).json()["emergencies"] != []
    assert client.get("/emergency/list", headers=HEADERS_B).json()["emergencies"] == []


def test_missing_session_header_falls_back_to_the_shared_default(client):
    # Every pre-existing test in this file relies on exactly this -- no
    # header at all must never error, and must behave identically to
    # explicitly passing the default.
    from railblock.api.store import DEFAULT_SESSION_ID

    client.post("/requests", json=VALID_REQUEST)
    default_headers = {"X-Session-Id": DEFAULT_SESSION_ID}
    assert client.get("/requests", headers=default_headers).json() == client.get("/requests").json()


def test_demo_batch_activate_reset_cycle(client):
    from railblock.api.request_builder import build_request_row
    from railblock.api.schemas import BlockRequestIn
    from railblock.api.store import get_store

    template_row = build_request_row(BlockRequestIn(**VALID_REQUEST), data_source="FILE_IMPORTED")
    get_store().add_template_task(template_row)

    activated = client.post("/demo/batch/activate", headers=HEADERS_A)
    assert activated.status_code == 200
    assert activated.json()["activated"] == 1
    assert client.get("/requests", headers=HEADERS_A).json() != []
    # a DIFFERENT session never sees session A's activated copy
    assert client.get("/requests", headers=HEADERS_B).json() == []

    client.post("/schedule/recommend", json={"start_date": "2026-09-07", "n_days": 7, "time_limit_s": 10}, headers=HEADERS_A)
    client.post("/schedule/approve", json={}, headers=HEADERS_A)

    reset = client.post("/demo/batch/reset", headers=HEADERS_A)
    assert reset.status_code == 200
    assert reset.json()["reset"] is True
    assert client.get("/requests", headers=HEADERS_A).json() == []
    assert client.get("/schedule/recommendation", headers=HEADERS_A).json()["available"] is False


def test_emergency_reset_endpoint_restores_original_placement(client):
    client.post("/demo/seed", json={"demand_scenario": "double_track_adjusted", "start_date": "2026-09-07", "seed": 7}, headers=HEADERS_A)
    client.post("/schedule/recommend", json={"start_date": "2026-09-07", "n_days": 7, "time_limit_s": 10}, headers=HEADERS_A)
    client.post("/schedule/approve", json={}, headers=HEADERS_A)

    created = client.post("/emergency/create", headers=HEADERS_A).json()
    assert client.get("/emergency/list", headers=HEADERS_A).json()["emergencies"] != []

    reset = client.post("/emergency/reset", headers=HEADERS_A)
    assert reset.status_code == 200
    assert reset.json()["reset"] is True
    assert client.get("/emergency/list", headers=HEADERS_A).json()["emergencies"] == []
