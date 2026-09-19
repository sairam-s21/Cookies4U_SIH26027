"""FastAPI layer (Session 5, Task 3).

Wraps the existing Layer 2-5 functions -- no scheduling/KPI logic lives
here, this module only does request validation, in-memory state
management (railblock.api.store), and response shaping. See store.py's
docstring for the synchronous-vs-cached performance judgement call.

Endpoints:
  GET    /corridor             station/section structure incl. real lat/lon (Session 9) for the 19 major stations
  GET    /trains/positions      Session 9: real train positions at a given date+minute, from the real timetable
  POST   /requests              submit a new block request
  GET    /requests              list requests (optional ?status= filter)
  GET    /requests/{task_id}    a single request's row (fast lookup, no full-list fetch)
  POST   /requests/{task_id}/reject  Session 13: COA rejects a request outright (not required)
  DELETE /requests              Session 9: bulk-delete pending requests (demo-batch retraction)
  POST   /demo/seed             populate pending requests from a Task 1 demand scenario
  POST   /schedule/recommend    run Layer 2/3 over all pending requests (single result)
  GET    /schedule/recommendation  read-only peek at the last recommendation (Session 7)
  POST   /schedule/options       Session 9: 3-5 genuinely different real schedule options
  GET    /schedule/options       read-only peek at the last /schedule/options result
  POST   /schedule/approve      approve some/all of the last recommendation or a chosen option
  GET    /schedule/weekly       approved schedule, shaped as a section x date matrix
  GET    /schedule/monthly      Layer 4 monthly plan, shaped as a section x week matrix
  GET    /tasks/history         approved schedule entries (completion_verified shown as-is)
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import date, timedelta

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import pandas as pd

from railblock.api.schemas import (
    ApproveRequest,
    BlockRequestIn,
    BlockRequestOut,
    DeleteRequestsRequest,
    EmergencyResolveRequest,
    MonthlyPlanRequest,
    RecommendRequest,
    ScheduleOptionsRequest,
    SeedDemoRequest,
)
from railblock.api.request_builder import RequestValidationError, build_request_row
from railblock.api.store import DEFAULT_SESSION_ID, CorridorContext, get_corridor_context, get_store
from railblock.availability.train_positions import enrich_many_with_live_status, trains_at
from railblock.corridor.geo import interpolate_all_station_geo, load_real_route_geometry
from railblock.integrations.import_tasks import check_for_new_rows
from railblock.scheduling.adaptive_allocation import apply_adaptive_allocation
from railblock.prioritization.whittle import rank_tasks
from railblock.scheduling.layer4 import generate_monthly_plan
from railblock.scheduling.capacity import compute_daily_window_capacity
from railblock.scheduling.combining import (
    approved_rows_to_window_rows,
    combine_into_occupied_windows,
    subtract_consumed_capacity,
)
from railblock.scheduling.emergency import create_demo_emergency, find_affected_rows, propose_reschedule
from railblock.scheduling.orchestrator import FullScheduleResult, solve_schedule_with_options
from railblock.scheduling.schedule_options import ScheduleOption, generate_schedule_options, summarize_option
from railblock.synthetic.goods_forecast import generate_goods_forecast
from railblock.synthetic.granted_history import IST, classify_blocks_by_time, demo_active_row, load_granted_history, now_ist
from railblock.synthetic.maintenance_tasks import (
    DEFECT_TYPES,
    DEMAND_SCENARIOS,
    DEPARTMENTS,
    PRIORITY_WEIGHTS,
    SM_DIRECT_APPROVAL_MAX_HOURS,
    generate_maintenance_tasks,
)

log = logging.getLogger("railblock.tasks_csv_watch")
TASKS_CSV_POLL_INTERVAL_S = 2.0


def get_session_id(x_session_id: str | None = Header(default=None, alias="X-Session-Id")) -> str:
    """Session 32, at explicit user request: per-visitor isolation for a
    publicly-hosted, shared-link deployment. A request with no session
    header at all (every pre-existing test, script, or curl call) is
    treated as one shared DEFAULT_SESSION_ID -- never an error, and fully
    backward compatible with anything written before this change. A real
    browser client always sends a real per-visitor UUID (see
    frontend/src/api/session.js), so only it ever gets true isolation."""
    return x_session_id or DEFAULT_SESSION_ID


async def _watch_tasks_csv() -> None:
    """Session 13: background poll loop, started at app startup -- see
    railblock.integrations.import_tasks.check_for_new_rows for why this is
    polling (row-count based, append-only) rather than an inotify/
    filesystem-event watcher. Runs for the lifetime of the app process;
    a single bad row never stops the loop (check_for_new_rows already
    reports and skips it), and any unexpected exception is logged and
    the loop keeps going rather than silently dying."""
    while True:
        try:
            imported, failed = check_for_new_rows()
            if imported or failed:
                log.info(f"tasks.csv watch: {imported} new row(s) imported, {failed} rejected")
        except Exception:
            log.exception("tasks.csv watch: error checking for new rows")
        await asyncio.sleep(TASKS_CSV_POLL_INTERVAL_S)


@asynccontextmanager
async def lifespan(app: FastAPI):
    watch_task = asyncio.create_task(_watch_tasks_csv())
    yield
    watch_task.cancel()


app = FastAPI(title="RailBlock Co-Pilot API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo/hackathon scope -- tighten before any real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)



def _df_records(df):
    """DataFrame -> list[dict] with NaN replaced by None -- needed because
    concatenating frames with different columns (e.g. orchestrator.py's
    full_df + negotiated_df, which have distinct option-specific fields)
    fills the gaps with NaN, which is not valid JSON (json.dumps raises on
    a bare NaN unless allow_nan, which still emits non-standard `NaN`
    tokens most JSON parsers reject)."""
    if df is None or df.empty:
        return []
    return df.astype(object).where(df.notna(), None).to_dict("records")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/now")
def now():
    """Session 29, at explicit user request, after a real reported bug:
    the frontend's own `todayIsoLocal()` (Schedule.jsx, RecommendedScheduling.jsx,
    CorridorMapPage.jsx) trusted the VIEWER's own machine clock for
    "today" -- confirmed live to genuinely disagree with this server's
    real clock (a user's Monthly Schedule "Today" button landed on
    August when this server's own real IST "today" is September).
    Every real-current-date UI default should come from HERE, the one
    real, authoritative clock every other part of this system (granted-
    history generation, live/upcoming/active classification) already
    anchors to -- never the browser's own `new Date()`, which has no
    reason to agree with it.

    `date`: plain "YYYY-MM-DD", day granularity, for anything that only
    needs "which day is it" (Schedule.jsx/RecommendedScheduling.jsx).
    `datetime`: the SAME real moment, but explicitly timezone-aware
    (real IST offset attached, unlike this module's other naive-IST
    convention) -- CorridorMapPage.jsx's live minute-of-day tracking
    needs an unambiguous instant a JS Date() can parse correctly
    regardless of the VIEWER's own browser timezone; a bare
    "YYYY-MM-DDTHH:MM:SS" with no offset would otherwise get
    reinterpreted as browser-local time, silently wrong for any viewer
    not physically in IST."""
    n = now_ist()
    return {"date": n.date().isoformat(), "datetime": n.replace(tzinfo=IST).isoformat()}


@app.get("/meta")
def meta():
    """Session 7: static form metadata (departments, defect types per
    department, priority levels) for the "Raise Block Request" form --
    served from the same source of truth (railblock.synthetic.
    maintenance_tasks) the backend itself validates against, rather than
    letting the frontend hold its own copy that could drift out of sync.
    """
    return {
        "departments": DEPARTMENTS,
        "defect_types_by_department": DEFECT_TYPES,
        "priorities": list(PRIORITY_WEIGHTS.keys()),
        "sm_direct_approval_max_hours": SM_DIRECT_APPROVAL_MAX_HOURS,
    }


@app.get("/corridor")
def corridor():
    """Session 7: the corridor structure the frontend needs to populate a
    section dropdown and draw the corridor map -- stations ordered by real
    distance_km, and the fine-grained (Session 4 default, Session 16 scope
    reduced to MAS-JTJ) block sections between them. Not previously
    exposed; every other endpoint that needs this data loaded it via
    get_corridor_context() internally, but nothing returned it to a
    caller until the frontend needed to know what sections exist at all.

    Session 17, at explicit user request: every station now gets a
    lat/lon (not just the 9 majors) -- see
    railblock.corridor.geo.interpolate_all_station_geo for exactly how
    the non-major ones are positioned (real distance-fraction
    interpolation between bracketing majors, honestly tagged via
    geo_source, not a survey position) -- this lets the map draw
    maintenance highlighting at the real fine-section granularity instead
    of collapsing dozens of real sections onto one oversized major-to-
    major line.
    """
    ctx = get_corridor_context()
    geo = interpolate_all_station_geo(ctx.stations)
    stations = _df_records(ctx.stations)
    for s in stations:
        g = geo.get(s["station_code"])
        s["lat"] = g["lat"] if g else None
        s["lon"] = g["lon"] if g else None
        s["geo_source"] = g["geo_source"] if g else None
    return {
        "stations": stations,
        "sections": _df_records(ctx.sections),
        "geo_note": (
            "lat/lon for the 9 major stations (Session 9, real datasets/india_railway_stations.csv "
            "coordinates) is real; every other station's lat/lon (Session 17) is interpolated between "
            "its bracketing major stations by real distance_km fraction, since these fine-grained "
            "signal cabins/halts have no public real lat/lon of their own -- see geo_source per station."
        ),
        "route_geometry": load_real_route_geometry(),
    }


@app.get("/corridor/sections/{section_id}/trains")
def section_train_schedule(section_id: str, date_: date = Query(..., alias="date")):
    """Session 9: every real train transiting `section_id` on `date_` (per
    the real timetable's passenger_occupancy + its assigned weekly running
    pattern) -- used to overlay real train-passing times on the Weekly/
    Monthly Schedule grid, even for sections/days with no maintenance
    block scheduled."""
    ctx = get_corridor_context()
    weekday = date_.weekday()
    pax = ctx.passenger_occupancy
    rows = pax[pax["section_id"] == section_id]
    out = []
    for _, row in rows.iterrows():
        wd_set = row.get("weekdays")
        if wd_set is not None and weekday not in wd_set:
            continue
        train_name = row.get("train_name", "")
        out.append(
            {
                "train_no": str(row["train_no"]),
                "train_name": train_name,
                "start_minute": row["start_minute"],
                "end_minute": row["end_minute"],
            }
        )
    return {"section_id": section_id, "date": date_.isoformat(), "trains": out}


@app.get("/trains/positions")
def train_positions(
    date_: date = Query(..., alias="date"),
    minute: float = Query(..., ge=0, lt=1440),
    live: bool = Query(False),
):
    """Session 9: real train positions at `date`/`minute` (minute-of-day,
    0-1439), derived from the real timetable's passenger_occupancy -- see
    railblock.availability.train_positions for exactly what "real" means
    here (every train's real transit times + a documented weekly-running
    assumption, never a fabricated position).

    Session 12: `live=true` additionally tries a real RailRadar lookup for
    each of these trains and, only if one succeeds, overlays its real
    lat/lon/delay onto that entry (source="live") -- a train RailRadar
    doesn't have data for, or if the API/key is unavailable, keeps its
    schedule-computed entry untouched (source="computed"), never an error.
    """
    ctx = get_corridor_context()
    trains = trains_at(ctx.passenger_occupancy, date_, minute)
    for t in trains:
        t["source"] = "computed"
    if live:
        enrich_many_with_live_status(trains)
    return {"date": date_.isoformat(), "minute": minute, "trains": trains}


# ------------------------------------------------------------- requests

@app.post("/requests", response_model=BlockRequestOut)
def submit_request(req: BlockRequestIn, session_id: str = Depends(get_session_id)):
    """Session 13: the "Raise Block Request" UI form was removed (real
    COA doesn't work that way -- departments deliver batch task lists, not
    one-at-a-time web forms). This endpoint is kept for any programmatic
    caller (tests, a future real TDMS/SMMS integration) and now shares its
    validation/derivation logic with railblock.integrations.import_tasks
    (the CSV/JSON batch importer that replaced the form) via
    build_request_row(), so the two paths can never drift apart."""
    try:
        row = build_request_row(req)
    except RequestValidationError as exc:
        raise HTTPException(400, str(exc)) from exc
    get_store().add_request(row, session_id)
    return row


@app.get("/requests")
def list_requests(status: str | None = Query(None), session_id: str = Depends(get_session_id)):
    df = get_store().all_requests_df(session_id)
    if df.empty:
        return []
    if status:
        df = df[df["status"] == status]
    return _df_records(df)


@app.get("/requests/whittle-rank")
def requests_whittle_rank(session_id: str = Depends(get_session_id)):
    """Session 21, at explicit user request: every non-approved request's
    real Whittle-index score, so the Waiting List can offer a "Whittle
    index rank order" sort without the user needing to run a full
    recommendation first. Session 24: the raw whittle_index is still
    included (real, internal, unbounded), but the UI is told to show
    `risk_percentage` instead ("Risk percentage") -- a real, bounded
    [0, 100] percentile rank against every other task in this same
    response, computed by rank_tasks() itself (see its docstring). Uses a
    fixed reference horizon (today, 7 days, deterministically-seeded
    goods forecast, same convention as /schedule/recommend) purely to
    compute the ranking -- this does NOT run CP-SAT and does not touch
    any stored recommendation/schedule state, it is a read-only ranking
    peek. Sorted highest-risk first."""
    ctx = get_corridor_context()
    store = get_store()
    pending = store.non_approved_requests_df(session_id)
    if pending.empty:
        return []
    start_date = date.today()
    goods = generate_goods_forecast(ctx.sections, start_date, n_days=7, seed=int(start_date.strftime("%Y%m%d")))
    ranked = rank_tasks(pending, ctx.sections, ctx.passenger_occupancy, goods, start_date, n_days=7)
    ranked = ranked.sort_values("whittle_index", ascending=False)
    return _df_records(ranked)


def _lookup_task_detail(task_id: str, session_id: str) -> dict | None:
    """A single-task detail lookup, shared by GET /requests/{task_id} and
    the Emergency Handling reschedule flow (the latter needs a displaced
    task's real defect_type/raised_date/due_date/days_overdue to rank and
    re-solve it, exactly the same fields this endpoint already surfaces).

    Session 28, at explicit user request, after a real reported "No
    details found" bug: a granted-history task_id (see railblock.
    synthetic.granted_history -- the fixed, already-"approved" dataset
    that GET /schedule/weekly, /blocks/active, and /blocks/upcoming all
    already show) was never inserted as a real request, so it lives only
    in that separate fixed dataset. Falls back to that dataset (never
    generated/reindexed here, so still O(1)-ish in practice at this
    dataset's real size) before finally giving up (returns None)."""
    row = get_store().get_request(task_id, session_id)
    if row is not None:
        return row

    hist = _granted_history_df(session_id)
    if not hist.empty:
        match = hist[hist["task_id"] == task_id]
        if not match.empty:
            # _df_records (not manual dict-building from .iloc[0]) so
            # numpy scalar types (e.g. estimated_block_hours as
            # numpy.float64) go through the same object/None-safe
            # conversion every other pandas-backed endpoint already uses.
            r = _df_records(match)[0]
            return {
                "task_id": r["task_id"],
                "department": r["department"],
                "section_id": r["section_id"],
                "defect_type": r["defect_type"],
                "requester_priority": r["requester_priority"],
                # Session 29 fix: these are real, already-computed values
                # (railblock.synthetic.granted_history's own row-building
                # generates them the same way any live request would) --
                # .get() only as a defensive fallback for a stale xlsx
                # generated before this field was added, never expected
                # to actually miss on the current file.
                "raised_date": r.get("raised_date"),
                "due_date": r.get("due_date"),
                "days_overdue": r.get("days_overdue"),
                "estimated_block_hours": r["estimated_block_hours"],
                "splittable": r.get("splittable"),
                "approval_path": r.get("approval_path"),
                "data_source": r.get("data_source", "GRANTED_HISTORY_SYNTHETIC"),
                # Session 28: honestly "approved" -- railblock.synthetic.
                # granted_history's own docstring: "every row here already
                # went through approval in the past" -- not fabricated,
                # just the one real status value that's actually true for
                # every row in that dataset, whether it now reads as
                # completed/active/upcoming (a live, time-based question
                # this static field was never meant to answer).
                "status": "approved",
            }
    return None


def _apply_emergency_override(detail: dict, task_id: str, session_id: str) -> dict:
    """Session 30: tags a task's own detail response with whether an
    emergency displaced it -- `_lookup_task_detail`'s shape never carries
    a real date/window (BlockDetailsModal.jsx never showed one), so
    there's nothing else to override here."""
    reassignment = get_store().get_emergency_reassignments(session_id).get(task_id)
    if reassignment is None:
        return detail
    detail = dict(detail)
    detail["rescheduled_due_to_emergency"] = "removed" if reassignment.get("removed") else True
    detail["emergency_task_id"] = reassignment.get("emergency_task_id")
    return detail


@app.get("/requests/{task_id}")
def get_request(task_id: str, session_id: str = Depends(get_session_id)):
    """A single-row lookup -- used by BlockDetailsModal.jsx so opening one
    task's details doesn't have to fetch and filter every request in the
    system (GET /requests) just to find it. See _lookup_task_detail for
    where the granted-history fallback comes from.

    Session 30, at explicit user request: also resolves an emergency
    block's own task_id (see railblock.scheduling.emergency) -- an
    emergency is never a real "request" (it skips that pipeline
    entirely), but the Weekly/Monthly Schedule page lets a user click
    into one expecting the same kind of details modal as any other
    block, and applies a `emergency_reassignments` override to a normal
    task's date/window if it was displaced and rescheduled because of
    one."""
    detail = _lookup_task_detail(task_id, session_id)
    if detail is not None:
        return _apply_emergency_override(detail, task_id, session_id)

    emergency = get_store().get_emergency(task_id, session_id)
    if emergency is not None:
        segments = emergency["segments"]
        affected = emergency.get("affected", [])
        return {
            "task_id": emergency["task_id"],
            "department": emergency["department"],
            "section_id": emergency["section_id"],
            "defect_type": emergency["defect_type"],
            "requester_priority": None,
            "raised_date": None,
            "due_date": None,
            "days_overdue": None,
            "estimated_block_hours": emergency["estimated_block_hours"],
            "approval_path": None,
            "data_source": "EMERGENCY",
            "status": emergency["status"],
            "is_emergency": True,
            "created_at": emergency["created_at"],
            # Session 30, at explicit user request, after a real reported
            # gap: the details modal showed mostly "-" for an emergency
            # (Requester priority/Raised date/Due date/Approval path are
            # all genuinely inapplicable) -- these are the fields that
            # actually matter for an emergency: its real window, and a
            # one-line summary of what it displaced.
            "window_start": {"date": segments[0]["date"], "minute": segments[0]["start_minute"]},
            "window_end": {"date": segments[-1]["date"], "minute": segments[-1]["end_minute"]},
            "affected_count": len(affected),
            "affected_task_ids": [a["task_id"] for a in affected],
        }

    raise HTTPException(404, f"no request with task_id {task_id!r}")


@app.post("/requests/{task_id}/reject")
def reject_request(task_id: str, session_id: str = Depends(get_session_id)):
    """Session 13, at explicit user request: gives COA the authority to
    reject a block request outright (found not required), rather than the
    only outcomes being "gets scheduled" or "sits in the Waiting List
    forever". A rejected request is excluded from the Waiting List and
    every future recommend/options run (see
    store.non_approved_requests_df), same treatment as 'approved' -- a
    real decision, not a pending one. Cannot reject a task that's already
    been approved -- that's a real, already-granted block, a different
    and much bigger decision than this endpoint is for."""
    store = get_store()
    row = store.get_request(task_id, session_id)
    if row is None:
        raise HTTPException(404, f"no request with task_id {task_id!r}")
    if row["status"] == "approved":
        raise HTTPException(400, f"task_id {task_id!r} is already approved -- cannot reject an approved block")
    store.update_request_status(task_id, "rejected", session_id)
    return {"task_id": task_id, "status": "rejected"}


@app.delete("/requests")
def delete_requests(req: DeleteRequestsRequest, session_id: str = Depends(get_session_id)):
    """Session 9: used by the "Demo: seed a full batch" retraction flow --
    the frontend remembers the task_ids of the batch it just seeded and,
    on the NEXT page load, deletes that batch here before re-enabling the
    seed button, so repeated demo runs don't pile up stale synthetic
    requests. Only rows still status='pending' are removed -- a task
    that has since been recommended/approved is never silently deleted."""
    deleted = get_store().delete_pending_requests(req.task_ids, session_id)
    return {"deleted": deleted}


@app.post("/demo/batch/activate")
def activate_demo_batch(session_id: str = Depends(get_session_id)):
    """Session 32, at explicit user request: "Create demo batch of tasks"
    on the Waiting List page -- copies the shared, fixed 70-task template
    pool (see store.template_tasks/add_template_task, populated by
    railblock.integrations.import_tasks from tasks.csv) into THIS
    visitor's own, isolated requests, so a publicly-shared hosted link
    doesn't need me to reload it by hand between demo rounds anymore."""
    activated = get_store().activate_demo_batch(session_id)
    return {"activated": activated}


@app.post("/demo/batch/reset")
def reset_demo_batch(session_id: str = Depends(get_session_id)):
    """Session 32, at explicit user request: the Waiting List page's
    "Reset" button -- clears this visitor's entire requests/approved/
    recommendation/schedule-options footprint (see
    store.reset_session_tasks), i.e. the demo batch (whatever's left of
    it) AND anything scheduled/approved from it, everywhere including the
    Weekly/Monthly Schedule."""
    get_store().reset_session_tasks(session_id)
    return {"reset": True}


@app.post("/emergency/reset")
def reset_emergencies(session_id: str = Depends(get_session_id)):
    """Session 32, at explicit user request: the Emergency Handling page's
    own "Reset" button -- clears every emergency and reassignment override
    THIS visitor created (see store.reset_session_emergencies). A
    displaced task's reassignment is only ever an override on top of its
    real stored data, never an overwrite, so removing these rows alone
    puts every displaced task back at its original slot."""
    get_store().reset_session_emergencies(session_id)
    return {"reset": True}


@app.post("/demo/seed")
def seed_demo(req: SeedDemoRequest, session_id: str = Depends(get_session_id)):
    if req.demand_scenario not in DEMAND_SCENARIOS:
        raise HTTPException(400, f"demand_scenario must be one of {list(DEMAND_SCENARIOS)}")
    ctx = get_corridor_context()
    tasks = generate_maintenance_tasks(
        ctx.sections, as_of_date=req.start_date, seed=req.seed, demand_scenario=req.demand_scenario
    )
    store = get_store()
    for _, row in tasks.iterrows():
        store.add_request(row.to_dict(), session_id)
    return {
        "added": len(tasks),
        "demand_scenario": req.demand_scenario,
        "n_tasks": DEMAND_SCENARIOS[req.demand_scenario],
        "task_ids": tasks["task_id"].tolist(),
    }


# --------------------------------------------------------------- schedule

def _shape_recommendation(result: FullScheduleResult) -> dict:
    return {
        "status": result.status,
        "solve_time_s": result.solve_time_s,
        "objective_value": result.objective_value,
        "option_counts": result.option_counts,
        "schedule": _df_records(result.schedule),
        "partial": _df_records(result.partial),
        "unscheduled_task_ids": result.unscheduled["task_id"].tolist() if not result.unscheduled.empty else [],
        "windows": _df_records(result.windows),
    }


@app.post("/schedule/recommend")
def recommend(req: RecommendRequest, session_id: str = Depends(get_session_id)):
    ctx = get_corridor_context()
    store = get_store()
    pending = store.non_approved_requests_df(session_id)
    if pending.empty:
        raise HTTPException(400, "no waiting-list requests to schedule -- submit requests via POST /requests or POST /demo/seed first")

    # Session 13: seeded deterministically from start_date, not left random
    # -- an unseeded goods forecast meant "Generate Recommendation" could
    # show a different completion result on every click for the exact same
    # waiting-list batch, which is real synthetic variance but genuinely
    # bad for a live demo (found while diagnosing a completion-rate gap
    # between a standalone test run and the live API for the same task
    # batch). Still a real, freshly-generated synthetic forecast -- just
    # reproducible for a given date range instead of re-randomized per call.
    goods_occupancy = generate_goods_forecast(
        ctx.sections, req.start_date, n_days=req.n_days, seed=int(req.start_date.strftime("%Y%m%d"))
    )
    ranked = rank_tasks(pending, ctx.sections, ctx.passenger_occupancy, goods_occupancy, req.start_date, n_days=req.n_days)
    result = solve_schedule_with_options(
        ranked, ctx.sections, ctx.passenger_occupancy, goods_occupancy, req.start_date,
        n_days=req.n_days, time_limit_s=req.time_limit_s,
    )

    full_ids = set(result.schedule["task_id"]) if not result.schedule.empty else set()
    partial_ids = set(result.partial["task_id"]) if not result.partial.empty else set()
    for task_id in pending["task_id"]:
        if task_id in full_ids:
            store.update_request_status(task_id, "recommended_full", session_id)
        elif task_id in partial_ids:
            store.update_request_status(task_id, "recommended_partial", session_id)
        else:
            store.update_request_status(task_id, "recommended_unscheduled", session_id)

    params = {"start_date": req.start_date.isoformat(), "n_days": req.n_days}
    store.set_last_recommendation(result, pending, params, goods_occupancy, session_id)

    return _shape_recommendation(result)


@app.get("/schedule/recommendation")
def get_last_recommendation(session_id: str = Depends(get_session_id)):
    """Session 7: read-only peek at the last POST /schedule/recommend
    result, without re-triggering a solve -- the frontend has separate
    "Recommended Scheduling" (trigger + inspect) and "Pending Approval"
    (review + approve) views that both need this same data without forcing
    the user back through the trigger action just to look at it again.
    """
    store = get_store()
    rec = store.get_last_recommendation(session_id)
    if rec is None:
        return {"available": False, "recommendation": None}
    return {
        "available": True,
        "params": store.get_last_recommendation_params(session_id),
        "recommendation": _shape_recommendation(rec),
    }


def _tag_adaptive_allocation(opt: ScheduleOption, rescued_ids: set[str], demand_lookup: dict[str, float]) -> ScheduleOption:
    """Marks which of opt's scheduled/partial rows were rescued by the
    adaptive-allocation regression tier -- `adaptive_allocation: True`
    plus the task's real original `demanded_block_hours`, alongside its
    (smaller) `estimated_block_hours` -- so a reduced-allocation block is
    never indistinguishable from an ordinary full-duration completion."""
    if not rescued_ids:
        return opt
    r = opt.result

    def tag(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        df["adaptive_allocation"] = df["task_id"].isin(rescued_ids)
        df["demanded_block_hours"] = df["task_id"].map(demand_lookup)
        # `partial` rows (splittable tasks with only SOME sessions placed)
        # don't carry an estimated_block_hours column at all -- only
        # `schedule` rows do -- so the un-rescued-row fallback only
        # applies where that column actually exists.
        if "estimated_block_hours" in df.columns:
            df["demanded_block_hours"] = df["demanded_block_hours"].fillna(df["estimated_block_hours"])
        return df

    tagged_schedule, tagged_partial = tag(r.schedule), tag(r.partial)
    actually_placed = 0
    if not tagged_schedule.empty:
        actually_placed += int(tagged_schedule["adaptive_allocation"].sum())
    if not tagged_partial.empty:
        actually_placed += int(tagged_partial["adaptive_allocation"].sum())

    new_counts = dict(r.option_counts)
    new_counts["adaptive_allocation"] = actually_placed  # trimmed tasks that ended up SCHEDULED, not every trimmed task
    new_result = FullScheduleResult(
        schedule=tagged_schedule, partial=tagged_partial, unscheduled=r.unscheduled, windows=r.windows,
        status=r.status, solve_time_s=r.solve_time_s, objective_value=r.objective_value, option_counts=new_counts,
    )
    return ScheduleOption(key=opt.key, label=opt.label, description=opt.description, result=new_result, recommended=opt.recommended)


def _apply_adaptive_allocation_upfront(ranked: pd.DataFrame) -> tuple[pd.DataFrame, set[str], dict[str, float]]:
    """Session 14, per explicit user direction: run EVERY ranked task
    through the adaptive-allocation regression BEFORE it ever reaches
    CP-SAT -- not as a fallback retried only on whatever CP-SAT couldn't
    place. Every task the model offers a safe reduction for gets its
    `estimated_block_hours` replaced by that reduced value right here,
    unconditionally -- CP-SAT then only ever sees the (possibly trimmed)
    duration and has no opportunity to use the task's original full
    demand, even for a task that would have fit at full size. This was
    an explicit, deliberate choice over the alternative (let CP-SAT
    choose full vs. trimmed per task) after that tradeoff was raised
    directly -- see PROGRESS.md.

    One consequence worth being plain about: because full duration is
    never even attempted for a trimmed task, "was rescued by trimming"
    and "would have fit at full size anyway" are no longer
    distinguishable -- every task the model trims is reported as
    adaptive_allocation if it ends up scheduled, regardless of which was
    true. Returns (modified_ranked, trimmed_ids, demand_lookup) --
    `demand_lookup`: task_id -> real original demanded hours, for
    tagging the final schedule rows so a reduced allocation is never
    indistinguishable from an ordinary full-duration completion."""
    reduced = apply_adaptive_allocation(ranked)
    changed = reduced[reduced["adaptive_allocation_applied"]]
    if changed.empty:
        return ranked, set(), {}

    modified_ranked = ranked.set_index("task_id")
    modified_ranked.loc[changed["task_id"], "estimated_block_hours"] = changed.set_index("task_id")["estimated_block_hours"]
    modified_ranked = modified_ranked.reset_index()

    trimmed_ids = set(changed["task_id"])
    demand_lookup = changed.set_index("task_id")["demanded_block_hours"].to_dict()
    return modified_ranked, trimmed_ids, demand_lookup


def _combine_into_occupied_and_subtract(
    occupied_windows: pd.DataFrame,
    ranked_tasks: pd.DataFrame,
    capacity: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], pd.DataFrame]:
    """Session 26, at explicit user request, after a real reported gap:
    "whenever allocating block for a department task, you also analyze
    the previously scheduled blocks and try to combine with earlier
    scheduled other department tasks" -- covers blocks approved in a
    genuinely PAST scheduling run (store.approved).

    `occupied_windows` (approved_rows_to_window_rows()'s view of
    store.approved) is given a real chance to absorb a different-
    department task from `ranked_tasks` into its own spare "max not sum"
    room (see combine_into_occupied_windows) BEFORE `capacity` is
    conservatively reduced for it -- a plain subtract_consumed_capacity
    call would otherwise zero out that spare room using only the
    ORIGINAL occupant's own usage, forever hiding a real, valid
    combining opportunity from the solve that follows.

    Returns (remaining_ranked_tasks, capacity_after, forced_full_rows,
    updated_windows) -- merge forced_full_rows into the caller's own
    forced_full_rows list, and updated_windows into forced_windows, so
    both the finalized schedule and the frontend's combined-block modal
    reflect what actually happened here."""
    if occupied_windows.empty:
        return ranked_tasks, capacity, [], occupied_windows
    remaining, forced_rows, updated_windows = combine_into_occupied_windows(occupied_windows, ranked_tasks, capacity)
    capacity_after = subtract_consumed_capacity(capacity, updated_windows)
    return remaining, capacity_after, forced_rows, updated_windows


def _finalize_schedule(
    pending: pd.DataFrame,
    ctx: CorridorContext,
    goods_occupancy: pd.DataFrame,
    req,
    ranked: pd.DataFrame,
    trimmed_ids: set[str],
    demand_lookup: dict[str, float],
    store,
    capacity: pd.DataFrame,
    session_id: str,
    forced_full_rows: list[dict] | None = None,
    forced_windows: pd.DataFrame | None = None,
) -> dict:
    """Session 30, at explicit user request: train cancellation is out of
    scope, so the former critical-tasks-first two-solve architecture
    (Step A solving Critical alone, this function as "Step D" for
    whatever was left, plus human-confirmed escalation in between) is
    gone -- `ranked` is now the WHOLE pending pool (every priority
    together), and CP-SAT runs exactly once, here, via
    generate_schedule_options. Critical tasks compete for the same
    windows as Moderate/Routine, just with a higher whittle_index weight
    (see prioritization/whittle.py) -- a Critical task that doesn't fit
    simply stays unscheduled, same as any other priority tier.

    `forced_full_rows`/`forced_windows`: from `_combine_into_occupied_
    and_subtract` against store.approved's already-occupied windows (a
    task from a genuinely PAST scheduling run) -- still merged in here
    exactly as before, this part of the flow is unrelated to the
    critical/non-critical split that was removed. (The former partially-
    forced-task machinery, `forced_extra_sessions`, was only ever
    produced by `apply_cross_tier_combining`, also removed -- every
    remaining forcing path only ever places a task WHOLE, never
    partially, so there's nothing left to merge that way.)

    `capacity` (Session 25, at explicit user request for a large
    scheduling-time reduction): the SAME real per-(section,date)-window
    capacity table the caller already computed once -- the caller is
    responsible for it already reflecting whatever forced placements
    consumed (see `subtract_consumed_capacity`) before it reaches here."""
    options = generate_schedule_options(
        ranked, ctx.sections, ctx.passenger_occupancy, goods_occupancy, req.start_date,
        n_days=req.n_days, time_limit_s=req.time_limit_s, capacity=capacity,
    )
    if trimmed_ids:
        options = [_tag_adaptive_allocation(o, trimmed_ids, demand_lookup) for o in options]

    forced_full_df = pd.DataFrame(forced_full_rows) if forced_full_rows else pd.DataFrame()
    forced_windows = forced_windows if forced_windows is not None else pd.DataFrame()

    def _merge(opt: ScheduleOption) -> ScheduleOption:
        r = opt.result
        merged_schedule = pd.concat([r.schedule, forced_full_df], ignore_index=True) if not forced_full_df.empty else r.schedule
        merged_partial = r.partial
        covered_ids = (set(merged_schedule["task_id"]) if not merged_schedule.empty else set()) | (
            set(merged_partial["task_id"]) if not merged_partial.empty else set()
        )
        # a forced-covered id must not ALSO linger in unscheduled -- it's
        # now reflected in merged_schedule above.
        merged_unscheduled = (
            r.unscheduled[~r.unscheduled["task_id"].isin(covered_ids)] if not r.unscheduled.empty else r.unscheduled
        )
        merged_windows = pd.concat([r.windows, forced_windows], ignore_index=True) if not forced_windows.empty else r.windows
        # Recomputed fresh from the final merged frames, not incrementally
        # patched -- forced rows introduce a new "combined" option value
        # and can move a task between partial/unscheduled, so trusting
        # the pre-merge r.option_counts here would drift out of sync with
        # what merged_schedule/merged_partial/merged_unscheduled actually
        # contain.
        new_counts = dict(r.option_counts)
        new_counts.update({
            "whole": int((merged_schedule["option"] == "whole").sum()) if not merged_schedule.empty else 0,
            "split": int((merged_schedule["option"] == "split").sum()) if not merged_schedule.empty else 0,
            "combined": int((merged_schedule["option"] == "combined").sum()) if not merged_schedule.empty else 0,
            "partial": int(len(merged_partial)),
            "unscheduled": int(len(merged_unscheduled)),
        })
        new_result = FullScheduleResult(
            schedule=merged_schedule, partial=merged_partial, unscheduled=merged_unscheduled, windows=merged_windows,
            status=r.status, solve_time_s=r.solve_time_s, objective_value=r.objective_value, option_counts=new_counts,
        )
        return ScheduleOption(key=opt.key, label=opt.label, description=opt.description, result=new_result, recommended=opt.recommended)

    merged_options = [_merge(o) for o in options]
    merged_baseline = next(o for o in merged_options if o.key == "balanced")

    total_critical = int((pending["requester_priority"] == "Critical").sum())
    shaped = [_shape_option(o, len(pending), total_critical) for o in merged_options]

    params = {"start_date": req.start_date.isoformat(), "n_days": req.n_days}
    store.set_last_options(shaped, params, session_id)
    store.set_last_recommendation(merged_baseline.result, pending, params, goods_occupancy, session_id)

    full_ids = set(merged_baseline.result.schedule["task_id"]) if not merged_baseline.result.schedule.empty else set()
    partial_ids = set(merged_baseline.result.partial["task_id"]) if not merged_baseline.result.partial.empty else set()
    for task_id in pending["task_id"]:
        if task_id in full_ids:
            store.update_request_status(task_id, "recommended_full", session_id)
        elif task_id in partial_ids:
            store.update_request_status(task_id, "recommended_partial", session_id)
        else:
            store.update_request_status(task_id, "recommended_unscheduled", session_id)

    return {"params": params, "options": shaped}


def _shape_option(opt, total_tasks: int, total_critical: int) -> dict:
    summary = summarize_option(opt, total_tasks, total_critical)
    return {
        "key": opt.key,
        "label": opt.label,
        "description": opt.description,
        "recommended": opt.recommended,
        "summary": summary,
        "option_counts": opt.result.option_counts,
        "status": opt.result.status,
        **_shape_recommendation(opt.result),
    }


@app.post("/schedule/options")
def schedule_options(req: ScheduleOptionsRequest, session_id: str = Depends(get_session_id)):
    """Session 30, at explicit user request: train cancellation is out of
    scope, so the former "Critical tasks first" two-solve architecture
    (a separate, earlier CP-SAT solve for Critical alone, escalation for
    whatever didn't fit, then a second solve for everything else) is
    gone. Every pending request -- every priority -- is ranked and solved
    TOGETHER, in exactly one CP-SAT pass (inside _finalize_schedule).
    Critical still gets scheduled first in practice, purely because it
    carries a higher whittle_index weight (prioritization/whittle.py),
    not because of any special-cased solve order."""
    ctx = get_corridor_context()
    store = get_store()
    pending = store.non_approved_requests_df(session_id)
    if pending.empty:
        raise HTTPException(400, "no waiting-list requests to schedule -- submit requests via POST /requests or POST /demo/seed first")

    # Session 13: same deterministic-seeding fix as /schedule/recommend --
    # see that endpoint's comment for why.
    goods_occupancy = generate_goods_forecast(
        ctx.sections, req.start_date, n_days=req.n_days, seed=int(req.start_date.strftime("%Y%m%d"))
    )
    # Session 25, at explicit user request for a large scheduling-time
    # reduction: one real per-(section,date) availability cache, shared
    # by ranking's congestion calc and the capacity table below -- real
    # profiling showed the same expensive availability sweep otherwise
    # ran redundantly against byte-identical occupancy data. See
    # compute_availability's docstring.
    availability_cache: dict = {}
    ranked = rank_tasks(
        pending, ctx.sections, ctx.passenger_occupancy, goods_occupancy, req.start_date, n_days=req.n_days,
        availability_cache=availability_cache,
    )

    # Session 14, at explicit user request: every ranked task passes
    # through the adaptive-allocation regression BEFORE CP-SAT ever runs.
    ranked, trimmed_ids, demand_lookup = _apply_adaptive_allocation_upfront(ranked)

    capacity = compute_daily_window_capacity(
        ctx.sections, ctx.passenger_occupancy, goods_occupancy, req.start_date, req.n_days,
        availability_cache=availability_cache,
    )
    # Session 26, at explicit user request, after a real reported gap:
    # a task approved in a genuinely PAST scheduling run still occupies
    # a real window here -- give it the same chance to absorb a NEW
    # different-department task's spare "max not sum" room as anything
    # else below, before capacity is (conservatively) reduced for it.
    #
    # Session 37: _live_approved_rows (not raw store.get_approved), so a
    # task an earlier emergency already displaced is subtracted at its
    # REAL current window, not its stale original one -- otherwise this
    # would (a) subtract capacity from a slot the task no longer
    # occupies, wasting real room a new task could have used, and (b)
    # never account for the slot it actually moved into, risking a
    # genuine double-booking there.
    approved_windows = approved_rows_to_window_rows(_live_approved_rows(session_id), capacity)
    ranked, capacity, forced_full_rows, forced_windows = _combine_into_occupied_and_subtract(
        approved_windows, ranked, capacity
    )

    return _finalize_schedule(
        pending=pending, ctx=ctx, goods_occupancy=goods_occupancy, req=req,
        ranked=ranked, trimmed_ids=trimmed_ids, demand_lookup=demand_lookup,
        store=store, capacity=capacity, session_id=session_id,
        forced_full_rows=forced_full_rows, forced_windows=forced_windows,
    )


@app.get("/schedule/options")
def get_last_schedule_options(session_id: str = Depends(get_session_id)):
    """Read-only peek at the last POST /schedule/options result, mirroring
    GET /schedule/recommendation's pattern."""
    store = get_store()
    if store.get_last_options(session_id) is None:
        return {"available": False, "options": None, "params": None}
    return {
        "available": True,
        "params": store.get_last_options_params(session_id),
        "options": store.get_last_options(session_id),
    }


@app.post("/schedule/approve")
def approve(req: ApproveRequest, session_id: str = Depends(get_session_id)):
    store = get_store()

    if req.option_key:
        options = store.get_last_options(session_id)
        if not options:
            raise HTTPException(400, "no schedule options to approve from -- POST /schedule/options first")
        option = next((o for o in options if o["key"] == req.option_key), None)
        if option is None:
            raise HTTPException(404, f"option_key {req.option_key!r} not found in last /schedule/options result")
        schedule_df = pd.DataFrame(option["schedule"])
        approved_at_start_date = store.get_last_options_params(session_id)["start_date"]
    else:
        if store.get_last_recommendation(session_id) is None:
            raise HTTPException(400, "no recommendation to approve -- POST /schedule/recommend first")
        schedule_df = store.get_last_recommendation(session_id).schedule
        approved_at_start_date = store.get_last_recommendation_params(session_id)["start_date"]

    if schedule_df.empty:
        raise HTTPException(400, "nothing fully-scheduled to approve")

    if req.task_ids:
        to_approve = schedule_df[schedule_df["task_id"].isin(req.task_ids)]
        missing = set(req.task_ids) - set(to_approve["task_id"])
        if missing:
            raise HTTPException(404, f"task_ids not found in last recommendation's fully-scheduled set: {sorted(missing)}")
    else:
        to_approve = schedule_df

    approved_rows = _df_records(to_approve)
    for row in approved_rows:
        row["approved_at_start_date"] = approved_at_start_date
        if store.has_request(row["task_id"], session_id):
            store.update_request_status(row["task_id"], "approved", session_id)
    store.append_approved(approved_rows, session_id)

    return {"approved_count": len(approved_rows), "approved": approved_rows}


def _expanded_schedule_rows(session_id: str) -> list[dict]:
    """Session 28: factored out of GET /schedule/weekly so it can be
    reused as-is by the Emergency Handling flow (finding which real,
    currently-scheduled blocks an emergency displaces needs the exact
    same merged live+granted-history view the Weekly/Monthly Schedule
    page itself renders -- never a second, subtly different notion of
    "the current schedule").

    Merges store.approved (live-approved tasks from this running
    session) with the fixed granted-history dataset (railblock.synthetic.
    granted_history -- already-"approved" blocks spanning one month
    before today through the SIH evaluation window), expands each into
    one row per real (date, window) session, then (Session 30) applies
    any `emergency_reassignments` override -- a task an earlier emergency
    displaced shows at its NEW placement (or not at all, if removed with
    no slot found/declined), never at its stale original slot."""
    store = get_store()
    all_rows = store.get_approved(session_id) + _df_records(_granted_history_df(session_id))
    expanded = []
    for r in all_rows:
        sessions = r.get("sessions")
        entries = sessions if sessions else [(r["date"], r.get("window_index"), r.get("start_minute"), r.get("end_minute"))]
        for entry in entries:
            d, k = entry[0], entry[1]
            start_minute = entry[2] if len(entry) > 2 else None
            end_minute = entry[3] if len(entry) > 3 else None
            expanded.append(
                {
                    "section_id": r["section_id"],
                    "date": d,
                    "window_index": k,
                    "start_minute": start_minute,
                    "end_minute": end_minute,
                    "task_id": r["task_id"],
                    "department": r["department"],
                    "option": r.get("option"),
                    "requester_priority": r["requester_priority"],
                    "negotiated_exception": r.get("negotiated_exception", False),
                    "completion_verified": r.get("completion_verified", False),
                    # Session 28: explicit provenance -- lets a caller tell a
                    # real live-approved row apart from a granted-history
                    # synthetic one now that both are merged into this same
                    # matrix (granted-history rows already carry this field
                    # themselves; a live row never does, hence the fallback).
                    "data_source": r.get("data_source", "LIVE_APPROVED"),
                    # Session 14, at explicit user request: the real allocated
                    # duration, alongside the original demand where the
                    # adaptive-allocation regression trimmed it -- .get()
                    # with a fallback since an approved row from before this
                    # feature existed won't have demanded_block_hours/
                    # adaptive_allocation at all.
                    "estimated_block_hours": r.get("estimated_block_hours"),
                    "demanded_block_hours": r.get("demanded_block_hours", r.get("estimated_block_hours")),
                    "adaptive_allocation": r.get("adaptive_allocation", False),
                }
            )
    return _apply_emergency_overrides(expanded, session_id)


def _apply_emergency_overrides(expanded: list[dict], session_id: str) -> list[dict]:
    """Session 30: `expanded` is the per-session row list built above --
    for every task_id with an `emergency_reassignments` entry, either
    drops it entirely (removed: no alternative slot found, or the human
    declined the reschedule) or replaces its session(s) with the
    proposal's new one(s), tagging `rescheduled_due_to_emergency` so the
    frontend can show why a block moved. A task can have MULTIPLE
    sessions here if the reschedule solve had to split it (every task is
    splittable now -- see maintenance_tasks.splittable_for)."""
    reassignments = get_store().get_emergency_reassignments(session_id)
    if not reassignments:
        return expanded
    by_task: dict[str, list[dict]] = {}
    for e in expanded:
        by_task.setdefault(e["task_id"], []).append(e)

    out = []
    for task_id, entries in by_task.items():
        reassignment = reassignments.get(task_id)
        if reassignment is None:
            out.extend(entries)
            continue
        if reassignment.get("removed"):
            continue
        template = dict(entries[0])
        # Session 37 fix, after a real reported gap: `template` can be a
        # naturally-split task's own row, which carries its OWN
        # pre-reassignment `sessions` array (see splitting.py/
        # collapse_split_results) -- copying it as-is and only
        # overwriting the top-level date/start/end below would leave
        # that STALE array dangling on the new, reassigned row, showing
        # the task's OLD split sessions alongside its real NEW single
        # placement. Dropped here; each exploded row below is a clean,
        # single-session entry -- multiple such rows sharing this same
        # task_id (if the reassignment itself has multiple sessions) is
        # the correct way a reassigned multi-session placement shows up.
        template.pop("sessions", None)
        sessions = reassignment.get("sessions") or [reassignment]
        for sess in sessions:
            new_entry = dict(template)
            new_entry.update({
                "date": sess["date"],
                "window_index": sess.get("window_index"),
                "start_minute": sess.get("start_minute"),
                "end_minute": sess.get("end_minute"),
                "rescheduled_due_to_emergency": True,
                "emergency_task_id": reassignment.get("emergency_task_id"),
            })
            out.append(new_entry)
    return out


def _live_approved_rows(session_id: str) -> list[dict]:
    """Session 37, at explicit user request, after a real reported gap:
    task_history/blocks_active/blocks_upcoming used to read
    store.get_approved(session_id) directly -- only the GRANTED-HISTORY
    portion of the Dashboard/history endpoints ever went through
    _apply_reassignment_overrides_df (via _granted_history_df). A LIVE
    (not granted-history) task an emergency displaced kept showing its
    stale original date/window in Approved Tasks / Currently Active /
    Upcoming Blocks, even though the exact same override was already
    correctly applied to it in the Weekly/Monthly Schedule view (via
    _expanded_schedule_rows). _apply_emergency_overrides is generic over
    any list of dicts carrying task_id/date/window_index/start_minute/
    end_minute -- store.approved's own rows already have exactly that
    shape, so it's directly reusable here with no adaptation needed."""
    return _apply_emergency_overrides(get_store().get_approved(session_id), session_id)


def _emergency_block_rows(session_id: str) -> list[dict]:
    """One row per real (date, segment) an emergency itself occupies --
    every emergency ever created keeps showing here permanently (it's a
    real historical/current fact once created, regardless of whether its
    reschedule proposal was later approved or discarded)."""
    rows = []
    for emergency in get_store().get_emergencies(session_id):
        for seg in emergency["segments"]:
            rows.append({
                "section_id": emergency["section_id"],
                "date": seg["date"],
                "window_index": None,
                "start_minute": seg["start_minute"],
                "end_minute": seg["end_minute"],
                "task_id": emergency["task_id"],
                "department": emergency["department"],
                "option": "emergency",
                "requester_priority": None,
                "negotiated_exception": False,
                "completion_verified": False,
                "data_source": "EMERGENCY",
                "estimated_block_hours": emergency["estimated_block_hours"],
                "demanded_block_hours": emergency["estimated_block_hours"],
                "adaptive_allocation": False,
                "is_emergency": True,
            })
    return rows


@app.get("/schedule/weekly")
def weekly_matrix(session_id: str = Depends(get_session_id)):
    """Session 28, at explicit user request, after a real reported gap:
    this used to read ONLY store.approved (live-approved tasks from this
    running session), so the fixed granted-history dataset never showed
    up on the Weekly/Monthly Schedule page at all, even though the exact
    same dataset already powers the Dashboard's Currently Active/Upcoming
    sections and the Completed History page. Merges both sources now,
    same as those other endpoints already do -- the WHOLE granted-history
    set (past AND future rows), not just the not-yet-completed subset
    /tasks/history uses, since this is a real calendar view where a past
    date should show what already happened.

    Session 30: also includes every emergency's own block (see
    _emergency_block_rows) -- this is how "in the weekly and monthly
    schedule, the time at when the emergency task was created and till
    the end of the emergency situation, it should show emergency
    situation" is satisfied, in the exact same grid every other block
    already renders in."""
    expanded = _expanded_schedule_rows(session_id) + _emergency_block_rows(session_id)
    if not expanded:
        return {"dates": [], "sections": [], "cells": {}}

    dates = sorted({e["date"] for e in expanded})
    sections = sorted({e["section_id"] for e in expanded})
    cells: dict = {s: {d: [] for d in dates} for s in sections}
    for e in expanded:
        cells[e["section_id"]][e["date"]].append(e)
    return {"dates": dates, "sections": sections, "cells": cells}


@app.get("/schedule/monthly")
def monthly_matrix(
    start_date: date, n_weeks: int = 4, time_limit_s: float = 30.0, session_id: str = Depends(get_session_id)
):
    ctx = get_corridor_context()
    all_tasks = get_store().all_requests_df(session_id)
    if all_tasks.empty:
        return {"weeks": list(range(1, n_weeks + 1)), "sections": [], "cells": {}, "status": "NO_REQUESTS"}

    n_days = n_weeks * 7
    # Session 13: same deterministic-seeding fix as /schedule/recommend.
    goods_occupancy = generate_goods_forecast(ctx.sections, start_date, n_days=n_days, seed=int(start_date.strftime("%Y%m%d")))
    ranked = rank_tasks(all_tasks, ctx.sections, ctx.passenger_occupancy, goods_occupancy, start_date, n_days=n_days)
    result = generate_monthly_plan(
        ranked, ctx.sections, ctx.passenger_occupancy, goods_occupancy, start_date, n_weeks=n_weeks, time_limit_s=time_limit_s
    )

    targets = result.weekly_targets
    if targets.empty:
        return {"weeks": list(range(1, n_weeks + 1)), "sections": [], "cells": {}, "status": result.status}

    weeks = list(range(1, n_weeks + 1))
    sections = sorted(targets["section_id"].unique().tolist())
    cells = {s: {w: 0.0 for w in weeks} for s in sections}
    for _, row in targets.iterrows():
        cells[row["section_id"]][int(row["week_number"])] = row["target_hours"]
    return {"weeks": weeks, "sections": sections, "cells": cells, "status": result.status}


# ------------------------------------------------------------------ history

def _granted_history_df(session_id: str) -> pd.DataFrame:
    """Session 30: overridden by any emergency reassignment before
    classification, so a granted-history task an emergency displaced
    classifies (completed/active/upcoming) by its real NEW window, or
    disappears entirely if removed with no slot found/declined -- never
    still shown "active" at a slot it's no longer actually in. (Scoped to
    granted-history rows only, not live store.approved ones -- the bulk
    of pre-existing schedule data an emergency demo interacts with.)

    Session 40: one extra row from demo_active_row(), computed fresh
    against the real current moment on every call (never baked into the
    cached, static load_granted_history() result -- see that function's
    own docstring for why it would go stale), appended BEFORE the
    reassignment override step above so it's just as eligible to be
    displaced by an emergency as any other row here."""
    ctx = get_corridor_context()
    base = pd.concat([load_granted_history(), pd.DataFrame([demo_active_row(ctx.sections, now_ist())])], ignore_index=True)
    return _apply_reassignment_overrides_df(base, session_id)


def _apply_reassignment_overrides_df(df: pd.DataFrame, session_id: str) -> pd.DataFrame:
    reassignments = get_store().get_emergency_reassignments(session_id)
    if df.empty or not reassignments or "task_id" not in df.columns:
        return df
    df = df.copy()
    removed_ids = {tid for tid, r in reassignments.items() if r.get("removed")}
    df = df[~df["task_id"].isin(removed_ids)]
    if not {"start_minute", "end_minute", "date"}.issubset(df.columns):
        return df
    # Session 30 fix, after a real reported crash: granted-history's own
    # start_minute/end_minute are whole-minute ints (see granted_history.
    # py's _place_in_free_time), but a reassignment's new placement can
    # be a real fractional minute (the reschedule solve's own float
    # arithmetic) -- assigning a float into an int64 column via .loc
    # raises TypeError instead of silently truncating. Widen to float64
    # first, same fix combining.py's force_combinable_placements already
    # applies to window_minutes for the identical int/float mixing reason.
    df["start_minute"] = df["start_minute"].astype(float)
    df["end_minute"] = df["end_minute"].astype(float)
    present_ids = set(df["task_id"])
    reassigned_ids = {tid for tid in reassignments if tid in present_ids and not reassignments[tid].get("removed")}
    if not reassigned_ids:
        return df

    # Session 37 fix, at explicit user request, after a real reported
    # gap ("the rescheduled tasks should show their updated schedule
    # date, time, sessions"): the old in-place `.loc[mask, col] = value`
    # only ever wrote the FIRST session of a reassignment onto the
    # single existing row -- a granted-history task the reschedule solve
    # had to SPLIT across multiple sessions silently lost every session
    # past the first, and never got a `rescheduled_due_to_emergency` tag
    # at all (unlike a live store.approved task via
    # _apply_emergency_overrides, which already explodes multi-session
    # reassignments into one row per session). Rebuilt the same way:
    # one output row per real session, tagged consistently.
    kept = df[~df["task_id"].isin(reassigned_ids)].to_dict("records")
    new_rows = []
    for tid in reassigned_ids:
        r = reassignments[tid]
        template = df[df["task_id"] == tid].iloc[0].to_dict()
        sessions = r.get("sessions") or [r]
        for sess in sessions:
            row = dict(template)
            row["date"] = sess["date"]
            row["start_minute"] = sess.get("start_minute")
            row["end_minute"] = sess.get("end_minute")
            row["rescheduled_due_to_emergency"] = True
            row["emergency_task_id"] = r.get("emergency_task_id")
            new_rows.append(row)
    return pd.DataFrame(kept + new_rows)


def _emergency_blocks_df(session_id: str) -> pd.DataFrame:
    rows = _emergency_block_rows(session_id)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["task_id", "section_id", "date", "start_minute", "end_minute"])


@app.get("/tasks/history")
def task_history(session_id: str = Depends(get_session_id)):
    """Session 17: now also includes the fixed granted-history dataset's
    NOT-YET-COMPLETED rows (currently active or still-upcoming granted
    blocks), merged alongside the live store.approved rows -- an
    already-granted block that hasn't finished yet belongs here, not in
    GET /tasks/history/completed, exactly as with any live approved task.
    Each row's real data_source field (USER_SUBMITTED/FILE_IMPORTED/
    SYNTHETIC for live rows, GRANTED_HISTORY_SYNTHETIC for these) makes
    the provenance honest and inspectable, never silently blended.

    Session 37, at explicit user request, after a real reported gap:
    _live_approved_rows (not raw store.get_approved) -- a live task an
    emergency displaced must show its real CURRENT window here, the
    same way a granted-history one already did via _granted_history_df."""
    live = _live_approved_rows(session_id)
    classified = classify_blocks_by_time(_granted_history_df(session_id))
    still_relevant = pd.concat([classified["active"], classified["upcoming"]], ignore_index=True)
    merged = live + _df_records(still_relevant)
    return {"count": len(merged), "tasks": merged}


@app.get("/tasks/history/completed")
def task_history_completed(session_id: str = Depends(get_session_id)):
    """Session 17: the fixed granted-history dataset's rows whose real
    grant window has already ended, relative to real current time -- the
    Completed History page. Live store.approved rows are deliberately NOT
    included here (this project has no real "block execution confirmed
    complete" signal for a live-approved task -- see completion_verified's
    existing honest always-False handling elsewhere)."""
    classified = classify_blocks_by_time(_granted_history_df(session_id))
    completed = _df_records(classified["completed"])
    return {"count": len(completed), "tasks": completed}


@app.get("/blocks/active")
def blocks_active(session_id: str = Depends(get_session_id)):
    """Session 17, for the Dashboard's Currently Active Blocks section:
    every block (live-approved, granted-history, or -- Session 30 -- an
    emergency) whose real window (date, start_minute, end_minute)
    contains right now. Negotiated-exception rows (no modeled start/end)
    can never appear here -- see classify_blocks_by_time's docstring.

    Session 37: _live_approved_rows, same reasoning as task_history."""
    live_active = classify_blocks_by_time(pd.DataFrame(_live_approved_rows(session_id)))["active"]
    hist_active = classify_blocks_by_time(_granted_history_df(session_id))["active"]
    emrg_active = classify_blocks_by_time(_emergency_blocks_df(session_id))["active"]
    merged = pd.concat([live_active, hist_active, emrg_active], ignore_index=True)
    rows = _df_records(merged)
    rows.sort(key=lambda r: (r["date"], r["start_minute"]))
    return {"count": len(rows), "blocks": rows}


@app.get("/blocks/upcoming")
def blocks_upcoming(limit: int = Query(5, ge=1, le=50), session_id: str = Depends(get_session_id)):
    """Session 17, for the Dashboard's Upcoming Blocks section: the next
    `limit` blocks (live-approved, granted-history, or -- Session 30 -- an
    emergency) whose real window hasn't started yet, soonest first.

    Session 37: _live_approved_rows, same reasoning as task_history."""
    live_upcoming = classify_blocks_by_time(pd.DataFrame(_live_approved_rows(session_id)))["upcoming"]
    hist_upcoming = classify_blocks_by_time(_granted_history_df(session_id))["upcoming"]
    emrg_upcoming = classify_blocks_by_time(_emergency_blocks_df(session_id))["upcoming"]
    merged = pd.concat([live_upcoming, hist_upcoming, emrg_upcoming], ignore_index=True)
    rows = _df_records(merged)
    rows.sort(key=lambda r: (r["date"], r["start_minute"]))
    return {"count": len(rows[:limit]), "blocks": rows[:limit]}


# --------------------------------------------------------------- emergency

@app.post("/emergency/create")
def emergency_create(session_id: str = Depends(get_session_id)):
    """Session 30, at explicit user request: creates a brand-new demo
    emergency EVERY call (no dedup, no reuse) -- an instantly-placed,
    UNAPPROVED block that occupies real time on a section right now, no
    approval needed ("a demo emergency situation has to be done as fast
    as possible"). Immediately computes which already-scheduled blocks it
    displaces (any section -- an incident can cause trains to be
    stopped/rerouted elsewhere too) and a PROPOSED single-option
    reschedule for each (never auto-applied -- see POST /emergency/
    {task_id}/resolve), so the human can review before anything actually
    moves."""
    ctx = get_corridor_context()
    store = get_store()
    now = now_ist()
    rng = np.random.default_rng()

    existing_rows = _expanded_schedule_rows(session_id)
    emergency = create_demo_emergency(ctx.sections, existing_rows, now, rng, store.next_emergency_id(session_id))

    affected_rows = find_affected_rows(emergency, existing_rows, now)
    enriched = []
    for row in affected_rows:
        detail = _lookup_task_detail(row["task_id"], session_id)
        if detail is None:
            continue
        enriched.append({
            **row,
            "defect_type": detail.get("defect_type"),
            "raised_date": detail.get("raised_date") or now.date().isoformat(),
            "due_date": detail.get("due_date") or (now.date() + timedelta(days=14)).isoformat(),
            "days_overdue": detail.get("days_overdue") or 0,
            "splittable": detail.get("splittable", True),
            "approval_path": detail.get("approval_path", "sm_direct"),
        })

    affected_task_ids = {t["task_id"] for t in enriched}
    # Session 37: _live_approved_rows, so a task an EARLIER emergency
    # already relocated is treated as occupying its real current window
    # here too, not the stale original one this row still carries in
    # store.approved itself.
    other_occupied_rows = [r for r in _live_approved_rows(session_id) if r["task_id"] not in affected_task_ids]

    proposals = propose_reschedule(
        enriched, ctx.sections, ctx.passenger_occupancy, emergency, other_occupied_rows,
        now.date(), now,
    )

    affected_payload = [
        {
            "task_id": p.task_id,
            "department": next(t["department"] for t in enriched if t["task_id"] == p.task_id),
            "section_id": next(t["section_id"] for t in enriched if t["task_id"] == p.task_id),
            "old": p.old,
            "proposed": p.proposed,
        }
        for p in proposals
    ]

    emergency["affected"] = affected_payload
    store.add_emergency(emergency, session_id)

    return {"emergency": emergency, "affected": affected_payload}


@app.post("/emergency/{task_id}/resolve")
def emergency_resolve(task_id: str, req: EmergencyResolveRequest, session_id: str = Depends(get_session_id)):
    """Applies (or discards) the reschedule proposal POST /emergency/
    create already computed for this emergency. On apply, every affected
    task with a real proposed slot gets it via `emergency_reassignments`
    (consulted by GET /schedule/weekly and the history/Dashboard
    endpoints); one with no slot found is marked removed -- it genuinely
    can't happen where it was, so its stale old slot is vacated either
    way, approved or not."""
    store = get_store()
    emergency = store.get_emergency(task_id, session_id)
    if emergency is None:
        raise HTTPException(404, f"no emergency with task_id {task_id!r}")

    for a in emergency.get("affected", []):
        if req.apply and a["proposed"] is not None:
            store.set_emergency_reassignment(a["task_id"], {**a["proposed"], "emergency_task_id": task_id}, session_id)
        else:
            store.set_emergency_reassignment(a["task_id"], {"removed": True, "emergency_task_id": task_id}, session_id)

    emergency["status"] = "resolved" if req.apply else "discarded"
    store.update_emergency(task_id, emergency, session_id)
    return {"task_id": task_id, "status": emergency["status"]}


@app.get("/emergency/list")
def emergency_list(session_id: str = Depends(get_session_id)):
    return {"emergencies": get_store().get_emergencies(session_id)}
