"""Application state for the API layer -- PostgreSQL-backed (see
railblock.config.DATABASE_URL), so submitted requests, the last
recommendation, and the approved schedule survive a process restart. The
corridor structure cache (`CorridorContext`) is deliberately kept
separate from this: it is real, expensive-to-derive, read-only,
never-changing reference data, not mutable application state, so it stays
a simple in-process cache rather than living in the database.

`last_recommendation` (a `FullScheduleResult` of DataFrames) has no
natural relational shape, so it is persisted as one JSON blob per
recommend call in a `recommendations` table -- a completely standard way
to persist a computed, non-tabular result. Reading it back returns a
`FullScheduleResult`-shaped object (DataFrames reconstructed from the
stored records), good enough for every existing consumer
(`_shape_recommendation`, etc.) without changing
their code.

Requires a running PostgreSQL server -- there is no in-process/embedded
fallback. Tests connect to a separate `railblock_test` database (see
reset_store_for_tests) so a `pytest` run never touches real demo data.

EVERY mutable table below is scoped by a `session_id` column, one per
visitor's browser (see api/app.py's `get_session_id` dependency and
frontend/src/api/session.js): since this app is hosted publicly behind a
single shared link, without per-session scoping every visitor would see
and mutate the exact same waiting list/approved schedule/emergencies as
everyone else. A request with no session header at all (a script or curl
call bypassing the frontend) is treated as one shared
`DEFAULT_SESSION_ID` rather than erroring.

The one exception is `demo_task_templates`: the FIXED pool of real,
file-imported tasks (see railblock.integrations.import_tasks) that "Create
demo batch of tasks" copies from -- this is shared, read-only-per-visitor
reference data, not any one visitor's own mutable state, so it deliberately
stays unscoped, the same reasoning that keeps `CorridorContext` unscoped.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass

import pandas as pd
import psycopg
from psycopg.rows import dict_row

from railblock.availability.corridor_availability import build_passenger_occupancy
from railblock.config import DATABASE_URL
from railblock.corridor.derive_stations import load_timetable
from railblock.corridor.fine_stations import load_fine_corridor_stations
from railblock.corridor.real_overrides import apply_real_train_overrides
from railblock.corridor.sections import build_block_sections
from railblock.scheduling.orchestrator import FullScheduleResult
from railblock.synthetic.granted_history import now_ist
from railblock.synthetic.maintenance_tasks import SOURCE_SYSTEM, refresh_demo_batch_dates

_id_counter = itertools.count(1)

DEFAULT_SESSION_ID = "default-session"


@dataclass
class CorridorContext:
    stations: pd.DataFrame
    sections: pd.DataFrame
    passenger_occupancy: pd.DataFrame


_corridor_context: CorridorContext | None = None


def get_corridor_context() -> CorridorContext:
    """Lazily load and cache the corridor structure + real passenger
    occupancy once per process -- see module docstring's performance note.
    """
    global _corridor_context
    if _corridor_context is None:
        stations = load_fine_corridor_stations()
        sections = build_block_sections(stations)
        timetable, _ = load_timetable()
        timetable = apply_real_train_overrides(timetable)
        passenger_occupancy = build_passenger_occupancy(timetable, stations)
        _corridor_context = CorridorContext(stations, sections, passenger_occupancy)
    return _corridor_context


def next_request_id(department: str) -> str:
    return f"{SOURCE_SYSTEM[department]}-REQ-{next(_id_counter):05d}"


_SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS requests (
        session_id TEXT NOT NULL DEFAULT 'default-session',
        task_id TEXT NOT NULL,
        status TEXT NOT NULL,
        data_json TEXT NOT NULL,
        PRIMARY KEY (session_id, task_id)
    )
    """,
    # The Waiting List shows the most-recently-submitted task first. The
    # table has no natural insertion-order column (task_id's per-department
    # counter doesn't give a single global order across departments), so
    # this adds one -- IF NOT EXISTS makes it a safe no-op on a fresh
    # database and a real migration on an existing one. Never touched on
    # conflict-update (see add_request), so an existing row's original
    # insertion time is never bumped just because its status changed
    # later.
    "ALTER TABLE requests ADD COLUMN IF NOT EXISTS inserted_at TIMESTAMPTZ NOT NULL DEFAULT NOW()",
    """
    CREATE TABLE IF NOT EXISTS recommendations (
        id SERIAL PRIMARY KEY,
        session_id TEXT NOT NULL DEFAULT 'default-session',
        created_at TIMESTAMPTZ NOT NULL,
        params_json TEXT NOT NULL,
        tasks_json TEXT NOT NULL,
        result_json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS approved (
        id SERIAL PRIMARY KEY,
        session_id TEXT NOT NULL DEFAULT 'default-session',
        task_id TEXT NOT NULL,
        row_json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS schedule_options (
        id SERIAL PRIMARY KEY,
        session_id TEXT NOT NULL DEFAULT 'default-session',
        created_at TIMESTAMPTZ NOT NULL,
        params_json TEXT NOT NULL,
        options_json TEXT NOT NULL
    )
    """,
    # Emergency Handling. `emergencies` holds one row per demo emergency
    # ever created (never overwritten except to update its own
    # status/embedded reschedule proposal) -- `inserted_at` gives a stable
    # creation order since a session can create several.
    # `emergency_reassignments` holds, per DISPLACED task_id, either its
    # new placement or {"removed": true} (no slot found, or the human
    # declined) -- consulted by GET /schedule/weekly and friends so a
    # moved/removed block never keeps showing at its old, now-invalid
    # slot. See railblock.scheduling.emergency.
    """
    CREATE TABLE IF NOT EXISTS emergencies (
        session_id TEXT NOT NULL DEFAULT 'default-session',
        task_id TEXT NOT NULL,
        row_json TEXT NOT NULL,
        inserted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (session_id, task_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS emergency_reassignments (
        session_id TEXT NOT NULL DEFAULT 'default-session',
        task_id TEXT NOT NULL,
        row_json TEXT NOT NULL,
        PRIMARY KEY (session_id, task_id)
    )
    """,
    # The shared, unscoped pool of real file-imported tasks (see
    # railblock.integrations.import_tasks) that "Create demo batch of
    # tasks" copies from into a specific visitor's own `requests` rows --
    # see module docstring.
    """
    CREATE TABLE IF NOT EXISTS demo_task_templates (
        task_id TEXT PRIMARY KEY,
        data_json TEXT NOT NULL,
        inserted_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
]


def _df_to_json(df: pd.DataFrame) -> str:
    if df is None:
        return "null"
    return df.astype(object).where(df.notna(), None).to_json(orient="records")


def _json_to_df(text: str) -> pd.DataFrame:
    if text is None or text == "null":
        return pd.DataFrame()
    records = json.loads(text)
    return pd.DataFrame(records)


class BlockRequestStore:
    """PostgreSQL-backed store. One persistent `autocommit` connection per
    instance (matches the prior SQLite store's single-connection design;
    this is a low-traffic local app, not a pool-worthy service)."""

    def __init__(self, dsn: str = DATABASE_URL):
        self.dsn = dsn
        self._conn = psycopg.connect(dsn, autocommit=True, row_factory=dict_row)
        with self._conn.cursor() as cur:
            for statement in _SCHEMA_STATEMENTS:
                cur.execute(statement)

    # ------------------------------------------------------------ requests

    def add_request(self, row: dict, session_id: str) -> None:
        row = dict(row)
        row.setdefault("status", "pending")
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO requests (session_id, task_id, status, data_json) VALUES (%s, %s, %s, %s)
                ON CONFLICT (session_id, task_id) DO UPDATE SET status = EXCLUDED.status, data_json = EXCLUDED.data_json
                """,
                (session_id, row["task_id"], row["status"], json.dumps(row, default=str)),
            )

    def get_request(self, task_id: str, session_id: str) -> dict | None:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT data_json FROM requests WHERE session_id = %s AND task_id = %s", (session_id, task_id)
            )
            row = cur.fetchone()
        return json.loads(row["data_json"]) if row else None

    def has_request(self, task_id: str, session_id: str) -> bool:
        return self.get_request(task_id, session_id) is not None

    def update_request_status(self, task_id: str, status: str, session_id: str) -> None:
        row = self.get_request(task_id, session_id)
        if row is None:
            return
        row["status"] = status
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE requests SET status = %s, data_json = %s WHERE session_id = %s AND task_id = %s",
                (status, json.dumps(row, default=str), session_id, task_id),
            )

    def delete_pending_requests(self, task_ids: list[str], session_id: str) -> int:
        """Bulk-delete, but ONLY rows still status='pending' -- a
        demo-seeded batch the frontend is retracting on page reload (see
        RaiseRequest.jsx) must never delete a task that has since been
        recommended/approved."""
        if not task_ids:
            return 0
        with self._conn.cursor() as cur:
            cur.execute(
                "DELETE FROM requests WHERE status = 'pending' AND session_id = %s AND task_id = ANY(%s)",
                (session_id, task_ids),
            )
            return cur.rowcount

    def pending_requests_df(self, session_id: str) -> pd.DataFrame:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT data_json FROM requests WHERE session_id = %s AND status = 'pending'", (session_id,)
            )
            rows = [json.loads(r["data_json"]) for r in cur.fetchall()]
        return pd.DataFrame(rows)

    def non_approved_requests_df(self, session_id: str) -> pd.DataFrame:
        """Every request still in the Waiting List, whatever its status --
        pending, recommended_full, recommended_partial, or
        recommended_unscheduled. A new recommend/options run must consider
        ALL of these, not just status='pending': a task an earlier run
        left as recommended_unscheduled (a fuller corridor that week, a
        different competing pool) genuinely might fit this run, and one
        left recommended_full but never approved must stay reachable too
        -- otherwise it is silently frozen out of every future run the
        moment its status first changes away from 'pending'.

        'rejected' is excluded too, same as 'approved' -- once COA
        explicitly rejects a request, it's a final decision, not a pending
        one; it must not keep reappearing in the Waiting List or be
        considered by the next recommend/options run."""
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT data_json FROM requests WHERE session_id = %s AND status NOT IN ('approved', 'rejected')",
                (session_id,),
            )
            rows = [json.loads(r["data_json"]) for r in cur.fetchall()]
        return pd.DataFrame(rows)

    def all_requests_df(self, session_id: str) -> pd.DataFrame:
        # Newly-raised-first by each request's own real raised_date, not
        # DB insertion order -- a bulk-activated demo batch inserts all 70
        # rows in template order within the same instant, which has no
        # relationship to which task was actually raised most recently in
        # the real data. inserted_at DESC (most-recently-inserted first)
        # is kept as the query order so ties on raised_date fall back to
        # it via the stable sort below.
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT data_json FROM requests WHERE session_id = %s ORDER BY inserted_at DESC", (session_id,)
            )
            rows = [json.loads(r["data_json"]) for r in cur.fetchall()]
        df = pd.DataFrame(rows)
        if not df.empty and "raised_date" in df.columns:
            df = df.sort_values("raised_date", ascending=False, kind="stable").reset_index(drop=True)
        return df

    # ------------------------------------------------------- recommendation

    def set_last_recommendation(
        self, result: FullScheduleResult, tasks_df: pd.DataFrame, params: dict, goods_occupancy: pd.DataFrame,
        session_id: str,
    ) -> None:
        result_json = json.dumps(
            {
                "schedule": _df_to_json(result.schedule),
                "partial": _df_to_json(result.partial),
                "unscheduled": _df_to_json(result.unscheduled),
                "windows": _df_to_json(result.windows),
                "status": result.status,
                "solve_time_s": result.solve_time_s,
                "objective_value": result.objective_value,
                "option_counts": result.option_counts,
                "goods_occupancy": _df_to_json(goods_occupancy),
            }
        )
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO recommendations (session_id, created_at, params_json, tasks_json, result_json) "
                "VALUES (%s, NOW(), %s, %s, %s)",
                (session_id, json.dumps(params), _df_to_json(tasks_df), result_json),
            )

    def get_last_recommendation(self, session_id: str) -> FullScheduleResult | None:
        row = self._latest_recommendation_row(session_id)
        if row is None:
            return None
        r = json.loads(row["result_json"])
        return FullScheduleResult(
            schedule=_json_to_df(r["schedule"]),
            partial=_json_to_df(r["partial"]),
            unscheduled=_json_to_df(r["unscheduled"]),
            windows=_json_to_df(r["windows"]),
            status=r["status"],
            solve_time_s=r["solve_time_s"],
            objective_value=r["objective_value"],
            option_counts=r["option_counts"],
        )

    def get_last_recommendation_tasks(self, session_id: str) -> pd.DataFrame | None:
        row = self._latest_recommendation_row(session_id)
        return _json_to_df(row["tasks_json"]) if row else None

    def get_last_recommendation_params(self, session_id: str) -> dict | None:
        row = self._latest_recommendation_row(session_id)
        return json.loads(row["params_json"]) if row else None

    def get_last_goods_occupancy(self, session_id: str) -> pd.DataFrame | None:
        row = self._latest_recommendation_row(session_id)
        if row is None:
            return None
        r = json.loads(row["result_json"])
        return _json_to_df(r["goods_occupancy"])

    def _latest_recommendation_row(self, session_id: str) -> dict | None:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT params_json, tasks_json, result_json FROM recommendations "
                "WHERE session_id = %s ORDER BY id DESC LIMIT 1",
                (session_id,),
            )
            return cur.fetchone()

    # ------------------------------------------------------------- approved

    def append_approved(self, rows: list[dict], session_id: str) -> None:
        with self._conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO approved (session_id, task_id, row_json) VALUES (%s, %s, %s)",
                [(session_id, r["task_id"], json.dumps(r, default=str)) for r in rows],
            )

    def get_approved(self, session_id: str) -> list[dict]:
        with self._conn.cursor() as cur:
            cur.execute("SELECT row_json FROM approved WHERE session_id = %s ORDER BY id ASC", (session_id,))
            return [json.loads(r["row_json"]) for r in cur.fetchall()]

    def delete_approved(self, task_id: str, session_id: str) -> None:
        """Removes every row for this task_id from `approved` (a split
        task can legitimately have more than one, one per session) --
        used when an emergency discard returns a live task to the
        Waiting List, so its stale pre-emergency approved slot doesn't
        keep existing alongside its new 'pending' request row."""
        with self._conn.cursor() as cur:
            cur.execute(
                "DELETE FROM approved WHERE session_id = %s AND task_id = %s", (session_id, task_id)
            )

    # ---------------------------------------------------------- options

    def set_last_options(self, options_payload: list[dict], params: dict, session_id: str) -> None:
        """options_payload: already-JSON-shaped list of option dicts
        (see app.py's _shape_option) -- stored verbatim as one row."""
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO schedule_options (session_id, created_at, params_json, options_json) "
                "VALUES (%s, NOW(), %s, %s)",
                (session_id, json.dumps(params), json.dumps(options_payload)),
            )

    def get_last_options(self, session_id: str) -> list[dict] | None:
        row = self._latest_options_row(session_id)
        return json.loads(row["options_json"]) if row else None

    def get_last_options_params(self, session_id: str) -> dict | None:
        row = self._latest_options_row(session_id)
        return json.loads(row["params_json"]) if row else None

    def _latest_options_row(self, session_id: str) -> dict | None:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT params_json, options_json FROM schedule_options "
                "WHERE session_id = %s ORDER BY id DESC LIMIT 1",
                (session_id,),
            )
            return cur.fetchone()

    # ------------------------------------------------------------ emergency

    def next_emergency_id(self, session_id: str) -> str:
        """Derived from the real `emergencies` table, not an in-process
        itertools.count() like next_request_id above: an in-memory counter
        restarts from 1 on every process restart, but `emergencies` is
        PostgreSQL-backed and survives one, so a restart would collide
        with an id already INSERTed in a prior process run
        (UniqueViolation, no ON CONFLICT fallback here unlike
        add_request's upsert).

        Reads the highest REAL id that exists (MAX by lexicographic
        order, safe since every id is fixed-width zero-padded -- see the
        row_number parse below), not a COUNT(*)-based "next": COUNT drops
        the instant any row is ever deleted (e.g. clearing stale
        demo/test data) while the highest real id doesn't, so a
        COUNT-based scheme would collide with an id already taken. Reading
        the max instead stays correct regardless of any gaps left by
        deletions -- each new emergency must get a genuinely new id,
        restart or deletion or not.

        Scoped per session_id -- each visitor's own emergencies start
        fresh at EMRG-00001, since they only ever share a corridor with
        themselves."""
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT task_id FROM emergencies WHERE session_id = %s ORDER BY task_id DESC LIMIT 1",
                (session_id,),
            )
            row = cur.fetchone()
        last_n = int(row["task_id"].split("-", 1)[1]) if row else 0
        return f"EMRG-{last_n + 1:05d}"

    def add_emergency(self, row: dict, session_id: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO emergencies (session_id, task_id, row_json) VALUES (%s, %s, %s)",
                (session_id, row["task_id"], json.dumps(row, default=str)),
            )

    def update_emergency(self, task_id: str, row: dict, session_id: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE emergencies SET row_json = %s WHERE session_id = %s AND task_id = %s",
                (json.dumps(row, default=str), session_id, task_id),
            )

    def get_emergency(self, task_id: str, session_id: str) -> dict | None:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT row_json FROM emergencies WHERE session_id = %s AND task_id = %s", (session_id, task_id)
            )
            row = cur.fetchone()
        return json.loads(row["row_json"]) if row else None

    def get_emergencies(self, session_id: str) -> list[dict]:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT row_json FROM emergencies WHERE session_id = %s ORDER BY inserted_at ASC", (session_id,)
            )
            return [json.loads(r["row_json"]) for r in cur.fetchall()]

    def set_emergency_reassignment(self, task_id: str, data: dict, session_id: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO emergency_reassignments (session_id, task_id, row_json) VALUES (%s, %s, %s)
                ON CONFLICT (session_id, task_id) DO UPDATE SET row_json = EXCLUDED.row_json
                """,
                (session_id, task_id, json.dumps(data, default=str)),
            )

    def clear_emergency_reassignment(self, task_id: str, session_id: str) -> None:
        """Removes a single task_id's override, once it's no longer
        needed -- a discarded LIVE task's stale approved row is deleted
        outright (see delete_approved), so there's nothing left for the
        'removed' flag to keep suppressing; leaving it in place would
        silently hide that same task_id's NEXT, genuinely new approval
        too, since the override lookup is keyed on task_id alone with no
        way to tell 'stale' from 'current'."""
        with self._conn.cursor() as cur:
            cur.execute(
                "DELETE FROM emergency_reassignments WHERE session_id = %s AND task_id = %s", (session_id, task_id)
            )

    def get_emergency_reassignments(self, session_id: str) -> dict[str, dict]:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT task_id, row_json FROM emergency_reassignments WHERE session_id = %s", (session_id,)
            )
            return {r["task_id"]: json.loads(r["row_json"]) for r in cur.fetchall()}

    def reset_session_emergencies(self, session_id: str) -> None:
        """Emergency Handling's own "Reset" button: clears every emergency
        AND its reassignment overrides for this one visitor. Since a
        reassignment is only ever an override layered on top of a task's
        real stored data (never an overwrite -- see api/app.py's
        _apply_reassignment_overrides_df), deleting these rows alone makes
        a displaced task reappear at its original slot with no further
        "undo" step needed."""
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM emergencies WHERE session_id = %s", (session_id,))
            cur.execute("DELETE FROM emergency_reassignments WHERE session_id = %s", (session_id,))

    # ------------------------------------------------------- demo templates

    def add_template_task(self, row: dict) -> None:
        """The shared, unscoped pool import_tasks.py populates (both the
        one-shot CLI import and the live tasks.csv watcher) -- see module
        docstring. Never touched by any per-visitor action; only read
        from, by activate_demo_batch below."""
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO demo_task_templates (task_id, data_json) VALUES (%s, %s)
                ON CONFLICT (task_id) DO UPDATE SET data_json = EXCLUDED.data_json
                """,
                (row["task_id"], json.dumps(row, default=str)),
            )

    def template_tasks(self) -> list[dict]:
        with self._conn.cursor() as cur:
            cur.execute("SELECT data_json FROM demo_task_templates ORDER BY inserted_at ASC")
            return [json.loads(r["data_json"]) for r in cur.fetchall()]

    def activate_demo_batch(self, session_id: str) -> int:
        """"Create demo batch of tasks": copies the shared template pool
        into THIS visitor's own `requests` rows, status reset to 'pending'
        regardless of whatever status the template itself carries. ON
        CONFLICT makes a repeat click idempotent -- it just re-activates
        cleanly rather than erroring or duplicating.

        raised_date/due_date are refreshed to real dates anchored to
        TODAY (the moment this is clicked), not the template's own
        frozen, CSV-import-time values -- see maintenance_tasks.
        refresh_demo_batch_dates's own module note for why. Uses
        now_ist(), not a naive date.today(), same reasoning as every
        other current-time read in this project: this deployment's host
        OS clock runs UTC, but every date here is implicitly IST
        wall-clock time."""
        templates = self.template_tasks()
        if not templates:
            return 0
        templates = refresh_demo_batch_dates(templates, now_ist().date())
        with self._conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO requests (session_id, task_id, status, data_json) VALUES (%s, %s, 'pending', %s)
                ON CONFLICT (session_id, task_id) DO UPDATE SET status = 'pending', data_json = EXCLUDED.data_json
                """,
                [
                    (session_id, t["task_id"], json.dumps({**t, "status": "pending"}, default=str))
                    for t in templates
                ],
            )
        return len(templates)

    def reset_session_tasks(self, session_id: str) -> None:
        """Waiting List's own "Reset" button: clears this visitor's entire
        requests/approved/recommendation/schedule-options footprint in one
        shot. Since every table here is scoped per session_id, that
        footprint IS exactly "the tasks in this session (or whatever's
        left of them) plus anything scheduled from them" -- there is
        nothing else in these four tables for this session_id to have put
        there."""
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM requests WHERE session_id = %s", (session_id,))
            cur.execute("DELETE FROM approved WHERE session_id = %s", (session_id,))
            cur.execute("DELETE FROM recommendations WHERE session_id = %s", (session_id,))
            cur.execute("DELETE FROM schedule_options WHERE session_id = %s", (session_id,))

    # --------------------------------------------------------------- misc

    def reset(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "TRUNCATE requests, recommendations, approved, schedule_options, "
                "emergencies, emergency_reassignments, demo_task_templates RESTART IDENTITY"
            )

    def close(self) -> None:
        self._conn.close()


_store: BlockRequestStore | None = None


def get_store() -> BlockRequestStore:
    global _store
    if _store is None:
        _store = BlockRequestStore()
    return _store


def _test_database_url() -> str:
    """Same server/credentials as DATABASE_URL, but the `railblock_test`
    database instead of the real one -- created alongside it during
    PostgreSQL setup, so a `pytest` run's TRUNCATEs never touch real demo
    data."""
    return DATABASE_URL.rsplit("/", 1)[0] + "/railblock_test"


def reset_store_for_tests() -> None:
    """Test-only helper: point at the separate `railblock_test` database
    and wipe it, so tests never touch the real `railblock` database or
    leak state between test runs (the corridor context cache is left
    intact -- it's expensive and never changes)."""
    global _store
    _store = BlockRequestStore(dsn=_test_database_url())
    _store.reset()
