"""Batch-import block/maintenance task requests from a CSV
file. This replaces a one-at-a-time "Raise Block Request" web form: real
COA departments deliver batch task lists rather than submitting requests
individually, and this models that directly instead of faking a live
form.

Shares its validation and derived-field logic with POST /requests
(see railblock.api.request_builder.build_request_row) so a row imported
this way is indistinguishable in the database from one that came in
through the API directly -- same required fields, same checks, same
defaulting -- except its `data_source` is honestly recorded as
FILE_IMPORTED rather than USER_SUBMITTED, since that's what actually
happened.

Rows land in the SHARED, unscoped `demo_task_templates` table
(store.add_template_task) rather than any particular visitor's own
`requests` -- each browser gets its own session-scoped view, and a task
appended to tasks.csv on the server should become available for the
NEXT "Create demo batch of tasks" click (store.activate_demo_batch), not
silently injected into whichever browser session happens to be open at
that moment.

Expected CSV columns (header row required, exact names):
    department, section_id, defect_type, requester_priority,
    estimated_block_hours, due_date, raised_date

`raised_date` is optional -- leave the cell empty to default to today.
`due_date` and `raised_date` must be ISO format (YYYY-MM-DD).

Every row is validated independently and NEVER silently skipped or
guessed at: an invalid row is reported with its exact reason and row
number, and no row is inserted until it passes every check real
POST /requests would apply.

Manual one-shot run:
    python -m railblock.integrations.import_tasks path/to/tasks.csv

Live-watched (automatic): the backend runs check_for_new_rows() against
TASKS_CSV_WATCH_PATH on a short poll loop (see api/app.py's startup
hook) -- appending a row to that file and saving picks it up within a
couple of seconds, no manual re-run needed. This is deliberately a
row-COUNT-based append-only model, not a diff: it tracks how many data
rows have already been imported (persisted in
TASKS_CSV_IMPORT_STATE_JSON so it survives a backend restart) and only
processes rows past that point. Editing or reordering an already-imported
row won't be picked up -- only genuinely new rows appended at the end.
Polling rather than a filesystem-event watcher (e.g. `watchdog`/inotify)
was a deliberate choice: this project runs under WSL, where inotify
events don't reliably fire for files touched from the Windows side of a
cross-boundary mount -- a short poll interval has no such gap and needs
no extra dependency.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import date

from pydantic import ValidationError

from railblock.api.request_builder import RequestValidationError, build_request_row
from railblock.api.schemas import BlockRequestIn
from railblock.api.store import get_store
from railblock.paths import TASKS_CSV_IMPORT_STATE_JSON, TASKS_CSV_WATCH_PATH


def _parse_date(value: str, field: str) -> date:
    v = value.strip()
    try:
        return date.fromisoformat(v)
    except ValueError:
        pass
    # Also accept DD-MM-YYYY (the natural format for an Indian date entry,
    # confirmed a real user actually used it) -- tried second, never
    # silently guessed: if neither format parses, this still raises.
    try:
        from datetime import datetime

        return datetime.strptime(v, "%d-%m-%Y").date()
    except ValueError as exc:
        raise ValueError(f"{field} must be YYYY-MM-DD or DD-MM-YYYY, got {value!r}") from exc


def _import_rows(rows: list[dict], start_line_no: int) -> tuple[int, int]:
    """Shared by both the one-shot CLI import and the incremental watcher.
    `start_line_no` is only used for REJECTED's printed line number."""
    imported, failed = 0, 0
    for offset, raw_row in enumerate(rows):
        line_no = start_line_no + offset
        try:
            due_date = _parse_date(raw_row["due_date"], "due_date")
            raised_date = (
                _parse_date(raw_row["raised_date"], "raised_date")
                if raw_row.get("raised_date", "").strip()
                else None
            )
            req = BlockRequestIn(
                department=raw_row["department"].strip(),
                section_id=raw_row["section_id"].strip(),
                defect_type=raw_row["defect_type"].strip(),
                requester_priority=raw_row["requester_priority"].strip(),
                estimated_block_hours=float(raw_row["estimated_block_hours"]),
                due_date=due_date,
                raised_date=raised_date,
            )
            row = build_request_row(req, data_source="FILE_IMPORTED")
        except (KeyError, ValueError, ValidationError, RequestValidationError) as exc:
            print(f"  line {line_no}: REJECTED -- {exc}")
            failed += 1
            continue

        get_store().add_template_task(row)
        print(f"  line {line_no}: imported as {row['task_id']}")
        imported += 1

    return imported, failed


def import_csv(path: str) -> tuple[int, int]:
    """One-shot: import every row in the file. Returns (imported_count,
    failed_count). Never raises for a bad row -- only for a genuinely
    unreadable file."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return _import_rows(rows, start_line_no=2)  # header is line 1


def _load_import_state() -> dict:
    if TASKS_CSV_IMPORT_STATE_JSON.exists():
        with open(TASKS_CSV_IMPORT_STATE_JSON) as f:
            return json.load(f)
    return {"rows_imported": 0}


def _save_import_state(state: dict) -> None:
    with open(TASKS_CSV_IMPORT_STATE_JSON, "w") as f:
        json.dump(state, f, indent=2)


def check_for_new_rows(path=TASKS_CSV_WATCH_PATH) -> tuple[int, int]:
    """Import only the data rows appended since the last check (tracked
    persistently). Safe to call repeatedly/on a poll loop -- a no-op
    (0, 0) if the file doesn't exist yet or has no new rows. Returns
    (imported_count, failed_count) for whatever was newly found this
    call."""
    if not path.exists():
        return 0, 0

    with open(path, newline="") as f:
        all_rows = list(csv.DictReader(f))

    state = _load_import_state()
    already_imported = state.get("rows_imported", 0)
    new_rows = all_rows[already_imported:]
    if not new_rows:
        return 0, 0

    imported, failed = _import_rows(new_rows, start_line_no=already_imported + 2)
    _save_import_state({"rows_imported": len(all_rows)})
    return imported, failed


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python -m railblock.integrations.import_tasks path/to/tasks.csv")
        sys.exit(1)

    imported, failed = import_csv(sys.argv[1])
    print(f"\nDone: {imported} imported, {failed} rejected.")


if __name__ == "__main__":
    main()
