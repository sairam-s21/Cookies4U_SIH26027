"""Validation + derived-field logic for turning one BlockRequestIn into a
stored request row -- shared by POST /requests (app.py) and
railblock.integrations.import_tasks (the CSV/JSON batch importer that
replaced the "Raise Block Request" UI form, since real COA receives batch
task lists from TDMS/SMMS/etc., not one-at-a-time web submissions). Kept
in its own module, separate from both app.py (so the importer doesn't
need FastAPI) and store.py (a pure persistence layer), so there is
exactly one place this logic can drift.
"""

from __future__ import annotations

from datetime import date

from railblock.api.schemas import BlockRequestIn
from railblock.api.store import get_corridor_context, next_request_id
from railblock.synthetic.maintenance_tasks import (
    DEFECT_TYPES,
    DEPARTMENTS,
    PRIORITY_WEIGHTS,
    SOURCE_SYSTEM,
    approval_path_for,
    splittable_for,
)


class RequestValidationError(ValueError):
    """A request row failed validation -- the caller decides how to
    surface this (an HTTP 400 in app.py, a per-row error line in the
    batch importer)."""


def build_request_row(req: BlockRequestIn, data_source: str = "USER_SUBMITTED") -> dict:
    if req.department not in DEPARTMENTS:
        raise RequestValidationError(f"department must be one of {DEPARTMENTS}")
    if req.defect_type not in DEFECT_TYPES.get(req.department, []):
        raise RequestValidationError(
            f"defect_type {req.defect_type!r} is not valid for department {req.department!r}"
        )
    if req.requester_priority not in PRIORITY_WEIGHTS:
        raise RequestValidationError(f"requester_priority must be one of {list(PRIORITY_WEIGHTS)}")

    ctx = get_corridor_context()
    if req.section_id not in set(ctx.sections["section_id"]):
        raise RequestValidationError(f"section_id {req.section_id!r} is not a real corridor section")

    raised = req.raised_date or date.today()
    days_overdue = max(0, (date.today() - req.due_date).days)
    return {
        "task_id": next_request_id(req.department),
        "department": req.department,
        "source_system": SOURCE_SYSTEM[req.department],
        "section_id": req.section_id,
        "defect_type": req.defect_type,
        "requester_priority": req.requester_priority,
        "raised_date": raised.isoformat(),
        "due_date": req.due_date.isoformat(),
        "days_overdue": days_overdue,
        "estimated_block_hours": req.estimated_block_hours,
        "splittable": splittable_for(req.requester_priority, req.defect_type),
        "approval_path": approval_path_for(req.estimated_block_hours),
        "data_source": data_source,
        "status": "pending",
    }
