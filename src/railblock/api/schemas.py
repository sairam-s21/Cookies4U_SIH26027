"""Pydantic request/response models for the API layer."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field

from railblock.config import SCHEDULE_OPTIONS_TIME_LIMIT_S
from railblock.synthetic.maintenance_tasks import DEFECT_TYPES, DEPARTMENTS


class BlockRequestIn(BaseModel):
    """A block/maintenance request submitted by a department -- the human
    BDMS judgement call (priority) is the requester's, taken as given
    input, never predicted by this system (master prompt Section 2)."""

    department: str = Field(..., description=f"One of {DEPARTMENTS}")
    section_id: str
    defect_type: str = Field(..., description="Must be valid for the given department")
    requester_priority: str = Field(..., description="Critical | Moderate | Routine")
    estimated_block_hours: float = Field(..., gt=0)
    due_date: date
    raised_date: date | None = None


class BlockRequestOut(BlockRequestIn):
    task_id: str
    source_system: str
    days_overdue: int
    splittable: bool
    approval_path: str
    status: str
    data_source: str = "USER_SUBMITTED"


class SeedDemoRequest(BaseModel):
    demand_scenario: str = Field(..., description="stress_test | double_track_adjusted")
    start_date: date
    seed: int | None = None


class RecommendRequest(BaseModel):
    start_date: date
    n_days: int = Field(7, ge=1, le=28)
    # Session 13: raised 30 -> 120 at explicit user request ("remove the
    # max solver time") -- not literally unbounded, since a live HTTP
    # request with no time limit at all risks hanging indefinitely if
    # CP-SAT can't prove optimality quickly, which would be worse for a
    # live demo than a generous-but-bounded budget. 120s gives CP-SAT far
    # more room to reach real OPTIMAL instead of being cut off at
    # FEASIBLE (see the run-to-run variance found and explained earlier
    # this session).
    time_limit_s: float = Field(120.0, gt=0)


class ApproveRequest(BaseModel):
    task_ids: list[str] | None = Field(
        None, description="Subset to approve; omit to approve every fully-scheduled task from the last recommendation"
    )
    option_key: str | None = Field(
        None, description="Session 9: approve from a specific POST /schedule/options result instead of the last /schedule/recommend result"
    )


class DeleteRequestsRequest(BaseModel):
    task_ids: list[str] = Field(..., description="Only rows still status='pending' are actually deleted")


class ScheduleOptionsRequest(BaseModel):
    start_date: date
    n_days: int = Field(7, ge=1, le=28)
    # Session 14, at explicit user request: the frontend no longer
    # exposes this as an editable field (Recommended Scheduling) -- a
    # live 120s-per-strategy budget could run 2x that in real wall time
    # (the "balanced" strategy solves first, then the other two run
    # concurrently, each up to their own full budget), on top of real
    # non-solver overhead (ranking, splitting, the adaptive-allocation
    # regression pass) -- which reliably ran well past what a number on
    # screen suggested, an awkward thing to have visible live in front
    # of judges. Lowered back down to a bound that's actually held to in
    # practice (see the controlled before/after comparison run earlier
    # this session, which got real, good results at 15s) -- fixed
    # internally now, not a user-facing dial.
    time_limit_s: float = Field(default_factory=lambda: SCHEDULE_OPTIONS_TIME_LIMIT_S, gt=0)


class MonthlyPlanRequest(BaseModel):
    start_date: date
    n_weeks: int = Field(4, ge=1, le=12)
    time_limit_s: float = Field(120.0, gt=0)


class EmergencyResolveRequest(BaseModel):
    """Session 30, at explicit user request: applies or discards the
    reschedule proposal POST /emergency/create already computed for one
    emergency -- see railblock.scheduling.emergency."""

    apply: bool = Field(..., description="True: apply the proposed reschedule. False: discard it (affected tasks are vacated, not moved).")
