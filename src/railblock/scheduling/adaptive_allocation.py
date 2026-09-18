"""Adaptive-allocation regression integration -- runs on EVERY ranked
task, right after Whittle-index ranking and BEFORE CP-SAT ever sees the
batch, at explicit user direction. This is deliberately NOT a fallback
retried only on whatever CP-SAT failed to place -- every task the
regression offers a safe reduction for gets trimmed unconditionally,
whether or not its full demanded duration would have fit somewhere. See
api/app.py's _apply_adaptive_allocation_upfront for the exact call site
and PROGRESS.md for why this ordering was chosen over the alternative
(letting CP-SAT choose full vs. trimmed per task) after that tradeoff
was raised directly.

Unlike critical_task_escalation.py (which finds real TRAIN adjustments
and always requires human confirmation, since it touches the real
timetable), this tier only ever adjusts OUR OWN task's requested
duration -- it never touches a real train. Consistent with this
project's "solver supremacy" principle: the regression only ever
proposes a smaller allocation for our own task and hands that to CP-SAT
like any other task's duration; it never assigns a window itself and
never overrides a capacity constraint.

A trimmed task that ends up scheduled keeps BOTH its original demand and
what it was actually given (see how the caller in api/app.py tags
`adaptive_allocation` / `demanded_block_hours` onto the final schedule
row) -- never silently merged into an ordinary full-duration completion,
so nobody mistakes a reduced-allocation block for the original demand
being fully met. See railblock.ml.predict_allocation's own docstring for
what "safe" means here (a real statistical margin, not zero-risk).
"""

from __future__ import annotations

import pandas as pd

from railblock.ml.predict_allocation import predict_safe_allocation, predict_safe_allocation_batch


def apply_adaptive_allocation(ranked_tasks: pd.DataFrame) -> pd.DataFrame:
    """For EVERY row in `ranked_tasks` (the full ranked batch, not just
    unscheduled ones), ask the regression for a safe reduced allocation.
    Returns a COPY with `estimated_block_hours` replaced by the reduced
    value for every task the model offered a genuine reduction for
    (`adaptive_allocation_applied` == True on that row) -- unconditionally,
    regardless of whether the task's full duration would have fit
    somewhere; every other row is returned unchanged, with its original
    hours preserved in `demanded_block_hours` either way. Does NOT itself
    check whether the reduced duration then fits anywhere -- that's
    decided by CP-SAT, exactly like any other task's duration.

    Session 25, at explicit user request for a large scheduling-time
    reduction: tries predict_safe_allocation_batch() first (one real
    model call for the whole DataFrame, identical math to calling
    predict_safe_allocation() per row -- see that function's own
    docstring), falling back to the original row-by-row loop only if the
    batch call itself failed (model unavailable, or any exception) --
    this keeps the exact same graceful-degradation guarantee this module
    already had, just without paying the per-row cost in the common case.
    """
    out = ranked_tasks.copy()
    # Deliberately NOT an early `if out.empty: return out` -- a 0-row (but
    # still real-columned) input is real and reachable (e.g. an
    # escalation round where every pending task just got resolved), and
    # the two columns below must still exist on the result either way, or
    # a caller checking `reduced["adaptive_allocation_applied"]` crashes
    # with a real KeyError -- the same class of empty-DataFrame bug
    # already found once this session in model.py's solve_schedule().
    out["demanded_block_hours"] = out["estimated_block_hours"]
    out["adaptive_allocation_applied"] = False

    batch = predict_safe_allocation_batch(out) if not out.empty else None
    if batch is not None:
        applied = batch[batch["safe_fraction"] < 1.0]
        out.loc[applied.index, "estimated_block_hours"] = applied["safe_allocation_hours"]
        out.loc[applied.index, "adaptive_allocation_applied"] = True
        return out

    for idx, row in out.iterrows():
        pred = predict_safe_allocation(
            department=row["department"],
            defect_type=row["defect_type"],
            criticality=row["requester_priority"],
            demanded_hours=row["estimated_block_hours"],
        )
        if pred is not None and pred["safe_fraction"] < 1.0:
            out.at[idx, "estimated_block_hours"] = pred["safe_allocation_hours"]
            out.at[idx, "adaptive_allocation_applied"] = True
    return out
