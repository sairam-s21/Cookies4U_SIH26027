"""Layer 2a -- Urgency scoring.

Deterministic formula per docs/MASTER_PROMPT_SIH_26027.md Section 3:

    urgency_score = days_overdue * requester_priority_weight

Fully transparent and auditable -- no learned parameters, no ML. The
priority weights below are an illustrative severity multiplier (a Critical
task accrues urgency 2.5x faster per day overdue than a Moderate one, and
5x faster than a Routine one) -- a reasonable assumption documented here,
not sourced from any real IR SLA policy (none is publicly available, per
docs/MASTER_PROMPT_SIH_26027.md Section 5).

requester_priority itself is the human BDMS judgement call from Session 1's
task generator -- this module ranks and weighs it, it never predicts it.
"""

from __future__ import annotations

import pandas as pd

PRIORITY_WEIGHT = {"Critical": 5.0, "Moderate": 2.0, "Routine": 1.0}


def score_urgency(tasks: pd.DataFrame) -> pd.Series:
    """Return urgency_score aligned to `tasks`' index.

    tasks needs `days_overdue` and `requester_priority` columns (as
    produced by railblock.synthetic.maintenance_tasks.generate_maintenance_tasks).
    """
    weights = tasks["requester_priority"].map(PRIORITY_WEIGHT)
    if weights.isna().any():
        bad = sorted(tasks.loc[weights.isna(), "requester_priority"].unique())
        raise ValueError(f"Unknown requester_priority value(s): {bad}")
    return tasks["days_overdue"] * weights
