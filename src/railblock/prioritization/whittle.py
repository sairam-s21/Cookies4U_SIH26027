"""Layer 2b -- Whittle-index ranking, and the Layer 2 orchestrator.

Adapted conceptually from Gerum, Altay & Baykal-Guersoy (2019), "Data-driven
predictive maintenance scheduling policies", which derive Whittle indices
for restless multi-armed bandits to decide which of several deteriorating
machines to inspect/service each period under a limited inspection budget.
We reuse the *shape* of their result -- an index that grows with accrued
deterioration and is boosted by resource scarcity -- not their exact
machine-degradation MDP or its closed-form derivation. Computing a formally
exact Whittle index would mean solving a full restless-bandit MDP per task
and section, which is out of scope for this build; the master prompt is
explicit that this is an "adaptation... applied here to task ranking rather
than their original inspection-scheduling context," not a literal port.

Each pending maintenance task is treated as a restless arm:
  - "passive" (not scheduled) -> its holding cost keeps accruing
  - "active" (scheduled)      -> the task closes out, cost stops accruing

The index used here:

    continuous_score(task) = base_score(task) * (1 + congestion(section))
    base_score(task) = (days_overdue + 1) * priority_weight * impact_weight
    congestion(section) = pending_demand_hours(section) / max(avg_free_hours_per_day(section), eps)

    whittle_index(task) = tier_rank(priority) * TIER_WIDTH
                           + continuous_score(task) / (1 + continuous_score(task)) * (TIER_WIDTH - 1)

Session 30, at explicit user request, after a real reported gap: "I want
all critical to rank above moderate tasks and all moderate tasks should
rank above routine tasks ... whittle index should also be considered for
ranking, but the order I said should remain." Before this, the index was
simply `base_score * (1 + congestion)` with no tier structure at all --
`priority_weight` (5.0/2.0/1.0, urgency.py) was just one multiplicative
factor among several UNBOUNDED ones (days_overdue, congestion), so a
Moderate task with enough accrued urgency/congestion could and did
legitimately out-rank a fresh Critical one -- confirmed live against a
real 70-task batch (whittle range ~1.9-43.4, Moderate's own max 17.7
exceeding Critical's own min 12.8).

A first fix tried a fixed additive per-tier gap -- rejected once a
regression test built to stress it (days_overdue=1000 on an
near-saturated section) broke straight through it: congestion is an
unbounded ratio, so no FIXED finite gap can ever be an UNCONDITIONAL
guarantee, only a "large enough for data seen so far" one. The formula
above is instead a real, provable guarantee: `x / (1 + x)` maps any
non-negative real number, however large, strictly into [0, 1) -- so a
tier's own whittle_index values can only ever occupy
[rank*TIER_WIDTH, rank*TIER_WIDTH + TIER_WIDTH), which can never reach
the next tier's floor, by construction rather than by hoping the inputs
stay reasonable. The existing continuous formula's own relative ordering
is preserved exactly (the `x/(1+x)` transform is strictly monotonic), so
whatever it already rewarded WITHIN a tier -- accrued urgency, section
congestion -- still works the same way, just rescaled to fit inside that
tier's own fixed-width slot.

Two properties this preserves from the real Whittle-index literature, and
why they matter here:

  1. Monotonic in accrued cost (days_overdue) and severity (priority x
     impact) -- on its own this would just reduce to a static priority
     sort, which is exactly what a naive urgency-only ranking already
     gives you.
  2. Boosted by resource scarcity in the task's own section -- an arm
     competing for a nearly-saturated resource (e.g. MAS-AJJ, see Session
     1's PROGRESS.md finding) has a higher OPPORTUNITY COST of staying
     passive this period: if it isn't scheduled now, the scarce window may
     simply not reopen soon. A static priority sort has no way to express
     this; the congestion multiplier is precisely what makes this index
     behave differently from urgency_score alone -- WITHIN a priority
     tier (Session 30: no longer ACROSS tiers -- see above;
     tests/test_prioritization.py::test_congestion_reorders_within_a_tier_
     but_never_crosses_one).

Note base_score pads days_overdue with +1. score_urgency() (urgency.py)
reports the literal, unpadded master-prompt formula (days_overdue *
weight, which is exactly 0 for not-yet-due tasks) as the transparent audit
column. Without the +1 pad here, every not-yet-overdue task would carry an
identical zero base_score regardless of priority/impact, collapsing the
Whittle index to an arbitrary tie for the majority of tasks that aren't
overdue yet -- the pad is a deliberate, documented refinement so the index
can still meaningfully separate a not-yet-due Critical task from a
not-yet-due Routine one.
"""

from __future__ import annotations

from datetime import date as Date, timedelta

import pandas as pd

from railblock.availability.corridor_availability import compute_availability
from railblock.prioritization.impact import score_impact
from railblock.prioritization.urgency import PRIORITY_WEIGHT, score_urgency

CONGESTION_EPS_HOURS = 1e-3

# Session 30, at explicit user request -- see the module docstring's
# `tier_offset` note above for the real gap this closes.
#
# A first version of this used a fixed additive gap (Routine=0,
# Moderate=1000, Critical=2000) sized against this corridor's REAL
# observed whittle range (~1.9-43.4). A regression test deliberately
# constructed to stress-test that gap (days_overdue=1000, a
# near-saturated section) immediately broke it: congestion is a ratio
# (demand_hours / free_hours) with no ceiling at all, so it can always
# be pushed past ANY fixed finite gap by combining extreme-enough
# congestion with extreme-enough days_overdue -- a FIXED additive
# constant can never be an UNCONDITIONAL guarantee, only a "large enough
# for data seen so far" one, which is not what "always" means.
#
# TIER_RANK/TIER_WIDTH below give a real unconditional guarantee
# instead: `continuous_score / (1 + continuous_score)` maps ANY
# non-negative real number, however astronomically large, into [0, 1)
# -- strictly, provably, with no exception. Multiplying that bounded
# fraction by (TIER_WIDTH - 1) and adding it to `tier_rank * TIER_WIDTH`
# means a tier's own whittle_index values can only ever occupy
# [rank*TIER_WIDTH, rank*TIER_WIDTH + TIER_WIDTH), which can never reach
# the next tier's floor -- by construction, not by hoping the inputs
# stay reasonable. TIER_WIDTH=1000 still gives ~999 units of resolution
# for realistic within-tier differences (this corridor's real
# continuous_score range of ~1.9-43.4 maps to roughly 65-977 of that).
TIER_RANK = {"Critical": 2, "Moderate": 1, "Routine": 0}
TIER_WIDTH = 1000.0


def compute_section_free_hours(
    sections: pd.DataFrame,
    passenger_occupancy: pd.DataFrame,
    goods_occupancy: pd.DataFrame,
    start_date: Date,
    n_days: int = 7,
    availability_cache: dict | None = None,
) -> pd.DataFrame:
    """Average daily free capacity (hours) per section over n_days, from
    Layer 1's compute_availability -- this IS the scarce "budget" the
    Whittle-style index reasons about opportunity cost against.

    `availability_cache` (optional, Session 25): see
    compute_availability's docstring -- shares this request's already-
    computed (section, date) results with whoever else needs them (Layer
    3's compute_daily_window_capacity) instead of recomputing the same
    real occupancy data a second time."""
    rows = []
    for section_id in sections["section_id"]:
        total_free_minutes = 0.0
        for d in range(n_days):
            the_date = start_date + timedelta(days=d)
            avail = compute_availability(section_id, the_date, passenger_occupancy, goods_occupancy, cache=availability_cache)
            total_free_minutes += avail["free_minutes"]
        rows.append(
            {
                "section_id": section_id,
                "avg_free_hours_per_day": (total_free_minutes / n_days) / 60.0,
            }
        )
    return pd.DataFrame(rows)


def compute_section_congestion(tasks: pd.DataFrame, section_free_hours: pd.DataFrame) -> pd.DataFrame:
    """congestion(section) = pending demand hours / max(free hours, eps).

    demand hours = sum of estimated_block_hours of every pending task
    competing for that section -- the more work is chasing a section's
    scarce free capacity, the higher its congestion.
    """
    demand = (
        tasks.groupby("section_id")["estimated_block_hours"]
        .sum()
        .rename("demand_hours")
        .reset_index()
    )
    merged = section_free_hours.merge(demand, on="section_id", how="left")
    merged["demand_hours"] = merged["demand_hours"].fillna(0.0)
    merged["congestion"] = merged["demand_hours"] / merged["avg_free_hours_per_day"].clip(
        lower=CONGESTION_EPS_HOURS
    )
    return merged


def rank_tasks(
    tasks: pd.DataFrame,
    sections: pd.DataFrame,
    passenger_occupancy: pd.DataFrame,
    goods_occupancy: pd.DataFrame,
    start_date: Date,
    n_days: int = 7,
    availability_cache: dict | None = None,
) -> pd.DataFrame:
    """Layer 2 orchestrator: urgency scoring + impact weighting + Whittle-
    index ranking, in that order.

    Returns `tasks` with added columns: urgency_score (transparent audit
    formula), impact_weight, avg_free_hours_per_day, demand_hours,
    congestion, whittle_index, risk_percentage, priority_rank (1 =
    schedule first). Sorted by priority_rank ascending. This is the exact
    shape Layer 3 consumes.

    `availability_cache` (optional, Session 25, at explicit user request
    for a large scheduling-time reduction): pass a fresh {} created once
    per real scheduling request and reused across every call that needs
    real per-section availability this same request (this function AND
    railblock.scheduling.capacity.compute_daily_window_capacity) -- real
    profiling showed the exact same (section, date) availability sweep
    being recomputed up to 3x per request (once here, once each for the
    critical-tasks-first Step A and Step D CP-SAT solves) against
    byte-identical occupancy data. See compute_availability's own
    docstring. Omit for the old always-recompute behavior (still the
    correct choice for a call that can't guarantee stable occupancy data
    across its own lifetime, e.g. spanning more than one real request)."""
    out = tasks.copy()
    out["urgency_score"] = score_urgency(out)
    out["impact_weight"] = score_impact(out)

    section_free_hours = compute_section_free_hours(
        sections, passenger_occupancy, goods_occupancy, start_date, n_days, availability_cache=availability_cache
    )
    congestion_table = compute_section_congestion(out, section_free_hours)
    out = out.merge(
        congestion_table[["section_id", "avg_free_hours_per_day", "demand_hours", "congestion"]],
        on="section_id",
        how="left",
    )

    priority_w = out["requester_priority"].map(PRIORITY_WEIGHT)
    base_score = (out["days_overdue"] + 1) * priority_w * out["impact_weight"]
    continuous_score = base_score * (1 + out["congestion"])
    tier_rank = out["requester_priority"].map(TIER_RANK)
    bounded_within_tier = continuous_score / (1 + continuous_score) * (TIER_WIDTH - 1)
    out["whittle_index"] = tier_rank * TIER_WIDTH + bounded_within_tier

    # Session 24, at explicit user request: whittle_index itself has no
    # natural upper bound (congestion = demand_hours / free_hours alone
    # can grow arbitrarily large for a badly congested section, and
    # days_overdue is open-ended too), so showing the raw number in the
    # UI as a "score out of some limit" would need an invented cap with
    # no real basis. Expressed instead as a real, bounded [0, 100]
    # PERCENTILE RANK against every other task in THIS SAME `tasks`
    # batch: "this task is more urgent than risk_percentage% of the
    # tasks currently in the waiting list" -- a real, computed, always-
    # meaningful number, deliberately RELATIVE to whichever batch is
    # passed in (moves as the waiting list itself changes), not an
    # absolute scale. A single-task batch is defined as 100% (trivially
    # the most urgent task right now).
    if len(out) > 1:
        out["risk_percentage"] = (out["whittle_index"].rank(method="average") - 1) / (len(out) - 1) * 100
    else:
        out["risk_percentage"] = 100.0

    out = out.sort_values("whittle_index", ascending=False).reset_index(drop=True)
    out["priority_rank"] = out.index + 1
    return out
