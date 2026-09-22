"""SYNTHETIC maintenance-task generator standing in for TMS/SMMS/TDMS.

No public dataset exists for Indian Railways defect/overdue-task records
(see docs/MASTER_PROMPT_SIH_26027.md Section 5), so this generator produces
realistic-but-fabricated task records parameterized by department, defect
type, requester-assigned priority, and overdue timing. Every row is tagged
data_source="SYNTHETIC" so downstream code and dashboards never mistake it
for real data. Re-running with a different seed produces a fresh scenario.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

DEPARTMENTS = ["Engineering", "Signalling", "Traction"]
SOURCE_SYSTEM = {"Engineering": "TMS", "Signalling": "SMMS", "Traction": "TDMS"}

DEFECT_TYPES = {
    # Defect list per docs/defect_maintenance_reference.md -- that
    # document's own honesty note applies here too: splittability/
    # impact-weight judgement calls are informed engineering estimates
    # for this corridor's real characteristics, not sourced from an
    # official IR document.
    "Engineering": [
        "Rail fracture",
        "Weld defect",
        "Ballast deficiency",
        "Track geometry deviation",
        "Rail wear (flange contact)",
        "Fishplate/joint looseness",
        "Points & crossing wear",
        "Rail corrugation",
        "Bridge/girder deterioration",
        "Level crossing gate fault",
        "Sleeper damage",
        "Cess/formation erosion",
    ],
    "Signalling": [
        "Signal lamp failure",
        "Point machine fault",
        "Track circuit failure",
        "Axle counter fault",
        "Interlocking relay fault",
        "Signal cable fault",
        "LED signal aspect fault",
        "Block instrument fault",
        "Level crossing interlocking fault",
        "Data logger fault",
        "Panel/VDU fault",
        "Point detection circuit fault",
    ],
    "Traction": [
        "OHE contact wire wear",
        "Insulator flashover damage",
        "Dropper/hanger damage",
        "Feeder cable fault",
        "Tension length anomaly",
        "Earthing/bonding fault",
        "Traction substation fault",
        "Pantograph-OHE interaction damage",
        "Return conductor fault",
        "Auto-tensioning device fault",
        "Jumper wire fault",
        "Overhead mast/structure damage",
    ],
}

# Whether a defect type's repair can realistically be done across multiple
# separate shorter possessions rather than needing one continuous block.
# True = splittable (staged/incremental work: individual joints, droppers,
# bonds, cable sections, lamp swaps, grinding passes -- each unit of work
# is independently completable). False = a single continuous possession is
# a genuine safety/technical requirement: interlocking cutover/testing
# work, traction isolation procedures, or urgent structural repair that
# must be completed and verified in one sitting before the section can be
# safely reopened.
SPLITTABLE = {
    # Engineering (TMS)
    "Rail fracture": False,               # urgent structural repair, weld + test in one continuous possession
    "Weld defect": False,                 # weld repair needs continuous work + cooling/testing
    "Ballast deficiency": True,           # tamping/packing can be phased across sections of the same stretch
    "Track geometry deviation": True,     # tamping can be phased in sub-sections
    "Rail wear (flange contact)": True,   # grinding can be done in staged passes
    "Fishplate/joint looseness": True,    # individual joints tightened incrementally
    "Points & crossing wear": False,      # interlocking/geometry integrity needs one continuous possession
    "Rail corrugation": True,             # grinding done in staged passes
    "Bridge/girder deterioration": False, # structural repair needs continuous possession (often multi-day)
    "Level crossing gate fault": True,    # mechanical adjustment, short window between train movements
    "Sleeper damage": True,               # individual sleepers replaced incrementally
    "Cess/formation erosion": True,       # earthwork staged in sections
    # Signalling (SMMS)
    "Signal lamp failure": True,          # simple swap, can be phased across a few lamps/nights
    "Point machine fault": False,         # mechanical/electrical adjustment + testing needs continuity
    "Track circuit failure": False,       # circuit commissioning/testing is meaningless if interrupted
    "Axle counter fault": False,          # calibration + testing needs continuity
    "Interlocking relay fault": False,    # signalling cutover work -- explicitly continuous, safety-critical
    "Signal cable fault": True,           # cable sections can be repaired incrementally
    "LED signal aspect fault": True,      # module swap, phased across signals
    "Block instrument fault": False,      # commissioning/testing needs continuity
    "Level crossing interlocking fault": False,  # interlock test needs continuity, safety-critical
    "Data logger fault": True,            # not safety-blocking, low-traffic window fits any time
    "Panel/VDU fault": True,              # module-level diagnosis and swap, staged if multiple elements
    "Point detection circuit fault": False,  # detection-circuit commissioning test needs continuity
    # Traction (TDMS)
    "OHE contact wire wear": True,        # staged repair along sections of contact wire
    "Insulator flashover damage": False,  # isolation + replacement + testing needs continuity
    "Dropper/hanger damage": True,        # individual droppers repaired incrementally
    "Feeder cable fault": False,          # traction power feeder work needs continuous isolation
    "Tension length anomaly": False,      # traction isolation procedure -- explicitly continuous
    "Earthing/bonding fault": True,       # individual bonds fixed incrementally
    "Traction substation fault": False,   # switching/isolation work needs continuity
    "Pantograph-OHE interaction damage": True,  # localized repair, staged if spans multiple sections
    "Return conductor fault": True,       # staged along the affected stretch
    "Auto-tensioning device fault": False,  # final calibration needs continuity
    "Jumper wire fault": True,            # individual jumpers repaired incrementally
    "Overhead mast/structure damage": False,  # structural repair needs continuous possession
}


def splittable_for(requester_priority: str, defect_type: str, rng=None) -> bool:
    """Every task is splittable, Critical included -- train-cancellation
    escalation (the scenario that previously required some Critical tasks
    to stay non-splittable) is out of scope for this project. Critical
    priority is expressed only through a higher whittle_index weight, not
    by forcing a continuous possession. Signature/params are kept as-is
    (both call sites still pass them) even though they're no longer used,
    to avoid churn at either caller."""
    return True


# Requester-assigned priority in BDMS -- a human judgement call taken as
# given input, not predicted by this system (see master prompt Section 2).
PRIORITY_WEIGHTS = {"Critical": 0.15, "Moderate": 0.35, "Routine": 0.50}

# ============================================================================
# UNSOURCED ASSUMPTION -- flagged plainly here, not deleted or silently
# changed.
#
# DUE_WINDOW_DAYS and RAISE_LOOKBACK_DAYS below are NOT sourced from any
# Indian Railways document; they are an internal placeholder assumption
# only, despite sometimes being discussed as if they were an established
# SLA policy.
#
# A real, dated IR document exists: a 2024 Railway Board "Rolling Block
# Plan guidelines" circular. It describes a STRUCTURALLY DIFFERENT scheme
# -- a 26-week rolling block plan reviewed every 2 months, with "mega
# blocks" (>4 hours) requiring Railway Board approval communicated at
# least 4 weeks in advance. This does not map cleanly onto a 3-tier
# Critical/Moderate/Routine day-window model (it's a rolling-horizon +
# advance-notice scheme, not a per-task due-date SLA), and reconciling the
# two remains a non-trivial redesign, not attempted here.
#
# An earlier configuration used RAISE_LOOKBACK_DAYS=120 and a
# DUE_WINDOW_DAYS lower bound as low as 3 days, which meant `raised_date +
# due_offset` could land BEFORE `as_of_date` -- measured at 60-62% of
# generated tasks already overdue the moment they were created, a
# confusing default for a live demo (a task raised "just now" showing e.g.
# 73 days overdue). due_date is now anchored to as_of_date directly (not
# raised_date), and every tier's lower bound is >=10 days, so a freshly
# generated task is NEVER already overdue -- `pre_existing_backlog_pct`
# (due_date_completion.py) is 0% by construction from this point on. The
# 60-62% figure, and any KPI reference/precomputed file computed under the
# old constants, describes THAT prior configuration only; it is not a
# claim about the current generator and must not be quoted as this
# generator's present behaviour.
#
# What remains valid: demand-scenario comparisons made under the OLD
# constants are still meaningful as INTERNALLY CONSISTENT comparisons
# against each other from that period (same assumption applied uniformly
# both times) -- they were not, and should not be, presented as validated
# absolute real-world figures either.
# ============================================================================
DUE_WINDOW_DAYS = {"Critical": (10, 20), "Moderate": (20, 40), "Routine": (45, 60)}
RAISE_LOOKBACK_DAYS = 10  # raised within the last ~10 days -- flavor only, does not affect due_date below

# The fixed 70-task demo template pool's own raised_date/due_date are
# baked in once, at whatever date tasks.csv was imported on (see
# railblock.integrations.import_tasks) -- they never move with the
# calendar. An evaluator opening the hosted link weeks (or months) after
# that import would click "Create demo batch of tasks" and see every
# single task already overdue on arrival, since due_date is a fixed past
# date by then, not a live one. `refresh_demo_batch_dates` (below)
# recomputes raised_date/due_date fresh at ACTIVATION time instead,
# anchored to the real date the button was clicked -- never the
# template's own frozen values. A FIXED seed, applied in the templates'
# own stable order (store.template_tasks() is always ORDER BY
# inserted_at ASC), means the same 70 tasks get the same relative
# raised/due offsets every time, regardless of which calendar date
# activation happens to fall on -- "the same set of tasks are generated
# each time," just anchored to a different "today."
#
# raised_date is never AFTER the click date (a freshly-raised task
# reported in the future makes no sense) -- at most
# REACTIVATION_RAISE_LOOKBACK_DAYS days before it. due_date is
# REACTIVATION_DUE_WINDOW_DAYS[priority] days after THAT raised_date, not
# after the click date -- narrower, priority-scaled sub-ranges within an
# overall 10-30 day band (Critical still gets the shortest window,
# Routine the longest, same relative ordering as DUE_WINDOW_DAYS above,
# just rescaled to fit 10-30 instead of 10-60). Because due_date can land
# as close as REACTIVATION_DUE_WINDOW_DAYS's own lower bound (10) days
# after a raised_date that's already up to REACTIVATION_RAISE_LOOKBACK_DAYS
# (5) days in the past, a handful of tasks can end up scheduled past
# their own due_date once corridor congestion is factored in during the
# actual CP-SAT solve -- a normal, expected scheduling outcome, not a
# data bug; days_overdue itself (computed against the click date, not a
# later "now") is always 0 at the moment of activation, matching this
# module's own established "never already overdue" convention (see the
# assumption note above).
REACTIVATION_RAISE_LOOKBACK_DAYS = 5
REACTIVATION_DUE_WINDOW_DAYS = {"Critical": (10, 15), "Moderate": (15, 22), "Routine": (22, 30)}
REACTIVATION_SEED = 20260917


def refresh_demo_batch_dates(templates: list[dict], as_of_date: date) -> list[dict]:
    """Returns a copy of `templates` with raised_date/due_date/
    days_overdue recomputed fresh, anchored to `as_of_date` (the real
    date "Create demo batch of tasks" was clicked) -- see the module
    note above for why this exists and why a fixed seed is used."""
    rng = np.random.default_rng(REACTIVATION_SEED)
    out = []
    for row in templates:
        row = dict(row)
        priority = row.get("requester_priority", "Routine")
        raise_offset = int(rng.integers(0, REACTIVATION_RAISE_LOOKBACK_DAYS + 1))
        raised_date = as_of_date - timedelta(days=raise_offset)
        lo, hi = REACTIVATION_DUE_WINDOW_DAYS.get(priority, REACTIVATION_DUE_WINDOW_DAYS["Routine"])
        due_offset = int(rng.integers(lo, hi + 1))
        due_date = raised_date + timedelta(days=due_offset)
        row["raised_date"] = raised_date.isoformat()
        row["due_date"] = due_date.isoformat()
        row["days_overdue"] = max(0, (as_of_date - due_date).days)
        out.append(row)
    return out
# Per-defect-type (best, avg, worst) fix durations in hours.
#
# An early version invented durations up to 72 hours for severe defects
# (Bridge/girder deterioration, etc.) without checking them against real
# duration data. Values below are instead grounded against real data
# already in this project: datasets/blocks_04-09-2026 (1).xlsx and
# block_requests_04-09-2026_to_04-09-2026.xlsx -- real Southern Railway
# COA block records for an actual date. Their real Duration/demanded-
# time-range values, checked directly: min 0.5h, mean ~2.2-2.3h, median
# 2.0h, MAX 6.0h, across both files, 160 real block records combined. No
# real block in either file exceeds 6 hours. Every value below is capped
# at that observed maximum -- relative severity ordering between defect
# types is still an informed judgement call (not sourced), but the
# actual NUMBERS are grounded against real evidence rather than invented
# independently of it.
DEFECT_DURATION_HOURS = {
    # Engineering (TMS)
    "Rail fracture": (1.0, 2.5, 5.0),
    "Weld defect": (1.0, 2.0, 4.0),
    "Ballast deficiency": (1.5, 3.0, 6.0),
    "Track geometry deviation": (1.5, 3.0, 6.0),
    "Rail wear (flange contact)": (1.0, 2.5, 5.0),
    "Fishplate/joint looseness": (0.5, 1.0, 2.0),
    "Points & crossing wear": (2.0, 4.0, 6.0),
    "Rail corrugation": (1.0, 2.5, 5.0),
    "Bridge/girder deterioration": (2.0, 4.0, 6.0),
    "Level crossing gate fault": (0.5, 1.5, 3.0),
    "Sleeper damage": (1.0, 2.0, 4.0),
    "Cess/formation erosion": (1.5, 3.5, 6.0),
    # Signalling (SMMS)
    "Signal lamp failure": (0.25, 0.5, 1.0),
    "Point machine fault": (1.0, 2.5, 5.0),
    "Track circuit failure": (1.0, 2.5, 5.0),
    "Axle counter fault": (1.0, 2.0, 4.0),
    "Interlocking relay fault": (1.5, 3.0, 6.0),
    "Signal cable fault": (0.5, 1.5, 3.0),
    "LED signal aspect fault": (0.25, 0.75, 1.5),
    "Block instrument fault": (1.0, 2.0, 4.0),
    "Level crossing interlocking fault": (1.0, 2.5, 5.0),
    "Data logger fault": (0.5, 1.0, 2.0),
    "Panel/VDU fault": (1.0, 2.0, 4.0),
    "Point detection circuit fault": (0.75, 1.5, 3.0),
    # Traction (TDMS)
    "OHE contact wire wear": (1.0, 2.0, 4.0),
    "Insulator flashover damage": (1.0, 2.5, 5.0),
    "Dropper/hanger damage": (0.5, 1.0, 2.0),
    "Feeder cable fault": (1.5, 3.0, 6.0),
    "Tension length anomaly": (1.5, 3.0, 5.0),
    "Earthing/bonding fault": (0.5, 1.0, 2.0),
    "Traction substation fault": (2.0, 4.0, 6.0),
    "Pantograph-OHE interaction damage": (1.0, 2.5, 5.0),
    "Return conductor fault": (1.0, 2.0, 4.0),
    "Auto-tensioning device fault": (1.0, 2.0, 4.0),
    "Jumper wire fault": (0.5, 1.0, 2.0),
    "Overhead mast/structure damage": (2.0, 4.0, 6.0),
}

# Grounded in the real Southern Railway disconnection/reconnection
# procedure (SR 3.51.6 & App. XIII, ASM training guide
# docs/1429770763529-Pro ASM study material.pdf p.27): "Disconnection upto
# one hour should normally be allowed by SM depending upon trains in the
# section... For works involving disconnection for more than one hour, a
# Disconnection schedule jointly signed by Sr.DSTE, Sr.DOM, Sr.DEN & Sr.DEE
# shall be issued." This is metadata only -- it does NOT change the CP-SAT
# scheduling constraint (does a window exist is unchanged); it's how a
# scheduled task is later reported/displayed, reflecting which real-world
# approval track it would actually take.
SM_DIRECT_APPROVAL_MAX_HOURS = 1.0


def approval_path_for(estimated_block_hours: float) -> str:
    return "sm_direct" if estimated_block_hours <= SM_DIRECT_APPROVAL_MAX_HOURS else "joint_schedule"


# Demand calibration against a real, citable benchmark -- Lidén,
# "Coordinating maintenance windows and train traffic": 2,915 maintenance
# tasks/year on Sweden's Ockelbo-Ljusdal line (Trafikverket budget data),
# a 167km SINGLE-TRACK line, i.e. ~17.5 tasks/km/year. Scaled to this
# corridor's real length (214.1km, MAS-JTJ): ~3,740 tasks/year, ~72
# tasks/week -- the STRESS_TEST scenario below, grounded in a citable
# source rather than picked arbitrarily.
#
# IMPORTANT CAVEAT, not a footnote: Lidén's own paper explicitly EXCLUDES
# double-track lines from this volume estimate, stating double-track
# maintenance is comparatively easy to fit around running trains without a
# full closure. This corridor is a double-track Southern Railway mainline,
# not the single-track line the benchmark was measured on. The real reason
# why, per the ASM guide's SR 3.51.6: maintenance work is classified into
# Group A (no consent needed at all), Group B (SM consent only, no formal
# block), and Group C (formal Disconnection Notice required) -- only Group
# C is what this system actually schedules. No public source gives the
# real Group A/B/C split for this or any Indian corridor, so
# GROUP_C_FRACTION below is a DOCUMENTED ASSUMPTION, not a measured or
# sourced figure -- 40% is a defensible starting point in a reasonable
# 30-50% range for a double-track mainline. State this plainly as an
# estimate everywhere it's used (PROGRESS.md, KPI output) -- never present
# it as authoritative.
LIDEN_TASKS_PER_KM_PER_YEAR = 2915 / 167  # Ockelbo-Ljusdal, single-track, excludes double-track lines
CORRIDOR_LENGTH_KM = 214.1  # MAS-JTJ, train 12243's own recorded distance to JTJ
GROUP_C_FRACTION = 0.4  # ASSUMPTION -- see caveat above, not a sourced figure

DEMAND_SCENARIOS = {
    # Lidén's single-track benchmark scaled to this corridor's length, unadjusted.
    "stress_test": round(LIDEN_TASKS_PER_KM_PER_YEAR * CORRIDOR_LENGTH_KM / 52),
    # The same figure scaled down by the documented Group-C-fraction assumption,
    # reflecting that this is a double-track corridor where most maintenance
    # doesn't need a scheduled block at all.
    "double_track_adjusted": round(LIDEN_TASKS_PER_KM_PER_YEAR * CORRIDOR_LENGTH_KM / 52 * GROUP_C_FRACTION),
}


def n_tasks_for_scenario(demand_scenario: str) -> int:
    """Weekly task count for a named demand scenario -- see DEMAND_SCENARIOS
    and the caveat above for what each one represents and why."""
    if demand_scenario not in DEMAND_SCENARIOS:
        raise ValueError(f"Unknown demand_scenario {demand_scenario!r}; choose from {list(DEMAND_SCENARIOS)}")
    return DEMAND_SCENARIOS[demand_scenario]


# Relative real-world frequency per defect type -- routine wear-and-tear
# defects genuinely occur far more often than rare structural/
# catastrophic ones (a bridge deteriorating to the point of needing
# repair is a rare event; a loose fishplate is not). Selecting defect
# types uniformly at random treats a 72-hour-worst-case bridge repair as
# equally likely as a 4-hour fishplate tightening, which is unrealistic
# and, empirically, drags generated task durations up sharply. Same
# documented-assumption status as service_frequency.py's
# SERVICE_TYPE_FREQUENCY_MIX -- a reasoned real-world judgement call, not
# a sourced frequency table (none exists publicly for this corridor).
_COMMON, _OCCASIONAL, _RARE = 10, 4, 1
DEFECT_FREQUENCY_WEIGHT = {
    # Engineering (TMS)
    "Ballast deficiency": _COMMON,
    "Track geometry deviation": _COMMON,
    "Rail wear (flange contact)": _COMMON,
    "Fishplate/joint looseness": _COMMON,
    "Rail corrugation": _COMMON,
    "Sleeper damage": _COMMON,
    "Weld defect": _OCCASIONAL,
    "Points & crossing wear": _OCCASIONAL,
    "Level crossing gate fault": _OCCASIONAL,
    "Cess/formation erosion": _OCCASIONAL,
    "Rail fracture": _RARE,
    "Bridge/girder deterioration": _RARE,
    # Signalling (SMMS)
    "Signal lamp failure": _COMMON,
    "Signal cable fault": _COMMON,
    "LED signal aspect fault": _COMMON,
    "Data logger fault": _COMMON,
    "Panel/VDU fault": _COMMON,
    "Point machine fault": _OCCASIONAL,
    "Track circuit failure": _OCCASIONAL,
    "Axle counter fault": _OCCASIONAL,
    "Block instrument fault": _OCCASIONAL,
    "Point detection circuit fault": _OCCASIONAL,
    "Interlocking relay fault": _RARE,
    "Level crossing interlocking fault": _RARE,
    # Traction (TDMS)
    "OHE contact wire wear": _COMMON,
    "Dropper/hanger damage": _COMMON,
    "Earthing/bonding fault": _COMMON,
    "Jumper wire fault": _COMMON,
    "Return conductor fault": _COMMON,
    "Insulator flashover damage": _OCCASIONAL,
    "Tension length anomaly": _OCCASIONAL,
    "Pantograph-OHE interaction damage": _OCCASIONAL,
    "Auto-tensioning device fault": _OCCASIONAL,
    "Feeder cable fault": _RARE,
    "Traction substation fault": _RARE,
    "Overhead mast/structure damage": _RARE,
}


def _defect_type_probabilities(department: str) -> list[float]:
    weights = [DEFECT_FREQUENCY_WEIGHT[dt] for dt in DEFECT_TYPES[department]]
    total = sum(weights)
    return [w / total for w in weights]


def generate_maintenance_tasks(
    sections: pd.DataFrame,
    n_tasks: int = 150,
    as_of_date: date | None = None,
    seed: int | None = None,
    demand_scenario: str | None = None,
) -> pd.DataFrame:
    """Generate n_tasks synthetic maintenance/defect records across the
    corridor's block sections.

    sections: block_sections DataFrame (needs a section_id column).
    as_of_date: reference "today" for computing days_overdue; defaults to
    the real current date.
    seed: fixes the RNG for reproducible test runs; omit for a fresh
    scenario each call.
    demand_scenario: if given (one of DEMAND_SCENARIOS' keys), overrides
    n_tasks with that scenario's weekly figure -- see DEMAND_SCENARIOS
    above. Leave as None to keep using an explicit n_tasks (fully
    backward-compatible default; existing callers are unaffected).
    """
    if demand_scenario is not None:
        n_tasks = n_tasks_for_scenario(demand_scenario)
    if as_of_date is None:
        as_of_date = date.today()
    rng = np.random.default_rng(seed)

    section_ids = sections["section_id"].tolist()
    priorities = list(PRIORITY_WEIGHTS.keys())
    priority_p = list(PRIORITY_WEIGHTS.values())
    defect_p_by_department = {dep: _defect_type_probabilities(dep) for dep in DEPARTMENTS}

    rows = []
    for i in range(n_tasks):
        department = rng.choice(DEPARTMENTS)
        priority = rng.choice(priorities, p=priority_p)
        section_id = rng.choice(section_ids)
        defect_type = rng.choice(DEFECT_TYPES[department], p=defect_p_by_department[department])

        raise_offset = int(rng.integers(0, RAISE_LOOKBACK_DAYS))
        raised_date = as_of_date - timedelta(days=raise_offset)

        # due_date is anchored to as_of_date (not raised_date) so a freshly
        # generated task is NEVER already overdue -- every task's due date
        # is at least DUE_WINDOW_DAYS' lower bound (>=10) days in the future.
        lo, hi = DUE_WINDOW_DAYS[priority]
        due_offset = int(rng.integers(lo, hi + 1))
        due_date = as_of_date + timedelta(days=due_offset)

        days_overdue = max(0, (as_of_date - due_date).days)

        # Triangular, not uniform: most real repairs land near the "avg
        # case" duration, with best/worst case as rarer tails -- a more
        # realistic shape than a flat range, and keyed by the SPECIFIC
        # defect_type now, not a department-wide average (see
        # DEFECT_DURATION_HOURS above).
        best_h, avg_h, worst_h = DEFECT_DURATION_HOURS[defect_type]
        estimated_block_hours = round(float(rng.triangular(best_h, avg_h, worst_h)), 2)

        rows.append(
            {
                "task_id": f"{SOURCE_SYSTEM[department]}-{i + 1:05d}",
                "department": department,
                "source_system": SOURCE_SYSTEM[department],
                "section_id": section_id,
                "defect_type": defect_type,
                "requester_priority": priority,
                "raised_date": raised_date.isoformat(),
                "due_date": due_date.isoformat(),
                "days_overdue": days_overdue,
                "estimated_block_hours": estimated_block_hours,
                "splittable": splittable_for(priority, defect_type, rng=rng),
                "approval_path": approval_path_for(estimated_block_hours),
                "data_source": "SYNTHETIC",
            }
        )

    return pd.DataFrame(rows)
