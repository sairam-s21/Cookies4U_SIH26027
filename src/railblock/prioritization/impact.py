"""Layer 2c -- Impact / delay-risk weighting.

Conceptually informed by Odolinski (2019): failures that would actually
disrupt train running (force a speed restriction, a stoppage, manual
signalling/block working) should be weighted more heavily than failures
that are real defects but carry little risk of delaying a train. No public
IR-specific dataset exists to calibrate exact multipliers (see
docs/MASTER_PROMPT_SIH_26027.md Section 5), so this is an illustrative,
clearly-labelled weighting scheme keyed by defect type -- a reasonable
assumption, not fitted to data.

1.0 = baseline (defect is real but has little direct delay risk).
Higher = judged more likely to force operational disruption if left open.
"""

from __future__ import annotations

import pandas as pd

from railblock.synthetic.maintenance_tasks import DEFECT_TYPES

IMPACT_WEIGHT = {
    # Engineering (TMS)
    "Rail fracture": 2.0,              # immediate speed restriction / possible line block
    "Weld defect": 1.6,
    "Track geometry deviation": 1.5,   # speed restriction risk
    "Points & crossing wear": 1.4,     # can force manual working, delays
    "Fishplate/joint looseness": 1.2,
    "Ballast deficiency": 1.1,
    "Rail wear (flange contact)": 1.1,
    "Bridge/girder deterioration": 2.2,  # structural, can force speed restriction/closure
    "Cess/formation erosion": 1.4,       # foundational/structural support risk
    "Level crossing gate fault": 1.5,    # direct road-safety hazard
    "Sleeper damage": 1.3,
    "Rail corrugation": 1.2,
    # Signalling (SMMS)
    "Interlocking relay fault": 2.0,   # can force manual signalling, major delay
    "Track circuit failure": 1.8,      # forces caution/manual block working
    "Point machine fault": 1.6,
    "Axle counter fault": 1.5,
    "Signal cable fault": 1.3,
    "Signal lamp failure": 1.1,
    "Block instrument fault": 1.9,       # affects inter-station block working
    "Level crossing interlocking fault": 1.9,  # direct road/rail collision risk
    "Point detection circuit fault": 1.7,  # direct derailment risk if mis-detected
    "Panel/VDU fault": 1.3,
    "LED signal aspect fault": 1.1,
    "Data logger fault": 1.0,            # not a live safety risk on its own
    # Traction (TDMS)
    "Feeder cable fault": 1.9,         # can suspend electric traction on the section
    "Insulator flashover damage": 1.6,
    "OHE contact wire wear": 1.5,
    "Dropper/hanger damage": 1.4,
    "Tension length anomaly": 1.3,
    "Earthing/bonding fault": 1.1,
    "Overhead mast/structure damage": 2.1,  # structural, OHE collapse risk
    "Traction substation fault": 1.8,       # can escalate to wider power interruption
    "Auto-tensioning device fault": 1.4,
    "Pantograph-OHE interaction damage": 1.3,
    "Return conductor fault": 1.2,
    "Jumper wire fault": 1.1,
}

DEFAULT_IMPACT_WEIGHT = 1.0

_ALL_DEFECT_TYPES = {dt for types in DEFECT_TYPES.values() for dt in types}
_missing = _ALL_DEFECT_TYPES - set(IMPACT_WEIGHT)
if _missing:
    raise RuntimeError(
        f"IMPACT_WEIGHT is missing entries for defect types: {sorted(_missing)} "
        "-- every defect type the generator can produce must have an explicit weight."
    )


def score_impact(tasks: pd.DataFrame) -> pd.Series:
    """Return impact_weight aligned to `tasks`' index (needs `defect_type`)."""
    return tasks["defect_type"].map(IMPACT_WEIGHT).fillna(DEFAULT_IMPACT_WEIGHT)
