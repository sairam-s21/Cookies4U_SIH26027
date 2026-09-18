"""SYNTHETIC, real-anchored historical block-utilization dataset, at
explicit user request: a history of *completed* maintenance blocks (one
row per block) recording how much time was demanded, how much was
sanctioned, and how much was actually utilized -- feeding the adaptive-
allocation regression step (railblock.ml.train_utilization_model /
predict_allocation) that later tries a historically-safe reduced
allocation for a task that cannot fit its full demanded duration anywhere
in the horizon, instead of leaving it in the waiting list untouched.

No sufficiently large public dataset of (demanded, sanctioned, actually-
utilized) block durations exists for Indian Railways (confirmed --
consistent with this project's established data inventory). Every row
generated here is synthetic and tagged data_source="SYNTHETIC_REAL_ANCHORED",
but the DISTRIBUTIONS used to generate it are not invented outright --
they are fitted to real numbers wherever real numbers exist, with every
extension beyond that real anchor disclosed explicitly below rather than
silently blended in.

=====================================================================
WHAT IS REAL, AND EXACTLY HOW IT WAS USED
=====================================================================

1. Two real COA block-request exports already in datasets/:
   - "blocks_04-09-2026 (1).xlsx" (94 real rows, including a real
     Availed Time column -- i.e. real ACTUAL utilized duration -- for 73
     rows with Status == "Block Closed")
   - "block_requests_04-09-2026_to_04-09-2026.xlsx" (66 real rows,
     Demanded/Sanctioned only, no Availed Time)

   From these, two real ratios were computed directly and used to fit
   this module's department-level distribution parameters:

   a) sanction_ratio = Sanctioned Time / Demanded Time, computed for
      every row with both fields present across BOTH files combined
      (160 rows total after excluding a handful of zero/near-3x parsing
      outliers), fitted as a truncated-normal per department:

         Engineering:  n=74  mean=0.837  std=0.199  (COA usually
                        sanctions LESS than demanded for Engineering)
         Signalling:   n=29  mean=1.092  std=0.266  (COA often sanctions
                        AT OR ABOVE demanded for Signalling)
         Traction:     n=55  mean=0.947  std=0.379  (close to demanded,
                        widest spread of the three)

   b) utilization_pct = Availed Time / Sanctioned Time, computed for the
      73 "Block Closed" rows in file (a) with both fields present and a
      non-zero sanctioned duration (72 usable rows after that filter),
      fitted as a LOGNORMAL per department (a ratio with a long right
      tail -- observed range was 0% to 410% -- is a natural lognormal
      shape, not a normal one):

         Engineering:  n=35  mu_log=-0.0897  sigma_log=0.6571
                        (implied median ~91% -- mixed, mostly at-or-
                        under sanctioned time)
         Signalling:   n=13  mu_log=-0.6995  sigma_log=0.8438
                        (implied median ~50% -- clearly underruns on
                        average, by far the WIDEST relative spread of
                        the three, several real rows from 9% to 96%)
         Traction:     n=21  mu_log=0.0332   sigma_log=0.7072
                        (implied median ~103% -- the only department
                        whose real median is an OVERRUN, not an underrun)

      Signalling's n=13 and Traction's n=21 are genuinely thin samples --
      real, but not large. Treat the department-level shape as real-
      grounded, not as a precise population estimate.

   c) A SECOND, independent real anchor for Engineering's sanction_ratio
      only: CAG Report No. 22 of 2022 ("Performance Audit on Derailments
      in Indian Railways"), Table 2.9.1 -- yard-line maintenance blocks,
      5 Zonal Railways, 2017-18 to 2020-21: 7339:28 hours demanded,
      4667:56 hours granted = 63.60% granted, a real, independently-
      audited, national figure -- but for a narrower category (yard-line
      maintenance specifically) than our own Excel rows cover, and
      notably lower than (a)'s 83.7%. Rather than picking one real
      source over the other, SANCTION_RATIO_ANCHOR_CAG + a 50/50
      per-row mixture (SANCTION_RATIO_MIXTURE_WEIGHT) is used so
      Engineering's generated sanction_ratio genuinely spans both real
      anchors instead of averaging them into a single new number neither
      source actually reported. No equivalent CAG figure was found for
      Signalling or Traction despite a genuine search.

2. Three real external sources, used ONLY to decide how each
   INDIVIDUAL defect type deviates from its department's real base
   distribution above (see DEFECT_TIER below) -- because no public
   dataset exists at the individual-defect-type level, and the two real
   Excel files have only 1-3 rows for most specific activities, far too
   thin to fit per-defect-type parameters directly:

   - Track tamping machines have a known, fixed productivity rate
     (sleepers/hour, metres/pass), meaning machine-based track work
     duration is largely calculable in advance from the length of track
     involved -- used to justify a NARROWER (more predictable) spread
     for machine/geometry-related Engineering defect types.
     https://www.qlrailwaymachine.com/info/efficient-railway-track-tamping-operations-103218771.html
   - A Southern Railway zonal training institute document (working of
     trains on electrified sections) describes a mandatory earthing +
     Authority-to-Work procedure before traction work, and removal of
     earths + verification before cancelling that authority afterward --
     real fixed procedural overhead at BOTH ends of a traction block,
     independent of the actual repair duration -- used to justify a
     HIGHER mean (more overrun-prone) for isolation-heavy Traction
     defect types.
     https://railnet.in/sr/mdzti/content/files/Chapter-17.pdf
   - Academic literature on railway signal-equipment fault diagnosis
     describes it as experience-based and non-standardized (unlike a
     fixed procedure) -- used to justify a WIDER spread for diagnostic-
     heavy Signalling defect types.
     https://thesai.org/Downloads/Volume13No12/Paper_79-Fault_Diagnosis_Technology_of_Railway_Signal_Equipment.pdf

   A fourth source -- a real "lost trains by cause, year-over-year"
   operational chart the user supplied (datasets/trains_improved_or_worsened.jpeg)
   -- showed "Planned Block Overrun" as the second-largest WORSENING
   cause (+83 lost trains) and Engineering-caused losses also worsening,
   while Signal & Telecom-caused losses improved sharply. Its exact
   scope (this corridor vs. a wider zone/all-India) was not confirmed
   before this module was written, so it is used only as qualitative
   corroboration that overrun risk is real and operationally significant
   enough to model seriously -- no number from it is used as a
   distribution parameter here.

3. Every defect-type-level adjustment beyond the two real department-
   level fits above (DEFECT_TIER) is a disclosed, reasoned judgement
   call informed by (2), not a separately-sourced statistic -- the same
   status DEFECT_DURATION_HOURS's own per-type judgement calls already
   have in railblock.synthetic.maintenance_tasks.

4. demanded_block_hours, section_id, criticality, and defect-type mix in
   the generated rows all reuse this project's EXISTING real-anchored
   generators (DEFECT_DURATION_HOURS's triangular distributions, the
   real corridor's own block sections, PRIORITY_WEIGHTS) rather than
   re-deriving a second, inconsistent set of numbers.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from railblock.paths import HISTORICAL_UTILIZATION_CSV
from railblock.synthetic.maintenance_tasks import (
    DEFECT_DURATION_HOURS,
    DEFECT_TYPES,
    DEPARTMENTS,
    PRIORITY_WEIGHTS,
    SOURCE_SYSTEM,
    _defect_type_probabilities,
)

DATA_SOURCE_TAG = "SYNTHETIC_REAL_ANCHORED"

# Real, fitted from the two COA Excel exports -- see module docstring
# section 1(a). Clipped at generation time to [0.3, 2.2] -- the real
# observed range across both files, with a small margin -- so an
# extreme tail sample never produces a nonsensical sanctioned duration.
SANCTION_RATIO_ANCHOR_EXCEL = {
    "Engineering": {"mean": 0.837, "std": 0.199},
    "Signalling": {"mean": 1.092, "std": 0.266},
    "Traction": {"mean": 0.947, "std": 0.379},
}
SANCTION_RATIO_CLIP = (0.3, 2.2)

# A second, independent real anchor for Engineering ONLY: CAG Report No.
# 22 of 2022 ("Performance Audit on Derailments in Indian Railways"),
# Table 2.9.1 -- real hours, extracted directly from the report's own
# text: yard-line maintenance blocks across 5 Zonal Railways, 2017-18 to
# 2020-21, demanded 7339:28 hours, granted 4667:56 hours = 63.60% granted.
# std=0.199 here REUSES the Excel anchor's std, not a separately-fitted
# one -- the CAG table only gives 4 year-level aggregate totals (whose
# own std across those 4 years is a much tighter 0.023, but that's a
# year-to-year aggregate spread, not a block-to-block one, and would
# understate real per-block variance) -- disclosed as a substitution,
# not a real block-level figure. This anchor is narrower in scope than
# our own Excel rows (yard-line maintenance specifically, not every
# Engineering defect type) but far larger in real sample (thousands of
# real hours vs. 74 real rows) and independently audited.
#
# No equivalent CAG figure was found for Signalling or Traction despite
# a genuine search (an S&T-specific CAG chapter existed but covered
# equipment procurement, not block hours) -- those two departments use
# the Excel anchor only.
#
# At explicit user request: rather than picking one real anchor over the
# other, sample from a 50/50 MIXTURE of both for Engineering -- each
# real SOURCE weighted equally, not weighted by each source's raw row
# count (the CAG figure's much larger n reflects a narrower, different
# sub-category, not simply "more data of the exact same population" our
# Excel rows cover, so letting it dominate by sample size alone would
# overstate its relevance). This is deliberately more honest than
# averaging the two into one new compromise number that neither real
# source actually reported.
SANCTION_RATIO_ANCHOR_CAG = {
    "Engineering": {"mean": 0.6360, "std": 0.199},
}
SANCTION_RATIO_MIXTURE_WEIGHT = 0.5  # weight on the CAG anchor, where a second anchor exists

# Real, fitted from the 72 usable "Block Closed" rows -- see module
# docstring section 1(b). log(utilization_pct/100) ~ Normal(mu, sigma).
UTILIZATION_LOGNORMAL_PARAMS = {
    "Engineering": {"mu": -0.0897, "sigma": 0.6571},
    "Signalling": {"mu": -0.6995, "sigma": 0.8438},
    "Traction": {"mu": 0.0332, "sigma": 0.7072},
}
# Real observed range was 0%-410%; clip generated values to a slightly
# wider band so the tails stay plausible without ever hitting exactly 0.
UTILIZATION_PCT_CLIP = (5.0, 450.0)

# Real, fitted from the 72 usable "Block Closed" rows: sanction_ratio and
# utilization_pct are NOT independent -- a block sanctioned generously
# relative to what was demanded tends to show LOWER utilization (the
# extra grant goes unused), not higher. Overall r=-0.255 (n=72),
# corroborated by the Signalling-specific subset alone (r=-0.239, n=13).
# Sampling the two independently (an earlier version of this module did)
# can produce "over-sanctioned AND high-utilization" rows that the real
# data says should be rare -- this correlation is applied via a shared
# bivariate-normal draw per row instead.
SANCTION_UTILIZATION_CORR = -0.25

# Per-defect-type deviation from the department base distribution above,
# informed by the three real external sources in the module docstring
# section 2 -- disclosed judgement calls, not separately-sourced
# statistics. mu_shift is added to the department's mu_log (in log-
# space, so +0.15 =~ +16% relative to the department median);
# sigma_mult scales the department's sigma_log. Any defect type not
# listed here uses its department's base distribution unchanged.
DEFECT_TIER: dict[str, dict[str, float]] = {
    # --- Engineering: machine-based/geometry work is predictable
    # (real tamping-productivity citation) -- narrower spread, no mean
    # shift. Manual/reactive repairs are the opposite -- wider spread.
    "Ballast deficiency": {"sigma_mult": 0.70},
    "Track geometry deviation": {"sigma_mult": 0.70},
    "Rail corrugation": {"sigma_mult": 0.75},
    "Rail fracture": {"sigma_mult": 1.20},
    "Weld defect": {"sigma_mult": 1.15},
    "Points & crossing wear": {"sigma_mult": 1.20},
    "Bridge/girder deterioration": {"sigma_mult": 1.25},
    "Cess/formation erosion": {"sigma_mult": 1.15},
    # --- Traction: isolation/earthing-heavy work overruns more (real
    # Authority-to-Work/earthing-procedure citation) -- mean shifted up.
    # Quick, localized fixes shifted down instead.
    "OHE contact wire wear": {"mu_shift": 0.15},
    "Pantograph-OHE interaction damage": {"mu_shift": 0.18},
    "Overhead mast/structure damage": {"mu_shift": 0.20},
    "Traction substation fault": {"mu_shift": 0.15},
    "Earthing/bonding fault": {"mu_shift": -0.12, "sigma_mult": 0.85},
    "Jumper wire fault": {"mu_shift": -0.10, "sigma_mult": 0.85},
    "Dropper/hanger damage": {"mu_shift": -0.08},
    # --- Signalling: diagnostic-heavy faults are unpredictable (real
    # fault-diagnosis-variability citation) -- wider spread. Simple
    # physical swaps are the opposite -- narrower spread, closer to 100%.
    "Track circuit failure": {"sigma_mult": 1.25},
    "Interlocking relay fault": {"sigma_mult": 1.30},
    "Point machine fault": {"sigma_mult": 1.20},
    "Level crossing interlocking fault": {"sigma_mult": 1.25},
    "Point detection circuit fault": {"sigma_mult": 1.15},
    "Signal lamp failure": {"mu_shift": 0.15, "sigma_mult": 0.65},
    "LED signal aspect fault": {"mu_shift": 0.15, "sigma_mult": 0.65},
    "Signal cable fault": {"mu_shift": 0.08, "sigma_mult": 0.80},
    "Data logger fault": {"mu_shift": 0.08, "sigma_mult": 0.75},
}

# Real, computed from the Department column of BOTH Excel files combined
# (n=160: 76 Engineering + 55 Traction + 29 Signalling real rows) --
# departments do not raise/close blocks equally often in the real data.
# Used to weight which department a generated row belongs to, instead of
# an equal-probability rng.choice(DEPARTMENTS) (which the live task
# generator in maintenance_tasks.py also uses -- an existing project-wide
# simplification, not something invented for this module specifically).
REAL_DEPARTMENT_WEIGHTS = {
    "Engineering": 76 / 160,
    "Traction": 55 / 160,
    "Signalling": 29 / 160,
}


def _department_for_defect(defect_type: str) -> str:
    for dep, types in DEFECT_TYPES.items():
        if defect_type in types:
            return dep
    raise KeyError(f"unknown defect_type: {defect_type!r}")


def generate_historical_utilization(
    sections: pd.DataFrame,
    n_records: int = 3000,
    end_date: date | None = None,
    lookback_days: int = 730,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate `n_records` synthetic-but-real-anchored completed-block
    history rows, spread over the `lookback_days` window ending
    `end_date` (defaults to today). `sections`: the real corridor's
    block_sections DataFrame (needs a section_id column) -- reused so
    every row's section is a real one, not invented.
    """
    if end_date is None:
        end_date = date.today()
    rng = np.random.default_rng(seed)

    section_ids = sections["section_id"].tolist()
    priorities = list(PRIORITY_WEIGHTS.keys())
    priority_p = list(PRIORITY_WEIGHTS.values())
    defect_p_by_department = {dep: _defect_type_probabilities(dep) for dep in DEPARTMENTS}

    # Two independent standard-normal draws per row, transformed so
    # z_sanction and z_utilization are correlated at SANCTION_UTILIZATION_CORR
    # (a standard Cholesky-style transform: corr(a, rho*a + sqrt(1-rho^2)*b) = rho
    # for independent standard normals a, b) -- see that constant's comment
    # for why this correlation exists and is real, not assumed.
    rho = SANCTION_UTILIZATION_CORR
    z_a = rng.standard_normal(n_records)
    z_b = rng.standard_normal(n_records)
    z_sanction = z_a
    z_utilization = rho * z_a + np.sqrt(1 - rho**2) * z_b
    # Which real anchor this row's sanction_ratio draws from, for
    # departments with a second one (see SANCTION_RATIO_ANCHOR_CAG) --
    # decided once per row, independent of z_sanction/z_utilization.
    anchor_draw = rng.random(n_records)

    department_p = [REAL_DEPARTMENT_WEIGHTS[d] for d in DEPARTMENTS]

    rows = []
    for i in range(n_records):
        department = rng.choice(DEPARTMENTS, p=department_p)
        defect_type = rng.choice(DEFECT_TYPES[department], p=defect_p_by_department[department])
        priority = rng.choice(priorities, p=priority_p)
        section_id = rng.choice(section_ids)

        day_offset = int(rng.integers(0, lookback_days))
        record_date = end_date - timedelta(days=day_offset)

        best_h, avg_h, worst_h = DEFECT_DURATION_HOURS[defect_type]
        demanded_hours = float(rng.triangular(best_h, avg_h, worst_h))

        cag_anchor = SANCTION_RATIO_ANCHOR_CAG.get(department)
        if cag_anchor is not None and anchor_draw[i] < SANCTION_RATIO_MIXTURE_WEIGHT:
            sp = cag_anchor
        else:
            sp = SANCTION_RATIO_ANCHOR_EXCEL[department]
        sanction_ratio = float(np.clip(sp["mean"] + sp["std"] * z_sanction[i], *SANCTION_RATIO_CLIP))
        sanctioned_hours = demanded_hours * sanction_ratio

        up = UTILIZATION_LOGNORMAL_PARAMS[department]
        tier = DEFECT_TIER.get(defect_type, {})
        mu = up["mu"] + tier.get("mu_shift", 0.0)
        sigma = up["sigma"] * tier.get("sigma_mult", 1.0)
        utilization_pct = float(np.clip(np.exp(mu + sigma * z_utilization[i]) * 100.0, *UTILIZATION_PCT_CLIP))
        actual_utilized_hours = sanctioned_hours * (utilization_pct / 100.0)

        rows.append(
            {
                "record_id": f"HIST-{i + 1:06d}",
                "date": record_date.isoformat(),
                "department": department,
                "source_system": SOURCE_SYSTEM[department],
                "defect_type": defect_type,
                "section_id": section_id,
                "criticality": priority,
                "demanded_block_hours": round(demanded_hours, 3),
                "sanctioned_block_hours": round(sanctioned_hours, 3),
                "actual_utilized_hours": round(actual_utilized_hours, 3),
                "sanction_ratio": round(sanction_ratio, 4),
                "utilization_pct": round(utilization_pct, 2),
                "data_source": DATA_SOURCE_TAG,
            }
        )

    return pd.DataFrame(rows)


def save_historical_utilization(df: pd.DataFrame, path=None) -> None:
    path = path or HISTORICAL_UTILIZATION_CSV
    df.to_csv(path, index=False)


if __name__ == "__main__":
    from railblock.api.app import get_corridor_context

    ctx = get_corridor_context()
    df = generate_historical_utilization(ctx.sections, n_records=3000, seed=13)
    save_historical_utilization(df)
    print(f"wrote {len(df)} rows to {HISTORICAL_UTILIZATION_CSV}")
    print()
    print("utilization_pct by department:")
    print(df.groupby("department")["utilization_pct"].describe()[["count", "mean", "50%", "std", "min", "max"]])
