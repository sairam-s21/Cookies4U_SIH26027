"""Option 1 -- realistic day-of-week service frequency, applied to every
train in the real timetable.

`Train_details_22122017.csv` has no day-of-week/run-frequency column (see
docs/MASTER_PROMPT_SIH_26027.md Section 5 and Session 1/2's PROGRESS.md
notes). Session 1/2 treated every train as running literally every day,
which Session 2 showed collapses free-window time to near zero on several
sections (e.g. ED-TUP: 0 free minutes across an entire week).

We looked for a reliable per-train frequency signal in this dataset before
resorting to a statistical assumption. The only usable signal is the train
NAME: a small number of services are explicitly marked "SPL"/"SPECIAL"/
"SPEC" (59+9 = 68 trains dataset-wide), which in real Indian Railways
naming convention reliably means a non-daily (often weekly/seasonal)
service. That signal is real but sparse -- it correctly flags a handful of
trains and says nothing about the vast majority (the ~11,000 other
services, mostly Mail/Express/Passenger, whose names carry no frequency
marker at all). In reality a large fraction of Indian Railways mail/
express services do NOT run daily (commonly cited real-world figures put
daily-run mail/express services at roughly half of all such services, with
the rest running 5-6 days/week or on a fixed 2-3 day/week pattern) --
these do NOT get an explicit name marker, so name-matching alone cannot
recover their true frequency from this dataset. Per the session's
instruction, this is used as a DOCUMENTED STATISTICAL SUBSTITUTE, not a
per-train fact: each train is deterministically bucketed into a service
type from its name, then assigned a weekly running pattern by a stable
hash of its own train number against illustrative, stated real-world
proportions below. This is a reasonable, clearly-labelled assumption, not
a real per-train fact recovered from data -- it is not claimed to be more
than that anywhere downstream.

Determinism: the same (train_no, train_name) always yields the same
pattern (via a stable SHA-256-based hash), so the corridor's "operating
pattern" is fixed and reproducible across runs, not re-randomized -- this
is a documented reinterpretation of the real, frequency-less timetable,
not a fresh synthetic scenario like Session 1's task/goods generators.

Session 12: this assumption is now overridable, per train, by a REAL
answer fetched from the RailRadar API and cached to
data/derived/real_running_days.json -- see this module's
assign_weekdays(). A train with no cached real answer still falls back to
the assumption below exactly as before.

Session 13: the first attempt at fetching that confirmed answer guessed a
bare `GET /trains/{no}` endpoint that never worked (repeated 429s even
when the account's real remaining balance contradicted "quota exceeded",
then a connection reset -- consistent with it not being a real endpoint,
not a rate limit). In the meantime, a real live observation (RailRadar's
live tracker showing 12243 genuinely running on 2026-09-06, a Sunday)
directly falsified this module's statistical guess for that train
(Tue/Thu/Sat only) -- MANUAL_OVERRIDE_RUNNING_DAYS below was a stop-gap
"at least daily" correction for the two trains we had direct contrary
evidence for, added before the real endpoint was found.

The real running-days field turned out to be sitting in the
ALREADY-CONFIRMED-WORKING /live endpoint the whole time
(`data.train.runDays`) -- see railradar.get_train_static_profile and the
one-time fetch script railblock.integrations.fetch_real_train_data, which
populates real_running_days.json (checked first, taking priority over
both MANUAL_OVERRIDE_RUNNING_DAYS and the statistical assumption) for up
to 500 corridor trains at once.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache

from railblock.paths import DERIVED_DIR

ALL_WEEKDAYS = frozenset(range(7))  # Python date.weekday(): 0=Mon .. 6=Sun

REAL_RUNNING_DAYS_JSON = DERIVED_DIR / "real_running_days.json"

# See module docstring (Session 13) -- a one-point manual correction, not
# a RailRadar-confirmed per-day answer. Checked after the real cache (a
# genuinely confirmed answer, once fetched, should win) but before the
# statistical guess.
MANUAL_OVERRIDE_RUNNING_DAYS: dict[str, frozenset[int]] = {
    "12243": ALL_WEEKDAYS,
    "12244": ALL_WEEKDAYS,
}

# Illustrative real-world-informed weekly frequency mix per service type.
# Each entry is [(cumulative_probability, pattern_name), ...] -- NOT fitted
# to this dataset (no ground truth exists), a stated, documented assumption.
SERVICE_TYPE_FREQUENCY_MIX = {
    # Suburban/local/passenger services: almost always daily in practice.
    "passenger_local": [(0.90, "daily"), (1.00, "six_day")],
    # Mail/Express/Superfast and everything else unmarked -- the majority
    # bucket. Real IR mail/express services are commonly reported as only
    # ~half running daily; the rest run 6 days/week (one rest day) or a
    # fixed 3-day/week pattern (a very common long-distance express cadence).
    "mail_express": [(0.45, "daily"), (0.80, "six_day"), (1.00, "tri_weekly")],
    # Explicitly name-marked SPL/SPECIAL services: by IR convention these
    # are seasonal/occasional, essentially never daily.
    "special": [(0.00, "daily"), (0.30, "tri_weekly"), (1.00, "bi_weekly")],
}

# A few canonical spread-out weekday combinations for non-daily patterns
# (avoids e.g. picking two adjacent days for a "tri-weekly" pattern, which
# real IR scheduling avoids for even service spacing).
_TRI_WEEKLY_TRIPLETS = [(0, 2, 4), (1, 3, 5), (2, 4, 6), (0, 3, 5), (1, 4, 6)]
_BI_WEEKLY_PAIRS = [(0, 4), (1, 5), (2, 6), (0, 3), (2, 5)]


def _stable_unit_interval(key: str) -> float:
    """Deterministic pseudo-random value in [0, 1) from a stable hash of key."""
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def classify_service_type(train_name: str) -> str:
    name = (train_name or "").upper()
    if any(k in name for k in ("SPL", "SPECIAL", "SPEC")):
        return "special"
    if any(k in name for k in ("PASS", "MEMU", "EMU", "LOCAL")):
        return "passenger_local"
    return "mail_express"


@lru_cache(maxsize=1)
def _load_real_running_days() -> dict[str, frozenset[int]]:
    """Session 12: real running-days data fetched from RailRadar (see
    railblock.integrations.fetch_real_running_days) -- a train confirmed
    here overrides the statistical assumption below entirely. Returns {}
    (falls through to the assumption for every train) if the cache file
    doesn't exist yet -- this cache is opt-in, fetched manually, never
    required."""
    if not REAL_RUNNING_DAYS_JSON.exists():
        return {}
    with open(REAL_RUNNING_DAYS_JSON) as f:
        raw = json.load(f)
    return {train_no: frozenset(days) for train_no, days in raw.items()}


def assign_weekdays(train_no: str, train_name: str) -> frozenset[int]:
    """The set of weekdays (0=Mon..6=Sun) this train is treated as
    running on -- a real, RailRadar-confirmed answer if one has been
    fetched for this train_no (see _load_real_running_days), otherwise
    the documented statistical assumption below."""
    real = _load_real_running_days().get(str(train_no))
    if real is not None:
        return real

    override = MANUAL_OVERRIDE_RUNNING_DAYS.get(str(train_no))
    if override is not None:
        return override

    service_type = classify_service_type(train_name)
    mix = SERVICE_TYPE_FREQUENCY_MIX[service_type]

    r = _stable_unit_interval(f"{train_no}|{train_name}|pattern")
    pattern_name = next(name for cutoff, name in mix if r < cutoff)

    if pattern_name == "daily":
        return ALL_WEEKDAYS

    day_r = _stable_unit_interval(f"{train_no}|{train_name}|days")
    if pattern_name == "six_day":
        skip_day = int(day_r * 7) % 7
        return frozenset(d for d in range(7) if d != skip_day)
    if pattern_name == "tri_weekly":
        idx = int(day_r * len(_TRI_WEEKLY_TRIPLETS)) % len(_TRI_WEEKLY_TRIPLETS)
        return frozenset(_TRI_WEEKLY_TRIPLETS[idx])
    if pattern_name == "bi_weekly":
        idx = int(day_r * len(_BI_WEEKLY_PAIRS)) % len(_BI_WEEKLY_PAIRS)
        return frozenset(_BI_WEEKLY_PAIRS[idx])

    raise AssertionError(f"unhandled pattern {pattern_name}")  # pragma: no cover
