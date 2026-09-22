"""Minimal .env loader -- avoids pulling in a new dependency
(python-dotenv) just for a couple of keys. Reads PROJECT_ROOT/.env
(gitignored, local-only) into os.environ on import, without overriding a
variable already set in the real environment. Safe to import even if
.env doesn't exist.
"""

from __future__ import annotations

import os

from railblock.paths import PROJECT_ROOT

_ENV_PATH = PROJECT_ROOT / ".env"


def _load_dotenv() -> None:
    if not _ENV_PATH.exists():
        return
    for line in _ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

RAILRADAR_API_KEY = os.environ.get("RAILRADAR_API_KEY")

# PostgreSQL is used for all app state (requests/approved/recommendations/
# schedule_options) -- see railblock.api.store. Default matches the local
# `railblock` role/database created for development; override via
# DATABASE_URL in .env for any other environment.
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://railblock:railblock@localhost:5432/railblock")

# CP-SAT's own thread count per solve (model.py) and how many of
# schedule_options.py's 3 strategies run concurrently. Both default to
# values tuned for a real multi-core dev machine -- 8 search workers x up
# to 3 concurrent solves is a reasonable ask on real hardware, but is a
# severe mismatch for a constrained free-tier host (Render's free plan is
# ~0.1 CPU / 512MB): confirmed live that this combination can starve the
# whole process (every other request on the same instance stalls for
# minutes, not just the scheduling one) rather than merely running slower.
# Override both via env vars on a constrained host without touching code;
# local dev and any better-resourced deploy keep the full defaults.
CPSAT_SEARCH_WORKERS = int(os.environ.get("CPSAT_SEARCH_WORKERS", "8"))
SCHEDULE_OPTIONS_MAX_CONCURRENCY = int(os.environ.get("SCHEDULE_OPTIONS_MAX_CONCURRENCY", "3"))
# schemas.py's ScheduleOptionsRequest.time_limit_s field default -- lower
# this on a constrained host alongside SCHEDULE_OPTIONS_MAX_CONCURRENCY=1
# (sequential strategies) so worst-case total wall time (n_strategies x
# this value) stays bounded, instead of multiplying the existing 30s.
SCHEDULE_OPTIONS_TIME_LIMIT_S = float(os.environ.get("SCHEDULE_OPTIONS_TIME_LIMIT_S", "30.0"))
