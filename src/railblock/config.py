"""Session 12: minimal .env loader -- no new dependency (python-dotenv)
added just for one key. Reads PROJECT_ROOT/.env (gitignored, local-only)
into os.environ on import, without overriding a variable already set in
the real environment. Safe to import even if .env doesn't exist.
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

# Session 13: PostgreSQL replaces SQLite entirely for app state (requests/
# approved/recommendations/schedule_options) -- see railblock.api.store.
# Default matches the local `railblock` role/database this session's setup
# created (see PROGRESS.md); override via DATABASE_URL in .env for any
# other environment.
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://railblock:railblock@localhost:5432/railblock")
