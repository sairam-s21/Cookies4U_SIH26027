"""ONE-TIME script -- fetches train 12243's real route
geometry from RailRadar and caches it to
data/derived/real_route_geometry.json. Train 12243 is the same train
whose real stop-by-stop timetable defines this project's entire
fine-grained corridor model (see corridor/fine_stations_source.py), so
its route geometry IS the corridor's real physical track shape end to
end -- one call covers the whole corridor, not one per train.

NOT run automatically on startup (costs one real API call against the
1,000/month free-tier quota) -- run manually, once:

    python -m railblock.integrations.fetch_real_route_geometry

The Dashboard corridor map (CorridorMap.jsx) draws this real curve as the
base track if the cache file exists, falling back to straight lines
between the 19 major stations (the previous approximation) if it doesn't.
"""

from __future__ import annotations

import json

from railblock.integrations.railradar import get_route_geometry, is_configured
from railblock.paths import DERIVED_DIR

TRAIN_NO = "12243"
OUTPUT_PATH = DERIVED_DIR / "real_route_geometry.json"


def main() -> None:
    if not is_configured():
        print("RAILRADAR_API_KEY not set (see .env) -- nothing to do.")
        return

    print(f"Fetching real route geometry for train {TRAIN_NO}...")
    points = get_route_geometry(TRAIN_NO)
    if not points:
        print("No usable route geometry returned -- see railradar.py's parsing caveat; nothing written.")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(points, f)
    print(f"Wrote {len(points)} real track points -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
