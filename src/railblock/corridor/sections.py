"""Block section model: consecutive station pairs along the derived corridor.

Each block section is the unit that Layers 2-4 schedule maintenance blocks
against. Run derive_stations.main() first (or import build_block_sections
directly on an already-derived stations DataFrame).
"""

from __future__ import annotations

import pandas as pd

from railblock.paths import BLOCK_SECTIONS_CSV, CORRIDOR_STATIONS_CSV


def build_block_sections(stations: pd.DataFrame) -> pd.DataFrame:
    """Turn an ordered corridor station list into consecutive block sections.

    stations must be sorted by sequence_order/distance_km already (as
    produced by derive_corridor_stations). Returns a DataFrame with one row
    per section: section_id, from_code, from_name, to_code, to_name,
    from_km, to_km, length_km, sequence_order.
    """
    stations = stations.sort_values("sequence_order").reset_index(drop=True)
    rows = []
    for i in range(len(stations) - 1):
        a = stations.iloc[i]
        b = stations.iloc[i + 1]
        rows.append(
            {
                "sequence_order": i + 1,
                "section_id": f"{a['station_code']}-{b['station_code']}",
                "from_code": a["station_code"],
                "from_name": a["station_name"],
                "to_code": b["station_code"],
                "to_name": b["station_name"],
                "from_km": a["distance_km"],
                "to_km": b["distance_km"],
                "length_km": round(b["distance_km"] - a["distance_km"], 3),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    stations = pd.read_csv(CORRIDOR_STATIONS_CSV)
    sections = build_block_sections(stations)
    sections.to_csv(BLOCK_SECTIONS_CSV, index=False)
    print(f"Built {len(sections)} block sections from {len(stations)} stations -> {BLOCK_SECTIONS_CSV}")
    print(sections.to_string(index=False))


if __name__ == "__main__":
    main()
