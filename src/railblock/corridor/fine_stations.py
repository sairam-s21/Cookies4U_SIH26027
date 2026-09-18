"""Fine-grained corridor station model (Session 4).

Session 3's fragmentation diagnosis traced part of the scheduling
bottleneck to coarse block sections: sections derived from major-station
stop patterns meant one saturated long section (e.g. MAS-AJJ, 68 km) blocks
scheduling across its whole length even when only part of it is actually
busy. Train 12243's full stop-by-stop timetable (see
fine_stations_source.py) gives real points on this exact corridor --
including signal cabins and block huts, which are genuine block-boundary
points, not noise -- letting block sections be rebuilt at a granularity
close to the real physical signalling layout instead of just major
stations.

Session 16, at explicit user request: project scope reduced to MAS-JTJ
only (see fine_stations_source.py) -- this module's own logic is
unchanged, it just now runs over a shorter real point list.

load_fine_corridor_stations() returns the same column shape as
derive_stations.derive_corridor_stations() (sequence_order, station_code,
station_name, distance_km, source) so it's a drop-in input to
sections.build_block_sections() and everything downstream.

cross_validate_against_coarse_derivation() checks this source against the
independently-derived 19-station list (train-union method over
Train_details_22122017.csv) on the km values for their shared stations --
two real, independently-sourced datasets agreeing validates both.
"""

from __future__ import annotations

import pandas as pd

from railblock.corridor.fine_stations_source import TRAIN_12243_STOPS


def load_fine_corridor_stations() -> pd.DataFrame:
    """The corridor station list from train 12243's real timetable, MAS to
    JTJ inclusive (Session 16 scope reduction), ordered by distance."""
    rows = [
        {
            "sequence_order": i + 1,
            "station_code": code,
            "station_name": name,
            "distance_km": km,
            "x_o_flag": x_o,
            "code_source": code_source,
            "source": "train_12243_timetable",
        }
        for i, (code, name, x_o, km, code_source) in enumerate(TRAIN_12243_STOPS)
    ]
    return pd.DataFrame(rows)


def cross_validate_against_coarse_derivation(
    fine_stations: pd.DataFrame, coarse_stations: pd.DataFrame
) -> pd.DataFrame:
    """For every station_code present in BOTH the fine (train 12243) and
    coarse (Train_details_22122017.csv cross-train union) derivations,
    compare their independently-computed distance_km. Returns one row per
    shared code with both values and the discrepancy -- reported for
    review, never silently reconciled.
    """
    merged = fine_stations[["station_code", "station_name", "distance_km"]].merge(
        coarse_stations[["station_code", "distance_km"]],
        on="station_code",
        how="inner",
        suffixes=("_fine_12243", "_coarse_union"),
    )
    merged["discrepancy_km"] = (
        merged["distance_km_fine_12243"] - merged["distance_km_coarse_union"]
    ).round(3)
    return merged.sort_values("distance_km_fine_12243").reset_index(drop=True)


def main() -> None:
    from railblock.corridor.derive_stations import derive_corridor_stations
    from railblock.corridor.sections import build_block_sections
    from railblock.paths import DERIVED_DIR

    fine = load_fine_corridor_stations()
    coarse, _ = derive_corridor_stations()

    comparison = cross_validate_against_coarse_derivation(fine, coarse)
    print(f"Fine model: {len(fine)} points. Coarse model: {len(coarse)} stations.")
    print(f"\nCross-validation on {len(comparison)} shared stations:")
    print(comparison.to_string(index=False))
    print(f"\nMax |discrepancy|: {comparison['discrepancy_km'].abs().max():.3f} km")
    print(f"Mean |discrepancy|: {comparison['discrepancy_km'].abs().mean():.3f} km")

    fine_path = DERIVED_DIR / "fine_corridor_stations.csv"
    fine.to_csv(fine_path, index=False)
    print(f"\nWrote {fine_path}")

    sections = build_block_sections(fine)
    sections_path = DERIVED_DIR / "fine_block_sections.csv"
    sections.to_csv(sections_path, index=False)
    print(f"Built {len(sections)} fine block sections -> {sections_path}")


if __name__ == "__main__":
    main()
