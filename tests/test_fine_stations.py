import pytest

from railblock.corridor.derive_stations import KNOWN_VERIFIED_9, TRAIN_DETAILS_CSV
from railblock.corridor.fine_stations import (
    cross_validate_against_coarse_derivation,
    load_fine_corridor_stations,
)
from railblock.corridor.sections import build_block_sections


def test_fine_stations_has_57_unique_points():
    # Session 16: scope reduced from MAS-CBE (102 points) to MAS-JTJ (57).
    fine = load_fine_corridor_stations()
    assert len(fine) == 57
    assert fine["station_code"].is_unique


def test_fine_stations_distance_strictly_increasing():
    fine = load_fine_corridor_stations()
    distances = fine.sort_values("sequence_order")["distance_km"].tolist()
    assert distances == sorted(distances)
    assert len(set(distances)) == len(distances)  # strictly increasing, no ties


def test_fine_stations_endpoints_are_mas_and_jtj():
    fine = load_fine_corridor_stations()
    ordered = fine.sort_values("sequence_order")
    assert ordered.iloc[0]["station_code"] == "MAS"
    assert ordered.iloc[0]["distance_km"] == 0.0
    assert ordered.iloc[-1]["station_code"] == "JTJ"


def test_fine_stations_contains_all_known_verified_9():
    fine = load_fine_corridor_stations()
    assert set(KNOWN_VERIFIED_9) <= set(fine["station_code"])


def test_fine_stations_flags_truncated_codes_distinctly():
    fine = load_fine_corridor_stations()
    inferred = fine[fine["code_source"] == "inferred_truncated"]
    assert len(inferred) == 1  # only ICF remains within the MAS-JTJ scope
    assert set(fine["code_source"]) <= {"timetable", "inferred_truncated"}


def test_build_block_sections_works_on_fine_stations():
    fine = load_fine_corridor_stations()
    sections = build_block_sections(fine)
    assert len(sections) == len(fine) - 1 == 56
    assert (sections["length_km"] > 0).all()
    assert sections["length_km"].sum() == pytest.approx(214.1)


@pytest.mark.skipif(not TRAIN_DETAILS_CSV.exists(), reason="real dataset not present")
def test_cross_validation_against_coarse_derivation_agrees_closely():
    from railblock.corridor.derive_stations import derive_corridor_stations

    fine = load_fine_corridor_stations()
    coarse, _ = derive_corridor_stations()
    comparison = cross_validate_against_coarse_derivation(fine, coarse)

    # every one of the coarse model's stations should be found in the fine model
    assert len(comparison) == len(coarse)
    # two independent real sources should agree closely, not exactly (different
    # kilometrage-table conventions/eras -- see Session 1's own ~1-3km note)
    assert comparison["discrepancy_km"].abs().max() < 5.0
    assert comparison["discrepancy_km"].abs().mean() < 3.0
