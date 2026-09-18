import pandas as pd
import pytest

from railblock.corridor.sections import build_block_sections


@pytest.fixture
def toy_stations():
    return pd.DataFrame(
        [
            {"sequence_order": 1, "station_code": "MAS", "station_name": "CHENNAI CENTRAL", "distance_km": 0.0},
            {"sequence_order": 2, "station_code": "AJJ", "station_name": "ARAKKONAM JN", "distance_km": 68.0},
            {"sequence_order": 3, "station_code": "CBE", "station_name": "COIMBATORE", "distance_km": 493.0},
        ]
    )


def test_n_sections_is_n_stations_minus_one(toy_stations):
    sections = build_block_sections(toy_stations)
    assert len(sections) == len(toy_stations) - 1


def test_sections_are_consecutive_pairs_in_order(toy_stations):
    sections = build_block_sections(toy_stations)
    assert sections.iloc[0]["from_code"] == "MAS"
    assert sections.iloc[0]["to_code"] == "AJJ"
    assert sections.iloc[1]["from_code"] == "AJJ"
    assert sections.iloc[1]["to_code"] == "CBE"


def test_section_lengths_are_positive_and_sum_to_total(toy_stations):
    sections = build_block_sections(toy_stations)
    assert (sections["length_km"] > 0).all()
    assert sections["length_km"].sum() == pytest.approx(493.0)


def test_section_id_format(toy_stations):
    sections = build_block_sections(toy_stations)
    assert sections.iloc[0]["section_id"] == "MAS-AJJ"


def test_real_corridor_produces_one_fewer_section_than_stations():
    # Session 16: scope reduced to MAS-JTJ -- the real derived count moved
    # from 19 stations/18 sections to whatever the train-union derivation
    # finds within the new, shorter span, so this only asserts the
    # invariant (n_sections == n_stations - 1), not a specific count.
    from railblock.paths import CORRIDOR_STATIONS_CSV

    if not CORRIDOR_STATIONS_CSV.exists():
        pytest.skip("corridor_stations.csv not generated yet")
    stations = pd.read_csv(CORRIDOR_STATIONS_CSV)
    sections = build_block_sections(stations)
    assert len(sections) == len(stations) - 1
    assert stations.sort_values("sequence_order").iloc[-1]["station_code"] == "JTJ"
