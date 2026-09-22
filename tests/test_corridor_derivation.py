import pandas as pd
import pytest

from railblock.corridor.derive_stations import (
    KNOWN_VERIFIED_9,
    derive_corridor_stations,
    load_timetable,
)
from railblock.paths import TRAIN_DETAILS_CSV

pytestmark = pytest.mark.skipif(
    not TRAIN_DETAILS_CSV.exists(), reason="real dataset not present"
)


@pytest.fixture(scope="module")
def derived():
    stations, n_dropped = derive_corridor_stations()
    return stations, n_dropped


def test_derivation_contains_all_known_verified_stations(derived):
    stations, _ = derived
    derived_codes = set(stations["station_code"])
    missing = set(KNOWN_VERIFIED_9) - derived_codes
    assert not missing, f"derivation is missing known-verified stations: {missing}"


def test_derivation_is_deterministic():
    stations_a, _ = derive_corridor_stations()
    stations_b, _ = derive_corridor_stations()
    pd.testing.assert_frame_equal(
        stations_a.reset_index(drop=True), stations_b.reset_index(drop=True)
    )


def test_stations_ordered_by_ascending_distance(derived):
    stations, _ = derived
    distances = stations.sort_values("sequence_order")["distance_km"].tolist()
    assert distances == sorted(distances)
    assert distances[0] == 0.0  # MAS anchor


def test_mas_and_jtj_are_the_two_endpoints(derived):
    stations, _ = derived
    ordered = stations.sort_values("sequence_order")
    assert ordered.iloc[0]["station_code"] == "MAS"
    assert ordered.iloc[-1]["station_code"] == "JTJ"


def test_sequence_order_is_contiguous_from_one(derived):
    stations, _ = derived
    assert stations["sequence_order"].tolist() == list(range(1, len(stations) + 1))


def test_no_duplicate_station_codes(derived):
    stations, _ = derived
    assert stations["station_code"].is_unique


def test_malformed_rows_are_dropped_not_silently_kept(tmp_path):
    # Session 41: was asserting the real Train_details_22122017.csv has
    # exactly 10 known-malformed rows (unrelated Karnataka UBL/BJP/SUR/GDG
    # trains, per load_timetable()'s own docstring -- none of which touch
    # this corridor). That file is now trimmed to just the 741 real trains
    # that do touch it (see docs/GCP_DEPLOY.md's memory-reduction work),
    # so those 10 irrelevant rows are gone from the shipped dataset and
    # n_dropped is legitimately 0 there now. Verified here instead with a
    # synthetic malformed row, so this test exercises the real dropping
    # BEHAVIOR directly rather than depending on what a specific real
    # file's historical row count happens to be.
    csv_path = tmp_path / "synthetic_timetable.csv"
    csv_path.write_text(
        "Train No,Train Name,SEQ,Station Code,Station Name,Arrival time,"
        "Departure Time,Distance,Source Station,Source Station Name,"
        "Destination Station,Destination Station Name\n"
        "100,TEST EXP,1,AAA,STATION A,00:00:00,00:05:00,0,AAA,STATION A,BBB,STATION B\n"
        "100,TEST EXP,2,BBB,STATION B,01:00:00,01:05:00,50,AAA,STATION A,BBB,STATION B\n"
        # malformed: an embedded newline shifted the fields, so SEQ/Distance
        # hold a time string instead of a number -- the exact real-world
        # corruption load_timetable()'s docstring describes.
        "101,BROKEN EXP,x,CCC,STATION C,02:00:00,x,00:10:00,AAA,STATION A,BBB,STATION B\n"
    )
    df, n_dropped = load_timetable(csv_path)
    assert n_dropped == 1
    assert len(df) == 2
    assert set(df["Train No"]) == {"100"}
