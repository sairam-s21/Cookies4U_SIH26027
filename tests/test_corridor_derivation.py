import pandas as pd
import pytest

from railblock.corridor.derive_stations import (
    KNOWN_VERIFIED_9,
    derive_corridor_stations,
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


def test_malformed_rows_are_dropped_not_silently_kept(derived):
    _, n_dropped = derived
    # the Dec-2017 snapshot has exactly 10 known-malformed rows (unrelated
    # UBL/BJP trains); if this changes, the dataset itself has changed.
    assert n_dropped == 10
