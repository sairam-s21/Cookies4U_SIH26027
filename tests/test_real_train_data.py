"""Session 13: tests for the real-train-data rebuild pipeline (RailRadar
running-days + per-station schedule, replacing the 2017 CSV for exactly
the trains fetch_real_train_data.py validates) -- see that module and
railblock.corridor.real_overrides.
"""

from unittest.mock import patch

import pandas as pd
import pytest

import railblock.integrations.railradar as rr
from railblock.corridor.real_overrides import apply_real_train_overrides
from railblock.integrations.fetch_real_train_data import _minute_to_hhmmss, _validate_and_normalize

# The exact real /live payload shape confirmed by the user for train 12243
# (2026-09-06) -- trimmed to the fields get_train_static_profile touches.
REAL_12243_PAYLOAD = {
    "trainNumber": "12243",
    "trainName": "Chennai Central - Coimbatore Shatabdi Express",
    "train": {
        "type": "Shatabdi Express",
        "source": {"code": "MAS", "name": "MGR Chennai Central"},
        "destination": {"code": "CBE", "name": "Coimbatore Jn"},
        "runDays": ["mon", "wed", "thu", "fri", "sat", "sun"],
        "distance": 493.2,
        "duration": 420,
        "returnTrain": "12244",
    },
    "route": [
        {"sequence": 1, "stationCode": "MAS", "distance": 0, "scheduledDeparture": "2026-09-06T07:15:00+05:30"},
        {"sequence": 2, "stationCode": "BBQ", "distance": 2.3,
         "scheduledArrival": "2026-09-06T07:17:00+05:30", "scheduledDeparture": "2026-09-06T07:17:00+05:30"},
        {"sequence": 69, "stationCode": "CBE", "distance": 493.2, "scheduledArrival": "2026-09-06T14:15:00+05:30"},
    ],
}


def test_get_train_static_profile_parses_real_payload():
    with patch.object(rr, "_get", return_value=REAL_12243_PAYLOAD):
        profile = rr.get_train_static_profile("12243")
    assert profile["train_no"] == "12243"
    assert profile["run_days"] == frozenset({0, 2, 3, 4, 5, 6})  # skips Tuesday (1)
    assert [s["station_code"] for s in profile["stops"]] == ["MAS", "BBQ", "CBE"]
    assert profile["stops"][0]["scheduled_departure_minute"] == 435.0  # 07:15
    assert profile["stops"][2]["scheduled_arrival_minute"] == 855.0  # 14:15


def test_get_train_static_profile_returns_none_on_fetch_failure():
    with patch.object(rr, "_get", return_value=None):
        assert rr.get_train_static_profile("99999") is None


def test_get_train_static_profile_handles_missing_rundays():
    payload = {**REAL_12243_PAYLOAD, "train": {**REAL_12243_PAYLOAD["train"], "runDays": None}}
    with patch.object(rr, "_get", return_value=payload):
        profile = rr.get_train_static_profile("12243")
    assert profile["run_days"] is None


def test_minute_to_hhmmss_formats_correctly():
    assert _minute_to_hhmmss(435.0) == "07:15:00"
    assert _minute_to_hhmmss(855.0) == "14:15:00"
    assert _minute_to_hhmmss(None) is None


def test_validate_and_normalize_accepts_genuine_corridor_train():
    with patch.object(rr, "_get", return_value=REAL_12243_PAYLOAD):
        profile = rr.get_train_static_profile("12243")
    corridor_codes = {"MAS", "BBQ", "CBE", "ED", "TUP"}
    normalized = _validate_and_normalize(profile, corridor_codes)
    assert normalized is not None
    assert normalized["train_no"] == "12243"
    assert len(normalized["stops"]) == 3


def test_validate_and_normalize_rejects_mismatched_train():
    """The train-11028 scenario: a real, valid RailRadar response whose
    route simply doesn't overlap this corridor (at most one shared,
    coincidental station) must be discarded, not silently trusted."""
    payload = {
        "trainNumber": "11028",
        "trainName": "Dadar Central Express",
        "train": {"runDays": ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]},
        "route": [
            {"sequence": 1, "stationCode": "CSTM", "scheduledDeparture": "2026-09-06T08:00:00+05:30"},
            {"sequence": 2, "stationCode": "PUNE", "scheduledArrival": "2026-09-06T12:00:00+05:30"},
            {"sequence": 3, "stationCode": "SUR", "scheduledArrival": "2026-09-06T16:00:00+05:30"},
        ],
    }
    with patch.object(rr, "_get", return_value=payload):
        profile = rr.get_train_static_profile("11028")
    corridor_codes = {"MAS", "BBQ", "CBE", "ED", "TUP"}
    assert _validate_and_normalize(profile, corridor_codes) is None


def test_validate_and_normalize_requires_at_least_two_overlapping_stops():
    """A single shared station (e.g. a junction many unrelated trains
    pass through) proves nothing -- must not validate on its own."""
    payload = {
        "trainNumber": "55555",
        "trainName": "Unrelated Train",
        "train": {"runDays": ["mon"]},
        "route": [
            {"sequence": 1, "stationCode": "ED", "scheduledDeparture": "2026-09-06T08:00:00+05:30"},
            {"sequence": 2, "stationCode": "XYZ", "scheduledArrival": "2026-09-06T12:00:00+05:30"},
        ],
    }
    with patch.object(rr, "_get", return_value=payload):
        profile = rr.get_train_static_profile("55555")
    corridor_codes = {"MAS", "BBQ", "CBE", "ED", "TUP"}
    assert _validate_and_normalize(profile, corridor_codes) is None


def test_apply_real_train_overrides_replaces_only_matched_trains(tmp_path, monkeypatch):
    real_csv = tmp_path / "real_train_details_2026.csv"
    real_csv.write_text(
        "Train No,Train Name,SEQ,Station Code,Station Name,Arrival time,Departure Time,Distance,"
        "Source Station,Source Station Name,Destination Station,Destination Station Name\n"
        "12243,REAL SHATABDI,1,MAS,Chennai,07:15:00,07:15:00,0,MAS,Chennai,CBE,Coimbatore\n"
        "12243,REAL SHATABDI,2,CBE,Coimbatore,14:15:00,14:15:00,493,MAS,Chennai,CBE,Coimbatore\n"
    )
    import railblock.corridor.real_overrides as ro
    monkeypatch.setattr(ro, "REAL_TRAIN_DETAILS_CSV", real_csv)

    old_timetable = pd.DataFrame({
        "Train No": ["12243", "12243", "99999"],
        "Train Name": ["OLD 2017 NAME", "OLD 2017 NAME", "UNRELATED TRAIN"],
        "SEQ": [1, 2, 1],
        "Station Code": ["MAS", "CBE", "XYZ"],
        "Arrival time": ["07:10:00", "14:20:00", "10:00:00"],
        "Departure Time": ["07:10:00", "14:20:00", "10:00:00"],
        "Distance": [0, 495, 0],
    })

    merged = apply_real_train_overrides(old_timetable)

    train_99999 = merged[merged["Train No"] == "99999"]
    assert len(train_99999) == 1
    assert train_99999.iloc[0]["Train Name"] == "UNRELATED TRAIN"  # untouched

    train_12243 = merged[merged["Train No"] == "12243"]
    assert len(train_12243) == 2
    assert set(train_12243["Train Name"]) == {"REAL SHATABDI"}  # replaced, not merged with old
    assert train_12243[train_12243["Station Code"] == "MAS"].iloc[0]["Departure Time"] == "07:15:00"


def test_apply_real_train_overrides_passthrough_when_no_real_data(tmp_path, monkeypatch):
    import railblock.corridor.real_overrides as ro
    monkeypatch.setattr(ro, "REAL_TRAIN_DETAILS_CSV", tmp_path / "does_not_exist.csv")

    df = pd.DataFrame({"Train No": ["1"], "Station Code": ["MAS"]})
    result = apply_real_train_overrides(df)
    pd.testing.assert_frame_equal(result, df)
