import pytest

from railblock.availability.service_frequency import (
    ALL_WEEKDAYS,
    SERVICE_TYPE_FREQUENCY_MIX,
    assign_weekdays,
    classify_service_type,
)


def test_classify_service_type_detects_special():
    assert classify_service_type("MAS-CBE SPL") == "special"
    assert classify_service_type("JP-HYB SPECI") == "special"
    assert classify_service_type("MAO-KOP SPEC") == "special"


def test_classify_service_type_detects_passenger_local():
    assert classify_service_type("SA-ED PASS") == "passenger_local"
    assert classify_service_type("MAS MEMU") == "passenger_local"


def test_classify_service_type_defaults_to_mail_express():
    assert classify_service_type("MAS-CBE CHER") == "mail_express"
    assert classify_service_type("") == "mail_express"


def test_assign_weekdays_is_deterministic():
    a = assign_weekdays("12673", "MAS-CBE CHER")
    b = assign_weekdays("12673", "MAS-CBE CHER")
    assert a == b


def test_assign_weekdays_different_trains_can_differ():
    patterns = {assign_weekdays(str(n), "MAS-CBE EXP") for n in range(1000, 1050)}
    assert len(patterns) > 1  # not everyone gets the same pattern


def test_assign_weekdays_never_empty():
    for n in range(2000, 2100):
        wd = assign_weekdays(str(n), "MAS-QLN SPL")
        assert len(wd) >= 2  # even bi-weekly has 2 days
        assert wd <= ALL_WEEKDAYS


def test_special_trains_are_never_daily():
    for n in range(3000, 3200):
        wd = assign_weekdays(str(n), "MAS-CBE SPL")
        assert wd != ALL_WEEKDAYS


def test_frequency_mix_distribution_roughly_matches_documented_proportions():
    n = 4000
    daily_count = sum(
        1 for i in range(n) if assign_weekdays(str(10000 + i), "MAS-CBE EXP") == ALL_WEEKDAYS
    )
    observed = daily_count / n
    expected = next(cutoff for cutoff, name in SERVICE_TYPE_FREQUENCY_MIX["mail_express"] if name == "daily")
    assert observed == pytest.approx(expected, abs=0.05)
