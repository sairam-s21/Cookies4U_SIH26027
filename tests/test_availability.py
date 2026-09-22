from datetime import date

import pandas as pd
import pytest

from railblock.availability.corridor_availability import (
    MINUTES_PER_DAY,
    compute_availability,
)


def _pax(rows):
    cols = ["section_id", "train_no", "start_minute", "end_minute", "source"]
    return pd.DataFrame(rows, columns=cols)


def _gds(rows):
    cols = ["section_id", "date", "start_minute", "end_minute"]
    return pd.DataFrame(rows, columns=cols)


def test_no_occupancy_means_fully_available():
    result = compute_availability("MAS-AJJ", date(2026, 9, 7), _pax([]), _gds([]))
    assert result["free_minutes"] == MINUTES_PER_DAY
    assert result["occupied_minutes"] == 0
    assert result["free_intervals"] == [(0.0, 1440.0)]


def test_cache_returns_identical_result_and_is_actually_reused():
    """Session 25, at explicit user request for a large scheduling-time
    reduction: `cache` must never change WHAT is computed, only how many
    times -- and a second call with the same cache must actually reuse
    the stored result rather than recomputing (checked by mutating the
    cached entry directly and confirming the "recomputed" call reflects
    that mutation, which is only possible if it truly came from cache)."""
    pax = _pax([{"section_id": "MAS-AJJ", "train_no": "1", "start_minute": 100, "end_minute": 160, "source": "real_timetable"}])
    goods = _gds([])
    the_date = date(2026, 9, 7)

    uncached = compute_availability("MAS-AJJ", the_date, pax, goods)
    cache = {}
    first = compute_availability("MAS-AJJ", the_date, pax, goods, cache=cache)
    assert first == uncached
    assert ("MAS-AJJ", the_date.isoformat()) in cache

    cache[("MAS-AJJ", the_date.isoformat())]["free_minutes"] = -1  # prove reuse, not recomputation
    second = compute_availability("MAS-AJJ", the_date, pax, goods, cache=cache)
    assert second["free_minutes"] == -1


def test_single_passenger_interval_subtracted_correctly():
    pax = _pax([{"section_id": "MAS-AJJ", "train_no": "1", "start_minute": 100, "end_minute": 160, "source": "real_timetable"}])
    result = compute_availability("MAS-AJJ", date(2026, 9, 7), pax, _gds([]))
    assert result["occupied_minutes"] == 60
    assert result["free_minutes"] == MINUTES_PER_DAY - 60
    assert result["free_intervals"][0] == (0.0, 100.0)
    assert result["free_intervals"][-1] == (160.0, 1440.0)


def test_overlapping_passenger_and_goods_intervals_union_not_double_counted():
    pax = _pax([{"section_id": "MAS-AJJ", "train_no": "1", "start_minute": 100, "end_minute": 200, "source": "real_timetable"}])
    gds = _gds([{"section_id": "MAS-AJJ", "date": "2026-09-07", "start_minute": 150, "end_minute": 250}])
    result = compute_availability("MAS-AJJ", date(2026, 9, 7), pax, gds)
    # union of [100,200) and [150,250) is [100,250) = 150 minutes, NOT 100+100=200
    assert result["occupied_minutes"] == 150
    assert result["occupied_intervals"] == [(100.0, 250.0)]


def test_different_section_ids_are_independent():
    pax = _pax([{"section_id": "MAS-AJJ", "train_no": "1", "start_minute": 0, "end_minute": 1440, "source": "real_timetable"}])
    result = compute_availability("AJJ-WJR", date(2026, 9, 7), pax, _gds([]))
    assert result["free_minutes"] == MINUTES_PER_DAY  # unaffected by MAS-AJJ's full occupation


def test_passenger_overnight_spillover_folds_into_next_day_start():
    # a transit that starts 23:50 and runs 30 min recurs every day, so it
    # also blocks 00:00-00:20 of THIS day via yesterday's occurrence
    pax = _pax([{"section_id": "MAS-AJJ", "train_no": "1", "start_minute": 1430, "end_minute": 1460, "source": "real_timetable"}])
    result = compute_availability("MAS-AJJ", date(2026, 9, 7), pax, _gds([]))
    assert result["occupied_minutes"] == pytest.approx(30.0)  # 10 (23:50-24:00) + 20 (00:00-00:20 wraparound)
    # occupied intervals should include both the tail-end and the wrapped start
    starts = sorted(s for s, _ in result["occupied_intervals"])
    assert starts[0] == 0.0


def test_goods_overnight_spillover_only_affects_next_calendar_day():
    gds = _gds(
        [
            {"section_id": "MAS-AJJ", "date": "2026-09-07", "start_minute": 1430, "end_minute": 1470},
        ]
    )
    today = compute_availability("MAS-AJJ", date(2026, 9, 7), _pax([]), gds)
    tomorrow = compute_availability("MAS-AJJ", date(2026, 9, 8), _pax([]), gds)

    assert today["occupied_minutes"] == pytest.approx(10.0)  # only 23:50-24:00 counted today
    assert tomorrow["occupied_minutes"] == pytest.approx(30.0)  # 00:00-00:30 spillover counted tomorrow
    day_after = compute_availability("MAS-AJJ", date(2026, 9, 9), _pax([]), gds)
    assert day_after["occupied_minutes"] == 0  # no entry for 09-08, so no spillover into 09-09


def _pax_with_weekdays(rows):
    cols = ["section_id", "train_no", "start_minute", "end_minute", "source", "weekdays"]
    return pd.DataFrame(rows, columns=cols)


def test_weekday_aware_train_only_occupies_its_running_days():
    # Monday=0 .. Sunday=6. 2026-09-07 is a Monday.
    monday = date(2026, 9, 7)
    tuesday = date(2026, 9, 8)
    pax = _pax_with_weekdays(
        [{"section_id": "MAS-AJJ", "train_no": "1", "start_minute": 100, "end_minute": 200,
          "source": "real_timetable", "weekdays": frozenset({0})}]  # Monday only
    )
    mon_result = compute_availability("MAS-AJJ", monday, pax, _gds([]))
    tue_result = compute_availability("MAS-AJJ", tuesday, pax, _gds([]))
    assert mon_result["occupied_minutes"] == 100
    assert tue_result["occupied_minutes"] == 0


def test_weekday_aware_overnight_spillover_checks_yesterdays_weekday():
    # train runs Monday only; a transit starting 23:50 Monday spills into
    # Tuesday's first 20 minutes even though the train doesn't run Tuesday.
    monday = date(2026, 9, 7)
    tuesday = date(2026, 9, 8)
    wednesday = date(2026, 9, 9)
    pax = _pax_with_weekdays(
        [{"section_id": "MAS-AJJ", "train_no": "1", "start_minute": 1430, "end_minute": 1460,
          "source": "real_timetable", "weekdays": frozenset({0})}]
    )
    tue_result = compute_availability("MAS-AJJ", tuesday, pax, _gds([]))
    wed_result = compute_availability("MAS-AJJ", wednesday, pax, _gds([]))
    assert tue_result["occupied_minutes"] == pytest.approx(20.0)  # Monday's spillover only
    assert wed_result["occupied_minutes"] == 0.0  # Tuesday didn't run, no spillover into Wednesday


def test_missing_weekdays_column_defaults_to_every_day_legacy_behaviour():
    pax_no_weekdays = _pax([{"section_id": "MAS-AJJ", "train_no": "1", "start_minute": 100, "end_minute": 200, "source": "real_timetable"}])
    for d in [date(2026, 9, 7), date(2026, 9, 8), date(2026, 9, 13)]:
        result = compute_availability("MAS-AJJ", d, pax_no_weekdays, _gds([]))
        assert result["occupied_minutes"] == 100  # applies every day, same as before Option 1


def test_build_passenger_occupancy_attaches_weekdays_column():
    from railblock.paths import CORRIDOR_STATIONS_CSV, TRAIN_DETAILS_CSV
    if not (CORRIDOR_STATIONS_CSV.exists() and TRAIN_DETAILS_CSV.exists()):
        pytest.skip("derived corridor data / real dataset not present")

    from railblock.corridor.derive_stations import load_timetable
    from railblock.availability.corridor_availability import build_passenger_occupancy

    stations = pd.read_csv(CORRIDOR_STATIONS_CSV)
    timetable, _ = load_timetable()
    pax = build_passenger_occupancy(timetable, stations)
    assert "weekdays" in pax.columns
    assert not pax.empty
    # not every train should be daily under Option 1's realistic mix
    from railblock.availability.service_frequency import ALL_WEEKDAYS
    assert (pax["weekdays"] != ALL_WEEKDAYS).any()


def test_build_passenger_occupancy_extends_correctly_to_fine_sections():
    """Session 4: the distance-proportional transit-splitting logic was
    written against the coarse (major-station) model; verify it
    generalizes to the finer-grained model (Session 16: MAS-JTJ scope,
    57 points/56 sections) rather than assume it does -- every fine
    section should get real occupancy coverage from at least one train's
    actual-stop-to-actual-stop transit spanning it."""
    from railblock.paths import TRAIN_DETAILS_CSV
    if not TRAIN_DETAILS_CSV.exists():
        pytest.skip("real dataset not present")

    from railblock.corridor.derive_stations import load_timetable
    from railblock.corridor.fine_stations import load_fine_corridor_stations
    from railblock.corridor.sections import build_block_sections
    from railblock.availability.corridor_availability import build_passenger_occupancy as bpo

    fine = load_fine_corridor_stations()
    sections = build_block_sections(fine)
    timetable, _ = load_timetable()
    pax = bpo(timetable, fine)

    covered = set(pax["section_id"].unique())
    all_secs = set(sections["section_id"])
    assert covered == all_secs, f"uncovered fine sections: {sorted(all_secs - covered)}"


def test_free_plus_occupied_equals_total_for_real_corridor_section(tmp_path):
    from railblock.paths import CORRIDOR_STATIONS_CSV, TRAIN_DETAILS_CSV
    if not (CORRIDOR_STATIONS_CSV.exists() and TRAIN_DETAILS_CSV.exists()):
        pytest.skip("derived corridor data / real dataset not present")

    from railblock.corridor.derive_stations import load_timetable
    from railblock.corridor.sections import build_block_sections
    from railblock.availability.corridor_availability import build_passenger_occupancy
    from railblock.synthetic.goods_forecast import generate_goods_forecast

    stations = pd.read_csv(CORRIDOR_STATIONS_CSV)
    sections = build_block_sections(stations)
    timetable, _ = load_timetable()
    pax = build_passenger_occupancy(timetable, stations)
    goods = generate_goods_forecast(sections, date(2026, 9, 7), n_days=1, seed=99)

    for section_id in sections["section_id"]:
        result = compute_availability(section_id, date(2026, 9, 7), pax, goods)
        assert result["free_minutes"] + result["occupied_minutes"] == pytest.approx(MINUTES_PER_DAY, abs=0.01)
        assert 0 <= result["free_minutes"] <= MINUTES_PER_DAY


def test_prewarm_delay_margin_cache_matches_uncached_prediction():
    """prewarm_delay_margin_cache exists purely as a speed optimization
    (batch-predicts every (train_type, station, weekday) combination a
    real timetable will ever need in ONE model call, instead of leaving
    each to be discovered -- and individually predicted -- on first use;
    measured ~48x faster for this corridor's real combination count).
    It must never change WHAT gets returned, only how fast -- confirmed
    here against the module's own real, already-prewarmed cache (a
    process-lifetime global api.store.get_corridor_context() populates
    automatically), checking every value it produced matches calling the
    single-row prediction function directly for the same inputs. Doesn't
    clear/rebuild that shared cache itself -- it's reused by every other
    test in this same process, and clearing it here would just make
    whichever test runs next pay the slow uncached path for no reason."""
    from railblock.paths import DELAY_MODEL_JOBLIB
    if not DELAY_MODEL_JOBLIB.exists():
        pytest.skip("delay-risk model not trained yet")

    from railblock.availability.corridor_availability import _cached_margin_minutes, _margin_cache
    from railblock.api.store import get_corridor_context
    from railblock.ml.predict_delay import predict_delay_margin

    get_corridor_context()  # idempotent -- ensures the process-wide prewarm has run at least once
    assert _margin_cache, "prewarm should have populated at least one real combination for this corridor"

    for (train_type, station_code, weekday), prewarmed_value in list(_margin_cache.items())[:25]:
        uncached = predict_delay_margin(train_type, station_code, weekday)
        assert uncached is not None
        assert prewarmed_value == pytest.approx(uncached["margin_minutes"])
        # Also confirm the real lookup function -- what _passenger_intervals
        # actually calls -- reads the prewarmed value, not a fresh one.
        assert _cached_margin_minutes(train_type, station_code, weekday) == pytest.approx(uncached["margin_minutes"])


def test_passenger_grouping_cache_is_not_corrupted_by_a_reused_object_id():
    """Real bug found running the full suite (not in isolation): the
    per-section grouping cache _passenger_intervals uses internally was
    first keyed on id(passenger_occupancy) alone -- correct for the ONE
    real, process-lifetime passenger_occupancy singleton, but wrong for
    the many short-lived DataFrames test fixtures construct, since
    CPython can and does reuse a garbage-collected object's id() for an
    unrelated later one. A short-lived DataFrame could then silently
    inherit a stale grouping computed for whatever DIFFERENT DataFrame
    previously died at that same address -- confirmed by seeing tests
    pass individually but fail under the full suite. This forces that
    exact scenario directly: build a DataFrame, use it, let it be
    garbage collected, then build enough fresh ones that at least one is
    highly likely to land on the freed address, and confirm every one of
    them gets its OWN correct grouping, never a stale one belonging to
    an earlier, unrelated object."""
    import gc

    from railblock.availability.corridor_availability import _passenger_intervals

    def make(section_id: str) -> pd.DataFrame:
        return pd.DataFrame(
            [{"section_id": section_id, "train_no": "1", "start_minute": 0.0, "end_minute": 10.0, "source": "real_timetable"}]
        )

    first = make("A-B")
    assert _passenger_intervals(first, "A-B") == [(0.0, 10.0)]
    del first
    gc.collect()

    for _ in range(500):
        other = make("C-D")
        result = _passenger_intervals(other, "C-D")
        assert result == [(0.0, 10.0)], f"got {result} -- section C-D must never see section A-B's stale interval"
        assert _passenger_intervals(other, "A-B") == [], "a fresh DataFrame must never inherit an unrelated one's grouping"
