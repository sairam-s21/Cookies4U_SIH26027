from datetime import date

import pandas as pd
import pytest

from railblock.synthetic.maintenance_tasks import (
    DEPARTMENTS,
    DEFECT_TYPES,
    DEMAND_SCENARIOS,
    GROUP_C_FRACTION,
    PRIORITY_WEIGHTS,
    REACTIVATION_DUE_WINDOW_DAYS,
    REACTIVATION_RAISE_LOOKBACK_DAYS,
    SPLITTABLE,
    generate_maintenance_tasks,
    n_tasks_for_scenario,
    refresh_demo_batch_dates,
)
from railblock.synthetic.goods_forecast import (
    HOUR_WEIGHTS,
    generate_goods_forecast,
)


@pytest.fixture
def toy_sections():
    return pd.DataFrame(
        [
            {"section_id": "MAS-AJJ", "length_km": 68.0},
            {"section_id": "AJJ-WJR", "length_km": 36.0},
            {"section_id": "BQI-SA", "length_km": 43.0},
        ]
    )


def test_maintenance_tasks_row_count_and_labelling(toy_sections):
    tasks = generate_maintenance_tasks(toy_sections, n_tasks=200, seed=1)
    assert len(tasks) == 200
    assert (tasks["data_source"] == "SYNTHETIC").all()


def test_maintenance_tasks_within_parameter_ranges(toy_sections):
    tasks = generate_maintenance_tasks(toy_sections, n_tasks=500, seed=2)

    assert set(tasks["department"]).issubset(set(DEPARTMENTS))
    assert set(tasks["requester_priority"]).issubset(set(PRIORITY_WEIGHTS))
    assert set(tasks["section_id"]).issubset(set(toy_sections["section_id"]))

    for _, row in tasks.iterrows():
        assert row["defect_type"] in DEFECT_TYPES[row["department"]]

    assert (tasks["days_overdue"] >= 0).all()
    assert (tasks["estimated_block_hours"] > 0).all()
    assert (pd.to_datetime(tasks["due_date"]) >= pd.to_datetime(tasks["raised_date"])).all()


def test_maintenance_tasks_priority_distribution_roughly_matches_weights(toy_sections):
    tasks = generate_maintenance_tasks(toy_sections, n_tasks=5000, seed=3)
    counts = tasks["requester_priority"].value_counts(normalize=True)
    for priority, expected in PRIORITY_WEIGHTS.items():
        assert counts[priority] == pytest.approx(expected, abs=0.04)


def test_splittable_covers_every_defect_type_with_bool_values(toy_sections):
    all_types = {dt for types in DEFECT_TYPES.values() for dt in types}
    assert set(SPLITTABLE) == all_types
    assert all(isinstance(v, bool) for v in SPLITTABLE.values())


def test_generated_tasks_carry_correct_splittable_flag(toy_sections):
    # Session 30, at explicit user request: every task is splittable now,
    # Critical included -- train cancellation (the only reason a Critical
    # task was ever marked non-splittable) is out of scope.
    tasks = generate_maintenance_tasks(toy_sections, n_tasks=5000, seed=5)
    assert tasks["splittable"].all()


def test_demand_scenarios_are_correctly_related():
    assert DEMAND_SCENARIOS["stress_test"] == n_tasks_for_scenario("stress_test")
    assert DEMAND_SCENARIOS["double_track_adjusted"] == n_tasks_for_scenario("double_track_adjusted")
    # double-track adjusted must be exactly the Group-C fraction of stress_test (within rounding)
    expected = round(DEMAND_SCENARIOS["stress_test"] * GROUP_C_FRACTION)
    assert DEMAND_SCENARIOS["double_track_adjusted"] == expected
    assert DEMAND_SCENARIOS["double_track_adjusted"] < DEMAND_SCENARIOS["stress_test"]


def test_n_tasks_for_scenario_rejects_unknown_name():
    with pytest.raises(ValueError):
        n_tasks_for_scenario("not_a_real_scenario")


def test_demand_scenario_overrides_n_tasks(toy_sections):
    tasks = generate_maintenance_tasks(toy_sections, n_tasks=999, seed=1, demand_scenario="double_track_adjusted")
    assert len(tasks) == DEMAND_SCENARIOS["double_track_adjusted"]


def test_no_demand_scenario_keeps_explicit_n_tasks_unchanged(toy_sections):
    tasks = generate_maintenance_tasks(toy_sections, n_tasks=37, seed=1)
    assert len(tasks) == 37


def test_maintenance_tasks_seed_is_reproducible(toy_sections):
    a = generate_maintenance_tasks(toy_sections, n_tasks=50, seed=42, as_of_date=date(2026, 9, 4))
    b = generate_maintenance_tasks(toy_sections, n_tasks=50, seed=42, as_of_date=date(2026, 9, 4))
    pd.testing.assert_frame_equal(a, b)


def test_maintenance_tasks_unseeded_calls_vary(toy_sections):
    a = generate_maintenance_tasks(toy_sections, n_tasks=50)
    b = generate_maintenance_tasks(toy_sections, n_tasks=50)
    assert not a["section_id"].equals(b["section_id"]) or not a["defect_type"].equals(b["defect_type"])


# ------------------------------------------------- refresh_demo_batch_dates

def _template_row(task_id, priority):
    return {
        "task_id": task_id, "department": "Engineering", "section_id": "MAS-AJJ",
        "requester_priority": priority, "estimated_block_hours": 2.0,
        # deliberately stale, CSV-import-time values -- refresh_demo_batch_dates
        # must overwrite these unconditionally, never trust them.
        "raised_date": "2020-01-01", "due_date": "2020-01-15", "days_overdue": 500,
    }


def test_refresh_demo_batch_dates_never_raises_the_task_after_the_click_date():
    as_of = date(2026, 10, 20)
    rows = [_template_row(f"T{i}", "Routine") for i in range(50)]
    refreshed = refresh_demo_batch_dates(rows, as_of)
    for r in refreshed:
        raised = date.fromisoformat(r["raised_date"])
        assert raised <= as_of
        assert (as_of - raised).days <= REACTIVATION_RAISE_LOOKBACK_DAYS


def test_refresh_demo_batch_dates_due_window_matches_priority_and_is_never_overdue_on_activation():
    as_of = date(2026, 10, 20)
    for priority, (lo, hi) in REACTIVATION_DUE_WINDOW_DAYS.items():
        rows = [_template_row(f"T-{priority}-{i}", priority) for i in range(30)]
        refreshed = refresh_demo_batch_dates(rows, as_of)
        for r in refreshed:
            raised = date.fromisoformat(r["raised_date"])
            due = date.fromisoformat(r["due_date"])
            offset = (due - raised).days
            assert lo <= offset <= hi
            # a freshly-activated task is never already overdue relative
            # to the real click date it was just anchored to -- see
            # Session 10's "never already overdue" convention.
            assert r["days_overdue"] == 0
            assert due >= as_of


def test_refresh_demo_batch_dates_is_deterministic_across_different_click_dates():
    # "the same set of tasks are generated each time" -- the SAME
    # template, activated on two DIFFERENT real calendar dates, must get
    # the same RELATIVE raised/due offsets both times, just anchored
    # differently.
    rows = [_template_row(f"T{i}", ["Critical", "Moderate", "Routine"][i % 3]) for i in range(70)]

    refreshed_a = refresh_demo_batch_dates(rows, date(2026, 9, 18))
    refreshed_b = refresh_demo_batch_dates(rows, date(2026, 11, 20))  # over a month later

    for a, b in zip(refreshed_a, refreshed_b):
        raised_offset_a = (date(2026, 9, 18) - date.fromisoformat(a["raised_date"])).days
        raised_offset_b = (date(2026, 11, 20) - date.fromisoformat(b["raised_date"])).days
        assert raised_offset_a == raised_offset_b

        due_offset_a = (date.fromisoformat(a["due_date"]) - date.fromisoformat(a["raised_date"])).days
        due_offset_b = (date.fromisoformat(b["due_date"]) - date.fromisoformat(b["raised_date"])).days
        assert due_offset_a == due_offset_b


def test_refresh_demo_batch_dates_does_not_mutate_the_input_rows():
    rows = [_template_row("T1", "Critical")]
    refresh_demo_batch_dates(rows, date(2026, 10, 20))
    assert rows[0]["raised_date"] == "2020-01-01"  # the original template row is untouched


def test_goods_forecast_row_shape_and_labelling(toy_sections):
    goods = generate_goods_forecast(toy_sections, date(2026, 9, 7), n_days=3, seed=1)
    assert set(goods["section_id"]).issubset(set(toy_sections["section_id"]))
    assert set(goods["date"]) == {"2026-09-07", "2026-09-08", "2026-09-09"}
    assert (goods["data_source"] == "SYNTHETIC").all()
    assert (goods["end_minute"] > goods["start_minute"]).all()
    assert (goods["duration_minutes"] > 0).all()


def test_goods_forecast_biased_toward_night_hours(toy_sections):
    goods = generate_goods_forecast(toy_sections, date(2026, 9, 7), n_days=30, seed=7)
    start_hour = (goods["start_minute"] // 60) % 24
    night = start_hour.isin([22, 23, 0, 1, 2, 3, 4]).mean()
    day = start_hour.isin([10, 11, 12, 13, 14, 15]).mean()
    assert night > day


def test_hour_weights_favour_night_over_midday():
    # index 0=00:00, 12=12:00 etc; night hours should carry more weight
    assert HOUR_WEIGHTS[0] > HOUR_WEIGHTS[12]
    assert HOUR_WEIGHTS[23] > HOUR_WEIGHTS[12]
    assert abs(HOUR_WEIGHTS.sum() - 1.0) < 1e-9
