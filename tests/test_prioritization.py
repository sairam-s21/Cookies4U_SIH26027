from datetime import date

import pandas as pd
import pytest

from railblock.prioritization.urgency import PRIORITY_WEIGHT, score_urgency
from railblock.prioritization.impact import IMPACT_WEIGHT, score_impact
from railblock.prioritization.whittle import rank_tasks
from railblock.synthetic.maintenance_tasks import DEFECT_TYPES


def _pax_df(rows):
    cols = ["section_id", "train_no", "start_minute", "end_minute", "source"]
    return pd.DataFrame(rows, columns=cols)


def _goods_df(rows=()):
    cols = ["section_id", "date", "start_minute", "end_minute"]
    return pd.DataFrame(list(rows), columns=cols)


def _toy_sections():
    return pd.DataFrame(
        [
            {"section_id": "A-B", "length_km": 20.0},
            {"section_id": "C-D", "length_km": 20.0},
        ]
    )


def _toy_task(task_id, section_id, priority, days_overdue, hours, defect_type="Rail fracture", department="Engineering"):
    return {
        "task_id": task_id,
        "department": department,
        "source_system": "TMS",
        "section_id": section_id,
        "defect_type": defect_type,
        "requester_priority": priority,
        "raised_date": "2026-01-01",
        "due_date": "2026-12-31",
        "days_overdue": days_overdue,
        "estimated_block_hours": hours,
        "data_source": "SYNTHETIC",
    }


# ---------------------------------------------------------------- urgency

def test_score_urgency_is_exact_product():
    tasks = pd.DataFrame(
        [
            _toy_task("T1", "A-B", "Critical", 10, 1.0),
            _toy_task("T2", "A-B", "Moderate", 4, 1.0),
            _toy_task("T3", "A-B", "Routine", 0, 1.0),
        ]
    )
    scores = score_urgency(tasks)
    assert scores.tolist() == [10 * PRIORITY_WEIGHT["Critical"], 4 * PRIORITY_WEIGHT["Moderate"], 0]


def test_score_urgency_rejects_unknown_priority():
    tasks = pd.DataFrame([_toy_task("T1", "A-B", "Urgent!!", 10, 1.0)])
    with pytest.raises(ValueError):
        score_urgency(tasks)


# ----------------------------------------------------------------- impact

def test_impact_weight_covers_every_generator_defect_type():
    all_types = {dt for types in DEFECT_TYPES.values() for dt in types}
    assert all_types <= set(IMPACT_WEIGHT)


def test_score_impact_uses_exact_lookup():
    tasks = pd.DataFrame([_toy_task("T1", "A-B", "Critical", 1, 1.0, defect_type="Rail fracture")])
    assert score_impact(tasks).iloc[0] == IMPACT_WEIGHT["Rail fracture"]


# --------------------------------------------------------- whittle ranking

def test_more_overdue_critical_outranks_less_overdue_moderate_same_section():
    # holding section (and therefore congestion) constant isolates the
    # urgency/priority effect on the index.
    tasks = pd.DataFrame(
        [
            _toy_task("CRIT", "A-B", "Critical", 40, 2.0),
            _toy_task("MOD", "A-B", "Moderate", 5, 2.0),
        ]
    )
    sections = _toy_sections()
    ranked = rank_tasks(tasks, sections, _pax_df([]), _goods_df(), date(2026, 9, 7), n_days=1)
    assert ranked.iloc[0]["task_id"] == "CRIT"
    assert ranked.iloc[0]["whittle_index"] > ranked.iloc[1]["whittle_index"]


def test_congestion_changes_ranking_order_within_the_same_tier():
    """The whole point of Whittle-index ranking over a plain urgency sort:
    a lower-urgency task competing for an almost-saturated section should
    be able to outrank a higher-urgency task sitting in a wide-open
    section -- WITHIN the same priority tier (Session 30: TIER_OFFSET
    keeps this real and working within a tier; see the test right below
    for why it must never let this cross a tier boundary instead)."""
    sections = _toy_sections()

    # A-B: occupied nearly the entire day (recurring), leaving ~1 minute free.
    pax = _pax_df([{"section_id": "A-B", "train_no": "1", "start_minute": 0, "end_minute": 1439, "source": "real_timetable"}])
    goods = _goods_df()

    low_urgency_congested = _toy_task("LOW_CONGESTED", "A-B", "Moderate", 5, 5.0)
    high_urgency_open = _toy_task("HIGH_OPEN", "C-D", "Moderate", 50, 0.01)
    tasks = pd.DataFrame([low_urgency_congested, high_urgency_open])

    naive_order = score_urgency(tasks).sort_values(ascending=False).index.map(tasks["task_id"]).tolist()
    assert naive_order == ["HIGH_OPEN", "LOW_CONGESTED"]  # 50*2=100 beats 5*2=10

    ranked = rank_tasks(tasks, sections, pax, goods, date(2026, 9, 7), n_days=1)
    whittle_order = ranked["task_id"].tolist()

    assert whittle_order == ["LOW_CONGESTED", "HIGH_OPEN"]  # order actually flips
    assert whittle_order != naive_order


def test_congestion_reorders_within_a_tier_but_never_crosses_one():
    """Session 30, at explicit user request, after a real reported gap:
    "I want all critical to rank above moderate tasks and all moderate
    tasks should rank above routine tasks ... whittle index should also
    be considered for ranking, but the order I said should remain." A
    Moderate task sitting in an almost-saturated section (near-maximum
    real congestion) must never outrank a Critical task in a wide-open
    one, however extreme the congestion gets -- TIER_OFFSET guarantees
    this by construction, not by hoping the continuous factors stay
    small."""
    sections = _toy_sections()
    pax = _pax_df([{"section_id": "A-B", "train_no": "1", "start_minute": 0, "end_minute": 1439, "source": "real_timetable"}])
    goods = _goods_df()

    extremely_congested_moderate = _toy_task("MOD_CONGESTED", "A-B", "Moderate", 1000, 5.0)
    barely_urgent_critical = _toy_task("CRIT_OPEN", "C-D", "Critical", 0, 0.01)
    tasks = pd.DataFrame([extremely_congested_moderate, barely_urgent_critical])

    ranked = rank_tasks(tasks, sections, pax, goods, date(2026, 9, 7), n_days=1)
    assert ranked["task_id"].tolist() == ["CRIT_OPEN", "MOD_CONGESTED"]


def test_rank_tasks_output_has_expected_columns_and_is_sorted():
    tasks = pd.DataFrame(
        [
            _toy_task("T1", "A-B", "Critical", 20, 2.0),
            _toy_task("T2", "C-D", "Routine", 1, 1.0),
        ]
    )
    sections = _toy_sections()
    ranked = rank_tasks(tasks, sections, _pax_df([]), _goods_df(), date(2026, 9, 7), n_days=1)

    expected_cols = {
        "urgency_score", "impact_weight", "avg_free_hours_per_day",
        "demand_hours", "congestion", "whittle_index", "priority_rank", "risk_percentage",
    }
    assert expected_cols <= set(ranked.columns)
    assert ranked["priority_rank"].tolist() == [1, 2]
    assert (ranked["whittle_index"].diff().dropna() <= 0).all()  # non-increasing


def test_risk_percentage_is_bounded_percentile_rank_of_whittle_index():
    # Session 24, at explicit user request: whittle_index itself has no
    # natural upper bound, so risk_percentage is a real [0, 100]
    # percentile rank against every other task in the SAME batch --
    # highest whittle_index -> 100%, lowest -> 0%.
    tasks = pd.DataFrame(
        [
            _toy_task("T1", "A-B", "Critical", 20, 2.0),
            _toy_task("T2", "A-B", "Moderate", 5, 1.0),
            _toy_task("T3", "C-D", "Routine", 0, 0.5),
        ]
    )
    sections = _toy_sections()
    ranked = rank_tasks(tasks, sections, _pax_df([]), _goods_df(), date(2026, 9, 7), n_days=1)
    assert ((ranked["risk_percentage"] >= 0) & (ranked["risk_percentage"] <= 100)).all()
    most_urgent = ranked.loc[ranked["whittle_index"].idxmax()]
    least_urgent = ranked.loc[ranked["whittle_index"].idxmin()]
    assert most_urgent["risk_percentage"] == 100.0
    assert least_urgent["risk_percentage"] == 0.0


def test_risk_percentage_is_100_for_a_single_task_batch():
    tasks = pd.DataFrame([_toy_task("T1", "A-B", "Critical", 5, 2.0)])
    sections = _toy_sections()
    ranked = rank_tasks(tasks, sections, _pax_df([]), _goods_df(), date(2026, 9, 7), n_days=1)
    assert ranked.iloc[0]["risk_percentage"] == 100.0


def test_rank_tasks_deterministic_given_same_inputs():
    tasks = pd.DataFrame(
        [
            _toy_task("T1", "A-B", "Critical", 20, 2.0),
            _toy_task("T2", "C-D", "Routine", 1, 1.0),
        ]
    )
    sections = _toy_sections()
    a = rank_tasks(tasks, sections, _pax_df([]), _goods_df(), date(2026, 9, 7), n_days=1)
    b = rank_tasks(tasks, sections, _pax_df([]), _goods_df(), date(2026, 9, 7), n_days=1)
    pd.testing.assert_frame_equal(a, b)
