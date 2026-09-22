"""End-to-end wiring: synthetic data -> Layer 1 availability (with Option
1's realistic day-of-week occupancy) -> Layer 2 urgency/Whittle-index
ranking -> Layer 3 CP-SAT scheduling with Options 1-2 (day-of-week
occupancy, automatic task splitting). Option 3 (bounded negotiated
exceptions) is out of scope: train cancellation/negotiation is not
handled by this project.

Not a new layer of logic itself -- every function called here already
lives in its own layer's module; this just sequences them the way a real
caller (an API endpoint, or a demo script) would.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date

import pandas as pd

from railblock.availability.corridor_availability import build_passenger_occupancy
from railblock.corridor.derive_stations import load_timetable
from railblock.corridor.fine_stations import load_fine_corridor_stations
from railblock.corridor.sections import build_block_sections
from railblock.prioritization.whittle import rank_tasks
from railblock.scheduling.orchestrator import FullScheduleResult, solve_schedule_with_options
from railblock.synthetic.goods_forecast import generate_goods_forecast
from railblock.synthetic.maintenance_tasks import generate_maintenance_tasks


@dataclass
class PipelineResult:
    stations: pd.DataFrame
    sections: pd.DataFrame
    tasks: pd.DataFrame
    ranked_tasks: pd.DataFrame
    schedule_result: FullScheduleResult


def load_corridor(corridor_stations_path=None, granularity: str = "fine") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the corridor's station/section model.

    granularity="fine" (default): the 102-point/101-section model derived
    from train 12243's real timetable (railblock.corridor.fine_stations)
    -- materially more schedulable capacity than the coarse model, since
    finer section boundaries fragment less. This is what the scheduler
    uses going forward.
    granularity="coarse": the original 19-station/18-section model
    (train-union derivation over Train_details_22122017.csv), kept
    available for comparison/regression testing, not for new demos.
    `corridor_stations_path` only applies to the coarse model.
    """
    if granularity == "fine":
        stations = load_fine_corridor_stations()
    elif granularity == "coarse":
        from railblock.paths import CORRIDOR_STATIONS_CSV

        stations = pd.read_csv(corridor_stations_path or CORRIDOR_STATIONS_CSV)
    else:
        raise ValueError(f"granularity must be 'fine' or 'coarse', got {granularity!r}")

    sections = build_block_sections(stations)
    return stations, sections


def run_end_to_end(
    start_date: Date,
    n_tasks: int = 150,
    n_days: int = 7,
    seed: int | None = None,
    solver_time_limit_s: float = 30.0,
    granularity: str = "fine",
    demand_scenario: str | None = None,
) -> PipelineResult:
    """Run the full data -> Layer 2 -> Layer 3 (Options 1-2)
    pipeline on a fresh (or seeded) synthetic scenario for the real MAS-JTJ
    corridor. `passenger_occupancy` already reflects Option 1's realistic
    day-of-week running patterns (railblock.availability.service_frequency);
    `solve_schedule_with_options` applies Option 2 (splitting) automatically.

    `granularity`: see load_corridor -- "fine" (default) is the 102-point
    model now used going forward; "coarse" reproduces the original
    19-station model for comparison.

    `demand_scenario`: if given ("stress_test" or
    "double_track_adjusted", see railblock.synthetic.maintenance_tasks.
    DEMAND_SCENARIOS), overrides n_tasks with that scenario's citable
    weekly figure instead of the arbitrary default.
    """
    stations, sections = load_corridor(granularity=granularity)

    timetable, _ = load_timetable()
    passenger_occupancy = build_passenger_occupancy(timetable, stations)
    goods_occupancy = generate_goods_forecast(sections, start_date, n_days=n_days, seed=seed)
    tasks = generate_maintenance_tasks(
        sections, n_tasks=n_tasks, as_of_date=start_date, seed=seed, demand_scenario=demand_scenario
    )

    ranked_tasks = rank_tasks(tasks, sections, passenger_occupancy, goods_occupancy, start_date, n_days=n_days)

    schedule_result = solve_schedule_with_options(
        ranked_tasks, sections, passenger_occupancy, goods_occupancy, start_date, n_days=n_days,
        time_limit_s=solver_time_limit_s,
    )

    return PipelineResult(
        stations=stations,
        sections=sections,
        tasks=tasks,
        ranked_tasks=ranked_tasks,
        schedule_result=schedule_result,
    )


if __name__ == "__main__":
    import time
    from datetime import date

    t0 = time.time()
    result = run_end_to_end(start_date=date(2026, 9, 7), n_tasks=150, n_days=7, seed=42)
    elapsed = time.time() - t0

    sr = result.schedule_result
    print(f"End-to-end run in {elapsed:.2f}s (solver: {sr.solve_time_s:.2f}s, status={sr.status})")
    print(f"Tasks generated: {len(result.tasks)}")
    print(f"Option counts: {sr.option_counts}")
    print(f"Partial: {len(sr.partial)}  Unscheduled: {len(sr.unscheduled)}")
    print(f"Windows opened: {len(sr.windows)}  Combined (multi-dept): {sr.windows['combined'].sum() if not sr.windows.empty else 0}")
    print(f"Objective value: {sr.objective_value}")
    print("\nSample schedule (first 20 rows):")
    print(sr.schedule.head(20).to_string())
    if not sr.partial.empty:
        print("\nPartial completions:")
        print(sr.partial.to_string())
