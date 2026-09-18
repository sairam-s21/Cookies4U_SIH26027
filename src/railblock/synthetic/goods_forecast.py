"""SYNTHETIC goods/freight train forecast standing in for the Control
Office's freight demand (FOIS has no public API -- see master prompt
Section 5). Produces freight occupancy windows per block section per day,
weighted towards night/off-peak hours the way real Indian Railways freight
paths typically are (passenger traffic gets daytime priority).
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

# Relative likelihood of a freight train starting in each hour of the day
# (index 0 = 00:00-01:00, ... index 23 = 23:00-24:00). Night/off-peak hours
# are weighted several times higher than the daytime passenger-heavy hours,
# not just a flat random draw.
HOUR_WEIGHTS = np.array(
    [
        6, 6, 6, 5, 4, 3, 2, 1,  # 00-07: late night high, tapering into morning peak
        1, 1, 2, 2, 3, 3, 2, 2,  # 08-15: daytime, passenger priority, low freight
        1, 1, 2, 3, 4, 5, 6, 6,  # 16-23: evening taper back up into night
    ],
    dtype=float,
)
HOUR_WEIGHTS = HOUR_WEIGHTS / HOUR_WEIGHTS.sum()

FREIGHT_SPEED_KMPH_RANGE = (25.0, 45.0)
TRAINS_PER_SECTION_PER_DAY_RANGE = (2, 7)


def generate_goods_forecast(
    sections: pd.DataFrame,
    start_date: date,
    n_days: int = 7,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate freight occupancy windows for every section over n_days.

    sections: block_sections DataFrame (needs section_id, length_km).
    Returns one row per freight movement: section_id, date, freight_id,
    start_minute, end_minute (minutes since local midnight of `date`; may
    exceed 1439 if the movement runs past midnight), duration_minutes.
    """
    rng = np.random.default_rng(seed)
    hours = np.arange(24)

    rows = []
    counter = 0
    for day_offset in range(n_days):
        the_date = start_date + timedelta(days=day_offset)
        for _, section in sections.iterrows():
            n_trains = int(
                rng.integers(
                    TRAINS_PER_SECTION_PER_DAY_RANGE[0],
                    TRAINS_PER_SECTION_PER_DAY_RANGE[1] + 1,
                )
            )
            for _ in range(n_trains):
                start_hour = int(rng.choice(hours, p=HOUR_WEIGHTS))
                start_minute_in_hour = int(rng.integers(0, 60))
                start_minute = start_hour * 60 + start_minute_in_hour

                speed = rng.uniform(*FREIGHT_SPEED_KMPH_RANGE)
                duration_minutes = max(10, round(section["length_km"] / speed * 60))
                # small dwell/pathing buffer, freight rarely runs exactly at line speed
                duration_minutes = int(duration_minutes * rng.uniform(1.05, 1.35))

                counter += 1
                rows.append(
                    {
                        "freight_id": f"GDS-{counter:06d}",
                        "section_id": section["section_id"],
                        "date": the_date.isoformat(),
                        "start_minute": start_minute,
                        "end_minute": start_minute + duration_minutes,
                        "duration_minutes": duration_minutes,
                        "data_source": "SYNTHETIC",
                    }
                )

    return pd.DataFrame(rows)
