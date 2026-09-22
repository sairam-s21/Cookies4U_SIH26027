"""Layer 3 input prep: per (section, day) candidate maintenance-window
capacity, derived directly from Layer 1's compute_availability.

Each (section, day) can offer several disjoint free intervals (busy trunk
sections in particular have their free time badly fragmented by frequent
train traffic -- see PROGRESS.md's "window fragmentation" finding). Rather
than collapsing a day down to a single largest gap, this exposes up to
MAX_WINDOWS_PER_DAY separate candidate windows (the largest ones, above
MIN_WINDOW_MINUTES) per (section, day), each independently choosable by
Layer 3's CP-SAT model -- a closer, if still simplified, match to Lidén's
general window-CHOICE sub-problem (choosing among multiple available
windows, not just whether to use "the" window).
"""

from __future__ import annotations

from datetime import date as Date, timedelta

import pandas as pd

from railblock.availability.corridor_availability import compute_availability

MAX_WINDOWS_PER_DAY = 8
MIN_WINDOW_MINUTES = 15.0  # gaps shorter than this aren't practically usable for a block


def compute_daily_window_capacity(
    sections: pd.DataFrame,
    passenger_occupancy: pd.DataFrame,
    goods_occupancy: pd.DataFrame,
    start_date: Date,
    n_days: int = 7,
    availability_cache: dict | None = None,
) -> pd.DataFrame:
    """One row per (section_id, date, window_index): window_minutes = the
    length of that candidate free interval. window_index 0 is always the
    largest gap that day, 1 the next largest, etc. -- at most
    MAX_WINDOWS_PER_DAY rows per (section, date), and only for gaps
    >= MIN_WINDOW_MINUTES. A (section, date) with no usable gap at all
    contributes zero rows (not a zero-minute row).

    `availability_cache` (optional): pass the SAME dict given to
    railblock.prioritization.whittle.rank_tasks for this request, so
    this doesn't recompute compute_availability for every (section, date)
    pair a SECOND time when ranking already computed it moments earlier
    for the exact same occupancy data -- see compute_availability's own
    docstring for the full reasoning. Omit to always recompute.
    """
    rows = []
    for section_id in sections["section_id"]:
        for d in range(n_days):
            the_date = start_date + timedelta(days=d)
            avail = compute_availability(section_id, the_date, passenger_occupancy, goods_occupancy, cache=availability_cache)
            intervals = sorted(avail["free_intervals"], key=lambda iv: iv[1] - iv[0], reverse=True)
            intervals = [iv for iv in intervals if iv[1] - iv[0] >= MIN_WINDOW_MINUTES][:MAX_WINDOWS_PER_DAY]
            for k, (start_minute, end_minute) in enumerate(intervals):
                rows.append(
                    {
                        "section_id": section_id,
                        "date": the_date.isoformat(),
                        "window_index": k,
                        "window_minutes": end_minute - start_minute,
                        "start_minute": start_minute,
                        "end_minute": end_minute,
                    }
                )
    return pd.DataFrame(
        rows, columns=["section_id", "date", "window_index", "window_minutes", "start_minute", "end_minute"]
    )
