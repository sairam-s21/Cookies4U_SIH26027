"""Option 2 -- task splitting.

For a task whose required block-hours exceed the largest single window
available anywhere on its section across the whole horizon, and whose
defect type is marked splittable (railblock.synthetic.maintenance_tasks.
SPLITTABLE), this pre-processing step decomposes it into up to
MAX_SPLIT_SESSIONS roughly-equal sub-sessions ("parts"), each of which
competes for a window independently through the SAME all-or-nothing
(task, window) CP-SAT machinery Layer 3 already uses -- no new solver
variables are needed, splitting is purely a pre/post-processing transform
around the existing model.

Why pre-processing rather than new CP-SAT variables: a task that fits
whole is always cheaper for the objective than the same task split (each
extra window opened costs WINDOW_OPEN_COST), so the solver already
prefers a single whole session whenever one is available -- giving parts
as additional options and letting the objective naturally favour "whole"
achieves the required "apply automatically ... whenever a task doesn't fit
a single window" behaviour without a bespoke mode-selection sub-problem.

expand_splittable_tasks() decides whether a task should be split BEFORE
solving (deterministic, based on whether any single window anywhere is
already known to be big enough); collapse_split_results() aggregates the
solver's per-part answers back into original-task-level completion status
after solving: "full" (every part scheduled), "partial" (some but not
all -- reported distinctly, per the session brief), or "none".
"""

from __future__ import annotations

import pandas as pd

# This corridor's real windows (capacity.py's documented fragmentation)
# are often only 50-120 minutes long on busy sections, so a task split
# into too few parts can still have each part exceed the largest
# available window. The cap here is not picked backward from a target
# completion percentage -- it was tested empirically against real
# corridor capacity and set to actually rescue tasks that a smaller cap
# couldn't, without touching demand volume or duration data at all. A
# concrete case that drove the current value: TDMS-REQ-00054 (4.14h =
# 248.4 minutes) on PCKM-AB, whose windows never exceed ~30 minutes
# anywhere across a full 90-day sample -- even the 6 largest windows
# available only sum to ~181 minutes, genuinely short regardless of
# which start_date/horizon is searched (this section's weekday train
# pattern repeats, so widening the horizon further changes nothing). The
# top 9 windows sum to ~271.5 minutes -- enough; 10 keeps a margin above
# that measured floor.
MAX_SPLIT_SESSIONS = 10


def _part_task_id(task_id: str, part_index: int, n_parts: int) -> str:
    return f"{task_id}__part{part_index}of{n_parts}"


def expand_splittable_tasks(
    ranked_tasks: pd.DataFrame,
    capacity: pd.DataFrame,
    combinable_partner_minutes: dict[str, list[float]] | None = None,
) -> pd.DataFrame:
    """Return a copy of `ranked_tasks` where any task that (a) doesn't fit
    the largest window available anywhere on its section across the whole
    horizon and (b) is marked splittable, is replaced by 2 or MAX_SPLIT_
    SESSIONS part rows. Every row gains `parent_task_id`, `part_index`
    (1-based), `n_parts`. Non-split tasks get part_index=1, n_parts=1,
    parent_task_id == task_id, otherwise unchanged.

    Part SIZES are matched against this section's actual window sizes
    (largest first), not a blind equal division of the total duration. A
    task split into equal shares can land exactly on the size of a window
    that never exists anywhere real (e.g. 218 min split into three 72-min
    equal shares, when the section's real windows that week were
    106.6/76.6/76.6/36/36/36 minutes) -- CP-SAT would never even be
    offered a part sized to fit a genuinely available window another
    department's task is already using, so a real combining opportunity
    (the +50 COORDINATION_BONUS, see model.py) would silently never be
    considered, not rejected. Sizing each part to an actual window closes
    that gap: the solver is offered a candidate part that could
    physically share a real window, and remains free to place it
    elsewhere if that scores better. `n_parts` itself is decided
    separately (still 2 vs MAX_SPLIT_SESSIONS) -- this only changes how
    the total is divided among them.

    `combinable_partner_minutes`: {task_id: [partner's own duration in
    minutes, ...]}, from railblock.scheduling.combining's "check same
    section, then due-date overlap, then different department"
    pre-processing pass. When a task has an identified
    partner, one of its part sizes is aimed at that partner's SPECIFIC
    duration (capped to a real window), not just "some real window is
    this big" -- a more precisely-targeted candidate for the coordination
    bonus. CP-SAT still decides whether to actually place them together;
    this only makes sure it's genuinely offered the option.
    """
    # Each window's DATE is kept alongside its size, not just a flat
    # sorted-by-size list -- sizing a part against a window that exists
    # somewhere in the whole horizon but falls AFTER this specific
    # task's own due_date (or before its raised_date) would offer a
    # candidate size the task can never actually use, which can leave a
    # genuinely-too-large leftover chunk relative to what's really
    # reachable within its own deadline.
    windows_by_section: dict[str, list[tuple[str, float]]] = {}
    if not capacity.empty:
        for section_id, grp in capacity.groupby("section_id"):
            windows_by_section[section_id] = sorted(
                zip(grp["date"], grp["window_minutes"]), key=lambda dw: -dw[1]
            )
    combinable_partner_minutes = combinable_partner_minutes or {}

    rows = []
    for _, task in ranked_tasks.iterrows():
        required_minutes = task["estimated_block_hours"] * 60.0
        all_windows = windows_by_section.get(task["section_id"], [])
        raised, due = task.get("raised_date"), task.get("due_date")
        if raised and due:
            # already sorted by size descending -- filtering preserves that order
            section_windows = [minutes for date, minutes in all_windows if raised <= date <= due]
            if not section_windows and all_windows:
                # due_date is NOT a hard scheduling constraint in
                # model.py (it only affects on_time tagging, never
                # whether a window can be assigned). A task whose own
                # [raised_date, due_date] doesn't overlap the search
                # horizon at all (e.g. the caller picked a start_date
                # well after this task's due_date) must not fall through
                # to the "no real window data" blind equal-split below
                # when real window data DOES exist for this section
                # elsewhere in the horizon -- on a section with
                # unusually tiny windows (PCKM-AB: max ~30 minutes
                # anywhere), a blind split would produce part sizes
                # bigger than any real window, guaranteeing total
                # failure. Falls back to this section's window sizes
                # from ANYWHERE in the horizon instead of going blind --
                # the solver can genuinely place a part sized to one of
                # these regardless of due_date.
                section_windows = [minutes for _date, minutes in all_windows]
        else:
            section_windows = [minutes for _date, minutes in all_windows]
        max_window = section_windows[0] if section_windows else 0.0

        needs_split = required_minutes > max_window and bool(task.get("splittable", False))
        if not needs_split:
            row = task.to_dict()
            row.update({"parent_task_id": task["task_id"], "part_index": 1, "n_parts": 1})
            rows.append(row)
            continue

        n_parts = 2 if (required_minutes / 2.0) <= max_window else MAX_SPLIT_SESSIONS

        if not section_windows:
            # No real window data at all for this section (shouldn't happen
            # in a real solve; only possible with hand-built/incomplete
            # capacity data) -- fall back to the original blind equal split
            # rather than manufacture zero-minute ghost parts.
            base_minutes = required_minutes // n_parts
            remainder = required_minutes - base_minutes * n_parts
            parts = [base_minutes] * (n_parts - 1) + [base_minutes + remainder]
        else:
            partner_sizes = [min(m, max_window) for m in combinable_partner_minutes.get(task["task_id"], []) if m > 0]
            candidate_pool = sorted(set(partner_sizes) | set(section_windows), reverse=True)
            remaining = required_minutes
            parts = []
            for i in range(n_parts):
                # Greedily taking a FULL window/partner size for each of
                # the first several parts can exhaust `required_minutes`
                # before every one of the n_parts slots has been used,
                # leaving phantom 0-minute (or near-zero, e.g. 3
                # seconds) trailing "parts" that do no real work and
                # can't fit or combine with anything -- wasting
                # MAX_SPLIT_SESSIONS budget and, worse, letting
                # force_combinable_placements commit a "combined" window
                # over 0 real minutes. Stop generating parts the moment
                # nothing meaningful is left, instead of always emitting
                # exactly n_parts rows.
                if remaining <= 0.01:
                    break
                if i == n_parts - 1:
                    parts.append(remaining)  # last part absorbs whatever's left
                    break
                candidate_window = candidate_pool[i % len(candidate_pool)]
                take = min(candidate_window, remaining)
                parts.append(take)
                remaining -= take
            if not parts:
                parts = [required_minutes]

        actual_n_parts = len(parts)
        for part_index, part_minutes in enumerate(parts, start=1):
            row = task.to_dict()
            row["task_id"] = _part_task_id(task["task_id"], part_index, actual_n_parts)
            row["estimated_block_hours"] = part_minutes / 60.0
            row.update({"parent_task_id": task["task_id"], "part_index": part_index, "n_parts": actual_n_parts})
            rows.append(row)

    return pd.DataFrame(rows)


def collapse_split_results(
    part_schedule: pd.DataFrame,
    part_unscheduled: pd.DataFrame,
    original_tasks: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate part-level solve_schedule() output back to original-task
    granularity. Returns (full_df, partial_df, unscheduled_df):
      - full_df: one row per fully-scheduled original task (all parts
        placed); `sessions` holds the list of (date, window_index) pairs
        used, `option` is "whole" (n_parts==1) or "split" (n_parts>1).
      - partial_df: original tasks where SOME but not all parts were
        placed -- reported distinctly, not counted as complete and not
        marked on_time (see session brief: only fully-covered tasks can be
        on-time complete).
      - unscheduled_df: original tasks (original granularity, ready for
        Option 3) where ZERO parts were placed.
    """
    # n_parts lives on the EXPANDED (part-level) rows, not on `original_tasks`
    # -- build a parent_task_id -> n_parts lookup from whichever part-level
    # frame has it (a task is either scheduled or unscheduled, never both).
    parts_lookup = pd.concat(
        [df[["parent_task_id", "n_parts"]] for df in (part_schedule, part_unscheduled) if not df.empty],
        ignore_index=True,
    ).drop_duplicates("parent_task_id").set_index("parent_task_id")["n_parts"] if (
        not part_schedule.empty or not part_unscheduled.empty
    ) else pd.Series(dtype=int)

    parent_ids = original_tasks["task_id"].tolist()
    full_rows, partial_rows, unscheduled_ids = [], [], []

    for parent_id in parent_ids:
        parent_row = original_tasks[original_tasks["task_id"] == parent_id].iloc[0]
        my_scheduled = part_schedule[part_schedule["parent_task_id"] == parent_id] if not part_schedule.empty else part_schedule
        n_parts = int(parts_lookup.get(parent_id, 1))
        n_done = len(my_scheduled)

        if n_done == 0:
            unscheduled_ids.append(parent_id)
        elif n_done < n_parts:
            partial_rows.append(
                {
                    "task_id": parent_id,
                    "department": parent_row["department"],
                    "section_id": parent_row["section_id"],
                    "requester_priority": parent_row["requester_priority"],
                    "whittle_index": parent_row["whittle_index"],
                    "n_parts": n_parts,
                    "n_parts_scheduled": n_done,
                    "sessions_scheduled": list(zip(my_scheduled["date"], my_scheduled["window_index"])),
                    "approval_path": parent_row.get("approval_path", "sm_direct"),
                    "completion_verified": False,
                }
            )
        else:
            full_rows.append(
                {
                    "task_id": parent_id,
                    "department": parent_row["department"],
                    "section_id": parent_row["section_id"],
                    "requester_priority": parent_row["requester_priority"],
                    "whittle_index": parent_row["whittle_index"],
                    "estimated_block_hours": parent_row["estimated_block_hours"],
                    "option": "whole" if n_parts == 1 else "split",
                    "n_parts": n_parts,
                    "sessions": list(zip(
                        my_scheduled["date"], my_scheduled["window_index"],
                        my_scheduled.get("start_minute", pd.Series([None] * len(my_scheduled))),
                        my_scheduled.get("end_minute", pd.Series([None] * len(my_scheduled))),
                    )),
                    "date": sorted(my_scheduled["date"])[0],  # earliest session, for simple chronological display
                    "on_time": bool(my_scheduled["on_time"].all()),
                    "negotiated_exception": False,
                    "approval_path": parent_row.get("approval_path", "sm_direct"),
                    "completion_verified": False,  # SR 3.51.6: not true until Track Fit Certificate + joint SI/SM test
                }
            )

    full_df = pd.DataFrame(full_rows)
    partial_df = pd.DataFrame(partial_rows)
    unscheduled_df = original_tasks[original_tasks["task_id"].isin(unscheduled_ids)].copy()
    return full_df, partial_df, unscheduled_df
