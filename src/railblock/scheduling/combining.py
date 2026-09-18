"""Option 2 extension, at explicit user request: identify which pending
tasks are genuine department-combining candidates BEFORE the CP-SAT solve
ever runs, using exactly the three checks asked for -- same section, a
real overlap in the days each could plausibly happen ([raised_date,
due_date] windows intersect), and different departments (SR 3.51.6's
joint disconnection only lets *different* crews work the same window
concurrently -- two tasks from the SAME department still queue
sequentially, one crew can't be in two places).

`find_combinable_pairs`/`partner_minutes_by_task` only ever identify
candidates, used to bias split-sizing toward a specific partner's real
duration. `force_combinable_placements` goes further, at explicit user
request after CP-SAT's own combined_window/COORDINATION_BONUS incentive
(model.py) repeatedly failed to actually realize a real, verified
combining opportunity in production, even after sizing was fixed --
COORDINATION_BONUS is real money on the table in the objective, but
CP-SAT is solving the WHOLE horizon under a time limit with 8 parallel
search workers, and a locally obvious combine can simply lose out to
whatever the search happened to explore first elsewhere. This function
deterministically PRE-ASSIGNS a verified-feasible combination the moment
one exists, instead of leaving it to chance: real section, real
overlapping window, each task's own hours independently fitting the
window (the same "max not sum" per-department rule model.py enforces),
within both tasks' own [raised_date, due_date]. What it finds is removed
from the pool CP-SAT ever sees for those specific parts (and the
consumed capacity subtracted), so CP-SAT can never re-decide it away --
but it never forces a pairing where a real window genuinely doesn't
exist for both, and it makes no claim about being the GLOBALLY best
possible pairing across the whole horizon, only a guaranteed-valid one.
"""

from __future__ import annotations

import pandas as pd

from railblock.scheduling.disconnection_procedure import REQUIRED_SIGNOFF_ROLES_JOINT_SCHEDULE
from railblock.scheduling.splitting import expand_splittable_tasks
from railblock.synthetic.maintenance_tasks import SM_DIRECT_APPROVAL_MAX_HOURS


def find_combinable_pairs(ranked_tasks: pd.DataFrame) -> dict[str, list[str]]:
    """{task_id: [combinable partner task_id, ...]} for every task that
    has at least one real combining candidate: same section_id, a
    different department, and an overlapping [raised_date, due_date]
    window (max(raised_dates) <= min(due_dates) -- there's at least one
    real day both tasks could genuinely be done on)."""
    partners: dict[str, list[str]] = {}
    if ranked_tasks.empty:
        return partners

    for _, group in ranked_tasks.groupby("section_id"):
        rows = group.to_dict("records")
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i], rows[j]
                if a["department"] == b["department"]:
                    continue
                lo = max(a["raised_date"], b["raised_date"])
                hi = min(a["due_date"], b["due_date"])
                if lo > hi:
                    continue
                partners.setdefault(a["task_id"], []).append(b["task_id"])
                partners.setdefault(b["task_id"], []).append(a["task_id"])
    return partners


def partner_minutes_by_task(ranked_tasks: pd.DataFrame, pairs: dict[str, list[str]]) -> dict[str, list[float]]:
    """{task_id: [partner's estimated_block_hours in minutes, ...]} for
    every task with at least one combinable partner -- used to size a
    split part against a SPECIFIC real partner's own duration, a more
    precisely-aimed candidate than just "some real window happens to be
    this big", which is all sizing against section windows alone can
    offer."""
    if not pairs:
        return {}
    hours_by_id = ranked_tasks.set_index("task_id")["estimated_block_hours"]
    result: dict[str, list[float]] = {}
    for task_id, partner_ids in pairs.items():
        result[task_id] = [float(hours_by_id[pid]) * 60.0 for pid in partner_ids if pid in hours_by_id.index]
    return result


_SCHEDULE_COLUMNS = [
    "task_id", "parent_task_id", "n_parts", "part_index", "department", "section_id",
    "date", "window_index", "start_minute", "end_minute", "estimated_block_hours",
    "requester_priority", "whittle_index", "on_time", "approval_path", "completion_verified",
]
_WINDOW_COLUMNS = [
    "section_id", "date", "window_index", "start_minute", "end_minute", "minutes_used",
    "minutes_capacity", "departments", "task_ids", "combined", "required_signoffs",
]


def force_combinable_placements(
    expanded_tasks: pd.DataFrame,
    capacity: pd.DataFrame,
    combinable_pairs: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (combined_schedule, remaining_tasks, capacity_after_combine).

    Session 26, rewritten at explicit user request after two real
    problems with the original pairwise version: (1) it let multiple
    tasks from the SAME department stack into one "combined" window
    with no cap at all (a real window showed 6 tasks combined, when a
    joint disconnection genuinely only makes sense as one crew per
    department working concurrently); (2) it processed arbitrary pairs
    in due-date order rather than reasoning about a section's real
    department mix, so a case like MAF-TO (one Engineering task, two
    Traction tasks, zero Signalling) never got tried in the way that
    actually reflects what's really being competed for there.

    Rewritten a SECOND time, again at explicit user request, from
    task-COUNT-based scarcity to TIME-based (sum of hours) scarcity, per
    this exact user-given example: "if total time of engineering tasks
    is 5 hrs, total time of signaling tasks is 3 hrs, and total time of
    traction tasks is 2 hrs ... combine all 3 departments for 2 hrs,
    engineering and signaling departments for remaining 1 hr and
    remaining 2 hrs of engineering department should take place
    separately." Counting TASKS (as the first rewrite did) is the wrong
    scarce resource: a department with one 5-hour task and a department
    with five 1-hour tasks have the same real combining budget, but the
    old count-based rule would have called the first "scarcer". The real
    bound is each department's own TOTAL remaining minutes in that
    section, and it is a POOL, not a fixed parent-to-parent pairing --
    ANY still-unplaced part from ANY task of that department can fill
    the next shared window, so a department's second (third, ...) task
    gets its fair chance too, not just its first.

    The algorithm: per section, repeat --
      1. Among departments with real remaining minutes left (and not
         already found infeasible to combine further this pass), rank
         by remaining minutes ascending -- the scarcest by TIME.
      2. Take the 3 scarcest (or both, if only 2 remain) as this round's
         group.
      3. Pool-match: for each real shared window (earliest date, then
         largest, never reused), if every department in the group has
         at least one still-unplaced part (from any of its tasks) that
         independently fits that window and lies within its own
         [raised_date, due_date], place the most urgent (earliest due,
         then largest) such part from each -- exactly like before, "max
         not sum" per department, never more than one task per
         department per window. Keep doing this for the SAME group
         until nothing more matches.
      4. If that group made zero placements, its scarcest member has no
         real window left to combine on right now -- it can't be forced,
         so drop it from consideration and retry with what's left.
         Otherwise, loop back to 1: minutes just moved, so the ranking
         (and which departments are even still active) can change.
      5. Stop once fewer than 2 departments have any real remaining
         minutes left to offer. Whatever never got pulled into a
         combined window (always ending with the single most time-rich
         department, since nothing else is left to pair it with) is
         never forced -- it goes to the normal CP-SAT solve alone.

    This reproduces the user's 5h/3h/2h example exactly: triple-combine
    Engineering+Signalling+Traction until Traction (scarcest, 2h) is
    exhausted; then Engineering+Signalling pair-combine until Signalling
    (now scarcest, 1h left) is exhausted; Engineering's remaining 2h is
    left for the normal solve.

    `expanded_tasks`: post expand_splittable_tasks() -- each row is a
    concretely-sized whole task or split part, still carrying its
    parent's `raised_date`/`due_date`. `combinable_pairs` is accepted
    for backward compatibility but no longer drives this function's own
    logic (department scarcity does) -- it's still what biases split
    part sizing upstream, in expand_splittable_tasks.
    """
    empty = pd.DataFrame(columns=_SCHEDULE_COLUMNS)
    if expanded_tasks.empty or capacity.empty:
        return empty, expanded_tasks, capacity

    windows_by_section: dict[str, list[tuple[str, int, float, float | None, float | None]]] = {}
    for row in capacity.to_dict("records"):
        windows_by_section.setdefault(row["section_id"], []).append(
            (row["date"], int(row["window_index"]), row["window_minutes"], row.get("start_minute"), row.get("end_minute"))
        )
    for section_id in windows_by_section:
        windows_by_section[section_id].sort(key=lambda w: (w[0], -w[2]))  # earliest date, then largest window

    placed_ids: set[str] = set()
    placed_rows: list[dict] = []
    consumed_windows: set[tuple[str, str, int]] = set()  # a window used for ONE combined block is never reused for another

    def _unplaced_parts_for_dept(section_id: str, dept: str) -> list[dict]:
        """Pooled across ALL of this department's tasks in this section
        -- not one pre-chosen parent -- so a second/third task from the
        same department gets its own real chance at combining, not just
        whichever task happened to be picked first."""
        mask = (expanded_tasks["section_id"] == section_id) & (expanded_tasks["department"] == dept)
        rows = expanded_tasks[mask]
        return [r for r in rows.to_dict("records") if r["task_id"] not in placed_ids]

    def _dept_remaining_minutes(section_id: str, dept: str) -> float:
        return sum(r["estimated_block_hours"] * 60.0 for r in _unplaced_parts_for_dept(section_id, dept))

    def _try_combine_group_once(depts: tuple[str, ...], section_id: str) -> bool:
        """One real window, earliest-then-largest, where EVERY department
        in `depts` has at least one still-unplaced part -- pooled across
        all of that department's own tasks, not a single fixed parent --
        independently fitting (each department's own budget is that
        part's minutes alone, never summed with anyone else's, since at
        most one task per department ever shares a window). On success,
        the most urgent (earliest due_date, then largest) still-fitting
        part of each department is committed there."""
        parts_by_dept = {d: _unplaced_parts_for_dept(section_id, d) for d in depts}
        if any(not parts for parts in parts_by_dept.values()):
            return False

        for (date, idx, window_minutes, w_start, w_end) in windows_by_section.get(section_id, []):
            key = (section_id, date, idx)
            if key in consumed_windows:
                continue
            choice: dict[str, dict] = {}
            for d, parts in parts_by_dept.items():
                # Session 26 fix, after a real reported gap: a part's
                # size and a window's size are both derived through
                # independent chains of floating-point arithmetic
                # (proportional-distance splitting, delay-margin
                # padding, etc.) that can legitimately land a part
                # that's SUPPOSED to exactly fill a window a fraction of
                # a second over it (e.g. 55.644444444... vs
                # 55.644444) -- a real precision artifact, not a real
                # capacity shortfall. 0.01 real minutes (well under a
                # second) of tolerance absorbs that noise without
                # meaningfully loosening the actual constraint.
                fitting = [
                    p for p in parts
                    if p["raised_date"] <= date <= p["due_date"] and p["estimated_block_hours"] * 60.0 <= window_minutes + 0.01
                ]
                if not fitting:
                    break
                fitting.sort(key=lambda p: (p["due_date"], -p["estimated_block_hours"]))
                choice[d] = fitting[0]
            if len(choice) < len(depts):
                continue

            consumed_windows.add(key)
            for row in choice.values():
                placed_rows.append({
                    "task_id": row["task_id"],
                    "parent_task_id": row["parent_task_id"],
                    "n_parts": row.get("n_parts", 1),
                    "part_index": row.get("part_index", 1),
                    "department": row["department"],
                    "section_id": section_id,
                    "date": date,
                    "window_index": idx,
                    "start_minute": w_start,
                    "end_minute": w_end,
                    "estimated_block_hours": row["estimated_block_hours"],
                    "requester_priority": row["requester_priority"],
                    "whittle_index": row["whittle_index"],
                    "on_time": bool(date <= row["due_date"]),
                    "approval_path": row.get("approval_path", "sm_direct"),
                    "completion_verified": False,
                })
                placed_ids.add(row["task_id"])
            return True
        return False

    def _try_combine_group(depts: tuple[str, ...], section_id: str) -> bool:
        """Keeps calling _try_combine_group_once (each call already
        enforces "one task per department per window" via
        consumed_windows) until this EXACT department combination
        genuinely has nothing left to place -- either a real window/date
        fit no longer exists for it, or one of its departments' pooled
        parts are exhausted. Returns True if at least one real
        combination was made."""
        made_any = False
        while _try_combine_group_once(depts, section_id):
            made_any = True
        return made_any

    for section_id in expanded_tasks["section_id"].unique():
        depts_present = expanded_tasks.loc[expanded_tasks["section_id"] == section_id, "department"].unique().tolist()
        if len(depts_present) < 2:
            continue  # nothing else in this section to combine with

        blocked: set[str] = set()  # departments found to have no real window/date fit left for their current group
        while True:
            active = [d for d in depts_present if d not in blocked and _dept_remaining_minutes(section_id, d) > 1e-6]
            if len(active) < 2:
                break
            remaining_by_dept = {d: _dept_remaining_minutes(section_id, d) for d in active}
            depts_sorted = sorted(active, key=lambda d: remaining_by_dept[d])  # scarcest by remaining TIME first
            group = tuple(depts_sorted[:3]) if len(depts_sorted) >= 3 else tuple(depts_sorted[:2])
            if not _try_combine_group(group, section_id):
                # This exact group's scarcest member has no real window
                # left right now -- never force it; retry without it.
                blocked.add(group[0])

    combined_schedule = pd.DataFrame(placed_rows, columns=_SCHEDULE_COLUMNS) if placed_rows else empty
    remaining = expanded_tasks[~expanded_tasks["task_id"].isin(placed_ids)].copy()

    capacity_after = capacity.copy()
    if consumed_windows:
        # window_minutes is a real duration and must stay float -- a
        # caller (or a test fixture) that happened to pass whole-number
        # minutes as int64 would otherwise make this assignment raise.
        capacity_after["window_minutes"] = capacity_after["window_minutes"].astype(float)
        used_minutes_by_window: dict[tuple[str, str, int], float] = {}
        for row in placed_rows:
            key = (row["section_id"], row["date"], row["window_index"])
            used_minutes_by_window[key] = max(used_minutes_by_window.get(key, 0.0), row["estimated_block_hours"] * 60.0)
        for (section_id, date, idx), max_used in used_minutes_by_window.items():
            mask = (
                (capacity_after["section_id"] == section_id)
                & (capacity_after["date"] == date)
                & (capacity_after["window_index"] == idx)
            )
            capacity_after.loc[mask, "window_minutes"] = (capacity_after.loc[mask, "window_minutes"] - max_used).clip(lower=0)

    return combined_schedule, remaining, capacity_after


def combined_window_rows(combined_schedule: pd.DataFrame, capacity: pd.DataFrame) -> pd.DataFrame:
    """Reshapes force_combinable_placements()'s per-task rows into the
    same `windows` schema model.py's own solve produces, so the frontend
    sees these as real "Combined block (shared window)" entries -- not a
    separate, invisible code path.

    `capacity`: the ORIGINAL table (before force_combinable_placements'
    own subtraction) -- minutes_capacity must be the window's real total
    size, not what's left after this combination consumed some of it.

    Session 26 fix, after a real reported "No details found" bug: `task_
    ids` is built from `parent_task_id` (deduped), not the raw `task_id`
    column -- `combined_schedule` rows are at PART granularity (e.g.
    "TDMS-REQ-00202__part1of6" for a split task), which never exists as
    a real request task_id, so the frontend's per-id GET /requests
    lookup 404s and shows blank details for the whole block. This is the
    single place every within-tier combining path builds its combined
    windows from, so fixing it here covers all of them instead of
    collapsing separately (and inconsistently) at each call site."""
    if combined_schedule.empty:
        return pd.DataFrame(columns=_WINDOW_COLUMNS)

    capacity_by_key = capacity.set_index(["section_id", "date", "window_index"])["window_minutes"]

    rows = []
    for (section_id, date, idx), group in combined_schedule.groupby(["section_id", "date", "window_index"]):
        minutes_by_dept: dict[str, float] = {}
        for _, r in group.iterrows():
            minutes_by_dept[r["department"]] = minutes_by_dept.get(r["department"], 0.0) + r["estimated_block_hours"] * 60.0
        needs_joint_signoff = any(group["estimated_block_hours"] > SM_DIRECT_APPROVAL_MAX_HOURS)
        capacity_key = (section_id, date, idx)
        rows.append({
            "section_id": section_id,
            "date": date,
            "window_index": idx,
            "start_minute": group.iloc[0]["start_minute"],
            "end_minute": group.iloc[0]["end_minute"],
            "minutes_used": int(round(max(minutes_by_dept.values()))),
            "minutes_capacity": int(round(capacity_by_key.get(capacity_key, 0.0))),
            "departments": ",".join(sorted(set(group["department"]))),
            "task_ids": list(dict.fromkeys(group["parent_task_id"].tolist())),
            "combined": True,
            "required_signoffs": REQUIRED_SIGNOFF_ROLES_JOINT_SCHEDULE if needs_joint_signoff else [],
        })
    return pd.DataFrame(rows, columns=_WINDOW_COLUMNS)


def subtract_consumed_capacity(capacity: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """Session 15, at explicit user request: reduce each window's real
    available minutes by however much a PRIOR solve/forced-placement
    already used there (from that pass's own real windows), so a later
    solve on a different task set can never double-book capacity the
    first pass already committed to.

    Deliberately conservative: subtracts the whole amount from the
    window's total budget, not just from the consuming department's own
    share -- a real joint disconnection (SR 3.51.6) could technically
    still fit a genuinely different department concurrently in the same
    window, so this can under-use a little real capacity in that
    specific case. Chosen over a full per-department reservation model
    because it reuses the EXACT SAME capacity/window_minutes mechanism
    model.py already enforces (no new constraint code, no new way to get
    the safety-critical no-double-booking guarantee wrong) -- it can only
    ever make the second pass MORE conservative than reality, never less,
    which is the direction that's safe to be wrong in.

    Session 30: moved here from api/app.py (was `_subtract_consumed_
    capacity`) -- needed by both the normal /schedule/options flow and
    the Emergency Handling reschedule flow (railblock.scheduling.
    emergency), so it belongs alongside this module's other shared
    capacity/window helpers rather than as private API-layer glue."""
    if windows is None or windows.empty:
        return capacity
    consumed = windows.set_index(["section_id", "date", "window_index"])["minutes_used"]
    capacity = capacity.copy()
    key_index = pd.MultiIndex.from_arrays([capacity["section_id"], capacity["date"], capacity["window_index"]])
    capacity["window_minutes"] = (capacity["window_minutes"] - key_index.map(consumed).fillna(0).to_numpy()).clip(lower=0)
    return capacity


def approved_rows_to_window_rows(approved: list[dict], capacity: pd.DataFrame) -> pd.DataFrame:
    """Session 26, at explicit user request, after finding a real,
    deeper gap: a task approved in an EARLIER /schedule/options ->
    /schedule/approve round occupies a real (section, date, window_index)
    slot forever after -- but nothing about a LATER scheduling run ever
    looked at `store.approved` at all, so (a) that slot's spare room
    (max-not-sum, a different department) was never offered to a new
    task as a combining candidate, and (b) its capacity was never even
    subtracted, so a later run could in principle double-book the SAME
    real window for a second, unrelated task. This expands `approved`
    (store.approved's own per-task rows -- each carrying a `sessions`
    list of (date, window_index, start_minute, end_minute) for a whole/
    split/combined task, or no real window_index at all for a negotiated
    exception) into the same one-row-per-(section,date,window_index)
    shape combined_window_rows() produces,
    restricted to keys that exist in THIS run's own `capacity` table
    (an approved slot outside the current horizon can't be double-booked
    by a run that never offers it, so it's simply dropped, not an error).
    A negotiated-exception row (no real window_index) is skipped for the
    same structural reason force_combinable_placements never targets one
    -- there is no discrete shared slot to offer anyone else."""
    if not approved or capacity.empty:
        return pd.DataFrame(columns=_WINDOW_COLUMNS)

    valid_keys = set(zip(capacity["section_id"], capacity["date"], capacity["window_index"]))
    slots: dict[tuple, dict] = {}
    for r in approved:
        window_index = r.get("window_index")
        sessions = r.get("sessions")
        entries = sessions if sessions else ([(r.get("date"), window_index, r.get("start_minute"), r.get("end_minute"))] if window_index is not None else [])
        for entry in entries:
            entry_date, entry_idx = entry[0], entry[1]
            if entry_idx is None or (r["section_id"], entry_date, entry_idx) not in valid_keys:
                continue
            key = (r["section_id"], entry_date, entry_idx)
            slot = slots.setdefault(key, {
                "section_id": r["section_id"], "date": entry_date, "window_index": entry_idx,
                "start_minute": entry[2] if len(entry) > 2 else None,
                "end_minute": entry[3] if len(entry) > 3 else None,
                "minutes_by_dept": {}, "task_ids": [],
            })
            hours = r.get("estimated_block_hours") or 0.0
            slot["minutes_by_dept"][r["department"]] = max(slot["minutes_by_dept"].get(r["department"], 0.0), hours * 60.0)
            if r["task_id"] not in slot["task_ids"]:
                slot["task_ids"].append(r["task_id"])

    if not slots:
        return pd.DataFrame(columns=_WINDOW_COLUMNS)

    capacity_by_key = capacity.set_index(["section_id", "date", "window_index"])["window_minutes"]
    rows = []
    for key, slot in slots.items():
        rows.append({
            "section_id": slot["section_id"],
            "date": slot["date"],
            "window_index": slot["window_index"],
            "start_minute": slot["start_minute"],
            "end_minute": slot["end_minute"],
            "minutes_used": int(round(max(slot["minutes_by_dept"].values(), default=0.0))),
            "minutes_capacity": int(round(capacity_by_key.get(key, 0.0))),
            "departments": ",".join(sorted(slot["minutes_by_dept"])),
            "task_ids": slot["task_ids"],
            "combined": len(slot["task_ids"]) > 1,
            "required_signoffs": [],
        })
    return pd.DataFrame(rows, columns=_WINDOW_COLUMNS)


def combine_into_occupied_windows(
    occupied_windows: pd.DataFrame,
    ranked_tasks: pd.DataFrame,
    capacity: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict], pd.DataFrame]:
    """Session 26, at explicit user request: "you can also combine
    blocks with the already scheduled blocks which got scheduled in a
    scheduling run long ago also. not just limit this to step A's
    critical task solve." `occupied_windows` is any already-real,
    already-placed set of windows -- either `critical_result.windows`
    from THIS run's own Step A (a Critical task's block, which Step D's
    separate solve over non-critical tasks would otherwise never see
    again) or `approved_rows_to_window_rows(store.approved, ...)` from a
    PAST, already-approved run. Either way the shape is identical
    (combined_window_rows()'s own schema), so one function handles both.

    For each occupied window with fewer than 3 departments already
    present, repeatedly looks for the most urgent (earliest due_date,
    then largest) still-unplaced task in `ranked_tasks` from a
    DIFFERENT department, same section, whose own [raised_date,
    due_date] covers that date, and whose own hours independently fit
    the window's REAL total size (capacity's own window_minutes for that
    exact key -- read BEFORE any capacity-subtraction step, so a new
    department genuinely sees the full "max not sum" room a joint
    disconnection allows, not whatever a separate, deliberately
    conservative subtraction pass already zeroed out for it). Only WHOLE
    (unsplit) tasks are matched here -- a task needing to be split to
    fit is left for the normal solve/combining machinery downstream,
    which already knows how to size parts against real windows; this
    pass's job is only to catch the case that machinery structurally
    can't reach: a slot that's already committed rather than one still
    open for this run's own solve to place things into.

    Returns (remaining_ranked_tasks, forced_full_rows, updated_occupied_
    windows) -- forced_full_rows is ready to merge into _finalize_
    schedule's own `forced_full_rows` list; updated_occupied_windows has
    the same rows as `occupied_windows`, with `departments`/`task_ids`/
    `combined` reflecting whatever got added here (merge into
    `forced_windows` so the frontend's combined-block modal shows the
    full, real group)."""
    if occupied_windows.empty or ranked_tasks.empty or capacity.empty:
        return ranked_tasks, [], occupied_windows

    capacity_by_key = capacity.set_index(["section_id", "date", "window_index"])["window_minutes"]
    placed_ids: set[str] = set()
    forced_full_rows: list[dict] = []
    updated_rows: list[dict] = []

    for _, w in occupied_windows.iterrows():
        key = (w["section_id"], w["date"], w["window_index"])
        window_minutes = capacity_by_key.get(key)
        current_depts: set[str] = set(w["departments"].split(",")) if w["departments"] else set()
        task_ids: list[str] = list(w["task_ids"]) if isinstance(w["task_ids"], list) else []
        # Session 26: `w["minutes_used"]` is already "max across whichever
        # department(s) were there before" -- track it forward so a newly
        # -added department's own minutes can grow it further (never
        # shrink it), giving _subtract_consumed_capacity a real, current
        # figure to conservatively subtract for the WHOLE window, exactly
        # as it always did for a purely-CP-SAT-produced window.
        running_max_minutes = float(w.get("minutes_used") or 0.0)

        if window_minutes is not None and w["window_index"] is not None:
            while len(current_depts) < 3:
                candidates = ranked_tasks[
                    (ranked_tasks["section_id"] == w["section_id"])
                    & (~ranked_tasks["department"].isin(current_depts))
                    & (~ranked_tasks["task_id"].isin(placed_ids))
                    & (ranked_tasks["raised_date"] <= w["date"])
                    & (w["date"] <= ranked_tasks["due_date"])
                    & (ranked_tasks["estimated_block_hours"] * 60.0 <= window_minutes + 0.01)
                ]
                if candidates.empty:
                    break
                most_urgent_due = candidates["due_date"].min()
                pick = candidates[candidates["due_date"] == most_urgent_due].sort_values(
                    "estimated_block_hours", ascending=False
                ).iloc[0]

                placed_ids.add(pick["task_id"])
                current_depts.add(pick["department"])
                task_ids.append(pick["task_id"])
                running_max_minutes = max(running_max_minutes, pick["estimated_block_hours"] * 60.0)
                forced_full_rows.append({
                    "task_id": pick["task_id"],
                    "department": pick["department"],
                    "section_id": w["section_id"],
                    "requester_priority": pick["requester_priority"],
                    "whittle_index": pick["whittle_index"],
                    "estimated_block_hours": pick["estimated_block_hours"],
                    "option": "combined",
                    "n_parts": 1,
                    "sessions": [(w["date"], w["window_index"], w["start_minute"], w["end_minute"])],
                    "date": w["date"],
                    "on_time": bool(w["date"] <= pick["due_date"]),
                    "negotiated_exception": False,
                    "approval_path": pick.get("approval_path", "sm_direct"),
                    "completion_verified": False,
                })

        updated = w.to_dict()
        updated["departments"] = ",".join(sorted(current_depts))
        updated["task_ids"] = task_ids
        updated["combined"] = len(task_ids) > 1
        updated["minutes_used"] = int(round(running_max_minutes))
        updated_rows.append(updated)

    remaining = ranked_tasks[~ranked_tasks["task_id"].isin(placed_ids)].copy()
    updated_occupied = pd.DataFrame(updated_rows, columns=occupied_windows.columns)
    return remaining, forced_full_rows, updated_occupied
