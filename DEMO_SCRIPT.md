# DEMO_SCRIPT — RailBlock Co-Pilot (SIH PS 26027)

This is the exact live sequence to run for judges, plus how to honestly narrate every number
shown — including the ones that are not flattering. Every screen in this demo pulls from the
real FastAPI backend; nothing in the frontend is mock data.

## Before the demo

```bash
# terminal 1 — backend
cd "RailBlock Co-Pilot"
.venv/bin/uvicorn railblock.api.app:app --port 8000

# terminal 2 — frontend
cd "RailBlock Co-Pilot/frontend"
npm run dev
```

Open the URL Vite prints (usually `http://localhost:5173`, but check the terminal — it auto-picks
the next free port if 5173 is taken). Confirm the backend is reachable: `curl http://127.0.0.1:8000/health`
should return `{"status":"ok"}`.

**State now persists in SQLite** (`data/railblock.db`, Session 8) — it survives a backend
restart. For a clean demo, delete that file (or call the reset-for-tests helper) before starting.

---

## Live sequence

### 1. Dashboard (empty state)
Land on the Dashboard. Point out:
- **Pending Requests: 0**, stat cards show placeholders — nothing has been submitted yet.
- The **corridor map** at the bottom: all 102 real points from train 12243's actual timetable
  (Session 4), positioned by real distance-km along the corridor. Say plainly: *this is a
  schematic mileage diagram, not a GPS map* — not all 101 signal cabins/halts have public
  lat/long, so distance-km (which is real, precise data for every point) is the honest axis to
  use, not an approximated geographic layout.

### 2. Raise Block Request
- Fill in one request by hand live (e.g. Signalling / Signal lamp failure / any section / Critical
  / 0.75h / a due date a few weeks out) and submit it — show the returned `task_id`,
  `approval_path` (`sm_direct` since ≤1h), and `splittable` flag, all computed server-side by the
  same logic the backend validates against.
- Then use **"Demo: seed a full batch"** with `double_track_adjusted` (66 tasks/week) to populate
  a realistic queue quickly, instead of submitting 65 more by hand.

### 3. Recommended Scheduling
Click **Generate recommendations**. While it solves (a few seconds — real CP-SAT over the real
101-section corridor), narrate what's happening: Layer 2 ranks by Whittle-index (urgency ×
priority × section congestion), Layer 3's CP-SAT solver places tasks into real leftover capacity,
trying Option 2 (splitting) automatically before falling back, and Option 3 (negotiated exception)
only for Critical tasks that still don't fit.

Point at the result breakdown:
- **Whole / Split** — split is usually the larger number. Say why: *splitting is the mechanism
  that unlocks Engineering work at all on this corridor* — a single Engineering block needs 2h+
  and only 4 of 101 sections ever offer a gap that long (Session 4's finding); splitting a task
  into 2-3 shorter sessions is often the only way it gets done.
- **Negotiated: likely 0.** This is expected, not broken — it only fires for Critical tasks whose
  shortfall is small enough that shifting/cancelling a single goods movement (≤30 min, capped at
  2/section/week) closes the gap. The mechanism itself is proven working by dedicated automated
  tests (Session 3); it just doesn't happen to trigger in every random batch.
- **Partial / Unscheduled** — be upfront that a meaningful fraction won't fit this week. That's
  the corridor's real, capacity-constrained reality, not a solver defect.

### 4. Pending Approval
Show the list of fully-scheduled tasks awaiting sign-off. Approve a few individually (checkbox +
"Approve selected") to show the granular path, then **Approve all** for the rest.

### 5. Weekly / Monthly Schedule
- **Weekly tab**: the section × date matrix now has colored entries — hover one to see which task/
  department/priority it is. Point out a lightning-bolt icon if any negotiated exceptions
  happened to fire.
- **Monthly tab**: click Generate to run Layer 4's tactical monthly plan live — a coarser,
  per-section weekly-hour target over a 4-week horizon, explicitly *not* window-fragmentation-
  aware (say so) — it's a soft target for the weekly runs, not a commitment.

### 6. Completed History
Show the approved list. **Read the banner on this page out loud**: every row's
`completion_verified` is `false` and stays that way — reconnection in real SR practice (SR
3.51.6) requires a Track Fit Certificate and joint SI/SM testing before a block is genuinely
closed out, not just that the scheduled time passed. This system doesn't fabricate a "done"
status it hasn't earned.

### 7. Back to Dashboard (populated state), then KPI Scorecard
This is the number judges will remember — narrate it carefully, in this order:

1. **KPI Scorecard's due-date-aware completion card** — *"Of tasks that had a genuine chance to
   be on time, this is the percentage the system actually schedules on time."* Since Session 10,
   the synthetic task generator anchors every task's due date to its own creation date (every
   priority tier gets at least 10 days), so a freshly generated task is never already overdue —
   the "excluding pre-existing backlog" and "blended" figures are now the same number, and
   `pre_existing_backlog_pct` reads 0% by construction. Say this plainly if asked rather than
   narrating a backlog story that no longer applies: **through Session 9** the generator could
   backdate a task's raise date well past its own due window, and Session 6 measured that at
   60-62% of a batch already overdue at creation — that number describes the OLD generator, not
   the one running now.
   The SLA due-date windows themselves (Critical 10-20 days / Moderate 20-40 / Routine 45-60) are
   still an **unsourced internal placeholder assumption**, not a real Indian Railways policy — a
   real, dated 2024 Railway Board "Rolling Block Plan guidelines" circular exists (26-week rolling
   horizon, 4-week advance notice for >4h mega blocks) but describes a structurally different
   scheme not yet reconciled with this generator — flagged honestly as a known limitation, not
   silently assumed correct.
2. **By-department bars** — point at Engineering's low bar explicitly. *"This is real and robust
   — confirmed under two independent measurement approaches (a fixed test-window and this
   due-date-aware one) — only 4 of 101 block sections on this corridor ever offer a contiguous gap
   long enough for a typical Engineering possession without splitting. That's a genuine structural
   finding about this corridor's capacity, not something we're hiding or explaining away."*
3. **Asset uptime (~99%+)** — say plainly this is expected *by construction*, not a claimed win:
   the system only ever schedules into leftover capacity trains don't need, so it was never going
   to show meaningfully less than ~100%.
4. **Corridor map with day selector** — click through a couple of dates, showing which sections
   are lit up on which day, colored by department.

---

## If asked hard questions

- **"Is this trained on real data?"** Layer 2/3 (Whittle-index ranking, CP-SAT constraint
  programming) are algorithmic, not learned — say that plainly. Session 8 added two *actually
  trained* scikit-learn classifiers on top, each with a published model card
  (`GET /ml/model-cards`): `unscheduled_risk` is trained on REAL computed outcomes from this
  system's own rolling-horizon runs (still over synthetic task batches — no real IR defect
  dataset exists); `delay_risk` is trained on ENTIRELY SIMULATED overrun labels (a documented
  noise model, since no real block-execution-duration data exists anywhere). Say this distinction
  out loud if asked — neither is a validated real-world predictor, and the second is explicitly a
  demo mechanism, not a claim.
- **"Why is the on-time rate so low?"** No pre-existing backlog is involved (Session 10 removed
  that by construction, see point 1 above) — it's genuine corridor capacity constraint, most
  visibly on Engineering. Say that plainly, and note the SLA due-date windows themselves are still
  an unsourced assumption, flagged for future correction against the real Railway Board circular.
- **"Does this handle [X real BDMS/COA feature]?"** BDMS itself is explicitly out of scope — this
  system takes requester-assigned priority as given input, never predicts it. If X is genuinely
  out of scope, say so rather than improvising an answer.
- **"What real datasets back this?"** Train_details_22122017.csv (186k rows, real IR timetable),
  india_railway_stations.csv, train 12243's real timetable (102-point corridor), the Southern
  Railway ASM training guide (SR 3.51.6 disconnection procedure), and real Rajya Sabha
  parliamentary delay/punctuality data for the KPI baselines. Freight forecasts, defect/task
  records, and demand volume are synthetic — clearly labeled as such everywhere in the code and
  API, never presented as real.

## Known limitations to volunteer if the demo runs long or judges probe deeper
- Corridor map is schematic (distance-km), not a literal GPS map.
- SLA due-date windows are an unsourced placeholder assumption (see above) — flagged, not fixed,
  this session; a real but structurally different IR document exists for future reconciliation.
- No completion-verification workflow exists yet (Track Fit Certificate / joint SI-SM test) —
  "History" means "approved", never "verified done".
- Backend state is now SQLite-persisted (Session 8) but still single-process, no auth — a real
  deployment would need a production datastore and access control, out of scope for this prototype.
- The two ML models (Session 8) are real trained classifiers, but bounded by what's described
  above — `unscheduled_risk`'s labels come from this system's own synthetic-task runs, and
  `delay_risk`'s labels are entirely simulated (no real execution-time data exists).
