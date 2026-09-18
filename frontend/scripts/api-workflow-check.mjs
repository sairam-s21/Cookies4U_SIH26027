// Drives the real submit -> seed -> recommend -> approve workflow against
// the live API and verifies every shape the frontend reads post-workflow
// (weekly matrix cells, history rows, live KPI "current", the approval
// queue) -- the substitute for browser testing described in
// api-shape-check.mjs's header comment.
const BASE = "http://127.0.0.1:8000";

async function post(path, body) {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`POST ${path} -> ${res.status}: ${await res.text()}`);
  return res.json();
}
async function get(path) {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`GET ${path} -> ${res.status}: ${await res.text()}`);
  return res.json();
}

async function main() {
  let ok = true;
  const fail = (msg) => {
    ok = false;
    console.log(`  FAIL: ${msg}`);
  };

  console.log("1) submit one explicit request (Raise Block Request page path)");
  const submitted = await post("/requests", {
    department: "Signalling",
    section_id: "MAS-BBQ",
    defect_type: "Signal lamp failure",
    requester_priority: "Critical",
    estimated_block_hours: 0.75,
    due_date: "2026-12-31",
  });
  console.log(`  created ${submitted.task_id}, status=${submitted.status}`);

  console.log("2) seed a demo batch (double_track_adjusted)");
  const seeded = await post("/demo/seed", { demand_scenario: "double_track_adjusted", start_date: "2026-09-07", seed: 777 });
  console.log(`  added ${seeded.added} pending requests`);

  console.log("3) generate recommendations");
  const rec = await post("/schedule/recommend", { start_date: "2026-09-07", n_days: 7, time_limit_s: 20 });
  console.log(`  status=${rec.status} scheduled=${rec.schedule.length} partial=${rec.partial.length} unscheduled=${rec.unscheduled_task_ids.length}`);
  if (rec.schedule.length === 0) fail("nothing fully scheduled -- can't verify the approve/history/weekly path with this seed");
  const firstRow = rec.schedule[0];
  for (const k of ["task_id", "department", "section_id", "date", "option", "requester_priority", "on_time"]) {
    if (!(k in firstRow)) fail(`schedule row missing '${k}' (RecommendedScheduling.jsx/PendingApproval.jsx read this)`);
  }

  console.log("4) GET /schedule/recommendation (Pending Approval page's data source)");
  const peek = await get("/schedule/recommendation");
  if (!peek.available) fail("peek.available should be true after a recommend call");
  const allRequests = await get("/requests");
  const byId = Object.fromEntries(allRequests.map((r) => [r.task_id, r]));
  const awaiting = peek.recommendation.schedule.filter((r) => byId[r.task_id]?.status === "recommended_full");
  console.log(`  ${awaiting.length} task(s) awaiting approval`);
  if (awaiting.length === 0) fail("expected at least one recommended_full task awaiting approval");

  console.log("5) approve all");
  const approved = await post("/schedule/approve", {});
  console.log(`  approved_count=${approved.approved_count}`);
  if (approved.approved_count !== rec.schedule.length) fail("approved_count should match the full recommend schedule length");

  console.log("6) GET /schedule/weekly (Schedule.jsx weekly tab's data source)");
  const weekly = await get("/schedule/weekly");
  if (!weekly.dates.length) fail("weekly.dates should be non-empty after approval");
  if (!weekly.sections.length) fail("weekly.sections should be non-empty after approval");
  const sampleSection = weekly.sections[0];
  const sampleDate = weekly.dates[0];
  const cellEntries = weekly.cells[sampleSection]?.[sampleDate];
  console.log(`  sample cells[${sampleSection}][${sampleDate}] = ${JSON.stringify(cellEntries)}`);
  // CorridorMap.jsx / Schedule.jsx read: e.department, e.task_id, e.window_index, e.requester_priority, e.negotiated_exception, e.completion_verified
  let anyEntry = null;
  for (const s of weekly.sections) {
    for (const d of weekly.dates) {
      const es = weekly.cells[s]?.[d];
      if (es && es.length) { anyEntry = es[0]; break; }
    }
    if (anyEntry) break;
  }
  if (!anyEntry) fail("could not find any populated weekly cell to verify entry shape");
  else {
    for (const k of ["task_id", "department", "window_index", "requester_priority", "negotiated_exception", "completion_verified"]) {
      if (!(k in anyEntry)) fail(`weekly cell entry missing '${k}'`);
    }
  }

  console.log("7) GET /tasks/history (History.jsx's data source)");
  const history = await get("/tasks/history");
  if (history.count === 0) fail("history.count should be > 0 after approval");
  const histRow = history.tasks[0];
  for (const k of ["task_id", "department", "section_id", "date", "option", "requester_priority", "approval_path", "on_time", "completion_verified"]) {
    if (!(k in histRow)) fail(`history row missing '${k}'`);
  }
  console.log(`  completion_verified on sample row = ${histRow.completion_verified} (must be false)`);
  if (histRow.completion_verified !== false) fail("completion_verified must be false — never silently implied complete");

  console.log("8) GET /kpi (Dashboard.jsx's live stat cards, now that a recommendation exists)");
  const kpi = await get("/kpi");
  if (kpi.current === null) fail("kpi.current should be populated now that a recommendation has run");
  else {
    for (const k of ["asset_uptime_pct", "block_utilization_rate_pct", "priority_task_completion", "coordination_rate"]) {
      if (!(k in kpi.current.metrics)) fail(`kpi.current.metrics missing '${k}'`);
    }
    console.log(`  current.metrics.asset_uptime_pct = ${kpi.current.metrics.asset_uptime_pct}`);
    console.log(`  current.metrics.coordination_rate = ${JSON.stringify(kpi.current.metrics.coordination_rate)}`);
  }

  console.log(`\n${ok ? "WORKFLOW CHECK PASSED" : "WORKFLOW CHECK FAILED"}`);
  process.exit(ok ? 0 : 1);
}

main().catch((e) => {
  console.error("UNCAUGHT:", e.message);
  process.exit(1);
});
