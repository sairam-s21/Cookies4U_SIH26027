// Verifies the REAL running API's response shapes match exactly what each
// page component reads (dot-path access), without needing a browser. This
// is a substitute for visual browser testing, which isn't possible in this
// sandbox (missing libnspr4/libnss3, no passwordless sudo to install them)
// -- run against the live server so it's checking reality, not a fixture.
const BASE = "http://127.0.0.1:8000";

function get(obj, path) {
  return path.split(".").reduce((o, k) => (o == null ? undefined : o[k]), obj);
}

async function check(label, path, fetcher, requiredPaths) {
  console.log(`\n=== ${label} ===`);
  const data = await fetcher();
  let ok = true;
  for (const p of requiredPaths) {
    const v = get(data, p);
    const present = v !== undefined;
    if (!present) ok = false;
    console.log(`  ${present ? "OK  " : "MISS"} ${p} = ${JSON.stringify(v)?.slice(0, 80)}`);
  }
  return { ok, data };
}

async function main() {
  let allOk = true;

  const { ok: ok1 } = await check("GET /health", "/health", () => fetch(`${BASE}/health`).then((r) => r.json()), ["status"]);
  allOk &&= ok1;

  const { ok: ok2, data: corridor } = await check(
    "GET /corridor",
    "/corridor",
    () => fetch(`${BASE}/corridor`).then((r) => r.json()),
    ["stations", "sections"]
  );
  allOk &&= ok2;
  console.log(`  stations[0] keys: ${Object.keys(corridor.stations[0]).join(", ")}`);
  console.log(`  sections[0] keys: ${Object.keys(corridor.sections[0]).join(", ")}`);
  // CorridorMap.jsx reads: st.distance_km, st.station_code, st.station_name; sec.from_km, sec.to_km, sec.length_km, sec.section_id
  for (const k of ["station_code", "station_name", "distance_km"]) {
    if (!(k in corridor.stations[0])) { console.log(`  MISSING on station: ${k}`); allOk = false; }
  }
  for (const k of ["section_id", "from_km", "to_km", "length_km", "from_name", "to_name"]) {
    if (!(k in corridor.sections[0])) { console.log(`  MISSING on section: ${k}`); allOk = false; }
  }

  const { ok: ok3 } = await check(
    "GET /meta",
    "/meta",
    () => fetch(`${BASE}/meta`).then((r) => r.json()),
    ["departments", "defect_types_by_department", "priorities", "sm_direct_approval_max_hours"]
  );
  allOk &&= ok3;

  const { ok: ok4 } = await check(
    "GET /kpi (Dashboard.jsx's exact read paths)",
    "/kpi",
    () => fetch(`${BASE}/kpi`).then((r) => r.json()),
    [
      "current",
      "current_note",
      "demand_scenario_comparison.primary_measurement",
      "demand_scenario_comparison.due_date_aware",
      "demand_scenario_comparison.due_date_aware_note",
      "demand_scenario_comparison.due_date_aware.scenarios.stress_test.completion.blended.on_time_pct",
      "demand_scenario_comparison.due_date_aware.scenarios.stress_test.completion.excluding_pre_existing_backlog.on_time_pct",
      "demand_scenario_comparison.due_date_aware.scenarios.stress_test.completion.pre_existing_backlog_pct",
      "demand_scenario_comparison.due_date_aware.scenarios.stress_test.completion.assumption_caveat",
      "demand_scenario_comparison.due_date_aware.scenarios.stress_test.completion.by_department",
      "demand_scenario_comparison.due_date_aware.scenarios.double_track_adjusted.completion.blended.on_time_pct",
      "demand_scenario_comparison.fixed_window_7day_legacy",
    ]
  );
  allOk &&= ok4;

  const { ok: ok5 } = await check(
    "GET /schedule/recommendation (before any recommend call)",
    "/schedule/recommendation",
    () => fetch(`${BASE}/schedule/recommendation`).then((r) => r.json()),
    ["available", "recommendation"]
  );
  allOk &&= ok5;

  const { ok: ok6 } = await check(
    "GET /schedule/weekly",
    "/schedule/weekly",
    () => fetch(`${BASE}/schedule/weekly`).then((r) => r.json()),
    ["dates", "sections", "cells"]
  );
  allOk &&= ok6;

  const { ok: ok7 } = await check(
    "GET /tasks/history",
    "/tasks/history",
    () => fetch(`${BASE}/tasks/history`).then((r) => r.json()),
    ["count", "tasks"]
  );
  allOk &&= ok7;

  console.log(`\n${allOk ? "ALL SHAPES OK" : "SOME SHAPES MISSING — fix before trusting the UI"}`);
  process.exit(allOk ? 0 : 1);
}

main();
