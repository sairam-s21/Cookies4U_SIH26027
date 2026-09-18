// Verifies the Session 9 backend additions the redesigned UI depends on:
// real station geo, real train positions, section train schedule,
// multi-option scheduling + approve-by-option, and the seed-batch
// retract-on-reload flow (DELETE /requests). Run against a throwaway
// backend state.
const BASE = "http://127.0.0.1:8000";

async function post(path, body) {
  const res = await fetch(`${BASE}${path}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!res.ok) throw new Error(`POST ${path} -> ${res.status}: ${await res.text()}`);
  return res.json();
}
async function del(path, body) {
  const res = await fetch(`${BASE}${path}`, { method: "DELETE", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!res.ok) throw new Error(`DELETE ${path} -> ${res.status}: ${await res.text()}`);
  return res.json();
}
async function get(path) {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`GET ${path} -> ${res.status}: ${await res.text()}`);
  return res.json();
}

async function main() {
  let ok = true;
  const fail = (msg) => { ok = false; console.log(`  FAIL: ${msg}`); };

  console.log("1) GET /corridor has real lat/lon for major stations (Dashboard's map)");
  const corridor = await get("/corridor");
  const mas = corridor.stations.find((s) => s.station_code === "MAS");
  if (mas.lat == null || mas.lon == null) fail("MAS missing real lat/lon");
  else console.log(`  MAS = (${mas.lat}, ${mas.lon})`);

  console.log("2) GET /trains/positions returns real transits at 06:00 on a Monday");
  const positions = await get("/trains/positions?date=2026-09-07&minute=360");
  if (!positions.trains.length) fail("expected real trains transiting at 06:00");
  else console.log(`  ${positions.trains.length} trains, e.g. ${JSON.stringify(positions.trains[0])}`);

  console.log("3) GET /corridor/sections/{id}/trains (Weekly/Monthly train overlay)");
  const sectionId = corridor.sections[0].section_id;
  const sectionTrains = await get(`/corridor/sections/${sectionId}/trains?date=2026-09-07`);
  console.log(`  ${sectionId}: ${sectionTrains.trains.length} real train transit(s) that day`);

  console.log("4) seed a batch, generate 3-5 schedule options, approve by option_key");
  const seeded = await post("/demo/seed", { demand_scenario: "double_track_adjusted", start_date: "2026-09-07", seed: 9 });
  if (!seeded.task_ids || seeded.task_ids.length !== seeded.added) fail("seed response missing task_ids for retract-on-reload tracking");
  const opts = await post("/schedule/options", { start_date: "2026-09-07", n_days: 7, time_limit_s: 10 });
  if (opts.options.length < 3 || opts.options.length > 5) fail(`expected 3-5 options, got ${opts.options.length}`);
  for (const o of opts.options) {
    if (!o.pros.length || !o.cons.length) fail(`option ${o.key} missing pros/cons`);
    if (o.start_minute !== undefined) fail("unexpected field leak");
  }
  const balanced = opts.options.find((o) => o.key === "balanced");
  if (balanced.schedule.length) {
    const row = balanced.schedule.find((r) => !r.negotiated_exception);
    const hasTime = row && ((row.sessions && row.sessions[0] && row.sessions[0][2] != null) || row.start_minute != null);
    if (!hasTime) fail("schedule rows missing a start_minute (top-level or inside sessions) for the hour-grid");
  }
  if (balanced.schedule.length) {
    const approved = await post("/schedule/approve", { option_key: "balanced" });
    console.log(`  approved ${approved.approved_count} task(s) from the balanced option`);
  }

  console.log("5) DELETE /requests retracts a still-pending demo batch, leaves approved tasks alone");
  const seeded2 = await post("/demo/seed", { demand_scenario: "double_track_adjusted", start_date: "2026-09-07", seed: 11 });
  const beforeCount = (await get("/requests?status=pending")).length;
  const delResult = await del("/requests", { task_ids: seeded2.task_ids });
  if (delResult.deleted !== seeded2.task_ids.length) fail(`expected to delete ${seeded2.task_ids.length}, deleted ${delResult.deleted}`);
  const afterCount = (await get("/requests?status=pending")).length;
  if (afterCount !== beforeCount - seeded2.task_ids.length) fail("pending count did not drop by the retracted batch size");
  const historyAfter = await get("/tasks/history");
  if (!historyAfter.count) fail("approved tasks from step 4 should still be present after retracting the unrelated batch");
  else console.log(`  history still has ${historyAfter.count} approved task(s) -- untouched by the retraction`);

  console.log(`\n${ok ? "SESSION 9 CHECK PASSED" : "SESSION 9 CHECK FAILED"}`);
  process.exit(ok ? 0 : 1);
}

main().catch((e) => {
  console.error("UNCAUGHT:", e.message);
  process.exit(1);
});
