// Shared day x hour (24 one-hour columns) schedule grid, used by both the
// Recommended Scheduling "view full schedule" modal and the Weekly/Monthly
// Schedule page. Renders one row per date; each hour cell is filled
// proportionally by real minute-level overlap -- a train that only passes
// through for 10 minutes shades ~17% of the cell, not the whole hour, and
// a maintenance block sharing that hour with a train shades its own real
// share alongside it. Maintenance windows never actually overlap a real
// train transit in this system's data (Layer 1 only offers windows inside
// FREE time), so the two kinds of segment never need to fight for the
// same minute.
//
// Session 29, at explicit user request: this file exports ONLY its
// default component now -- trainKey moved to utils/trainKey.js, since a
// file mixing a component export with a plain-function export can't be
// cleanly Fast-Refreshed by Vite (see that file's own comment for the
// real bug this caused).
import { trainKey } from "../utils/trainKey.js";

const DEPT_CLASS = { Engineering: "eng", Signalling: "sig", Traction: "trd" };
const HOURS = Array.from({ length: 24 }, (_, h) => h);

function segmentsForHour(entries, hourStart, hourEnd) {
  const segs = [];
  for (const e of entries) {
    if (e.start_minute == null || e.end_minute == null) continue;
    const s = Math.max(e.start_minute, hourStart);
    const en = Math.min(e.end_minute, hourEnd);
    if (en > s) segs.push({ ...e, segStart: s, segEnd: en });
  }
  segs.sort((a, b) => a.segStart - b.segStart);
  return segs;
}

// Session 29, at explicit user request, after a real reported bug: a
// multi-hour block's department-initial label wasn't showing AT ALL, in
// either Weekly or Monthly, for every single block -- not new, and not
// specific to any one dataset. Root cause: the old code tracked "have I
// already labeled this task_id in this row" with a plain Set MUTATED as
// a side effect during render (shownTaskIds.add(...)), shared across
// every HourCell in that row. React 18 StrictMode (enabled in main.jsx)
// deliberately double-invokes render functions to catch exactly this
// kind of impurity -- the second invocation of an hour's HourCell saw
// the Set already containing its own task_id (added by the first,
// throwaway invocation), so even the block's OWN FIRST hour rendered as
// "already labeled" and showed nothing. Replaced with a pure, read-only
// lookup computed once per day from `blocks` itself (safe to compute
// twice under StrictMode -- same input, same answer, no shared mutable
// state) instead of a stateful side effect.
function firstLabelHourByTask(blocks) {
  const map = new Map();
  for (const b of blocks) {
    if (b.start_minute == null) continue; // negotiated-exception approx blocks use their own hour-0 marker, not this
    const hour = Math.floor(b.start_minute / 60);
    const existing = map.get(b.task_id);
    if (existing === undefined || hour < existing) map.set(b.task_id, hour);
  }
  return map;
}

function HourCell({ date, blocks, trains, hour, onBlockClick, onTrainClick, labelHourByTask, highlightedTrains }) {
  const hourStart = hour * 60;
  const hourEnd = hourStart + 60;

  // Negotiated-exception blocks (no known start/end minute) get a
  // full-cell approximate marker at hour 0 -- never invented precision.
  // Session 23: colored by department like any other block, not a
  // special "combined" color. Session 28, at explicit user request: the
  // ⚡ icon that used to fill the cell was removed as unnecessary -- the
  // department initial (same convention the normal per-segment cells
  // below use) plus the tooltip/details modal already say "negotiated
  // exception" clearly enough on their own.
  const approxNegotiated = blocks.find((b) => b.negotiated_exception && b.start_minute == null);
  if (approxNegotiated && hour === 0) {
    return (
      <td>
        <div
          className={`hour-cell-block ${DEPT_CLASS[approxNegotiated.department] || "eng"}`}
          style={{ width: "100%", height: "100%" }}
          title={`${approxNegotiated.task_id} (negotiated exception — approximate time, not modeled) · ${approxNegotiated.department}`}
          onClick={() => onBlockClick?.(approxNegotiated)}
        >
          {approxNegotiated.department[0]}
        </div>
      </td>
    );
  }

  const blockSegs = segmentsForHour(blocks, hourStart, hourEnd).map((s) => ({ ...s, kind: "block" }));
  const trainSegs = segmentsForHour(trains, hourStart, hourEnd).map((s) => ({ ...s, kind: "train" }));
  const all = [...blockSegs, ...trainSegs].sort((a, b) => a.segStart - b.segStart);

  if (all.length === 0) return <td />;

  const pieces = [];
  let cursor = hourStart;
  for (const seg of all) {
    if (seg.segStart > cursor) {
      pieces.push({ width: ((seg.segStart - cursor) / 60) * 100, kind: "free" });
    }
    pieces.push({ width: ((seg.segEnd - seg.segStart) / 60) * 100, kind: seg.kind, data: seg });
    cursor = Math.max(cursor, seg.segEnd);
  }
  if (cursor < hourEnd) pieces.push({ width: ((hourEnd - cursor) / 60) * 100, kind: "free" });

  return (
    <td>
      <div style={{ display: "flex", width: "100%", height: "100%" }}>
        {pieces.map((p, i) => {
          if (p.kind === "free") return <div key={i} style={{ width: `${p.width}%`, height: "100%" }} />;
          if (p.kind === "train") {
            const color = highlightedTrains?.[trainKey(date, p.data)];
            return (
              <div
                key={i}
                className="hour-cell-train"
                style={{
                  width: `${p.width}%`,
                  height: "100%",
                  ...(color
                    ? { background: `repeating-linear-gradient(45deg, ${color}, ${color} 3px, #fff 3px, #fff 6px)` }
                    : {}),
                }}
                title={`${p.data.train_name || p.data.train_no} (${p.data.train_no}) passing`}
                onClick={() => onTrainClick?.(date, p.data)}
              />
            );
          }
          const b = p.data;
          // Session 30, at explicit user request: an emergency block gets
          // its own distinct alert color instead of a department color --
          // it's not routine departmental work, and should read as an
          // "EMERGENCY SITUATION" at a glance.
          const cls = b.is_emergency ? "emg" : b.combined ? "combined" : DEPT_CLASS[b.department] || "eng";
          const isFirstHourForThisTask = labelHourByTask.get(b.task_id) === hour;
          return (
            <div
              key={i}
              className={`hour-cell-block ${cls}`}
              style={{ width: `${p.width}%`, height: "100%", fontSize: 8 }}
              title={b.is_emergency ? `${b.task_id} — EMERGENCY SITUATION · ${b.department}` : `${b.task_id}${b.negotiated_exception ? " (negotiated exception)" : ""} · ${b.department}`}
              onClick={() => onBlockClick?.(b)}
            >
              {isFirstHourForThisTask ? (b.is_emergency ? "!" : b.department[0]) : ""}
            </div>
          );
        })}
      </div>
    </td>
  );
}

export default function HourGrid({ days, onBlockClick, onTrainClick, highlightedTrains }) {
  return (
    <div className="matrix-wrap">
      <table className="hour-matrix">
        <thead>
          <tr>
            <th className="day-col">Date</th>
            {HOURS.map((h) => (
              <th key={h}>{String(h).padStart(2, "0")}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {days.map((day) => {
            const labelHourByTask = firstLabelHourByTask(day.blocks || []);
            return (
              <tr key={day.date}>
                <td className="day-label">{day.label || day.date}</td>
                {HOURS.map((h) => (
                  <HourCell
                    key={h}
                    date={day.date}
                    hour={h}
                    blocks={day.blocks || []}
                    trains={day.trains || []}
                    onBlockClick={onBlockClick}
                    onTrainClick={onTrainClick}
                    labelHourByTask={labelHourByTask}
                    highlightedTrains={highlightedTrains}
                  />
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
