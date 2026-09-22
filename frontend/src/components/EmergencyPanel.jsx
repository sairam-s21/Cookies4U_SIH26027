import { useEffect, useMemo, useState } from "react";
import api from "../api/client.js";
import HourGrid from "./HourGrid.jsx";
import BlockDetailsModal from "./BlockDetailsModal.jsx";
import TrainDetailsModal from "./TrainDetailsModal.jsx";
import { trainKey } from "../utils/trainKey.js";
import { ErrorBanner, InfoBanner, Spinner } from "./StatusBanner.jsx";

// Same palette/behavior as Schedule.jsx and RecommendedScheduling.jsx's
// own persistent train-block highlighting -- kept identical here too, at
// explicit user request, so a train clicked in this preview reads the
// same way as everywhere else in the app.
const HIGHLIGHT_COLORS = [
  "#E4572E", "#17BEBB", "#FFC914", "#2E86AB", "#A23B72",
  "#8E44AD", "#27AE60", "#D81159", "#F18F01", "#0B7A75",
];

// Same rounding convention as Dashboard.jsx's own fmtMinute -- a
// rescheduled slot's minute values come straight out of the CP-SAT
// solve's float arithmetic (e.g. 694.5639...), which must be rounded
// for display, not truncated raw.
function fmtMinute(m) {
  if (m == null) return "—";
  const wrapped = ((m % 1440) + 1440) % 1440;
  const h = Math.floor(wrapped / 60);
  const min = Math.round(wrapped % 60);
  return `${String(h).padStart(2, "0")}:${String(min).padStart(2, "0")}`;
}

const STATUS_TAG = { proposed: "crit", resolved: "rou", discarded: "pending" };

// Groups the emergency's own block plus every affected task's PROPOSED
// session by section, in the same {date, blocks} shape HourGrid already
// renders everywhere else, so approving an emergency reschedule is a
// genuine "I looked at where this lands" decision, not a blind click
// over a plain table.
//
// Same window-grouping Schedule.jsx's own days-builder uses, and for the
// same reason: a session's own `proposed.combined` flag only reflects
// whichever OTHER task it organically combined with, which may not even
// be part of THIS emergency's affected list -- two affected tasks that
// both land in this preview's same (section, date, window_index) need
// their own local grouping to render as one shared block and open
// together on click, rather than trusting each one's flag in isolation
// (which can disagree even when they're genuinely sharing the window).
function buildSectionDays(emergency) {
  const bySection = {};
  const push = (sectionId, block) => {
    (bySection[sectionId] ??= []).push(block);
  };

  for (const seg of emergency.segments) {
    push(emergency.section_id, {
      task_id: emergency.task_id,
      department: emergency.department,
      is_emergency: true,
      combined: false,
      related: [emergency.task_id],
      date: seg.date,
      start_minute: seg.start_minute,
      end_minute: seg.end_minute,
    });
  }

  const affectedEntries = [];
  for (const a of emergency.affected || []) {
    if (!a.proposed) continue;
    const sessions = a.proposed.sessions?.length
      ? a.proposed.sessions
      : [{
          date: a.proposed.date,
          window_index: a.proposed.window_index,
          start_minute: a.proposed.start_minute,
          end_minute: a.proposed.end_minute,
        }];
    for (const s of sessions) {
      affectedEntries.push({
        task_id: a.task_id,
        department: a.department,
        section_id: a.section_id,
        date: s.date,
        window_index: s.window_index,
        start_minute: s.start_minute,
        end_minute: s.end_minute,
      });
    }
  }

  const byWindow = {};
  for (const e of affectedEntries) {
    if (e.window_index == null) continue;
    const key = `${e.section_id}|${e.date}|${e.window_index}`;
    (byWindow[key] ??= []).push(e);
  }
  for (const e of affectedEntries) {
    const key = e.window_index != null ? `${e.section_id}|${e.date}|${e.window_index}` : null;
    const sameWindow = key ? byWindow[key] : [e];
    const combined = sameWindow.length > 1 && new Set(sameWindow.map((x) => x.department)).size > 1;
    push(e.section_id, {
      task_id: e.task_id,
      department: e.department,
      is_emergency: false,
      combined,
      related: combined ? sameWindow.map((x) => x.task_id) : [e.task_id],
      date: e.date,
      start_minute: e.start_minute,
      end_minute: e.end_minute,
    });
  }

  const out = {};
  for (const [sectionId, blocks] of Object.entries(bySection)) {
    const byDate = {};
    for (const b of blocks) (byDate[b.date] ??= []).push(b);
    out[sectionId] = Object.keys(byDate)
      .sort()
      .map((date) => ({ date, label: date, blocks: byDate[date], trains: [] }));
  }
  return out;
}

export default function EmergencyPanel() {
  const [emergencies, setEmergencies] = useState([]);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState(null);
  const [resolving, setResolving] = useState(null);
  const [previewing, setPreviewing] = useState(null); // the emergency currently being reviewed
  const [resetting, setResetting] = useState(false);

  function load() {
    api
      .listEmergencies()
      .then((r) => setEmergencies(r.emergencies))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }

  useEffect(() => {
    load();
  }, []);

  // Every click always POSTs a fresh emergency, never disabled/deduped
  // once one already exists.
  async function handleCreate() {
    setCreating(true);
    setError(null);
    try {
      await api.createEmergency();
      load();
    } catch (e) {
      setError(e.message);
    } finally {
      setCreating(false);
    }
  }

  async function handleResolve(taskId, apply) {
    setResolving(taskId);
    setError(null);
    try {
      await api.resolveEmergency(taskId, apply);
      setPreviewing(null);
      load();
    } catch (e) {
      setError(e.message);
    } finally {
      setResolving(null);
    }
  }

  // Clears every emergency AND its reassignment overrides for this
  // visitor -- a displaced task's reassignment is only ever an override
  // on top of its stored data (never an overwrite), so this alone puts
  // every displaced task back at its original Weekly/Monthly Schedule
  // slot.
  async function handleReset() {
    setResetting(true);
    setError(null);
    try {
      await api.resetEmergencies();
      load();
    } catch (e) {
      setError(e.message);
    } finally {
      setResetting(false);
    }
  }

  return (
    <div className="stack">
      <p className="text-small text-muted" style={{ marginTop: -4 }}>
        Simulates a real emergency (e.g. a rail fracture) that has to be handled immediately, with no approval
        needed -- it's placed on the schedule right away. Any already-scheduled maintenance block it displaces
        (anywhere on the corridor, since an incident can cause trains to be stopped or rerouted elsewhere too) is
        offered a proposed new slot below -- view where it lands, then approve or discard.
      </p>
      <div style={{ display: "flex", gap: 8 }}>
        <button className="btn btn-primary" onClick={handleCreate} disabled={creating} style={{ alignSelf: "flex-start" }}>
          {creating ? "Creating…" : "Create a demo emergency situation"}
        </button>
        {emergencies.length > 0 && (
          <button className="btn btn-danger" onClick={handleReset} disabled={resetting} style={{ alignSelf: "flex-start" }}>
            {resetting ? "Resetting…" : "Reset"}
          </button>
        )}
      </div>
      <ErrorBanner message={error} />
      {loading ? (
        <Spinner />
      ) : emergencies.length === 0 ? (
        <InfoBanner>No emergencies created yet this session.</InfoBanner>
      ) : (
        <div className="stack" style={{ gap: 14 }}>
          {[...emergencies].reverse().map((em) => (
            <EmergencyCard
              key={em.task_id}
              emergency={em}
              resolving={resolving}
              onResolve={handleResolve}
              onPreview={() => setPreviewing(em)}
            />
          ))}
        </div>
      )}

      {previewing && (
        <SchedulePreviewModal
          emergency={previewing}
          resolving={resolving}
          onApprove={() => handleResolve(previewing.task_id, true)}
          onClose={() => setPreviewing(null)}
        />
      )}
    </div>
  );
}

function EmergencyCard({ emergency, resolving, onResolve, onPreview }) {
  // The emergency's window is anchored to whatever task it was
  // guaranteed to overlap (see create_demo_emergency), which can start
  // later than `created_at` (when the button was actually clicked) --
  // read the displayed window from the segments themselves, not
  // recomputed from created_at + duration, which would show the wrong
  // time whenever the two diverge.
  const firstSeg = emergency.segments[0];
  const lastSeg = emergency.segments[emergency.segments.length - 1];
  const windowLabel = `${firstSeg.date} ${fmtMinute(firstSeg.start_minute)} – ${lastSeg.date} ${fmtMinute(lastSeg.end_minute)}`;
  const affected = emergency.affected || [];

  return (
    <div className="card" style={{ background: "var(--red-bg)", border: "1px solid #F3C7C1" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: 8 }}>
        <div>
          <div style={{ fontWeight: 700 }}>{emergency.task_id} — {emergency.defect_type}</div>
          <div className="text-small text-muted">
            {emergency.department} · {emergency.section_id} · {windowLabel} ({emergency.estimated_block_hours}h)
          </div>
        </div>
        <span className={`tag ${STATUS_TAG[emergency.status] || "pending"}`}>
          <span className="tag-dot" />
          {emergency.status}
        </span>
      </div>

      {affected.length === 0 ? (
        <p className="text-small text-muted" style={{ marginTop: 10, marginBottom: 0 }}>No already-scheduled blocks were affected.</p>
      ) : (
        <div className="table-scroll" style={{ marginTop: 10 }}>
          <table>
            <thead>
              <tr>
                <th>Task</th>
                <th>Department</th>
                <th>Section</th>
                <th>Old slot</th>
                <th>Proposed slot</th>
              </tr>
            </thead>
            <tbody>
              {affected.map((a) => (
                <tr key={a.task_id}>
                  <td className="id-cell">{a.task_id}</td>
                  <td>{a.department}</td>
                  <td>{a.section_id}</td>
                  <td>{a.old.date} {fmtMinute(a.old.start_minute)}–{fmtMinute(a.old.end_minute)}</td>
                  <td>
                    {a.proposed ? (
                      <>
                        {a.proposed.date} {fmtMinute(a.proposed.start_minute)}–{fmtMinute(a.proposed.end_minute)}
                        {a.proposed.combined ? " (combined)" : ""}
                        {a.proposed.n_parts > 1 ? ` · split across ${a.proposed.n_parts} sessions` : ""}
                      </>
                    ) : (
                      <span className="text-muted">No alternative slot found</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {emergency.status === "proposed" && (
        <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
          <button
            className="btn btn-primary"
            disabled={resolving === emergency.task_id || affected.length === 0}
            onClick={onPreview}
          >
            View proposed schedule & approve
          </button>
          <button
            className="btn btn-outline"
            disabled={resolving === emergency.task_id || affected.length === 0}
            onClick={() => onResolve(emergency.task_id, false)}
          >
            Discard
          </button>
        </div>
      )}
    </div>
  );
}

function SchedulePreviewModal({ emergency, resolving, onApprove, onClose }) {
  const sectionDays = useMemo(() => buildSectionDays(emergency), [emergency]);
  const sectionIds = Object.keys(sectionDays).sort();
  const [sectionId, setSectionId] = useState(sectionIds[0] || "");
  const [detail, setDetail] = useState(null);
  const [detailTrain, setDetailTrain] = useState(null);
  const [trainsByDate, setTrainsByDate] = useState({});
  const [highlightedTrains, setHighlightedTrains] = useState({});

  useEffect(() => {
    if (!sectionId && sectionIds.length) setSectionId(sectionIds[0]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sectionIds.length]);

  const dates = sectionId ? (sectionDays[sectionId] || []).map((d) => d.date) : [];

  // Same real-train overlay every other schedule view has, using the
  // identical AbortController pattern as Schedule.jsx's own effect, so
  // rapid section switches can't pile up superseded requests (see that
  // file's comment for the ERR_INSUFFICIENT_RESOURCES bug this guards
  // against).
  useEffect(() => {
    if (!sectionId || dates.length === 0) return;
    const controller = new AbortController();
    Promise.all(dates.map((d) => api.sectionTrainSchedule(sectionId, d, controller.signal).catch(() => ({ trains: [] }))))
      .then((results) => {
        if (controller.signal.aborted) return;
        const out = {};
        dates.forEach((d, i) => (out[d] = results[i].trains));
        setTrainsByDate(out);
      });
    return () => controller.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sectionId, dates.join(",")]);

  function handleTrainClick(date, train) {
    setDetailTrain(train);
    const key = trainKey(date, train);
    setHighlightedTrains((prev) => {
      if (prev[key]) return prev;
      const used = new Set(Object.values(prev));
      const color = HIGHLIGHT_COLORS.find((c) => !used.has(c)) || HIGHLIGHT_COLORS[Object.keys(prev).length % HIGHLIGHT_COLORS.length];
      return { ...prev, [key]: color };
    });
  }

  const days = (sectionDays[sectionId] || []).map((d) => {
    // Same "no in-between trains" rule Schedule.jsx's Weekly/Monthly
    // view applies -- an emergency block on THIS day suppresses any
    // real train shown running through its own exact window, so the
    // preview matches what the actual schedule will show once approved.
    const emergencyBlocksToday = d.blocks.filter((b) => b.is_emergency);
    const trains = (trainsByDate[d.date] || []).filter(
      (t) => !emergencyBlocksToday.some((e) => t.start_minute < e.end_minute && t.end_minute > e.start_minute)
    );
    return { ...d, trains };
  });

  return (
    <div className="overlay" onClick={onClose}>
      <div className="modal wide" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <h3>{emergency.task_id} — proposed reschedule</h3>
          <button className="modal-close" onClick={onClose}>×</button>
        </div>
        <p className="text-small text-muted" style={{ marginTop: -8 }}>
          Where the emergency block (red) sits, and where every affected task is proposed to move to. Nothing is
          applied yet -- approve below to make it real, or close this and discard instead.
        </p>

        {sectionIds.length === 0 ? (
          <InfoBanner>Nothing to preview.</InfoBanner>
        ) : (
          <>
            <div className="matrix-toolbar">
              <select value={sectionId} onChange={(e) => setSectionId(e.target.value)}>
                {sectionIds.map((s) => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>
            </div>
            <HourGrid
              days={days}
              onBlockClick={(b) => setDetail(b)}
              onTrainClick={handleTrainClick}
              highlightedTrains={highlightedTrains}
            />
            <div className="map-legend" style={{ marginTop: 14 }}>
              <div className="legend-item"><div className="legend-swatch" style={{ background: "var(--red)" }} />Emergency situation</div>
              <div className="legend-item"><div className="legend-swatch" style={{ background: "#3B82C4" }} />Engineering</div>
              <div className="legend-item"><div className="legend-swatch" style={{ background: "#8A5DAE" }} />Signalling</div>
              <div className="legend-item"><div className="legend-swatch" style={{ background: "#C4842F" }} />Traction (TRD)</div>
              <div className="legend-item"><div className="legend-swatch" style={{ background: "var(--blue)" }} />Combined block (shared window)</div>
              <div className="legend-item"><div style={{ width: 14, height: 5, borderRadius: 2, background: "repeating-linear-gradient(45deg,#D0D5DD,#D0D5DD 2px,#fff 2px,#fff 4px)" }} />Train passing (real duration)</div>
            </div>
          </>
        )}

        <div className="modal-actions">
          <button className="btn btn-outline" onClick={onClose}>Close (don't apply yet)</button>
          <button className="btn btn-primary" disabled={resolving === emergency.task_id} onClick={onApprove}>
            {resolving === emergency.task_id ? "Applying…" : "Approve & apply this reschedule"}
          </button>
        </div>
      </div>

      {detailTrain && <TrainDetailsModal train={detailTrain} sectionId={sectionId} onClose={() => setDetailTrain(null)} />}

      {detail && (
        <BlockDetailsModal
          taskId={detail.task_id}
          relatedTaskIds={detail.combined ? detail.related : null}
          showReject={false}
          onClose={() => setDetail(null)}
        />
      )}
    </div>
  );
}
