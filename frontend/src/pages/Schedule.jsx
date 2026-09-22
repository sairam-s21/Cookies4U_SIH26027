import { useEffect, useMemo, useState } from "react";
import api from "../api/client.js";
import Card from "../components/Card.jsx";
import HourGrid from "../components/HourGrid.jsx";
import { trainKey } from "../utils/trainKey.js";
import BlockDetailsModal from "../components/BlockDetailsModal.jsx";
import TrainDetailsModal from "../components/TrainDetailsModal.jsx";
import { ErrorBanner, InfoBanner, Spinner } from "../components/StatusBanner.jsx";

// Each train block a viewer has inspected gets its own distinct color
// from this palette (deliberately none of the department/negotiated-
// exception colors used elsewhere on this page), assigned in order as
// new blocks are clicked. Resets to the default grey hatch on every page
// reload -- this is plain in-memory React state, never persisted.
const HIGHLIGHT_COLORS = [
  "#E4572E", "#17BEBB", "#FFC914", "#2E86AB", "#A23B72",
  "#8E44AD", "#27AE60", "#D81159", "#F18F01", "#0B7A75",
];

// Formats a Date using its LOCAL getters (getFullYear/getMonth/getDate),
// not `.toISOString().slice(0, 10)`, which reads UTC. Every Date object
// built above (`new Date(anchorDate + "T00:00:00")`, `new Date(year,
// month, i + 1)`) is constructed in local time, so converting it back to
// a string via UTC silently shifts the date backward by one whenever the
// viewer's timezone is ahead of UTC (IST is +5:30, this app's timezone) --
// and since Next/Previous each re-parse an already-shifted string, the
// drift compounds with every click. Same pattern CorridorMapPage.jsx's
// isoLocalDateOf() uses for the identical reason.
function isoDate(d) {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

// Weekly = 7 consecutive calendar days starting from `anchorDate`.
// Monthly = every day in the calendar month containing that same start
// date. Both always render every day, whether or not it has a
// maintenance block that day -- an empty row is real information
// (nothing scheduled), not something to hide.
function buildDateRange(mode, anchorDate) {
  // `anchorDate` starts out null (set only once GET /now resolves, see
  // SectionScheduleView) -- this useMemo still runs on the very first
  // render, before the `loading` gate below has a chance to hide
  // anything. Without this guard, `new Date(null + "T00:00:00")` is an
  // Invalid Date -- every getter on it (getFullYear/getMonth/getDate)
  // returns NaN, silently producing garbage "NaN-NaN-NaN" date strings
  // instead of a real range, with no error boundary to catch it.
  if (!anchorDate) return [];
  const start = new Date(anchorDate + "T00:00:00");
  if (mode === "weekly") {
    return Array.from({ length: 7 }, (_, i) => {
      const d = new Date(start);
      d.setDate(d.getDate() + i);
      return isoDate(d);
    });
  }
  const year = start.getFullYear();
  const month = start.getMonth();
  const daysInMonth = new Date(year, month + 1, 0).getDate();
  return Array.from({ length: daysInMonth }, (_, i) => isoDate(new Date(year, month, i + 1)));
}

// GET /schedule/weekly returns the fixed granted-history dataset (spans
// ~one month before today through the SIH evaluation window, see
// railblock.synthetic.granted_history) alongside live-approved rows, so
// anchoring to the earliest date across the whole dataset would pin both
// tabs to that dataset's very first week/month with no way to reach
// today or any later date. `anchorDate` is user-movable state,
// defaulting to today, with Previous/Next/Today controls below to move
// it.
function shiftAnchor(mode, anchorDate, direction) {
  const d = new Date(anchorDate + "T00:00:00");
  if (mode === "weekly") {
    d.setDate(d.getDate() + direction * 7);
  } else {
    d.setMonth(d.getMonth() + direction, 1); // pin to the 1st, avoids month-length rollover bugs
  }
  return isoDate(d);
}

function rangeLabel(mode, dates) {
  if (!dates.length) return "";
  if (mode === "monthly") {
    const d = new Date(dates[0] + "T00:00:00");
    return d.toLocaleDateString("en-IN", { month: "long", year: "numeric" });
  }
  const fmt = (iso) => new Date(iso + "T00:00:00").toLocaleDateString("en-IN", { day: "numeric", month: "short", year: "numeric" });
  return `${fmt(dates[0])} – ${fmt(dates[dates.length - 1])}`;
}

export default function Schedule() {
  const [tab, setTab] = useState("weekly");

  return (
    <div className="stack">
      <div className="horizon-toggle">
        <button className={tab === "weekly" ? "active" : ""} onClick={() => setTab("weekly")}>Weekly</button>
        <button className={tab === "monthly" ? "active" : ""} onClick={() => setTab("monthly")}>Monthly</button>
      </div>

      <SectionScheduleView mode={tab} />
    </div>
  );
}

function SectionScheduleView({ mode }) {
  const [weekly, setWeekly] = useState(null);
  const [corridor, setCorridor] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [sectionId, setSectionId] = useState("");
  const [trainsByDate, setTrainsByDate] = useState({});
  const [detailBlock, setDetailBlock] = useState(null);
  const [detailTrain, setDetailTrain] = useState(null);
  const [highlightedTrains, setHighlightedTrains] = useState({});
  // Defaulting this to the viewer's own machine clock is unreliable --
  // it can disagree with the server's clock. null until GET /now
  // resolves below; `loading` gates every render that would actually
  // use it, so there's no visible flash of a wrong date.
  const [anchorDate, setAnchorDate] = useState(null);
  const [serverToday, setServerToday] = useState(null);

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

  useEffect(() => {
    Promise.all([api.weeklySchedule(), api.corridor(), api.now()])
      .then(([w, c, n]) => {
        setWeekly(w);
        setCorridor(c);
        setServerToday(n.date);
        setAnchorDate(n.date);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  // Strict corridor sequence order (MAS-BBQ ... KDY-JTJ), not "affected
  // sections first" -- GET /corridor already returns sections in that
  // order (sorted by distance_km via sequence_order), so this just uses
  // it as-is.
  const sectionOptions = corridor?.sections?.map((s) => s.section_id) || [];

  useEffect(() => {
    if (!sectionId && sectionOptions.length) setSectionId(sectionOptions[0]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sectionOptions.length]);

  const blocksByDate = useMemo(() => {
    if (!weekly || !sectionId) return {};
    return weekly.cells[sectionId] || {};
  }, [weekly, sectionId]);

  const dates = useMemo(() => buildDateRange(mode, anchorDate), [mode, anchorDate]);

  useEffect(() => {
    if (!sectionId || dates.length === 0) return;
    // A plain `cancelled` flag would only suppress the STATE UPDATE from
    // a stale effect run -- it wouldn't stop the underlying fetch()es.
    // Clicking Next/Previous repeatedly (Monthly mode fires up to 31
    // requests per click, one per day in view) would pile up every
    // previous click's still-in-flight batch on top of the new one,
    // until Chromium ran out of connections to the origin
    // (ERR_INSUFFICIENT_RESOURCES) and every request on the page -- not
    // just these -- started failing, making the whole UI look frozen.
    // AbortController actually cancels the superseded batch's network
    // requests.
    const controller = new AbortController();
    Promise.all(dates.map((d) => api.sectionTrainSchedule(sectionId, d, controller.signal).catch(() => ({ trains: [] }))))
      .then((results) => {
        if (controller.signal.aborted) return;
        const out = {};
        dates.forEach((d, i) => (out[d] = results[i].trains));
        setTrainsByDate(out);
      });
    return () => {
      controller.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sectionId, dates.join(",")]);

  if (loading) return <Spinner />;
  if (error) return <ErrorBanner message={error} />;
  if (!weekly || sectionOptions.length === 0)
    return <InfoBanner>No corridor data available.</InfoBanner>;

  // GET /schedule/weekly's entries carry the window_index each task was
  // actually placed in -- group by it (per date, already segmented by
  // section above) and treat >=2 distinct departments sharing the same
  // window as combined. Hardcoding combined:false here would render two
  // departments sharing the exact same window as two adjacent single-
  // department blocks instead of one shared "combined" block, and
  // clicking either one would never surface the other task sharing that
  // window.
  const days = dates.map((date) => {
    const entries = blocksByDate[date] || [];
    const byWindow = {};
    entries.forEach((e) => {
      if (e.window_index == null) return;
      (byWindow[e.window_index] ??= []).push(e);
    });
    return {
      date,
      label: date,
      blocks: entries.map((e) => {
        const sameWindow = e.window_index != null ? byWindow[e.window_index] : [e];
        const combined = sameWindow.length > 1 && new Set(sameWindow.map((x) => x.department)).size > 1;
        return {
          task_id: e.task_id,
          department: e.department,
          negotiated_exception: e.negotiated_exception,
          // `e.is_emergency` (from GET /schedule/weekly) must be carried
          // through explicitly -- dropping it here would render an
          // emergency block with its ordinary department color instead
          // of the distinct red "emg" style HourGrid.jsx has for it.
          is_emergency: e.is_emergency,
          combined,
          related: combined ? sameWindow.map((x) => x.task_id) : [e.task_id],
          start_minute: e.start_minute,
          end_minute: e.end_minute,
          estimated_block_hours: e.estimated_block_hours,
          demanded_block_hours: e.demanded_block_hours,
          adaptive_allocation: e.adaptive_allocation,
        };
      }),
      // A train's own schedule is fixed data this system never
      // fabricates away, but showing it calmly passing through the same
      // window as an active emergency reads as a contradiction, so it's
      // suppressed from this display specifically for that window (the
      // block itself, and every other non-emergency block, is
      // unaffected).
      trains: (trainsByDate[date] || []).filter((t) => {
        const emergencyBlocksToday = (entries || []).filter((e) => e.is_emergency);
        return !emergencyBlocksToday.some(
          (e) => t.start_minute < e.end_minute && t.end_minute > e.start_minute
        );
      }),
    };
  });

  return (
    <Card
      title={`${mode === "weekly" ? "Weekly" : "Monthly"} schedule — ${sectionId}`}
    >
      <div className="matrix-toolbar" style={{ justifyContent: "space-between" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <button className="btn btn-outline btn-sm" onClick={() => setAnchorDate((d) => shiftAnchor(mode, d, -1))}>
            ← Previous
          </button>
          <span className="text-small text-muted" style={{ minWidth: 190, textAlign: "center" }}>
            {rangeLabel(mode, dates)}
          </span>
          <button className="btn btn-outline btn-sm" onClick={() => setAnchorDate((d) => shiftAnchor(mode, d, 1))}>
            Next →
          </button>
          <button className="btn btn-outline btn-sm" onClick={() => setAnchorDate(serverToday)}>
            Today
          </button>
        </div>
        <select value={sectionId} onChange={(e) => setSectionId(e.target.value)}>
          {sectionOptions.map((s) => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>
      </div>

      <HourGrid
        days={days}
        onBlockClick={(b) => setDetailBlock(b)}
        onTrainClick={handleTrainClick}
        highlightedTrains={highlightedTrains}
      />

      <div className="map-legend" style={{ marginTop: 14 }}>
        <div className="legend-item"><div className="legend-swatch" style={{ background: "#3B82C4" }} />Engineering</div>
        <div className="legend-item"><div className="legend-swatch" style={{ background: "#8A5DAE" }} />Signalling</div>
        <div className="legend-item"><div className="legend-swatch" style={{ background: "#C4842F" }} />Traction (TRD)</div>
        <div className="legend-item"><div className="legend-swatch" style={{ background: "var(--blue)" }} />Combined block (shared window)</div>
        <div className="legend-item"><div style={{ width: 14, height: 5, borderRadius: 2, background: "repeating-linear-gradient(45deg,#D0D5DD,#D0D5DD 2px,#fff 2px,#fff 4px)" }} />Train passing (real duration)</div>
      </div>

      {detailBlock && (
        <BlockDetailsModal
          taskId={detailBlock.task_id}
          relatedTaskIds={detailBlock.combined ? detailBlock.related : null}
          scheduleInfoByTaskId={{
            [detailBlock.task_id]: {
              estimated_block_hours: detailBlock.estimated_block_hours,
              demanded_block_hours: detailBlock.demanded_block_hours,
              adaptive_allocation: detailBlock.adaptive_allocation,
            },
          }}
          showReject={false}
          onClose={() => setDetailBlock(null)}
        />
      )}
      {detailTrain && <TrainDetailsModal train={detailTrain} sectionId={sectionId} onClose={() => setDetailTrain(null)} />}
    </Card>
  );
}
