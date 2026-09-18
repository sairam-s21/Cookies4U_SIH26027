import { useEffect, useMemo, useState } from "react";
import api from "../api/client.js";
import Card from "../components/Card.jsx";
import CorridorMap from "../components/CorridorMap.jsx";
import TrainDetailsModal from "../components/TrainDetailsModal.jsx";
import { ErrorBanner, InfoBanner, Spinner } from "../components/StatusBanner.jsx";

const LIVE_FOLLOW_TICK_MS = 30000;

// Session 17: real LOCAL date, not toISOString()'s UTC date -- the map
// previously showed "yesterday" for roughly the first 5.5 hours of every
// IST day (00:00-05:30 IST is still the previous day in UTC), a real bug,
// not a display quirk.
function isoLocalDateOf(d) {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}
function minuteOfDay(d) {
  return d.getHours() * 60 + d.getMinutes();
}

export default function CorridorMapPage() {
  const [corridor, setCorridor] = useState(null);
  const [weekly, setWeekly] = useState(null);
  const [grantedHistory, setGrantedHistory] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  // Session 29, at explicit user request, after a real reported bug:
  // these used to read the VIEWER's own machine clock directly --
  // confirmed live to genuinely disagree with the server's real clock.
  // `clockOffsetMs` (server-real-now minus this browser's own Date.now(),
  // captured once via GET /now on mount) corrects every subsequent
  // local clock read without needing to re-poll the server every tick --
  // `correctedNow()` is this browser's own fast clock, just shifted by
  // that one real, measured offset.
  const [clockOffsetMs, setClockOffsetMs] = useState(0);
  const correctedNow = () => new Date(Date.now() + clockOffsetMs);
  const [selectedDate, setSelectedDate] = useState(isoLocalDateOf(correctedNow()));
  const [minute, setMinute] = useState(minuteOfDay(correctedNow()));
  const [trains, setTrains] = useState([]);
  const [trainDetail, setTrainDetail] = useState(null);
  // Live-follow is the default, no manual toggle: the slider tracks real
  // "now" automatically until the user drags it or picks another date, at
  // which point it's frozen for free past/future browsing until they
  // click "Reset to live time".
  const [liveFollow, setLiveFollow] = useState(true);

  useEffect(() => {
    api.now().then((n) => {
      setClockOffsetMs(new Date(n.datetime).getTime() - Date.now());
    }).catch(() => {});
  }, []);

  useEffect(() => {
    if (!liveFollow) return;
    const tick = () => {
      const now = correctedNow();
      setSelectedDate(isoLocalDateOf(now));
      setMinute(minuteOfDay(now));
    };
    tick();
    const id = setInterval(tick, LIVE_FOLLOW_TICK_MS);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [liveFollow, clockOffsetMs]);

  useEffect(() => {
    let cancelled = false;
    Promise.all([api.corridor(), api.weeklySchedule(), api.history(), api.completedHistory()])
      .then(([c, w, hist, completedHist]) => {
        if (cancelled) return;
        setCorridor(c);
        setWeekly(w);
        // Session 18, at explicit user request: the map previously only
        // ever drew highlighting from GET /schedule/weekly (live-approved
        // tasks only), so a currently-active block from the fixed
        // granted-history dataset (Dashboard's "Currently Active Blocks")
        // never showed up here even while it was genuinely happening.
        // Both endpoints below can include live-approved rows too (see
        // their own docstrings) -- keep only the granted-history ones
        // (they never carry completion_verified, unlike a live row) so a
        // live-approved task isn't double-drawn (it's already in `weekly`).
        const isGrantedHistory = (row) => !("completion_verified" in row);
        setGrantedHistory([
          ...hist.tasks.filter(isGrantedHistory),
          ...completedHist.tasks.filter(isGrantedHistory),
        ]);
      })
      .catch((e) => !cancelled && setError(e.message))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, []);

  const stations = useMemo(() => corridor?.stations || [], [corridor]);
  const orderedStations = useMemo(
    () => [...stations].sort((a, b) => a.distance_km - b.distance_km),
    [stations]
  );
  const sectionsById = useMemo(() => {
    const out = {};
    (corridor?.sections || []).forEach((s) => (out[s.section_id] = s));
    return out;
  }, [corridor]);

  // One real fine section per row -- no bucketing onto major-station
  // stretches. Every one of the 56 real sections is represented, whether
  // or not it has maintenance right now.
  //
  // Session 18, at explicit user request, after a real bug: this
  // previously highlighted a section for its ENTIRE day the moment any
  // block existed on that date, ignoring the Time slider entirely --
  // once granted-history rows were wired in (each with its own real
  // hour-of-day window, unlike the old weekly-only view which mostly
  // showed only what was picked to be "today"'s single recommendation
  // run), this surfaced as blocks that had already ended or hadn't
  // started yet still showing as highlighted "right now." Both sources
  // are now filtered by whether the selected minute actually falls
  // inside the block's real (start_minute, end_minute) window -- a
  // negotiated-exception entry with no modeled time (start/end both
  // null) is the one exception, shown for the whole day since there's
  // no real time to filter it by.
  const daySegments = useMemo(() => {
    const sections = (corridor?.sections || []).map((s) => ({ ...s, department: null, combined: false, is_emergency: false, departments: new Set() }));
    if (!selectedDate) return sections;
    const bySectionId = {};
    sections.forEach((s) => (bySectionId[s.section_id] = s));

    const isNowInWindow = (startMinute, endMinute) =>
      startMinute == null || endMinute == null || (minute >= startMinute && minute <= endMinute);

    if (weekly) {
      for (const sec of sections) {
        const entries = weekly.cells[sec.section_id]?.[selectedDate] || [];
        entries.filter((e) => isNowInWindow(e.start_minute, e.end_minute)).forEach((e) => {
          sec.departments.add(e.department);
          // Session 34, at explicit user request, after a real reported
          // gap: an emergency's own block was already in `weekly.cells`
          // (GET /schedule/weekly already includes it, see
          // api/app.py's weekly_matrix/_emergency_block_rows), but it
          // was only ever treated as an ordinary department block here
          // -- never flagged so the map could give it its own distinct
          // red highlight (CorridorMap.jsx) instead of blending into
          // whatever department color it happens to carry.
          if (e.is_emergency) sec.is_emergency = true;
        });
      }
    }
    for (const row of grantedHistory) {
      if (row.date !== selectedDate) continue;
      if (!isNowInWindow(row.start_minute, row.end_minute)) continue;
      const sec = bySectionId[row.section_id];
      if (sec) sec.departments.add(row.department);
    }

    sections.forEach((s) => {
      s.combined = s.departments.size > 1;
      s.department = s.departments.size === 1 ? [...s.departments][0] : null;
    });
    return sections;
  }, [corridor, weekly, grantedHistory, selectedDate, minute]);

  useEffect(() => {
    if (!selectedDate) return;
    // Real live RailRadar positions only make sense exactly when we're
    // showing real "now" (liveFollow); a past/future date+time can only
    // ever use the schedule-computed estimate, live data doesn't exist
    // for a moment that hasn't happened yet or is already over.
    api
      .trainPositions(selectedDate, minute, liveFollow)
      .then((r) => setTrains(r.trains))
      .catch(() => setTrains([]));
  }, [selectedDate, minute, liveFollow]);

  const trainMarkers = useMemo(() => {
    function latLonAtKm(km) {
      if (orderedStations.length < 2) return null;
      let i = orderedStations.findIndex((s) => s.distance_km >= km);
      if (i <= 0) i = 1;
      if (i >= orderedStations.length) i = orderedStations.length - 1;
      const a = orderedStations[i - 1], b = orderedStations[i];
      const span = b.distance_km - a.distance_km;
      const frac = span <= 0 ? 0 : (km - a.distance_km) / span;
      return { lat: a.lat + (b.lat - a.lat) * frac, lon: a.lon + (b.lon - a.lon) * frac };
    }

    return trains
      .map((t) => {
        // A "live" entry gives a real distance-from-origin km (RailRadar
        // has no direct lat/lon -- see railradar.py's get_live_status
        // docstring) -- place it using that REAL km value along the real
        // station geography. Otherwise (schedule-computed, or a live
        // lookup with no data for this train) fall back to the section
        // midpoint km.
        const sec = sectionsById[t.section_id];
        const km = t.source === "live" && t.distance_from_origin_km != null
          ? t.distance_from_origin_km
          : sec ? (sec.from_km + sec.to_km) / 2 : null;
        if (km == null || Number.isNaN(km)) return null;
        const pos = latLonAtKm(km);
        if (!pos) return null;
        return { ...t, lat: pos.lat, lon: pos.lon };
      })
      .filter(Boolean);
  }, [trains, orderedStations, sectionsById]);

  if (loading) return <Spinner />;

  return (
    <div className="stack">
      <ErrorBanner message={error} />
      <Card title="Corridor map — maintenance blocks by time">
        <div className="map-controls">
          <div className="grp">
            Date:
            <input
              type="date"
              value={selectedDate || ""}
              onChange={(e) => {
                setLiveFollow(false);
                setSelectedDate(e.target.value);
              }}
            />
          </div>
          {weekly?.dates?.length > 0 && (
            <div className="grp">
              Scheduled:
              <div className="day-pills">
                {weekly.dates.map((d) => (
                  <div
                    key={d}
                    className={`day-pill${selectedDate === d ? " active" : ""}`}
                    onClick={() => {
                      setLiveFollow(false);
                      setSelectedDate(d);
                    }}
                  >
                    {d?.slice(5)}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
        <div className="map-time-row">
          Time:
          <input
            type="range"
            min={0}
            max={1439}
            step={5}
            value={minute}
            onChange={(e) => {
              setLiveFollow(false);
              setMinute(parseInt(e.target.value, 10));
            }}
          />
          <span className="map-time-label">{String(Math.floor(minute / 60)).padStart(2, "0")}:{String(minute % 60).padStart(2, "0")}</span>
          {liveFollow ? (
            <span className="text-small" style={{ marginLeft: 16, color: "var(--red)", fontWeight: 700 }}>● LIVE</span>
          ) : (
            <button className="btn btn-outline btn-sm" style={{ marginLeft: 16 }} onClick={() => setLiveFollow(true)}>
              Reset to live time
            </button>
          )}
        </div>
        {stations.length > 0 ? (
          <CorridorMap
            stations={stations}
            sections={daySegments}
            trains={trainMarkers}
            onTrainClick={setTrainDetail}
            routeGeometry={corridor?.route_geometry}
          />
        ) : (
          <InfoBanner>No station geo data available.</InfoBanner>
        )}
        <div className="map-legend">
          <div className="legend-item"><div className="legend-swatch" style={{ background: "#3B82C4" }} />Engineering</div>
          <div className="legend-item"><div className="legend-swatch" style={{ background: "#8A5DAE" }} />Signalling</div>
          <div className="legend-item"><div className="legend-swatch" style={{ background: "#C4842F" }} />Traction (TRD)</div>
          <div className="legend-item"><div className="legend-swatch" style={{ background: "#3457D5" }} />Combined block</div>
          <div className="legend-item"><div className="legend-swatch" style={{ background: "#C0392B" }} />Emergency situation</div>
          <div className="legend-item"><div className="legend-swatch" style={{ background: "#D0D5DD" }} />No maintenance</div>
        </div>
      </Card>

      {trainDetail && <TrainDetailsModal train={trainDetail} sectionId={trainDetail.section_id} date={selectedDate} onClose={() => setTrainDetail(null)} />}
    </div>
  );
}
