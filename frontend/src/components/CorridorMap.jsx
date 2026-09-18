// Real OpenStreetMap-based corridor map (Session 9).
//
// Session 17, at explicit user request: previously only drew maintenance
// highlighting between the 9 major stations (the only ones with real
// public lat/lon), collapsing all 56 real fine sections onto 8 oversized
// major-to-major lines. GET /corridor now returns an interpolated lat/lon
// for every one of the 57 fine stations (see
// railblock.corridor.geo.interpolate_all_station_geo), so this component
// draws one highlight line per REAL fine section, using each section's own
// from_code/to_code endpoints -- no bucketing, no oversized lines. Station
// markers are only drawn for the 9 real-geo majors (the interpolated ones
// are real block boundaries, not places worth cluttering the map with a
// dot for).
//
// The base track line is RailRadar's real route-geometry curve for train
// 12243 (GET /corridor's route_geometry) when available, else straight
// lines between the 9 majors. The colored maintenance highlight overlay
// is always a straight sub-line between its section's two endpoints
// (real for majors, honestly-interpolated for the rest) -- exact
// positioning along a curved base track isn't attempted.
//
// A train marker's position is either a real live RailRadar position
// (GET /trains/positions?live=true, source="live") or interpolated along
// its segment using its section's real progress-through-section value
// (source="computed") -- never a fabricated position either way.
import { useEffect, useRef } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

const DEPT_COLORS = { Engineering: "#3B82C4", Signalling: "#8A5DAE", Traction: "#C4842F" };
const COMBINED_COLOR = "#3457D5";
const IDLE_COLOR = "#D0D5DD";
// Session 34, at explicit user request, after a real reported gap: an
// active emergency situation's section wasn't visually distinguishable
// on the map -- it was just blended into ordinary department coloring
// (its own department got added to the section's departments set like
// any other task). Matches --red (index.css) and HourGrid.jsx's own
// distinct red "emg" treatment, and takes priority over combined/
// department coloring since an emergency is the most urgent thing
// happening on a section.
const EMERGENCY_COLOR = "#C0392B";

function trainIcon() {
  return L.divIcon({
    className: "",
    html: `<div style="width:14px;height:14px;border-radius:50%;background:#101828;border:2px solid #fff;box-shadow:0 0 0 1px #101828;"></div>`,
    iconSize: [14, 14],
    iconAnchor: [7, 7],
  });
}

export default function CorridorMap({ stations, sections, trains, onTrainClick, routeGeometry }) {
  const mapDivRef = useRef(null);
  const mapRef = useRef(null);
  const layerRef = useRef(null);
  const trainLayerRef = useRef(null);

  useEffect(() => {
    if (!mapDivRef.current || mapRef.current) return;
    const map = L.map(mapDivRef.current, { scrollWheelZoom: true }).setView([12.9, 79.2], 8);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      maxZoom: 18,
    }).addTo(map);
    mapRef.current = map;
    layerRef.current = L.layerGroup().addTo(map);
    trainLayerRef.current = L.layerGroup().addTo(map);
    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!mapRef.current || !layerRef.current) return;
    layerRef.current.clearLayers();

    const byCode = {};
    stations.forEach((st) => (byCode[st.station_code] = st));

    stations
      .filter((st) => st.geo_source === "real")
      .forEach((st) => {
        L.circleMarker([st.lat, st.lon], { radius: 4, color: "#475467", fillColor: "#667085", fillOpacity: 1, weight: 1 })
          .bindTooltip(`${st.station_name} (${st.station_code})`, { direction: "top" })
          .addTo(layerRef.current);
      });

    const hasRealGeometry = Array.isArray(routeGeometry) && routeGeometry.length > 1;
    if (hasRealGeometry) {
      L.polyline(routeGeometry, { color: IDLE_COLOR, weight: 3, opacity: 0.8 })
        .bindTooltip("Real track geometry (RailRadar, train 12243's route)")
        .addTo(layerRef.current);
    }

    sections.forEach((sec) => {
      const from = byCode[sec.from_code];
      const to = byCode[sec.to_code];
      if (!from || !to) return;
      // With real geometry drawn as the base track, only draw the colored
      // highlight overlay where there's actually something to highlight --
      // an "idle" straight line would just duplicate the real curve.
      if (hasRealGeometry && !sec.department && !sec.combined && !sec.is_emergency) return;
      const highlighted = sec.department || sec.combined || sec.is_emergency;
      const color = sec.is_emergency ? EMERGENCY_COLOR : sec.combined ? COMBINED_COLOR : sec.department ? DEPT_COLORS[sec.department] : IDLE_COLOR;
      L.polyline(
        [[from.lat, from.lon], [to.lat, to.lon]],
        { color, weight: highlighted ? 5 : 2, opacity: highlighted ? 0.95 : 0.5 }
      )
        .bindTooltip(
          sec.is_emergency
            ? `${sec.section_id} — EMERGENCY SITUATION`
            : `${sec.section_id}${sec.department ? ` — ${sec.combined ? "combined block" : sec.department}` : " — no maintenance"}`
        )
        .addTo(layerRef.current);
    });
  }, [stations, sections, routeGeometry]);

  useEffect(() => {
    if (!mapRef.current || !trainLayerRef.current) return;
    trainLayerRef.current.clearLayers();
    trains.forEach((t) => {
      const marker = L.marker([t.lat, t.lon], { icon: trainIcon() })
        .bindTooltip(`${t.train_name || t.train_no}`, { direction: "top" })
        .on("click", () => onTrainClick?.(t));
      marker.addTo(trainLayerRef.current);
    });
  }, [trains, onTrainClick]);

  return <div ref={mapDivRef} className="leaflet-map-wrap" style={{ height: 480, width: "100%" }} />;
}
