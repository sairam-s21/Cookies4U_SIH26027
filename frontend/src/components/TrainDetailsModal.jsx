function fmtMinute(m) {
  if (m == null) return "—";
  const wrapped = ((m % 1440) + 1440) % 1440;
  const h = Math.floor(wrapped / 60);
  const min = Math.round(wrapped % 60);
  return `${String(h).padStart(2, "0")}:${String(min).padStart(2, "0")}`;
}

export default function TrainDetailsModal({ train, sectionId, date, onClose }) {
  return (
    <div className="overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <h3>{train.train_name || "Train"} ({train.train_no})</h3>
          <button className="modal-close" onClick={onClose}>×</button>
        </div>
        <div className="modal-row"><span className="k">Train number</span><span className="v mono">{train.train_no}</span></div>
        <div className="modal-row"><span className="k">Train name</span><span className="v">{train.train_name || "—"}</span></div>
        {sectionId && <div className="modal-row"><span className="k">Section</span><span className="v">{sectionId}</span></div>}
        {date && <div className="modal-row"><span className="k">Date</span><span className="v">{date}</span></div>}
        <div className="modal-row"><span className="k">Transit window (this section)</span><span className="v">{fmtMinute(train.start_minute)} – {fmtMinute(train.end_minute)}</span></div>
        {train.source === "live" && (
          <>
            <div className="modal-row"><span className="k">Live status</span><span className="v">{train.live_status || "—"}</span></div>
            <div className="modal-row"><span className="k">Delay</span><span className="v">{train.delay_minutes != null ? `${train.delay_minutes} min` : "—"}</span></div>
            <div className="modal-row"><span className="k">Nearest station</span><span className="v">{train.current_halt || "—"}</span></div>
            <div className="modal-row"><span className="k">Distance from origin</span><span className="v">{train.distance_from_origin_km != null ? `${train.distance_from_origin_km} km` : "—"}</span></div>
          </>
        )}
        <div className="modal-actions">
          <button className="btn btn-outline" onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  );
}
