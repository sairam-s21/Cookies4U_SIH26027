export default function BreakdownBars({ rows, labelKey }) {
  return (
    <div>
      {rows.map((row) => {
        const total = row.total || 0;
        return (
          <div className="breakdown-row" key={row[labelKey]}>
            <div className="breakdown-head">
              <span className="n">{row[labelKey]}</span>
              <span className="v">
                {row.on_time_pct}% on-time · {total} task{total === 1 ? "" : "s"}
              </span>
            </div>
            <div className="breakdown-track">
              {total === 0 ? null : (
                <>
                  <div className="breakdown-seg-ontime" style={{ width: `${row.on_time_pct}%` }} title={`On time: ${row.on_time} (${row.on_time_pct}%)`} />
                  <div className="breakdown-seg-late" style={{ width: `${row.late_pct}%` }} title={`Late: ${row.late} (${row.late_pct}%)`} />
                  <div className="breakdown-seg-unscheduled" style={{ width: `${row.unscheduled_pct}%` }} title={`Unscheduled: ${row.unscheduled} (${row.unscheduled_pct}%)`} />
                </>
              )}
            </div>
          </div>
        );
      })}
      <div className="map-legend" style={{ marginTop: 4 }}>
        <Legend color="var(--green)" label="On time" />
        <Legend color="var(--amber)" label="Late" />
        <Legend color="var(--ink-faint)" label="Unscheduled" />
      </div>
    </div>
  );
}

function Legend({ color, label }) {
  return (
    <span className="legend-item">
      <span className="legend-swatch" style={{ background: color }} />
      {label}
    </span>
  );
}
