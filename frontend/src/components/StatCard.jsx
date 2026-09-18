export default function StatCard({ label, value, sublabel, tone = "default" }) {
  const valueClass = tone === "bad" ? "value red" : tone === "good" ? "value green" : "value";
  return (
    <div className="card stat-card">
      <div className="label">{label}</div>
      <div className={valueClass}>{value}</div>
      {sublabel ? <div className="delta">{sublabel}</div> : null}
    </div>
  );
}
