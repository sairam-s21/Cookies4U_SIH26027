export default function Card({ title, subtitle, actions, children, className = "", style }) {
  return (
    <section className={`card ${className}`} style={style}>
      {(title || actions) && (
        <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 16, marginBottom: 14, flexWrap: "wrap" }}>
          <div>
            {title && <div className="section-title" style={{ margin: 0 }}>{title}</div>}
            {subtitle && <p className="text-small text-muted" style={{ margin: "4px 0 0 0" }}>{subtitle}</p>}
          </div>
          {actions}
        </div>
      )}
      {children}
    </section>
  );
}
