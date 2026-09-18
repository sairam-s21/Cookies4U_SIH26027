export function ErrorBanner({ message }) {
  if (!message) return null;
  return <div className="banner banner-error">{message}</div>;
}

export function InfoBanner({ children, tone = "slate" }) {
  const toneClass = tone === "amber" ? "banner-amber" : tone === "emerald" ? "banner-emerald" : "banner-info";
  return <div className={`banner ${toneClass}`}>{children}</div>;
}

export function Spinner() {
  return (
    <div className="spinner-row">
      <span className="spinner-dot" />
      Loading…
    </div>
  );
}
