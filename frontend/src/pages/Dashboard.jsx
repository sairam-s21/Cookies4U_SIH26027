import { useEffect, useState } from "react";
import api from "../api/client.js";
import Card from "../components/Card.jsx";
import StatCard from "../components/StatCard.jsx";
import { ErrorBanner, InfoBanner, Spinner } from "../components/StatusBanner.jsx";

function fmtMinute(m) {
  if (m == null) return "—";
  const wrapped = ((m % 1440) + 1440) % 1440;
  const h = Math.floor(wrapped / 60);
  const min = Math.round(wrapped % 60);
  return `${String(h).padStart(2, "0")}:${String(min).padStart(2, "0")}`;
}

const DEPT_CLASS = { Engineering: "eng", Signalling: "sig", Traction: "trd" };

function BlockRow({ row }) {
  return (
    <div className="block-row">
      <span className={`block-row-dot ${DEPT_CLASS[row.department] || ""}`} />
      <div className="block-row-main">
        <div className="block-row-title">{row.task_id} — {row.section_id}</div>
        <div className="text-small text-muted">{row.department} · {row.date} · {fmtMinute(row.start_minute)}–{fmtMinute(row.end_minute)}</div>
      </div>
    </div>
  );
}

export default function Dashboard() {
  const [history, setHistory] = useState(null);
  const [pending, setPending] = useState([]);
  const [active, setActive] = useState([]);
  const [upcoming, setUpcoming] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const [historyRes, pendingRes, activeRes, upcomingRes] = await Promise.all([
          api.history(),
          api.listRequests(),
          api.activeBlocks(),
          api.upcomingBlocks(5),
        ]);
        if (cancelled) return;
        setHistory(historyRes);
        // Same definition as WaitingList.jsx uses -- everything not yet
        // approved, not just literally status="pending". A task that's
        // already been through one recommendation run (recommended_full/
        // partial/unscheduled) still belongs in this count; otherwise this
        // card and the Waiting List page disagree the moment ANY
        // recommend run has happened, which is confusing, not honest.
        setPending(pendingRes.filter((r) => r.status !== "approved"));
        setActive(activeRes.blocks);
        setUpcoming(upcomingRes.blocks);
      } catch (e) {
        if (!cancelled) setError(e.message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, []);

  if (loading) return <Spinner />;

  const criticalPending = pending.filter((r) => r.requester_priority === "Critical").length;

  return (
    <div className="stack">
      <ErrorBanner message={error} />

      <div className="grid-3">
        <StatCard label="Pending Requests" value={pending.length} />
        <StatCard label="Approved (all time)" value={history?.count ?? 0} />
        <StatCard label="Critical — pending" value={criticalPending} tone={criticalPending > 0 ? "bad" : "default"} />
      </div>

      <div className="grid-2">
        <Card title="Currently active blocks">
          {active.length === 0 ? (
            <InfoBanner>Nothing is running right now.</InfoBanner>
          ) : (
            <div className="stack" style={{ gap: 8 }}>
              {active.map((row) => (
                <BlockRow key={row.task_id} row={row} />
              ))}
            </div>
          )}
        </Card>

        <Card title="Upcoming blocks">
          {upcoming.length === 0 ? (
            <InfoBanner>Nothing upcoming.</InfoBanner>
          ) : (
            <div className="stack" style={{ gap: 8 }}>
              {upcoming.map((row) => (
                <BlockRow key={row.task_id} row={row} />
              ))}
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
