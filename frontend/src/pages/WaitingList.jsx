import { useEffect, useMemo, useState } from "react";
import api from "../api/client.js";
import BlockDetailsModal from "../components/BlockDetailsModal.jsx";
import { ErrorBanner, InfoBanner, Spinner } from "../components/StatusBanner.jsx";

const PRIORITY_CLASS = { Critical: "crit", Moderate: "mod", Routine: "rou" };

const STATUS_LABEL = {
  pending: "Pending (not yet scheduled)",
  recommended_full: "Scheduled — awaiting approval",
  recommended_partial: "Partially scheduled",
  recommended_unscheduled: "Could not be scheduled this run",
};

// Read-only: nothing is approved here. Approval happens on Recommended
// Scheduling when the COA selects a schedule option; a task leaves this
// list automatically once that happens.
//
// Session 32, at explicit user request: this page is now the home of the
// per-visitor "Create demo batch of tasks" / "Reset" toggle -- each
// browser gets its own isolated copy of the 70 real tasks (see
// api/session.js + api/app.py's /demo/batch/* endpoints), so a publicly-
// shared hosted link no longer needs manual reloading between demo
// rounds. `totalCount` (the UNFILTERED request count, before the
// approved/rejected filter below) decides which button to show -- a
// session whose entire batch has since been fully approved must still
// show "Reset", not "Create demo batch" again.
export default function WaitingList() {
  const [requests, setRequests] = useState([]);
  const [totalCount, setTotalCount] = useState(0);
  const [whittleById, setWhittleById] = useState({});
  const [sortMode, setSortMode] = useState("raised"); // "raised" | "whittle"
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [detailTaskId, setDetailTaskId] = useState(null);
  const [activating, setActivating] = useState(false);
  const [resetting, setResetting] = useState(false);

  function load() {
    return Promise.all([api.listRequests(), api.requestsWhittleRank()])
      .then(([all, ranked]) => {
        setTotalCount(all.length);
        setRequests(all.filter((r) => r.status !== "approved" && r.status !== "rejected"));
        const byId = {};
        ranked.forEach((r, i) => (byId[r.task_id] = { risk_percentage: r.risk_percentage, rank: i }));
        setWhittleById(byId);
      })
      .catch((e) => setError(e.message));
  }

  useEffect(() => {
    load().finally(() => setLoading(false));
  }, []);

  async function handleActivate() {
    setActivating(true);
    setError(null);
    try {
      await api.activateDemoBatch();
      await load();
    } catch (e) {
      setError(e.message);
    } finally {
      setActivating(false);
    }
  }

  async function handleReset() {
    setResetting(true);
    setError(null);
    try {
      await api.resetDemoBatch();
      await load();
    } catch (e) {
      setError(e.message);
    } finally {
      setResetting(false);
    }
  }

  // Session 21, at explicit user request: two real orders --
  // "Whittle index rank order" (real Layer 2 risk score, highest first,
  // from GET /requests/whittle-rank) and "newly raised block request
  // order" (GET /requests' own most-recent-first order, unchanged).
  const sortedRequests = useMemo(() => {
    if (sortMode !== "whittle") return requests;
    return [...requests].sort((a, b) => {
      const ra = whittleById[a.task_id]?.rank ?? Infinity;
      const rb = whittleById[b.task_id]?.rank ?? Infinity;
      return ra - rb;
    });
  }, [requests, sortMode, whittleById]);

  if (loading) return <Spinner />;

  return (
    <div className="stack">
      <ErrorBanner message={error} />

      <div className="table-card-head" style={{ padding: 0 }}>
        <div className="section-title" style={{ margin: 0 }}>Waiting list — {requests.length} request(s)</div>
        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          {requests.length > 0 && (
            <select value={sortMode} onChange={(e) => setSortMode(e.target.value)}>
              <option value="raised">Newly raised block request order</option>
              <option value="whittle">Whittle index rank order</option>
            </select>
          )}
          {totalCount > 0 ? (
            <button className="btn btn-danger" onClick={handleReset} disabled={resetting}>
              {resetting ? "Resetting…" : "Reset"}
            </button>
          ) : (
            <button className="btn btn-primary" onClick={handleActivate} disabled={activating}>
              {activating ? "Loading…" : "Create demo batch of tasks"}
            </button>
          )}
        </div>
      </div>

      {requests.length === 0 ? (
        <InfoBanner>
          {totalCount > 0
            ? `All ${totalCount} task(s) in this batch have been approved or rejected -- none left waiting. Check Approved Tasks or the Weekly/Monthly Schedule to see them, or click "Reset" above to start a fresh batch.`
            : 'Waiting list is empty. Click "Create demo batch of tasks" above to load the demo dataset.'}
        </InfoBanner>
      ) : (
        <div className="card table-card">
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Task</th>
                  <th>Department</th>
                  <th>Section</th>
                  <th>Priority</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {sortedRequests.map((row) => (
                  <tr key={row.task_id} className="clickable" onClick={() => setDetailTaskId(row.task_id)}>
                    <td className="id-cell">{row.task_id}</td>
                    <td>{row.department}</td>
                    <td>{row.section_id}</td>
                    <td>
                      <span className={`tag ${PRIORITY_CLASS[row.requester_priority] || "pending"}`}>
                        <span className="tag-dot" />
                        {row.requester_priority}
                      </span>
                    </td>
                    <td>
                      <span className="tag pending">{STATUS_LABEL[row.status] || row.status}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {detailTaskId && (
        <BlockDetailsModal
          taskId={detailTaskId}
          scheduleInfoByTaskId={{ [detailTaskId]: { risk_percentage: whittleById[detailTaskId]?.risk_percentage } }}
          onClose={() => setDetailTaskId(null)}
          onChanged={(taskId) => setRequests((rs) => rs.filter((r) => r.task_id !== taskId))}
        />
      )}
    </div>
  );
}
