import { useEffect, useState } from "react";
import api from "../api/client.js";
import GrantedBlockDetailsModal from "../components/GrantedBlockDetailsModal.jsx";
import { ErrorBanner, InfoBanner, Spinner } from "../components/StatusBanner.jsx";

const PRIORITY_CLASS = { Critical: "crit", Moderate: "mod", Routine: "rou" };

// Session 17: granted blocks whose real (date, start_minute, end_minute)
// window has already ended, relative to real current time -- see
// GET /tasks/history/completed and railblock.synthetic.granted_history.
export default function CompletedHistory() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [detail, setDetail] = useState(null);

  useEffect(() => {
    api
      .completedHistory()
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="stack">
      <ErrorBanner message={error} />
      {loading ? (
        <Spinner />
      ) : !data || data.count === 0 ? (
        <InfoBanner>No completed blocks yet.</InfoBanner>
      ) : (
        <div className="card table-card">
          <div className="table-card-head">
            <div className="section-title" style={{ margin: 0 }}>Completed history — {data.count}</div>
          </div>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Task</th>
                  <th>Department</th>
                  <th>Section</th>
                  <th>Date</th>
                  <th>Priority</th>
                  <th>Approval path</th>
                  <th>On time</th>
                </tr>
              </thead>
              <tbody>
                {data.tasks.map((row) => (
                  <tr key={row.task_id} className="clickable" onClick={() => setDetail(row)}>
                    <td className="id-cell">{row.task_id}</td>
                    <td>{row.department}</td>
                    <td>{row.section_id}</td>
                    <td>{row.date}</td>
                    <td>
                      <span className={`tag ${PRIORITY_CLASS[row.requester_priority] || "pending"}`}>
                        <span className="tag-dot" />
                        {row.requester_priority}
                      </span>
                    </td>
                    <td>{row.approval_path}</td>
                    <td>{String(row.on_time)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {detail && <GrantedBlockDetailsModal row={detail} onClose={() => setDetail(null)} />}
    </div>
  );
}
