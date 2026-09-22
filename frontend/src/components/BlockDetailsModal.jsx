import { useEffect, useState } from "react";
import api from "../api/client.js";
import { ErrorBanner, Spinner } from "./StatusBanner.jsx";

// task_id -> whichever schedule row(s) reference it, so combined-window
// details (multiple tasks sharing one window) can be shown together.
// `onChanged(taskId)` is optional -- called after a successful reject so
// the caller (e.g. WaitingList.jsx) can remove it from its own list
// immediately, without a full re-fetch.
// `scheduleInfoByTaskId` (optional): { [task_id]: { estimated_block_hours,
// demanded_block_hours, adaptive_allocation, risk_percentage } } -- the
// REAL allocated duration from the schedule/timetable itself, not the
// original request record (api.getRequest only ever has the original
// demand), plus the task's risk percentage -- a [0,100] percentile rank
// of its Whittle-index score against every other task currently in the
// waiting list (see railblock.prioritization.whittle.rank_tasks), shown
// as "Risk percentage" when supplied. Not the raw Whittle-index itself,
// which has no natural upper bound and so isn't meaningful shown on its
// own.
// api.getRequest never has this either since it's not stored on the
// request row. Whichever fields a caller has are shown; whichever it
// doesn't just don't render.
// `showReject` (default true): the reject action only makes sense for a
// still-pending waiting-list request -- callers viewing an already-
// generated/approved schedule (Recommended Scheduling's "view full
// schedule", Weekly/Monthly Schedule) pass false.
function fmtWindowPoint(point) {
  if (!point) return "—";
  const h = Math.floor(point.minute / 60);
  const m = Math.round(point.minute % 60);
  return `${point.date} ${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
}

export default function BlockDetailsModal({ taskId, relatedTaskIds, scheduleInfoByTaskId, showReject = true, onClose, onChanged }) {
  const [rows, setRows] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [rejecting, setRejecting] = useState(false);
  const [rejectError, setRejectError] = useState(null);

  const ids = relatedTaskIds && relatedTaskIds.length ? relatedTaskIds : [taskId];

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    Promise.all(ids.map((id) => api.getRequest(id).catch(() => null)))
      .then((rowResults) => {
        if (cancelled) return;
        setRows(rowResults.filter(Boolean));
      })
      .catch((e) => !cancelled && setError(e.message))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [taskId]);

  return (
    <div className="overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <h3>{ids.length > 1 ? `Combined block — ${ids.length} tasks` : taskId}</h3>
          <button className="modal-close" onClick={onClose}>×</button>
        </div>

        {loading ? (
          <Spinner />
        ) : error ? (
          <ErrorBanner message={error} />
        ) : !rows || rows.length === 0 ? (
          <p className="text-small text-muted">No details found for this task.</p>
        ) : (
          rows.map((row) => {
            const scheduleInfo = scheduleInfoByTaskId?.[row.task_id];
            return (
            <div key={row.task_id} style={{ marginBottom: rows.length > 1 ? 18 : 0 }}>
              {rows.length > 1 && <div className="section-title" style={{ fontSize: 13, marginBottom: 8 }}>{row.task_id}</div>}
              <Row k="Section" v={row.section_id} />
              <Row k="Department" v={row.department} />
              <Row k="Defect / work type" v={row.defect_type} />
              {row.is_emergency ? (
                // Requester priority/Raised date/Due date/Approval path
                // are all inapplicable to an emergency (it skips the
                // request pipeline entirely), so its own relevant fields
                // are shown here instead of blank rows for those.
                <>
                  <Row k="Started" v={fmtWindowPoint(row.window_start)} />
                  <Row k="Ends" v={fmtWindowPoint(row.window_end)} />
                  <Row k="Duration" v={`${row.estimated_block_hours}h`} />
                  <Row k="Already-scheduled blocks displaced" v={row.affected_count} />
                  {row.affected_task_ids?.length > 0 && (
                    <Row k="Affected task IDs" v={row.affected_task_ids.join(", ")} />
                  )}
                  <Row k="Status" v={row.status} />
                </>
              ) : (
                <>
                  <Row k="Requester priority" v={row.requester_priority} />
                  {scheduleInfo?.risk_percentage != null && (
                    <Row k="Risk percentage" v={`${scheduleInfo.risk_percentage.toFixed(1)}% (vs. current waiting list)`} />
                  )}
                  <Row k="Estimated block hours" v={`${row.estimated_block_hours}h`} />
                  {scheduleInfo && scheduleInfo.estimated_block_hours != null && (
                    <Row k="Allocated block hours (timetable)" v={`${scheduleInfo.estimated_block_hours}h`} />
                  )}
                  {scheduleInfo?.adaptive_allocation && (
                    <Row k="Originally demanded" v={`${scheduleInfo.demanded_block_hours}h`} />
                  )}
                  <Row k="Raised date" v={row.raised_date} />
                  <Row k="Due date" v={row.due_date} />
                  <Row k="Days overdue (at generation)" v={row.days_overdue} />
                  <Row k="Approval path" v={row.approval_path} />
                  {row.rescheduled_due_to_emergency && (
                    <Row
                      k="Rescheduled due to emergency"
                      v={row.rescheduled_due_to_emergency === "removed" ? `Removed (${row.emergency_task_id})` : row.emergency_task_id}
                    />
                  )}
                  <Row k="Status" v={row.status} />
                </>
              )}
            </div>
            );
          })
        )}

        {rejectError && <ErrorBanner message={rejectError} />}

        <div className="modal-actions">
          <button className="btn btn-outline" onClick={onClose}>Close</button>
          {showReject && rows && rows.length === 1 && !rows[0].is_emergency && rows[0].status !== "approved" && rows[0].status !== "rejected" && (
            <button
              className="btn btn-danger"
              disabled={rejecting}
              onClick={() => {
                setRejecting(true);
                setRejectError(null);
                api
                  .rejectRequest(rows[0].task_id)
                  .then(() => {
                    onChanged?.(rows[0].task_id);
                    onClose();
                  })
                  .catch((e) => setRejectError(e.message))
                  .finally(() => setRejecting(false));
              }}
            >
              {rejecting ? "Rejecting…" : "Reject block request"}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

function Row({ k, v }) {
  return (
    <div className="modal-row">
      <span className="k">{k}</span>
      <span className="v">{v === null || v === undefined || v === "" ? "—" : String(v)}</span>
    </div>
  );
}
