// Shared read-only details modal for any already-granted block row --
// live-approved (GET /tasks/history), granted-history completed (GET
// /tasks/history/completed), or currently-active/upcoming (GET
// /blocks/active, /blocks/upcoming). All four share the same row shape,
// so one modal covers Approved Tasks, Completed History, and the
// Dashboard's Currently Active / Upcoming Blocks sections.
export default function GrantedBlockDetailsModal({ row, onClose }) {
  return (
    <div className="overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <h3>{row.task_id}</h3>
          <button className="modal-close" onClick={onClose}>×</button>
        </div>
        <div className="modal-row"><span className="k">Section</span><span className="v">{row.section_id}</span></div>
        <div className="modal-row"><span className="k">Department</span><span className="v">{row.department}</span></div>
        <div className="modal-row"><span className="k">Option</span><span className="v">{row.option}{row.negotiated_exception ? " (negotiated exception)" : ""}</span></div>
        <div className="modal-row"><span className="k">Requester priority</span><span className="v">{row.requester_priority}</span></div>
        <div className="modal-row"><span className="k">Estimated block hours</span><span className="v">{row.estimated_block_hours ?? "—"}h</span></div>
        <div className="modal-row"><span className="k">Approval path</span><span className="v">{row.approval_path}</span></div>
        <div className="modal-row"><span className="k">On time (vs due date)</span><span className="v">{String(row.on_time)}</span></div>
        <div className="modal-actions">
          <button className="btn btn-outline" onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  );
}
