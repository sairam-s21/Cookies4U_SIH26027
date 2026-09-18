// Session 17, at explicit user request: a train-passing block a user just
// inspected is otherwise indistinguishable from every other one once the
// details modal closes, making it hard to relocate. Clicking a train
// block marks its exact (date, train_no, start_minute, end_minute)
// instance with a persistent, distinct color -- see HourGrid.jsx's own
// HIGHLIGHT_COLORS usage.
//
// Session 29, at explicit user request, after a real reported bug: this
// used to live inside HourGrid.jsx, exported alongside its default
// component export. Vite's React Fast Refresh cannot cleanly hot-swap a
// file that exports both a component and a plain function -- editing
// HourGrid.jsx repeatedly during this session triggered exactly that
// ("Could not Fast Refresh (trainKey export is incompatible)"), which
// left a real, already-open browser tab with a crashed, half-updated
// component tree (an uncaught render error, no error boundary to
// recover) that no amount of clicking anything on the page could fix --
// only a manual reload. Moved to its own file so HourGrid.jsx exports
// only its component, and any future edit to it hot-swaps cleanly.
export function trainKey(date, t) {
  return `${date}__${t.train_no}__${t.start_minute}__${t.end_minute}`;
}
