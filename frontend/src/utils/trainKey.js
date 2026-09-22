// A train-passing block a viewer just inspected is otherwise
// indistinguishable from every other one once the details modal closes,
// making it hard to relocate. Clicking a train block marks its exact
// (date, train_no, start_minute, end_minute) instance with a persistent,
// distinct color -- see HourGrid.jsx's own HIGHLIGHT_COLORS usage.
//
// Lives in its own file, separate from HourGrid.jsx's default component
// export, because Vite's React Fast Refresh cannot cleanly hot-swap a
// file that exports both a component and a plain function -- editing
// such a file triggers "Could not Fast Refresh (export is incompatible)"
// and leaves an already-open browser tab with a crashed, half-updated
// component tree (an uncaught render error, no error boundary to
// recover) that only a manual reload can fix. Keeping HourGrid.jsx's
// exports to just its component means any future edit to it hot-swaps
// cleanly.
export function trainKey(date, t) {
  return `${date}__${t.train_no}__${t.start_minute}__${t.end_minute}`;
}
