import api from "./client.js";

// Single shared check every page can await BEFORE fetching its own
// data, so no page -- Dashboard included -- ever reads an empty waiting
// list a fresh session hasn't actually finished loading yet. Keeping
// this in one place (rather than duplicated per page) means a visitor
// landing on Dashboard sees the same auto-loaded batch as one landing
// on Waiting List first.
//
// Cached as a module-level promise (not re-checked per call) so mounting
// several pages during one visit -- or React StrictMode's dev-mode
// double-invoke -- never fires two concurrent activate calls; every
// caller just awaits the same one in-flight/settled promise.
let _ensured = null;

export function ensureDemoBatch() {
  if (!_ensured) {
    _ensured = api
      .listRequests()
      .then((all) => (all.length === 0 ? api.activateDemoBatch() : null))
      .catch(() => null); // a failed check must never block the app from rendering
  }
  return _ensured;
}
