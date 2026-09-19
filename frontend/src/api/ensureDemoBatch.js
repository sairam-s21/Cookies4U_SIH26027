import api from "./client.js";

// Session 39, at explicit user request, after a real reported gap: the
// auto-load lived only in WaitingList.jsx, so a visitor landing on
// Dashboard (the default route) first saw "0 pending" until they
// separately visited Waiting List and triggered it there. This is the
// single shared check every page can await BEFORE fetching its own
// data, so no page -- Dashboard included -- ever reads an empty waiting
// list a fresh session hasn't actually finished loading yet.
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
