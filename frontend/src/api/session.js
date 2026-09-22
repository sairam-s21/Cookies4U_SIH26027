// Per-visitor isolation for a publicly-hosted, shared-link deployment --
// every browser tab gets its own random id, sent as the X-Session-Id
// header on every backend call (see client.js). Two different visitors
// (two different browsers/tabs) never share one -- the backend treats
// every distinct value as a fully separate waiting list / approved
// schedule / emergencies, with no login required.
//
// sessionStorage, not localStorage: the id (and so the demo batch it's
// tied to, see ensureDemoBatch.js) must survive a page refresh within
// the same visit, but reset the next time the tab/browser is closed and
// reopened, so a fresh visit always gets a clean batch. localStorage
// would persist forever across restarts; sessionStorage clears exactly
// on tab/browser close, which is the actual requirement here.
const STORAGE_KEY = "railblock_session_id";

// In-memory fallback for when sessionStorage itself is unavailable
// (private browsing, blocked storage, etc.) -- cached at module scope so
// every call during this one page load still agrees on the same id;
// only cross-reload persistence is lost in that case, never isolation
// within the page.
let _memoryFallbackId = null;

export function getSessionId() {
  try {
    let id = sessionStorage.getItem(STORAGE_KEY);
    if (!id) {
      id = crypto.randomUUID();
      sessionStorage.setItem(STORAGE_KEY, id);
    }
    return id;
  } catch {
    if (!_memoryFallbackId) _memoryFallbackId = crypto.randomUUID();
    return _memoryFallbackId;
  }
}
