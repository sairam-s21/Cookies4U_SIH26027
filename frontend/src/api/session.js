// Session 32, at explicit user request: per-visitor isolation for a
// publicly-hosted, shared-link deployment -- every browser gets its own
// random id, persisted in localStorage so it survives a page reload, and
// sent as the X-Session-Id header on every backend call (see client.js).
// Two different visitors (two different browsers) never share one -- the
// backend treats every distinct value as a fully separate waiting list /
// approved schedule / emergencies, with no login required.
const STORAGE_KEY = "railblock_session_id";

// In-memory fallback for when localStorage itself is unavailable (private
// browsing, blocked storage, etc.) -- cached at module scope so every call
// during this one page load still agrees on the same id; only cross-reload
// persistence is lost in that case, never isolation within the page.
let _memoryFallbackId = null;

export function getSessionId() {
  try {
    let id = localStorage.getItem(STORAGE_KEY);
    if (!id) {
      id = crypto.randomUUID();
      localStorage.setItem(STORAGE_KEY, id);
    }
    return id;
  } catch {
    if (!_memoryFallbackId) _memoryFallbackId = crypto.randomUUID();
    return _memoryFallbackId;
  }
}
