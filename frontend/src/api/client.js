// Thin wrapper around the real RailBlock Co-Pilot API (railblock.api.app).
// No mock data anywhere in this module -- every function here hits the
// live FastAPI backend. Set VITE_API_BASE_URL to point elsewhere; defaults
// to the local dev server started with `uvicorn railblock.api.app:app`.

import { getSessionId } from "./session.js";

const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

async function request(path, options = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json", "X-Session-Id": getSessionId() },
    ...options,
  });
  if (!res.ok) {
    let detail;
    try {
      detail = (await res.json()).detail;
    } catch {
      detail = res.statusText;
    }
    throw new Error(`${options.method || "GET"} ${path} -> ${res.status}: ${detail}`);
  }
  return res.json();
}

export const api = {
  health: () => request("/health"),
  // Session 29, at explicit user request, after a real reported bug:
  // the server's own real current date -- see app.py's GET /now for why
  // every "today" default in this frontend should come from here, not
  // the viewer's own machine clock.
  now: () => request("/now"),
  corridor: () => request("/corridor"),

  listRequests: (status) =>
    request(`/requests${status ? `?status=${encodeURIComponent(status)}` : ""}`),
  requestsWhittleRank: () => request("/requests/whittle-rank"),
  getRequest: (taskId) => request(`/requests/${encodeURIComponent(taskId)}`),
  rejectRequest: (taskId) =>
    request(`/requests/${encodeURIComponent(taskId)}/reject`, { method: "POST" }),

  // Session 32, at explicit user request: per-visitor "Create demo batch
  // of tasks" / "Reset" on the Waiting List page -- see api/app.py's
  // /demo/batch/* endpoints.
  activateDemoBatch: () => request("/demo/batch/activate", { method: "POST" }),
  resetDemoBatch: () => request("/demo/batch/reset", { method: "POST" }),

  recommend: (payload) =>
    request("/schedule/recommend", { method: "POST", body: JSON.stringify(payload) }),
  lastRecommendation: () => request("/schedule/recommendation"),
  scheduleOptions: (payload) =>
    request("/schedule/options", { method: "POST", body: JSON.stringify(payload) }),
  lastScheduleOptions: () => request("/schedule/options"),
  approve: (payload = {}) =>
    request("/schedule/approve", { method: "POST", body: JSON.stringify(payload) }),

  weeklySchedule: () => request("/schedule/weekly"),
  monthlySchedule: ({ start_date, n_weeks = 4, time_limit_s = 30 }) =>
    request(
      `/schedule/monthly?start_date=${start_date}&n_weeks=${n_weeks}&time_limit_s=${time_limit_s}`
    ),

  trainPositions: (date, minute, live = false) =>
    request(`/trains/positions?date=${date}&minute=${minute}${live ? "&live=true" : ""}`),
  // Session 29, at explicit user request, after a real reported bug:
  // Weekly/Monthly Schedule's Next/Previous fetches one of these per day
  // in view (up to 31 at once in Monthly mode) -- clicking Next/Previous
  // repeatedly without an `signal` to cancel the PREVIOUS click's
  // still-in-flight batch let them pile up across clicks until Chromium
  // hit ERR_INSUFFICIENT_RESOURCES (too many pending connections to one
  // origin), which starves every other request on the page too, making
  // the whole UI look frozen. `signal` (optional) lets a caller cancel
  // this specific request via AbortController when it's superseded.
  sectionTrainSchedule: (sectionId, date, signal) =>
    request(`/corridor/sections/${encodeURIComponent(sectionId)}/trains?date=${date}`, { signal }),

  history: () => request("/tasks/history"),
  completedHistory: () => request("/tasks/history/completed"),
  activeBlocks: () => request("/blocks/active"),
  upcomingBlocks: (limit = 5) => request(`/blocks/upcoming?limit=${limit}`),

  // Session 30, at explicit user request: Emergency Handling -- see
  // railblock.scheduling.emergency.
  createEmergency: () => request("/emergency/create", { method: "POST" }),
  resolveEmergency: (taskId, apply) =>
    request(`/emergency/${encodeURIComponent(taskId)}/resolve`, {
      method: "POST",
      body: JSON.stringify({ apply }),
    }),
  listEmergencies: () => request("/emergency/list"),
  // Session 32, at explicit user request: Emergency Handling's own
  // per-visitor "Reset" button.
  resetEmergencies: () => request("/emergency/reset", { method: "POST" }),
};

export default api;
