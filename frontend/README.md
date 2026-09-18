# RailBlock Co-Pilot — Frontend (Session 7)

Vite + React + React Router + Tailwind. Talks to the real FastAPI backend
(`railblock.api.app`) — no mock data anywhere in `src/`.

## Run

```bash
# backend, from the project root
.venv/bin/uvicorn railblock.api.app:app --port 8000

# frontend
cd frontend
npm install
npm run dev
```

Vite defaults to port 5173 but will pick the next free port if that's
taken — check the terminal output. `VITE_API_BASE_URL` env var overrides
the API base (default `http://127.0.0.1:8000`, see `src/api/client.js`).

## Pages

`Dashboard`, `RaiseRequest`, `PendingApproval`, `RecommendedScheduling`,
`Schedule` (weekly/monthly), `History` — one per real backend workflow
stage, see `../DEMO_SCRIPT.md` for the intended walkthrough order.

## Verification scripts (`scripts/`)

Browser automation (`chromium-cli`, Playwright) wasn't fully usable in the
sandbox this was built in — Chromium's `libnspr4`/`libnss3` runtime
dependencies need `sudo apt-get`, and no passwordless sudo was available.
`vite build` passed cleanly (no compile errors), and two dependency-free
Node scripts substitute for a visual check by exercising the real live API
and asserting every property path each page component actually reads:

- `node scripts/api-shape-check.mjs` — checks every endpoint's response
  shape against a fresh backend (empty state).
- `node scripts/api-workflow-check.mjs` — drives the real submit → seed →
  recommend → approve workflow and checks the resulting weekly/history/KPI
  shapes. Both passed against the live server before this session ended.
- `node scripts/visual-check.mjs` — the actual Playwright screenshot
  driver, left in place for an environment with full Chromium system
  dependencies (`npx playwright install --with-deps chromium` on a normal
  dev machine); not runnable as-is in the sandbox this was authored in.

Run the workflow/shape scripts against a **throwaway** backend state (they
submit real requests and approve them) — restart the backend afterward for
a clean demo.
