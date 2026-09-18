# Deploying RailBlock Co-Pilot

## One-time setup: the private datasets repo

Railway authorities' instruction: the real railway-sourced files below
must never be pushed to this public repo. They're already excluded by
`.gitignore` (see its "Session 36" section) and instead need to live in
a **separate private** GitHub repo that gets pulled down at build time.

**1. Create the private repo yourself** (on github.com, not via Claude/CLI):
   - New repository → name it anything, e.g. `railblock-private-data` →
     visibility **Private** → create, empty (no README).

**2. Push the staged files to it.** A ready-to-push folder with the exact
   layout this needs has already been prepared at:
   `/home/sairam_s/Projects/railblock-private-data-staging/`
   It mirrors this project's own relative paths exactly (`tasks.csv` at
   its root, `datasets/...`, `data/derived/...`), so pushing it as-is is
   enough:
   ```bash
   cd /home/sairam_s/Projects/railblock-private-data-staging
   git init
   git add .
   git commit -m "Private railway datasets"
   git branch -M main
   git remote add origin https://github.com/<your-username>/railblock-private-data.git
   git push -u origin main
   ```
   (If that folder ever goes missing, just re-copy the 9 files listed
   under "What's in the private repo" below from this project into a
   fresh folder with the same layout.)

**3. Generate a read-only access token scoped to ONLY that repo:**
   - GitHub → Settings → Developer settings → Personal access tokens →
     **Fine-grained tokens** → Generate new token.
   - Repository access: **Only select repositories** → pick
     `railblock-private-data`.
   - Permissions → Repository permissions → **Contents: Read-only**
     (nothing else needs any access).
   - Generate, copy the token (starts with `github_pat_...`) — you won't
     see it again.

**4. Set two environment variables on the Render service** (Render
   dashboard → your service → Environment):
   - `PRIVATE_DATA_REPO` = `github.com/<your-username>/railblock-private-data`
   - `DATASETS_REPO_TOKEN` = the token from step 3 (paste as a secret —
     `render.yaml` deliberately never contains this value, so it's safe
     that `render.yaml` itself is in the public repo)

That's it — every Render build now runs `scripts/fetch_private_data.sh`
(see `render.yaml`'s `buildCommand`) before installing dependencies,
which clones the private repo fresh and copies its files into place.

## What's in the private repo

| Path (same in both repos) | What it is |
|---|---|
| `tasks.csv` | The 70 real maintenance tasks the demo batch is built from |
| `granted_blocks_history.xlsx` | The fixed granted-history dataset (Dashboard/Completed History) |
| `datasets/Train_details_22122017.csv` | Real train timetable |
| `datasets/india_railway_stations.csv` | Real station geo data |
| `data/derived/real_train_details_2026.csv` | Real train-roster override |
| `data/derived/train_delay_history.csv` | Real delay history (feeds the live delay-margin model) |
| `data/derived/real_running_days.json` | Real weekly-running-days data |
| `data/derived/ml/block_utilization_regressor.joblib` | Trained adaptive-allocation model |
| `data/derived/ml/delay_risk_regressor.joblib` | Trained delay-risk model |

**Not included on purpose**: `data/derived/tasks_csv_import_state.json`.
It isn't a dataset — it's a mutable counter the app itself writes as it
runs (how many `tasks.csv` rows have been imported so far). It must
regenerate fresh on every new deployment (it starts at "0 imported" when
absent); seeding it from this dev machine's own current value would make
a fresh deploy's importer think there's nothing left to import, silently
leaving the demo-batch template pool empty.

## Local development

Nothing changes for local dev on a machine that already has these files
(like this one) — they're just regular files on disk; `.gitignore` only
affects what `git add`/`git commit` picks up, not what's already there
or what the app reads at runtime.

For a **teammate cloning the public repo fresh**, running locally needs
the same private-repo access:
```bash
export PRIVATE_DATA_REPO=github.com/<your-username>/railblock-private-data
export DATASETS_REPO_TOKEN=<a token scoped to that repo, see step 3 above>
bash scripts/fetch_private_data.sh
```

## Frontend (Vercel)

Set the Vercel project's `VITE_API_BASE_URL` env var to the Render
service's real URL (shown in the Render dashboard once deployed).
