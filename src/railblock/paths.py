"""Central path constants so every module agrees on where real data lives
and where derived/generated artifacts go."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = PROJECT_ROOT / "datasets"
DERIVED_DIR = PROJECT_ROOT / "data" / "derived"

TRAIN_DETAILS_CSV = DATASETS_DIR / "Train_details_22122017.csv"
STATIONS_CSV = DATASETS_DIR / "india_railway_stations.csv"

RS_SESSION_246_CSV = DATASETS_DIR / "RS_Session_246_AU471_1.1.csv"
RS_SESSION_247_CSV = DATASETS_DIR / "RS_Session_247_AU_604.csv"
RS_SESSION_250_CSV = DATASETS_DIR / "RS_Session_250_AU1409.csv"
SESSION_244_CSV = DATASETS_DIR / "Session_244_AS127_1.1.csv"

CORRIDOR_STATIONS_CSV = DERIVED_DIR / "corridor_stations.csv"
BLOCK_SECTIONS_CSV = DERIVED_DIR / "block_sections.csv"
GEO_FLAGGED_CSV = DERIVED_DIR / "geographic_cross_check_flagged.csv"

# Session 13: real-train-data rebuild (see railblock.integrations.fetch_real_train_data)
TRAIN_FETCH_CANDIDATES_JSON = DERIVED_DIR / "train_fetch_candidates.json"
TRAIN_FETCH_PROGRESS_JSON = DERIVED_DIR / "train_fetch_progress.json"
REAL_TRAIN_DETAILS_CSV = DERIVED_DIR / "real_train_details_2026.csv"

# Session 26: real per-station delay history (see railblock.integrations.fetch_train_delay_history)
DELAY_HISTORY_PROGRESS_JSON = DERIVED_DIR / "delay_history_progress.json"
TRAIN_DELAY_HISTORY_CSV = DERIVED_DIR / "train_delay_history.csv"
TRAIN_DELAY_STATION_SUMMARY_CSV = DERIVED_DIR / "train_delay_station_summary.csv"

# Session 13: live-watched task import (see railblock.integrations.import_tasks
# and the background watcher started in api/app.py) -- appending a row to
# this file and saving is picked up automatically, no manual re-run needed.
TASKS_CSV_WATCH_PATH = PROJECT_ROOT / "tasks.csv"
TASKS_CSV_IMPORT_STATE_JSON = DERIVED_DIR / "tasks_csv_import_state.json"

# Session 17, at explicit user request: a fixed, never-modified set of
# already-granted historical blocks (some completed, some approved but
# still running/upcoming) -- see railblock.synthetic.granted_history.
# Not live-watched like tasks.csv -- generated once, read many times.
GRANTED_HISTORY_XLSX = PROJECT_ROOT / "granted_blocks_history.xlsx"

# Adaptive-allocation regression: synthetic, real-anchored historical
# block-utilization dataset (see railblock.ml.historical_utilization) and
# the regression model trained on it.
HISTORICAL_UTILIZATION_CSV = DERIVED_DIR / "historical_block_utilization.csv"
ML_DIR = PROJECT_ROOT / "data" / "derived" / "ml"
UTILIZATION_MODEL_JOBLIB = ML_DIR / "block_utilization_regressor.joblib"
DELAY_MODEL_JOBLIB = ML_DIR / "delay_risk_regressor.joblib"

DERIVED_DIR.mkdir(parents=True, exist_ok=True)
ML_DIR.mkdir(parents=True, exist_ok=True)
