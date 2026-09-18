#!/usr/bin/env bash
# Session 36, at explicit user request ("we are instructed by railway
# authorities to not push the datasets publicly into github"): pulls the
# real railway-sourced files this app needs to run -- but must never be
# committed to THIS public repo -- from a separate PRIVATE GitHub repo,
# at build time. See DEPLOY.md for the one-time setup (creating the
# private repo, pushing these files there, generating the token).
#
# Safe to run repeatedly (e.g. on every Render build) -- it always
# re-fetches fresh rather than assuming a previous run's copy is still
# correct. Does nothing destructive to anything else in the working tree.
set -euo pipefail

: "${PRIVATE_DATA_REPO:?Set PRIVATE_DATA_REPO to the private repo path, e.g. github.com/you/railblock-private-data -- see DEPLOY.md}"
: "${DATASETS_REPO_TOKEN:?Set DATASETS_REPO_TOKEN to a GitHub personal access token with read-only access to ONLY that private repo -- see DEPLOY.md}"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

echo "Fetching private railway datasets from ${PRIVATE_DATA_REPO}..."
git clone --depth 1 "https://x-access-token:${DATASETS_REPO_TOKEN}@${PRIVATE_DATA_REPO}.git" "$TMP_DIR" --quiet

# The private repo mirrors this project's own relative layout exactly
# (tasks.csv and granted_blocks_history.xlsx at its root, datasets/...,
# data/derived/...), so copying it straight over the top just fills in
# the files .gitignore excludes here.
[ -f "$TMP_DIR/tasks.csv" ] && cp "$TMP_DIR/tasks.csv" .
[ -f "$TMP_DIR/granted_blocks_history.xlsx" ] && cp "$TMP_DIR/granted_blocks_history.xlsx" .
[ -d "$TMP_DIR/datasets" ] && cp -r "$TMP_DIR/datasets/." datasets/
[ -d "$TMP_DIR/data" ] && cp -r "$TMP_DIR/data/." data/

echo "Private datasets fetched successfully."
