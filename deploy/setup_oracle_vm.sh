#!/usr/bin/env bash
# One-time setup for a fresh Oracle Cloud Always Free Ubuntu VM. Installs
# everything (Python, Postgres, Caddy), clones the public repo, fetches
# the private datasets, and installs the systemd service + reverse proxy.
# Safe to re-run -- every step below is written to be idempotent, so a
# second run (e.g. after `git pull`ing app updates) just refreshes things
# in place rather than erroring on "already exists".
#
# Run as: sudo DOMAIN=your.domain.com PRIVATE_DATA_REPO=github.com/you/railblock-private-data \
#         DATASETS_REPO_TOKEN=github_pat_... bash setup_oracle_vm.sh
#
# See docs/ORACLE_DEPLOY.md for the full walkthrough this script is one
# step of (VM creation, firewall, DNS all happen before this).
set -euo pipefail

: "${DOMAIN:?Set DOMAIN to the real domain pointed at this VMs public IP, e.g. railblock.duckdns.org}"
: "${PRIVATE_DATA_REPO:?Set PRIVATE_DATA_REPO to the private datasets repo path, e.g. github.com/you/railblock-private-data}"
: "${DATASETS_REPO_TOKEN:?Set DATASETS_REPO_TOKEN to a GitHub token with read-only access to that private repo}"
PUBLIC_REPO_URL="${PUBLIC_REPO_URL:-https://github.com/sairam-s21/Cookies4U_SIH26027.git}"
APP_DIR=/opt/railblock/app
VENV_DIR=/opt/railblock/venv
DB_NAME=railblock
DB_USER=railblock
DB_PASS=railblock

if [ "$(id -u)" -ne 0 ]; then
  echo "Run this with sudo (it installs system packages and a systemd service)." >&2
  exit 1
fi

echo "=== 1/8: installing system packages ==="
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  python3 python3-venv python3-pip \
  postgresql postgresql-contrib libpq-dev \
  git curl build-essential

echo "=== 2/8: setting up Postgres role + database ==="
sudo -u postgres psql -tc "SELECT 1 FROM pg_roles WHERE rolname = '${DB_USER}'" | grep -q 1 \
  || sudo -u postgres psql -c "CREATE ROLE ${DB_USER} LOGIN PASSWORD '${DB_PASS}';"
sudo -u postgres psql -tc "SELECT 1 FROM pg_database WHERE datname = '${DB_NAME}'" | grep -q 1 \
  || sudo -u postgres psql -c "CREATE DATABASE ${DB_NAME} OWNER ${DB_USER};"

echo "=== 3/8: fetching the public app repo ==="
mkdir -p /opt/railblock
if [ -d "$APP_DIR/.git" ]; then
  git -C "$APP_DIR" pull --ff-only
else
  git clone "$PUBLIC_REPO_URL" "$APP_DIR"
fi

echo "=== 4/8: python venv + dependencies ==="
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --quiet --upgrade pip
"$VENV_DIR/bin/pip" install --quiet -r "$APP_DIR/requirements.txt"

echo "=== 5/8: fetching private datasets ==="
(cd "$APP_DIR" && PRIVATE_DATA_REPO="$PRIVATE_DATA_REPO" DATASETS_REPO_TOKEN="$DATASETS_REPO_TOKEN" \
  bash scripts/fetch_private_data.sh)

echo "=== 6/8: writing runtime environment file ==="
cat > /opt/railblock/app.env <<EOF
DATABASE_URL=postgresql://${DB_USER}:${DB_PASS}@localhost:5432/${DB_NAME}
# 4 real OCPUs on the Always Free ARM shape -- 2 concurrent strategies x
# 4 CP-SAT workers = 8 threads, a modest 2x oversubscription (compare to
# Render free tier's forced 1x2 on a fraction of a shared core). Tune
# these after watching real timings -- see PROGRESS.md Session 39 for
# the reasoning behind each knob.
CPSAT_SEARCH_WORKERS=4
SCHEDULE_OPTIONS_MAX_CONCURRENCY=2
SCHEDULE_OPTIONS_TIME_LIMIT_S=20
EOF

echo "=== 7/8: installing the systemd service ==="
cp "$APP_DIR/deploy/railblock-api.service" /etc/systemd/system/railblock-api.service
systemctl daemon-reload
systemctl enable railblock-api
systemctl restart railblock-api

echo "=== 8/8: installing Caddy (reverse proxy + automatic HTTPS) ==="
if ! command -v caddy >/dev/null 2>&1; then
  apt-get install -y -qq debian-keyring debian-archive-keyring apt-transport-https curl
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
    | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
    > /etc/apt/sources.list.d/caddy-stable.list
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq caddy
fi
sed "s/YOUR_DOMAIN_HERE/${DOMAIN}/" "$APP_DIR/deploy/Caddyfile" > /etc/caddy/Caddyfile
systemctl restart caddy

echo
echo "Done. Checking service status..."
sleep 2
systemctl --no-pager status railblock-api | head -5
echo
echo "If DNS for ${DOMAIN} already points at this VMs public IP, and the"
echo "Oracle Cloud Security List / NSG (see docs/ORACLE_DEPLOY.md) allows"
echo "ports 80 and 443, the API should be live at: https://${DOMAIN}/health"
