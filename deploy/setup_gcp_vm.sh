#!/usr/bin/env bash
# One-time setup for a fresh GCP Always Free e2-micro Ubuntu VM. Installs
# everything (Python, Postgres, Caddy, a swap file), clones the public
# repo, fetches the private datasets, and installs the systemd service +
# reverse proxy. Safe to re-run -- every step is idempotent.
#
# Run as: sudo DOMAIN=your.domain.com PRIVATE_DATA_REPO=github.com/you/railblock-private-data \
#         DATASETS_REPO_TOKEN=github_pat_... bash setup_gcp_vm.sh
#
# See docs/GCP_DEPLOY.md for the full walkthrough this script is one
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
SWAP_FILE=/swapfile
SWAP_SIZE_MB=2048  # 2GB swap on top of e2-micro's 1GB RAM -- a real safety
                    # net, not a performance feature: pandas/numpy/OR-Tools
                    # briefly touching more memory than physically present
                    # should page to disk and survive, not trigger the
                    # OOM killer and take the whole service down.

if [ "$(id -u)" -ne 0 ]; then
  echo "Run this with sudo (it installs system packages and a systemd service)." >&2
  exit 1
fi

echo "=== 1/9: swap file (e2-micro has only 1GB real RAM) ==="
if [ ! -f "$SWAP_FILE" ]; then
  fallocate -l "${SWAP_SIZE_MB}M" "$SWAP_FILE"
  chmod 600 "$SWAP_FILE"
  mkswap "$SWAP_FILE"
  swapon "$SWAP_FILE"
  echo "$SWAP_FILE none swap sw 0 0" >> /etc/fstab
else
  echo "swap file already exists, skipping"
fi

echo "=== 2/9: installing system packages ==="
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  python3 python3-venv python3-pip \
  postgresql postgresql-contrib libpq-dev \
  git curl build-essential

echo "=== 3/9: tuning Postgres for 1GB RAM ==="
PG_CONF=$(sudo -u postgres psql -tAc "SHOW config_file;")
sed -i "s/^#\?shared_buffers.*/shared_buffers = 64MB/" "$PG_CONF"
sed -i "s/^#\?work_mem.*/work_mem = 4MB/" "$PG_CONF"
sed -i "s/^#\?maintenance_work_mem.*/maintenance_work_mem = 32MB/" "$PG_CONF"
systemctl restart postgresql

echo "=== 4/9: setting up Postgres role + database ==="
sudo -u postgres psql -tc "SELECT 1 FROM pg_roles WHERE rolname = '${DB_USER}'" | grep -q 1 \
  || sudo -u postgres psql -c "CREATE ROLE ${DB_USER} LOGIN PASSWORD '${DB_PASS}';"
sudo -u postgres psql -tc "SELECT 1 FROM pg_database WHERE datname = '${DB_NAME}'" | grep -q 1 \
  || sudo -u postgres psql -c "CREATE DATABASE ${DB_NAME} OWNER ${DB_USER};"

echo "=== 5/9: dedicated system user to run the service ==="
id -u railblock >/dev/null 2>&1 || useradd --system --home-dir /opt/railblock --shell /usr/sbin/nologin railblock

echo "=== 6/9: fetching the public app repo ==="
mkdir -p /opt/railblock
if [ -d "$APP_DIR/.git" ]; then
  git -C "$APP_DIR" pull --ff-only
else
  git clone "$PUBLIC_REPO_URL" "$APP_DIR"
fi

echo "=== 7/9: python venv + dependencies ==="
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --quiet --upgrade pip
"$VENV_DIR/bin/pip" install --quiet -r "$APP_DIR/requirements.txt"

echo "=== 8/9: fetching private datasets + runtime env ==="
(cd "$APP_DIR" && PRIVATE_DATA_REPO="$PRIVATE_DATA_REPO" DATASETS_REPO_TOKEN="$DATASETS_REPO_TOKEN" \
  bash scripts/fetch_private_data.sh)
chown -R railblock:railblock /opt/railblock

cat > /opt/railblock/app.env <<EOF
DATABASE_URL=postgresql://${DB_USER}:${DB_PASS}@localhost:5432/${DB_NAME}
# e2-micro is 1 SHARED vCPU -- not meaningfully more raw power than
# Render's free tier, so these stay conservative (same values Render's
# render.yaml uses). The real benefit of this host is consistency: no
# cold starts, no other tenants' workloads stealing your CPU mid-solve.
# Tune up only after watching real timings on the actual hardware.
CPSAT_SEARCH_WORKERS=2
SCHEDULE_OPTIONS_MAX_CONCURRENCY=1
SCHEDULE_OPTIONS_TIME_LIMIT_S=12
EOF

echo "=== 9/9: systemd service + Caddy (reverse proxy, automatic HTTPS) ==="
cp "$APP_DIR/deploy/railblock-api-gcp.service" /etc/systemd/system/railblock-api.service
systemctl daemon-reload
systemctl enable railblock-api
systemctl restart railblock-api

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
free -h
echo
echo "If DNS for ${DOMAIN} already points at this VMs external IP, and the"
echo "GCP firewall allows ports 80/443 (see docs/GCP_DEPLOY.md), the API"
echo "should be live at: https://${DOMAIN}/health"
