#!/bin/bash
# AWS EC2 GPU/CPU setup — Ubuntu 22.04
# Run as: sudo bash ec2_setup.sh
set -euo pipefail

APP_DIR=/opt/cricgiri
REPO_URL="${REPO_URL:-}"  # set your git repo URL

apt-get update
apt-get install -y python3.11 python3.11-venv python3-pip nginx git \
  libgl1 libglib2.0-0 ffmpeg

mkdir -p "$APP_DIR"
if [ -n "$REPO_URL" ]; then
  git clone "$REPO_URL" "$APP_DIR" || true
fi

cd "$APP_DIR"
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements-prod.txt

# GPU (optional — g4dn instances)
if command -v nvidia-smi &>/dev/null; then
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  echo "DEVICE=0" >> .env
  echo "USE_HALF_PRECISION=true" >> .env
fi

mkdir -p uploads outputs/api logs
cp deploy/systemd/cricgiri-api.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable cricgiri-api
systemctl start cricgiri-api

cp deploy/nginx/cricgiri.conf /etc/nginx/sites-available/cricgiri
ln -sf /etc/nginx/sites-available/cricgiri /etc/nginx/sites-enabled/
nginx -t && systemctl reload nginx

echo "Done. Attach Elastic IP and point DNS to this instance."
