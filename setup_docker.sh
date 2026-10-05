#!/usr/bin/env bash
# Finish the IsaacLab-Arena (GR00T e2e tutorial) host setup.
# Run WITH sudo:  sudo ./setup_docker.sh
# Idempotent: safe to re-run if a step already succeeded.
set -e

if [ "$(id -u)" -ne 0 ]; then
  echo "Please run with sudo:  sudo $0" >&2
  exit 1
fi

REAL_USER="${SUDO_USER:-$USER}"
echo "== 1/5 apt packages (docker.io may already be installed) =="
apt-get update
apt-get install -y docker.io curl gnupg

echo "== 2/5 NVIDIA container toolkit apt repo =="
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
apt-get install -y nvidia-container-toolkit

echo "== 3/5 register the nvidia runtime with Docker =="
nvidia-ctk runtime configure --runtime=docker

echo "== 4/5 enable + restart Docker =="
systemctl enable --now docker
systemctl restart docker

echo "== 5/5 add ${REAL_USER} to the docker group =="
usermod -aG docker "${REAL_USER}"

echo
echo "== verification =="
docker info 2>/dev/null | grep -E "Server Version|nvidia" | head -3 || true
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi \
  && echo "GPU-in-docker OK" \
  || echo "GPU test container needs a pull; rerun this script if it failed on network"

cat <<'NOTE'

DONE. If this is the first run, log out and back in (or run: newgrp docker)
so group membership takes effect. Then as the normal user:

    cd ~/IsaacLab-Arena && ./docker/run_docker.sh

NOTE
