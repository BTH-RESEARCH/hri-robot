#!/usr/bin/env bash
# Raspberry Pi 5 setup (Raspberry Pi OS Bookworm 64-bit, Lite or Desktop). Run from the repo root:
#   bash scripts/setup_pi.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> System packages"
sudo apt-get update
# --no-install-recommends: on Pi OS Lite, python3-picamera2 would otherwise pull in Qt / desktop packages
# python3-pygame: Debian's SDL has the kmsdrm driver (draws on the LCD without X/Wayland); pip's does not
sudo apt-get install -y --no-install-recommends \
    python3-venv python3-dev python3-pip python3-picamera2 python3-libcamera python3-pygame \
    libportaudio2 portaudio19-dev libatlas-base-dev alsa-utils git

echo "==> Permissions: screen (video, render), keyboard under kmsdrm (input), sound (audio)"
sudo usermod -aG video,render,input,audio "$USER"

echo "==> venv (with --system-site-packages so apt's picamera2/libcamera/pygame are visible)"
if [ ! -d .venv ]; then
    python3 -m venv --system-site-packages .venv
fi
# a venv created earlier without the flag cannot see apt's packages -> switch it on
sed -i 's/^include-system-site-packages = false/include-system-site-packages = true/' .venv/pyvenv.cfg
source .venv/bin/activate
pip install --upgrade pip wheel
pip install -e ".[dev]" -f https://k2-fsa.github.io/icefall/piper_phonemize.html
# an older install may have put pip's pygame (no kmsdrm) into the venv: remove it so apt's is used
pip uninstall -y pygame pygame-ce >/dev/null 2>&1 || true

echo "==> Sanity checks"
python -c "import picamera2, numpy; print('picamera2 ok, numpy', numpy.__version__)"
python -c "import pygame; print('pygame', pygame.version.ver, 'SDL', pygame.get_sdl_version(), 'from', pygame.__file__)"

[ -f .env ] || cp .env.example .env

echo "==> Models"
hri --profile pi5 models download

hri --profile pi5 info
echo
echo "Done. Log out and back in (or reboot) so the new groups apply, then:"
echo "  source .venv/bin/activate && nano .env   # API keys"
echo "  hri check display                         # can we draw on the LCD?"
echo "  hri test                                  # camera + FER on the LCD"
echo "  hri run"
