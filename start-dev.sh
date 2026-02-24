#!/usr/bin/env bash
# VaultSpeed dev launcher
# Run from the VaultSpeed root folder: bash start-dev.sh

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo "========================================"
echo "  VaultSpeed Dev Server"
echo "========================================"
echo ""

# ── Find Python venv (check both common names) ───────────────────────────────
if [ -f "$ROOT/backend/.venv/Scripts/python" ]; then
  VENV="$ROOT/backend/.venv/Scripts/python"
elif [ -f "$ROOT/backend/.venv/bin/python" ]; then
  VENV="$ROOT/backend/.venv/bin/python"
elif [ -f "$ROOT/backend/env2/Scripts/python" ]; then
  VENV="$ROOT/backend/env2/Scripts/python"
elif [ -f "$ROOT/backend/env2/bin/python" ]; then
  VENV="$ROOT/backend/env2/bin/python"
else
  echo "[backend] No venv found. Create one first:"
  echo ""
  echo "  cd backend"
  echo "  python -m venv .venv"
  echo "  .venv/Scripts/pip install -r requirements.txt   # Windows"
  echo "  .venv/bin/pip install -r requirements.txt       # macOS/Linux"
  echo ""
  exit 1
fi

echo "[backend] Using venv: $VENV"
echo "[backend] Starting on http://0.0.0.0:8000 ..."
"$VENV" "$ROOT/backend/local_server.py" &
BACKEND_PID=$!

# Wait for backend to be ready (up to 15s)
echo -n "[backend] Waiting for server"
for i in {1..15}; do
  sleep 1
  if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo " ✓"
    break
  fi
  echo -n "."
done

# ── Show laptop IP ────────────────────────────────────────────────────────────
if command -v powershell.exe &>/dev/null; then
  LAPTOP_IP=$(powershell.exe -Command "Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias 'Wi-Fi' | Select -ExpandProperty IPAddress" 2>/dev/null | tr -d '\r')
elif command -v ipconfig &>/dev/null; then
  LAPTOP_IP=$(ipconfig | grep -A1 "Wireless" | grep "IPv4" | head -1 | awk '{print $NF}' | tr -d '\r')
else
  LAPTOP_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
fi

echo ""
echo "[network] Laptop IP:   ${LAPTOP_IP:-<run ipconfig to find your IP>}"
echo "[network] Backend URL: http://${LAPTOP_IP:-<YOUR_IP>}:8000"
echo ""
echo "  → Phone and laptop must be on the same WiFi"
echo "  → In app Setup screen: set Backend URL to the URL above"
echo ""

# ── Expo ─────────────────────────────────────────────────────────────────────
echo "[expo] Starting Expo (tunnel mode)..."
cd "$ROOT/app" && npx expo start --tunnel

# Cleanup on exit
trap "kill $BACKEND_PID 2>/dev/null" EXIT
