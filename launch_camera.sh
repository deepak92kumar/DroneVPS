#!/bin/bash
export DISPLAY="${DISPLAY:-:0}"
cd /home/abc/Desktop/DRONE

# If background autonomous daemon is running, pause it so GUI preview can access the camera
WAS_RUNNING=0
if systemctl is-active --quiet drone-vio 2>/dev/null; then
    echo "[*] Temporarily pausing background daemon for GUI preview..."
    sudo systemctl stop drone-vio
    WAS_RUNNING=1
fi

python3 live_preview.py

# When GUI closes, restore background daemon if it was running
if [ "$WAS_RUNNING" -eq 1 ]; then
    echo "[*] Restoring background flight daemon..."
    sudo systemctl start drone-vio
fi
