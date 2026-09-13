#!/bin/bash
# =============================================================================
# Autonomous Drone Navigation Autostart Uninstaller
# Stops and disables drone-vio.service
# =============================================================================

echo "================================================================="
echo " Disabling Autonomous Drone Navigation Service (drone-vio)"
echo "================================================================="

echo "[*] Stopping drone-vio service..."
sudo systemctl stop drone-vio.service 2>/dev/null

echo "[*] Disabling drone-vio service from boot..."
sudo systemctl disable drone-vio.service 2>/dev/null

echo "[*] Removing /etc/systemd/system/drone-vio.service..."
sudo rm -f /etc/systemd/system/drone-vio.service

echo "[*] Reloading systemd daemon..."
sudo systemctl daemon-reload

echo "[+] SUCCESS: drone-vio.service has been uninstalled."
echo "================================================================="
