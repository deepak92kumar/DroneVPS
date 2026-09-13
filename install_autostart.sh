#!/bin/bash
# =============================================================================
# Autonomous Drone Navigation Autostart Installer
# Installs and enables drone-vio.service to launch on boot without password
# =============================================================================

SERVICE_SRC="/home/abc/Desktop/DRONE/drone-vio.service"
SERVICE_DEST="/etc/systemd/system/drone-vio.service"

echo "================================================================="
echo " Installing Autonomous Drone Navigation Service (drone-vio)"
echo "================================================================="

if [ ! -f "$SERVICE_SRC" ]; then
    echo "[-] Error: Source service file $SERVICE_SRC not found!"
    exit 1
fi

echo "[*] Copying service unit to /etc/systemd/system/..."
sudo cp "$SERVICE_SRC" "$SERVICE_DEST"
sudo chmod 644 "$SERVICE_DEST"

echo "[*] Reloading systemd daemon..."
sudo systemctl daemon-reload

echo "[*] Enabling drone-vio service on boot..."
sudo systemctl enable drone-vio.service

echo ""
echo "================================================================="
echo "[+] SUCCESS: drone-vio.service is ENABLED!"
echo "    Jetson will now automatically start the Navigation Daemon"
echo "    upon battery connection / power boot without password!"
echo ""
echo "Commands to manage the service:"
echo "  * Start now     : sudo systemctl start drone-vio"
echo "  * Stop service  : sudo systemctl stop drone-vio"
echo "  * Check status  : sudo systemctl status drone-vio"
echo "  * View live logs: tail -f /home/abc/Desktop/DRONE/flight_logs/*.log"
echo "================================================================="
