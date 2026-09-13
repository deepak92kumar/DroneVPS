#!/bin/bash
# ============================================================
# Skydroid C12 Gimbal & SIYI Air Unit Ethernet Link Setup
# ============================================================
echo "============================================================"
echo " Configuring Jetson Ethernet (enP8p1s0) for SIYI Air Unit"
echo " Subnet: 192.168.144.0/24 | Jetson IP: 192.168.144.30"
echo "============================================================"

# Ensure link is UP
sudo ip link set enP8p1s0 up

# Assign static IP 192.168.144.30
sudo ip addr replace 192.168.144.30/24 dev enP8p1s0

echo "[+] Jetson Ethernet configured: 192.168.144.30/24"
echo ""
echo "[*] Testing ping to SIYI Air Unit (192.168.144.11)..."
if ping -c 2 -W 1 192.168.144.11 > /dev/null 2>&1; then
    echo "[+] SUCCESS: SIYI Air Unit is ONLINE (192.168.144.11)!"
else
    echo "[!] SIYI Air Unit (192.168.144.11) not reachable yet."
fi

echo ""
echo "[*] Testing ping to Skydroid C12 Gimbal Camera (192.168.144.108)..."
if ping -c 2 -W 1 192.168.144.108 > /dev/null 2>&1; then
    echo "[+] SUCCESS: Skydroid C12 Camera is ONLINE (192.168.144.108)!"
    echo "[+] RTSP Stream Ready: rtsp://192.168.144.108:554/stream=1"
else
    echo "[!] Skydroid C12 (192.168.144.108) not reachable yet."
    echo "    - Check: Is Ethernet cable plugged between Jetson (RJ45) and SIYI Air Unit (LAN 2)?"
    echo "    - Check: Is Skydroid C12 plugged into LAN 1 and powered on?"
fi
echo "============================================================"
