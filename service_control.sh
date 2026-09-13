#!/bin/bash
# =============================================================================
# Autonomous Drone Service Controller & Live Log Viewer
# =============================================================================

while true; do
    clear
    echo "================================================================="
    echo "       Autonomous Drone Navigation Service Controller"
    echo "================================================================="
    
    # Check if installed
    if [ -f /etc/systemd/system/drone-vio.service ]; then
        IS_ACTIVE=$(systemctl is-active drone-vio 2>/dev/null)
        IS_ENABLED=$(systemctl is-enabled drone-vio 2>/dev/null)
        echo "  Service Installation : INSTALLED"
        echo "  Boot Auto-Start      : $IS_ENABLED"
        echo "  Current Running State: $IS_ACTIVE"
    else
        echo "  Service Installation : NOT INSTALLED (Run install_autostart.sh)"
    fi
    echo "================================================================="
    echo "  1. Start Service Now (sudo systemctl start drone-vio)"
    echo "  2. Stop Service (sudo systemctl stop drone-vio)"
    echo "  3. Restart Service (sudo systemctl restart drone-vio)"
    echo "  4. View Live Tail Logs (Real-time telemetry & VIO stream)"
    echo "  5. Check Full Status & Diagnostics"
    echo "  6. Install / Enable Autostart on Boot"
    echo "  7. Uninstall / Disable Autostart"
    echo "  8. Exit"
    echo "================================================================="
    read -p "Select option (1-8): " choice

    case $choice in
        1)
            echo "[*] Starting drone-vio.service..."
            sudo systemctl start drone-vio
            sleep 1
            ;;
        2)
            echo "[*] Stopping drone-vio.service..."
            sudo systemctl stop drone-vio
            sleep 1
            ;;
        3)
            echo "[*] Restarting drone-vio.service..."
            sudo systemctl restart drone-vio
            sleep 1
            ;;
        4)
            LATEST_LOG=$(ls -t /home/abc/Desktop/DRONE/flight_logs/flight_*.log 2>/dev/null | head -n 1)
            if [ -n "$LATEST_LOG" ]; then
                echo "[*] Tailing latest log: $LATEST_LOG (Press Ctrl+C to stop)..."
                tail -n 25 -f "$LATEST_LOG"
            else
                echo "[-] No log files found in /home/abc/Desktop/DRONE/flight_logs/."
                read -p "Press Enter to continue..."
            fi
            ;;
        5)
            echo "--- Systemd Service Status ---"
            sudo systemctl status drone-vio --no-pager
            echo ""
            read -p "Press Enter to continue..."
            ;;
        6)
            bash /home/abc/Desktop/DRONE/install_autostart.sh
            read -p "Press Enter to continue..."
            ;;
        7)
            bash /home/abc/Desktop/DRONE/uninstall_autostart.sh
            read -p "Press Enter to continue..."
            ;;
        8)
            echo "Exiting."
            exit 0
            ;;
        *)
            echo "Invalid option."
            sleep 1
            ;;
    esac
done
