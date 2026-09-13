#!/usr/bin/env python3
"""
EKF3 Parameter Tuning for VIO Drift Reduction
===========================================
Reduces position drift by tuning EKF parameters for better VIO performance.
Run this script to optimize EKF for stable VIO hovering.
"""

import sys
import time
from cuav_mavlink_bridge import CuavMavlinkClient

def tune_ekf_for_stable_vio(client):
    """
    Tune EKF3 parameters to reduce VIO drift.
    Focuses on position hold stability and drift minimization.
    """
    print("[*] Tuning EKF3 parameters for VIO drift reduction...")
    
    # Position noise and innovation limits
    params = {
        # Reduce position noise - tighter position hold
        'EK3_POSNE_M_NSE': 0.1,    # Position measurement noise (default: 0.5)
        'EK3_ALT_M_NSE': 0.1,     # Altitude measurement noise (default: 2.0)
        
        # Innovation limits - reject bad VIO data
        'EK3_POS_I_GATE': 100,    # Position innovation gate (default: 500)
        'EK3_VEL_I_GATE': 100,    # Velocity innovation gate (default: 500)
        
        # Process noise - how much we trust VIO vs prediction
        'EK3_ACC_P_NSE': 0.1,     # Accelerometer process noise (default: 0.35)
        'EK3_GYRO_P_NSE': 0.01,   # Gyroscope process noise (default: 0.015)
        
        # External navigation delay compensation
        'EK3_EAS_I_GATE': 400,    # Airspeed innovation gate
        'EK3_RNG_I_GATE': 100,    # Range finder innovation gate
        
        # VIO specific tuning
        'EK3_VIS_DRFT': 0.02,     # Vision position drift rate (default: 0.1)
        'EK3_GYROFFSET': 1,       # Gyro offset learning (enable)
        
        # Velocity decay for hover
        'EK3_DRAG_MULT': 2.0,     # Drag multiplier for velocity decay (default: 0.0)
    }
    
    success_count = 0
    for param, value in params.items():
        if client.set_parameter(param, value):
            print(f"[+] Set {param} = {value}")
            success_count += 1
        else:
            print(f"[-] Failed to set {param}")
        time.sleep(0.1)
    
    print(f"[*] EKF tuning complete: {success_count}/{len(params)} parameters set")
    return success_count == len(params)

def set_conservative_vio_mode(client):
    """
    Set conservative VIO mode with drift compensation.
    """
    print("[*] Setting conservative VIO mode...")
    
    # Conservative VIO settings
    p1 = client.set_parameter('EK3_SRC1_POSXY', 6.0)  # ExternalNav position
    p2 = client.set_parameter('EK3_SRC1_VELXY', 6.0)  # ExternalNav velocity active (30Hz)
    p3 = client.set_parameter('EK3_SRC1_POSZ', 1.0)   # Barometer altitude
    p4 = client.set_parameter('EK3_SRC1_VELZ', 0.0)   # No Z velocity
    
    # Enable optical flow fallback
    p5 = client.set_parameter('FLOW_TYPE', 0)         # Optical flow disabled (camera-based)
    p6 = client.set_parameter('EK3_FLOW_USE', 0)      # Don't use optical flow
    
    return all([p1, p2, p3, p4, p5, p6])

def main():
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print("Usage: python3 tune_ekf_for_vio.py")
        print("Tunes EKF3 parameters to reduce VIO position drift.")
        return
    
    print("=" * 60)
    print("  EKF3 VIO Drift Reduction Tuner")
    print("=" * 60)
    
    client = CuavMavlinkClient()
    
    print("[*] Connecting to CUAV...")
    for attempt in range(10):
        if client.is_connected:
            break
        time.sleep(1)
        print(f"    Attempt {attempt + 1}/10...")
    
    if not client.is_connected:
        print("[-] Failed to connect to CUAV flight controller!")
        return
    
    print(f"[+] Connected to CUAV on {client.connection_string}")
    
    # Apply EKF tuning
    if tune_ekf_for_stable_vio(client):
        print("[+] EKF tuning successful!")
    else:
        print("[-] Some EKF parameters failed to set")
    
    # Apply VIO mode
    if set_conservative_vio_mode(client):
        print("[+] Conservative VIO mode enabled")
    else:
        print("[-] Failed to set VIO mode parameters")
    
    print("\n[*] Restart flight daemon for changes to take effect:")
    print("    sudo systemctl restart drone_flight")
    print("\n[*] Test with small hovering flights first!")
    
    client.close()

if __name__ == '__main__':
    main()