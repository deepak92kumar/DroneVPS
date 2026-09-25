#!/usr/bin/env python3
"""Phase 0 — VIO bench evaluation on the OAK-D.  BENCH ONLY. NO FLIGHT. NO FC.

What it does:
    1. Refuses to run while the live drone-vio daemon is active (it owns the OAK-D).
    2. Starts a VIO session on the OAK-D via the SpectacularAI DepthAI plugin.
    3. Logs pose (x,y,z, quaternion, timestamp) to a CSV.
    4. Prints the achieved output rate and pose-timestamp vs wall-clock latency —
       the first thing we need to know is whether the camera-IMU timing is sane
       (Phase 0 exit criterion in VPS_Upgrade_Plan.pdf).

This does NOT touch flight_daemon.py, cuav_mavlink_bridge.py, the FC, or any
parameter. It only reads the camera. Feeding pose to the FC is Phase 1, and will
go through cuav_mavlink_bridge.CuavMavlinkClient.send_vision_estimate() using the
vio_frame.VioToNed transform — not from here.

Run (after the demo, daemon stopped):
    sudo systemctl stop drone-vio
    python3 vps_vio/phase0_vio_eval.py --seconds 120 --out /mnt/ssd/DRONE/vps_vio/logs/vio_eval.csv
"""
from __future__ import annotations
import argparse
import csv
import os
import subprocess
import sys
import time


def daemon_is_active() -> bool:
    try:
        r = subprocess.run(["systemctl", "is-active", "drone-vio"],
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip() == "active"
    except Exception:
        return False  # if systemctl is unavailable, don't block; user is driving


def require_free_camera(force: bool) -> None:
    if daemon_is_active() and not force:
        print("REFUSING TO RUN: the drone-vio daemon is active and owns the OAK-D.")
        print("This is bench-only Phase-0 code. After the demo, stop it first:")
        print("    sudo systemctl stop drone-vio")
        print("Then re-run. (Use --force only if you are certain the camera is free.)")
        sys.exit(2)


def import_vio():
    try:
        import depthai  # noqa: F401
    except Exception as e:
        sys.exit(f"depthai import failed: {e}\n(depthai 2.x should already be installed.)")
    try:
        import spectacularAI  # noqa: F401
        import spectacularAI.depthai  # noqa: F401
        return "spectacularAI"
    except Exception:
        print("SpectacularAI SDK not installed. It is the fastest VIO path on OAK-D.")
        print("Install (Phase 0):  python3 -m pip install spectacularAI")
        print("Docs: https://spectacularai.github.io/docs/sdk/  (verify licence/version)")
        print("Fallback path if this is not viable: OpenVINS / VINS-Fusion (Phase 0 backup).")
        sys.exit(3)


def run(seconds: float, out_path: str, force: bool) -> None:
    require_free_camera(force)
    import_vio()
    import depthai
    import spectacularAI.depthai

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pipeline = depthai.Pipeline()
    vio_pipeline = spectacularAI.depthai.Pipeline(pipeline)

    n = 0
    t_wall0 = time.monotonic()
    lat_sum = 0.0
    with depthai.Device(pipeline) as device, open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["wall_s", "vio_time_s", "x", "y", "z", "qw", "qx", "qy", "qz", "tracking"])
        session = vio_pipeline.startSession(device)
        print(f"VIO session started. Logging {seconds:.0f}s to {out_path}")
        print("Move the aircraft in KNOWN directions (fwd/back, left/right) to later")
        print("measure the VIO->NED axis mapping (the placeholder R in vio_frame.py).")
        try:
            while time.monotonic() - t_wall0 < seconds:
                out = session.waitForOutput()
                p = out.pose
                wall = time.monotonic()
                # pose position (metres) + orientation quaternion in VIO world frame
                pos = p.position
                q = p.orientation
                status = getattr(out, "status", "")
                w.writerow([f"{wall - t_wall0:.4f}", f"{p.time:.4f}",
                            f"{pos.x:.5f}", f"{pos.y:.5f}", f"{pos.z:.5f}",
                            f"{q.w:.6f}", f"{q.x:.6f}", f"{q.y:.6f}", f"{q.z:.6f}",
                            str(status)])
                n += 1
                lat_sum += abs((wall - t_wall0) - p.time)
                if n % 100 == 0:
                    dt = wall - t_wall0
                    print(f"  {n:6d} poses  {n/dt:5.1f} Hz  mean|wall-vio|={lat_sum/n*1e3:5.1f} ms")
        finally:
            dt = max(time.monotonic() - t_wall0, 1e-6)
            print(f"\nDone. {n} poses, {n/dt:.1f} Hz average.")
            if n:
                print(f"Mean |wall - vio_time| = {lat_sum/n*1e3:.1f} ms")
            print("Phase-0 exit check: rate should be steady (>=20 Hz) and latency low/stable.")
            print("Next: plot the CSV against your known moves to fix vio_frame.R, then Phase 1.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phase 0 VIO bench eval (OAK-D). Bench only.")
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "logs", "vio_eval.csv"))
    ap.add_argument("--force", action="store_true", help="run even if the daemon looks active")
    a = ap.parse_args()
    run(a.seconds, a.out, a.force)
