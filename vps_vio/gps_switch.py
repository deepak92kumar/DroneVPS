#!/usr/bin/env python3
"""Choose by hand which GPS the aircraft flies on, and watch the other one.

Two receivers are present: GPS1 is the real one, GPS2 is the VPS injected over
GPS_INPUT. ArduPilot streams both to the ground station whatever it is flying
on, so the one not in command stays visible and can be compared - which is the
whole point of running them side by side before trusting the new one.

Selection is deliberately manual for now:

    GPS_AUTO_SWITCH = 0    fly the primary, nothing else
    GPS_PRIMARY     = 0    GPS1 (real)  |  1 = GPS2 (VPS)

Automatic failover (GPS_AUTO_SWITCH=1, "use the best") is what this becomes
later. It is left off until the VPS has earned it, because "use the best" means
the aircraft decides, and the decision is made on accuracy numbers we supply
ourselves.

    python3 gps_switch.py                # kaun chala raha hai, dono kahan hain
    python3 gps_switch.py --watch        # live, dono ka fark metre mein
    python3 gps_switch.py --use 2        # VPS par switch (disarmed only)
    python3 gps_switch.py --enable-vps   # GPS2_TYPE=14 + reboot
"""
from __future__ import annotations
import argparse, sys, time
import numpy as np
from pymavlink import mavutil

sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from fc_port import exclusive_fc

PORT, BAUD = "/dev/ttyTHS1", 115200
FIX = {0: "no GPS", 1: "no fix", 2: "2D", 3: "3D", 4: "DGPS", 5: "RTK float", 6: "RTK fixed"}


class FC:
    def __init__(self, require_disarmed=False):
        self.m = mavutil.mavlink_connection(PORT, baud=BAUD)
        if self.m.wait_heartbeat(timeout=20) is None:
            sys.exit("no heartbeat from the flight controller")
        self.armed = bool(self.m.motors_armed())
        if require_disarmed and self.armed:
            sys.exit("*** ARMED - refusing to change a parameter. Disarm first. ***")
        self.m.mav.request_data_stream_send(self.m.target_system,
                                            self.m.target_component,
                                            mavutil.mavlink.MAV_DATA_STREAM_ALL, 5, 1)

    def get(self, name, timeout=4.0):
        for _ in range(3):
            self.m.mav.param_request_read_send(self.m.target_system,
                                               self.m.target_component,
                                               name.encode(), -1)
            t0 = time.time()
            while time.time() - t0 < timeout:
                msg = self.m.recv_match(type="PARAM_VALUE", blocking=True, timeout=1)
                if msg and msg.param_id == name:
                    return msg.param_value
        return None

    def set(self, name, val):
        self.m.mav.param_set_send(self.m.target_system, self.m.target_component,
                                  name.encode(), float(val),
                                  mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
        time.sleep(1.0)
        return self.get(name)

    def reboot(self):
        self.m.mav.command_long_send(self.m.target_system, self.m.target_component,
                                     mavutil.mavlink.MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN,
                                     0, 1, 0, 0, 0, 0, 0, 0)

    def both(self, seconds=3.0):
        """Latest GPS1 and GPS2 reports."""
        out = {}
        t0 = time.time()
        while time.time() - t0 < seconds:
            msg = self.m.recv_match(type=["GPS_RAW_INT", "GPS2_RAW"],
                                    blocking=True, timeout=1)
            if msg:
                out[msg.get_type()] = msg
        return out.get("GPS_RAW_INT"), out.get("GPS2_RAW")


def gap_m(a, b):
    if a is None or b is None or a.fix_type < 2 or b.fix_type < 2:
        return None
    la1, lo1 = a.lat / 1e7, a.lon / 1e7
    la2, lo2 = b.lat / 1e7, b.lon / 1e7
    return float(np.hypot((la1 - la2) * 111_320.0,
                          (lo1 - lo2) * 111_320.0 * np.cos(np.radians(la1))))


def line(tag, msg, active):
    mark = ">>" if active else "  "
    if msg is None:
        return f" {mark} {tag}: no data"
    return (f" {mark} {tag}: {FIX.get(msg.fix_type, msg.fix_type):9s} "
            f"sats {msg.satellites_visible:2d}  hdop {msg.eph/100.0:5.2f}  "
            f"{msg.lat/1e7:10.6f}, {msg.lon/1e7:10.6f}"
            + ("   <-- YAHI CHALA RAHA HAI" if active else ""))


def show(fc):
    auto = fc.get("GPS_AUTO_SWITCH")
    prim = fc.get("GPS_PRIMARY")
    t2 = fc.get("GPS2_TYPE")
    mode = {0: "MANUAL (sirf primary)", 1: "AUTO (best waala)",
            2: "BLEND", 4: "primary agar 3D fix ho"}.get(int(auto or 0), str(auto))
    print(f"mode      : GPS_AUTO_SWITCH={auto:.0f}  ->  {mode}")
    print(f"primary   : GPS_PRIMARY={prim:.0f}  ->  GPS{int(prim or 0)+1}")
    print(f"GPS2_TYPE : {t2:.0f}" + ("  (VPS ke liye 14 chahiye)" if t2 != 14 else "  (MAV - VPS)"))
    print(f"armed     : {fc.armed}\n")
    g1, g2 = fc.both()
    act1 = int(prim or 0) == 0
    print(line("GPS1 (asli)", g1, act1))
    print(line("GPS2 (VPS) ", g2, not act1))
    d = gap_m(g1, g2)
    print(f"\n dono ke beech ka fark: " +
          (f"{d:.1f} m" if d is not None else "cannot be measured (one of them has no fix)"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use", type=int, choices=(1, 2), help="kaun sa GPS chalaye")
    ap.add_argument("--watch", action="store_true", help="live dono dikhao")
    ap.add_argument("--manual", action="store_true", help="GPS_AUTO_SWITCH=0")
    ap.add_argument("--auto", action="store_true", help="GPS_AUTO_SWITCH=1 (best waala)")
    ap.add_argument("--enable-vps", action="store_true", help="GPS2_TYPE=14 + reboot")
    ap.add_argument("--disable-vps", action="store_true", help="GPS2_TYPE=0 + reboot")
    a = ap.parse_args()

    needs_write = any((a.use, a.manual, a.auto, a.enable_vps, a.disable_vps))
    fc = FC(require_disarmed=needs_write)

    if a.enable_vps or a.disable_vps:
        v = 14.0 if a.enable_vps else 0.0
        print(f"GPS2_TYPE -> {fc.set('GPS2_TYPE', v)}")
        print("FC reboot kar raha hoon (GPS driver ke liye zaroori)...")
        fc.reboot()
        print("25 second baad dobara chalao.")
        return

    if a.manual or a.use:
        print(f"GPS_AUTO_SWITCH -> {fc.set('GPS_AUTO_SWITCH', 0)}  (manual)")
    if a.auto:
        print(f"GPS_AUTO_SWITCH -> {fc.set('GPS_AUTO_SWITCH', 1)}  (best waala)")
    if a.use:
        print(f"GPS_PRIMARY -> {fc.set('GPS_PRIMARY', a.use - 1)}  (GPS{a.use})")
    if needs_write:
        print()

    if a.watch:
        prim = int(fc.get("GPS_PRIMARY") or 0)
        print(f"live - Ctrl-C se band.  chal raha hai: GPS{prim+1}\n")
        try:
            while True:
                g1, g2 = fc.both(1.0)
                d = gap_m(g1, g2)
                f1 = "-" if g1 is None else FIX.get(g1.fix_type, "?")
                f2 = "-" if g2 is None else FIX.get(g2.fix_type, "?")
                s1 = 0 if g1 is None else g1.satellites_visible
                s2 = 0 if g2 is None else g2.satellites_visible
                print(f"  GPS1 {f1:9s} sats {s1:2d} | GPS2 {f2:9s} sats {s2:2d} | "
                      f"fark " + (f"{d:7.1f} m" if d is not None else "    -   ")
                      + f" | chal raha hai GPS{prim+1}", flush=True)
        except KeyboardInterrupt:
            print("\nband")
        return

    show(fc)


if __name__ == "__main__":
    with exclusive_fc():
        main()
