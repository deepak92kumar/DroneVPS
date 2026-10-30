#!/usr/bin/env python3
"""Put the parameters back on a hard-reset FC, one at a time, over USB.

Every one is read before, written, and read back. A write that is not read
back is not a write - that is the 23 Oct lesson from gps_rc_switch, and it
matters more here because nobody is watching 131 of them go past.

USB, not the router: telem2 is dead until SERIAL2_BAUD is set, and that is one
of the things being set. The router keeps /dev/ttyTHS1; this takes /dev/ttyACM0
and the two do not meet.

    python3 restore.py                 # dry run, writes nothing
    python3 restore.py --apply
"""
import argparse
import sys
import time

from pymavlink import mavutil

PARM = "/data/DRONE/reports/restore_after_hard_reset.parm"
PORT = "/dev/ttyACM0"

ap = argparse.ArgumentParser()
ap.add_argument("--apply", action="store_true")
ap.add_argument("--port", default=PORT)
ap.add_argument("--only", default=None, help="sirf is prefix wale")
a = ap.parse_args()

want = []
for line in open(PARM):
    line = line.split("#")[0].strip()
    if not line or "," not in line:
        continue
    k, v = line.split(",", 1)
    k = k.strip()
    if a.only and not k.startswith(a.only):
        continue
    want.append((k, float(v)))
print(f"{len(want)} parameters {'LIKHNE' if a.apply else 'DEKHNE (dry run)'} ke liye\n")

m = mavutil.mavlink_connection(a.port, baud=115200,
                               source_system=255, source_component=198)
if m.wait_heartbeat(timeout=15) is None:
    sys.exit("FC se heartbeat nahi aaya")
print(f"FC mila on {a.port}  (sys {m.target_system}, comp {m.target_component})\n")


def get(name, timeout=4.0):
    m.mav.param_request_read_send(m.target_system, m.target_component,
                                  name.encode(), -1)
    t0 = time.time()
    while time.time() - t0 < timeout:
        msg = m.recv_match(type="PARAM_VALUE", blocking=True, timeout=1.0)
        if msg is not None and msg.param_id == name:
            return float(msg.param_value)
    return None


def close(x, y):
    # Parameters come back as 32-bit floats, so an exact compare fails on
    # things like 1.1 and on a latitude it fails on everything.
    return x is not None and abs(x - y) <= max(abs(y) * 1e-6, 1e-6)


same = changed = failed = absent = 0
fails = []
for i, (k, v) in enumerate(want, 1):
    cur = get(k)
    if cur is None:
        print(f"{i:3d}/{len(want)}  {k:<20} FC par yeh param hai hi nahi")
        absent += 1
        continue
    if close(cur, v):
        print(f"{i:3d}/{len(want)}  {k:<20} {cur:<16g} pehle se sahi")
        same += 1
        continue
    if not a.apply:
        print(f"{i:3d}/{len(want)}  {k:<20} {cur:<16g} -> {v:g}   (dry run)")
        changed += 1
        continue
    for attempt in (1, 2, 3):
        m.mav.param_set_send(m.target_system, m.target_component, k.encode(),
                             float(v), mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
        time.sleep(0.6)
        got = get(k)
        if close(got, v):
            print(f"{i:3d}/{len(want)}  {k:<20} {cur:<16g} -> {got:<16g} OK")
            changed += 1
            break
        if attempt < 3:
            print(f"{i:3d}/{len(want)}  {k:<20} wapas {got}, chahiye {v:g}"
                  f" - phir se ({attempt + 1}/3)")
    else:
        print(f"{i:3d}/{len(want)}  {k:<20} FAIL - FC ne nahi maana"
              f" (wapas {got})")
        failed += 1
        fails.append(k)

print(f"\npehle se sahi {same}   {'badle' if a.apply else 'badlenge'} {changed}"
      f"   FC par nahi {absent}   FAIL {failed}")
if fails:
    print("FAIL hue:", ", ".join(fails))
