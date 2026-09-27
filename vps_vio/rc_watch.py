#!/usr/bin/env python3
"""Show which RC channel moves when you flip a switch.

Prints every channel that changes, live, so a switch can be identified by
flipping it and reading what appears. Nothing is written to the aircraft.
"""
from __future__ import annotations
import sys, time, collections
from pymavlink import mavutil


def main():
    m = mavutil.mavlink_connection("/dev/ttyTHS1", baud=115200)
    if m.wait_heartbeat(timeout=20) is None:
        sys.exit("FC se heartbeat nahi aaya")
    m.mav.request_data_stream_send(m.target_system, m.target_component,
                                   mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1)
    print("Switch hilao - jo channel badlega wo yahan dikhega.")
    print("Ctrl-C se band karo, phir positions ka summary aayega.\n")
    base = None
    seen: dict[int, collections.Counter] = {}
    last: dict[int, int] = {}
    try:
        while True:
            msg = m.recv_match(type="RC_CHANNELS", blocking=True, timeout=2)
            if msg is None:
                print("  (RC signal nahi aa raha - remote on hai?)", flush=True)
                continue
            vals = {i: getattr(msg, f"chan{i}_raw", 0) for i in range(1, 17)}
            if base is None:
                base = dict(vals)
                last = dict(vals)
                print("  shuruaati values yaad kar li. Ab hilao.\n", flush=True)
                continue
            for i, v in vals.items():
                if abs(v - last[i]) > 40:
                    print(f"  ch{i:<2d} : {last[i]:4d} -> {v:4d}", flush=True)
                    last[i] = v
                    seen.setdefault(i, collections.Counter())
                if i in seen:
                    seen[i][v] += 1
    except KeyboardInterrupt:
        pass
    print("\n=== jo channel hile ===")
    if not seen:
        print("  koi nahi")
    for i in sorted(seen):
        pos = sorted(v for v, n in seen[i].items() if n > 3)
        print(f"  ch{i:<2d} : {len(pos)} position -> {pos}")


if __name__ == "__main__":
    main()
