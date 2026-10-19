"""Restart the VPS node when it stops looping, not merely when it dies.

systemd's Restart= only catches a process that EXITS. The failure that matters
here is different: the node stays "active" while its loop is stuck - waiting on
a camera frame that never arrives, or on a device that aborted underneath it -
and systemd is perfectly happy. The 28 Sep report records the watchdog being
extended to cover the VPS itself through a liveness file for exactly this.

So the node writes a liveness file every loop and this checks its age.

TWO THINGS ARE DELIBERATE AND BOTH ARE ARGUABLE - say so rather than bury them.

1. By default it does NOT restart while the aircraft is armed.
   A stuck node in flight means GPS_INPUT stops arriving, and ArduPilot then
   marks GPS2 unhealthy and GPS_AUTO_SWITCH moves the aircraft onto the real
   receiver by itself - the FC has its own fallback. A restart in flight has no
   such grace: it guarantees a gap in the position source, and the same report
   notes "depthai can abort during rapid restart cycling", so the restart
   itself can fail and leave nothing. On the ground none of that applies.
   --restart-when-armed overrides this if you disagree; it is a real choice.

2. Restarts are rate limited and capped.
   Cycling a service that cannot start is how the depthai abort was reached.
   After --max-restarts consecutive failures it stops trying and keeps saying
   so in the log, because a loud stuck watchdog is easier to diagnose than a
   silent restart loop.

Nothing here writes to the aircraft.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time


def liveness_age(path: str) -> float | None:
    try:
        return time.time() - os.path.getmtime(path)
    except OSError:
        return None


def liveness_text(path: str) -> str:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return "(unreadable)"


def is_active(unit: str) -> bool:
    return subprocess.run(["systemctl", "is-active", "--quiet", unit]).returncode == 0


def armed_via_router(endpoint: str, timeout_s: float = 3.0) -> bool | None:
    """Ask the router, read-only, whether the FC reports itself armed.

    None means we could not tell - and "could not tell" must never be treated
    as disarmed, because that is the direction that restarts a node in flight.
    """
    try:
        from pymavlink import mavutil
    except Exception:
        return None
    try:
        m = mavutil.mavlink_connection(endpoint, source_system=255,
                                       source_component=191)
        hb = m.recv_match(type="HEARTBEAT", blocking=True, timeout=timeout_s)
        if hb is None:
            return None
        return bool(hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
    except Exception:
        return None
    finally:
        try:
            m.close()
        except Exception:
            pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--unit", default="drone-vps.service",
                    help="the unit to restart when its loop stalls")
    ap.add_argument("--liveness", default="/run/drone/vps.alive")
    ap.add_argument("--stale-after", type=float, default=15.0,
                    help="seconds without a loop before the node counts as stuck")
    ap.add_argument("--min-interval", type=float, default=60.0,
                    help="never restart more often than this")
    ap.add_argument("--max-restarts", type=int, default=3,
                    help="give up after this many consecutive restarts")
    ap.add_argument("--router", default="udpout:127.0.0.1:14650",
                    help="read-only endpoint used only to check armed state")
    ap.add_argument("--restart-when-armed", action="store_true",
                    help="restart even in flight. Read the module docstring first.")
    ap.add_argument("--state", default="/run/drone/watchdog.state",
                    help="where the restart counter is kept across timer runs")
    ap.add_argument("--once", action="store_true",
                    help="one check and exit, which is how the .timer runs it")
    ap.add_argument("--interval", type=float, default=10.0,
                    help="loop period when not --once")
    a = ap.parse_args(argv)

    state_path = a.state
    restarts, last_restart = 0, 0.0
    try:
        with open(state_path) as f:
            restarts, last_restart = (float(x) for x in f.read().split())
            restarts = int(restarts)
    except Exception:
        pass

    def save():
        try:
            os.makedirs(os.path.dirname(state_path), exist_ok=True)
            with open(state_path, "w") as f:
                f.write(f"{restarts} {last_restart}")
        except OSError:
            pass

    def check() -> None:
        nonlocal restarts, last_restart
        if not is_active(a.unit):
            # Not running at all: systemd's own Restart= owns this case, and
            # stepping on it would fight the thing that is already handling it.
            print(f"{a.unit} is not active - leaving it to systemd", flush=True)
            return
        age = liveness_age(a.liveness)
        if age is None:
            print(f"no liveness file at {a.liveness} yet - the node may still "
                  f"be starting; not restarting on a missing file", flush=True)
            return
        if age <= a.stale_after:
            if restarts:
                print(f"loop healthy again ({age:.1f}s) - clearing "
                      f"{restarts} restart(s)", flush=True)
                restarts = 0
                save()
            return

        detail = liveness_text(a.liveness)
        print(f"STALLED: {a.liveness} is {age:.1f}s old "
              f"(limit {a.stale_after}). Last state: {detail}", flush=True)

        if restarts >= a.max_restarts:
            print(f"already restarted {restarts} times with no recovery - "
                  f"NOT restarting again. This needs a person.", flush=True)
            return
        now = time.time()
        if now - last_restart < a.min_interval:
            print(f"last restart was {now-last_restart:.0f}s ago "
                  f"(min {a.min_interval:.0f}) - waiting", flush=True)
            return

        armed = armed_via_router(a.router)
        if armed is None and not a.restart_when_armed:
            print("could not determine armed state - treating as ARMED and "
                  "NOT restarting", flush=True)
            return
        if armed and not a.restart_when_armed:
            print("aircraft is ARMED - not restarting. The FC drops an "
                  "unhealthy GPS2 by itself; a restart here would only "
                  "guarantee a gap.", flush=True)
            return

        print(f"restarting {a.unit}", flush=True)
        r = subprocess.run(["systemctl", "restart", a.unit],
                           capture_output=True, text=True)
        restarts += 1
        last_restart = now
        save()
        if r.returncode != 0:
            print(f"restart FAILED: {r.stderr.strip()}", flush=True)

    if a.once:
        check()
        return 0
    while True:
        check()
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
