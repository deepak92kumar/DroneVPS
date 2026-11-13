#!/usr/bin/env python3
"""Stop and start the node from Studio, and report what the new run loaded.

A map change only reaches the node at startup. `live_node.py` builds its
MapCache once in main() and never reopens it, so moving active.tif while the
node runs changes nothing in the air: Studio shows the new sheet and the node
keeps flying the old one. That is exactly what happened on 13 Nov - the
symlink moved at 13:20, the node had started at 13:07, and for forty minutes
the honest answer to "is the new map loaded" was no.

The button that moves the symlink now has the restart beside it.

`systemctl restart` is deliberately NOT used. It brings the new process up
before the OAK-D has re-enumerated and the camera never opens. The sequence
is stop, wait for 03e7:2485 to come back, start - the same three steps that
were being run by hand.

The new run is read back from a byte offset taken BEFORE the stop, so what is
reported is this run and never a stale line from an earlier one. Reading the
whole file and taking the last match is not good enough: the log carries NUL
bytes, grep stops printing at the first one, and the "last" line it prints
can be an hour old.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time

UNIT = "drone-vps"
# The OAK-D answers to TWO ids and which one it shows is the whole point. It
# is 03e7:2485 (MyriadX, unbooted) while nothing holds it, and the node's own
# firmware upload re-enumerates it as 03e7:f63b (Myriad VPU) - on a different
# bus number. So "the camera is back and free to open" is 2485 specifically,
# while "a camera exists at all" is either. Waiting on the wrong one waits
# for ever.
CAMERA_FREE_ID = "03e7:2485"
CAMERA_BOOTED_ID = "03e7:f63b"
FALLBACK_LOG = "/data/DRONE/flight_logs/vps_stdout.log"

# Budgets. The whole call is bounded so the browser cannot be left hanging.
STOP_S = 15.0
CAMERA_S = 25.0
SETTLE_S = 45.0


def _sh(args, timeout=20.0):
    """Run a command, never raise. Returns (rc, combined output)."""
    try:
        p = subprocess.run(args, capture_output=True, text=True,
                           timeout=timeout)
        return p.returncode, (p.stdout + p.stderr).strip()
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout:.0f} s"
    except Exception as e:                           # noqa: BLE001
        return 125, str(e)


def _systemctl(*args, timeout=20.0):
    # sudoers carries NOPASSWD for `systemctl <verb> drone-*`, so this needs
    # no password and no sudoers change. If that rule is ever removed the
    # command fails cleanly and the page says so instead of hanging.
    return _sh(["sudo", "-n", "/usr/bin/systemctl", *args], timeout=timeout)


def log_path() -> str:
    """Where the unit appends its stdout, asked of the unit itself."""
    rc, out = _systemctl("show", "-p", "StandardOutput", "--value", UNIT)
    if rc == 0 and out.startswith("append:"):
        return out.split(":", 1)[1].strip()
    return FALLBACK_LOG


def is_active() -> bool:
    return _systemctl("is-active", UNIT)[0] == 0


def _lsusb(dev_id: str) -> bool:
    if not shutil.which("lsusb"):
        return True              # cannot look; do not block the restart
    return _sh(["lsusb", "-d", dev_id], timeout=5.0)[0] == 0


def camera_free() -> bool:
    """Unbooted and available - what must be true before the node starts."""
    return _lsusb(CAMERA_FREE_ID)


def camera_present() -> bool:
    """Plugged in at all, booted or not."""
    return camera_free() or _lsusb(CAMERA_BOOTED_ID)


def _size(path) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


_MAP = re.compile(r"^map (\d+)x(\d+) @ (.+)$")
_ORIGIN = re.compile(r"^origin from (.+?): ([-\d.]+), ([-\d.]+)")
_COVERS = re.compile(r"sheet covers this point: (YES|NO)")
_RECORDING = re.compile(r"^recording to (.+)$")


def read_run(path: str, offset: int) -> dict:
    """What the run that starts at `offset` has told us so far."""
    out = {"map": None, "map_px": None, "origin": None, "origin_from": None,
           "covered": None, "run": None, "no_origin": None, "lines": 0}
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            text = f.read().decode("utf-8", "replace")
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip("\x00").strip()
        if not line:
            continue
        out["lines"] += 1
        m = _MAP.match(line)
        if m:
            out["map_px"] = f"{m.group(1)}x{m.group(2)}"
            out["map"] = line
            continue
        m = _ORIGIN.match(line)
        if m:
            out["origin_from"] = m.group(1)
            out["origin"] = [float(m.group(3)), float(m.group(2))]  # lon, lat
            continue
        m = _COVERS.search(line)
        if m:
            out["covered"] = (m.group(1) == "YES")
            continue
        m = _RECORDING.match(line)
        if m:
            out["run"] = os.path.basename(os.path.dirname(m.group(1)))
            continue
        if line.startswith("NO ORIGIN:"):
            out["no_origin"] = line
    return out


def _settled(st: dict) -> bool:
    # The map line alone is not enough to fly on: a node with no origin
    # publishes nothing at all. Wait until the origin question is answered
    # one way or the other.
    return bool(st["map_px"]) and (st["origin"] or st["no_origin"])


def restart() -> dict:
    """stop -> wait for the camera -> start -> read the new run back."""
    path = log_path()
    offset = _size(path)          # before anything, so we read only this run
    steps = []
    t0 = time.time()

    def step(name, ok, detail=""):
        steps.append({"name": name, "ok": bool(ok), "detail": detail})

    rc, out = _systemctl("stop", UNIT, timeout=STOP_S + 5)
    if rc != 0:
        step("stop", False, out or f"systemctl stop exited {rc}")
        return _verdict(steps, {}, path, offset, t0)
    deadline = time.time() + STOP_S
    while is_active() and time.time() < deadline:
        time.sleep(0.5)
    stopped = not is_active()
    step("stop", stopped,
         "stopped" if stopped else f"still active after {STOP_S:.0f} s")
    if not stopped:
        return _verdict(steps, {}, path, offset, t0)

    # The OAK-D detaches on exit and comes back a second or two later. Start
    # before it does and the camera never opens - that is what makes
    # `systemctl restart` unusable here.
    deadline = time.time() + CAMERA_S
    time.sleep(1.0)
    while not camera_free() and time.time() < deadline:
        time.sleep(0.5)
    cam = camera_free()
    step("camera", cam,
         f"{CAMERA_FREE_ID} back and free" if cam
         else f"{CAMERA_FREE_ID} did not come back in {CAMERA_S:.0f} s - "
              f"replug the camera, do not fly")
    if not cam:
        return _verdict(steps, {}, path, offset, t0)

    rc, out = _systemctl("start", UNIT, timeout=30.0)
    step("start", rc == 0, out or ("started" if rc == 0 else f"exited {rc}"))
    if rc != 0:
        return _verdict(steps, {}, path, offset, t0)

    deadline = time.time() + SETTLE_S
    st = {}
    while time.time() < deadline:
        st = read_run(path, offset)
        if _settled(st):
            break
        time.sleep(1.0)
    else:
        st = read_run(path, offset)
    step("loaded", bool(st.get("map_px")),
         st.get("map") or f"no map line in {SETTLE_S:.0f} s - "
                          f"check {os.path.basename(path)}")
    return _verdict(steps, st, path, offset, t0)


def _verdict(steps, st, path, offset, t0) -> dict:
    """One sentence the pilot can act on, and the detail behind it."""
    ok = all(s["ok"] for s in steps) and bool(st.get("map_px"))
    flyable = bool(ok and st.get("origin") and st.get("covered"))
    if not steps or not steps[-1]["ok"]:
        verdict = "RESTART FAILED - " + (steps[-1]["detail"] if steps
                                         else "nothing ran")
    elif not st.get("map_px"):
        verdict = "node started but printed no map - do not fly"
    elif st.get("no_origin"):
        verdict = ("map loaded, but the node has NO ORIGIN and will publish "
                   "nothing. Power the FC, then restart again.")
    elif not st.get("origin"):
        verdict = "map loaded; still waiting for an origin"
    elif st.get("covered") is False:
        verdict = ("map loaded and origin taken, but this sheet does NOT "
                   "cover the take-off point - there will be no map fixes")
    else:
        verdict = f"ready - {st['map_px']} loaded, origin taken, sheet covers it"
    return {"ok": ok, "flyable": flyable, "verdict": verdict, "steps": steps,
            "state": st, "log": path, "offset": offset,
            "seconds": round(time.time() - t0, 1)}


def state() -> dict:
    """What the node is running right now, without touching it.

    The offset is the last map line's own position, so this reads the current
    run and not the whole history.
    """
    path = log_path()
    try:
        with open(path, "rb") as f:
            blob = f.read()
    except OSError:
        return {"active": is_active(), "log": path, "state": {}}
    hits = [m.start() for m in re.finditer(rb"(?m)^map \d+x\d+ @", blob)]
    off = hits[-1] if hits else 0
    return {"active": is_active(), "camera": camera_present(),
            "log": path, "offset": off, "state": read_run(path, off)}
