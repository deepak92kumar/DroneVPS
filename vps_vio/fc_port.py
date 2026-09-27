"""One serial line to the flight controller, and several things that want it.

/dev/ttyTHS1 is a single port. The recorder holds it continuously, so any tool
that opens it as well gets "device reports readiness to read but returned no
data (device disconnected or multiple access on port?)" - two readers pulling
bytes out of the same stream, each seeing half a MAVLink frame.

The proper fix is a MAVLink router owning the port and handing out UDP
endpoints. That sits in the flight-critical path, so it is not something to add
casually at night. Until then this pauses the recorder for the duration of a
ground tool and puts it back, which is safe because these tools only run on the
ground and the recorder only records while armed anyway.
"""
from __future__ import annotations
import contextlib
import subprocess
import time

SERVICES = ("drone-record", "drone-record-watchdog.timer")


def _active(name: str) -> bool:
    return subprocess.run(["systemctl", "is-active", "--quiet", name]).returncode == 0


@contextlib.contextmanager
def exclusive_fc(quiet: bool = False):
    """Hold the FC port alone, then restore whatever was running before."""
    paused = [s for s in SERVICES if _active(s)]
    if paused:
        if not quiet:
            print(f"  (recording rok raha hoon: {', '.join(paused)})", flush=True)
        subprocess.run(["sudo", "systemctl", "stop", *paused],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2.0)
    try:
        yield
    finally:
        if paused:
            subprocess.run(["sudo", "systemctl", "start", *paused],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if not quiet:
                print(f"\n  (recording wapas chalu: {', '.join(paused)})", flush=True)
