"""The camera is closed when the node stops, so its watchdog does not do it.

WHAT THE CRASH DUMPS SAID. Every bench run this week opened with
"Device ... has crashed", and it looked like the thing being benched. It was
not: DepthAI reports the PREVIOUS session's dump when it connects, so the
line at the start of a run belongs to the end of the one before.

The dumps are errorId 9001 and the device's own prints name it:

    92.046  WD:1876  TX:89394201     healthy
    93.047  WD:1360  TX:0            the host has gone
    === crashdumpmgr::stackTrace: Collecting...
    WATCHDOG_TIMEOUT=1500ms

The host must speak to the device every 1.5 s. systemd sends SIGTERM;
Python's default handler ends the process outright, with no exception and
therefore no finally; dai.Device is abandoned with its pipeline running; the
watchdog fires and resets the board.

Their uptimes agree - 1197 s, 98 s, 98 s - which are how long the NODE had
been running, not how long the bench had. And they go back to 22 Oct, ten
days before the FeatureTracker work that seemed to be causing them.

It is also why bringing the node back has needed the 03e7:2485 wait: a
device that has just reset itself is re-enumerating while the next process
is already asking for it.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vps_vio"))

fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}   {detail}")
    if not ok:
        fails.append(name)


src = open(os.path.join(os.path.dirname(__file__), "..", "vps_vio",
                        "live_node.py"), encoding="utf-8").read()

print("SIGTERM becomes an exit, so the finally blocks run at all")
check("there is a handler", "def _exit_on_sigterm" in src)
check("and main installs it", "signal.signal(signal.SIGTERM, _exit_on_sigterm)" in src)
check("it raises rather than returns", "raise SystemExit(0)" in src,
      "a handler that returns lets the default behaviour stand")
check("and the loop catches SystemExit", "except SystemExit:" in src,
      "otherwise it escapes before finally has done the work")

print("\nthe finally closes the camera, and does it FIRST")
fin = src[src.index("    finally:\n        node.stop()"):]
fin = fin[:fin.index("\n    dt = time.perf_counter")]
check("the camera is closed", "_close()" in fin)
check("before the file writers",
      fin.index("_close()") < fin.index("fwriter.close()"),
      "a slow disk must not be what keeps the device open")
check("and a source without one is not an error",
      'getattr(src, "close", None)' in fin,
      "replay has no camera to close")

print("\nLiveSource can actually be closed")
check("it has close()", "    def close(self):" in src)
cl = src[src.index("    def close(self):"):]
cl = cl[:cl.index("\n    def ", 10)]
check("it drops its reference first", "dev, self.dev = getattr(self, \"dev\", None), None" in cl,
      "so a second call is a no-op rather than a double close")
check("and it survives a device that never opened", "if dev is None:" in cl)
check("and a close that throws", "except Exception:" in cl)

print("\nthe behaviour, on a stand-in")
import types  # noqa: E402
from live_node import LiveSource  # noqa: E402


class FakeDev:
    def __init__(self):
        self.closed = 0

    def close(self):
        self.closed += 1


s = LiveSource.__new__(LiveSource)
d = FakeDev()
s.dev = d
s.close()
check("close() closes the device", d.closed == 1)
s.close()
check("and twice does not close it twice", d.closed == 1)

s2 = LiveSource.__new__(LiveSource)
s2.close()
check("a source that never opened one is fine", True)


class Angry(FakeDev):
    def close(self):
        raise RuntimeError("usb gone")


s3 = LiveSource.__new__(LiveSource)
s3.dev = Angry()
s3.close()
check("a device that throws on close does not take the node with it", True)

print()
if fails:
    print(f"{len(fails)} FAILED: " + ", ".join(fails))
    sys.exit(1)
print("ALL PASS")
