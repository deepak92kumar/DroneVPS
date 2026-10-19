"""Pin what the watchdog will and will not do.

A watchdog that restarts the wrong thing at the wrong moment is worse than
none, so each refusal is a test. Two matter most:

  * it must not restart while the aircraft is armed, and "cannot tell whether
    it is armed" counts as armed - that is the direction that errs toward not
    touching a flying aircraft
  * it must not cycle a service that will not come back, which is how the
    28 Sep note "depthai can abort during rapid restart cycling" was earned

systemctl and the FC are both stubbed; nothing on this machine is restarted.
"""
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import drone_watchdog as W  # noqa: E402

fails = []
restarts = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        fails.append(name)


class FakeRun:
    """Stands in for subprocess.run, recording restarts instead of doing them."""
    def __init__(self, rc=0):
        self.rc = rc

    def __call__(self, cmd, **kw):
        if "restart" in cmd:
            restarts.append(cmd[-1])

        class R:
            returncode = self.rc
            stderr = "" if self.rc == 0 else "stub failure"
        return R()


def run(*, age=None, active=True, armed=False, when_armed=False,
        restart_rc=0, state=None, extra=()):
    """One watchdog pass against a liveness file of a chosen age."""
    d = tempfile.mkdtemp()
    live = os.path.join(d, "vps.alive")
    if age is not None:
        with open(live, "w") as f:
            f.write("t=12.00 n=300 tracks=287 sigma=6.1 fixes=4 flow=299 vel=ok\n")
        os.utime(live, (time.time() - age, time.time() - age))
    st = state if state is not None else os.path.join(d, "state")
    W.is_active = lambda unit: active
    W.armed_via_router = lambda ep, timeout_s=3.0: armed
    W.subprocess.run = FakeRun(restart_rc)
    argv = ["--once", "--liveness", live, "--state", st,
            "--stale-after", "15", "--min-interval", "60", "--max-restarts", "3"]
    if when_armed:
        argv.append("--restart-when-armed")
    argv += list(extra)
    before = len(restarts)
    W.main(argv)
    return len(restarts) - before, st


print("healthy loop is left alone")
n, _ = run(age=2.0)
check("fresh liveness -> no restart", n == 0)

print("\nstalled loop on the ground is restarted")
n, _ = run(age=40.0)
check("40s stale -> restarted", n == 1, f"{n} restart(s)")
check("the right unit", restarts and restarts[-1] == "drone-vps.service",
      restarts[-1] if restarts else "none")

print("\nbut not while armed, and not when that cannot be determined")
n, _ = run(age=40.0, armed=True)
check("armed -> refused", n == 0)
n, _ = run(age=40.0, armed=None)
check("armed state unknown -> refused", n == 0)
n, _ = run(age=40.0, armed=True, when_armed=True)
check("--restart-when-armed overrides", n == 1, f"{n} restart(s)")

print("\nnot when the unit is already down - that is systemd's job")
n, _ = run(age=40.0, active=False)
check("inactive unit -> left to systemd", n == 0)

print("\nnot on a missing liveness file - the node may be starting")
n, _ = run(age=None)
check("no liveness file -> refused", n == 0)

print("\nrate limit holds across timer runs")
d = tempfile.mkdtemp()
st = os.path.join(d, "state")
n1, _ = run(age=40.0, state=st)
n2, _ = run(age=40.0, state=st)
check("first pass restarts", n1 == 1, f"{n1}")
check("second pass within min-interval refuses", n2 == 0, f"{n2}")

print("\nconsecutive restarts are capped")
d = tempfile.mkdtemp()
st = os.path.join(d, "state")
with open(st, "w") as f:
    f.write(f"3 {time.time() - 999}")      # already at the cap, interval clear
n, _ = run(age=40.0, state=st)
check("at max-restarts -> stops trying", n == 0, f"{n}")

print("\nrecovery clears the counter")
d = tempfile.mkdtemp()
st = os.path.join(d, "state")
with open(st, "w") as f:
    f.write(f"2 {time.time() - 999}")
run(age=2.0, state=st)                      # healthy again
with open(st) as f:
    after = f.read().split()
check("counter reset on recovery", after[0] == "0", f"state now {after}")

print("\na failing restart is reported, not silently counted as success")
d = tempfile.mkdtemp()
st = os.path.join(d, "state")
n, _ = run(age=40.0, state=st, restart_rc=1)
check("attempted", n == 1, f"{n}")
with open(st) as f:
    check("and still counted, so the cap applies", f.read().split()[0] == "1", "")

print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
