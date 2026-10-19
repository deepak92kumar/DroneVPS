"""Only one VPS node, one router, one switch - whoever started them.

The guard this replaces asked systemctl whether a unit called "drone-vio" was
active. No such unit exists; the units are drone-router, drone-vps,
drone-rc-switch and drone-watchdog. So it was always inactive, the guard never
fired once, and it read like a safety check while being nothing at all.

The pilot asked for this directly: "ye nahi ki 2-2 services chal gaye aur alt
hold le liya". That is the right thing to worry about. The OAK-D opens
exclusively, so a second copy cannot have the camera; with Restart=always in
front of it the loser fails, restarts five seconds later and does that for the
whole flight. And if the SERVICE is the loser, every one of those loops is a
hole in the GPS_INPUT stream - which is what takes the EKF's position away and
drops the aircraft into ALT_HOLD. Log 214 has exactly that at 8.6 s.
"""
import os
import subprocess
import sys
import textwrap

VPS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vps_vio")
sys.path.insert(0, VPS)
import single_instance as si  # noqa: E402

fails = []
NAME = "test_single_copy_probe"


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        fails.append(name)


def in_another_process(name, hold=False):
    """Try to take the lock in a separate process; returns (exit_code, stderr)."""
    code = textwrap.dedent(f"""
        import sys; sys.path.insert(0, {VPS!r})
        import single_instance as si
        fd = si.require({name!r})
        print("GOT", flush=True)
        {"import time; time.sleep(30)" if hold else ""}
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, timeout=60)
    return r.returncode, r.stdout + r.stderr


print("the first copy gets it")
rc, out = in_another_process(NAME)
check("a free lock is taken", rc == 0 and "GOT" in out, f"rc={rc}")

print("\nthe second copy is refused, and says who has it")
fd = si.acquire(NAME)
check("this process now holds it", fd is not None)
rc, out = in_another_process(NAME)
check("the second copy exits", rc != 0, f"rc={rc}")
check("with EXIT_BUSY, not a generic failure", rc == si.EXIT_BUSY,
      f"{rc} vs {si.EXIT_BUSY}")
check("it names the holder's pid", str(os.getpid()) in out, out.strip()[:80])
check("and says how to stop it", "systemctl stop" in out or "kill" in out)

print("\nthe lock dies with its holder, however it dies")
# A PID file would survive a kill -9 and lock the next start out for ever.
code = textwrap.dedent(f"""
    import sys, os, time; sys.path.insert(0, {VPS!r})
    import single_instance as si
    si.require({NAME + "_kill"!r})
    print(os.getpid(), flush=True)
    time.sleep(60)
""")
p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
pid = int(p.stdout.readline().strip())
rc, _ = in_another_process(NAME + "_kill")
check("while it lives, nobody else gets in", rc == si.EXIT_BUSY, f"rc={rc}")
p.kill()
p.wait(timeout=20)
rc, out = in_another_process(NAME + "_kill")
check("after a kill -9 the next start gets in", rc == 0 and "GOT" in out,
      f"rc={rc} - a stale lock would have blocked the aircraft's own service")

print("\neach program has its own lock")
other = si.acquire(NAME + "_other")
check("a different name is a different lock", other is not None)
if other:
    os.close(other)

print("\nthe real names are the ones the units use")
units = os.path.join(os.path.dirname(VPS), "systemd")
if os.path.isdir(units):
    import re
    for unit, name in (("drone-vps.service", "drone-vps"),
                       ("drone-router.service", "drone-router"),
                       ("drone-rc-switch.service", "drone-rc-switch")):
        up = os.path.join(units, unit)
        if not os.path.exists(up):
            continue
        text = open(up).read()
        m = re.search(r"^ExecStart=(.*?)(?=^\w+=|\Z)", text, re.M | re.S)
        script = next((w for w in " ".join(m.group(1).split()).split()
                       if w.endswith(".py")), None) if m else None
        if script and os.path.exists(script):
            src = open(script).read()
            check(f"{unit} takes the {name!r} lock",
                  f'"{name}"' in src or f"'{name}'" in src,
                  "the unit and the program must agree on the name")
        # And the unit must not loop on the busy exit.
        check(f"{unit} does not restart on EXIT_BUSY",
              "RestartPreventExitStatus=3" in text or "Restart=always" not in text)

if fd:
    os.close(fd)
print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
