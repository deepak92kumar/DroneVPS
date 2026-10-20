"""Do the service files actually start the programs they name.

This exists because of what 20 Oct found in drone-vps.service: it passed
--gps2, a flag live_node has never had. systemd started it, argparse refused
it, the process died in under a second, and Restart=always did that again five
seconds later, for ever. The service could not have worked on ANY boot - and
nothing would have said so until the aircraft was on the ground somewhere with
the companion rebooted and no VPS.

It was also missing --map, --fc-port and --send-gps, so even with the flag
fixed it would have matched nothing against nothing and sent it nowhere.

A unit file is not covered by any other test here: it is configuration, it is
only exercised at boot, and it fails silently into a restart loop. So every
flag in every ExecStart is checked against the program's own --help.
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNITS = os.path.join(HERE, "systemd")

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        fails.append(name)


def exec_start(text):
    """The ExecStart line, with systemd's backslash continuations joined."""
    m = re.search(r"^ExecStart=(.*?)(?=^\w+=|\Z)", text, re.M | re.S)
    if not m:
        return None
    return " ".join(m.group(1).replace("\\\n", " ").split())


def help_text(script):
    r = subprocess.run([sys.executable, script, "--help"],
                       capture_output=True, text=True, timeout=120)
    return r.stdout + r.stderr


if not os.path.isdir(UNITS):
    print(f"SKIP: {UNITS} not present")
    sys.exit(0)

units = sorted(f for f in os.listdir(UNITS) if f.endswith(".service"))
check("there are units to check", bool(units), str(units))

for u in units:
    text = open(os.path.join(UNITS, u)).read()
    cmd = exec_start(text)
    print(f"\n{u}")
    if cmd is None:
        check(f"{u} has an ExecStart", False)
        continue
    parts = cmd.split()
    script = next((p for p in parts if p.endswith(".py")), None)
    check(f"{u} names a python script", script is not None, cmd[:70])
    if script is None:
        continue
    check(f"{u}: {os.path.basename(script)} exists", os.path.exists(script), script)
    if not os.path.exists(script):
        continue
    check(f"{u} points inside /data/DRONE", script.startswith("/data/DRONE"), script)

    h = help_text(script)
    used = [p for p in parts if p.startswith("--")]
    unknown = [f for f in used if f.split("=")[0] not in h]
    check(f"{u}: every flag is real", not unknown,
          f"unknown: {unknown}" if unknown else f"{len(used)} flags")

    # A flag that needs a value and did not get one is just as fatal at boot
    # as a flag that does not exist. Appending --help and checking the exit
    # code does NOT catch that - argparse acts on --help before it validates
    # anything else, so it exits 0 no matter what else is on the line. The
    # help text says which flags take a value (argparse prints a metavar after
    # them), so check each one is followed by something that is not a flag.
    needs_value = set(re.findall(r"(--[a-z0-9-]+) [A-Z][A-Z0-9_]*", h))
    missing = []
    for i, tok in enumerate(parts):
        if tok in needs_value and "=" not in tok:
            nxt = parts[i + 1] if i + 1 < len(parts) else None
            if nxt is None or nxt.startswith("--"):
                missing.append(tok)
    check(f"{u}: every flag that needs a value has one", not missing,
          f"bare: {missing}" if missing else f"{len(needs_value & set(used))} valued")

    wd = re.search(r"^WorkingDirectory=(.+)$", text, re.M)
    if wd:
        check(f"{u}: WorkingDirectory exists", os.path.isdir(wd.group(1).strip()),
              wd.group(1))

print("\nrestart behaviour")
for u in units:
    text = open(os.path.join(UNITS, u)).read()
    if "Restart=always" in text:
        # single_instance exits 3 when another copy holds the lock. Restarting
        # on that is a loop that achieves nothing.
        check(f"{u} stands down on exit 3",
              "RestartPreventExitStatus=3" in text,
              "RestartPreventExitStatus missing")

print("\nno unit is ordered after the target that pulls it in")
# 20 Oct, after a real reboot in preparation for flying: the router came up
# and nothing else did. No failed unit, no error, no journal entry for the
# services at all - systemd had simply deleted their start jobs:
#
#   multi-user.target: Found ordering cycle on drone-vps.service/start
#   multi-user.target: Found dependency on drone-router.service/start
#   Job drone-vps.service/start deleted to break ordering cycle
#
# drone-router was WantedBy=multi-user.target AND After=multi-user.target. The
# target pulls in drone-vps, drone-vps is After the router, the router is
# After the target: a loop, which systemd breaks by dropping jobs. It dropped
# the two that matter. A unit must never be ordered after a target that wants
# it, and nothing else in this suite would have noticed.
for u in units:
    text = open(os.path.join(UNITS, u)).read()
    after = set()
    for line in re.findall(r"^After=(.+)$", text, re.M):
        after.update(line.split())
    wanted = set()
    for line in re.findall(r"^WantedBy=(.+)$", text, re.M):
        wanted.update(line.split())
    clash = after & wanted
    check(f"{u}: not After= a target it is WantedBy=", not clash,
          f"{clash} - this is an ordering cycle, systemd deletes the job")

print("\nunits that live on the NVMe wait for it to be mounted")
for u in units:
    text = open(os.path.join(UNITS, u)).read()
    m = re.search(r"^ExecStart=(.*?)(?=^\w+=|\Z)", text, re.M | re.S)
    cmd = " ".join(m.group(1).split()) if m else ""
    if "/data/DRONE" in cmd:
        check(f"{u} has RequiresMountsFor", "RequiresMountsFor=" in text,
              "it can otherwise start before /data is mounted")

print("\nsystemd itself is happy with them")
import shutil  # noqa: E402
if shutil.which("systemd-analyze"):
    for u in units:
        r = subprocess.run(["systemd-analyze", "verify",
                            os.path.join(UNITS, u)],
                           capture_output=True, text=True, timeout=120)
        # verify walks the whole dependency tree and will happily report on
        # unrelated units already installed on this machine. Only complaints
        # about THIS unit are ours to fix.
        bad = [l for l in (r.stderr or "").splitlines()
               if u in l and "Unknown" not in l and "ignoring" not in l.lower()]
        check(f"{u}: systemd-analyze verify is clean", not bad,
              (bad or [""])[0][:90])

print("\nthe ones that must survive a reboot say so")
for u in ("drone-router.service", "drone-vps.service"):
    p = os.path.join(UNITS, u)
    if os.path.exists(p):
        check(f"{u} has an [Install] target",
              "WantedBy=" in open(p).read(), "no WantedBy - enable would do nothing")

print("\nthe VPS unit sends what it is for")
p = os.path.join(UNITS, "drone-vps.service")
if os.path.exists(p):
    cmd = exec_start(open(p).read()) or ""
    for need, why in (("--send-gps", "without it nothing reaches the FC"),
                      ("--map", "without it there are no map fixes"),
                      ("--fc-port", "without it it opens the serial port the router owns"),
                      ("--live", "without it there is no camera")):
        check(f"drone-vps passes {need}", need in cmd, why)
    check("drone-vps goes through the router, not the serial device",
          "udpout:" in cmd, cmd[:80])

print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
