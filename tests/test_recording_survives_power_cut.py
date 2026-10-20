"""A flight record that only exists in RAM is not a flight record.

Why this exists. On 20 Oct the aircraft flew, landed, would not auto-disarm,
and the battery was pulled by hand. The battery is the Jetson's power too, so
the companion went down with it mid-write. What came back:

    vps_00005.csv   1081 NUL bytes at the tail
    vps_00006.csv    461 NUL bytes at the tail
    vps_00007.csv    survived - by luck, nothing more

The rows had been handed to `csv.writer` and were sitting in Python's buffer
and the kernel's page cache. Nothing had ever asked for them to be on the
disk. `vps_00006` was the session either side of the flight and `vps_00007`
was the flight itself: one more second of bad luck and the only record of the
first real flight would have been the FC's own dataflash, which knows what we
sent and nothing about why.

So this file pins three things, each of which was broken on 20 Oct:

  1. ROWS REACH THE DISK. A process killed without warning must leave the rows
     it wrote more than one sync interval ago readable. Note what this test can
     and cannot reach: SIGKILL loses Python's own buffer, which flush() fixes;
     a power cut also loses the kernel's page cache, which ONLY fsync() fixes.
     The test proves the flush half directly and that fsync is called at all -
     it cannot cut the power to this machine.

  2. THE UNIT ACTUALLY ASKS FOR FRAMES. --save-frames is off by default and
     drone-vps.service did not pass it, so frame_writer had never once run on
     a flight and 20 Oct produced a CSV with an empty frame_file column on
     every row and not a single image. A flag that exists and is not passed is
     a feature that does not exist.

  3. THE CAMERA IMU IS LOGGED PER SAMPLE. The estimate CSV only ever carried
     the gyro integrated over a frame. An integral over 33 ms cannot tell a
     0.9 Hz airframe wobble from a 47 Hz prop one, which is why "how much
     vibration was on the camera IMU" could not be answered from the flight at
     all. One row per sample, at the sample's own device timestamp.
"""
import csv
import os
import re
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "vps_vio"))

from live_node import IMU_LOG_COLUMNS, imu_rows  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        fails.append(name)


# ---------------------------------------------------------------- 1. the disk

WRITER = r"""
import csv, os, sys, time
path, sync_every = sys.argv[1], float(sys.argv[2])
fh = open(path, "w", newline="")
wr = csv.writer(fh)
wr.writerow(["n"])
last = time.time()
n = 0
while True:
    wr.writerow([n]); n += 1
    if sync_every > 0 and time.time() - last >= sync_every:
        last = time.time()
        fh.flush(); os.fsync(fh.fileno())
    time.sleep(0.002)
"""


def rows_after_kill(sync_every, run_for=1.6):
    """Start a writer, SIGKILL it, and count the rows that survived."""
    d = tempfile.mkdtemp()
    path = os.path.join(d, "rows.csv")
    script = os.path.join(d, "w.py")
    with open(script, "w") as f:
        f.write(WRITER)
    p = subprocess.Popen([sys.executable, script, path, str(sync_every)])
    time.sleep(run_for)
    p.kill()
    p.wait()
    with open(path, newline="") as f:
        return len(list(csv.reader(f))) - 1      # minus the header


print("rows a killed process leaves behind")
no_sync = rows_after_kill(0.0)
with_sync = rows_after_kill(0.2)
# At ~500 rows/s and a 0.2 s sync, a killed writer should leave hundreds of
# rows; unsynced it leaves whatever fitted in one 8 KB buffer, or nothing.
check("periodic sync leaves the rows on disk", with_sync >= 200,
      f"{with_sync} rows survived")
check("that is more than an unsynced writer leaves", with_sync > no_sync,
      f"synced {with_sync} vs unsynced {no_sync}")

# And the defence has to be cheap enough to sit in the flow loop, which has
# ~59 ms a frame and already spends 22 of them.
d = tempfile.mkdtemp()
fh = open(os.path.join(d, "cost.csv"), "w", newline="")
wr = csv.writer(fh)
row = ["%.4f" % i for i in range(40)]
worst = 0.0
for i in range(300):
    wr.writerow(row)
    if i % 30 == 29:
        t0 = time.perf_counter()
        fh.flush()
        os.fsync(fh.fileno())
        worst = max(worst, (time.perf_counter() - t0) * 1e3)
fh.close()
check("one flush+fsync stays well inside a frame budget", worst < 20.0,
      f"worst {worst:.2f} ms against 59 ms a frame")

# ---------------------------------------------------------------- 2. the unit

print("\nthe flying unit asks for images")
unit = open(os.path.join(HERE, "systemd", "drone-vps.service")).read()
exec_start = re.search(r"^ExecStart=(.*?)(?=^\w+=|\Z)", unit, re.M | re.S)
line = " ".join(exec_start.group(1).replace("\\\n", " ").split()) if exec_start else ""
m = re.search(r"--save-frames\s+([0-9.]+)", line)
check("drone-vps.service passes --save-frames", m is not None,
      "off by default, so without it no flight records an image")
if m:
    hz = float(m.group(1))
    check("at the rate the pilot asked for", abs(hz - 4.0) < 1e-9, f"{hz:g} Hz")
    check("and the rate is not zero", hz > 0)
check("--out is a directory, which --save-frames requires",
      re.search(r"--out\s+\S+/vps(\s|$)", line) is not None, line[-60:])

# The pilot's rule, 20 Oct: images start at arm and stop at disarm. The
# services run for as long as the Jetson has power, so without this the run
# folder fills with pictures of the ground the aircraft is parked on - 29 MB a
# minute, measured - and every one of them is a frame offline matching has to
# be told to ignore. The source of truth is the FC's own armed flag, and the
# three-way behaviour matters: True saves, False stops, None (no link at all)
# does not save, because "cannot tell" is the bench and a flight always has a
# link.
src = open(os.path.join(HERE, "vps_vio", "live_node.py")).read()
check("the unit does NOT pass --frames-any-time",
      "--frames-any-time" not in line,
      "that flag is bench-only; on the aircraft frames follow the arm switch")
check("the offer is gated on the FC's armed flag",
      re.search(r'armed\s*=\s*f\.get\("armed"\)', src) is not None
      and re.search(r"if armed is True:", src) is not None,
      "`armed is True`, not truthiness - None must not save")
check("an unknown armed state does not record",
      "elif not a.frames_any_time:" in src)
check("and the radio says which way the gate is",
      "VPS FRAMES OFF - DISARMED" in src)

# ------------------------------------------------------------- 3. the IMU log

print("\nthe camera IMU, one row per sample")
f = {
    "gyro": [(1.000, 0.01, -0.02, 0.03),
             (1.010, 0.02, -0.03, 0.04),
             (1.020, 0.03, -0.04, 0.05)],
    "accel": [(1.005, 0.11, -0.22, 0.33),
              (1.015, 0.12, -0.23, 0.34)],
}
rows = list(imu_rows(f))
check("every sample gets its own row", len(rows) == 5, f"{len(rows)} rows")
check("the header matches the row width",
      all(len(r) == len(IMU_LOG_COLUMNS) for r in rows))
kinds = [r[1] for r in rows]
check("gyro and accel are distinguishable",
      kinds.count("g") == 3 and kinds.count("a") == 2, str(kinds))
ts = [float(r[0]) for r in rows]
check("each row carries the sample's own device timestamp, not the frame's",
      len(set(ts)) == 5, str(sorted(set(ts))))
check("the gyro values survive the round trip",
      [float(r[2]) for r in rows if r[1] == "g"] == [0.01, 0.02, 0.03])

# A silent sensor must produce an empty log, never a log full of zeros - that
# is the mistake the accelerometer made for three weeks: a field that was
# never enabled reads as a perfect standstill rather than as absent.
check("a frame with no samples writes nothing",
      list(imu_rows({"gyro": [], "accel": []})) == []
      and list(imu_rows({})) == [])

# The integral stays where derotation needs it: this log is an addition, not a
# replacement.
check("the estimate CSV keeps its per-frame integral columns",
      "gyro_d_roll" in open(os.path.join(HERE, "vps_vio", "live_node.py")).read())

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
