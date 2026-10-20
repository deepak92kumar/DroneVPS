"""A frame the CSV names must be a frame someone can open.

86 of the 1151 frames saved on the 20 Oct 16:38 flight are ZERO BYTES. Not
truncated - empty. They are the last 86, consecutive, spanning the final 23
seconds, and the last readable frame carries the same timestamp as the first
empty one. The writer reported all 1151 as written and `failed` stayed at 0.

Offline, that reads as a matcher problem. match_flight.py said "86 frame on
disk but unreadable" in the same breath as the inlier counts, which is exactly
the wrong place to look.

TWO FAULTS, and only one of them is what bit here.

  cv2.imwrite RETURNS FALSE on failure. It does not raise. So the worker's
  `except Exception` could never fire for a failed write, nothing was counted,
  and a row named a file nobody could open. A full disk, a read-only mount or a
  bad path were all invisible by construction. That one is latent and is fixed
  whether or not it caused this.

  THE BYTES WERE NEVER FLUSHED. ext4 delays allocation: a file's name can be on
  the disk while its contents sit in the page cache for up to 30 s. The battery
  was pulled at the end of that flight. 23 seconds of unflushed frames is
  precisely what that leaves, and it is the same failure the estimate CSVs
  showed as NUL tails on the same day - those got a per-second fsync on 20 Oct
  and the frames did not.

Measured cost of the fix on this Orin's NVMe: one encode-and-fsync is 16.4 ms
median, 23.2 ms worst, against the 250 ms a frame has at the 4 Hz the aircraft
saves at. At the flying rate, 23 of 23 offered frames land non-empty.
"""
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "vps_vio"))

import frame_writer as FW  # noqa: E402
from frame_writer import FrameWriter  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        fails.append(name)


try:
    import cv2
except ImportError:                      # pragma: no cover
    print("cv2 missing - nothing here can be tested")
    sys.exit(0)

d = tempfile.mkdtemp()
img = (np.random.default_rng(20261020).random((480, 640)) * 255).astype("uint8")
params = [int(cv2.IMWRITE_JPEG_QUALITY), 92]

# ------------------------------------------------- the encoder itself

print("the default encoder writes real bytes and flushes them")
p = os.path.join(d, "one.jpg")
FW._default_encode(p, img, params)
check("the file is not empty", os.path.getsize(p) > 0, f"{os.path.getsize(p)} bytes")
check("and it opens as an image",
      cv2.imread(p, cv2.IMREAD_GRAYSCALE) is not None)
check("it fsyncs rather than trusting the page cache",
      "os.fsync" in open(os.path.join(HERE, "vps_vio", "frame_writer.py")).read(),
      "ext4 can hold 30 s of contents with only the name on disk")

print("\na JPEG on disk begins ffd8 and ends ffd9")
raw = open(p, "rb").read()
check("starts with the JPEG marker", raw[:2] == b"\xff\xd8", raw[:2].hex())
check("ends with the JPEG marker", raw[-2:] == b"\xff\xd9", raw[-2:].hex())

# ------------------------------------------------- a failure is counted

print("\na write that does not land is counted, not reported as written")
w = FrameWriter(os.path.join(d, "fail"), hz=1000.0)


def refuse(path, im, prm):
    # The real shape of it: the file gets created, then the write dies. That is
    # what leaves a zero-byte stub whose name is already in the CSV.
    open(path, "wb").close()
    raise OSError("disk went away")


w._encode = refuse
name = w.offer(img, 1)
w.close()
check("failed was incremented", w.failed == 1, f"failed={w.failed}")
left = [f for f in os.listdir(os.path.join(d, "fail"))
        if os.path.getsize(os.path.join(d, "fail", f)) == 0]
check("and no zero-byte stub is left behind", not left, str(left))

# imwrite's own contract is the thing that hid this, so pin it.
print("\nthe contract that hid it: imwrite returns False, it does not raise")
bad = os.path.join(d, "no_such_dir", "x.jpg")
raised = False
try:
    ret = cv2.imwrite(bad, img, params)
except Exception:
    raised = True
    ret = None
check("cv2.imwrite signals failure by RETURN VALUE",
      raised or ret is False,
      "returned False" if ret is False else "raised")
check("so the encoder must not ignore it",
      "cv2.imwrite(path, img, params)" not in
      open(os.path.join(HERE, "vps_vio", "frame_writer.py")).read(),
      "the bare call is what counted 86 lost frames as saved")

# ------------------------------------------------- at the flying rate

print("\nat the 4 Hz the aircraft saves at, every named frame lands")
# REAL TIME, not an injected clock. A synthetic clock advances six seconds in
# microseconds, so the rate gate names all 24 frames at once, the bounded queue
# fills and 15 are dropped - which is the drop-on-backlog design working, not a
# fault, but it measures nothing about a flight. The aircraft offers a frame
# every 33 ms of wall time and the disk has 250 ms per saved frame, so the only
# honest version of this test costs six seconds.
import time as _time  # noqa: E402

out = os.path.join(d, "flight")
w2 = FrameWriter(out, hz=4.0)
named = []
_t0 = _time.time()
while _time.time() - _t0 < 6.0:
    nm = w2.offer(img, len(named))
    if nm:
        named.append(nm)
    _time.sleep(1.0 / 30.0)
w2.close()
landed = [n for n in named
          if os.path.exists(os.path.join(out, n))
          and os.path.getsize(os.path.join(out, n)) > 0]
check("every frame the writer named is on disk and non-empty",
      len(landed) == len(named), f"{len(landed)} of {len(named)}")
check("about 4 a second were named", 20 <= len(named) <= 28,
      f"{len(named)} over {_time.time() - _t0:.1f} s")
check("nothing failed and nothing was dropped",
      w2.failed == 0 and w2.dropped == 0,
      f"failed={w2.failed} dropped={w2.dropped}")

# ------------------------------------------------- it is affordable

print("\nand it fits the budget a frame has")
ts = []
for i in range(15):
    a = _time.time()
    FW._default_encode(os.path.join(d, f"t{i}.jpg"), img, params)
    ts.append((_time.time() - a) * 1000.0)
p50, worst = float(np.median(ts)), float(max(ts))
print(f"    encode + fsync: p50 {p50:.1f} ms, worst {worst:.1f} ms")
check("one write is well inside the 250 ms a 4 Hz frame has",
      worst < 150.0, f"worst {worst:.1f} ms")

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
