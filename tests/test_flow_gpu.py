"""The flow tracks on the GPU, and the CPU gets its cores back.

WHY. The flow is what eats the CPU on this aircraft, and CPU is what frames
come out of. Flight 382 measured the node at 246% of a core while dropping
3.67 frames a second, and measured what those dropped frames cost: flow minus
GPS_RAW_INT was +0.770 m/s p50 where fewer than 3 a second were dropped and
+3.705 where 6 to 10 were.

Measured on two real 1280x800 frames from that flight, 300 corners, a 31 px
window, 4 levels, forward and backward:

    OpenCV, 6 threads   wall 53.39 ms   CPU 98.21 ms   1.84 cores
    VPI CUDA            wall 25.70 ms   CPU 15.01 ms   0.58 cores

85% of the CPU goes away - the arithmetic moves to the GPU and the CPU waits.
At 25 Hz with two trackers that is 491% of a core against 75%.

WHAT IT COSTS, because it is not free. From the same 300 seed points the GPU
keeps 251 tracks where OpenCV keeps 285, 12% fewer, and corr(tracks, phantom)
was -0.233 on 382. Against that, the median shift the two produce from those
same seeds agrees to 0.26 px in x and 0.03 px in y.

VPI'S DEFAULTS ARE NOT OURS and the difference is not cosmetic: winsize
defaults to 11 and maxiter to 6, and run that way it tracked 15 of 300 points
with the survivors 72 px out. Passed 31 and 30 it tracks 251 and agrees to a
fifth of a pixel.

VPI ALSO NEEDS DISPLAY UNSET. With one set it reaches for the X server's EGL
rather than the device's and every backend fails with EGL_BAD_DISPLAY. Under
TigerVNC, which is how this Jetson is driven, nothing works at all. The flying
systemd unit has no DISPLAY, so the aircraft is fine; a shell is not.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vps_vio"))
from flow_odom import FlowTracker  # noqa: E402
from config import VpsConfig  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}   {detail}")
    if not ok:
        fails.append(name)


print("the setting exists, defaults on, and the node PASSES it")
check("VpsConfig.flow_gpu is True", VpsConfig().flow_gpu is True)
live = open(os.path.join(os.path.dirname(__file__), "..", "vps_vio",
                         "live_node.py"), encoding="utf-8").read()
check("the primary tracker is given it",
      'use_gpu=bool(getattr(_vps_config(), "flow_gpu", True))' in live,
      "four settings on the map worker were dead for exactly this reason. "
      "_vps_config() and not _c0: the tracker is built before either of "
      "__init__'s config reads, and reaching forward to one was an "
      "UnboundLocalError this suite caught the moment the line was added")
check("and the partner follows the primary",
      "use_gpu=node.tracker.gpu" in live,
      "two trackers on different backends would make flow_disagree_px "
      "measure the backends instead of the cameras")

print("\nthe window and the iterations are OURS, not VPI's")
src = open(os.path.join(os.path.dirname(__file__), "..", "vps_vio",
                        "flow_odom.py"), encoding="utf-8").read()
check("LK_WIN is 31", "LK_WIN = 31" in src, "VPI defaults to 11")
check("LK_ITERS is 30", "LK_ITERS = 30" in src, "VPI defaults to 6")
check("and _vpi_track passes them",
      "winsize=self.LK_WIN" in src and "maxiter=self.LK_ITERS" in src)

print("\nstatus is inverted, because VPI and OpenCV disagree about it")
check("0 means tracked on the VPI side", "== 0).astype(np.uint8)" in src,
      "OpenCV says 1 for tracked; getting this backwards keeps exactly the "
      "points that should be dropped")

print("\nCPU-only still works, and says so")
t = FlowTracker(use_gpu=False)
check("use_gpu=False gives a CPU tracker", t.gpu is False)

print("\nand the two backends agree, if this machine has a GPU")
probe = FlowTracker(use_gpu=True)
if not probe.gpu:
    print(f"  SKIP  no usable VPI here: {probe.vpi_error}")
else:
    rng = np.random.default_rng(7)
    base = rng.integers(0, 255, (400, 640), dtype=np.uint8)
    base = np.asarray(base, np.uint8)
    nxt = np.roll(base, (3, -2), axis=(1, 0))
    probe.step(base)
    pts = probe.pts
    check("it seeded", pts is not None and len(pts) > 40, f"{0 if pts is None else len(pts)}")
    pc, sc, rc, s2c = probe._cv_track(base, nxt, pts)
    pg, sg, rg, s2g = probe._vpi_track(base, nxt, pts)
    check("the GPU returns OpenCV's shapes",
          pg.shape == pc.shape and sg.shape == sc.shape,
          f"{pg.shape} {sg.shape}")
    both = (sc.ravel() == 1) & (sg.ravel() == 1)
    d = np.linalg.norm((pg - pc)[both].reshape(-1, 2), axis=1)
    check("and the same answer where both tracked",
          both.sum() > 20 and float(np.median(d)) < 1.0,
          f"{both.sum()} points, median {float(np.median(d)):.3f} px")
    shc = np.median((pc - pts)[(sc.ravel() == 1)].reshape(-1, 2), axis=0)
    shg = np.median((pg - pts)[(sg.ravel() == 1)].reshape(-1, 2), axis=0)
    check("the median shift agrees to under a pixel",
          float(np.linalg.norm(shc - shg)) < 1.0,
          f"cpu {shc.round(3)} gpu {shg.round(3)}")

print()
if fails:
    print(f"{len(fails)} FAILED: " + ", ".join(fails))
    sys.exit(1)
print("ALL PASS")
