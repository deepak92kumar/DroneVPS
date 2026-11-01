"""The matcher runs in its own process, and the flow loop stops paying for it.

THE MEASUREMENT THAT MOVED IT. map_fix_worker's own header said threads were
enough "despite the GIL: OpenCV releases it around the heavy work". Flight
382, 1 Nov, was the first flight whose matcher ever actually ran - every
earlier one had covers() false and returned instantly - and it priced the
claim:

    loop rate        23.86 Hz (378, matcher idle)  ->  21.20 Hz (382)
    frames dropped    1.06 /s                      ->   3.67 /s

That is 5.3 frames per fix against 84 ms of measured matcher work, so four
fifths of it was never arithmetic. It was the Python glue between the OpenCV
calls holding the GIL while the flow loop wanted it.

Frames are what the velocity is made of. On the same flight, flow minus
GPS_RAW_INT was +0.770 m/s p50 where fewer than 3 frames a second were
dropped, and +3.705 where 6 to 10 were - 4.8x worse.

Reproduced on this machine with a 25 Hz loop doing two LK passes a tick and
one fix a second, 550 ticks, counting ticks that ran late:

    matcher in a THREAD     127 late  = 23.1%
    matcher in a PROCESS     51 late  =  9.3%
    no matcher at all        65 late  = 11.8%

A process costs the loop nothing measurable; a thread doubled it.

fork() is the deliberate start method. The child inherits the 30 MB map array
copy-on-write instead of pickling it per job, and _warm() loads torch INSIDE
the child, so the flying process never imports it.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vps_vio"))
from map_fix_worker import MapFixWorker  # noqa: E402
from map_db import GeoRef  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}   {detail}")
    if not ok:
        fails.append(name)


class FakeCache:
    """A map that covers everything and answers with flat grey."""

    def __init__(self):
        self.grey = np.full((900, 900), 128, np.uint8)
        self.gsd = 0.5
        self.gsd_x = 0.43
        self.h, self.w = self.grey.shape
        self.geo = GeoRef.from_gdal((74.0, 1e-5, 0.0, 30.6, 0.0, -1e-5))
        self.gt = (74.0, 1e-5, 0.0, 30.6, 0.0, -1e-5)

    def covers(self, lon, lat):
        return True

    def window(self, lon, lat, radius_m, kind="grey"):
        return self.grey[:400, :400], GeoRef.from_gdal(self.gt)


print("the default is a process, which is the whole point")
import inspect  # noqa: E402
sig = inspect.signature(MapFixWorker.__init__)
check("use_process defaults True", sig.parameters["use_process"].default is True,
      "a thread cost 23.1% late ticks against 9.3%")
check("and a core can be pinned if it is ever wanted",
      "worker_core" in sig.parameters
      and sig.parameters["worker_core"].default is None,
      "off by default - a pin that collides with the flow threads is worse "
      "than none")

src = open(os.path.join(os.path.dirname(__file__), "..", "vps_vio",
                        "map_fix_worker.py"), encoding="utf-8").read()
print("\nfork, not spawn")
check('the context is "fork"', 'get_context("fork")' in src,
      "spawn would pickle the 30 MB map into every child and re-import torch "
      "in the parent")

print("\nthe node PASSES it, so the config is not dead")
live = open(os.path.join(os.path.dirname(__file__), "..", "vps_vio",
                         "live_node.py"), encoding="utf-8").read()
check("VpsNode passes use_process from the config",
      'use_process=bool(getattr(_cfg, "matcher_process", True))' in live,
      "matcher, map_window_alt_frac, map_query_shape and fix_period_s were "
      "all dead this way first")
check("and the core with it",
      'worker_core=getattr(_cfg, "matcher_core", None)' in live)
cfg = open(os.path.join(os.path.dirname(__file__), "..", "vps_vio",
                        "config.py"), encoding="utf-8").read()
check("the config field exists and is True",
      "matcher_process: bool = True" in cfg)
check("and the pin is off by default",
      "matcher_core: int | None = None" in cfg)

print("\nthe child cannot touch the counters, so the parent does the counting")
body = src[src.index("    def _drain"):]
body = body[:body.index("\n    def ", 10)]
check("_drain is what clears the job slot", "self._job = None" in body)
check("and it is the only place completed is counted",
      body.count("self.completed += 1") == 1
      and src.count("self.completed += 1") == 2,
      "the other one is _loop, the thread fallback")
check("poll drains first", "def poll" in src and "self._drain()" in src)
check("busy drains first",
      src.index("self._drain()") < src.index("return self._job is not None"))

print("\nit really runs, and it answers")
w = MapFixWorker(FakeCache(), focal_px=795.1, matcher="ncc", fix_period_s=0.0,
                 min_alt_m=10.0)
try:
    check("a child process was started", w._proc is not None and w._proc.is_alive())
    t0 = time.time()
    while not w.model_ready and time.time() - t0 < 30.0:
        w.poll()
        time.sleep(0.05)
    check("the child reported its model ready", w.model_ready,
          f"{w.model_load_s:.2f} s")
    img = np.random.default_rng(0).integers(0, 255, (400, 640), dtype=np.uint8)
    ok = w.submit(img, prior_lon=74.002, prior_lat=30.598,
                  altitude_m=80.0, yaw_deg=0.0)
    check("submit accepted", ok)
    check("and the slot is busy until the child answers", w.busy)
    t0 = time.time()
    while w.busy and time.time() - t0 < 30.0:
        w.poll()
        time.sleep(0.02)
    check("the child answered", not w.busy, f"{time.time() - t0:.2f} s")
    check("and the parent counted it", w.completed == 1, f"completed={w.completed}")

    print("\nthe single slot still drops rather than queues")
    w.submit(img, prior_lon=74.002, prior_lat=30.598, altitude_m=80.0, yaw_deg=0.0)
    before = w.dropped
    second = w.submit(img, prior_lon=74.002, prior_lat=30.598,
                      altitude_m=80.0, yaw_deg=0.0)
    check("a second submit while busy is refused", second is False)
    check("and counted as dropped", w.dropped == before + 1)
finally:
    w.stop()
check("stop ended the child", w._proc is not None and not w._proc.is_alive())

print()
if fails:
    print(f"{len(fails)} FAILED: " + ", ".join(fails))
    sys.exit(1)
print("ALL PASS")
