"""Run the map fix on its own core so it never stalls the flow loop.

A fix costs 68 ms. The flow loop's budget is 33 ms at 30 Hz or 100 ms at 10 Hz,
so a fix taken inline blocks one frame outright - measured as 24 overruns in
244 frames at 10 Hz, and almost every frame at 30 Hz. The Orin has six cores
and was using roughly one.

Threads are enough here, despite the GIL: OpenCV releases it around the heavy
calls, so matchTemplate genuinely runs on another core. No second process, no
shared-memory plumbing, and the map stays in one place.

The queue holds exactly one job. If a fix is already running, a new request is
DROPPED rather than queued - a fix computed against a prior that is two seconds
stale is worse than no fix at all, and the flow keeps the position meanwhile.
"""
from __future__ import annotations
import threading
import time
from dataclasses import dataclass

import numpy as np

from map_match_ncc import locate


@dataclass
class FixResult:
    lon: float
    lat: float
    score: float
    margin: float
    ok: bool
    age_s: float          # how long the fix took, i.e. how stale it is on arrival
    prior_lon: float      # the prior it was computed against, so the caller can
    prior_lat: float      # apply it as a correction to that, not to "now"


class MapFixWorker:
    def __init__(self, cache, *, focal_px: float, radius_m: float = 150.0,
                 min_score: float = 0.05, min_margin: float = 0.01):
        self.cache = cache
        self.focal_px = float(focal_px)
        self.radius_m = float(radius_m)
        self.min_score = float(min_score)
        self.min_margin = float(min_margin)

        self._job = None
        self._result: FixResult | None = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._run = True
        self.submitted = self.dropped = self.completed = 0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, img: np.ndarray, *, prior_lon: float, prior_lat: float,
               altitude_m: float, yaw_deg: float) -> bool:
        """Queue a fix. Returns False if one was already running (request dropped)."""
        with self._lock:
            if self._job is not None:
                self.dropped += 1
                return False
            self._job = (img, prior_lon, prior_lat, altitude_m, yaw_deg, time.monotonic())
            self.submitted += 1
        self._wake.set()
        return True

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._job is not None

    def poll(self) -> FixResult | None:
        """Take the most recent finished fix, if there is one. Returns it once."""
        with self._lock:
            r, self._result = self._result, None
        return r

    def _loop(self):
        while self._run:
            self._wake.wait(0.2)
            self._wake.clear()
            with self._lock:
                job = self._job
            if job is None:
                continue
            img, plon, plat, alt, yaw, t0 = job
            try:
                if not self.cache.covers(plon, plat):
                    res = None
                else:
                    win, wg = self.cache.window(plon, plat, self.radius_m)
                    res = locate(img, win, wg, altitude_m=alt, focal_px=self.focal_px,
                                 yaw_deg=yaw, map_gsd_m=self.cache.gsd,
                                 min_score=self.min_score, min_margin=self.min_margin,
                                 window_is_edges=True)
            except Exception:
                res = None
            out = None
            if res is not None:
                out = FixResult(res.lon, res.lat, res.score, res.margin, res.ok,
                                time.monotonic() - t0, plon, plat)
            with self._lock:
                self._result = out
                self._job = None
                self.completed += 1

    def stop(self):
        self._run = False
        self._wake.set()
        self._thread.join(timeout=2.0)
