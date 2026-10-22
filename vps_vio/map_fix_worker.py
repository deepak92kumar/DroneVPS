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

Two rates matter and they are not the same. Dropping-while-busy is about not
queueing stale work; fix_period_s is about not asking too often in the first
place. At 0.4 s the map pushed a stationary aircraft at 1.18 m/s above 80 m
against a real drift of 0.15-0.37, and the worker ate 75% of a core. At 2.0 s
it pushes 0.26 and costs 17%. The period is enforced here as well as by the
caller, so a caller that forgets cannot put that behaviour back.
"""
from __future__ import annotations
import threading
import time
from dataclasses import dataclass

import numpy as np

# Default thresholds per matcher. They are not interchangeable: the NCC margin
# is read off a coarse correlation surface and peaks around 0.1-0.2 on
# four-year-old imagery, while the XFeat numbers are an inlier ratio and a
# cluster tightness. Swapping one matcher's gate onto the other would either
# pass everything or nothing.
_GATES = {
    "ncc":   dict(min_score=0.05, min_margin=0.01),   # swept: 50 frames, +-150 m
    # 0.10, NOT 0.25, and this is now the swept value rather than the guess.
    #
    # 12 Oct introduced 0.25 for a real reason: the gate "had never been
    # applied on the XFeat path at all - that branch hard-coded ok=True, so 189
    # of 189 fixes were accepted including ones 85.66 m out". That is an
    # argument for HAVING a gate, which this still is, not for 0.25
    # specifically - and the same report doubted the number in the same
    # paragraph: "Its rejections track our own error... because the matcher
    # takes its window from our estimate, so the gate closes exactly when it
    # needs to be open."
    #
    # Swept 21 Oct on the first frames this project has ever had above 72 m -
    # 601 above the 60 m floor from the 95 m flight, matched against the raw
    # receiver:
    #
    #     90-100 m, 477 matches      score p50 0.077, p95 0.187, MAX 0.248
    #     at min_score 0.25            0 of 477 pass. Not one, all flight.
    #     at 0.10, min_inliers 12    176 pass, error p50 8.3  p95 12.9  max 24.8 m
    #     at 0.00, min_inliers 12    228 pass, error p50 8.3  p95 12.7  max 24.8 m
    #
    # 8.3 m is better than the fuser's own fix_sigma_m of 20, and nothing like
    # 12 Oct's 85.66 m outliers. The score does not predict the error at all
    # there - r = -0.118 against it, inliers -0.158, margin +0.034 - so the gate
    # buys accuracy with yield and gets none back, which is why it goes to the
    # bottom of the swept range rather than to zero: it still catches the
    # degenerate case that 12 Oct found, and costs 52 of 228 fixes to do it.
    #
    # What bounds the damage is downstream and unchanged: map_fuse drops a fix
    # beyond 3 sigma outright, and max_correction_m caps an accepted one at
    # 0.5 m however confident it was.
    "xfeat": dict(min_score=0.10, min_margin=0.20),
}


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
    inliers: int = 0      # RANSAC inliers behind the fix
    total_matches: int = 0
    reason: str = ""      # why a rejected fix was rejected, for the recorder
    # The altitude the MATCH was made at, not the altitude now. Added 22 Oct:
    # the matcher's error depends strongly on height, because the search
    # window is window_alt_frac x altitude and a smaller window holds fewer
    # features - and until now the fix's own position was never written down,
    # so that dependence could only be taken from an archived report of a
    # configuration this code does not share. It is measurable here instead.
    altitude_m: float = 0.0


class MapFixWorker:
    def __init__(self, cache, *, focal_px: float, matcher: str = "xfeat",
                 radius_m: float = 150.0, window_alt_frac: float = 0.60,
                 min_alt_m: float = 60.0, fix_period_s: float = 2.0,
                 min_score: float | None = None,
                 min_margin: float | None = None,
                 principal_xy=None):
        self.cache = cache
        self.focal_px = float(focal_px)
        # (cx, cy) of the frames being handed in, when the caller knows them.
        # A fix is consumed as the AIRCRAFT's position, and the pixel that
        # corresponds to is the one the optical axis pierces, not the middle
        # of the image - 80 px apart on this camera's leg-cropped frames,
        # which is 11 m of ground at 110 m, in the same direction every frame.
        # None keeps the old geometric-centre behaviour for a caller that has
        # no K: wrong by that offset, but not silently different.
        self.principal_xy = (None if principal_xy is None
                             else (float(principal_xy[0]), float(principal_xy[1])))
        self.matcher = matcher.lower()
        self.radius_m = float(radius_m)
        self.window_alt_frac = float(window_alt_frac)
        self.min_alt_m = float(min_alt_m)
        self.fix_period_s = float(fix_period_s)

        # XFeat only. The NCC matcher was removed on 20 Oct at the pilot's
        # instruction - XFeat had already been the default here and the two
        # were never run together, so the alternative was dead weight on the
        # flying path. The geometry they shared lives in map_query.
        if self.matcher != "xfeat":
            raise ValueError(f"unknown matcher {matcher!r}, only 'xfeat' remains")
        from map_match_xfeat import locate
        self._window_kind = "grey"
        self._locate = locate

        gate = _GATES[self.matcher]
        self.min_score = gate["min_score"] if min_score is None else float(min_score)
        self.min_margin = gate["min_margin"] if min_margin is None else float(min_margin)

        self._job = None
        self._result: FixResult | None = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._run = True
        self._last_submit = 0.0
        self.submitted = self.dropped = self.completed = 0
        self.skipped_low = self.skipped_period = 0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, img: np.ndarray, *, prior_lon: float, prior_lat: float,
               altitude_m: float, yaw_deg: float) -> bool:
        """Queue a fix. False if it was refused, for any of three reasons.

        Below min_alt_m the map does not run at all. That is deliberate and it
        is also the project's largest single error: on a 6 m/s descent the
        estimate went 4.13 to 26.0 m in twelve seconds because nothing absolute
        corrects below this height. Raising the floor is not the fix - the map
        was measured as not starting at 80 m, which is why the floor came DOWN
        to 60 - so whatever closes the descent has to come from elsewhere.
        """
        now = time.monotonic()
        if altitude_m < self.min_alt_m:
            self.skipped_low += 1
            return False
        if now - self._last_submit < self.fix_period_s:
            self.skipped_period += 1
            return False
        with self._lock:
            if self._job is not None:
                self.dropped += 1
                return False
            self._job = (img, prior_lon, prior_lat, altitude_m, yaw_deg, now)
            self.submitted += 1
            self._last_submit = now
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

    def _radius_for(self, altitude_m: float) -> float:
        """Search window scaled to what the camera can actually see.

        It was a fixed 100 m until 4 Oct, which had the matcher guessing
        outside the frame's own footprint at low altitude and cropping the
        search too tight at high. 0.60 x altitude, floored so a window is
        never smaller than the query, and capped at radius_m so one bad
        altitude cannot ask for the whole map.
        """
        return float(np.clip(self.window_alt_frac * altitude_m, 40.0, self.radius_m))

    def _loop(self):
        while self._run:
            self._wake.wait(0.2)
            self._wake.clear()
            with self._lock:
                job = self._job
            if job is None:
                continue
            img, plon, plat, alt, yaw, t0 = job
            res = None
            try:
                if self.cache.covers(plon, plat):
                    win, wg = self.cache.window(plon, plat, self._radius_for(alt),
                                                kind=self._window_kind)
                    res = self._locate(
                        img, win, wg, altitude_m=alt, focal_px=self.focal_px,
                        yaw_deg=yaw, map_gsd_m=self.cache.gsd,
                        map_gsd_x_m=getattr(self.cache, "gsd_x", None),
                        min_score=self.min_score, min_margin=self.min_margin,
                        window_is_edges=(self._window_kind == "edges"),
                        principal_xy=self.principal_xy)
            except Exception:
                res = None
            out = None
            if res is not None:
                out = FixResult(res.lon, res.lat, res.score, res.margin, res.ok,
                                time.monotonic() - t0, plon, plat,
                                getattr(res, "inliers", 0),
                                getattr(res, "total_matches", 0),
                                getattr(res, "reason", ""),
                                float(alt))
            with self._lock:
                self._result = out
                self._job = None
                self.completed += 1

    def stop(self):
        self._run = False
        self._wake.set()
        self._thread.join(timeout=2.0)
