"""
Shadow optical-flow estimators. Logged only, never sent to the flight controller.

N: long-baseline velocity from OAK FeatureTracker IDs (~80 ms baseline instead of one 10 ms frame).
   Pixel noise stays the same while the displacement grows, so velocity noise should drop.
J: Jetson LK optical flow on the mono stream (~10 ms old on arrival vs ~110 ms for the on-device
   tracker output in flight). Runs in a worker thread on the newest frame only.

Derotation (gyro) and scaling are done by the daemon on its own thread with the same math as the
live flow, so gyro history is never read from the worker thread.
"""
import math
import threading
import time
from collections import deque

import cv2
import numpy as np


def partial_affine_translation(old_pts, new_pts, cx, cy, reproj_px=4.0):
    """(tx, ty, inliers) of a RANSAC partial affine between point sets centred on (cx, cy), else None.
    Same acceptance as the live flow: >=8 inliers, >=20 % inliers, scale 0.7-1.3."""
    old_pts = np.float32(old_pts).reshape(-1, 2)
    new_pts = np.float32(new_pts).reshape(-1, 2)
    if len(old_pts) < 8:
        return None
    M, inliers = cv2.estimateAffinePartial2D(old_pts - [cx, cy], new_pts - [cx, cy],
                                             method=cv2.RANSAC, ransacReprojThreshold=reproj_px)
    if M is None or inliers is None:
        return None
    n = int(inliers.sum())
    scale = float(np.linalg.norm(M[:2, 0]))
    if n < 8 or n < 0.20 * len(old_pts) or not (0.70 < scale < 1.30):
        return None
    return float(M[0, 2]), float(M[1, 2]), n


def px_to_ned_velocity(tx, ty, droll, dpitch, focal_px, height_m, heading_deg, dt_s):
    """Image translation (px) over dt_s -> (vN, vE) m/s, with the live flow's sign conventions:
    tilt shift x = -f*tan(droll), y = +f*tan(dpitch); body dx = -tx*mpp, dy = +ty*mpp."""
    tx = tx + focal_px * math.tan(droll)
    ty = ty - focal_px * math.tan(dpitch)
    mpp = height_m / focal_px
    dx_body = -(tx * mpp)
    dy_body = ty * mpp
    c = math.cos(math.radians(heading_deg))
    s = math.sin(math.radians(heading_deg))
    return (dy_body * c - dx_body * s) / dt_s, (dy_body * s + dx_body * c) / dt_s


class LongBaselineTracks:
    """Keeps recent FeatureTracker frames and matches the current frame against one ~baseline_s older."""

    def __init__(self, cx, cy, baseline_s=0.08, max_age_s=0.30):
        self.cx, self.cy = cx, cy
        self.baseline_s = baseline_s
        self.max_age_s = max_age_s
        self._hist = deque()

    def update(self, t, features):
        """features: {id: (x, y)}. Returns (t_ref, tx, ty, inliers) or None."""
        self._hist.append((t, features))
        while self._hist and t - self._hist[0][0] > self.max_age_s:
            self._hist.popleft()
        ref = None
        for tr, fr in reversed(self._hist):
            if t - tr >= self.baseline_s:
                ref = (tr, fr)
                break
        if ref is None:
            return None
        common = [k for k in features if k in ref[1]]
        if len(common) < 8:
            return None
        res = partial_affine_translation([ref[1][k] for k in common], [features[k] for k in common],
                                         self.cx, self.cy, reproj_px=4.0)
        return None if res is None else (ref[0], res[0], res[1], res[2])


class JetsonLKWorker(threading.Thread):
    """LK optical flow on the newest mono frame. Results: (t0, t1, tx, ty, inliers, proc_ms)."""

    def __init__(self, cx, cy, n_features=150, refresh_below=90, max_gap_s=0.10):
        super().__init__(daemon=True)
        self.cx, self.cy = cx, cy
        self.n_features = n_features
        self.refresh_below = refresh_below
        self.max_gap_s = max_gap_s
        self._cv = threading.Condition()
        self._job = None
        self._results = deque(maxlen=64)
        self.running = True
        self.start()

    def submit(self, img, t_frame):
        if img is None or t_frame is None:
            return
        with self._cv:
            self._job = (img, t_frame)
            self._cv.notify()

    def pop_results(self):
        with self._cv:
            out = list(self._results)
            self._results.clear()
        return out

    def stop(self):
        self.running = False

    def run(self):
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01)
        prev_img = prev_t = prev_pts = None
        while self.running:
            with self._cv:
                if self._job is None:
                    self._cv.wait(timeout=0.2)
                job, self._job = self._job, None
            if job is None:
                continue
            img, t = job
            t_start = time.perf_counter()
            try:
                if (prev_img is not None and prev_pts is not None and len(prev_pts) >= 8
                        and 0.0 < t - prev_t < self.max_gap_s):
                    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_img, img, prev_pts, None, winSize=(15, 15),
                                                         maxLevel=2, criteria=crit)
                    good = st.ravel() == 1
                    if int(good.sum()) >= 8:
                        res = partial_affine_translation(prev_pts[good], p1[good], self.cx, self.cy, reproj_px=2.0)
                        if res is not None:
                            ms = (time.perf_counter() - t_start) * 1000.0
                            with self._cv:
                                self._results.append((prev_t, t, res[0], res[1], res[2], ms))
                        prev_pts = p1[good].reshape(-1, 1, 2)
                    else:
                        prev_pts = None
                elif prev_t is not None and not (0.0 < t - prev_t < self.max_gap_s):
                    prev_pts = None
                if prev_pts is None or len(prev_pts) < self.refresh_below:
                    prev_pts = cv2.goodFeaturesToTrack(img, self.n_features, 0.01, 8, blockSize=5)
            except Exception:
                prev_pts = None
            prev_img, prev_t = img, t
