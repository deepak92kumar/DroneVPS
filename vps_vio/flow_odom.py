"""Metric x/y from the downward camera using the known height. No stereo.

At 107 m the OAK-D's 7.5 cm baseline is useless - disparity is under a third of
a pixel - so stereo cannot give depth and cuVSLAM degenerates to a monocular
tracker, which knows direction but not distance. Height fixes that: with a flat
ground plane at a known altitude, one pixel of image motion is exactly
alt/focal metres on the ground, so tracked features give metric translation
directly. This is the same trick the flow sensor uses, except with thousands of
features over the whole frame instead of one small patch, and it works at any
altitude the camera can still resolve texture at.

Rotation comes from the FC's yaw, so only translation is estimated here.
"""
from __future__ import annotations
import numpy as np
import cv2


def _R(roll_deg, pitch_deg, yaw_deg):
    r, p, y = np.radians([roll_deg, pitch_deg, yaw_deg])
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


# Pixels of image shift per degree of attitude change, fitted on 376 frame
# pairs of the 27 Sep flight above 60 m. Theory says focal * pi/180 = 13.8
# px/deg and the fit came back 13.14 and 13.17, with cross terms under 2.1 -
# close enough to confirm the geometry, and measured rather than assumed
# because a composed rotation written from the datasheet had the sign wrong and
# made the drift worse, not better.
ATT_PX_PER_DEG_ROLL_X = 13.14
ATT_PX_PER_DEG_PITCH_Y = 13.17


def attitude_shift_px(att_prev, att_cur) -> np.ndarray:
    """Image shift caused purely by the aircraft tilting, in pixels.

    Without removing this the measurement is nonsense: one degree of tilt slides
    the image 13.8 px, which at 107 m reads as 1.9 m of travel that never
    happened. Integrated over a loiter it dominated everything - the first run
    measured 1533 m of movement across a 545 m flight.
    """
    def wrap(a):
        return (a + 180.0) % 360.0 - 180.0
    d_roll = wrap(att_cur[0] - att_prev[0])
    d_pitch = wrap(att_cur[1] - att_prev[1])
    return np.array([ATT_PX_PER_DEG_ROLL_X * d_roll,
                     ATT_PX_PER_DEG_PITCH_Y * d_pitch])


def frame_shift_px(prev: np.ndarray, cur: np.ndarray, *,
                   max_corners: int = 1200, quality: float = 0.01,
                   min_dist: int = 8, min_tracks: int = 30,
                   use_similarity: bool = True):
    """Pixel translation from prev to cur, with a track count.

    Tracking itself is not the hard part - at 110 m over this ground, 1180 to
    1200 of 1200 corners survive the forward-backward check, and the flow field
    is visibly coherent. What matters is how those tracks are reduced to one
    number. A median treats every track alike; fitting a similarity with RANSAC
    separates the ones that agree on a single rigid motion from the moving cars,
    tree shadows and parallax on tall buildings, and that alone took the flight's
    median error from 19.6 m to 13.6 m.

    The fit also yields the frame-to-frame rotation, but integrating it as a
    visual gyro does NOT work: measured against the compass it came out at a
    slope of -0.365 instead of -1, and integrating it drifted 356 degrees over
    one flight. Per-step rotation here is a tenth of a degree, down in the noise,
    and the roll/pitch contribution to apparent rotation is not removed. Yaw
    stays with the magnetometer, which does not need GPS; the map corrects its
    bias, which measured -1.5 deg with a heading-dependent swing of about 3 deg.
    """
    p0 = cv2.goodFeaturesToTrack(prev, max_corners, quality, min_dist)
    if p0 is None or len(p0) < min_tracks:
        return None, 0
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None,
                                         winSize=(31, 31), maxLevel=4,
                                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    if p1 is None:
        return None, 0
    # Backward check: a track that does not return to where it started is junk.
    p0r, st2, _ = cv2.calcOpticalFlowPyrLK(cur, prev, p1, None,
                                           winSize=(31, 31), maxLevel=4)
    if p0r is None:
        return None, 0
    good = (st.ravel() == 1) & (st2.ravel() == 1) & \
           (np.linalg.norm(p0 - p0r, axis=2).ravel() < 1.0)
    if good.sum() < min_tracks:
        return None, int(good.sum())
    src = p0[good].reshape(-1, 1, 2)
    dst = p1[good].reshape(-1, 1, 2)
    if use_similarity:
        M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                             ransacReprojThreshold=1.5)
        if M is not None and inl is not None and int(inl.sum()) >= min_tracks:
            c = np.array([[[prev.shape[1] / 2.0, prev.shape[0] / 2.0]]], np.float32)
            t = cv2.transform(c, M)[0][0] - c[0][0]
            return t.astype(float), int(inl.sum())
    d = (p1 - p0)[good].reshape(-1, 2)
    return np.median(d, axis=0), int(good.sum())


def shift_to_ne(dpx: np.ndarray, *, altitude_m: float, focal_px: float,
                yaw_deg: float) -> tuple[float, float]:
    """Image-plane pixels -> metres North/East, using height for scale.

    Signs are measured, not reasoned about. Regressing flow against GPS over 376
    steps of the 27 Sep flight gave a slope of -1.096 on both axes with
    near-zero cross terms: the axes are right and the scale is right to 10%, but
    the direction was inverted from what the obvious argument ("the ground
    slides backwards as you fly forward") predicts. The mounting is what it is.
    """
    gsd = altitude_m / focal_px
    fwd = dpx[1] * gsd
    right = dpx[0] * gsd
    t = np.radians(yaw_deg)
    north = fwd * np.cos(t) - right * np.sin(t)
    east = fwd * np.sin(t) + right * np.cos(t)
    return float(north), float(east)


class FlowTracker:
    """Keeps its corners between frames instead of finding them again each time.

    Re-detecting every frame was 30 ms of the 60 ms a flow step cost, and it
    buys nothing: tracked over real flight frames, 707 of 714 corners were still
    alive after 59 frames - two seconds at 30 Hz. Detecting once and re-seeding
    only when the population thins takes a step from 60 ms to 31 ms, which is
    the difference between 17 Hz and 32 Hz on one core.

    That is the whole Part 0b answer for this workload: the win was algorithmic,
    not a GPU offload. VPI's CUDA PyrLK does beat OpenCV (16.6 ms against 24 ms
    forward, agreeing to 0.23 px) and is worth taking later, but VPI's Harris
    corners returned 3 points where OpenCV returned 714, so the detector cannot
    move to the GPU as configured.
    """

    def __init__(self, *, max_corners: int = 1200, quality: float = 0.01,
                 min_dist: int = 8, min_tracks: int = 30,
                 redetect_below: float = 0.6, redetect_every: int = 30):
        self.max_corners, self.quality, self.min_dist = max_corners, quality, min_dist
        self.min_tracks = min_tracks
        self.redetect_below = redetect_below
        self.redetect_every = redetect_every
        self.prev = None
        self.pts = None
        self.n_seed = 0
        self.age = 0
        self.redetections = 0

    def _detect(self, img):
        p = cv2.goodFeaturesToTrack(img, self.max_corners, self.quality, self.min_dist)
        self.pts = p
        self.n_seed = 0 if p is None else len(p)
        self.age = 0
        self.redetections += 1

    def step(self, img: np.ndarray):
        """Returns (median pixel shift, track count), or (None, n) if it failed."""
        if self.prev is None or self.pts is None or len(self.pts) < self.min_tracks:
            self._detect(img)
            self.prev = img
            return None, self.n_seed

        p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, img, self.pts, None,
                                             winSize=(31, 31), maxLevel=4)
        if p1 is None:
            self._detect(img); self.prev = img
            return None, 0
        p0r, st2, _ = cv2.calcOpticalFlowPyrLK(img, self.prev, p1, None,
                                               winSize=(31, 31), maxLevel=4)
        good = (st.ravel() == 1) & (st2.ravel() == 1) & \
               (np.linalg.norm(self.pts - p0r, axis=2).ravel() < 1.0)
        n = int(good.sum())
        shift = None
        if n >= self.min_tracks:
            src = self.pts[good].reshape(-1, 1, 2)
            dst = p1[good].reshape(-1, 1, 2)
            M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                                 ransacReprojThreshold=1.5)
            if M is not None and inl is not None and int(inl.sum()) >= self.min_tracks:
                c = np.array([[[img.shape[1] / 2.0, img.shape[0] / 2.0]]], np.float32)
                shift = (cv2.transform(c, M)[0][0] - c[0][0]).astype(float)
                n = int(inl.sum())
            else:
                shift = np.median((p1 - self.pts)[good].reshape(-1, 2), axis=0)

        self.prev = img
        self.age += 1
        surviving = 0 if self.n_seed == 0 else n / self.n_seed
        if (n < self.min_tracks or surviving < self.redetect_below
                or self.age >= self.redetect_every):
            self._detect(img)
        else:
            self.pts = p1[good].reshape(-1, 1, 2)
        return shift, n
