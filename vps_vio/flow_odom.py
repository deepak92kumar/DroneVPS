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

# THE GYRO'S OWN COEFFICIENTS, and they are not the two above.
#
# Gyro derotation is not a new idea on this project - it was built, flown and
# carried at 95% from 28 Sep onwards, and the rebuild lost it. The 3 Oct report
# gives the numbers and how they were got:
#
#   "Roll re-measured on hover only, pooled over 15 flights and 39723 samples:
#    -14.8 px/deg, steady when restricted to the strongest rotations, so
#    -13.82 was 6% short. Pitch deliberately left at 13.82 - its fit climbs
#    toward theory as signal grows, which is attenuation, not gain."
#
# 13.82 is the geometry: 791.72 * tan(1 deg). So roll is 7% above theory and
# measured that way over 39723 samples, and pitch is theory on purpose.
#
# THE SIGN IS THE WHOLE POINT. These are subtracted, exactly as the FC's are,
# so a negative coefficient is the inversion. Measured on the 20-21 Oct
# recordings against the FC's own attitude over matched intervals:
# r = -0.994 to -0.998 and scale -0.983 to -1.008 on both axes, across three
# flights. The camera gyro tracks the aircraft to one or two percent and points
# the other way.
#
# Replayed through the whole velocity path on the 21 Oct flight, against the
# real receiver's ground speed, 11398 samples above 5 m:
#
#     FC attitude (what flew)      error p50 0.725  p95 3.576  1.9% over 5 m/s
#     gyro, these coefficients     error p50 0.155  p95 1.243  0.0% over 5 m/s
#
# The coefficient barely matters - 13.82/13.82 gives p50 0.154 - and one flight
# cannot separate them. 15 flights and 39723 samples can, so the report wins.
GYRO_PX_PER_DEG_ROLL_X = -14.80
GYRO_PX_PER_DEG_PITCH_Y = -13.82


class TiltValidator:
    """Decide at RUN TIME whether a tilt source may be used, and with which sign.

    This exists because of 21 Oct. Derotation was switched to the camera gyro on
    a static measurement - gyro against the FC's attitude over matched
    intervals, r = -0.994, scale -0.99 - and the sign that measurement implied
    was hard-coded. Flown, the reported velocity came out 160 degrees from the
    truth with east r = -0.884, and in Loiter the aircraft ran away: told it was
    moving one way while it drifted the other, the controller pushed it further.
    Position error grew 0.5 -> 21.4 m and 18.2 -> 43.3 m across two VPS windows
    of log 00000223.

    The pre-SSD-loss daemon never had that hole, and this is its logic:

        "Compare ~0.4 s windows of gyro vs rotation-vector tilt. Gyro
         derotation turns ON only when both axes agree (>=15 moving windows,
         |corr|>=0.85, 0.7<=|slope|<=2.0) and OFF if agreement is lost."

    and it DERIVED the sign rather than assuming it:

        gyro_derot.update(ok=True, sign_r=sign(verdict[0][0]),
                                   sign_p=sign(verdict[1][0]))

    The reference here is the FC's attitude, because that is the one source
    measured to track the real motion on this airframe: velocity slope +0.795
    north and +0.892 east against the raw receiver, 33 degrees from the truth,
    73% of samples in the right half-plane. It is a poor source per FRAME - it
    arrives at ~4 Hz against a 30 Hz loop - which is exactly why the windows are
    0.4 s long: over that span the FC has published once or twice and the
    comparison is fair, while a single frame's comparison is not.

    Thresholds are the daemon's, unchanged. ON is deliberately harder than OFF:
    |corr| 0.85 to arm, 0.6 to stay, so a source does not flicker.
    """

    WINDOW_S = 0.4
    MIN_WINDOWS = 15
    MIN_REF_DEG = 0.005      # a window with no rotation in it says nothing
    ON_CORR = 0.85
    OFF_CORR = 0.60
    SLOPE_LO, SLOPE_HI = 0.7, 2.0
    MAX_PAIRS = 200          # rolling, so a source can lose agreement later

    def __init__(self, name: str = ""):
        self.name = name
        self.ok = False
        self.sign_roll = 0.0
        self.sign_pitch = 0.0
        self.verdict = None          # ((slope, corr) roll, (slope, corr) pitch)
        self.n_windows = 0
        self.turned_on = 0
        self.turned_off = 0
        self._acc = [0.0, 0.0, 0.0, 0.0, 0.0]
        self._pairs = []

    def update(self, cand_roll, cand_pitch, ref_roll, ref_pitch, dt_s):
        """One frame of both sources, in degrees across that frame."""
        if None in (cand_roll, cand_pitch, ref_roll, ref_pitch) or not dt_s:
            return
        a = self._acc
        a[0] += float(cand_roll); a[1] += float(cand_pitch)
        a[2] += float(ref_roll);  a[3] += float(ref_pitch)
        a[4] += float(dt_s)
        if a[4] < self.WINDOW_S:
            return
        self._pairs.append((a[0], a[1], a[2], a[3]))
        del self._pairs[:-self.MAX_PAIRS]
        self._acc = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.n_windows = len(self._pairs)
        v = []
        for ci, ri in ((0, 2), (1, 3)):
            pts = [(p[ci], p[ri]) for p in self._pairs
                   if abs(p[ri]) > self.MIN_REF_DEG]
            if len(pts) < self.MIN_WINDOWS:
                return
            c = np.array([p[0] for p in pts], float)
            r = np.array([p[1] for p in pts], float)
            denom = float(np.dot(r, r))
            if denom <= 0.0 or c.std() == 0.0 or r.std() == 0.0:
                return
            v.append((float(np.dot(c, r) / denom), float(np.corrcoef(c, r)[0, 1])))
        self.verdict = tuple(v)
        agree = all(abs(corr) >= self.ON_CORR
                    and self.SLOPE_LO <= abs(sl) <= self.SLOPE_HI
                    and np.sign(sl) == np.sign(corr) for sl, corr in v)
        if agree and not self.ok:
            self.ok = True
            self.sign_roll = float(np.sign(v[0][0]))
            self.sign_pitch = float(np.sign(v[1][0]))
            self.turned_on += 1
        elif self.ok:
            keep = all(abs(corr) >= self.OFF_CORR for _sl, corr in v) and \
                np.sign(v[0][0]) == self.sign_roll and \
                np.sign(v[1][0]) == self.sign_pitch
            if not keep:
                self.ok = False
                self.turned_off += 1

    def describe(self) -> str:
        if self.verdict is None:
            return f"{self.name}: no verdict yet ({self.n_windows} windows)"
        (sr, cr), (sp, cp) = self.verdict
        return (f"{self.name}: {'ON' if self.ok else 'off'} "
                f"roll slope {sr:+.2f} corr {cr:+.2f}, "
                f"pitch slope {sp:+.2f} corr {cp:+.2f}, "
                f"{self.n_windows} windows, on {self.turned_on} off {self.turned_off}")


def gyro_shift_px(d_roll_deg: float, d_pitch_deg: float,
                  sign_roll: float = 1.0, sign_pitch: float = 1.0) -> np.ndarray:
    """Image shift caused purely by tilting, from the CAMERA's own gyro.

    Same job as attitude_shift_px and a different source. The FC's attitude
    arrives at about 4 Hz against a 30 Hz loop, so on 88% of the 21 Oct
    flight's frames it reported no rotation at all - and on the few where it
    did, it reported several frames' worth and that got subtracted from one.
    The 30 Sep report has the extreme case: "the FC's attitude had sat frozen
    for four frames and then jumped 4.67 deg of roll and 11.82 of pitch in one,
    so the derotation subtracted 167 px from a 1 px shift - 802 m/s."
    Measured on the 21 Oct flight, FC-attitude derotation is WORSE than no
    derotation at all: per-frame speed error p50 2.97 against 2.50 raw.

    The gyro sees every frame, at 100 Hz, and the caller hands in its integral
    across exactly this frame interval.

    Only the DIFFERENCE matters here, never an absolute angle, which is why the
    gyro's unbounded integration drift costs nothing: a slowly accumulating
    offset cancels in a difference. It is also why GAME_ROTATION_VECTOR, which
    is bounded, buys nothing - it is logged beside this for comparison and not
    used.
    """
    # The MAGNITUDE is the 3 Oct report's; the SIGN is whatever TiltValidator
    # measured at run time. Hard-coding the sign is the mistake that ran the
    # aircraft away on 21 Oct, so the constants carry magnitude only and the
    # caller supplies what it has verified.
    return np.array([abs(GYRO_PX_PER_DEG_ROLL_X) * float(sign_roll) * float(d_roll_deg),
                     abs(GYRO_PX_PER_DEG_PITCH_Y) * float(sign_pitch) * float(d_pitch_deg)])


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
                yaw_deg: float, focal_y_px: float | None = None) -> tuple[float, float]:
    """Image-plane pixels -> metres North/East, using height for scale.

    Signs are measured, not reasoned about. Regressing flow against GPS over 376
    steps of the 27 Sep flight gave a slope of -1.096 on both axes with
    near-zero cross terms: the axes are right and the scale is right to 10%, but
    the direction was inverted from what the obvious argument ("the ground
    slides backwards as you fly forward") predicts. The mounting is what it is.
    """
    # fx for the x axis, fy for the y axis. On this camera they differ by
    # 0.018% - 1.8 cm per 100 m - so this is correctness, not a cure.
    fy = focal_px if focal_y_px is None else focal_y_px
    fwd = dpx[1] * (altitude_m / fy)
    # THE MINUS IS THE 21 OCT FIX, and it is the whole of the runaway.
    #
    # The sign above came from "regressing flow against GPS over 376 steps of
    # the 27 Sep flight", which is a different airframe and a camera that has
    # been remounted since - accel_to_ne's own docstring says so. It was never
    # re-measured, and it was wrong on this mount for the sideways axis only.
    #
    # Forward motion was therefore fine and sideways motion came out backwards,
    # which is exactly why this took three flights to see: how broken a flight
    # looked depended entirely on how much of it was flown sideways. Scored on
    # the direction of the reported velocity against the raw receiver:
    #
    #   flight   sideways   as it was        with this minus
    #   221        48%      33 deg, 73%      7 deg,  94%
    #   223        98%     160 deg,  8%     10 deg,  99%
    #   224        57%      21 deg, 62%      7 deg,  97%
    #
    # Log 223 is the one that gives it away: 98% of its energy was sideways,
    # the aircraft was flying 83 degrees off its own heading, and the reported
    # velocity pointed 160 degrees from the truth. In Loiter that is a runaway
    # and it ran away - position error 0.5 -> 21.4 m in fifteen seconds.
    #
    # Two whole days of derotation work were spent shuffling this error around:
    # the FC attitude, the camera gyro, the fused orientation, a run-time
    # validator for the sign. None of them could have fixed it, because the
    # tilt correction and the axis sign are different things and only one of
    # them was wrong.
    right = -dpx[0] * (altitude_m / focal_px)
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

    300 corners, not 1200. Swept over 900 flight frames the drift is flat from
    150 to 1200 - 6.8, 6.9, 6.8, 6.1 m - while the cost runs 23, 32, 47, 78 ms.
    Tracking more points buys precision the rest of the chain cannot use.

    That is the whole Part 0b answer for this workload: the win was algorithmic,
    not a GPU offload. VPI's CUDA PyrLK does beat OpenCV (16.6 ms against 24 ms
    forward, agreeing to 0.23 px) and is worth taking later, but VPI's Harris
    corners returned 3 points where OpenCV returned 714, so the detector cannot
    move to the GPU as configured.

    K and D are the camera's own intrinsics, and passing them is not a detail.
    Read off this OAK-D on 20 Oct at 1280x800: the principal point is cx 624.89,
    cy 391.80, which is 15.11 px and 8.20 px away from the geometric centre this
    used to evaluate the motion at. The fitted affine carries rotation as well
    as translation, so evaluating it at the wrong point turns rotation into
    phantom translation - 0.32 px at 1 deg/frame, which at 100 m and 17 Hz is
    0.64 m/s of speed the aircraft never had. The project's own unexplained
    hover phantom is 0.36-0.43 m/s, the same size.

    The distortion matters for the same reason and more: D is the 8-coefficient
    rational model and the coefficients are large (k1 24.09, k2 -39.14,
    k3 18.88), so a displacement measured near the edge of the frame does not
    mean the same number of metres as one measured in the middle. The corners
    are tracked on the raw image - LK has to be - but the affine is fitted on
    undistorted copies of them, which is where the metres come from.

    Without K and D it still runs, on the geometric centre and the raw pixels,
    and says so. That path is what every flight before 20 Oct used.
    """

    def __init__(self, *, max_corners: int = 300, quality: float = 0.01,
                 min_dist: int = 8, min_tracks: int = 30,
                 redetect_below: float = 0.6, redetect_every: int = 30,
                 K=None, D=None):
        self.max_corners, self.quality, self.min_dist = max_corners, quality, min_dist
        self.min_tracks = min_tracks
        self.K = None if K is None else np.asarray(K, np.float64).reshape(3, 3)
        self.D = None if D is None else np.asarray(D, np.float64).reshape(1, -1)
        self.centre = (None if self.K is None
                       else np.array([self.K[0, 2], self.K[1, 2]], float))
        self.redetect_below = redetect_below
        self.redetect_every = redetect_every
        self.prev = None
        self.pts = None
        self.n_seed = 0
        self.age = 0
        self.redetections = 0
        self.depth_med_m = None
        self.depth_dropped = 0

    def _undistort(self, pts):
        """Raw pixel points -> undistorted pixel points, in the same K units.

        P=K keeps the result in pixels at the same focal length, so everything
        downstream that divides by focal_px still means what it meant.
        """
        if self.K is None or self.D is None:
            return pts
        return cv2.undistortPoints(pts.astype(np.float64), self.K, self.D,
                                   P=self.K).astype(np.float32)

    def _detect(self, img):
        p = cv2.goodFeaturesToTrack(img, self.max_corners, self.quality, self.min_dist)
        self.pts = p
        self.n_seed = 0 if p is None else len(p)
        self.age = 0
        self.redetections += 1

    # Below this the stereo pair is worth listening to, above it the disparity
    # is smaller than its own noise. From the 7.5 cm baseline and fx 791.72:
    # 0.2 px of disparity noise is 3.4% of range at 10 m, 8.4% at 25 m and
    # 20.2% at 60 m, where the disparity itself has fallen under 1 px.
    DEPTH_MAX_M = 30.0
    # A track has to be this far off the surface everyone else is on before it
    # is thrown away: 1 m, or three times the depth noise at that range,
    # whichever is larger. Tight enough to drop a rooftop, loose enough not to
    # fight the sensor's own scatter.
    DEPTH_TOL_M = 1.0
    DEPTH_SIGMA_K = 3.0
    DEPTH_MIN_KEEP = 30
    # Fractions, and they are what the first live run taught. Sitting on the
    # floor the pair reported 1.3-1.8 m and the gate threw away 190 of 300
    # tracks - 63% - because at that range the depth map is mostly noise and
    # unmatched pixels. An absolute floor of 30 survivors passed that happily.
    # A gate that drops most of the frame is not finding a rooftop, it is
    # failing, and it has to say so instead of making the flow worse.
    DEPTH_MIN_VALID_FRAC = 0.5    # of the tracks must have a depth at all
    DEPTH_MIN_KEEP_FRAC = 0.6     # of the tracks must survive the gate

    def _depth_gate(self, pts, depth):
        """Which tracks sit on the same surface as the majority. None = no gate.

        shift_to_ne converts pixels to metres by dividing by ONE altitude, so
        it assumes every tracked feature lies on one plane. A feature on a roof
        or a treetop breaks that silently: it is a perfectly consistent track,
        RANSAC is happy with it, and it is simply scaled wrong. That is the
        "parallax on tall buildings" this module's own notes blame, and the
        stereo pair can see it directly where the single altitude cannot.
        """
        if depth is None or len(pts) < self.DEPTH_MIN_KEEP:
            return None
        h, w = depth.shape[:2]
        xy = pts.reshape(-1, 2)
        xi = np.clip(xy[:, 0].astype(np.int32), 0, w - 1)
        yi = np.clip(xy[:, 1].astype(np.int32), 0, h - 1)
        z = depth[yi, xi].astype(np.float32) / 1000.0     # mm -> m
        have = z > 0.0                                     # 0 means no match
        if (have.sum() < self.DEPTH_MIN_KEEP
                or have.mean() < self.DEPTH_MIN_VALID_FRAC):
            return None
        med = float(np.median(z[have]))
        if not (0.0 < med < self.DEPTH_MAX_M):
            return None
        tol = max(self.DEPTH_TOL_M,
                  self.DEPTH_SIGMA_K * med * med * 0.2 / (791.723 * 0.075))
        keep = have & (np.abs(z - med) <= tol)
        # Refuse to gate rather than gate down to nothing: too few survivors
        # means the surface assumption has failed, not that the tracks are bad.
        if (keep.sum() < self.DEPTH_MIN_KEEP
                or keep.mean() < self.DEPTH_MIN_KEEP_FRAC):
            return None
        return keep, med, tol

    def step(self, img: np.ndarray, depth: np.ndarray | None = None):
        """Returns (median pixel shift, track count), or (None, n) if it failed.

        depth is the stereo pair's range map, aligned to this image and in
        millimetres, used only to drop tracks that are not on the same surface
        as the rest. It is NOT a height source - that was tried on 9 Oct and
        came out worse than the barometer - and nothing here divides by it.
        """
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
        # One surface, if the stereo pair can tell us which one.
        self.depth_med_m = None
        self.depth_dropped = 0
        g = self._depth_gate(self.pts[good], depth)
        if g is not None:
            keep, self.depth_med_m, _tol = g
            full = np.zeros(len(good), bool)
            full[np.flatnonzero(good)[keep]] = True
            self.depth_dropped = int(good.sum() - full.sum())
            good = full

        n = int(good.sum())
        shift = None
        if n >= self.min_tracks:
            src = self.pts[good].reshape(-1, 1, 2)
            dst = p1[good].reshape(-1, 1, 2)
            # Undistorted copies for the fit only. The tracks themselves stay in
            # raw pixels because that is the image LK ran on; the metres come
            # out of the fit, so the fit is where the lens has to be removed.
            u_src, u_dst = self._undistort(src), self._undistort(dst)
            M, inl = cv2.estimateAffinePartial2D(u_src, u_dst, method=cv2.RANSAC,
                                                 ransacReprojThreshold=1.5)
            if M is not None and inl is not None and int(inl.sum()) >= self.min_tracks:
                # At the OPTICAL centre, not the geometric one. The affine
                # carries rotation, and rotation evaluated off-centre reads as
                # translation that never happened.
                c = (self.centre if self.centre is not None
                     else np.array([img.shape[1] / 2.0, img.shape[0] / 2.0]))
                cc = c.reshape(1, 1, 2).astype(np.float32)
                shift = (cv2.transform(cc, M)[0][0] - c).astype(float)
                n = int(inl.sum())
            else:
                shift = np.median((u_dst - u_src).reshape(-1, 2), axis=0)

        self.prev = img
        self.age += 1
        surviving = 0 if self.n_seed == 0 else n / self.n_seed
        if (n < self.min_tracks or surviving < self.redetect_below
                or self.age >= self.redetect_every):
            self._detect(img)
        else:
            self.pts = p1[good].reshape(-1, 1, 2)
        return shift, n
