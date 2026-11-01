"""Locate a nadir frame on the map by CORRELATION, not by features.

Restored 25 Oct, at the pilot's instruction, after being removed on 20 Oct.
Nothing was downloaded: NCC is cv2.matchTemplate, which OpenCV already has.

WHY IT IS BACK, and it is the reason the original docstring gave for
correlating EDGES rather than intensity - "edges survive the difference in
brightness, season and sun angle far better". On 25 Oct that stopped being a
remark and became the flight:

    flight 154, 11:38   XFeat accepted 78% of frames
    flight 160, 15:20   XFeat accepted  0% of 40, every one "only N inliers"

Same site, same camera, same code, and the late frames were the SHARPER of
the two (Laplacian variance 640 against 261). What moved was the sun. The
dominant dark-edge orientation is 128 deg in bhh.tif, 102 deg at 11:38 and
28 deg at 15:20 - a hundred degrees of shadow rotation away from the map.
XFeat matches descriptors, and a descriptor built on a shadow edge is a
different descriptor when the shadow swings. Correlation of gradient
magnitude does not care which side of the tree is dark.

The aircraft already knows two of the three unknowns: altitude gives scale
and yaw gives rotation, both from the FC. Once the query is rescaled and
rotated north-up only a TRANSLATION is left, and a normalised cross
correlation over the search window solves that directly - no descriptors, so
appearance change costs contrast rather than correspondences.

The geometry is NOT duplicated here any more. This module used to carry its
own copy of north_up_query; the shared one in map_query grew the east-west
anisotropy correction on 19 Oct and the copy would not have it, so the copy
is gone and map_query.north_up_query is called instead.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cv2

from map_query import north_up_query


@dataclass
class NccResult:
    """Shaped to match XFeatResult, because map_fix_worker reads both the
    same way - ok/lon/lat/score/margin/reason, plus the counters it logs."""
    ok: bool
    lon: float = 0.0
    lat: float = 0.0
    score: float = 0.0          # correlation peak, 1.0 = perfect
    margin: float = 0.0         # peak minus the best rival: confidence
    reason: str = ""
    inliers: int = 0            # not meaningful for correlation; kept for the log
    total_matches: int = 0
    altitude_m: float = 0.0


def _correlate(window: np.ndarray, templ: np.ndarray, *, pyramid: bool = True,
               pad: int = 12):
    """Normalised cross-correlation peak, coarse-to-fine.

    A full-resolution matchTemplate is 97% of the cost of a fix and it scales
    with window area times template area: a +-300 m window takes 253 ms for one
    rotation. Halving both images first, finding the peak there, then rechecking
    only a small box around it at full resolution gives the SAME peak - measured
    at zero pixels of difference across every window size - for a quarter of the
    time.

    Worth recording that this beat the GPU. VPI's CUDA templateMatching came in
    at 68 ms against the pyramid's 61 ms at +-300 m, and 15.8 against 10.0 at
    +-150 m, so the offload the plan expected to need was measured and rejected.
    """
    if not pyramid or min(window.shape) < 4 * pad or min(templ.shape) < 32:
        r = cv2.matchTemplate(window, templ, cv2.TM_CCOEFF_NORMED)
        _, peak, _, loc = cv2.minMaxLoc(r)
        return peak, loc, r
    wh = cv2.resize(window, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
    th = cv2.resize(templ, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
    if wh.shape[0] <= th.shape[0] or wh.shape[1] <= th.shape[1]:
        r = cv2.matchTemplate(window, templ, cv2.TM_CCOEFF_NORMED)
        _, peak, _, loc = cv2.minMaxLoc(r)
        return peak, loc, r
    rh = cv2.matchTemplate(wh, th, cv2.TM_CCOEFF_NORMED)
    _, _, _, lh = cv2.minMaxLoc(rh)
    x0 = max(0, lh[0] * 2 - pad); y0 = max(0, lh[1] * 2 - pad)
    x1 = min(window.shape[1], x0 + templ.shape[1] + 2 * pad)
    y1 = min(window.shape[0], y0 + templ.shape[0] + 2 * pad)
    sub = window[y0:y1, x0:x1]
    if sub.shape[0] <= templ.shape[0] or sub.shape[1] <= templ.shape[1]:
        r = cv2.matchTemplate(window, templ, cv2.TM_CCOEFF_NORMED)
        _, peak, _, loc = cv2.minMaxLoc(r)
        return peak, loc, r
    rf = cv2.matchTemplate(sub, templ, cv2.TM_CCOEFF_NORMED)
    _, peak, _, l = cv2.minMaxLoc(rf)
    # The confidence margin needs rival peaks from the WHOLE window, which only
    # the coarse surface has, so it is measured there and rescaled.
    return peak, (l[0] + x0, l[1] + y0), rh


def _edges(img: np.ndarray, notch: bool = False) -> np.ndarray:
    """Gradient magnitude. Correlating intensity across a four-year gap and a
    different sensor is weak; edges survive the difference in brightness,
    season and sun angle far better."""
    if notch:
        img = _denotch(img)
    eq = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(img)
    gx = cv2.Sobel(eq, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(eq, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.normalize(cv2.magnitude(gx, gy), None, 0, 255,
                         cv2.NORM_MINMAX).astype(np.uint8)


def _denotch(img: np.ndarray, n_peaks: int = 6) -> np.ndarray:
    """Remove the strongest PERIODIC components before correlating.

    This ground is ploughed, and both the sheet and the camera see it as
    parallel furrows. A periodic pattern correlates with itself at every
    multiple of its own period, so the correlation surface grows a row of
    rival peaks that are as tall as the true one and the matcher locks onto
    the wrong furrow. That is what flight 378 measured on the sheet that does
    cover the site: NCC passed 57% of 2069 attempts and landed 33.9 m from
    the receiver, with only 2.2% inside 10 m, and no margin threshold could
    separate the good from the bad - the top 1% by margin still sat at
    35.9 m, and corr(margin, error) was -0.155. A matcher that cannot tell
    when it is wrong is the whole problem.

    Periodicity is a few bright points in the Fourier plane, so it can be
    taken out there: find the n_peaks strongest components away from DC,
    blank each one and its conjugate, and transform back. What is left is the
    non-repeating structure - field boundaries, tracks, buildings - which is
    what actually says WHERE.

    Measured on 180 frames of flight 378 against demo/bam1.tif, in one
    harness, against the receiver:

        plain gradient   err p50 51.7 m    at 39 m: 58.7 m
        notch + gradient err p50 38.9 m    at 39 m: 15.0 m

    The low leg is the one that moves, and it moves 3.9x. That is why this
    exists. It is NOT enough on its own - 11% of frames inside 10 m against
    8% - so it is a step, not the answer.

    DC is kept. The 3-pixel block around the centre is excluded from the
    search, so the mean and the slowest shading survive; removing those
    would flatten the image, not sharpen its structure.
    """
    f = np.fft.fftshift(np.fft.fft2(img.astype(np.float32)))
    mag = np.abs(f).copy()
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    mag[cy - 3:cy + 4, cx - 3:cx + 4] = 0.0
    yy, xx = np.ogrid[:h, :w]
    for _ in range(n_peaks):
        iy, ix = np.unravel_index(int(np.argmax(mag)), mag.shape)
        for sy, sx in ((iy, ix), (2 * cy - iy, 2 * cx - ix)):
            f[(yy - sy) ** 2 + (xx - sx) ** 2 <= 16] = 0.0
        mag[max(iy - 5, 0):iy + 6, max(ix - 5, 0):ix + 6] = 0.0
    out = np.abs(np.fft.ifft2(np.fft.ifftshift(f)))
    return cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def locate(img_grey: np.ndarray, map_window: np.ndarray, geo_window, *,
           altitude_m: float, focal_px: float, yaw_deg: float, map_gsd_m: float,
           map_gsd_x_m: float | None = None,
           yaw_search_deg=(-6.0, -3.0, 0.0, 3.0, 6.0),
           min_score: float = 0.05, min_margin: float = 0.01,
           window_is_edges: bool = False, pyramid: bool = True,
           min_inliers: int = 0, principal_xy=None,
           query_shape: str = "full", notch: bool = False) -> NccResult:
    """Correlate one frame against a map window. geo_window maps window px -> lon/lat.

    yaw_search_deg covers compass error: the FC heading is good to a few
    degrees, and a few degrees of template rotation costs real correlation.
    It also absorbs the camera's own mount angle, which is why
    camera_yaw_offset_deg matters far less here than it does to XFeat.

    The thresholds suit the pyramid path, where the margin is read off the
    coarse correlation surface and so sits on a different scale than it did at
    full resolution. Swept over 50 frames with a +-150 m window: a margin of
    0.01 keeps 29 of them and every single one lands within 30 m, while 0.03
    keeps only 19 for no gain in accuracy. min_score is low because gradient
    correlation across a four-year-old map peaks around 0.1-0.2 - the margin
    is what actually separates a good fix from a bad one, not the peak height.

    min_inliers and principal_xy are accepted and ignored. They are XFeat's
    vocabulary, and map_fix_worker passes the same kwargs to whichever matcher
    is selected; refusing them here would make the choice a code change
    instead of a setting.
    """
    if not (altitude_m > 0.0):
        return NccResult(False, reason=f"altitude {altitude_m:.1f} m is not "
                                       f"above the ground",
                         altitude_m=float(altitude_m))
    W = map_window if window_is_edges else _edges(map_window, notch)
    peak, loc, q, r, scale = -9.0, None, None, None, 1
    for d in yaw_search_deg:
        t = north_up_query(img_grey, altitude_m=altitude_m, focal_px=focal_px,
                           yaw_deg=yaw_deg - d, map_gsd_m=map_gsd_m,
                           map_gsd_x_m=map_gsd_x_m, shape=query_shape)
        if isinstance(t, tuple):
            t = t[0]
        if t.shape[0] < 16 or W.shape[0] <= t.shape[0] or W.shape[1] <= t.shape[1]:
            continue
        pk, lc, rr = _correlate(W, _edges(t, notch), pyramid=pyramid)
        if pk > peak:
            peak, loc, q, r, scale = pk, lc, t, rr, (2 if pyramid else 1)
    if loc is None:
        return NccResult(False, reason="template does not fit the search window",
                         altitude_m=float(altitude_m))

    # Confidence: how far the winning peak stands above the best rival outside
    # its own neighbourhood. A flat correlation surface means "could be
    # anywhere". r is the coarse surface when the pyramid ran, so the peak
    # neighbourhood to blank out is in coarse coordinates too.
    rival = r.copy()
    k = max(1, max(q.shape) // (2 * scale))
    cx, cy = loc[0] // scale, loc[1] // scale
    x0, y0 = max(0, cx - k), max(0, cy - k)
    rival[y0:cy + k, x0:cx + k] = -1.0
    margin = float(peak - rival.max())

    px = loc[0] + q.shape[1] / 2.0
    py = loc[1] + q.shape[0] / 2.0
    lon, lat = geo_window.to_lonlat(px, py)
    if peak < min_score or margin < min_margin:
        return NccResult(False, float(lon), float(lat), float(peak), margin,
                         reason=f"weak peak {peak:.2f} margin {margin:.3f}",
                         altitude_m=float(altitude_m))
    return NccResult(True, float(lon), float(lat), float(peak), margin,
                     altitude_m=float(altitude_m))
