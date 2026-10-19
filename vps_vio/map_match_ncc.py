"""Locate a nadir frame on the satellite map by correlation, not by features.

AKAZE/SIFT matching against this map failed on real flight imagery: the 2022
satellite tiles and today's camera simply do not produce comparable
descriptors, even though a side-by-side shows the same buildings and roads.

But the aircraft already knows two of the three unknowns. Altitude gives scale
(footprint = width * alt / focal) and yaw gives rotation, both from the FC. Once
the query is rescaled and rotated north-up, only a TRANSLATION is left, and a
normalised cross-correlation over a search window solves that directly - no
descriptors, so appearance change costs contrast, not correspondences.

The search window comes from a prior: VIO in flight, GPS when measuring
accuracy offline. A window of +-300 m at 0.5 m/px is a 1200x1200 search, which
is a few milliseconds on this hardware.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cv2


@dataclass
class NccResult:
    ok: bool
    lon: float = 0.0
    lat: float = 0.0
    score: float = 0.0          # correlation peak, 1.0 = perfect
    margin: float = 0.0         # peak minus the best rival peak: confidence
    reason: str = ""


def north_up_query(img_grey: np.ndarray, *, altitude_m: float, focal_px: float,
                   yaw_deg: float, map_gsd_m: float,
                   map_gsd_x_m: float | None = None,
                   keep: float = 0.95) -> np.ndarray:
    """Rescale to the map's GSD and rotate north-up, then keep the centre.

    Rotation leaves black corners. Cropping them away entirely (keep 0.5-0.62)
    measured far worse than tolerating them - context is what makes the peak
    unambiguous, and 12 frames went from 2/12 to 9/12 within 30 m as keep rose
    from 0.50 to 0.95. Rotating the map instead, to give a template with no
    holes at all, was worse again: it blurs the map and widens the search.
    """
    scale = (altitude_m / float(focal_px)) / map_gsd_m
    q = cv2.resize(img_grey, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = q.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -yaw_deg, 1.0)
    q = cv2.warpAffine(q, M, (w, h))

    # The map's pixels are not square IN METRES, and that is a 16.5% error here.
    #
    # A geographic (EPSG:4326) GeoTIFF has equal pixel size in DEGREES, but a
    # degree of longitude is cos(latitude) of a degree of latitude - 0.858 at
    # 30.9 N. So this map is 0.5092 m/px north-south and 0.4369 m/px east-west,
    # while MapCache.gsd carries only the north-south figure and the scale
    # above applied it to both axes.
    #
    # 1/cos(30.89) = 1.1653, which is the 16.5% the 12 Oct report found behind
    # XFEAT_ANISO and "16 m that has never been explained". Over a 178 m
    # footprint at 110 m, a 16.5% east-west scale error is about 15 m.
    #
    # The correction has to come AFTER the rotation, not before: the anisotropy
    # lives in the map's axes, not the camera's, so the frame has to be north-up
    # before x means east. With an isotropic map the order does not matter and
    # this is skipped entirely, which is why it went unnoticed.
    if map_gsd_x_m and abs(map_gsd_x_m - map_gsd_m) > 1e-9:
        # Map pixels are SMALLER in metres east-west, so the same ground
        # distance spans MORE of them: stretch x.
        q = cv2.resize(q, None, fx=map_gsd_m / float(map_gsd_x_m), fy=1.0,
                       interpolation=cv2.INTER_LINEAR)
        h, w = q.shape

    s = int(min(h, w) * keep) // 2
    return q[h // 2 - s:h // 2 + s, w // 2 - s:w // 2 + s]


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


def _edges(img: np.ndarray) -> np.ndarray:
    """Gradient magnitude. Correlating intensity across a four-year gap and a
    different sensor is weak; edges survive the difference in brightness,
    season and sun angle far better."""
    eq = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(img)
    gx = cv2.Sobel(eq, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(eq, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.normalize(cv2.magnitude(gx, gy), None, 0, 255,
                         cv2.NORM_MINMAX).astype(np.uint8)


def locate(img_grey: np.ndarray, map_window: np.ndarray, geo_window, *,
           altitude_m: float, focal_px: float, yaw_deg: float, map_gsd_m: float,
           map_gsd_x_m: float | None = None,
           yaw_search_deg=(-6.0, -3.0, 0.0, 3.0, 6.0),
           min_score: float = 0.05, min_margin: float = 0.01,
           window_is_edges: bool = False, pyramid: bool = True) -> NccResult:
    """Correlate one frame against a map window. geo_window maps window px -> lon/lat.

    yaw_search_deg covers compass error: the FC heading is good to a few degrees,
    and a few degrees of template rotation costs real correlation.

    The thresholds suit the pyramid path, where the margin is read off the
    coarse correlation surface and so sits on a different scale than it did at
    full resolution. Swept over 50 frames with a +-150 m window: a margin of
    0.01 keeps 29 of them and every single one lands within 30 m, while 0.03
    keeps only 19 for no gain in accuracy. min_score is low because gradient
    correlation across a four-year-old map peaks around 0.1-0.2 - the margin is
    what actually separates a good fix from a bad one, not the peak height.
    """
    # MapCache hands over a window whose edges were computed once at startup.
    W = map_window if window_is_edges else _edges(map_window)
    peak, loc, q = -9.0, None, None
    for d in yaw_search_deg:
        t = north_up_query(img_grey, altitude_m=altitude_m, focal_px=focal_px,
                           yaw_deg=yaw_deg - d, map_gsd_m=map_gsd_m,
                           map_gsd_x_m=map_gsd_x_m)
        if t.shape[0] < 16 or W.shape[0] <= t.shape[0] or W.shape[1] <= t.shape[1]:
            continue
        pk, lc, rr = _correlate(W, _edges(t), pyramid=pyramid)
        if pk > peak:
            peak, loc, q, r, scale = pk, lc, t, rr, (2 if pyramid else 1)
    if loc is None:
        return NccResult(False, reason="template does not fit the search window")

    # Confidence: how far the winning peak stands above the best rival outside
    # its own neighbourhood. A flat correlation surface means "could be anywhere".
    # r is the coarse surface when the pyramid ran, so the peak neighbourhood
    # to blank out is in coarse coordinates too.
    rival = r.copy()
    k = max(1, max(q.shape) // (2 * scale))
    cx, cy = loc[0] // scale, loc[1] // scale
    x0, y0 = max(0, cx - k), max(0, cy - k)
    rival[y0:cy + k, x0:cx + k] = -1.0
    margin = float(peak - rival.max())

    cx = loc[0] + q.shape[1] / 2.0
    cy = loc[1] + q.shape[0] / 2.0
    lon, lat = geo_window.to_lonlat(cx, cy)
    if peak < min_score or margin < min_margin:
        return NccResult(False, float(lon), float(lat), float(peak), margin,
                         reason=f"weak peak {peak:.2f} margin {margin:.3f}")
    return NccResult(True, float(lon), float(lat), float(peak), margin)
