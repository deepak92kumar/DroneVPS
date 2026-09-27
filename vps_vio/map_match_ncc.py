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
    s = int(min(h, w) * keep) // 2
    return q[h // 2 - s:h // 2 + s, w // 2 - s:w // 2 + s]


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
           yaw_search_deg=(-6.0, -3.0, 0.0, 3.0, 6.0),
           min_score: float = 0.25, min_margin: float = 0.03) -> NccResult:
    """Correlate one frame against a map window. geo_window maps window px -> lon/lat.

    yaw_search_deg covers compass error: the FC heading is good to a few degrees,
    and a few degrees of template rotation costs real correlation.
    """
    W = _edges(map_window)
    peak, loc, q = -9.0, None, None
    for d in yaw_search_deg:
        t = north_up_query(img_grey, altitude_m=altitude_m, focal_px=focal_px,
                           yaw_deg=yaw_deg - d, map_gsd_m=map_gsd_m)
        if t.shape[0] < 16 or W.shape[0] <= t.shape[0] or W.shape[1] <= t.shape[1]:
            continue
        rr = cv2.matchTemplate(W, _edges(t), cv2.TM_CCOEFF_NORMED)
        _, pk, _, lc = cv2.minMaxLoc(rr)
        if pk > peak:
            peak, loc, q, r = pk, lc, t, rr
    if loc is None:
        return NccResult(False, reason="template does not fit the search window")

    # Confidence: how far the winning peak stands above the best rival outside
    # its own neighbourhood. A flat correlation surface means "could be anywhere".
    rival = r.copy()
    k = max(q.shape) // 2
    x0, y0 = max(0, loc[0] - k), max(0, loc[1] - k)
    rival[y0:loc[1] + k, x0:loc[0] + k] = -1.0
    margin = float(peak - rival.max())

    cx = loc[0] + q.shape[1] / 2.0
    cy = loc[1] + q.shape[0] / 2.0
    lon, lat = geo_window.to_lonlat(cx, cy)
    if peak < min_score or margin < min_margin:
        return NccResult(False, float(lon), float(lat), float(peak), margin,
                         reason=f"weak peak {peak:.2f} margin {margin:.3f}")
    return NccResult(True, float(lon), float(lat), float(peak), margin)
