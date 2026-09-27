"""Match a drone nadir image against the satellite map -> absolute lon/lat.

Pipeline: detect features in the (rescaled) query, match to the pre-built map
features, fit a similarity/affine transform with RANSAC, then push the query
centre through that transform and read off the geographic position.

The output is what AnchorFuser consumes — the same role ArUco plays near home,
but available anywhere the map covers.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cv2

from map_db import MapDatabase, prepare_query


@dataclass
class MatchResult:
    ok: bool
    lon: float = 0.0
    lat: float = 0.0
    inliers: int = 0
    total_matches: int = 0
    inlier_ratio: float = 0.0
    scale: float = 1.0          # map px per query px from the fitted transform
    rotation_deg: float = 0.0
    reason: str = ""


def _ratio_match(desc_q, desc_m, norm, ratio=0.8):
    if desc_q is None or desc_m is None or len(desc_q) < 4 or len(desc_m) < 4:
        return []
    bf = cv2.BFMatcher(norm)
    knn = bf.knnMatch(desc_q, desc_m, k=2)
    good = []
    for pair in knn:
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
            good.append(pair[0])
    return good


def match_image(query_grey: np.ndarray, db: MapDatabase, *,
                altitude_m: float, focal_px: float,
                min_inliers: int = 12, ratio: float = 0.8) -> MatchResult:
    """Locate a drone image on the map. Query is resampled to the map's GSD first."""
    if db.kp is None:
        return MatchResult(False, reason="map database not built")

    q, _ = prepare_query(query_grey, altitude_m, focal_px, db.gsd)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    q = clahe.apply(q)

    kq, dq = db.det.detectAndCompute(q, None)
    if dq is None or len(kq) < 8:
        return MatchResult(False, reason=f"too few query features ({0 if kq is None else len(kq)})")

    norm = cv2.NORM_L2 if db.detector_name.upper() == "SIFT" else cv2.NORM_HAMMING
    good = _ratio_match(dq, db.desc, norm, ratio)
    if len(good) < min_inliers:
        return MatchResult(False, total_matches=len(good),
                           reason=f"only {len(good)} ratio-test matches")

    src = np.float32([kq[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([db.kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # A nadir view over flat ground differs from the map by rotation + scale +
    # translation, so a partial-affine (similarity) fit is the right model and is
    # far more robust than a full homography with this many outliers.
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                         ransacReprojThreshold=3.0,
                                         maxIters=5000, confidence=0.995)
    if M is None or inl is None:
        return MatchResult(False, total_matches=len(good), reason="RANSAC failed")
    n_in = int(inl.sum())
    if n_in < min_inliers:
        return MatchResult(False, total_matches=len(good), inliers=n_in,
                           reason=f"only {n_in} inliers")

    qc = np.array([[[q.shape[1] / 2.0, q.shape[0] / 2.0]]], dtype=np.float32)
    mc = cv2.transform(qc, M)[0][0]
    lon, lat = db.geo.to_lonlat(mc[0], mc[1])
    scale = float(np.hypot(M[0, 0], M[1, 0]))
    rot = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    return MatchResult(True, float(lon), float(lat), n_in, len(good),
                       n_in / max(len(good), 1), scale, rot)


def geo_error_m(lon1, lat1, lon2, lat2) -> float:
    """Great-circle-ish distance in metres for small separations."""
    m_lat = 111_320.0
    m_lon = 111_320.0 * np.cos(np.radians((lat1 + lat2) / 2))
    return float(np.hypot((lon1 - lon2) * m_lon, (lat1 - lat2) * m_lat))
