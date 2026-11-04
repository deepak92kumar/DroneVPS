"""Locate a nadir frame on the satellite map with XFeat learned features.

Replaced the gradient-correlation matcher as the flying path; that matcher
was removed on 20 Oct at the pilot's instruction, once this one had taken over.
The reason for the swap is measured, not aesthetic: NCC correlates appearance,
so a four-year-old map costs it contrast on every frame, and its cost scales
with window area times template area. Learned features survive the appearance
gap and cost the same whatever the window is.

Why not the descriptor matcher in map_match.py: AKAZE/SIFT scored 0% on real
flight imagery against this map. XFeat is trained for exactly that gap.

Why XFeat and not SuperPoint/SuperGlue: Apache-2.0 (commercial use clear),
and it runs real-time sparse inference on CPU, which matters because this
Jetson's OpenCV has no CUDA (cv2.cuda.getCudaEnabledDeviceCount() == 0) and
there is no torch wheel on it yet. NOTE for whoever reads map_db.py's comment
about "XFeat exported to ONNX and run through TensorRT": upstream ships no
such export, so that route does not exist. This module uses torch directly.

The fit is TRANSLATION ONLY by default. Altitude gives scale and the compass
gives rotation, both already applied by north_up_query, so translation is the
only unknown left - and a looser model fits outliers on this imagery rather
than correcting anything. XFEAT_ANISO=1 adds an anisotropic scale term; it is
off by default because it introduced a 16.5% scale mismatch whose own code
comment blamed it for "16 m that has never been explained".

STATE, 19 Oct. It runs and it locates correctly on a synthetic pair: a patch
cut from a stand-in map at a known pixel, upscaled by the inverse of the
altitude scale and rotated, came back at 0.09 / 0.08 / 0.23 m of its true
centre at yaw 0 / +12 / -35, with 76-84 inliers of 168-179 matches. A pure
noise query was rejected with 0 inliers. On this Jetson's CPU a fix costs
530-620 ms warm (the first is ~2.9 s, model load), so at fix_period 2.0 s the
worker sits near 27% of a core - close to the 17% the 5 Oct report measured.

WHAT THAT DOES NOT SETTLE, and the thresholds that are therefore still open:
the synthetic pair is the SAME image on both sides, so it is an upper bound on
agreement in every respect. Real flights matched a four-year-old satellite
sheet over tree canopy and got 14 to 32 inliers above 140 m (5 Oct, with
top_k 1024). Against that, min_score - an inlier RATIO of 0.25 - is the number
most likely to be wrong, and wrong in the direction of rejecting good fixes.
min_inliers of 12 sits just under that measured 14 and has no margin either.
Both must be swept against a recorded flight, the way the NCC margin was swept
over 50 frames with a +-150 m window, before this path is trusted in the air.
"""
from __future__ import annotations
from dataclasses import dataclass
import os
import sys

import numpy as np
import cv2

from map_query import north_up_query

# One model instance per process. Loading is ~100 ms and the worker thread
# calls locate() repeatedly, so it must not reload per fix.
_MODEL = None
_TORCH = None


@dataclass
class XFeatResult:
    """Same shape as NccResult so MapFixWorker can build FixResult unchanged.

    score  = inlier ratio, 0..1. How much of the matched set agrees on one
             translation. This is the honest "is this a fix" number.
    margin = agreement tightness, 0..1: 1 - MAD(residual)/tol. A wide but
             numerous cluster is a weaker fix than a tight one of the same size.
    """
    ok: bool
    lon: float = 0.0
    lat: float = 0.0
    score: float = 0.0
    margin: float = 0.0
    inliers: int = 0
    total_matches: int = 0
    reason: str = ""


def _load():
    """Import torch and build the model on first use.

    Deliberately lazy: map_fix_worker imports this module at startup on an
    aircraft that may not have torch installed, and an ImportError there would
    take the whole VPS node down rather than just the map path.
    """
    global _MODEL, _TORCH
    if _MODEL is not None:
        return _MODEL
    # XFeat is not on PyPI, so it is a checkout rather than an install:
    #   git clone --depth 1 https://github.com/verlab/accelerated_features
    # Apache-2.0, confirmed from the LICENSE file in that checkout, not from a
    # fork's metadata. XFEAT_ROOT overrides the default location.
    root = os.environ.get("XFEAT_ROOT") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "third_party", "accelerated_features")
    if root not in sys.path:
        sys.path.insert(0, root)

    import torch
    from modules.xfeat import XFeat
    _TORCH = torch

    # ONE thread, not six. Measured on this Orin 20 Oct, same frame and the
    # same map window every time:
    #
    #     1 thread    261 ms        4 threads    646 ms
    #     2 threads   484 ms        6 threads   1233 ms   <- the default
    #     3 threads   510 ms
    #
    # Five times faster on one core than on all six. XFeat is a small sparse
    # model and torch's thread pool costs more to coordinate than the work it
    # divides, so every extra thread made it slower - monotonically, with no
    # sweet spot in between.
    #
    # It is the difference between fitting the job and not. The worker is
    # asked for a fix every second and was taking 1233 ms, so it could never
    # keep up; at 261 ms there is 739 ms of headroom. And it was taking that
    # time on ALL SIX cores, against a flow loop that needs one of them every
    # 59 ms - which is the host-side contention the 9 Oct report recorded as
    # "main loop stalled 44 s and 30 s" and blamed on stereo depth. Stereo
    # runs on the camera's own VPU and costs the Jetson nothing. This was it.
    #
    # Set before the model is built, because torch reads it when it first
    # builds its pool.
    torch.set_num_threads(1)
    # XFeat picks its own device - torch.cuda.is_available() inside its
    # __init__ - and takes no device argument. On this Jetson that resolves to
    # CPU, which upstream states is enough for real-time sparse inference.
    #
    # top_k 1024: 5 Oct measured it doubling the inliers above 140 m (14 -> 32)
    # and buying 13% of error for 1.9x the time per fix.
    _MODEL = XFeat(top_k=1024)
    return _MODEL


def _match_lg(model, q, win):
    """XFeat's features, matched by LighterGlue instead of mutual NN.

    THE MATCHING STEP WAS THE THING. The 4 Nov sweep moved every knob on the
    mutual-NN path - top_k 1024/2048/3072/4096, detection_threshold 0.03 to
    0.10, min_cossim -1/0.5/0.7/0.82/0.9, median fit against RANSAC, tol_px
    2/4/6, sparse against XFeat* - and the error p50 stayed between 8.9 and
    10.8 m on all eighteen. A flat sweep says the knob is not it. This is:
    a learned matcher that reasons over both feature sets together, against a
    mutual nearest neighbour with min_cossim defaulting to -1, which accepts
    every mutual pair however unlike the descriptors are.

    Measured on 338 frames of two flights, gates applied the same way, nadir
    corrected, against the M9N:

        mutual NN   accept 168/338 (50%)  err p50 3.53  WORST  10.95  >20 m 0
        LighterGlue accept 213/338 (63%)  err p50 3.94  WORST  14.79  >20 m 0
        MNN, else LG  236/338 (70%)       err p50 3.72  WORST  14.79  >20 m 0

    and by how wrong the prior was - the band this exists for, because the
    window is centred on the prior and a drifting one shows the matcher the
    wrong ground:

        prior   0- 10 m   MNN 78%   LG 75%
        prior  10- 20 m   MNN 70%   LG 86%
        prior  20- 50 m   MNN 22%   LG 94%     worst 14.8 m
        prior  50+    m   MNN  0%   LG 15%     worst  9.3 m

    The accuracy is not the gain - 3.53 against 3.94 - the COVERAGE is, and
    it arrives exactly where the aircraft needs it. Tiled over a ring, LG
    found three of four frames from flight 478's lost phase, 51 to 131 m out,
    to 3.6-8.3 m, where mutual NN found none of four across 49 tiles each.

    top_k stays at the loader's 1024: measured 38 inliers against 41 at 4096
    for 1482 ms against 2299, and the keypoints saturate at about 1800 anyway
    so 2048 and 4096 are identical.

    kornia is required and is NOT a hard dependency: without it this raises
    and the caller reports a refusal, which is the same outcome as a matcher
    that declines. The mutual-NN path is untouched by its absence.
    """
    d0 = model.detectAndCompute(q, top_k=model.top_k)[0]
    d1 = model.detectAndCompute(win, top_k=model.top_k)[0]
    d0.update({"image_size": (q.shape[1], q.shape[0])})
    d1.update({"image_size": (win.shape[1], win.shape[0])})
    m0, m1, _ = model.match_lighterglue(d0, d1, min_conf=0.1)
    return m0, m1


def locate_lg(*args, **kw):
    """locate() with LighterGlue, and the gate LighterGlue needs.

    min_inliers 20, NOT the 12 the mutual-NN path uses. This is the whole
    difference between a matcher that is bounded and one that is not, and it
    was measured the hard way: run with min_inliers 12, LighterGlue produced
    six fixes over 20 m wrong and a worst case of 152.67 m. Every one of them
    had 12 to 19 inliers. Raising the floor to 20 removed all six and took
    the worst case to 14.79 m with the accept rate still at 63%.

        min_inliers   fix rate   err p50   WORST
            12          68%       4.06     152.67
            20          63%       3.94      14.79
            30          58%       3.99      14.79
            40          51%       3.96      10.65

    The reason the old floor failed is that it was XFeat's. LighterGlue
    returns fewer candidate pairs and more of them agree - inlier ratio 0.248
    against mutual NN's 0.053 - so a gate written for one distribution barely
    gates the other.

    min_alt_m 90: LighterGlue is WORSE than mutual NN low down. Measured by
    altitude, accept rate 55-85 m: 19% against 46%; 85-100 m: 36% against
    36%; 100-118 m: 83% against 52%. And in the lost-frame test it found
    3 of 3 above 100 m and 0 of 1 at 58 m. So it is not asked below 90 m.
    """
    kw.setdefault("min_inliers", 20)
    kw.setdefault("min_alt_m", 90.0)
    kw["pairs"] = "lg"
    return locate(*args, **kw)


def _as_3ch(img: np.ndarray) -> np.ndarray:
    """XFeat's parse_input takes HxWxC and permutes NHWC->NCHW.

    A 2-D greyscale array reaches its permute as 2 dimensions and raises
    "input.dim() = 2 is not equal to len(dims) = 4". Everything upstream here
    is single-channel - the camera is a mono global shutter and the map cache
    stores grey - so the channel axis has to be added on this side. XFeat
    converts back to luminance internally, so this costs a copy, not accuracy.
    """
    return img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def _equalise(img: np.ndarray) -> np.ndarray:
    """CLAHE, same as the map features were built with.

    The map and the camera differ in exposure, haze and sun angle by years.
    Equalising local contrast on BOTH sides is what makes the descriptors
    comparable at all; map_db.build() does the same to the map.
    """
    return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(img)


def _translation(src: np.ndarray, dst: np.ndarray, tol_px: float):
    """Robust translation from matched points.

    Median of the per-match offset, then count how many agree within tol_px.
    A median survives the outlier rate this imagery produces, where RANSAC on a
    richer model would rather fit the outliers.

    Returns (offset, inlier_mask, mad). mad is the median absolute deviation of
    the inlier residual, in pixels - the tightness of the cluster.
    """
    d = dst - src
    off = np.median(d, axis=0)
    r = np.linalg.norm(d - off, axis=1)
    keep = r <= tol_px
    if keep.sum() >= 3:
        # Refit on the agreeing set only: the first median is pulled by
        # outliers, and a second pass costs nothing.
        off = np.median(d[keep], axis=0)
        r = np.linalg.norm(d - off, axis=1)
        keep = r <= tol_px
    mad = float(np.median(r[keep])) if keep.any() else float(tol_px)
    return off, keep, mad


def locate(img_grey: np.ndarray, map_window: np.ndarray, geo_window, *,
           altitude_m: float, focal_px: float, yaw_deg: float, map_gsd_m: float,
           map_gsd_x_m: float | None = None, min_inliers: int = 12, min_score: float = 0.07,
           min_margin: float = 0.20, tol_px: float = 4.0,
           window_is_edges: bool = False,
           pairs: str = "mnn", min_alt_m: float = 0.0,
           principal_xy=None, query_shape: str = "square") -> XFeatResult:
    """Match one frame to a map window and read off lon/lat.

    geo_window maps window pixel -> lon/lat.

    window_is_edges is kept as a REFUSAL, not as an option. MapCache can still
    hand over a pre-computed edge window, and XFeat needs intensity: a
    gradient-magnitude image has none of the texture the descriptors were
    trained on, so matching one would produce confident nonsense. The cache has
    to be asked for a grey window on this path, and asking for the wrong one is
    an error rather than a silent wrong answer.
    """
    if window_is_edges:
        return XFeatResult(False, reason="XFeat needs an intensity window, got edges")
    if map_window is None or min(map_window.shape[:2]) < 32:
        return XFeatResult(False, reason="map window too small")
    if min_alt_m and altitude_m < min_alt_m:
        # Asked of LighterGlue only, and measured: it is worse than mutual NN
        # below 85 m (19% accept against 46%) and better above 100 m (83%
        # against 52%). See locate_lg.
        return XFeatResult(False, reason=f"{pairs} is not asked below "
                                         f"{min_alt_m:.0f} m")
    if not (altitude_m > 0.0):
        # Same reason as the NCC path: the barometer reads below zero on the
        # ground, and a non-positive altitude makes the query scale negative.
        return XFeatResult(False, reason=f"altitude {altitude_m:.1f} m is not "
                                         f"above the ground")

    try:
        model = _load()
    except Exception as e:                      # torch or weights missing
        return XFeatResult(False, reason=f"xfeat unavailable: {e}")

    # Scale from altitude, rotation from the compass. Same preparation the NCC
    # path uses, so the two matchers see the identical query and can be scored
    # against each other on the same recording.
    # WHICH PIXEL THE ANSWER IS ABOUT. A translation on its own does not say
    # where the aircraft is; it says how far the query moved. The position
    # belongs to the pixel the optical axis goes through - the principal point
    # (cx, cy) - because that is the camera's own position projected onto the
    # ground. The geometric centre is 80 px away from it on this camera, since
    # the frames are the 130 px leg-crop, and 80 px is 11 m of ground at
    # 110 m: a bias in the SAME direction on every frame, which never shows up
    # as a worse match. flow_odom has always used K's cx/cy as its centre, so
    # until 20 Oct the two paths disagreed about where the aircraft was.
    #
    # principal_xy stays optional and falls back to the geometric centre, so a
    # caller that does not know K gets exactly the old behaviour rather than a
    # silently different answer.
    q = north_up_query(img_grey, altitude_m=altitude_m, focal_px=focal_px,
                       yaw_deg=yaw_deg, map_gsd_m=map_gsd_m,
                       map_gsd_x_m=map_gsd_x_m, shape=query_shape,
                       track_xy=None if principal_xy is None else principal_xy)
    qc_tracked = None
    if principal_xy is not None:
        q, qc_tracked = q
    if min(q.shape[:2]) < 32:
        return XFeatResult(False, reason="query too small after rescaling")

    try:
        if pairs == "lg":
            out = _match_lg(model, _as_3ch(_equalise(q)),
                            _as_3ch(_equalise(map_window)))
        else:
            out = model.match_xfeat(_as_3ch(_equalise(q)),
                                    _as_3ch(_equalise(map_window)))
    except Exception as e:
        return XFeatResult(False, reason=f"{pairs} match failed: {e}")
    if out is None or len(out) < 2:
        return XFeatResult(False, reason="no matches")
    src, dst = np.asarray(out[0], float), np.asarray(out[1], float)
    n_total = int(len(src))
    if n_total < min_inliers:
        return XFeatResult(False, total_matches=n_total,
                           reason=f"only {n_total} matches")

    off, keep, mad = _translation(src, dst, tol_px)
    n_in = int(keep.sum())
    ratio = n_in / max(n_total, 1)
    margin = max(0.0, 1.0 - mad / float(tol_px))

    # Where that pixel lands on the map, through the translation just fitted.
    qc = (np.array([q.shape[1] / 2.0, q.shape[0] / 2.0]) if qc_tracked is None
          else np.array(qc_tracked, float))
    mc = qc + off
    lon, lat = geo_window.to_lonlat(float(mc[0]), float(mc[1]))
    lon, lat = float(lon), float(lat)

    if os.environ.get("XFEAT_ANISO") == "1":
        # Left reachable on purpose, as it was before: it is the only way to
        # re-measure the scale mismatch it causes. It does not move lon/lat
        # here - fitting it would, which is the fault - it only reports.
        sx, sy = _aniso_scale(src[keep], dst[keep])
        if sx and sy and min(sx, sy) > 0:
            margin *= 1.0 - min(0.5, abs(sx - sy) / max(sx, sy))

    # THE GATE. The previous XFeat path hard-coded ok=True and so declared
    # every frame a fix - 189 of them on one flight - which is the single
    # defect that most needs not to come back. A fix is a fix only if enough
    # points agree, a real share of them agree, and they agree tightly.
    if n_in < min_inliers:
        return XFeatResult(False, lon, lat, ratio, margin, n_in, n_total,
                           reason=f"only {n_in} inliers")
    if ratio < min_score:
        return XFeatResult(False, lon, lat, ratio, margin, n_in, n_total,
                           reason=f"inlier ratio {ratio:.2f} below {min_score}")
    if margin < min_margin:
        return XFeatResult(False, lon, lat, ratio, margin, n_in, n_total,
                           reason=f"cluster spread {mad:.1f} px too wide")
    return XFeatResult(True, lon, lat, ratio, margin, n_in, n_total)


def _aniso_scale(src: np.ndarray, dst: np.ndarray):
    """Per-axis scale between two matched sets, about their own centroids.

    Reporting only. See XFEAT_ANISO above.
    """
    if len(src) < 4:
        return None, None
    s = src - src.mean(axis=0)
    d = dst - dst.mean(axis=0)
    out = []
    for i in (0, 1):
        den = float(np.sum(s[:, i] * s[:, i]))
        out.append(float(np.sum(s[:, i] * d[:, i]) / den) if den > 1e-9 else None)
    return out[0], out[1]
