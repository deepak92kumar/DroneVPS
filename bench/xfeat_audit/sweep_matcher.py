"""Matcher knob sweep. Same 20 frames, window frac 0.60 (the flown one).

Reuses the flight code's own query preparation and translation fit by import,
so only the knob under test differs. Nothing in vps_vio is modified.

Knobs: top_k, detection_threshold, min_cossim (the MNN descriptor floor, which
the flight path leaves at -1 = accept every mutual pair), the fit (median of
offsets vs RANSAC affine), tol_px, and sparse vs XFeat* semi-dense.
"""
import sys, os, time, math, csv as _csv, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2, torch
import replay as R
sys.path.insert(0, "/data/DRONE/vps_vio")
sys.path.insert(0, "/data/DRONE/third_party/accelerated_features")
from map_cache import MapCache
from map_match_xfeat import _as_3ch, _equalise, _translation
from map_query import north_up_query
from modules.xfeat import XFeat

OUT = "/data/DRONE/bench/xfeat_audit/sweep_matcher.csv"
torch.set_num_threads(1)
cache = MapCache(os.path.realpath("/data/DRONE/maps/active.tif"))
FRAC = 0.60
MODELS = {}


def model(top_k, det):
    k = (top_k, det)
    if k not in MODELS:
        MODELS[k] = XFeat(top_k=top_k, detection_threshold=det)
    return MODELS[k]


def pick(run, n=10):
    fr = R.frames(run, alt_min=58, step=3)
    for f in fr:
        f["prior_err"] = R.metres(f["est_lon"], f["est_lat"], f["true_lon"], f["true_lat"])
    fr = [f for f in fr if f["prior_err"] < 12.0]
    fr.sort(key=lambda f: f["alt"])
    if len(fr) <= n:
        return fr
    return [fr[i] for i in np.linspace(0, len(fr) - 1, n).round().astype(int)]


FRAMES = []
for run in ("vps_00476", "vps_00478"):
    for f in pick(run):
        img = cv2.imread(f["path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        rad = float(np.clip(FRAC * f["alt"], 40.0, 150.0))
        try:
            win, wg = cache.window(f["est_lon"], f["est_lat"], rad, kind="grey")
        except Exception:
            continue
        q = north_up_query(img, altitude_m=f["alt"], focal_px=R.CAM["fx"],
                           yaw_deg=f["yaw"] + R.YAW_OFFSET, map_gsd_m=cache.gsd,
                           map_gsd_x_m=getattr(cache, "gsd_x", None), shape="full",
                           track_xy=(R.CAM["cx"], R.CAM["cy"]))
        q, qc = q
        FRAMES.append(dict(run=run, **{k: f[k] for k in
                      ("t", "alt", "prior_err", "true_lon", "true_lat")},
                      q=_as_3ch(_equalise(q)), qc=np.array(qc, float),
                      win=_as_3ch(_equalise(win)), wg=wg))
print(f"{len(FRAMES)} frames prepared", flush=True)


def ransac_fit(src, dst, tol_px):
    M, inl = cv2.estimateAffinePartial2D(
        src.astype(np.float32).reshape(-1, 1, 2), dst.astype(np.float32).reshape(-1, 1, 2),
        method=cv2.RANSAC, ransacReprojThreshold=tol_px, maxIters=5000, confidence=0.995)
    if M is None or inl is None:
        return None, None, None, None
    keep = inl.ravel().astype(bool)
    scale = float(np.hypot(M[0, 0], M[1, 0]))
    rot = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    return M, keep, scale, rot


def one(f, *, top_k, det, cossim, fit, tol_px, mode):
    m = model(top_k, det)
    t1 = time.time()
    if mode == "star":
        out = m.match_xfeat_star(f["q"], f["win"], top_k=top_k)
    else:
        out = m.match_xfeat(f["q"], f["win"], top_k=top_k, min_cossim=cossim)
    src, dst = np.asarray(out[0], float), np.asarray(out[1], float)
    n_total = len(src)
    if n_total < 12:
        return dict(n_total=n_total, n_in=0, ratio=0.0, margin=0.0,
                    err=float("nan"), scale=float("nan"), rot=float("nan"),
                    ms=1000 * (time.time() - t1))
    if fit == "ransac":
        M, keep, scale, rot = ransac_fit(src, dst, tol_px)
        if M is None:
            return dict(n_total=n_total, n_in=0, ratio=0.0, margin=0.0,
                        err=float("nan"), scale=float("nan"), rot=float("nan"),
                        ms=1000 * (time.time() - t1))
        mc = cv2.transform(f["qc"].reshape(1, 1, 2).astype(np.float32), M)[0][0]
        res = np.linalg.norm(dst[keep] - cv2.transform(
            src[keep].astype(np.float32).reshape(-1, 1, 2), M).reshape(-1, 2), axis=1)
        mad = float(np.median(res)) if len(res) else tol_px
    else:
        off, keep, mad = _translation(src, dst, tol_px)
        mc = f["qc"] + off
        scale = rot = float("nan")
    n_in = int(keep.sum())
    lon, lat = f["wg"].to_lonlat(float(mc[0]), float(mc[1]))
    err = R.metres(float(lon), float(lat), f["true_lon"], f["true_lat"])
    return dict(n_total=n_total, n_in=n_in, ratio=n_in / max(n_total, 1),
                margin=max(0.0, 1.0 - mad / tol_px), err=err,
                scale=scale, rot=rot, ms=1000 * (time.time() - t1))


CONFIGS = []
# A. the flown configuration
CONFIGS.append(dict(tag="FLOWN", top_k=1024, det=0.05, cossim=-1, fit="median", tol_px=4.0, mode="sparse"))
# B. descriptor floor - the knob the flight path never set
for c in (0.5, 0.7, 0.82, 0.9):
    CONFIGS.append(dict(tag=f"cossim{c}", top_k=1024, det=0.05, cossim=c, fit="median", tol_px=4.0, mode="sparse"))
# C. top_k
for k in (2048, 3072, 4096):
    CONFIGS.append(dict(tag=f"topk{k}", top_k=k, det=0.05, cossim=-1, fit="median", tol_px=4.0, mode="sparse"))
# D. detection threshold
for d in (0.03, 0.07, 0.10):
    CONFIGS.append(dict(tag=f"det{d}", top_k=1024, det=d, cossim=-1, fit="median", tol_px=4.0, mode="sparse"))
# E. RANSAC instead of median-of-offsets
CONFIGS.append(dict(tag="ransac", top_k=1024, det=0.05, cossim=-1, fit="ransac", tol_px=4.0, mode="sparse"))
CONFIGS.append(dict(tag="ransac6", top_k=1024, det=0.05, cossim=-1, fit="ransac", tol_px=6.0, mode="sparse"))
# F. tol_px on the median fit
CONFIGS.append(dict(tag="tol6", top_k=1024, det=0.05, cossim=-1, fit="median", tol_px=6.0, mode="sparse"))
CONFIGS.append(dict(tag="tol2", top_k=1024, det=0.05, cossim=-1, fit="median", tol_px=2.0, mode="sparse"))
# G. semi-dense
CONFIGS.append(dict(tag="star4096", top_k=4096, det=0.05, cossim=-1, fit="median", tol_px=4.0, mode="star"))
# H. the two most promising crossed
CONFIGS.append(dict(tag="cossim0.82+ransac", top_k=1024, det=0.05, cossim=0.82, fit="ransac", tol_px=4.0, mode="sparse"))
CONFIGS.append(dict(tag="cossim0.82+topk4096", top_k=4096, det=0.05, cossim=0.82, fit="median", tol_px=4.0, mode="sparse"))

rows, t0 = [], time.time()
for cfg in CONFIGS:
    for f in FRAMES:
        try:
            r = one(f, **{k: v for k, v in cfg.items() if k != "tag"})
        except Exception as e:
            print("  FAIL", cfg["tag"], e, flush=True); continue
        rows.append(dict(tag=cfg["tag"], run=f["run"], t=f["t"], alt=f["alt"],
                         prior_err=f["prior_err"],
                         **{k: v for k, v in cfg.items() if k != "tag"}, **r))
    s = [x for x in rows if x["tag"] == cfg["tag"]]
    e = sorted(x["err"] for x in s if x["err"] == x["err"])
    print("%-20s n=%2d  matches %4.0f  inliers %3.0f  ratio %.3f  err p50 %5.1f  max %5.1f  %4.0f ms" % (
        cfg["tag"], len(s), np.median([x["n_total"] for x in s]),
        np.median([x["n_in"] for x in s]), np.median([x["ratio"] for x in s]),
        e[len(e) // 2] if e else -1, e[-1] if e else -1,
        np.median([x["ms"] for x in s])), flush=True)

with open(OUT, "w", newline="") as fh:
    w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"\n{len(rows)} rows -> {OUT}  ({time.time()-t0:.0f} s)")
