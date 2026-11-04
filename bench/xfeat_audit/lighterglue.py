#!/usr/bin/env python3
"""XFeat + LighterGlue against XFeat + mutual nearest neighbour.

    python3 lighterglue.py vps_00476 vps_00478 [--step 6]

The weight has been sitting in third_party/accelerated_features/weights the
whole time and nothing used it. The 4 Nov sweep moved every knob on the MNN
path - top_k 1024/2048/3072/4096, detection_threshold 0.03 to 0.10,
min_cossim -1/0.5/0.7/0.82/0.9, median fit against RANSAC, tol_px 2/4/6,
sparse against XFeat* - and the error p50 stayed between 8.9 and 10.8 m on
every one of the eighteen. A flat sweep says the knob is not the thing. The
MATCHER is, and LighterGlue replaces the matching step itself: a learned
transformer that reasons over both sets jointly instead of pairing each
feature with its nearest partner and no similarity floor at all
(the flight path calls match_xfeat with min_cossim defaulting to -1).

Everything else is held identical - the same frames, the same north-up query
through the flight code's own map_query, the same window at
clip(0.60 x alt, 40, 150) m, the same median-of-offsets translation fit from
map_match_xfeat, the same nadir correction from absolute attitude, and the
same M9N as truth. Only the correspondence step differs.
"""
import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replay as R                                      # noqa: E402
import fclog                                            # noqa: E402

sys.path.insert(0, "/data/DRONE/vps_vio")
sys.path.insert(0, "/data/DRONE/third_party/accelerated_features")
from map_cache import MapCache                          # noqa: E402
from map_fix_worker import MapFixWorker                 # noqa: E402
from map_match_xfeat import _as_3ch, _equalise, _translation   # noqa: E402
from map_query import north_up_query                    # noqa: E402

MDEG = 111320.0
TOL_PX = 4.0
MIN_INLIERS = 12
MIN_SCORE = 0.07
MIN_MARGIN = 0.20


def gate(n_in, n_total, mad):
    """map_match_xfeat.locate's gate, applied to whatever produced the pairs."""
    if n_total < MIN_INLIERS:
        return False, f"only {n_total} matches"
    if n_in < MIN_INLIERS:
        return False, f"only {n_in} inliers"
    ratio = n_in / max(n_total, 1)
    if ratio < MIN_SCORE:
        return False, f"inlier ratio {ratio:.2f}"
    if max(0.0, 1.0 - mad / TOL_PX) < MIN_MARGIN:
        return False, f"cluster spread {mad:.1f} px"
    return True, ""


def main():
    import cv2
    import torch
    from modules.xfeat import XFeat

    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--step", type=int, default=6)
    ap.add_argument("--min-alt", type=float, default=58.0)
    ap.add_argument("--top-k", type=int, default=4096,
                    help="LighterGlue's own example uses 4096")
    ap.add_argument("--min-conf", type=float, default=0.1)
    ap.add_argument("--map", default="/data/DRONE/maps/active.tif")
    a = ap.parse_args()

    torch.set_num_threads(4)
    cache = MapCache(os.path.realpath(a.map))
    model = XFeat(top_k=a.top_k)
    rows, t0 = [], time.time()

    for run in a.runs:
        fclog.attitude(run)
        fr = R.frames(run, alt_min=a.min_alt, step=a.step)
        print(f"{run}: {len(fr)} frames", flush=True)
        for k, f in enumerate(fr):
            img = cv2.imread(f["path"], cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            prior_err = R.metres(f["est_lon"], f["est_lat"],
                                 f["true_lon"], f["true_lat"])
            rad = float(np.clip(0.60 * f["alt"], 40.0, 150.0))
            try:
                win, wg = cache.window(f["est_lon"], f["est_lat"], rad,
                                       kind="grey")
            except Exception:
                continue
            q, qc = north_up_query(img, altitude_m=f["alt"],
                                   focal_px=R.CAM["fx"],
                                   yaw_deg=f["yaw"] + R.YAW_OFFSET,
                                   map_gsd_m=cache.gsd,
                                   map_gsd_x_m=getattr(cache, "gsd_x", None),
                                   shape="full",
                                   track_xy=(R.CAM["cx"], R.CAM["cy"]))
            qi, wi = _as_3ch(_equalise(q)), _as_3ch(_equalise(win))
            roll, pitch, _ = fclog.at(run, f["t"])
            dN, dE = MapFixWorker.nadir_offset_m(f["alt"], roll, pitch, f["yaw"])
            qc = np.array(qc, float)

            rec = dict(run=run, t=f["t"], alt=f["alt"], prior_err=prior_err)
            for tag in ("mnn", "lg"):
                t1 = time.time()
                try:
                    if tag == "mnn":
                        out = model.match_xfeat(qi, wi, top_k=a.top_k)
                        src, dst = (np.asarray(out[0], float),
                                    np.asarray(out[1], float))
                    else:
                        d0 = model.detectAndCompute(qi, top_k=a.top_k)[0]
                        d1 = model.detectAndCompute(wi, top_k=a.top_k)[0]
                        d0.update({"image_size": (qi.shape[1], qi.shape[0])})
                        d1.update({"image_size": (wi.shape[1], wi.shape[0])})
                        m0, m1, _ = model.match_lighterglue(
                            d0, d1, min_conf=a.min_conf)
                        src, dst = (np.asarray(m0, float), np.asarray(m1, float))
                except Exception as exc:
                    rec[f"{tag}_ok"] = 0
                    rec[f"{tag}_why"] = f"raised {exc}"
                    rec[f"{tag}_err"] = float("nan")
                    continue
                ms = 1000 * (time.time() - t1)
                n_total = len(src)
                if n_total >= 3:
                    off, keep, mad = _translation(src, dst, TOL_PX)
                    n_in = int(keep.sum())
                    mc = qc + off
                    lon, lat = wg.to_lonlat(float(mc[0]), float(mc[1]))
                    lat = lat - dN / MDEG
                    lon = lon - dE / (MDEG * math.cos(math.radians(lat)))
                    err = R.metres(lon, lat, f["true_lon"], f["true_lat"])
                else:
                    n_in, mad, err = 0, TOL_PX, float("nan")
                ok, why = gate(n_in, n_total, mad)
                rec.update({f"{tag}_ok": int(ok), f"{tag}_n": n_total,
                            f"{tag}_in": n_in,
                            f"{tag}_ratio": n_in / max(n_total, 1),
                            f"{tag}_err": err, f"{tag}_ms": ms,
                            f"{tag}_why": why})
            rows.append(rec)
            if k % 10 == 0:
                print("   %3d/%d alt%6.1f prior%6.1f | MNN ok%s %3s/%4s e%6s"
                      " | LG ok%s %3s/%4s e%6s" % (
                          k, len(fr), f["alt"], prior_err,
                          rec.get("mnn_ok"), rec.get("mnn_in"), rec.get("mnn_n"),
                          "%.1f" % rec.get("mnn_err", float("nan")),
                          rec.get("lg_ok"), rec.get("lg_in"), rec.get("lg_n"),
                          "%.1f" % rec.get("lg_err", float("nan"))), flush=True)

    import csv as _csv
    p = "/data/DRONE/bench/xfeat_audit/lighterglue.csv"
    with open(p, "w", newline="") as fh:
        ks = sorted({k for r in rows for k in r})
        w = _csv.DictWriter(fh, fieldnames=ks)
        w.writeheader(); w.writerows(rows)

    print(f"\n{'=' * 80}\n{len(rows)} frames\n{'=' * 80}")
    for tag, name in (("mnn", "mutual NN (as flown)"), ("lg", "LighterGlue")):
        acc = [r for r in rows if r.get(f"{tag}_ok")
               and np.isfinite(r.get(f"{tag}_err", np.nan))]
        e = np.array([r[f"{tag}_err"] for r in acc]) if acc else np.array([])
        allms = [r[f"{tag}_ms"] for r in rows if f"{tag}_ms" in r]
        print("  %-22s accept %3d/%3d (%3.0f%%) | matches %4.0f inliers %4.0f"
              " ratio %.3f | err p50 %5.2f p90 %6.2f WORST %6.2f | >20 m %2d"
              " | %4.0f ms" % (
                  name, len(acc), len(rows), 100 * len(acc) / max(len(rows), 1),
                  np.median([r[f"{tag}_n"] for r in rows if f"{tag}_n" in r]),
                  np.median([r[f"{tag}_in"] for r in rows if f"{tag}_in" in r]),
                  np.median([r[f"{tag}_ratio"] for r in rows if f"{tag}_ratio" in r]),
                  np.median(e) if len(e) else float("nan"),
                  np.percentile(e, 90) if len(e) else float("nan"),
                  e.max() if len(e) else float("nan"),
                  int((e > 20).sum()) if len(e) else 0,
                  np.median(allms) if allms else float("nan")))
    print("\n  by prior error - the band that decides whether a wider search "
          "could ever work:")
    for lo, hi in ((0, 10), (10, 20), (20, 50), (50, 999)):
        s = [r for r in rows if lo <= r["prior_err"] < hi]
        if not s:
            continue
        line = "    prior %3d-%3d m  n=%3d |" % (lo, hi, len(s))
        for tag in ("mnn", "lg"):
            acc = [r for r in s if r.get(f"{tag}_ok")
                   and np.isfinite(r.get(f"{tag}_err", np.nan))]
            e = [r[f"{tag}_err"] for r in acc]
            line += "  %-3s %3.0f%% / worst %6s" % (
                tag, 100 * len(acc) / len(s),
                "%.1f" % max(e) if e else "-")
        print(line)
    print(f"\n  -> {p}   ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
