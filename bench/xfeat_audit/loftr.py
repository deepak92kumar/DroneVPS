#!/usr/bin/env python3
"""LoFTR against XFeat's mutual NN. The question is the MAP, not the matcher.

    python3 loftr.py vps_00476 vps_00478 [--step 10]

This test exists to decide between two completely different pieces of work.

XFeat scores 69-77% matching the map sheet against ITSELF and behaves like
random pairing matching the camera to that sheet, which is a 2022 satellite
image of ground that has been farmed for four years since. Either

  (a) the fault is the DETECTOR - XFeat picks keypoints that do not survive
      the appearance gap - and the answer is a different model, or
  (b) the fault is the SHEET - the ground no longer looks like the picture -
      and no model will fix it; the answer is new imagery.

LoFTR separates them, because it is detector-free: it has no keypoints at
all, it correlates dense coarse features between the two images and refines.
Where a detector fails because its corners have been ploughed away, LoFTR can
still match on texture and layout. So if LoFTR works here, it is (a). If
LoFTR fails here too, it is (b) - and that is worth knowing before anyone
builds a place-recognition stage on top of a map that does not match.

Held identical to the flight path: the same frames, the same north-up query
through map_query, the same window at clip(0.60 x alt, 40, 150) m, the same
median-of-offsets fit, the same nadir correction from absolute attitude, the
same M9N as truth.
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
    if n_total < MIN_INLIERS:
        return False, f"only {n_total} matches"
    if n_in < MIN_INLIERS:
        return False, f"only {n_in} inliers"
    if n_in / max(n_total, 1) < MIN_SCORE:
        return False, "inlier ratio"
    if max(0.0, 1.0 - mad / TOL_PX) < MIN_MARGIN:
        return False, "cluster spread"
    return True, ""


def _pad8(g):
    """LoFTR's backbone strides by 8, so both sides need multiples of 8."""
    h, w = g.shape[:2]
    H, W = (h + 7) // 8 * 8, (w + 7) // 8 * 8
    if (H, W) == (h, w):
        return g
    out = np.zeros((H, W), g.dtype)
    out[:h, :w] = g
    return out


def main():
    import cv2
    import torch
    import kornia.feature as KF
    from modules.xfeat import XFeat

    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--min-alt", type=float, default=58.0)
    ap.add_argument("--conf", type=float, default=0.5,
                    help="LoFTR confidence floor on a correspondence")
    ap.add_argument("--map", default="/data/DRONE/maps/active.tif")
    a = ap.parse_args()

    torch.set_num_threads(4)
    cache = MapCache(os.path.realpath(a.map))
    xf = XFeat(top_k=1024)
    lo = KF.LoFTR(pretrained="outdoor").eval()
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
            qe, we = _equalise(q), _equalise(win)
            roll, pitch, _ = fclog.at(run, f["t"])
            dN, dE = MapFixWorker.nadir_offset_m(f["alt"], roll, pitch, f["yaw"])
            qc = np.array(qc, float)
            rec = dict(run=run, t=f["t"], alt=f["alt"], prior_err=prior_err)

            for tag in ("mnn", "loftr"):
                t1 = time.time()
                try:
                    if tag == "mnn":
                        out = xf.match_xfeat(_as_3ch(qe), _as_3ch(we))
                        src = np.asarray(out[0], float)
                        dst = np.asarray(out[1], float)
                    else:
                        q8, w8 = _pad8(qe), _pad8(we)
                        d = {"image0": torch.from_numpy(q8)[None, None].float() / 255.,
                             "image1": torch.from_numpy(w8)[None, None].float() / 255.}
                        with torch.inference_mode():
                            o = lo(d)
                        c = o["confidence"].cpu().numpy()
                        keep = c >= a.conf
                        src = o["keypoints0"].cpu().numpy()[keep].astype(float)
                        dst = o["keypoints1"].cpu().numpy()[keep].astype(float)
                except Exception as exc:
                    rec[f"{tag}_ok"] = 0
                    rec[f"{tag}_err"] = float("nan")
                    rec[f"{tag}_why"] = f"raised {exc}"
                    continue
                ms = 1000 * (time.time() - t1)
                n_total = len(src)
                if n_total >= 3:
                    off, kp, mad = _translation(src, dst, TOL_PX)
                    n_in = int(kp.sum())
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
                            f"{tag}_err": err, f"{tag}_ms": ms, f"{tag}_why": why})
            rows.append(rec)
            print("   %3d/%d alt%6.1f prior%6.1f | MNN ok%s %3s/%4s e%6s |"
                  " LoFTR ok%s %4s/%5s e%6s %5.0fms" % (
                      k, len(fr), f["alt"], prior_err,
                      rec.get("mnn_ok"), rec.get("mnn_in"), rec.get("mnn_n"),
                      "%.1f" % rec.get("mnn_err", float("nan")),
                      rec.get("loftr_ok"), rec.get("loftr_in"), rec.get("loftr_n"),
                      "%.1f" % rec.get("loftr_err", float("nan")),
                      rec.get("loftr_ms", float("nan"))), flush=True)

    import csv as _csv
    p = "/data/DRONE/bench/xfeat_audit/loftr.csv"
    with open(p, "w", newline="") as fh:
        ks = sorted({k for r in rows for k in r})
        w = _csv.DictWriter(fh, fieldnames=ks); w.writeheader(); w.writerows(rows)

    print(f"\n{'=' * 84}\n{len(rows)} frames\n{'=' * 84}")
    for tag, name in (("mnn", "XFeat + mutual NN"), ("loftr", "LoFTR (detector-free)")):
        acc = [r for r in rows if r.get(f"{tag}_ok")
               and np.isfinite(r.get(f"{tag}_err", np.nan))]
        e = np.array([r[f"{tag}_err"] for r in acc]) if acc else np.array([])
        print("  %-23s accept %3d/%3d (%3.0f%%) | matches %5.0f inliers %5.0f"
              " ratio %.3f | err p50 %5.2f WORST %6.2f | %5.0f ms" % (
                  name, len(acc), len(rows), 100 * len(acc) / max(len(rows), 1),
                  np.median([r[f"{tag}_n"] for r in rows if f"{tag}_n" in r]),
                  np.median([r[f"{tag}_in"] for r in rows if f"{tag}_in" in r]),
                  np.median([r[f"{tag}_ratio"] for r in rows if f"{tag}_ratio" in r]),
                  np.median(e) if len(e) else float("nan"),
                  e.max() if len(e) else float("nan"),
                  np.median([r[f"{tag}_ms"] for r in rows if f"{tag}_ms" in r])))
    print("\n  If LoFTR matches where XFeat does not, the detector was the "
          "fault and a model swap is the work.\n  If both fail, the sheet is "
          "the fault and new imagery is the work. That is the whole point of\n"
          "  this test, and it decides which of the two gets built.")
    print(f"\n  -> {p}   ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
