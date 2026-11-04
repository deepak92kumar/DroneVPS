#!/usr/bin/env python3
"""The lost drone's own frames. Who finds them on the map?

    python3 find_me.py

Flight 478's late phase, where the estimate was 50 to 130 m wrong and every
matcher that was handed a window around that estimate failed. The question is
not whether a wider window works - measured, it does not, the accept rate
collapses with width. The question is whether the frame is findable at all,
and by whom.

So the search region around the estimate is TILED instead of widened: tiles
the size the matcher already works on, overlapping, every one tried. The
winner is the tile whose match passes the gate with the most agreeing points.

Scored against the M9N. For each frame: did it find the truth, how far off,
and did any matcher confidently name a wrong tile - which is the dangerous
answer, not the missing one.
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replay as R
import fclog

sys.path.insert(0, "/data/DRONE/vps_vio")
sys.path.insert(0, "/data/DRONE/third_party/accelerated_features")
from map_cache import MapCache
from map_fix_worker import MapFixWorker
from map_match_xfeat import _as_3ch, _equalise, _translation
from map_query import north_up_query

MDEG = 111320.0
SEARCH_R = 250.0        # how lost we allow for: +/- 250 m around the estimate
TILE_M = 150.0          # the window the matcher already works on
STEP_M = 75.0           # 50% overlap, so nothing falls between tiles
TOL_PX = 4.0


def main():
    import cv2
    import torch
    from modules.xfeat import XFeat

    torch.set_num_threads(4)
    cache = MapCache(os.path.realpath("/data/DRONE/maps/active.tif"))
    model = XFeat(top_k=4096)
    fclog.attitude("vps_00478")

    fr = R.frames("vps_00478", alt_min=58, step=3)
    for f in fr:
        f["prior_err"] = R.metres(f["est_lon"], f["est_lat"],
                                  f["true_lon"], f["true_lat"])
    lost = [f for f in fr if 50.0 <= f["prior_err"] <= 200.0]
    print(f"{len(lost)} frames jab drone 50-200 m kho chuka tha", flush=True)
    idx = np.linspace(0, len(lost) - 1, min(4, len(lost))).round().astype(int)
    sel = [lost[i] for i in idx]

    results = []
    t0 = time.time()
    for f in sel:
        img = cv2.imread(f["path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        roll, pitch, _ = fclog.at("vps_00478", f["t"])
        dN, dE = MapFixWorker.nadir_offset_m(f["alt"], roll, pitch, f["yaw"])
        mlon = MDEG * math.cos(math.radians(f["true_lat"]))
        q, qc = north_up_query(img, altitude_m=f["alt"], focal_px=R.CAM["fx"],
                               yaw_deg=f["yaw"] + R.YAW_OFFSET,
                               map_gsd_m=cache.gsd,
                               map_gsd_x_m=getattr(cache, "gsd_x", None),
                               shape="full",
                               track_xy=(R.CAM["cx"], R.CAM["cy"]))
        qe = _as_3ch(_equalise(q))
        qc = np.array(qc, float)
        d0 = None

        grid = np.arange(-SEARCH_R, SEARCH_R + 1, STEP_M)
        print("\n  frame %s  t=%.0f  alt %.0f m  |  drone sochta tha wo yahan hai,"
              " asli jagah %.1f m door  |  %d x %d = %d tiles" % (
                  os.path.basename(f["path"]), f["t"], f["alt"],
                  f["prior_err"], len(grid), len(grid), len(grid) ** 2),
              flush=True)

        rec = dict(f=f, dN=dN, dE=dE, tiles={}, qe=qe, qc=qc)
        for tag in ("mnn", "lg"):
            best = None
            n_try = 0
            t1 = time.time()
            for dn in grid:
                for de in grid:
                    tlat = f["est_lat"] + dn / MDEG
                    tlon = f["est_lon"] + de / mlon
                    if not cache.covers(tlon, tlat):
                        continue
                    try:
                        win, wg = cache.window(tlon, tlat, TILE_M / 2.0,
                                               kind="grey")
                    except Exception:
                        continue
                    we = _as_3ch(_equalise(win))
                    n_try += 1
                    try:
                        if tag == "mnn":
                            o = model.match_xfeat(qe, we, top_k=4096)
                            src = np.asarray(o[0], float)
                            dst = np.asarray(o[1], float)
                        else:
                            if d0 is None:
                                d0 = model.detectAndCompute(qe, top_k=4096)[0]
                                d0.update({"image_size": (qe.shape[1], qe.shape[0])})
                            d1 = model.detectAndCompute(we, top_k=4096)[0]
                            d1.update({"image_size": (we.shape[1], we.shape[0])})
                            m0, m1, _ = model.match_lighterglue(d0, d1,
                                                                min_conf=0.1)
                            src = np.asarray(m0, float)
                            dst = np.asarray(m1, float)
                    except Exception:
                        continue
                    if len(src) < 12:
                        continue
                    off, keep, mad = _translation(src, dst, TOL_PX)
                    n_in = int(keep.sum())
                    ratio = n_in / max(len(src), 1)
                    margin = max(0.0, 1.0 - mad / TOL_PX)
                    mi = 20 if tag == "lg" else 12
                    if n_in < mi or ratio < 0.07 or margin < 0.20:
                        continue
                    mc = qc + off
                    lon, lat = wg.to_lonlat(float(mc[0]), float(mc[1]))
                    lat -= dN / MDEG
                    lon -= dE / (MDEG * math.cos(math.radians(lat)))
                    err = R.metres(lon, lat, f["true_lon"], f["true_lat"])
                    cand = dict(n_in=n_in, n=len(src), ratio=ratio, err=err,
                                lon=lon, lat=lat, dn=dn, de=de)
                    if best is None or n_in > best["n_in"]:
                        best = cand
            ms = time.time() - t1
            rec["tiles"][tag] = dict(best=best, n_try=n_try, secs=ms)
            if best is None:
                print("     %-4s %3d tiles, %5.1f s  ->  KUCH NAHI MILA" % (
                    tag, n_try, ms), flush=True)
            else:
                print("     %-4s %3d tiles, %5.1f s  ->  %3d points sehmat, "
                      "asli jagah se %6.2f m   %s" % (
                          tag, n_try, ms, best["n_in"], best["err"],
                          "SAHI DHOONDHA" if best["err"] <= 20
                          else "GALAT JAGAH BATAYI"), flush=True)
        results.append(rec)

    import pickle
    with open("/data/DRONE/bench/xfeat_audit/find_me.pkl", "wb") as fh:
        pickle.dump([{k: v for k, v in r_.items() if k not in ("qe",)}
                     for r_ in results], fh)
    print(f"\n{'=' * 76}")
    for tag, name in (("mnn", "XFeat + mutual NN"), ("lg", "XFeat + LighterGlue")):
        found = [r_ for r_ in results if r_["tiles"][tag]["best"]
                 and r_["tiles"][tag]["best"]["err"] <= 20]
        wrong = [r_ for r_ in results if r_["tiles"][tag]["best"]
                 and r_["tiles"][tag]["best"]["err"] > 20]
        none = [r_ for r_ in results if not r_["tiles"][tag]["best"]]
        print("  %-22s SAHI %d/%d | GALAT %d | kuch nahi %d | %5.1f s per frame"
              % (name, len(found), len(results), len(wrong), len(none),
                 np.mean([r_["tiles"][tag]["secs"] for r_ in results])))
    print(f"\n({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
