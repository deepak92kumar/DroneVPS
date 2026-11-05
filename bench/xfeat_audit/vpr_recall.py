#!/usr/bin/env python3
"""recall@K: does a global descriptor put the TRUE map tile in the top K?

This is the one number that decides whether a VPR coarse stage is worth
building. The whole sheet is cut into the same tiles the tiled search uses
(150 m, 75 m step), each gets one compact vector from XFeat's own
descriptors, and then every test frame is ranked against ALL of them.

If the true tile lands in the top 20, the coarse stage turns 49-110 s of
tiled matching into about 30 s. If it lands at rank 400, it does not work and
the honest answer is that this needs a real appearance-invariant backbone,
not that the idea is wrong.

XFeat's own descriptors deliberately, not DINOv2: nothing new has to be
installed, the query's features are already computed on the flying path, and
if these cannot tell camera from 2022 sheet globally then the record already
suspects they cannot tell it locally either (map_does_not_match, 3 Nov).
"""
import math, os, subprocess, sys, time
import numpy as np, cv2, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replay as R
import fclog
sys.path.insert(0, "/data/DRONE/vps_vio")
sys.path.insert(0, "/data/DRONE/third_party/accelerated_features")
from map_cache import MapCache
from map_match_xfeat import _as_3ch, _equalise
from map_query import north_up_query
from modules.xfeat import XFeat

torch.set_num_threads(4)
MDEG = 111320.0
TILE_M, STEP_M = 150.0, 75.0
TIF = os.path.realpath("/data/DRONE/maps/active.tif")


def sheet_bounds():
    """lon/lat corners, from gdalinfo rather than a new python dependency."""
    out = subprocess.run(["gdalinfo", TIF], capture_output=True, text=True).stdout
    lo = la = None
    import re
    xs, ys = [], []
    for key in ("Upper Left", "Lower Right"):
        m = re.search(key + r"\s*\(\s*([-\d.]+),\s*([-\d.]+)\)", out)
        if m:
            xs.append(float(m.group(1))); ys.append(float(m.group(2)))
    if len(xs) != 2:
        raise RuntimeError("could not read the sheet's corners from gdalinfo")
    return min(xs), max(xs), min(ys), max(ys)


def gem(d, p=3.0):
    """Generalised-mean pooling over a tile's descriptors, then L2."""
    if d is None or len(d) == 0:
        return None
    x = np.asarray(d, np.float32)
    x = np.clip(x, 1e-6, None) if x.min() >= 0 else np.abs(x) + 1e-6
    v = np.power(np.mean(np.power(x, p), axis=0), 1.0 / p)
    n = np.linalg.norm(v)
    return (v / n) if n > 1e-9 else None


def meanl2(d):
    if d is None or len(d) == 0:
        return None
    v = np.asarray(d, np.float32).mean(axis=0)
    n = np.linalg.norm(v)
    return (v / n) if n > 1e-9 else None


def main():
    cache = MapCache(TIF)
    model = XFeat(top_k=1024)
    lon0, lon1, lat0, lat1 = sheet_bounds()
    latm = 0.5 * (lat0 + lat1)
    mlon = MDEG * math.cos(math.radians(latm))
    print("sheet: lon %.5f..%.5f  lat %.5f..%.5f  = %.0f x %.0f m" % (
        lon0, lon1, lat0, lat1, (lon1 - lon0) * mlon, (lat1 - lat0) * MDEG), flush=True)

    # ---- tile grid, inset by half a tile so every tile is fully inside
    half = TILE_M / 2
    ns = np.arange(lat0 + half / MDEG, lat1 - half / MDEG, STEP_M / MDEG)
    es = np.arange(lon0 + half / mlon, lon1 - half / mlon, STEP_M / mlon)
    print("tiles: %d x %d = %d  (%.0f m, %.0f m step)" % (
        len(es), len(ns), len(es) * len(ns), TILE_M, STEP_M), flush=True)

    t0 = time.time()
    cen, G, M = [], [], []
    for j, la in enumerate(ns):
        for lo in es:
            try:
                if not cache.covers(lo, la):
                    continue
                win, _ = cache.window(lo, la, half, kind="grey")
                d = model.detectAndCompute(_as_3ch(_equalise(win)),
                                           top_k=1024)[0]["descriptors"]
                d = d.cpu().numpy()
            except Exception:
                continue
            g, m = gem(d), meanl2(d)
            if g is None or m is None:
                continue
            cen.append((lo, la)); G.append(g); M.append(m)
        if j % 5 == 0:
            print("   row %d/%d  %d tiles  %.0f s" % (j, len(ns), len(cen),
                                                      time.time() - t0), flush=True)
    cen = np.array(cen); G = np.array(G); M = np.array(M)
    print("built %d tile descriptors in %.0f s  (%.0f ms each)" % (
        len(cen), time.time() - t0, 1000 * (time.time() - t0) / max(len(cen), 1)),
        flush=True)
    np.savez("/data/DRONE/bench/xfeat_audit/vpr_tiles.npz",
             cen=cen, gem=G, meanl2=M, tile_m=TILE_M, step_m=STEP_M)

    # ---- test frames: the lost ones, and good ones as a control
    tests = []
    for run, bin_ in (("vps_00478", None), ("vps_00486",
                       "/home/abc/Downloads/2026-11-05 08-57-56.bin")):
        if bin_:
            fclog.BIN[run] = bin_
        try:
            fclog.attitude(run, verbose=False)
        except Exception as exc:
            print("  skip %s: %s" % (run, exc)); continue
        fr = R.frames(run, alt_min=85, step=3)
        for f in fr:
            f["prior_err"] = R.metres(f["est_lon"], f["est_lat"],
                                      f["true_lon"], f["true_lat"])
            f["run"] = run
        lost = [f for f in fr if f["prior_err"] >= 50]
        good = [f for f in fr if f["prior_err"] < 12]
        for tag, sel, k in (("LOST", lost, 8), ("GOOD", good, 8)):
            if not sel:
                continue
            idx = np.linspace(0, len(sel) - 1, min(k, len(sel))).round().astype(int)
            tests += [(tag, sel[i]) for i in idx]
    print("\n%d test frames" % len(tests), flush=True)

    res = []
    for tag, f in tests:
        img = cv2.imread(f["path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        q, _ = north_up_query(img, altitude_m=f["alt"], focal_px=R.CAM["fx"],
                              yaw_deg=f["yaw"] + R.YAW_OFFSET,
                              map_gsd_m=cache.gsd,
                              map_gsd_x_m=getattr(cache, "gsd_x", None),
                              shape="full", track_xy=(R.CAM["cx"], R.CAM["cy"]))
        d = model.detectAndCompute(_as_3ch(_equalise(q)),
                                   top_k=1024)[0]["descriptors"].cpu().numpy()
        qg, qm = gem(d), meanl2(d)
        if qg is None:
            continue
        # the true tile = whose centre is nearest the M9N
        dn = (cen[:, 1] - f["true_lat"]) * MDEG
        de = (cen[:, 0] - f["true_lon"]) * mlon
        dist = np.hypot(dn, de)
        true_i = int(np.argmin(dist))
        row = dict(tag=tag, run=f["run"], t=f["t"], alt=f["alt"],
                   prior=f["prior_err"], true_d=dist[true_i])
        for nm, Q, DB in (("gem", qg, G), ("meanl2", qm, M)):
            s = DB @ Q
            order = np.argsort(-s)
            row[nm] = int(np.where(order == true_i)[0][0]) + 1
            # how far the top-1 tile was from the truth
            row[nm + "_top1_m"] = float(dist[order[0]])
        res.append(row)
        print("  %-4s %s t=%5.0f alt %5.1f prior %6.1f | true tile rank: "
              "gem %4d  meanl2 %4d  | top1 %6.1f m door" % (
                  tag, f["run"][-3:], f["t"], f["alt"], f["prior"],
                  row["gem"], row["meanl2"], row["gem_top1_m"]), flush=True)

    print("\n%s\nrecall@K  (%d tiles total)\n%s" % ("=" * 64, len(cen), "=" * 64))
    for tag in ("LOST", "GOOD", "ALL"):
        s = [r for r in res if tag == "ALL" or r["tag"] == tag]
        if not s:
            continue
        line = "  %-5s n=%2d |" % (tag, len(s))
        for nm in ("gem", "meanl2"):
            rk = np.array([r[nm] for r in s])
            line += "  %-7s" % nm + " ".join(
                "@%d %2.0f%%" % (k, 100 * np.mean(rk <= k)) for k in (1, 5, 20, 50))
            line += " med %4.0f |" % np.median(rk)
        print(line)
    print("\n  chance level: top 20 of %d tiles = %.1f%%" % (len(cen), 100 * 20 / len(cen)))
    print("  faisla: @20 chance se bahut upar -> coarse stage banta hai.")
    print("          @20 chance ke paas       -> XFeat ke descriptors kaafi nahi.")


if __name__ == "__main__":
    main()
