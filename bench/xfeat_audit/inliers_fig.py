"""E: what the matches actually look like. One good frame and one bad one,
query beside map window, inliers green and rejected red.
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2, torch
import replay as R
sys.path.insert(0, "/data/DRONE/vps_vio")
sys.path.insert(0, "/data/DRONE/third_party/accelerated_features")
from map_cache import MapCache
from map_match_xfeat import _as_3ch, _equalise, _translation
from map_query import north_up_query
from modules.xfeat import XFeat
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/data/DRONE/bench/xfeat_audit"
torch.set_num_threads(1)
cache = MapCache(os.path.realpath("/data/DRONE/maps/active.tif"))
model = XFeat(top_k=1024)
MN = 111320.0

cands = []
for run in ("vps_00476", "vps_00478"):
    fr = R.frames(run, alt_min=58, step=3)
    for f in fr:
        f["prior_err"] = R.metres(f["est_lon"], f["est_lat"], f["true_lon"], f["true_lat"])
        f["run"] = run
    cands += [f for f in fr if f["prior_err"] < 12.0]

def run_one(f):
    img = cv2.imread(f["path"], cv2.IMREAD_GRAYSCALE)
    rad = float(np.clip(0.60 * f["alt"], 40.0, 150.0))
    win, wg = cache.window(f["est_lon"], f["est_lat"], rad, kind="grey")
    q, qc = north_up_query(img, altitude_m=f["alt"], focal_px=R.CAM["fx"],
                           yaw_deg=f["yaw"] + R.YAW_OFFSET, map_gsd_m=cache.gsd,
                           map_gsd_x_m=getattr(cache, "gsd_x", None), shape="full",
                           track_xy=(R.CAM["cx"], R.CAM["cy"]))
    qe, we = _as_3ch(_equalise(q)), _as_3ch(_equalise(win))
    out = model.match_xfeat(qe, we)
    src, dst = np.asarray(out[0], float), np.asarray(out[1], float)
    off, keep, mad = _translation(src, dst, 4.0)
    mc = np.array(qc, float) + off
    lon, lat = wg.to_lonlat(float(mc[0]), float(mc[1]))
    err = R.metres(float(lon), float(lat), f["true_lon"], f["true_lat"])
    return dict(q=qe, win=we, src=src, dst=dst, keep=keep, err=err,
                ratio=keep.sum() / max(len(src), 1), f=f)

res = [run_one(f) for f in cands[::max(1, len(cands) // 14)]]
res.sort(key=lambda r: r["err"])
pick = [res[0], res[-1]]

fig, axes = plt.subplots(2, 1, figsize=(11, 9))
for ax, r in zip(axes, pick):
    qg = cv2.cvtColor(r["q"], cv2.COLOR_BGR2GRAY)
    wgi = cv2.cvtColor(r["win"], cv2.COLOR_BGR2GRAY)
    H = max(qg.shape[0], wgi.shape[0])
    canvas = np.zeros((H, qg.shape[1] + wgi.shape[1]), np.uint8)
    canvas[:qg.shape[0], :qg.shape[1]] = qg
    canvas[:wgi.shape[0], qg.shape[1]:] = wgi
    ax.imshow(canvas, cmap="gray", interpolation="nearest")
    sx = r["src"]; dx = r["dst"].copy(); dx[:, 0] += qg.shape[1]
    k = r["keep"]
    for a, b in zip(sx[~k][::3], dx[~k][::3]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#e53e3e", lw=0.35, alpha=0.5)
    for a, b in zip(sx[k], dx[k]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#38a169", lw=0.55, alpha=0.9)
    ax.set_title("%s  t=%.0f s  alt %.0f m   |  %d matches, %d inliers (ratio %.3f)   "
                 "fix error %.1f m   [green = agree on one shift, red = rejected]"
                 % (r["f"]["run"], r["f"]["t"], r["f"]["alt"], len(sx), k.sum(),
                    r["ratio"], r["err"]), fontsize=9)
    ax.axis("off")
plt.tight_layout(); plt.savefig(f"{D}/E_inliers_outliers.png", dpi=140); plt.close()
print("wrote E_inliers_outliers.png")
for r in res:
    print("  %-10s t=%6.0f alt %5.1f  %4d matches %3d inliers  ratio %.3f  err %6.2f m"
          % (r["f"]["run"], r["f"]["t"], r["f"]["alt"], len(r["src"]),
             r["keep"].sum(), r["ratio"], r["err"]))
