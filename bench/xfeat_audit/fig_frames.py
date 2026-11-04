"""The actual frame, the actual map window, and the actual points.

Three cases drawn the same way, so they can be compared by eye:
  1  normal window, matched           - what a working fix looks like
  2  normal window, declined          - what the gate refusing looks like
  3  wide window at a 100 m offset    - the wide search, which recovered 0%

Each row: the camera frame as saved | the north-up query the matcher is
actually handed | the map window with every match drawn, green where the
points agree on one shift and red where they were thrown out.
"""
import os, sys, math
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replay as R
import fclog
sys.path.insert(0, "/data/DRONE/vps_vio")
sys.path.insert(0, "/data/DRONE/third_party/accelerated_features")
from map_cache import MapCache
from map_fix_worker import MapFixWorker
from map_match_xfeat import _as_3ch, _equalise, _translation
from map_query import north_up_query
from modules.xfeat import XFeat
import torch

torch.set_num_threads(2)
MDEG = 111320.0
cache = MapCache(os.path.realpath("/data/DRONE/maps/active.tif"))
model = XFeat(top_k=1024)

fr = R.frames("vps_00476", alt_min=58, step=3)
for f in fr:
    f["prior_err"] = R.metres(f["est_lon"], f["est_lat"], f["true_lon"], f["true_lat"])
hi = [f for f in fr if f["alt"] > 105 and f["prior_err"] < 12]


def attempt(f, *, offset_m=0.0, bearing=0.0, rad=None):
    img = cv2.imread(f["path"], cv2.IMREAD_GRAYSCALE)
    mlon = MDEG * math.cos(math.radians(f["true_lat"]))
    th = math.radians(bearing)
    plat = f["est_lat"] + offset_m * math.cos(th) / MDEG
    plon = f["est_lon"] + offset_m * math.sin(th) / mlon
    if rad is None:
        rad = float(np.clip(0.60 * f["alt"], 40.0, 150.0))
    win, wg = cache.window(plon, plat, rad, kind="grey")
    q, qc = north_up_query(img, altitude_m=f["alt"], focal_px=R.CAM["fx"],
                           yaw_deg=f["yaw"] + R.YAW_OFFSET, map_gsd_m=cache.gsd,
                           map_gsd_x_m=getattr(cache, "gsd_x", None),
                           shape="full", track_xy=(R.CAM["cx"], R.CAM["cy"]))
    qe, we = _as_3ch(_equalise(q)), _as_3ch(_equalise(win))
    out = model.match_xfeat(qe, we)
    src, dst = np.asarray(out[0], float), np.asarray(out[1], float)
    if len(src) >= 3:
        off, keep, mad = _translation(src, dst, 4.0)
        n_in = int(keep.sum())
        mc = np.array(qc, float) + off
        lon, lat = wg.to_lonlat(float(mc[0]), float(mc[1]))
        roll, pitch, _ = fclog.at("vps_00476", f["t"])
        dN, dE = MapFixWorker.nadir_offset_m(f["alt"], roll, pitch, f["yaw"])
        lat -= dN / MDEG
        lon -= dE / (MDEG * math.cos(math.radians(lat)))
        err = R.metres(lon, lat, f["true_lon"], f["true_lat"])
    else:
        off, keep, mad, n_in, err = np.zeros(2), np.zeros(0, bool), 4.0, 0, np.nan
    ratio = n_in / max(len(src), 1)
    margin = max(0.0, 1.0 - mad / 4.0)
    ok = (len(src) >= 12 and n_in >= 12 and ratio >= 0.07 and margin >= 0.20)
    return dict(img=img, q=cv2.cvtColor(qe, cv2.COLOR_BGR2GRAY),
                win=cv2.cvtColor(we, cv2.COLOR_BGR2GRAY), src=src, dst=dst,
                keep=keep, n_in=n_in, n=len(src), ratio=ratio, margin=margin,
                ok=ok, err=err, rad=rad, f=f, mad=mad)


fclog.attitude("vps_00476")
cases = []
for f in hi:
    a = attempt(f)
    if a["ok"] and np.isfinite(a["err"]) and a["err"] < 8:
        cases.append(("1. NORMAL window, MATCHED", a)); break
for f in hi:
    a = attempt(f)
    if not a["ok"]:
        cases.append(("2. NORMAL window, matcher ne MANA kiya", a)); break
f = cases[0][1]["f"]
cases.append(("3. WIDE window, prior 100 m galat (wide search)",
              attempt(f, offset_m=100.0, bearing=45.0,
                      rad=100.0 + 0.95 * f["alt"] + 30.0)))

fig, ax = plt.subplots(len(cases), 3, figsize=(14.5, 4.6 * len(cases)))
if len(cases) == 1:
    ax = ax[None, :]
for row, (title, a) in enumerate(cases):
    ax[row, 0].imshow(a["img"], cmap="gray")
    ax[row, 0].set_title("camera frame: %s\nalt %.0f m, yaw %.0f deg"
                         % (os.path.basename(a["f"]["path"]), a["f"]["alt"],
                            a["f"]["yaw"]), fontsize=8.5)
    ax[row, 1].imshow(a["q"], cmap="gray")
    ax[row, 1].set_title("north-up query (matcher ko yahi milta hai)\n%d x %d px"
                         % (a["q"].shape[1], a["q"].shape[0]), fontsize=8.5)
    # query and window side by side with the match lines
    qg, wgi = a["q"], a["win"]
    H = max(qg.shape[0], wgi.shape[0])
    canvas = np.zeros((H, qg.shape[1] + wgi.shape[1]), np.uint8)
    canvas[:qg.shape[0], :qg.shape[1]] = qg
    canvas[:wgi.shape[0], qg.shape[1]:] = wgi
    ax[row, 2].imshow(canvas, cmap="gray")
    sx = a["src"]; dx = a["dst"].copy()
    if len(dx):
        dx[:, 0] += qg.shape[1]
        k = a["keep"]
        for p, qq in zip(sx[~k][::2], dx[~k][::2]):
            ax[row, 2].plot([p[0], qq[0]], [p[1], qq[1]], color="#e53e3e",
                            lw=0.35, alpha=0.45)
        for p, qq in zip(sx[k], dx[k]):
            ax[row, 2].plot([p[0], qq[0]], [p[1], qq[1]], color="#38a169",
                            lw=0.6, alpha=0.95)
    ax[row, 2].set_title(
        "map window: radius %.0f m, %d x %d px\n%d points pakde, %d ne EK hi "
        "shift par sehmati di (ratio %.3f)\n%s   error %s"
        % (a["rad"], wgi.shape[1], wgi.shape[0], a["n"], a["n_in"], a["ratio"],
           "FIX bana" if a["ok"] else "MANA kiya",
           "%.2f m" % a["err"] if np.isfinite(a["err"]) else "-"),
        fontsize=8.5)
    for c in range(3):
        ax[row, c].axis("off")
    ax[row, 0].text(-0.06, 0.5, title, transform=ax[row, 0].transAxes,
                    rotation=90, va="center", ha="center", fontsize=10,
                    weight="bold")
plt.tight_layout()
plt.savefig("/data/DRONE/bench/xfeat_audit/frames_and_points.png", dpi=125)
print("wrote frames_and_points.png")
for title, a in cases:
    print("  %-46s %s  %d/%d inliers  ratio %.3f  margin %.2f  err %s"
          % (title, os.path.basename(a["f"]["path"]), a["n_in"], a["n"],
             a["ratio"], a["margin"],
             "%.2f m" % a["err"] if np.isfinite(a["err"]) else "-"))
