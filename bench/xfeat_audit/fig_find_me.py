"""The map, the lost drone, and who found it.

One panel per frame. The map as the matcher saw it, the tiles that were
searched, where the drone THOUGHT it was, where the M9N says it ACTUALLY was,
and where each matcher put it.
"""
import math
import os
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replay as R
sys.path.insert(0, "/data/DRONE/vps_vio")
from map_cache import MapCache

MDEG = 111320.0
SEARCH_R, TILE_M, STEP_M = 250.0, 150.0, 75.0
cache = MapCache(os.path.realpath("/data/DRONE/maps/active.tif"))
res = pickle.load(open("/data/DRONE/bench/xfeat_audit/find_me.pkl", "rb"))
C = {"est": "#2b6cb0", "true": "#2f855a", "mnn": "#c05621", "lg": "#805ad5"}

n = len(res)
fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 6.6))
if n == 1:
    axes = [axes]

for ax, r_ in zip(axes, res):
    f = r_["f"]
    mlon = MDEG * math.cos(math.radians(f["true_lat"]))
    # A map crop big enough to hold the search square and both positions.
    pad = SEARCH_R + TILE_M
    win, wg = cache.window(f["est_lon"], f["est_lat"], pad, kind="grey")
    H, W = win.shape[:2]

    def to_px(lon, lat):
        """metres east/north of the window centre -> pixel in this crop."""
        de = (lon - f["est_lon"]) * mlon
        dn = (lat - f["est_lat"]) * MDEG
        return W / 2.0 + de / (2 * pad / W), H / 2.0 - dn / (2 * pad / H)

    ax.imshow(win, cmap="gray", interpolation="nearest")
    # every tile that was searched
    grid = np.arange(-SEARCH_R, SEARCH_R + 1, STEP_M)
    px_per_m = W / (2 * pad)
    for dn in grid:
        for de in grid:
            cx, cy = to_px(f["est_lon"] + de / mlon, f["est_lat"] + dn / MDEG)
            s = TILE_M * px_per_m
            ax.add_patch(Rectangle((cx - s / 2, cy - s / 2), s, s, fill=False,
                                   edgecolor="#f6e05e", lw=0.25, alpha=0.5))
    ex, ey = to_px(f["est_lon"], f["est_lat"])
    tx, ty = to_px(f["true_lon"], f["true_lat"])
    ax.plot([ex, tx], [ey, ty], "-", color="white", lw=1.2, alpha=0.8)
    ax.plot(ex, ey, "o", ms=13, mfc="none", mec=C["est"], mew=2.6)
    ax.annotate("drone ko laga\nwo YAHAN hai", (ex, ey), color=C["est"],
                fontsize=9, weight="bold",
                textcoords="offset points", xytext=(14, 10))
    ax.plot(tx, ty, "*", ms=20, color=C["true"], mec="white", mew=0.8)
    ax.annotate("ASLI jagah (M9N)", (tx, ty), color=C["true"], fontsize=9,
                weight="bold", textcoords="offset points", xytext=(14, -16))

    lines = []
    for tag, nm in (("mnn", "XFeat+MNN (aaj udta hai)"),
                    ("lg", "XFeat+LighterGlue")):
        t = r_["tiles"].get(tag, {})
        b = t.get("best")
        if b is None:
            lines.append(f"{nm}: {t.get('n_try', 0)} tiles me KUCH NAHI MILA"
                         f"  ({t.get('secs', 0):.0f} s)")
            continue
        mx, my = to_px(b["lon"], b["lat"])
        ax.plot(mx, my, "X", ms=15, color=C[tag], mec="white", mew=1.2)
        # the tile it won on
        cx, cy = to_px(f["est_lon"] + b["de"] / mlon,
                       f["est_lat"] + b["dn"] / MDEG)
        s = TILE_M * px_per_m
        ax.add_patch(Rectangle((cx - s / 2, cy - s / 2), s, s, fill=False,
                               edgecolor=C[tag], lw=2.2))
        lines.append("%s: %d points sehmat, asli se %.2f m  (%.0f s)"
                     % (nm, b["n_in"], b["err"], t.get("secs", 0)))
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    ax.set_title("%s   alt %.0f m\ndrone %.1f m kho chuka tha,  %d tiles "
                 "dhoondhe\n%s" % (
                     os.path.basename(f["path"]), f["alt"], f["prior_err"],
                     len(grid) ** 2, "\n".join(lines)), fontsize=9)

plt.tight_layout()
p = "/data/DRONE/bench/xfeat_audit/find_me_map.png"
plt.savefig(p, dpi=120)
print("wrote", p)
for r_ in res:
    f = r_["f"]
    print("  %-14s lost %6.1f m |" % (os.path.basename(f["path"]),
                                      f["prior_err"]), end="")
    for tag in ("mnn", "lg"):
        b = r_["tiles"].get(tag, {}).get("best")
        print("  %-4s %s" % (tag, "KUCH NAHI" if b is None
                             else "%.2f m (%d pts)" % (b["err"], b["n_in"])),
              end="")
    print()
