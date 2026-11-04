"""D, E, F, G, H from the CSVs the sweeps already wrote. No re-matching except
one frame for the inlier picture.
"""
import sys, os, math, csv as _csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import replay as R

D = "/data/DRONE/bench/xfeat_audit"
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.axisbelow": True})
C = ["#2b6cb0", "#c05621", "#2f855a", "#805ad5", "#b83280"]


def rd(name, nums):
    rows = list(_csv.DictReader(open(f"{D}/{name}")))
    for x in rows:
        for k in nums:
            try:
                x[k] = float(x[k])
            except Exception:
                x[k] = float("nan")
    return rows


# ---------- D: altitude vs match quality ----------
b = rd("batch.csv", ["alt", "prior_err", "score", "err", "ok", "inliers", "total", "overlap"])
fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.6))
bins = [(55, 70), (70, 85), (85, 100), (100, 118)]
lbl = [f"{a}-{c}" for a, c in bins]
acc = [100 * np.mean([x["ok"] for x in b if a <= x["alt"] < c]) for a, c in bins]
rat = [np.median([x["score"] for x in b if a <= x["alt"] < c]) for a, c in bins]
ax[0].bar(lbl, acc, color=C[0]); ax[0].set_ylabel("accept %"); ax[0].set_title("D1  accept rate by altitude")
ax[0].set_xlabel("altitude band (m)")
ax[1].bar(lbl, rat, color=C[1]); ax[1].set_ylabel("inlier ratio (median)")
ax[1].set_title("D2  match quality by altitude"); ax[1].set_xlabel("altitude band (m)")
pe = [x["prior_err"] for x in b]; sc = [x["score"] for x in b]
al = [x["alt"] for x in b]
s = ax[2].scatter(pe, sc, c=al, cmap="viridis", s=14, alpha=0.85)
ax[2].set_xscale("log"); ax[2].set_xlabel("prior error (m, log)"); ax[2].set_ylabel("inlier ratio")
ax[2].set_title("D3  quality tracks the PRIOR, not height")
plt.colorbar(s, ax=ax[2], label="altitude (m)")
cp = np.corrcoef(pe, sc)[0, 1]; ca = np.corrcoef(al, sc)[0, 1]
ax[2].text(0.03, 0.95, f"corr(prior err) = {cp:+.3f}\ncorr(altitude)  = {ca:+.3f}",
           transform=ax[2].transAxes, va="top", fontsize=8,
           bbox=dict(fc="white", ec="0.7", alpha=0.9))
plt.tight_layout(); plt.savefig(f"{D}/D_altitude_vs_quality.png"); plt.close()

# ---------- H: the sweeps, before/after ----------
w = rd("sweep_window.csv", ["frac", "err", "ok", "score", "ms"])
m = rd("sweep_matcher.csv", ["err", "ok", "n_in", "n_total", "ratio", "ms"])
fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))
fr = sorted(set(x["frac"] for x in w))
e50 = [np.nanmedian([x["err"] for x in w if x["frac"] == f]) for f in fr]
acc = [100 * np.mean([x["ok"] for x in w if x["frac"] == f]) for f in fr]
ms = [np.median([x["ms"] for x in w if x["frac"] == f]) for f in fr]
ax[0].plot(fr, e50, "o-", color=C[0], label="error p50 (m)")
ax[0].plot(fr, acc, "s--", color=C[1], label="accept %")
ax[0].plot(fr, np.array(ms) / 10, "^:", color=C[2], label="cost (ms/10)")
ax[0].axvline(0.60, color="0.4", lw=1, ls="-.")
ax[0].text(0.61, ax[0].get_ylim()[1] * 0.92, "flown", fontsize=8, color="0.3")
ax[0].set_xlabel("window_alt_frac"); ax[0].legend(fontsize=8)
ax[0].set_title("H1  bigger window: no accuracy, less accept")

tags = [t for t in dict.fromkeys(x["tag"] for x in m)]
ev = [np.nanmedian([x["err"] for x in m if x["tag"] == t]) for t in tags]
col = [C[0] if t == "FLOWN" else "0.65" for t in tags]
y = np.arange(len(tags))
ax[1].barh(y, ev, color=col); ax[1].set_yticks(y); ax[1].set_yticklabels(tags, fontsize=7)
ax[1].invert_yaxis(); ax[1].set_xlabel("error p50 (m)")
ax[1].axvline(ev[0], color=C[0], lw=1, ls="--")
ax[1].set_title("H2  18 configurations, one flat line")

t = rd("tilt.csv", ["raw", "corrected", "tilt", "shift", "alt", "ok"])
ax[2].scatter([x["shift"] for x in t], [x["raw"] for x in t], s=26, color=C[1],
              label="no nadir correction")
ax[2].scatter([x["shift"] for x in t], [x["corrected"] for x in t], s=26, color=C[2],
              marker="s", label="corrected (as flown)")
for x in t:
    ax[2].plot([x["shift"]] * 2, [x["raw"], x["corrected"]], color="0.8", lw=0.8, zorder=0)
ax[2].set_xlabel("alt x tan(tilt)  (m)"); ax[2].set_ylabel("fix error (m)")
ax[2].legend(fontsize=8); ax[2].set_title("H3  what actually moves the error: tilt")
rr = np.corrcoef([x["shift"] for x in t], [x["raw"] for x in t])[0, 1]
ax[2].text(0.03, 0.95, f"corr = {rr:+.3f}", transform=ax[2].transAxes, va="top",
           fontsize=8, bbox=dict(fc="white", ec="0.7", alpha=0.9))
plt.tight_layout(); plt.savefig(f"{D}/H_sweeps_and_tilt.png"); plt.close()

# ---------- F + G: trajectory overlay and error vs time, flight 478 ----------
d, rows = R.load_run("vps_00478")
def col(k):
    return np.array([R.fnum(x, k) for x in rows])
t = col("t"); vn, ve = col("north_m"), col("east_m")
gn, ge = col("gps1_north_m"), col("gps1_east_m")
good = np.isfinite(gn) & np.isfinite(ge) & (np.abs(gn) < 400) & (np.abs(ge) < 400)
fu = np.array([x.get("fix_used") == "1" for x in rows])
fl, fa = col("fix_lon"), col("fix_lat")
MN = 111320.0
lat0 = np.nanmedian(col("lat")); me = MN * math.cos(math.radians(lat0))
olat = np.nanmedian(col("lat") - vn / MN); olon = np.nanmedian(col("lon") - ve / me)
fN = (fa - olat) * MN; fE = (fl - olon) * me
fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.0))
ax[0].plot(ge[good], gn[good], "-", color=C[2], lw=1.4, label="M9N (truth)")
ax[0].plot(ve, vn, "-", color=C[0], lw=1.1, label="VPS estimate")
hit = fu & np.isfinite(fN)
ax[0].scatter(fE[hit], fN[hit], s=18, color=C[1], zorder=5, label="applied map fix")
ax[0].set_xlabel("east (m)"); ax[0].set_ylabel("north (m)"); ax[0].legend(fontsize=8)
ax[0].set_aspect("equal"); ax[0].set_title("F  flight 478: route and where the fixes landed")

err = np.full(len(t), np.nan)
gi = np.where(good)[0]
for i in range(len(t)):
    if not np.isfinite(fN[i]) or not fu[i]:
        continue
    j = gi[np.argmin(np.abs(t[gi] - t[i]))]
    if abs(t[j] - t[i]) < 1.0:
        err[i] = math.hypot(fN[i] - gn[j], fE[i] - ge[j])
vsp = col("vel_speed")
ax[1].plot(t, np.where(good, np.hypot(vn - gn, ve - ge), np.nan), "-", color=C[0],
           lw=1.0, label="estimate error vs M9N")
ax[1].scatter(t[np.isfinite(err)], err[np.isfinite(err)], s=16, color=C[1],
              label="applied fix error")
ax[1].axhline(50, color=C[4], lw=1, ls="--")
ax[1].text(t[0], 53, "50 m", fontsize=8, color=C[4])
ax[1].set_xlabel("t (s)"); ax[1].set_ylabel("error (m)"); ax[1].legend(fontsize=8)
ax[1].set_title("G  error over time; the bad fixes are all late")
plt.tight_layout(); plt.savefig(f"{D}/FG_route_and_error.png"); plt.close()
print("wrote D_altitude_vs_quality.png  H_sweeps_and_tilt.png  FG_route_and_error.png")
