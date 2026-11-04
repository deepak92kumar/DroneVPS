"""The three claims that matter, drawn so they can be checked.

A  the declared sigma against the real error - the 15x understatement
B  the flow's scale per 20 s window, which is what actually drifts
C  drift rate against the map's correction authority
D  the detector: when each rule fires, and what it costs in false alarms
"""
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
D = "/data/DRONE/bench/xfeat_audit"
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.axisbelow": True})
C = ["#2b6cb0", "#c05621", "#2f855a", "#805ad5", "#b83280", "#718096"]

P = "/home/abc/NVME/DRONE/flight_logs/vps/vps_00478/vps_00478.csv"
r = list(csv.DictReader(open(P)))
F = lambda x, k: (float(x[k]) if x.get(k) not in (None, "") else np.nan)
g = lambda k: np.array([F(x, k) for x in r])
t, n, e = g("t"), g("north_m"), g("east_m")
gn, ge, sg = g("gps1_north_m"), g("gps1_east_m"), g("sigma_m")
ok = (np.isfinite(gn) & np.isfinite(ge) & (np.abs(gn) < 400)
      & (np.abs(ge) < 400) & np.isfinite(n) & np.isfinite(sg))
t, n, e, gn, ge, sg = [v[ok] for v in (t, n, e, gn, ge, sg)]
err = np.hypot(n - gn, e - ge)

fig, ax = plt.subplots(2, 2, figsize=(13, 8.4))

# A ---- sigma vs truth
a = ax[0, 0]
a.plot(t, err, "-", color=C[1], lw=1.4, label="REAL error vs M9N")
a.plot(t, sg, "-", color=C[0], lw=1.6, label="sigma the aircraft DECLARED")
a.axhline(25, color=C[4], ls="--", lw=1)
a.text(t[0] + 5, 27, "EK3_GLITCH_RAD 25 m", fontsize=8, color=C[4])
a.axvline(409, color=C[2], lw=1.4, ls=":")
a.text(412, 100, "detector fires\nt=409, err 24.7 m", fontsize=8, color=C[2])
a.set_yscale("log"); a.set_ylim(1, 300)
a.set_xlabel("t (s)"); a.set_ylabel("metres (log)")
a.legend(fontsize=8, loc="upper left")
a.set_title("A  124.7 m wrong while declaring 8.5 m")

# B ---- flow scale per 20 s window
b = ax[0, 1]
win, sc = [], []
for lo in range(200, 780, 20):
    m = (t >= lo) & (t < lo + 20)
    if m.sum() < 10:
        continue
    i0, i1 = np.where(m)[0][[0, -1]]
    de_ = np.hypot(n[i1] - n[i0], e[i1] - e[i0])
    dg = np.hypot(gn[i1] - gn[i0], ge[i1] - ge[i0])
    if dg < 3:
        continue
    win.append(lo + 10); sc.append(de_ / dg)
b.bar(win, sc, width=16, color=[C[1] if abs(s - 1) > 0.25 else C[2] for s in sc])
b.axhline(1.0, color="0.3", lw=1.2)
b.axhline(np.median(sc), color=C[0], ls="--", lw=1.2,
          label=f"median {np.median(sc):.3f}")
b.set_ylim(0, 2.0)
b.set_xlabel("t (s)"); b.set_ylabel("flow distance / M9N distance")
b.legend(fontsize=8)
b.set_title("B  the flow's scale swings 0.38 to 1.70 per 20 s")

# C ---- drift rate vs authority
c = ax[1, 0]
phases, rates = [], []
for lo, hi in ((200, 300), (300, 400), (400, 500), (500, 600), (600, 800)):
    m = (t >= lo) & (t < hi)
    if m.sum() < 5:
        continue
    phases.append(f"{lo}-{hi}")
    rates.append(np.polyfit(t[m], err[m], 1)[0])
c.bar(phases, rates, color=[C[1] if v > 0.25 else C[2] for v in rates])
c.axhline(0.25, color=C[0], lw=1.6, ls="--")
c.text(0.02, 0.26, "map's authority: 0.5 m per fix / 2 s = 0.250 m/s",
       transform=c.get_yaxis_transform(), fontsize=8, color=C[0], va="bottom")
c.set_ylabel("drift rate (m/s)"); c.set_xlabel("phase (s)")
c.set_title("C  drift outruns the map 2.5x - cap is not the limiter")

# D ---- detector rules
d = ax[1, 1]
rules = [("rate10 <= 0.3", 35, 4.0), ("rate10 <= 0.2", 25, 4.0),
         ("rate20 <= 0.3", 26, 4.0), ("gap >= 5", 18, 24.7),
         ("gap >= 8", 11, 21.8), ("secs >= 20", 2, 34.7),
         ("secs >= 40", 0, 69.2), ("rate10<=0.3 & alt>75", 2, 24.7),
         ("gap>=5 & alt>75", 1, 24.7), ("gap>=5 & alt>90", 1, 24.7),
         ("gap>=4 & score<0.04", 9, 23.2),
         ("gap>=5 & score<0.05 & alt>75", 0, 24.7)]
for name, fa, er in rules:
    best = (fa == 0 and er < 30)
    d.scatter(fa, er, s=70 if best else 36,
              color=C[2] if best else C[5], zorder=5 if best else 3,
              edgecolor="white" if best else "none", linewidth=1.2)
    if best or fa > 8:
        d.annotate(name, (fa, er), textcoords="offset points",
                   xytext=(7, 4 if best else -10), fontsize=7.5,
                   color=C[2] if best else "0.35")
d.axhspan(0, 30, color=C[2], alpha=0.07)
d.axvline(0, color="0.5", lw=0.8)
d.set_xlabel("false alarms on flight 476 (never lost)")
d.set_ylabel("error when it first fires on 478 (m)")
d.set_title("D  bottom-left is what is wanted")
d.set_xlim(-2, 38)

plt.tight_layout()
plt.savefig(f"{D}/lost_detector.png")
print("wrote lost_detector.png")
