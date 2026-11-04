"""Does the tilt correction account for the along-heading scatter?

The vps CSV carries only attitude DELTAS, so absolute roll/pitch come from the
FC log. The two clocks are aligned by cross-correlating yaw, which both record.

Then the flown configuration is replayed twice over the same frames: once as
the bench has been doing it (no nadir correction, which was my omission) and
once with map_fix_worker's own nadir_offset_m applied, imported not copied.
"""
import sys, os, math, csv as _csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
import replay as R
sys.path.insert(0, "/data/DRONE/vps_vio")
from map_cache import MapCache
from map_match_xfeat import locate
from map_fix_worker import MapFixWorker

BIN = "/home/abc/Downloads/2026-11-04 12-54-13.bin"
RUN = "vps_00478"
MN = 111320.0

from pymavlink import mavutil
m = mavutil.mavlink_connection(BIN)
att = []
while True:
    msg = m.recv_match(type="ATT")
    if msg is None:
        break
    att.append((msg.TimeUS / 1e6, msg.Roll, msg.Pitch, msg.Yaw))
att = np.array(att)
print("ATT samples:", len(att), "span %.1f s" % (att[-1, 0] - att[0, 0]), flush=True)

d, rows = R.load_run(RUN)
vt = np.array([R.fnum(r, "t") for r in rows])
vy = np.array([R.fnum(r, "yaw_deg") for r in rows])
good = np.isfinite(vt) & np.isfinite(vy)
vt, vy = vt[good], vy[good]

# Align by yaw: resample both to 5 Hz and slide.
def grid(t, y, dt=0.2):
    g = np.arange(t[0], t[-1], dt)
    return g, np.interp(g, t, np.unwrap(np.radians(y)))
gv, yv = grid(vt, vy)
ga, ya = grid(att[:, 0], att[:, 3])
yv = yv - yv.mean(); ya = ya - ya.mean()
best, bl = None, None
n = min(len(yv), len(ya))
for lag in range(-len(ya) + n // 2, len(yv) - n // 2):
    i0 = max(0, lag); j0 = max(0, -lag)
    L = min(len(yv) - i0, len(ya) - j0)
    if L < n // 2:
        continue
    c = float(np.dot(yv[i0:i0 + L], ya[j0:j0 + L]) / L)
    if best is None or c > best:
        best, bl = c, lag
off = gv[0] - ga[0] + bl * 0.2       # vps_t = fc_t + off
print("yaw alignment: lag %d samples -> vps_t = fc_t %+.2f s  (corr score %.3f)" % (bl, off, best), flush=True)

def att_at(t):
    ft = t - off
    i = np.searchsorted(att[:, 0], ft)
    i = min(max(i, 0), len(att) - 1)
    return att[i, 1], att[i, 2]      # roll, pitch in deg

cache = MapCache(os.path.realpath("/data/DRONE/maps/active.tif"))
fr = R.frames(RUN, alt_min=58, step=3)
for f in fr:
    f["prior_err"] = R.metres(f["est_lon"], f["est_lat"], f["true_lon"], f["true_lat"])
fr = [f for f in fr if f["prior_err"] < 12.0]
fr.sort(key=lambda f: f["alt"])
sel = [fr[i] for i in np.linspace(0, len(fr) - 1, min(14, len(fr))).round().astype(int)]

out = []
for f in sel:
    img = cv2.imread(f["path"], cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    roll, pitch = att_at(f["t"])
    rad = float(np.clip(0.60 * f["alt"], 40.0, 150.0))
    win, wg = cache.window(f["est_lon"], f["est_lat"], rad, kind="grey")
    res = locate(img, win, wg, altitude_m=f["alt"], focal_px=R.CAM["fx"],
                 yaw_deg=f["yaw"] + R.YAW_OFFSET, map_gsd_m=cache.gsd,
                 map_gsd_x_m=getattr(cache, "gsd_x", None),
                 min_score=0.07, min_margin=0.20,
                 principal_xy=(R.CAM["cx"], R.CAM["cy"]), query_shape="full")
    if not res.lon:
        continue
    me = MN * math.cos(math.radians(f["true_lat"]))
    raw = (R.metres(res.lon, res.lat, f["true_lon"], f["true_lat"]))
    dN, dE = MapFixWorker.nadir_offset_m(f["alt"], roll, pitch, f["yaw"])
    clat = res.lat - dN / MN
    clon = res.lon - dE / (MN * math.cos(math.radians(res.lat)))
    cor = R.metres(clon, clat, f["true_lon"], f["true_lat"])
    tilt = math.degrees(math.acos(max(-1.0, min(1.0,
           math.cos(math.radians(roll)) * math.cos(math.radians(pitch))))))
    out.append(dict(t=f["t"], alt=f["alt"], ok=int(res.ok), ratio=res.score,
                    roll=roll, pitch=pitch, tilt=tilt,
                    shift=math.hypot(dN, dE), raw=raw, corrected=cor,
                    eN=(clat - f["true_lat"]) * MN, eE=(clon - f["true_lon"]) * me,
                    yaw=f["yaw"]))
    print("alt%6.1f ok%d r%.3f | roll%+6.2f pitch%+6.2f tilt%5.2f -> shift%6.2f m "
          "| err %6.2f -> %6.2f m" % (f["alt"], res.ok, res.score, roll, pitch,
          tilt, out[-1]["shift"], raw, cor), flush=True)

a = np.array([[o["raw"], o["corrected"]] for o in out])
ok = np.array([o["ok"] for o in out], bool)
for nm, msk in (("all", np.ones(len(out), bool)), ("accepted", ok)):
    v = a[msk]
    print("\n%-9s n=%2d  raw p50 %5.2f p90 %5.2f max %5.2f  ->  corrected p50 %5.2f p90 %5.2f max %5.2f" % (
        nm, msk.sum(), np.median(v[:, 0]), np.percentile(v[:, 0], 90), v[:, 0].max(),
        np.median(v[:, 1]), np.percentile(v[:, 1], 90), v[:, 1].max()))
eN = np.array([o["eN"] for o in out]); eE = np.array([o["eE"] for o in out])
yw = np.radians(np.array([o["yaw"] for o in out]) - 4.42)
al = eN * np.cos(yw) + eE * np.sin(yw); cr = -eN * np.sin(yw) + eE * np.cos(yw)
print("corrected, in the camera frame: ALONG sd %5.2f  ACROSS sd %5.2f  (was 9.45 / 3.80 uncorrected)" % (al.std(), cr.std()))
print("corr(tilt, raw err) = %+.3f    corr(shift, raw err) = %+.3f" % (
    np.corrcoef([o["tilt"] for o in out], a[:, 0])[0, 1],
    np.corrcoef([o["shift"] for o in out], a[:, 0])[0, 1]))
with open("/data/DRONE/bench/xfeat_audit/tilt.csv", "w", newline="") as fh:
    w = _csv.DictWriter(fh, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
