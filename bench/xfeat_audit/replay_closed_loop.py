#!/usr/bin/env python3
"""Does the trust change stop the runaway? Closed loop, the fuser in it.

    python3 replay_closed_loop.py vps_00478 [--step 4]

The fix-level numbers say XFeat publishes nothing over 20 m wrong. They do
NOT say the estimate stops running away, because the two are coupled: the
search window is centred on the estimate, so a drifting estimate is shown the
wrong ground and the matcher's failure is then a CONSEQUENCE of the drift
rather than its cause. Flight 478 went prior 11 m at t=300-400 to 124 m at
t=500-600. Only a closed loop can answer it.

So this runs the real MapFuser, fed the flow steps the aircraft actually
measured (step_dn/step_de/step_dt_s from the log), and re-runs the matchers
INSIDE the loop with the window centred on the replayed estimate - which
diverges from the logged one as soon as the policy differs. Scored against
the M9N throughout.

What it cannot replay: the velocity the FC was given, the EKF, and therefore
ALT_HOLD. This is the position filter only. That is the question being asked.
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
from map_cache import MapCache                          # noqa: E402
from map_fuse import MapFuser                           # noqa: E402
from map_fix_worker import MapFixWorker, _GATES         # noqa: E402

MDEG = 111320.0
POLICIES = {"old (ncc wins)": ["ncc", "xfeat"],
            "new (xfeat trusted)": ["xfeat"],
            "no map at all": []}


def _locate(name):
    if name == "ncc":
        from map_match_ncc import locate
    else:
        from map_match_xfeat import locate
    return locate


def radius(alt, name, focal):
    r = 0.60 * alt
    if name == "ncc":
        r = max(r, 0.5 * 1280.0 * alt / focal + 45.0)
    return float(np.clip(r, 40.0, 150.0))


def load(run, step, min_alt):
    """Frames with a saved image, plus the flow step the aircraft measured."""
    d, rows = R.load_run(run)
    out = []
    for r in rows:
        ff = r.get("frame_file") or ""
        t = R.fnum(r, "t")
        gn, ge = R.fnum(r, "gps1_north_m"), R.fnum(r, "gps1_east_m")
        rec = dict(t=t, alt=R.fnum(r, "alt_m"), yaw=R.fnum(r, "yaw_deg"),
                   dn=R.fnum(r, "step_dn"), de=R.fnum(r, "step_de"),
                   dt=R.fnum(r, "step_dt_s"),
                   true_n=gn if abs(gn) < 400 else np.nan,
                   true_e=ge if abs(ge) < 400 else np.nan,
                   path=os.path.join(d, os.path.basename(ff)) if ff else None,
                   logged_n=R.fnum(r, "north_m"), logged_e=R.fnum(r, "east_m"))
        out.append(rec)
    # The origin, so metres and lon/lat convert the same way the node did.
    lat0 = np.nanmedian([R.fnum(r, "lat") for r in rows[:200]])
    mlon = MDEG * math.cos(math.radians(lat0))
    olat = np.nanmedian([R.fnum(r, "lat") - R.fnum(r, "north_m") / MDEG
                         for r in rows[:200]])
    olon = np.nanmedian([R.fnum(r, "lon") - R.fnum(r, "east_m") / mlon
                         for r in rows[:200]])
    # Only every `step`-th frame is a FIX ATTEMPT; every row still drives the
    # flow step, or the replayed trajectory is not the flown one.
    attempt = set()
    k = 0
    for i, rec in enumerate(out):
        if rec["path"] and rec["alt"] >= min_alt:
            if k % step == 0:
                attempt.add(i)
            k += 1
    return out, attempt, (olon, olat, mlon)


def run_policy(run, rows, attempt, origin, trust, cache, *, verbose=False,
               cap=None):
    import cv2
    olon, olat, mlon = origin
    kw = {} if cap is None else dict(max_correction_m=cap)
    fuser = MapFuser(flow_drift_frac=0.07, fix_sigma_m=20.0, **kw)
    started = False
    track = []
    n_fix = n_att = 0
    for i, rec in enumerate(rows):
        if not np.isfinite(rec["true_n"]):
            continue
        if not started:
            # Start where the aircraft started, which is the origin.
            started = True
        if np.isfinite(rec["dn"]) and np.isfinite(rec["de"]):
            fuser.predict(rec["dn"], rec["de"],
                          dt=rec["dt"] if np.isfinite(rec["dt"]) else None)
        if i in attempt and trust:
            n_att += 1
            # THE WINDOW IS CENTRED ON THE REPLAYED ESTIMATE. This is the
            # whole point: it is not where the aircraft thought it was.
            pn, pe = float(fuser.pos[0]), float(fuser.pos[1])
            plat = olat + pn / MDEG
            plon = olon + pe / mlon
            img = cv2.imread(rec["path"], cv2.IMREAD_GRAYSCALE)
            if img is not None and cache.covers(plon, plat):
                roll, pitch, _ = fclog.at(run, rec["t"])
                dN, dE = MapFixWorker.nadir_offset_m(rec["alt"], roll, pitch,
                                                     rec["yaw"])
                got = {}
                for name in ("ncc", "xfeat"):
                    try:
                        win, wg = cache.window(
                            plon, plat, radius(rec["alt"], name, R.CAM["fx"]),
                            kind="grey")
                        res = _locate(name)(
                            img, win, wg, altitude_m=rec["alt"],
                            focal_px=R.CAM["fx"],
                            yaw_deg=rec["yaw"] + R.YAW_OFFSET,
                            map_gsd_m=cache.gsd,
                            map_gsd_x_m=getattr(cache, "gsd_x", None),
                            principal_xy=(R.CAM["cx"], R.CAM["cy"]),
                            query_shape="full", **_GATES[name])
                    except Exception:
                        continue
                    if res.ok:
                        got[name] = res
                pick = next((got[m] for m in trust if m in got), None)
                if pick is not None:
                    flat = pick.lat - dN / MDEG
                    flon = pick.lon - dE / (MDEG * math.cos(
                        math.radians(pick.lat)))
                    if fuser.update((flat - olat) * MDEG, (flon - olon) * mlon,
                                    prior_north=pn, prior_east=pe, t=rec["t"]):
                        n_fix += 1
        track.append((rec["t"], float(fuser.pos[0]), float(fuser.pos[1]),
                      rec["true_n"], rec["true_e"], fuser.sigma_m))
        if verbose and i in attempt and n_att % 25 == 0:
            e = math.hypot(fuser.pos[0] - rec["true_n"],
                           fuser.pos[1] - rec["true_e"])
            print(f"      t={rec['t']:5.0f} err {e:7.2f} m  sigma {fuser.sigma_m:6.2f}"
                  f"  fixes {n_fix}/{n_att}", flush=True)
    return np.array(track), n_fix, n_att


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--step", type=int, default=4)
    ap.add_argument("--min-alt", type=float, default=58.0)
    ap.add_argument("--map", default="/data/DRONE/maps/active.tif")
    ap.add_argument("--caps", default="",
                    help="comma-separated max_correction_m to compare under "
                         "the xfeat-trusted policy, e.g. 0.5,2,5")
    a = ap.parse_args()

    cache = MapCache(os.path.realpath(a.map))
    fclog.attitude(a.run)
    rows, attempt, origin = load(a.run, a.step, a.min_alt)
    print(f"{a.run}: {len(rows)} log rows, {len(attempt)} fix attempts\n",
          flush=True)

    out = {}
    t0 = time.time()
    plans = list(POLICIES.items())
    if a.caps:
        # THE CAP IS THE MAP'S CORRECTION AUTHORITY. max_correction_m metres
        # per fix at fix_period seconds is a rate: 0.5 m / 2 s = 0.25 m/s.
        # Flight 478 drifted at 0.546 m/s from t=400 and 0.625 m/s from
        # t=500, so the map was outrun by two and a half times whatever it
        # said. Raising it is only defensible once the fixes are bounded,
        # which is what trusting XFeat buys: NCC was publishing fixes up to
        # 180.74 m wrong, and a larger cap would have applied those too.
        plans = [(f"xfeat, cap {c}", ["xfeat"], float(c))
                 for c in a.caps.split(",")]
        plans.insert(0, ("no map at all", [], None))
    for plan in plans:
        label, trust = plan[0], plan[1]
        cap = plan[2] if len(plan) > 2 else None
        print(f"  {label} ...", flush=True)
        track, n_fix, n_att = run_policy(a.run, rows, attempt, origin, trust,
                                         cache, verbose=True, cap=cap)
        err = np.hypot(track[:, 1] - track[:, 3], track[:, 2] - track[:, 4])
        out[label] = (track, err, n_fix, n_att)
        print(f"    fixes {n_fix}/{n_att}   err p50 {np.median(err):6.2f}"
              f"  p90 {np.percentile(err, 90):7.2f}  FINAL {err[-1]:7.2f}"
              f"  WORST {err.max():7.2f} m", flush=True)

    print(f"\n{'=' * 74}\nposition error against the M9N, by phase\n{'=' * 74}")
    print("  %-22s %s" % ("policy", "  ".join(f"t {lo}-{hi}" for lo, hi in
          ((200, 300), (300, 400), (400, 500), (500, 600), (600, 900)))))
    for label, (track, err, nf, na) in out.items():
        cells = []
        for lo, hi in ((200, 300), (300, 400), (400, 500), (500, 600), (600, 900)):
            m = (track[:, 0] >= lo) & (track[:, 0] < hi)
            cells.append("%8.1f" % np.median(err[m]) if m.any() else "       -")
        print(f"  {label:22s}" + "  ".join(cells))
    print("\n  (median metres of error in each window; the flown log drifted "
          "11 m at t=300-400 and 124 m at t=500-600)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
        pal = ["#c05621", "#2f855a", "#2b6cb0", "#805ad5", "#b83280"]
        cols = {k: ("#718096" if k == "no map at all" else pal[i % len(pal)])
                for i, k in enumerate(out)}
        tr = next(iter(out.values()))[0]
        ax[0].plot(tr[:, 4], tr[:, 3], "-", color="#2b6cb0", lw=1.6,
                   label="M9N (truth)")
        for label, (track, err, nf, na) in out.items():
            ax[0].plot(track[:, 2], track[:, 1], "-", color=cols[label], lw=1.1,
                       label=f"{label}  ({nf}/{na} fixes)")
            ax[1].plot(track[:, 0], err, "-", color=cols[label], lw=1.2,
                       label=label)
        ax[0].set_xlabel("east (m)"); ax[0].set_ylabel("north (m)")
        ax[0].legend(fontsize=8); ax[0].set_aspect("equal"); ax[0].grid(alpha=.25)
        ax[0].set_title(f"{a.run}: replayed trajectory, closed loop")
        ax[1].axhline(25, color="#b83280", ls="--", lw=1)
        ax[1].text(tr[0, 0], 27, "EK3_GLITCH_RAD 25 m", fontsize=8, color="#b83280")
        ax[1].set_xlabel("t (s)"); ax[1].set_ylabel("error vs M9N (m)")
        ax[1].legend(fontsize=8); ax[1].grid(alpha=.25)
        ax[1].set_title("position error over time")
        plt.tight_layout()
        p = f"/data/DRONE/bench/xfeat_audit/closed_loop_{a.run}.png"
        plt.savefig(p, dpi=130); plt.close()
        print(f"\n  -> {p}")
    except Exception as exc:
        print(f"  (no plot: {exc})")
    print(f"\n({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
