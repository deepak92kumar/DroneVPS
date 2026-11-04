#!/usr/bin/env python3
"""Which matcher should the aircraft believe? Measured, per flight.

    python3 who_is_right.py vps_00476 vps_00478 [--step 4] [--min-alt 58]

Runs every matcher the worker can run over the saved frames of each flight,
each with the radius map_fix_worker gives it, each nadir-corrected with
absolute roll and pitch from the FC log, and scores all of them against the
M9N - the raw receiver, which is the only position aboard that is not partly
our own injection coming back.

Then it prints what the aircraft would have done under each trust setting and
recommends one. The recommendation is the worst case, not the median: a map
aid is judged by how wrong it can be while sounding confident, because that
is what moves the aircraft. Run it after any flight that carried frames; it
needs no arguments beyond the run names and finds the FC log by correlating
yaw and altitude, since the log's filename carries the FC's own clock and
identifies nothing.

Nothing here touches flight code. It imports it.
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
from map_fix_worker import MapFixWorker, _GATES         # noqa: E402

MDEG = 111320.0
MATCHERS = ("ncc", "xfeat")


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


def run_flight(run, cache, *, step, min_alt):
    import cv2
    fclog.attitude(run)
    fr = R.frames(run, alt_min=min_alt, step=step)
    print(f"  {run}: {len(fr)} frames", flush=True)
    out = []
    for k, f in enumerate(fr):
        img = cv2.imread(f["path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        roll, pitch, _ = fclog.at(run, f["t"])
        dN, dE = MapFixWorker.nadir_offset_m(f["alt"], roll, pitch, f["yaw"])
        rec = dict(run=run, t=f["t"], alt=f["alt"],
                   prior_err=R.metres(f["est_lon"], f["est_lat"],
                                      f["true_lon"], f["true_lat"]),
                   tilt=math.degrees(math.acos(max(-1.0, min(1.0,
                       math.cos(math.radians(roll))
                       * math.cos(math.radians(pitch)))))))
        for name in MATCHERS:
            try:
                win, wg = cache.window(f["est_lon"], f["est_lat"],
                                       radius(f["alt"], name, R.CAM["fx"]),
                                       kind="grey")
                res = _locate(name)(
                    img, win, wg, altitude_m=f["alt"], focal_px=R.CAM["fx"],
                    yaw_deg=f["yaw"] + R.YAW_OFFSET, map_gsd_m=cache.gsd,
                    map_gsd_x_m=getattr(cache, "gsd_x", None),
                    principal_xy=(R.CAM["cx"], R.CAM["cy"]),
                    query_shape="full", **_GATES[name])
            except Exception as exc:
                rec[name] = dict(ok=False, err=float("nan"), why=f"raised {exc}")
                continue
            lat = res.lat - dN / MDEG
            lon = res.lon - dE / (MDEG * math.cos(math.radians(res.lat or 30.9)))
            rec[name] = dict(
                ok=bool(res.ok), lon=lon, lat=lat,
                err=(R.metres(lon, lat, f["true_lon"], f["true_lat"])
                     if res.lon else float("nan")),
                score=res.score, why=res.reason or "")
        a, b = (rec[m] for m in MATCHERS)
        rec["disagree"] = (R.metres(a["lon"], a["lat"], b["lon"], b["lat"])
                           if a.get("ok") and b.get("ok") else float("nan"))
        out.append(rec)
        if k and k % 50 == 0:
            print(f"     {k}/{len(fr)}", flush=True)
    return out


def _stats(errs):
    e = np.array([v for v in errs if np.isfinite(v)])
    if not len(e):
        return None
    return dict(n=len(e), p50=np.median(e), p90=np.percentile(e, 90),
                worst=e.max(), over20=int((e > 20).sum()),
                over50=int((e > 50).sum()))


def policy(rows, trust):
    """What would have been published, and how wrong it was."""
    errs = []
    for r in rows:
        for m in trust:
            if r.get(m, {}).get("ok"):
                errs.append(r[m]["err"])
                break
    return errs


def report(rows, label):
    print(f"\n{'=' * 78}\n{label}   ({len(rows)} attempts)\n{'=' * 78}")
    print("\nwhat each matcher did, and how wrong it was when it spoke")
    for m in MATCHERS:
        acc = [r for r in rows if r.get(m, {}).get("ok")]
        s = _stats([r[m]["err"] for r in acc])
        if s is None:
            print(f"  {m:6s} accepted nothing")
            continue
        print(f"  {m:6s} accepted {len(acc):4d}/{len(rows)} ({100*len(acc)/len(rows):3.0f}%)"
              f" | err p50 {s['p50']:6.2f}  p90 {s['p90']:7.2f}  WORST {s['worst']:7.2f}"
              f" | >20 m {s['over20']:4d}  >50 m {s['over50']:4d}")

    print("\nhonesty of each gate: does it decline when it should?")
    print("  %-6s %s" % ("", "  ".join(f"prior {lo:>3}-{hi:<3}" for lo, hi in
                                       ((0, 10), (10, 20), (20, 50), (50, 999)))))
    for m in MATCHERS:
        cells = []
        for lo, hi in ((0, 10), (10, 20), (20, 50), (50, 999)):
            s = [r for r in rows if lo <= r["prior_err"] < hi]
            acc = [r for r in s if r.get(m, {}).get("ok")]
            e = _stats([r[m]["err"] for r in acc])
            cells.append("%3.0f%% / %6s" % (
                100 * len(acc) / len(s) if s else 0,
                "%.1f" % e["worst"] if e else "-"))
        print(f"  {m:6s} " + "  ".join(cells) + "    (accept% / worst err)")

    print("\nwhat the aircraft would publish under each trust setting")
    best, best_key = None, None
    for trust in (["ncc", "xfeat"], ["xfeat", "ncc"], ["xfeat"], ["ncc"]):
        e = policy(rows, trust)
        s = _stats(e)
        if s is None:
            continue
        name = ",".join(trust)
        note = "  <- what the code did before 4 Nov" if trust == ["ncc", "xfeat"] else ""
        print(f"  trust={name:12s} fix {s['n']:4d}/{len(rows)} ({100*s['n']/len(rows):3.0f}%)"
              f" | p50 {s['p50']:6.2f}  p90 {s['p90']:7.2f}  WORST {s['worst']:7.2f}"
              f" | >20 m {s['over20']:4d}  >50 m {s['over50']:4d}{note}")
        # Ranked on the worst case first, then on how many fixes survive. A
        # median is not the question: every runaway this project has had came
        # from a confident fix that was badly wrong, not from a poor average.
        key = (-s["worst"], s["n"])
        if best is None or key > best_key:
            best, best_key = name, key
    print(f"\n  RECOMMENDED  map_trust = {best!r}")
    if best == "ncc,xfeat":
        print("  (which is the old behaviour - on this flight it cost nothing)")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="flight folder names, e.g. vps_00478")
    ap.add_argument("--step", type=int, default=4, help="take every Nth frame")
    ap.add_argument("--min-alt", type=float, default=58.0)
    ap.add_argument("--map", default="/data/DRONE/maps/active.tif")
    a = ap.parse_args()

    cache = MapCache(os.path.realpath(a.map))
    t0 = time.time()
    allrows = []
    for run in a.runs:
        rows = run_flight(run, cache, step=a.step, min_alt=a.min_alt)
        allrows += rows
        if len(a.runs) > 1:
            report(rows, run)
    report(allrows, "ALL FLIGHTS" if len(a.runs) > 1 else a.runs[0])
    print(f"\n({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
