#!/usr/bin/env python3
"""The thread test, analysed by itself when a flight ends.

A 10 m thread hangs from the landing gear with a small stone on the end. At
the height where the stone just lifts off the ground, the aircraft is exactly
10.00 m up - measured by the thread, not by a sensor. That is the altitude
reference this aircraft lost when the lidar came off: alt_m is now the raw
barometer with no correction (alt_bias_m reads 0.00 on every flight since,
and rngfnd_m is empty), and the altitude it reports multiplies four things -
the flow's velocity, the matcher's query scale, the search radius and the
tilt correction.

Flight plan this expects, and it finds the phases itself:

    arm, take off to 10 m by the thread      ->  hold 10 s
    loiter, move 10-20 m forward/back/sideways
    climb                                    ->  come back down
    10 m by the thread again                 ->  hold 10-20 s
    land

Reports: what the barometer called 10.00 m, whether that offset drifted
across the flight, what the flow claimed while the aircraft was still, and
what the flow's scale looks like with and without the altitude correction.

Run by drone-thread-test.timer. Writes THREAD_REPORT.txt beside the log and
prints the same to the journal.
"""
import csv, glob, math, os, sys, time
import numpy as np

VPS = "/home/abc/NVME/DRONE/flight_logs/vps"
THREAD_M = 10.00          # the thread's length, metres
HOLD_LO, HOLD_HI = 5.0, 18.0     # altitudes that could be the thread hold
STILL_M = 1.5             # M9N movement over a window that still counts as held
MIN_HOLD_S = 6.0

# A thread report only means something if a thread was actually tied on, and
# no signal in the log says whether it was. So the pilot arms it: this marker
# file is touched before a thread flight and the timer ignores every run whose
# CSV is older than it. Without the marker nothing is analysed at all. The
# first version had no marker, swept the whole back-catalogue, and wrote
# "+27% off" onto a 3 November flight that never had a thread.
SINCE = "/home/abc/NVME/DRONE/flight_logs/vps/.thread_test_from"


def cols(r):
    F = lambda x, k: (float(x[k]) if x.get(k) not in (None, "") else np.nan)
    g = lambda k: np.array([F(x, k) for x in r])
    d = {k: g(k) for k in ("t", "alt_m", "step_dn", "step_de", "step_dt_s",
                           "gps1_north_m", "gps1_east_m", "north_m", "east_m",
                           "tracks", "gyro_d_roll", "gyro_d_pitch",
                           "frame_age_s", "sigma_m")}
    d["used"] = np.array([x.get("fix_used") == "1" for x in r])
    d["src"] = [x.get("fix_source", "") for x in r]
    return d


def segments(t, mask, min_s):
    """Contiguous runs of mask lasting at least min_s seconds."""
    out = []
    i = 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(mask) and mask[j + 1]:
            j += 1
        if t[j] - t[i] >= min_s:
            out.append((i, j))
        i = j + 1
    return out


def analyse(run, path, w):
    r = list(csv.DictReader(open(path)))
    if len(r) < 200:
        w("  too few rows (%d)" % len(r))
        return
    d = cols(r)
    t, alt = d["t"], d["alt_m"]
    gn, ge = d["gps1_north_m"], d["gps1_east_m"]
    ok = (np.isfinite(gn) & np.isfinite(ge) & (np.abs(gn) < 400)
          & (np.abs(ge) < 400))
    w("run %s   %.0f s   alt max %.1f m   rows %d" % (
        run, t[-1] - t[0], np.nanmax(alt), len(r)))

    # ---- still, at a height the thread could be holding
    still = np.zeros(len(t), bool)
    step = max(1, int(0.5 / max(np.nanmedian(np.diff(t)), 1e-3)))
    for i in range(0, len(t), step):
        j = min(len(t) - 1, i + int(3.0 / max(np.nanmedian(np.diff(t)), 1e-3)))
        if not (HOLD_LO <= np.nanmedian(alt[i:j + 1]) <= HOLD_HI):
            continue
        m = ok[i:j + 1]
        if m.sum() < 10:
            continue
        a, b = np.where(m)[0][[0, -1]]
        if math.hypot(gn[i + b] - gn[i + a], ge[i + b] - ge[i + a]) <= STILL_M:
            still[i:j + 1] = True
    holds = segments(t, still, MIN_HOLD_S)
    w("")
    if not holds:
        w("  NO THREAD HOLD FOUND between %.0f and %.0f m." % (HOLD_LO, HOLD_HI))
        w("  (needs %.0f s still, within %.1f m by the M9N)" % (MIN_HOLD_S, STILL_M))
    else:
        w("  ALTITUDE, against the thread's %.2f m" % THREAD_M)
        w("  (ye section SIRF tab maayne rakhta hai jab dhaga sach me laga ho")
        w("   aur pathar bas uth raha ho. Warna ye bas us unchai ko dohra raha")
        w("   hai jahan drone khada tha.)")
        offs = []
        for n, (i, j) in enumerate(holds, 1):
            a = np.nanmedian(alt[i:j + 1])
            offs.append(a - THREAD_M)
            res = math.hypot(np.nansum(d["step_dn"][i:j + 1]),
                             np.nansum(d["step_de"][i:j + 1]))
            span = t[j] - t[i]
            w("    hold %d  t=%6.1f-%6.1f (%4.1f s) | alt_m p50 %6.2f m"
              "  -> baro %+6.2f m off  (%+5.1f%%)" % (
                  n, t[i], t[j], span, a, a - THREAD_M,
                  100 * (a - THREAD_M) / THREAD_M))
            w("            flow ne khada drone par %5.2f m kaha  = %.3f m/s"
              "  | tracks %4.0f" % (res, res / max(span, .1),
                                    np.nanmedian(d["tracks"][i:j + 1])))
        if len(offs) >= 2:
            w("    offset ka drift pehli se aakhri hold tak: %+.2f m" %
              (offs[-1] - offs[0]))
        OFF = float(np.median(offs))
        w("")
        w("  => barometer %+.2f m %s padh raha hai. 100 m par wo %+.1f%% velocity"
          % (OFF, "zyada" if OFF > 0 else "kam", 100 * OFF / 100.0))
        w("     aur matcher ka query bhi usi %+.1f%% se galat size ka banta hai."
          % (100 * OFF / 100.0))

        # ---- flow scale on the low moves, with and without the correction
        w("")
        w("  FLOW KA SCALE (10-20 m ki moves, M9N ke khilaf)")
        rows = []
        for lo in np.arange(t[0], t[-1], 8.0):
            m = (t >= lo) & (t < lo + 8.0)
            mo = m & ok
            if m.sum() < 60 or mo.sum() < 25:
                continue
            A = np.nanmedian(alt[m])
            if not (HOLD_LO <= A <= 40.0):
                continue
            a, b = np.where(mo)[0][[0, -1]]
            truth = math.hypot(gn[b] - gn[a], ge[b] - ge[a])
            if truth < 5.0:
                continue
            flow = math.hypot(np.nansum(d["step_dn"][m]), np.nansum(d["step_de"][m]))
            rows.append((A, flow, truth))
        if len(rows) < 3:
            w("    moves nahi mili (8 s me 5 m se zyada chalna chahiye)")
        else:
            R = np.array(rows)
            sc = R[:, 1] / R[:, 2]
            w("    n=%d windows | scale p50 %.3f  (p10 %.3f  p90 %.3f)" % (
                len(R), np.median(sc), np.percentile(sc, 10),
                np.percentile(sc, 90)))
            corr = sc * R[:, 0] / np.maximum(R[:, 0] - OFF, 1e-3)
            w("    altitude theek karne ke baad: scale p50 %.3f" % np.median(corr))
            w("    (M9N ka apna noise in windows par %.1f m ka hai, to ye"
              " indicative hai)" % 1.5)

    # ---- the climb, which inherits the same offset
    hi = alt > 30.0
    if hi.sum() > 50:
        w("")
        w("  UNCHAI WALA HISSA: alt_m max %.1f m" % np.nanmax(alt))
        if holds:
            w("    offset hata kar asli unchai lagbhag %.1f m" % (
                np.nanmax(alt) - float(np.median(offs))))
        w("    us dauraan fix %d | tracks p50 %4.0f | frame_age p50 %.4f" % (
            int(d["used"][hi].sum()), np.nanmedian(d["tracks"][hi]),
            np.nanmedian(d["frame_age_s"][hi])))

    # ---- tilt, free of any reference: rotation with the aircraft still
    if holds:
        w("")
        w("  DEROTATION (hold ke dauraan - camera ghooma, drone nahi hila)")
        for n, (i, j) in enumerate(holds, 1):
            rot = np.nansum(np.hypot(d["gyro_d_roll"][i:j + 1],
                                     d["gyro_d_pitch"][i:j + 1]))
            res = math.hypot(np.nansum(d["step_dn"][i:j + 1]),
                             np.nansum(d["step_de"][i:j + 1]))
            A = np.nanmedian(alt[i:j + 1])
            px = (res / max(A, 1e-3)) * 795.0
            w("    hold %d | rotation %6.2f deg | residual %5.2f m = %5.2f px"
              " | %.3f px per deg" % (n, rot, res, px, px / max(rot, 1e-6)))
        w("    (px per deg ko 100 m par guna karo: 1 px = 0.126 m/frame)")


def main():
    force = sys.argv[1] if len(sys.argv) > 1 else None
    since = os.path.getmtime(SINCE) if os.path.exists(SINCE) else None
    if since is None and not force:
        print("thread test not armed (touch %s before the flight)" % SINCE)
        return
    runs = []
    for p in sorted(glob.glob(os.path.join(VPS, "vps_*"))):
        run = os.path.basename(p)
        csvp = os.path.join(p, run + ".csv")
        if not os.path.exists(csvp):
            continue
        if force and run != force:
            continue
        done = os.path.join(p, "THREAD_REPORT.txt")
        if not force:
            if os.path.exists(done):
                continue
            if time.time() - os.path.getmtime(csvp) < 60:
                continue          # still flying
            if os.path.getmtime(csvp) < since:
                continue          # flown before the pilot armed the test
        runs.append((run, p, csvp))
    if not runs:
        return
    for run, folder, csvp in runs[-3:]:
        lines = []
        w = lambda s="": (lines.append(s), print(s, flush=True))
        try:
            quick = list(csv.DictReader(open(csvp)))[:4000]
            a = [float(x["alt_m"]) for x in quick
                 if x.get("alt_m") not in (None, "")]
            if not a or max(a) < 5.0:
                open(os.path.join(folder, "THREAD_REPORT.txt"), "w").write(
                    "no flight (alt max %.1f m)\n" % (max(a) if a else 0))
                continue
        except Exception as exc:
            print("skip %s: %s" % (run, exc))
            continue
        w("=" * 72)
        w("THREAD TEST  —  analysed %s" % time.strftime("%Y-%m-%d %H:%M"))
        w("run %s   flown %s" % (
            run, time.strftime("%Y-%m-%d %H:%M",
                               time.localtime(os.path.getmtime(csvp)))))
        w("=" * 72)
        try:
            analyse(run, csvp, w)
        except Exception as exc:
            import traceback
            w("  FAILED: %s" % exc)
            w(traceback.format_exc())
        w("=" * 72)
        open(os.path.join(folder, "THREAD_REPORT.txt"), "w").write(
            "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
