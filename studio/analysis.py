#!/usr/bin/env python3
"""The automatic flight review: what an assistant would say about a log.

Every rule in here is one of this project's own measured findings, turned into
a check that runs without anyone asking for it. The thresholds are not taste -
each one comes from a flight in `flight_logs` and the number is quoted in the
rule's own text, so a verdict can always be traced back to the flight that
earned it.

THREE RULES OF MEASUREMENT ARE BUILT IN, because breaking them is how this
project has repeatedly fooled itself (`memory/dronevps-measure-fairly.md`):

  1. TRUTH IS `gps1_north_m` / `gps1_east_m` - GPS_RAW_INT, the receiver
     before the EKF. NEVER `gps_north_m`, which is GLOBAL_POSITION_INT, the
     EKF's fused output; with EK3_SRC1_POSXY = 3 and our GPS_INPUT injected as
     GPS2, that column is partly our own injection coming back. Scoring
     against it once moved a conclusion by a factor of four. If a log has no
     usable `gps1_*`, the position findings say so and are skipped rather than
     computed against the wrong column.

  2. THE gps1 COLUMNS CARRY CORRUPT ROWS, up to 23 km out. Anything over
     400 m from the origin is dropped before use, and the count is reported.

  3. BOTH SIDES OF A COMPARISON GET THE SAME SMOOTHING. Every velocity
     comparison here differentiates the receiver over a window and averages
     our own signal over the SAME window, and the window is named in the
     output. A 2.22x "scale error" once came from comparing a 30-sample mean
     against a raw sample.

  4. A HOLD IS SCORED BY HOW FAR THE AIRCRAFT RAN, in metres from the switch
     point against the raw receiver - not by the EKF's variance. SV is the EKF
     agreeing with itself, and anything that makes our output quieter
     improves it by construction. SV is still reported, under a heading that
     says what it does and does not answer.
"""
from __future__ import annotations

import math
import os

import numpy as np

import logs
from logs import series

# Altitude bands, chosen to match the bands this project's own reports use, so
# a number here can be compared with a number in a handoff without rebinning.
BANDS = [(0, 25), (25, 40), (40, 55), (55, 70), (70, 85), (85, 100), (100, 130)]

# Settings the node flew with. Read from vps_vio/config.py when it can be, so
# a review of an old log is not scored against today's values by accident.
DEFAULTS = dict(max_correction_m=0.5, fix_sigma_m=20.0, deadband_base=0.35,
                deadband_per_m=0.010, flow_vel_frames=40, fix_period_s=2.0)


def _p(a, q):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, q)) if a.size else float("nan")


def _frac(mask, within=None):
    m = np.asarray(mask, bool)
    if within is not None:
        w = np.asarray(within, bool)
        return float(m[w].mean()) if w.any() else float("nan")
    return float(m.mean()) if m.size else float("nan")


def _smooth(t, v, win_s):
    """Boxcar mean over win_s seconds, NaN-aware, on an irregular clock.

    The node's rows are not evenly spaced (dropped frames, slow iterations),
    so a fixed-sample window would be a different time window at different
    altitudes - which is exactly the unequal-smoothing trap. This one is in
    seconds on both sides of every comparison.
    """
    t = np.asarray(t, float)
    v = np.asarray(v, float)
    out = np.full(v.shape, np.nan)
    ok = np.isfinite(v)
    if not ok.any():
        return out
    i0 = 0
    cs_t = t
    for i in range(len(t)):
        lo, hi = t[i] - win_s / 2.0, t[i] + win_s / 2.0
        a = np.searchsorted(cs_t, lo)
        b = np.searchsorted(cs_t, hi)
        seg = v[a:b]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            out[i] = seg.mean()
    return out


def _deriv(t, x, win_s):
    """Central difference over win_s seconds: the receiver's own speed.

    Differencing adjacent samples of a 5 Hz receiver is noise; differencing
    over a named window is a measurement, and the same window then smooths
    our side of the comparison.
    """
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    out = np.full(x.shape, np.nan)
    for i in range(len(t)):
        a = np.searchsorted(t, t[i] - win_s / 2.0)
        b = min(len(t) - 1, np.searchsorted(t, t[i] + win_s / 2.0))
        if b <= a:
            continue
        dt = t[b] - t[a]
        if dt <= 0 or not (np.isfinite(x[a]) and np.isfinite(x[b])):
            continue
        out[i] = (x[b] - x[a]) / dt
    return out


def band_label(lo, hi):
    return f"{lo}-{hi} m"


# --------------------------------------------------------------------------- #
#  loading one flight
# --------------------------------------------------------------------------- #
def load(fc_path=None, node_csv=None):
    """Everything about one flight, from either log or both."""
    b = dict(fc=None, node=None, align=None, config=_read_config())
    if fc_path:
        arrays, meta = logs.read_bin(fc_path)
        b["fc"] = dict(path=fc_path, name=os.path.basename(fc_path),
                       arrays=arrays, meta=meta)
    if node_csv:
        c = logs.read_csv(node_csv)
        b["node"] = dict(path=node_csv, name=os.path.basename(os.path.dirname(node_csv))
                         or os.path.basename(node_csv), csv=c,
                         origin=logs.origin_of(c))
    if b["fc"] and b["node"]:
        b["align"] = logs.align(b["node"]["csv"], b["fc"]["arrays"])
    if b["node"]:
        b["derived"] = derive(b)
    b["vps_fc"] = _vps_intervals_fc(b)
    return b


def _vps_intervals_fc(b):
    """When the node was injecting, on the FLIGHT CONTROLLER's clock.

    Several of the flight controller's own numbers mean nothing averaged over
    a whole log: the EKF's variances are near zero while the aircraft flies on
    its receiver and only the VPS stretch is being asked about. With both logs
    aligned this gives those findings the right window; without an alignment
    they fall back to the whole log and say so.
    """
    if not (b.get("fc") and b.get("node") and b.get("align")
            and b["align"].get("trust")):
        return None
    d, lag = b["derived"], b["align"]["lag_s"]
    vps = d["vps"]
    if not vps.any():
        return None
    t = d["t"] + lag
    edges = np.diff(vps.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0] + 1)
    if vps[0]:
        starts = [0] + starts
    if vps[-1]:
        ends = ends + [len(vps) - 1]
    return [(float(t[a]), float(t[bb])) for a, bb in zip(starts, ends)
            if t[bb] - t[a] > 1.0]


def _in(t, intervals):
    t = np.asarray(t, float)
    if not intervals:
        return np.ones(t.shape, bool)
    m = np.zeros(t.shape, bool)
    for a, b_ in intervals:
        m |= (t >= a) & (t <= b_)
    return m


def _read_config():
    """The node's settings as they are on disk now, for the rules that need one.

    Parsed rather than imported: importing vps_vio.config pulls in the node's
    own dependencies, and a review tool must open a log on a machine where the
    camera stack is not installed.
    """
    out = dict(DEFAULTS)
    path = os.path.join(logs.ROOT, "vps_vio", "config.py")
    try:
        txt = open(path).read()
    except OSError:
        return out
    import re
    for key in list(out):
        m = re.search(rf"^\s*{key}\s*[:=]\s*([0-9.]+)", txt, re.M)
        if m:
            try:
                out[key] = float(m.group(1))
            except ValueError:
                pass
    return out


def derive(b):
    """Derived series every finding and every plot shares.

    Computed once, because several rules need the same masks and because a
    rule that recomputed its own truth mask would be free to use a different
    one - which is how two findings in the same report came to disagree.
    """
    c = b["node"]["csv"]
    num, txt = c["num"], c["txt"]
    t = num.get("t", np.array([]))
    n = len(t)
    d = dict(t=t, n=n)

    src = np.array(txt.get("src", ["?"] * n), dtype=object)
    d["src"] = src
    d["vps"] = src == "VPS"
    d["alt"] = num.get("alt_m", np.full(n, np.nan))

    # TRUTH. gps1_* is the raw receiver; gps_* is the EKF's fused output and is
    # NOT truth on a VPS flight. Rows beyond 400 m of the origin are the known
    # corruption in these columns.
    gn = num.get("gps1_north_m", np.full(n, np.nan))
    ge = num.get("gps1_east_m", np.full(n, np.nan))
    raw_ok = np.isfinite(gn) & np.isfinite(ge)
    sane = raw_ok & (np.abs(gn) < 400) & (np.abs(ge) < 400)
    d["truth_n"], d["truth_e"] = gn, ge
    d["truth_ok"] = sane
    d["truth_rows"] = int(sane.sum())
    d["truth_dropped"] = int((raw_ok & ~sane).sum())
    d["has_truth"] = bool(sane.sum() > 50)

    # POSITION ERROR against the receiver. north_m - gps1_north_m is
    # origin-invariant, so no masking of the handover step is needed - and
    # must not be applied, because subtracting the window's own start is
    # exactly the step the pilot is asking about.
    nm = num.get("north_m", np.full(n, np.nan))
    em = num.get("east_m", np.full(n, np.nan))
    d["ours_n"], d["ours_e"] = nm, em
    err = np.full(n, np.nan)
    m = sane & np.isfinite(nm) & np.isfinite(em)
    err[m] = np.hypot(nm[m] - gn[m], em[m] - ge[m])
    d["pos_err"] = err

    # VELOCITY, both sides smoothed over the same window.
    win = 1.0
    d["vel_win_s"] = win
    sent = np.hypot(num.get("vel_sent_vn", np.full(n, np.nan)),
                    num.get("vel_sent_ve", np.full(n, np.nan)))
    d["sent_speed"] = sent
    d["sent_speed_sm"] = _smooth(t, sent, win)
    tvn = _deriv(t, np.where(sane, gn, np.nan), win)
    tve = _deriv(t, np.where(sane, ge, np.nan), win)
    d["truth_speed"] = np.hypot(tvn, tve)
    d["truth_vn"], d["truth_ve"] = tvn, tve

    d["deadband"] = num.get("vel_deadband", np.full(n, np.nan))
    d["deadbanded"] = num.get("vel_deadbanded", np.zeros(n)) > 0.5
    d["vel_valid"] = num.get("vel_valid", np.zeros(n)) > 0.5
    d["vel_age"] = num.get("vel_age_s", np.full(n, np.nan))

    # Row-to-row interval: the node's own loop rate, which starves at height.
    dt = np.diff(t, prepend=t[0] if n else 0.0)
    dt[0] = np.nan
    d["dt"] = dt
    return d


# --------------------------------------------------------------------------- #
#  the findings
# --------------------------------------------------------------------------- #
def _f(id, level, title, headline, detail, rows=None, basis=""):
    return dict(id=id, level=level, title=title, headline=headline,
                detail=detail, rows=rows or [], basis=basis)


LEVEL_ORDER = {"alarm": 0, "warn": 1, "info": 2, "ok": 3}


def findings(b):
    out = []
    if b.get("node"):
        out += _node_findings(b)
    if b.get("fc"):
        out += _fc_findings(b)
    if b.get("node") and b.get("fc"):
        out += _combined_findings(b)
    out.sort(key=lambda f: (LEVEL_ORDER.get(f["level"], 9), f["id"]))
    return out


def _node_findings(b):
    d, c = b["derived"], b["node"]["csv"]
    num, cfg = c["num"], b["config"]
    t, n = d["t"], d["n"]
    vps = d["vps"]
    out = []

    # ---- how much of this log is even a VPS flight
    dur = float(t[-1] - t[0]) if n > 1 else 0.0
    vps_s = float(np.nansum(d["dt"][vps])) if vps.any() else 0.0
    out.append(_f("a_overview", "info", "What this log is",
                  f"{dur:.0f} s of rows, {vps_s:.0f} s of them on the VPS",
                  f"{n} rows at {n / max(dur, 1e-6):.1f} Hz. "
                  f"Altitude median {_p(d['alt'], 50):.1f} m, max "
                  f"{np.nanmax(d['alt']) if n else float('nan'):.1f} m. "
                  f"Source column: "
                  + ", ".join(f"{s} {int((d['src'] == s).sum())} rows"
                              for s in sorted(set(d["src"].tolist())))))

    # ---- is there independent truth at all
    if not d["has_truth"]:
        out.append(_f("b_truth", "warn", "No independent truth in this log",
                      f"{d['truth_rows']} usable gps1 rows",
                      "`gps1_north_m`/`gps1_east_m` (GPS_RAW_INT) is the only "
                      "column that is not downstream of our own injection. "
                      "Without it no position or velocity error can be "
                      "measured, and `gps_north_m` must NOT be substituted - "
                      "it is the EKF's fused output and on a VPS flight that "
                      "is partly what we sent. Those findings are skipped.",
                      basis="memory/dronevps-measure-fairly.md, 23 Oct"))
    else:
        out.append(_f("b_truth", "ok", "Truth is the raw receiver",
                      f"{d['truth_rows']} rows of gps1, "
                      f"{d['truth_dropped']} corrupt rows dropped",
                      "Scored against `gps1_*` (GPS_RAW_INT), never `gps_*` "
                      "(GLOBAL_POSITION_INT, the EKF's fused output). Rows "
                      "further than 400 m from the origin are the known "
                      "corruption in these columns and are dropped.",
                      basis="memory/dronevps-measure-fairly.md"))

    # ---- POSITION ERROR
    if d["has_truth"]:
        err = d["pos_err"]
        m = vps & np.isfinite(err)
        if m.sum() > 50:
            p50, p90 = _p(err[m], 50), _p(err[m], 90)
            rows = []
            for lo, hi in BANDS:
                bm = m & (d["alt"] >= lo) & (d["alt"] < hi)
                if bm.sum() > 20:
                    rows.append([band_label(lo, hi), f"{int(bm.sum())}",
                                 f"{_p(err[bm], 50):.2f} m",
                                 f"{_p(err[bm], 90):.2f} m"])
            # 20 m is the figure this project has called broken (flight 239
            # ended at 20.6 m at 110 m). Flight 245, the best so far, sits at
            # 10 m overall and 1.3 m low down, so an alarm at 10 would fire on
            # the good flight and mean nothing.
            lvl = "alarm" if p50 > 20 else "warn" if p50 > 5 else "ok"
            out.append(_f("c_position", lvl, "Position against the receiver",
                          f"p50 {p50:.2f} m, p90 {p90:.2f} m over "
                          f"{int(m.sum())} VPS rows",
                          "`north_m - gps1_north_m` on VPS rows only. This "
                          "difference is origin-invariant, so no masking of "
                          "the handover is applied - and none should be: "
                          "measuring a window's error from its own start "
                          "subtracts exactly the handover step. Flight 245 "
                          "read 0.44 m at 38 m and 1.42 m at 82 m.",
                          rows=[["band", "rows", "p50", "p90"]] + rows,
                          basis="flight 245, 25 Oct"))

    # ---- PHANTOM VELOCITY: what we send while the aircraft is standing still
    if d["has_truth"]:
        still = d["truth_speed"] < 0.5
        m = vps & still & np.isfinite(d["sent_speed_sm"])
        if m.sum() > 50:
            rows, worst, worst_band = [], 0.0, ""
            for lo, hi in BANDS:
                bm = m & (d["alt"] >= lo) & (d["alt"] < hi)
                if bm.sum() > 20:
                    med = _p(d["sent_speed_sm"][bm], 50)
                    p90 = _p(d["sent_speed_sm"][bm], 90)
                    rows.append([band_label(lo, hi), f"{int(bm.sum())}",
                                 f"{med:.2f} m/s", f"{p90:.2f} m/s"])
                    if med > worst:
                        worst, worst_band = med, band_label(lo, hi)
            lvl = "alarm" if worst > 1.5 else "warn" if worst > 0.6 else "ok"
            out.append(_f("d_phantom", lvl,
                          "Speed sent while the receiver says standing still",
                          f"worst band {worst_band}: {worst:.2f} m/s median",
                          f"Rows where the receiver's own speed is under "
                          f"0.5 m/s. Both sides over the same "
                          f"{d['vel_win_s']:.1f} s window - the receiver "
                          f"differentiated over it, our sent speed averaged "
                          f"over it. LOITER tilts to cancel motion that is "
                          f"not happening, which is what the pilot means by "
                          f"'jaise GPS par chalta hai waise nahi chala'. "
                          f"Flight 239 read 2.40 m/s at 70-85 m; flight 245 "
                          f"was far lower. A longer velocity baseline is "
                          f"ruled out - flight 240 swept N and the phantom is "
                          f"flat from N=40 on, so it is a correlated error "
                          f"averaging cannot reach.",
                          rows=[["band", "rows", "median sent", "p90"]] + rows,
                          basis="flights 239/240, 24 Oct"))

    # ---- DEADBAND CHATTER: the jerks
    sent = d["sent_speed"]
    m = vps & np.isfinite(sent)
    if m.sum() > 100:
        z = sent[m] <= 1e-9
        flips = int(np.count_nonzero(np.diff(z.astype(int)) != 0))
        per_s = flips / max(vps_s, 1e-6)
        zeroed = _frac(d["deadbanded"], vps)
        lvl = "alarm" if per_s > 0.4 else "warn" if per_s > 0.15 else "ok"
        out.append(_f("e_deadband", lvl, "Deadband chatter - the jerks",
                      f"{flips} zero/non-zero transitions in {vps_s:.0f} s "
                      f"= {per_s:.2f} per second; {zeroed * 100:.1f}% of rows zeroed",
                      f"The deadband is a HARD threshold: below "
                      f"{cfg['deadband_base']:.2f} + "
                      f"{cfg['deadband_per_m']:.3f} x altitude the flight "
                      f"controller is told exactly 0, above it the full "
                      f"value. Every transition is therefore a step input, "
                      f"and the aircraft answers each one with a tilt. "
                      f"Flight 245 measured 307 crossings in 634 s (0.48/s) "
                      f"with 39.9% of rows zeroed, and that is the flight the "
                      f"pilot described as holding but jerking. A soft "
                      f"deadband - shrink by the threshold instead of zeroing "
                      f"below it - is the known fix and is NOT written, "
                      f"because it changes a guard.",
                      basis="flight 245, 25 Oct"))

    # ---- MAP FIXES
    fix_d = num.get("fix_disagree_m", np.full(n, np.nan))
    attempts = np.isfinite(num.get("fix_lon", np.full(n, np.nan)))
    used = attempts & (num.get("fix_used", np.zeros(n)) > 0.5)
    n_att, n_used = int(attempts.sum()), int(used.sum())
    if n_att:
        rate = n_used / max(vps_s, 1e-6)
        clamp = float(b["config"]["max_correction_m"])
        asked = fix_d[used & np.isfinite(fix_d)]
        clamped = float(np.mean(asked > clamp)) if asked.size else float("nan")
        score = num.get("fix_score", np.full(n, np.nan))
        inl = num.get("fix_inliers", np.full(n, np.nan))
        rows = []
        for lo, hi in BANDS:
            bm = used & (d["alt"] >= lo) & (d["alt"] < hi)
            if bm.sum() > 3:
                im = bm & (inl > 0)
                rows.append([band_label(lo, hi), f"{int(bm.sum())}",
                             f"{_p(fix_d[bm], 50):.2f} m",
                             f"{_p(score[bm], 50):.3f}",
                             f"{_p(inl[im], 50):.0f}" if im.sum() else "-"])
        closing = n_used * clamp / max(vps_s, 1e-6)
        lvl = "warn" if (rate < 0.1 or clamped > 0.9) else "ok"
        out.append(_f("f_map", lvl, "Map fixes",
                      f"{n_att} attempts, {n_used} accepted = {rate:.3f} per "
                      f"second; they asked to move {_p(asked, 50):.2f} m p50, "
                      f"{clamped * 100:.0f}% of it over the {clamp:.2f} m cap, "
                      f"so the map can close at most {closing:.3f} m/s",
                      "`fix_used` is the gate's verdict, `fix_lon` only says "
                      "a matcher answered - on flight 245 that was 234 "
                      "answers and 159 accepted, and quoting the larger "
                      "number as the fix rate overstates it by half. "
                      "`fix_disagree_m` is how far the matcher's answer sat "
                      "from the node's own position: the matcher's error and "
                      "our drift together, not either alone. The fix RATE is "
                      "what sets the system's error - 0 fixes gave 38-68 m on "
                      "log 227 and 19 gave 4.50 m. Flight 245 ran 0.251/s "
                      "with the matcher at p50 5.64 m against truth, 2.89 m "
                      "at 60-80 m and 11.37 m above 95 m: height, not the "
                      "matcher, is the limit. When the cap binds on nearly "
                      "every fix the map is arriving at a fixed rate rather "
                      "than correcting what it sees: on flight 239 all 68 "
                      "fixes were clamped while each asked for a median 14.45 m, "
                      "and opening the cap was still not the fix, because the "
                      "matcher's own error was 11.74 m - the node would have "
                      "gone from 14.04 m to 11.74 m. Inliers are blank for the NCC "
                      "matcher, so the column is shown only where it is "
                      "non-zero.",
                      rows=[["band", "accepted", "p50 disagreement",
                             "score p50", "inliers p50"]] + rows,
                      basis="flight 245 / log 227"))
    else:
        out.append(_f("f_map", "warn", "Map fixes", "none in this log",
                      "No row carries a `fix_lon`. Either the flight never "
                      "cleared the matcher's minimum altitude, or the "
                      "take-off point is off the sheet - which means no map "
                      "fixes for the whole flight. Check the sheet coverage "
                      "in the Map tab against this log's origin."))

    # ---- FLOW HEALTH
    fd = num.get("flow_disagree_px", np.full(n, np.nan))
    if np.isfinite(fd).any():
        p50, p90 = _p(fd[vps], 50), _p(fd[vps], 90)
        tracks = _p(num.get("tracks", np.full(n, np.nan))[vps], 50)
        ndr = _p(num.get("n_dr", np.full(n, np.nan))[vps], 50)
        stale = _frac(d["vel_age"] > 0.6, vps)
        invalid = _frac(~d["vel_valid"], vps)
        lvl = "alarm" if p50 > 0.5 else "warn" if p50 > 0.2 else "ok"
        out.append(_f("g_flow", lvl, "Optical flow health",
                      f"flow_disagree_px p50 {p50:.3f}, p90 {p90:.3f}; "
                      f"{tracks:.0f} tracks; velocity stale > 0.6 s on "
                      f"{stale * 100:.1f}% of rows",
                      f"The two cameras' disagreement in pixels is the "
                      f"fastest read on whether derotation is working. "
                      f"Flight 163, before per-track derotation, read 2.32 px "
                      f"with the 0.5 px veto firing on 95% of frames; flight "
                      f"245 read 0.050 px with it firing on 2.2%. "
                      f"vel_valid false on {invalid * 100:.1f}% of rows, "
                      f"dead-reckoned frames median {ndr:.0f}. Note the veto "
                      f"does NOT catch the phantom: on flight 239's worst "
                      f"over-reports the cameras agreed to 0.04-0.45 px - "
                      f"both gave the same wrong answer.",
                      basis="flights 163/245"))

    # ---- LOOP HEALTH
    dt = d["dt"]
    if np.isfinite(dt).sum() > 100:
        slow = _frac(dt > 0.060, vps & np.isfinite(dt))
        rows = []
        for lo, hi in BANDS:
            bm = vps & np.isfinite(dt) & (d["alt"] >= lo) & (d["alt"] < hi)
            if bm.sum() > 20:
                rows.append([band_label(lo, hi), f"{int(bm.sum())}",
                             f"{_p(dt[bm] * 1000, 50):.0f} ms",
                             f"{_frac(dt[bm] > 0.060) * 100:.1f}%"])
        skipped = num.get("frames_skipped", np.full(n, np.nan))
        tot_skip = (float(np.nanmax(skipped)) if np.isfinite(skipped).any()
                    else float("nan"))
        lvl = "warn" if slow > 0.25 else "ok"
        out.append(_f("h_loop", lvl, "Loop rate and dropped frames",
                      f"{slow * 100:.1f}% of iterations slower than 60 ms; "
                      f"frames_skipped reached {tot_skip:.0f}",
                      "The loop starves at height because the matcher's "
                      "window radius is clip(0.60 x altitude, 40, 150) and "
                      "XFeat's Python-level work does not release the GIL: "
                      "locate() cost 0.098 s at 42 m and 0.319 s at 110 m on "
                      "flight 239, where 47.7% of iterations ran slow at "
                      "110 m against 3.6% at 37 m. Flight 240 brought that to "
                      "9.7%.",
                      rows=[["band", "rows", "median interval", "> 60 ms"]] + rows,
                      basis="flights 239/240"))

    # ---- ALTITUDE: the barometer against the lidar
    rng = num.get("rngfnd_m", np.full(n, np.nan))
    alt = d["alt"]
    # Both sensors must be off the ground. Below ~2 m the lidar is reading its
    # own mounting offset and the ratio is dominated by it: on flight 245 the
    # 8901 rows with alt under 3 m have a lidar median of 0.63 m, which turns
    # into a ratio of 2.86 that says nothing about the barometer.
    air = np.isfinite(rng) & np.isfinite(alt) & (rng > 2.0) & (alt > 3.0)
    if air.sum() > 50:
        rows, worst, worst_band = [], 1.0, ""
        for lo, hi in [(3, 15), (15, 30), (30, 60), (60, 90), (90, 130)]:
            bm = air & (alt >= lo) & (alt < hi)
            if bm.sum() > 20:
                r = _p(alt[bm] / rng[bm], 50)
                rows.append([band_label(lo, hi), f"{int(bm.sum())}",
                             f"{_p(rng[bm], 50):.1f} m", f"x{r:.2f}"])
                if abs(r - 1) > abs(worst - 1):
                    worst, worst_band = r, band_label(lo, hi)
        gnd = np.isfinite(rng) & np.isfinite(alt) & (alt < 3.0)
        gnd_txt = (f" On the {int(gnd.sum())} rows under 3 m the lidar reads "
                   f"{_p(rng[gnd], 50):.2f} m while the height in use reads "
                   f"{_p(alt[gnd], 50):.2f} m." if gnd.sum() > 20 else "")
        lvl = "alarm" if worst > 1.4 else "warn" if worst > 1.15 else "ok"
        out.append(_f("i_altitude", lvl, "Height: the barometer against the lidar",
                      f"worst ratio x{worst:.2f} at {worst_band}",
                      "Velocity is pixels x altitude / focal, so a height "
                      "that reads high scales the velocity up by the same "
                      "factor - 72% at 10 m on flight 245. That flight "
                      "measured 1.07 at 100 m, 1.17 at 40, 1.29 at 20 and "
                      "1.72 at 10, the same climbing and descending, so an "
                      "offset rather than hysteresis. Rows where either "
                      "sensor reads near the ground are excluded." + gnd_txt,
                      rows=[["band", "rows", "lidar p50", "height / lidar"]] + rows,
                      basis="flight 245, 25 Oct"))

    # ---- CIRCLING / TOILET BOWL
    src_xy = (d["truth_n"], d["truth_e"]) if d["has_truth"] else (d["ours_n"], d["ours_e"])
    ok = (d["truth_ok"] if d["has_truth"]
          else np.isfinite(d["ours_n"]) & np.isfinite(d["ours_e"]))
    m = vps & ok
    if m.sum() > 200:
        yaw = num.get("yaw_deg", np.full(n, np.nan))
        res = _circling(t[m], src_xy[0][m], src_xy[1][m], yaw[m])
        if res:
            lvl = "warn" if res["sweep_deg"] > 360 else "ok"
            out.append(_f("j_circling", lvl, "Circling (toilet bowl)",
                          f"bearing swept {res['sweep_deg']:.0f} deg in "
                          f"{res['window_s']:.0f} s while yaw moved "
                          f"{res['yaw_deg']:.1f} deg; radius "
                          f"{res['radius_m']:.1f} m, period "
                          f"{res['period_s']:.1f} s, one-way fraction "
                          f"{res['coherence']:.2f}",
                          "A bearing that sweeps while the heading does not "
                          "is circling, not turning. Flight 245 swept 754 deg "
                          "in 24 s at 35 m with yaw moving 9.6 deg. The "
                          "period to compare it against is 2*pi times the "
                          "velocity baseline's lag: at 40 frames that is "
                          "1.6 s of lag and 10.1 s of period, and 245 "
                          "measured 11.4 s.",
                          basis="flight 245, 25 Oct"))

    # ---- DECLARED SIGMA, reported and not acted on
    sig = num.get("sigma_m", np.full(n, np.nan))
    if d["has_truth"] and np.isfinite(sig).any():
        m = vps & np.isfinite(sig) & np.isfinite(d["pos_err"])
        if m.sum() > 50:
            out.append(_f("k_sigma", "info", "Declared accuracy against actual",
                          f"declared p50 {_p(sig[m], 50):.2f} m, actual error "
                          f"p50 {_p(d['pos_err'][m], 50):.2f} m",
                          "Reported, not scored. Declaring an honest, larger "
                          "position sigma was flown and made the hold WORSE, "
                          "because this aircraft has no other horizontal "
                          "position source: an EKF told its only source is "
                          "poor does not find a better one. The entry at "
                          "map_fuse.py:96 records it.",
                          basis="memory/dronevps-measure-fairly.md"))
    return out


def _circling(t, nn, ee, yaw):
    """The worst 30 s window for bearing sweep about its own centre."""
    if len(t) < 100:
        return None
    best = None
    win = 30.0
    i = 0
    while i < len(t):
        j = np.searchsorted(t, t[i] + win)
        if j - i < 50:
            i += 25
            continue
        n_, e_ = nn[i:j], ee[i:j]
        cn, ce = np.nanmean(n_), np.nanmean(e_)
        r = np.hypot(n_ - cn, e_ - ce)
        if np.nanmedian(r) < 2.5:
            i += 25
            continue
        ang = np.degrees(np.arctan2(e_ - ce, n_ - cn))
        step = (np.diff(ang) + 180) % 360 - 180
        sweep = float(np.nansum(np.abs(step)))
        net = abs(float(np.nansum(step)))
        # A toilet bowl goes round one way, so the net rotation is most of the
        # total. Receiver noise on a 1.5 m radius sweeps the bearing just as
        # far but cancels out, and scoring the total alone reports a circle
        # the aircraft never flew.
        coh = net / sweep if sweep > 1 else 0.0
        if coh < 0.5:
            i += 25
            continue
        if best is None or sweep > best["sweep_deg"]:
            y = yaw[i:j]
            yspan = (float(np.nanmax(y) - np.nanmin(y))
                     if np.isfinite(y).any() else float("nan"))
            period = (360.0 / (sweep / (t[j - 1] - t[i])) if sweep > 1 else float("nan"))
            best = dict(sweep_deg=sweep, window_s=float(t[j - 1] - t[i]),
                        yaw_deg=yspan, radius_m=float(np.nanmedian(r)),
                        period_s=period, t0=float(t[i]), coherence=coh)
        i += 25
    return best


def _fc_findings(b):
    arrays, meta = b["fc"]["arrays"], b["fc"]["meta"]
    parm = meta["params"]
    out = []
    # The EKF's variances and the commanded tilt are near nothing while the
    # aircraft flies on its receiver, so averaging them over a whole log
    # answers a question nobody asked. With both logs aligned they are scoped
    # to the stretch the node was actually injecting.
    win = b.get("vps_fc")
    where = (f"over the {sum(bb - a for a, bb in win):.0f} s the node was "
             f"injecting" if win else "over the whole log")

    # ---- MODE CHANGES, and which of them the aircraft made by itself
    ms = logs.modes(meta)
    auto = [m for m in ms if m["reason"] not in ("RC command", "GCS command",
                                                 "unknown")]
    drops = [m for m in ms if m["name"] == "ALT_HOLD"]
    rows = [["t", "mode", "why"]] + [[f"{m['t']:.1f} s", m["name"], m["reason"]]
                                     for m in ms]
    lvl = "alarm" if drops or auto else "ok"
    out.append(_f("m_modes", lvl, "Mode changes",
                  (f"{len(drops)} drop(s) to ALT_HOLD, "
                   f"{len(auto)} change(s) the aircraft made itself")
                  if (drops or auto) else
                  f"{len(ms)} change(s), all commanded",
                  "An ALT_HOLD the pilot did not ask for is the VPS being "
                  "dropped: the EKF's velocity innovation failsafe fires and "
                  "the aircraft gives up horizontal control. Every flight "
                  "before 245 fell to ALT_HOLD two to four times; 245 did it "
                  "zero times in 634 s. The reason code is the flight "
                  "controller's own.",
                  rows=rows, basis="flight 245"))

    # ---- WHAT THE FLIGHT CONTROLLER COMPLAINED ABOUT
    errs = logs.errors(meta)
    lanes = [e for e in errs if e["n"] == 24 and _in([e["t"]], win)[0]]
    rest = [e for e in errs if e["n"] != 24]
    if rest:
        # A glitch on the receiver matters because the receiver is our TRUTH,
        # not because the flight depends on it - nothing here is gated on
        # GNSS. So it is a warning; the navigation subsystems are the alarm.
        bad = [e for e in rest if e["serious"] and e["code"] != 0
               and e["n"] != 11]
        soft = [e for e in rest if e["n"] == 11 and e["code"] != 0]
        rows = [["t", "subsystem", "what"]] + [
            [f"{e['t']:.1f} s", e["subsys"], e["code_name"]] for e in rest[:40]]
        out.append(_f("n_errors",
                      "alarm" if bad else "warn" if soft else "info",
                      "Flight controller errors",
                      f"{len(rest)} entries"
                      + (f", {len(bad)} on a navigation subsystem" if bad else
                         f", {len(soft)} on the receiver (our truth)" if soft
                         else ", none on a navigation subsystem"),
                      "ERR entries are the flight controller's own verdict, "
                      "not ours. A 'pilot input' pair is the sticks leaving "
                      "and returning to centre and is normal; EKF check, GPS, "
                      "barometer and the failsafes are not. The ECode means "
                      "different things per subsystem, so it is named per "
                      "subsystem rather than through one generic table.",
                      rows=rows))
    if len(lanes) > 2:
        # ArduPilot logs an EKF primary change as subsystem 24 with the new
        # core number in the ECode - it is not an error. Frequent lane
        # changes mean the two cores disagree about the aircraft, which on
        # this airframe means they disagree about what we are injecting.
        cores = sorted({e["code"] for e in lanes})
        rows = [["t", "primary core"]] + [[f"{e['t']:.1f} s", str(e["code"])]
                                          for e in lanes[:30]]
        out.append(_f("n2_ekflane", "warn", "EKF primary core changes",
                      f"{len(lanes)} lane changes between cores "
                      + "/".join(str(c) for c in cores) + f" - {where}",
                      "The EKF runs two cores and promotes whichever it "
                      "trusts; a change is logged as subsystem 24 with the "
                      "new core in the ECode, which is NOT an error code. "
                      "Changes this often mean the cores disagree, and on "
                      "this aircraft the thing they can disagree about is "
                      "the GPS2 we inject. Worth seeing beside the velocity "
                      "variance: a lane change at the same moment as an "
                      "innovation spike is one event, not two.",
                      rows=rows))

    # ---- EKF VARIANCES: a failsafe question, not a hold score
    t_sv, sv = series(arrays, "XKF4", "SV", where=("C", 0))
    t_sp, sp = series(arrays, "XKF4", "SP", where=("C", 0))
    thr = parm.get("FS_EKF_THRESH", 1.0)
    sv = sv[_in(t_sv, win)]
    sp = sp[_in(t_sp, win)]
    if len(sv):
        over = float(np.mean(sv > thr))
        lvl = "alarm" if over > 0.2 else "warn" if over > 0.02 else "ok"
        out.append(_f("o_ekf", lvl, "EKF innovation variances",
                      f"SV p50 {_p(sv, 50):.3f}, max {np.nanmax(sv):.2f}; "
                      f"{over * 100:.1f}% of samples over FS_EKF_THRESH "
                      f"{thr:.2f}, SP p50 {_p(sp, 50):.3f} - {where}",
                      "This answers ONE question - will the EKF trip its "
                      "failsafe - and it is not a score for the hold. SV is "
                      "the EKF agreeing with itself, so anything that makes "
                      "our output quieter (a longer baseline, a wider "
                      "deadband, a larger declared sigma) improves it by "
                      "construction. Two changes once improved SV while "
                      "making the aircraft hold twelve times worse. The hold "
                      "is scored in 'Distance run from the handover'.",
                      basis="memory/dronevps-measure-fairly.md, 22 Oct"))

    # ---- VIBRATION
    t_v, vx = series(arrays, "VIBE", "VibeX", where=("IMU", 0))
    _, vy = series(arrays, "VIBE", "VibeY", where=("IMU", 0))
    _, vz = series(arrays, "VIBE", "VibeZ", where=("IMU", 0))
    _, clip = series(arrays, "VIBE", "Clip", where=("IMU", 0))
    if len(vx):
        clips = int(np.nanmax(clip) - np.nanmin(clip)) if len(clip) else 0
        worst = max(_p(vx, 99), _p(vy, 99), _p(vz, 99))
        lvl = "alarm" if clips or worst > 30 else "warn" if worst > 15 else "ok"
        out.append(_f("p_vibe", lvl, "Vibration",
                      f"p50 X {_p(vx, 50):.1f} Y {_p(vy, 50):.1f} "
                      f"Z {_p(vz, 50):.1f}, p99 {worst:.1f} m/s2, "
                      f"{clips} clipping events",
                      "At rest on 20 Oct this airframe read 0.8 / 1.6 / 2.3 "
                      "m/s2. Vibration was measured and RULED OUT as the "
                      "cause of the phantom velocity on flight 239 (p99 "
                      "4.4 m/s2, no clipping), so a clean reading here does "
                      "not clear the flow - it only removes one suspect.",
                      basis="flight 239, 24 Oct"))

    # ---- COMMANDED TILT AND RATE
    t_att = arrays.get("ATT.t", np.array([]))
    droll = arrays.get("ATT.DesRoll", np.array([]))
    dpitch = arrays.get("ATT.DesPitch", np.array([]))
    if len(t_att) and len(droll) == len(t_att) and len(dpitch) == len(t_att):
        keep = _in(t_att, win) & np.isfinite(droll) & np.isfinite(dpitch)
        droll, dpitch = droll[keep], dpitch[keep]
    else:
        droll = dpitch = np.array([])
    amax = parm.get("ATC_ANGLE_MAX", 0.0) / 100.0 if parm.get("ATC_ANGLE_MAX", 0) > 100 \
        else parm.get("ATC_ANGLE_MAX", 15.0)
    if len(droll):
        tilt = np.hypot(droll, dpitch)
        sat = float(np.mean(tilt > 0.95 * amax)) if amax else float("nan")
        t_r, rr = series(arrays, "RATE", "R")
        t_p, pr = series(arrays, "RATE", "P")
        rr, pr = rr[_in(t_r, win)], pr[_in(t_p, win)]
        rate_max = max(_p(np.abs(rr), 99.9) if len(rr) else 0,
                       _p(np.abs(pr), 99.9) if len(pr) else 0)
        lvl = "warn" if sat > 0.02 else "ok"
        out.append(_f("q_tilt", lvl, "Commanded tilt and body rates",
                      f"commanded tilt p99 {_p(tilt, 99):.1f} deg against "
                      f"ATC_ANGLE_MAX {amax:.0f}; saturated on "
                      f"{sat * 100:.1f}% of samples; body rate p99.9 "
                      f"{rate_max:.1f} deg/s - {where}",
                      "Saturated tilt is the aircraft being asked for more "
                      "than it is allowed to give, which is what a step in "
                      "the commanded velocity produces. On flight 245 the "
                      "commanded tilt saturated at 15 deg with rates to "
                      "56.9 deg/s, and the deadband's zero crossings are "
                      "where those steps come from.",
                      basis="flight 245"))

    # ---- THE RECEIVER AND OUR INJECTION
    t_g0, st0 = series(arrays, "GPS", "Status", where=("I", 0))
    t_g1, st1 = series(arrays, "GPS", "Status", where=("I", 1))
    _, sats = series(arrays, "GPS", "NSats", where=("I", 0))
    _, hdop = series(arrays, "GPS", "HDop", where=("I", 0))
    if len(st0):
        out.append(_f("r_gps", "ok" if len(st1) else "warn",
                      "Receiver and injected GPS",
                      f"GPS1 (real) status p50 {_p(st0, 50):.0f}, "
                      f"{_p(sats, 50):.0f} satellites, HDop {_p(hdop, 50):.2f}; "
                      f"GPS2 (our GPS_INPUT) "
                      + (f"{len(st1)} records, status p50 {_p(st1, 50):.0f}"
                         if len(st1) else "ABSENT from this log"),
                      "The node injects as GPS2 and the receiver stays GPS1. "
                      "GPS1 is kept for truth only - EK3_SRC1_POSXY is "
                      f"{parm.get('EK3_SRC1_POSXY', float('nan')):.0f} and "
                      "nothing is gated on it. If GPS2 is absent the node was "
                      "not injecting at all for this log."))

    # ---- THE NODE'S OWN VOICE, as the flight controller heard it
    nm = logs.node_messages(meta)
    if nm:
        import re as _re
        kinds = {}
        for m in nm:
            # Every number replaced by #, so the node's periodic status line
            # ("t401 s6m f0 a1 v0.1 GPS1") collapses to one row instead of
            # one row per distinct track count.
            key = _re.sub(r"[-+]?\d+(\.\d+)?", "#", m["text"]).strip()
            kinds[key] = kinds.get(key, 0) + 1
        rows = [["message", "count"]] + [[k, str(v)] for k, v in
                                         sorted(kinds.items(), key=lambda kv: -kv[1])[:25]]
        out.append(_f("s_nodemsg", "info", "What the node told the aircraft",
                      f"{len(nm)} status lines from the Jetson",
                      "The node sends its state to the flight controller as "
                      "MAVLink MSGs, so they are timestamped on the FC's own "
                      "clock. This is the bridge between the two logs when "
                      "the altitude correlation fails, and the only record of "
                      "TRACK LOST, VEL stale and source changes as the "
                      "aircraft saw them.", rows=rows))

    # ---- ORIGIN SANITY
    lat_p, lon_p = parm.get("SCR_USER1"), parm.get("SCR_USER2")
    t_lat, lat0 = series(arrays, "GPS", "Lat", where=("I", 0))
    _, lon0 = series(arrays, "GPS", "Lng", where=("I", 0))
    if lat_p and lon_p and len(lat0):
        la, lo = _p(lat0, 50), _p(lon0, 50)
        dn = (la - lat_p) * 111_320.0
        de = (lo - lon_p) * 111_320.0 * math.cos(math.radians(la))
        dist = math.hypot(dn, de)
        lvl = "alarm" if dist > 200 else "warn" if dist > 30 else "ok"
        out.append(_f("t_origin", lvl, "The typed origin against where it flew",
                      f"SCR_USER1/2 = {lat_p:.6f}, {lon_p:.6f} is "
                      f"{dist:.0f} m from the receiver's median position",
                      "SCR_USER1/2 is the origin the node is told. On 20 Oct "
                      "it was 150 m from the take-off point; a sheet that "
                      "does not cover the real point means no map fixes for "
                      "the whole flight. Preflight checks this at 30 m.",
                      basis="vps_vio/preflight.py"))
    return out


def _combined_findings(b):
    """Findings that need both logs on one clock."""
    out = []
    al = b["align"]
    if not al:
        return out
    lvl = "ok" if al.get("trust") else "warn"
    out.append(_f("u_align", lvl, "The two logs on one clock",
                  f"node t + {al['lag_s']:.0f} s = flight controller t, "
                  f"correlation {al['corr']:.4f} over {al['overlap_s']:.0f} s",
                  "Recovered by correlating the two altitude traces at 1 Hz, "
                  "because nothing in either file states the offset. Below "
                  "0.9 correlation the combined plots are not trustworthy and "
                  "every single-log finding above still is."))
    if not al.get("trust"):
        return out

    d, arrays = b["derived"], b["fc"]["arrays"]
    lag = al["lag_s"]
    num = b["node"]["csv"]["num"]
    n = d["n"]

    # ---- THE SWITCH AND WHAT WE DID WITH IT
    t_rc, c6 = series(arrays, "RCIN", "C6")
    if len(t_rc):
        want = np.where(c6 < 1200, "GPS", np.where(c6 < 1700, "centre", "VPS"))
        node_t_on_fc = d["t"] + lag
        asked = np.array([want[min(len(want) - 1, np.searchsorted(t_rc, tt))]
                          for tt in node_t_on_fc], dtype=object)
        disagree = (asked == "GPS") & (d["src"] == "VPS")
        bad_s = float(np.nansum(d["dt"][disagree]))
        lvl = "alarm" if bad_s > 2 else "ok"
        out.append(_f("v_switch", lvl, "Switch against what the node did",
                      f"{bad_s:.1f} s of rows where channel 6 asked for GPS "
                      f"and the node was still on the VPS",
                      "Switch means switch: when the pilot flips to GPS the "
                      "Jetson must stop injecting immediately and completely. "
                      "On 23 Oct the aircraft landed on the VPS while the "
                      "switch was asking for GPS, because a cached belief in "
                      "gps_rc_switch ate the pilot's command. Channel 6 "
                      "below 1200 is GPS, above 1700 is VPS, in between is "
                      "whatever was last set.",
                      basis="memory/dronevps-switch-disagreement.md"))

    # ---- DISTANCE RUN FROM EACH HANDOVER: the project's own hold metric
    if d["has_truth"]:
        vps = d["vps"]
        edges = np.where(np.diff(vps.astype(int)) == 1)[0] + 1
        rows = [["handover", "altitude", "30 s", "60 s", "120 s", "drift"]]
        worst = 0.0
        for i in edges:
            t0 = d["t"][i]
            cols = []
            for horizon in (30.0, 60.0, 120.0):
                j = np.searchsorted(d["t"], t0 + horizon)
                seg = slice(i, min(j, n))
                m = d["truth_ok"][seg] & vps[seg]
                if m.sum() < 10:
                    cols.append("-")
                    continue
                nn = d["truth_n"][seg][m]
                ee = d["truth_e"][seg][m]
                ran = float(np.hypot(nn - nn[0], ee - ee[0]).max())
                cols.append(f"{ran:.1f} m")
                worst = max(worst, ran if horizon == 60.0 else 0.0)
            j60 = np.searchsorted(d["t"], t0 + 60.0)
            dur = max(1e-6, d["t"][min(j60, n - 1)] - t0)
            seg = slice(i, min(j60, n))
            m = d["truth_ok"][seg]
            drift = (float(np.hypot(*(d["truth_n"][seg][m][-1] - d["truth_n"][seg][m][0],
                                      d["truth_e"][seg][m][-1] - d["truth_e"][seg][m][0])) / dur)
                     if m.sum() > 10 else float("nan"))
            rows.append([f"{t0:.0f} s", f"{d['alt'][i]:.0f} m", *cols,
                         f"{drift:.3f} m/s"])
        if len(rows) > 1:
            lvl = "alarm" if worst > 10 else "warn" if worst > 3 else "ok"
            out.append(_f("w_handover", lvl, "Distance run from the handover",
                          f"{len(edges)} handover(s); worst 60 s run "
                          f"{worst:.1f} m",
                          "THE metric for a hold: how far the aircraft "
                          "actually went from the point where the VPS took "
                          "over, measured against the raw receiver, plus the "
                          "drift rate. Log 228 ran 0.9 m in 23 s at 31 m; the "
                          "next flight ran 11.1 m in 24 s at the same height "
                          "while its EKF variance looked five times better. "
                          "Measured from the receiver, not from our own "
                          "position, and not from the window's own start.",
                          rows=rows,
                          basis="memory/dronevps-measure-fairly.md, 22 Oct"))
    return out


# --------------------------------------------------------------------------- #
#  the track, for drawing on the satellite sheet
# --------------------------------------------------------------------------- #
def track(b, stride=5):
    """lon/lat of what we believed, what the receiver said, and every fix."""
    if not b.get("node"):
        return dict(ours=[], truth=[], fixes=[], origin=None)
    d, c = b["derived"], b["node"]["csv"]
    num = c["num"]
    origin = b["node"]["origin"]
    out = dict(origin=list(origin) if origin else None, ours=[], truth=[],
               fixes=[], vps_segments=[])
    lat, lon = num.get("lat"), num.get("lon")
    n = d["n"]
    if lat is not None and lon is not None:
        m = np.isfinite(lat) & np.isfinite(lon) & (lat != 0)
        idx = np.where(m)[0][::stride]
        out["ours"] = [[float(lon[i]), float(lat[i]), float(d["alt"][i]),
                        float(d["t"][i]), bool(d["vps"][i])] for i in idx]
    if origin and d["has_truth"]:
        olon, olat = origin
        mlon = 111_320.0 * math.cos(math.radians(olat))
        idx = np.where(d["truth_ok"])[0][::stride]
        out["truth"] = [[float(olon + d["truth_e"][i] / mlon),
                         float(olat + d["truth_n"][i] / 111_320.0),
                         float(d["alt"][i]), float(d["t"][i])] for i in idx]
    fl, fa = num.get("fix_lon"), num.get("fix_lat")
    if fl is not None and fa is not None:
        m = np.isfinite(fl) & np.isfinite(fa)
        dis = num.get("fix_disagree_m", np.full(n, np.nan))
        out["fixes"] = [[float(fl[i]), float(fa[i]), float(dis[i]),
                         float(d["alt"][i]), float(d["t"][i])]
                        for i in np.where(m)[0]]
    return out


def summary(b):
    """The headline numbers, for the strip across the top of the tab."""
    s = {}
    if b.get("node"):
        d = b["derived"]
        t = d["t"]
        s["node"] = b["node"]["name"]
        s["rows"] = d["n"]
        s["duration_s"] = float(t[-1] - t[0]) if d["n"] > 1 else 0.0
        s["vps_s"] = float(np.nansum(d["dt"][d["vps"]]))
        s["alt_max"] = float(np.nanmax(d["alt"])) if d["n"] else None
        s["alt_med"] = _p(d["alt"], 50)
        if d["has_truth"]:
            m = d["vps"] & np.isfinite(d["pos_err"])
            s["pos_p50"] = _p(d["pos_err"][m], 50) if m.sum() else None
        fl = b["node"]["csv"]["num"].get("fix_lon")
        s["fixes"] = int(np.isfinite(fl).sum()) if fl is not None else 0
    if b.get("fc"):
        meta = b["fc"]["meta"]
        s["fc"] = b["fc"]["name"]
        ms = logs.modes(meta)
        s["modes"] = [m["name"] for m in ms]
        s["althold"] = sum(1 for m in ms if m["name"] == "ALT_HOLD")
        s["fc_messages"] = len(meta["records"].get("MSG", []))
    if b.get("align"):
        s["align"] = b["align"]
    return s
