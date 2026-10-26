#!/usr/bin/env python3
"""The panels. One function per figure, rendered server-side as a PNG.

Server-side matplotlib rather than a charting library in the browser, for two
reasons: the Jetson has no network in the field and a CDN would leave the page
blank, and a flight is 26 000 rows per log which is slow to ship and slower to
draw in canvas. The browser asks for a panel by name and gets an image.

Every panel that compares our signal against the receiver smooths both over
the same window, and says so in its own title. That is not decoration - this
project has twice published a conclusion that came from comparing a smoothed
series against a raw one.
"""
from __future__ import annotations

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analysis
from logs import series

BG = "#0f1419"
FG = "#c9d4e0"
GRID = "#243040"
ACC = "#4da3ff"          # ours
TRUTH = "#ffb34d"        # the receiver
WARN = "#ff6b6b"
OK = "#4ade80"
MUTED = "#6b7a8d"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "axes.edgecolor": GRID,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "figure.autolayout": True, "lines.linewidth": 1.1,
})


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return buf.getvalue()


def _shade_vps(ax, d):
    """Mark the stretches the node was injecting, on the node's own clock."""
    vps = d["vps"]
    t = d["t"]
    edges = np.diff(vps.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0] + 1)
    if vps.size and vps[0]:
        starts = [0] + starts
    if vps.size and vps[-1]:
        ends = ends + [len(vps) - 1]
    for a, b in zip(starts, ends):
        ax.axvspan(t[a], t[b], color="#1b3a5c", alpha=0.45, lw=0, zorder=0)


def _grid(ax):
    ax.grid(True, alpha=0.25, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


# --------------------------------------------------------------------------- #
def overview(b):
    d = b["derived"]
    num = b["node"]["csv"]["num"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                             gridspec_kw=dict(height_ratios=[2, 1]))
    ax = axes[0]
    _shade_vps(ax, d)
    ax.plot(d["t"], d["alt"], color=ACC, label="height in use (alt_m)")
    rng = num.get("rngfnd_m")
    if rng is not None:
        m = np.isfinite(rng) & (rng > 0.3)
        ax.plot(d["t"][m], rng[m], color=OK, lw=0.8, alpha=0.8, label="lidar")
    if b.get("fc") and b.get("align") and b["align"].get("trust"):
        ft, fa = series(b["fc"]["arrays"], "CTUN", "Alt")
        ax.plot(ft - b["align"]["lag_s"], fa, color=TRUTH, lw=0.8, alpha=0.9,
                label="flight controller (CTUN.Alt)")
    ax.set_ylabel("m")
    ax.set_title("Height, and when the VPS was driving (shaded)")
    ax.legend(loc="upper right", framealpha=0.2, fontsize=8)
    _grid(ax)

    ax = axes[1]
    _shade_vps(ax, d)
    tr = num.get("tracks")
    if tr is not None:
        ax.plot(d["t"], tr, color=MUTED, lw=0.8, label="tracks")
    ax.set_ylabel("tracks")
    ax.set_xlabel("node time (s)")
    ax2 = ax.twinx()
    fd = num.get("flow_disagree_px")
    if fd is not None:
        ax2.plot(d["t"], fd, color=WARN, lw=0.7, alpha=0.8,
                 label="camera disagreement (px)")
        ax2.set_ylabel("px", color=WARN)
        ax2.set_ylim(0, max(1.0, np.nanpercentile(fd[np.isfinite(fd)], 99)
                            if np.isfinite(fd).any() else 1))
    _grid(ax)
    return _png(fig)


def velocity(b):
    d = b["derived"]
    num = b["node"]["csv"]["num"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax = axes[0]
    _shade_vps(ax, d)
    ax.plot(d["t"], d["sent_speed_sm"], color=ACC,
            label=f"sent speed, {d['vel_win_s']:.0f} s mean")
    if d["has_truth"]:
        ax.plot(d["t"], d["truth_speed"], color=TRUTH, lw=0.9,
                label=f"receiver speed, {d['vel_win_s']:.0f} s difference")
    db = d["deadband"]
    if np.isfinite(db).any():
        ax.plot(d["t"], db, color=MUTED, ls="--", lw=0.8, label="deadband")
    ax.set_ylabel("m/s")
    ax.set_title("What we sent against what the receiver did - same window on "
                 "both sides")
    ax.legend(loc="upper right", framealpha=0.2, fontsize=8)
    _grid(ax)

    ax = axes[1]
    _shade_vps(ax, d)
    sent = d["sent_speed"]
    z = np.isfinite(sent) & (sent <= 1e-9)
    ax.plot(d["t"], sent, color=ACC, lw=0.6, alpha=0.8, label="sent, raw")
    ax.plot(d["t"][z], np.zeros(z.sum()), ".", color=WARN, ms=1.5,
            label="zeroed by the deadband")
    ax.set_ylabel("m/s")
    ax.set_xlabel("node time (s)")
    ax.set_title("Every zero is a step input to the flight controller")
    ax.legend(loc="upper right", framealpha=0.2, fontsize=8)
    _grid(ax)
    return _png(fig)


def position(b):
    d = b["derived"]
    num = b["node"]["csv"]["num"]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    _shade_vps(ax, d)
    if d["has_truth"]:
        ax.plot(d["t"], d["pos_err"], color=ACC, lw=0.9,
                label="|ours - receiver| (gps1)")
    gl = num.get("gps_north_m")
    if gl is not None and d["has_truth"]:
        # Drawn deliberately, and labelled for what it is: the trap this
        # project fell into twice. Seeing the two curves apart on the VPS
        # stretch is the fastest way to understand why gps_* is not truth.
        fused = np.hypot(d["ours_n"] - gl,
                         d["ours_e"] - num.get("gps_east_m", gl))
        ax.plot(d["t"], fused, color=MUTED, lw=0.7, alpha=0.7,
                label="against gps_* (the EKF's fused output - NOT truth)")
    fl = num.get("fix_lon")
    used = (np.isfinite(fl) & (num.get("fix_used", np.zeros(d["n"])) > 0.5)
            if fl is not None else np.zeros(d["n"], bool))
    if used.any():
        ax.plot(d["t"][used], np.full(used.sum(), 0.0), "|", color=OK, ms=8,
                label=f"{int(used.sum())} accepted map fixes")
    ax.set_ylabel("m")
    ax.set_xlabel("node time (s)")
    ax.set_title("Position error against the raw receiver")
    ax.legend(loc="upper left", framealpha=0.2, fontsize=8)
    _grid(ax)
    ax2 = ax.twinx()
    ax2.plot(d["t"], d["alt"], color="#2f4a63", lw=0.8)
    ax2.set_ylabel("height (m)", color="#45627d")
    return _png(fig)


def phantom(b):
    d = b["derived"]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    if not d["has_truth"]:
        ax.text(0.5, 0.5, "no independent truth (gps1) in this log",
                ha="center", color=WARN, transform=ax.transAxes)
        return _png(fig)
    still = d["truth_speed"] < 0.5
    m = d["vps"] & still & np.isfinite(d["sent_speed_sm"])
    xs, med, p90, p10 = [], [], [], []
    for lo, hi in analysis.BANDS:
        bm = m & (d["alt"] >= lo) & (d["alt"] < hi)
        if bm.sum() > 20:
            v = d["sent_speed_sm"][bm]
            xs.append((lo + hi) / 2)
            med.append(np.nanmedian(v))
            p90.append(np.nanpercentile(v, 90))
            p10.append(np.nanpercentile(v, 10))
    if xs:
        ax.fill_between(xs, p10, p90, color=ACC, alpha=0.2, label="p10-p90")
        ax.plot(xs, med, "o-", color=ACC, label="median sent speed")
    ax.axhline(0.5, color=MUTED, ls="--", lw=0.8)
    ax.text(ax.get_xlim()[0], 0.52, " the receiver's own threshold for 'still'",
            color=MUTED, fontsize=8)
    ax.set_xlabel("height (m)")
    ax.set_ylabel("m/s sent")
    ax.set_title("Speed sent while the receiver says the aircraft is standing "
                 "still (under 0.5 m/s)")
    ax.legend(loc="upper left", framealpha=0.2, fontsize=8)
    _grid(ax)
    return _png(fig)


def mapfix(b):
    d = b["derived"]
    num = b["node"]["csv"]["num"]
    fl = num.get("fix_lon")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    if fl is None or not np.isfinite(fl).any():
        axes[0].text(0.5, 0.5, "no map fixes in this log", ha="center",
                     color=WARN, transform=axes[0].transAxes)
        return _png(fig)
    att = np.isfinite(fl)
    used = att & (num.get("fix_used", np.zeros(d["n"])) > 0.5)
    dis = num.get("fix_disagree_m", np.full(d["n"], np.nan))
    cap = float(b["config"]["max_correction_m"])
    ax = axes[0]
    ax.plot(d["alt"][att & ~used], dis[att & ~used], ".", color=MUTED, ms=4,
            label="refused")
    ax.plot(d["alt"][used], dis[used], ".", color=ACC, ms=5, label="accepted")
    ax.axhline(cap, color=WARN, ls="--", lw=0.9,
               label=f"max_correction_m {cap:.2f}")
    ax.set_xlabel("height (m)")
    ax.set_ylabel("disagreement with our position (m)")
    ax.set_title("How far each fix asked us to move")
    ax.legend(loc="upper left", framealpha=0.2, fontsize=8)
    _grid(ax)

    ax = axes[1]
    sc = num.get("fix_score", np.full(d["n"], np.nan))
    ax.plot(d["alt"][att & ~used], sc[att & ~used], ".", color=MUTED, ms=4)
    ax.plot(d["alt"][used], sc[used], ".", color=OK, ms=5)
    ax.set_xlabel("height (m)")
    ax.set_ylabel("fix_score")
    ax.set_title("Match score by height")
    _grid(ax)
    return _png(fig)


def height(b):
    d = b["derived"]
    num = b["node"]["csv"]["num"]
    rng = num.get("rngfnd_m")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    if rng is None or not np.isfinite(rng).any():
        axes[0].text(0.5, 0.5, "no rangefinder in this log", ha="center",
                     color=WARN, transform=axes[0].transAxes)
        return _png(fig)
    air = np.isfinite(rng) & np.isfinite(d["alt"]) & (rng > 2) & (d["alt"] > 3)
    ax = axes[0]
    ax.plot(rng[air], d["alt"][air], ".", color=ACC, ms=2, alpha=0.5)
    lim = max(1.0, np.nanmax(rng[air]) if air.any() else 1)
    ax.plot([0, lim], [0, lim], color=TRUTH, ls="--", lw=0.9, label="1:1")
    ax.set_xlabel("lidar (m)")
    ax.set_ylabel("height in use (m)")
    ax.set_title("The height the velocity is scaled by, against the lidar")
    ax.legend(loc="upper left", framealpha=0.2, fontsize=8)
    _grid(ax)

    ax = axes[1]
    ratio = np.where(air, d["alt"] / np.where(rng > 0, rng, np.nan), np.nan)
    ax.plot(d["alt"][air], ratio[air], ".", color=WARN, ms=2, alpha=0.5)
    ax.axhline(1.0, color=MUTED, ls="--", lw=0.8)
    ax.set_xlabel("height (m)")
    ax.set_ylabel("height / lidar")
    ax.set_ylim(0.8, 2.2)
    ax.set_title("Velocity is pixels x height / focal, so this ratio IS the "
                 "velocity error")
    _grid(ax)
    return _png(fig)


def loop(b):
    d = b["derived"]
    num = b["node"]["csv"]["num"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    dt = d["dt"] * 1000.0
    ax.plot(d["t"], dt, ".", color=ACC, ms=1.5, alpha=0.5)
    ax.axhline(60, color=WARN, ls="--", lw=0.9, label="60 ms")
    ax.set_ylim(0, max(200, np.nanpercentile(dt[np.isfinite(dt)], 99.5)
                       if np.isfinite(dt).any() else 200))
    ax.set_xlabel("node time (s)")
    ax.set_ylabel("iteration interval (ms)")
    ax.set_title("Loop interval - it starves at height")
    ax.legend(loc="upper right", framealpha=0.2, fontsize=8)
    _grid(ax)

    ax = axes[1]
    ok = np.isfinite(dt) & np.isfinite(d["alt"])
    ax.plot(d["alt"][ok], dt[ok], ".", color=MUTED, ms=1.5, alpha=0.4)
    ax.axhline(60, color=WARN, ls="--", lw=0.9)
    ax.set_ylim(0, 200)
    ax.set_xlabel("height (m)")
    ax.set_ylabel("ms")
    ax.set_title("locate() cost rises with the window radius, "
                 "clip(0.60 x alt, 40, 150)")
    _grid(ax)
    return _png(fig)


def ekf(b):
    arrays, meta = b["fc"]["arrays"], b["fc"]["meta"]
    thr = meta["params"].get("FS_EKF_THRESH", 1.0)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for name, col, lbl in (("SV", WARN, "velocity innovation (SV)"),
                           ("SP", ACC, "position innovation (SP)"),
                           ("SH", OK, "height innovation (SH)")):
        t, v = series(arrays, "XKF4", name, where=("C", 0))
        if len(t):
            ax.plot(t, v, color=col, lw=0.8, label=lbl)
    ax.axhline(thr, color=TRUTH, ls="--", lw=0.9,
               label=f"FS_EKF_THRESH {thr:.2f}")
    for a, bb in (b.get("vps_fc") or []):
        ax.axvspan(a, bb, color="#1b3a5c", alpha=0.45, lw=0, zorder=0)
    import logs as _l
    for e in _l.errors(meta):
        if e["n"] == 24:
            ax.axvline(e["t"], color="#8b5cf6", lw=0.5, alpha=0.6)
    ax.set_xlabel("flight controller time (s)")
    ax.set_ylabel("innovation ratio")
    ax.set_title("EKF innovations (shaded = VPS driving, purple = EKF primary "
                 "core change). This says whether a failsafe will fire, not "
                 "whether the hold is good")
    ax.legend(loc="upper right", framealpha=0.2, fontsize=8)
    _grid(ax)
    return _png(fig)


def attitude(b):
    arrays = b["fc"]["arrays"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax = axes[0]
    for f, col, lbl in (("DesRoll", WARN, "commanded roll"),
                        ("Roll", ACC, "roll"),
                        ("DesPitch", TRUTH, "commanded pitch"),
                        ("Pitch", OK, "pitch")):
        t, v = series(arrays, "ATT", f)
        if len(t):
            ax.plot(t, v, color=col, lw=0.7,
                    ls="--" if f.startswith("Des") else "-", label=lbl)
    amax = b["fc"]["meta"]["params"].get("ATC_ANGLE_MAX", 15.0)
    amax = amax / 100.0 if amax > 100 else amax
    for s in (amax, -amax):
        ax.axhline(s, color=MUTED, ls=":", lw=0.8)
    for a, bb in (b.get("vps_fc") or []):
        ax.axvspan(a, bb, color="#1b3a5c", alpha=0.45, lw=0, zorder=0)
    ax.set_ylabel("deg")
    ax.set_title(f"Attitude against what was commanded "
                 f"(dotted = ATC_ANGLE_MAX {amax:.0f} deg)")
    ax.legend(loc="upper right", framealpha=0.2, fontsize=8, ncol=2)
    _grid(ax)

    ax = axes[1]
    for f, col, lbl in (("R", ACC, "roll rate"), ("P", TRUTH, "pitch rate")):
        t, v = series(arrays, "RATE", f)
        if len(t):
            ax.plot(t, v, color=col, lw=0.6, label=lbl)
    ax.set_xlabel("flight controller time (s)")
    ax.set_ylabel("deg/s")
    ax.set_title("Body rates - a step in the commanded velocity shows up here")
    ax.legend(loc="upper right", framealpha=0.2, fontsize=8)
    _grid(ax)
    return _png(fig)


def vibe(b):
    arrays = b["fc"]["arrays"]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for f, col in (("VibeX", ACC), ("VibeY", TRUTH), ("VibeZ", OK)):
        t, v = series(arrays, "VIBE", f, where=("IMU", 0))
        if len(t):
            ax.plot(t, v, color=col, lw=0.7, label=f)
    ax.axhline(30, color=WARN, ls="--", lw=0.8, label="30 m/s2")
    ax.set_xlabel("flight controller time (s)")
    ax.set_ylabel("m/s2")
    ax.set_title("Vibration (at rest on 20 Oct this airframe read 0.8 / 1.6 / "
                 "2.3). Ruled out as the cause of the phantom on flight 239")
    ax.legend(loc="upper right", framealpha=0.2, fontsize=8)
    _grid(ax)
    return _png(fig)


def flow(b):
    d = b["derived"]
    num = b["node"]["csv"]["num"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax = axes[0]
    _shade_vps(ax, d)
    for k, col, lbl in (("flow_disagree_px", WARN, "camera disagreement (px)"),
                        ("step_px", ACC, "frame-to-frame shift (px)")):
        v = num.get(k)
        if v is not None and np.isfinite(v).any():
            ax.plot(d["t"], v, color=col, lw=0.6, label=lbl)
    ax.axhline(0.5, color=MUTED, ls="--", lw=0.8, label="0.5 px veto")
    ax.set_ylabel("px")
    ax.set_title("Flow: the two cameras against each other. 2.32 px on flight "
                 "163, 0.050 on 245 - but on 239's worst over-reports they "
                 "agreed to 0.04 px")
    ax.legend(loc="upper right", framealpha=0.2, fontsize=8)
    ax.set_ylim(0, 3)
    _grid(ax)

    ax = axes[1]
    _shade_vps(ax, d)
    for k, col, lbl in (("vel_age_s", WARN, "velocity age (s)"),
                        ("frame_age_s", MUTED, "frame age (s)")):
        v = num.get(k)
        if v is not None and np.isfinite(v).any():
            ax.plot(d["t"], v, color=col, lw=0.7, label=lbl)
    ax.axhline(0.6, color=TRUTH, ls="--", lw=0.8, label="0.6 s stale gate")
    ax.set_xlabel("node time (s)")
    ax.set_ylabel("s")
    ax.legend(loc="upper right", framealpha=0.2, fontsize=8)
    _grid(ax)
    return _png(fig)


def derotation(b):
    """What the derotation is doing - the 25 Oct fix, and its residual."""
    d = b["derived"]
    num = b["node"]["csv"]["num"]
    fig, ax = plt.subplots(figsize=(11, 4.4))
    _shade_vps(ax, d)
    pairs = (("fc_d_roll", ACC, "FC roll delta"),
             ("gyro_d_roll", TRUTH, "camera gyro roll delta"),
             ("quat_d_roll", OK, "quaternion roll delta"))
    for k, col, lbl in pairs:
        v = num.get(k)
        if v is not None and np.isfinite(v).any():
            ax.plot(d["t"], v, color=col, lw=0.7, label=lbl)
    ax.set_xlabel("node time (s)")
    ax.set_ylabel("rad per frame")
    ax.set_title("Rotation per frame, from each source. The phantom was a "
                 "model error: one uniform shift cannot derotate a 77.7 deg "
                 "frame, so derotation is now per track")
    ax.legend(loc="upper right", framealpha=0.2, fontsize=8)
    _grid(ax)
    return _png(fig)


PANELS = {
    "overview":   dict(fn=overview,   needs="node", title="Height and coverage"),
    "velocity":   dict(fn=velocity,   needs="node", title="Velocity sent against the receiver"),
    "position":   dict(fn=position,   needs="node", title="Position error"),
    "phantom":    dict(fn=phantom,    needs="node", title="Phantom velocity by height"),
    "mapfix":     dict(fn=mapfix,     needs="node", title="Map fixes"),
    "height":     dict(fn=height,     needs="node", title="Barometer against lidar"),
    "flow":       dict(fn=flow,       needs="node", title="Optical flow health"),
    "derotation": dict(fn=derotation, needs="node", title="Derotation sources"),
    "loop":       dict(fn=loop,       needs="node", title="Loop rate"),
    "ekf":        dict(fn=ekf,        needs="fc",   title="EKF innovations"),
    "attitude":   dict(fn=attitude,   needs="fc",   title="Attitude and body rates"),
    "vibe":       dict(fn=vibe,       needs="fc",   title="Vibration"),
}


def render(b, name):
    p = PANELS.get(name)
    if not p:
        raise KeyError(name)
    if p["needs"] == "node" and not b.get("node"):
        raise ValueError(f"{name} needs a node log")
    if p["needs"] == "fc" and not b.get("fc"):
        raise ValueError(f"{name} needs a flight controller log")
    return p["fn"](b)
