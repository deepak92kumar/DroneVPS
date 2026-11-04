"""Absolute attitude from the FC log, on the node's clock.

The vps CSV records attitude DELTAS only, so roll and pitch have to come from
the .bin. The two clocks are aligned by cross-correlating yaw AND altitude -
both, because either alone can lock onto a wrong peak, and the alignment is
only trusted when the two agree.

Measured 4 Nov:
    vps_00476 <- 2026-11-04 11-07-52.bin   yaw r +0.999  alt r +1.000  same lag
    vps_00478 <- 2026-11-04 12-54-13.bin   yaw r +1.000  alt r +1.000  same lag
"""
import sys, csv, os
sys.path.insert(0, "/home/abc/.local/lib/python3.10/site-packages")
import numpy as np
from pymavlink import mavutil

BIN = {"vps_00476": "/home/abc/Downloads/2026-11-04 11-07-52.bin",
       "vps_00478": "/home/abc/Downloads/2026-11-04 12-54-13.bin"}
BIN_DIRS = ["/home/abc/Downloads", "/data/DRONE/flight_logs/fc"]
_CACHE = {}
_FOUND = {}


def _grid(t, y, dt=0.25):
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    g = np.arange(t[0], t[-1], dt)
    return g, np.interp(g, t, y)


def _best_lag(a, b, min_overlap=400):
    best = (-9.0, 0, 0)
    for lag in range(-len(b) + min_overlap, len(a) - min_overlap):
        i0, j0 = max(0, lag), max(0, -lag)
        L = min(len(a) - i0, len(b) - j0)
        if L < min_overlap:
            continue
        x, y = a[i0:i0 + L], b[j0:j0 + L]
        sx, sy = x.std(), y.std()
        if sx < 1e-6 or sy < 1e-6:
            continue
        r = float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))
        if r > best[0]:
            best = (r, lag, L)
    return best


def _csv_grids(run):
    rows = list(csv.DictReader(open(
        f"/home/abc/NVME/DRONE/flight_logs/vps/{run}/{run}.csv")))
    g = lambda k: np.array([float(x[k]) if x.get(k) not in (None, "") else np.nan
                            for x in rows])
    ct = g("t")
    gv, yv = _grid(ct, np.unwrap(np.radians(g("yaw_deg"))))
    gva, av = _grid(ct, g("alt_m"))
    return gv, yv, gva, av


def _read_bin(path):
    m = mavutil.mavlink_connection(path)
    att, ctu = [], []
    while True:
        msg = m.recv_match(type=["ATT", "CTUN"])
        if msg is None:
            break
        if msg.get_type() == "ATT":
            att.append((msg.TimeUS / 1e6, msg.Roll, msg.Pitch, msg.Yaw))
        else:
            ctu.append((msg.TimeUS / 1e6, getattr(msg, "Alt", np.nan)))
    return np.array(att), np.array(ctu)


def _score(run, path):
    """-> (offset, r_yaw, r_alt, disagree_s) or None if the two do not match.

    BOTH yaw and altitude have to pick the SAME shift. Either alone can lock
    onto a wrong peak: a first attempt at this used an unnormalised dot
    product and reported correlations above 1.0, which is impossible, and the
    file-modification times said the bin was recorded AFTER the flight it was
    supposed to belong to. Agreement between two independent channels is what
    makes an alignment checkable rather than asserted.
    """
    att, ctu = _read_bin(path)
    if len(att) < 500 or len(ctu) < 500:
        return None
    gv, yv, gva, av = _csv_grids(run)
    ga, ya = _grid(att[:, 0], np.unwrap(np.radians(att[:, 3])))
    gb, ab = _grid(ctu[:, 0], ctu[:, 1])
    ry, ly, _ = _best_lag(yv, ya)
    ra, la, _ = _best_lag(av, ab)
    off_y = gv[0] - ga[0] + 0.25 * ly
    off_a = gva[0] - gb[0] + 0.25 * la
    return (0.5 * (off_y + off_a), ry, ra, abs(off_y - off_a), att)


def find_bin(run, verbose=True):
    """The FC log that belongs to `run`, found by correlation, not by name.

    The filename carries the FC's own clock, which is not wall time: the log
    of the 12:30 flight is named 12-54-13. Nothing about the name or the file
    date identifies a flight, so the flight identifies itself.
    """
    if run in _FOUND:
        return _FOUND[run]
    if run in BIN:
        best = _score(run, BIN[run])
        if best and min(best[1], best[2]) > 0.9 and best[3] < 1.0:
            if verbose:
                print("%s <- %s  (named; yaw r %.3f, alt r %.3f, they agree "
                      "on the clock to %.2f s)" % (
                          run, os.path.basename(BIN[run]), best[1], best[2],
                          best[3]), flush=True)
            _FOUND[run] = (BIN[run], best)
            return _FOUND[run]
    cands = []
    for d in BIN_DIRS:
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith(".bin"):
                cands.append(os.path.join(d, fn))
    best, best_path = None, None
    for path in cands:
        try:
            s = _score(run, path)
        except Exception:
            continue
        if s is None or s[3] > 1.0:
            continue
        key = min(s[1], s[2])
        if best is None or key > min(best[1], best[2]):
            best, best_path = s, path
    if best is None or min(best[1], best[2]) < 0.9:
        raise RuntimeError(f"{run}: no FC log in {BIN_DIRS} matches this "
                           f"flight (best yaw/alt correlation "
                           f"{'none' if best is None else min(best[1], best[2]):.3f}, "
                           f"0.9 needed)")
    if verbose:
        print("%s <- %s  (found by correlation: yaw r %.3f, alt r %.3f, "
              "they agree on the clock to %.2f s)" % (
                  run, os.path.basename(best_path), best[1], best[2], best[3]),
              flush=True)
    _FOUND[run] = (best_path, best)
    return _FOUND[run]


def attitude(run, verbose=True):
    """-> (att array [t_node, roll, pitch, yaw], info dict).

    t_node is the node clock, so the array indexes straight against the vps
    CSV's `t`. The FC log is found by correlation - see find_bin.
    """
    if run in _CACHE:
        return _CACHE[run]
    path, (off, ry, ra, dis, att) = find_bin(run, verbose=verbose)
    att = att.copy()
    att[:, 0] += off                      # FC clock -> node clock
    info = dict(bin=os.path.basename(path), off=off, r_yaw=ry, r_alt=ra,
                disagree_s=dis)
    if verbose:
        print("   node_t = fc_t %+.2f s   (%d ATT samples, %.1f .. %.1f s)" % (
            off, len(att), att[0, 0], att[-1, 0]), flush=True)
    _CACHE[run] = (att, info)
    return att, info


def at(run, t_node):
    """roll, pitch, yaw in degrees at a node timestamp, linearly interpolated.

    Interpolated, not nearest: ATT logs at 10 Hz here and the aircraft rolls
    through several degrees in 100 ms on a waypoint turn.
    """
    a, _ = attitude(run, verbose=False)
    return (float(np.interp(t_node, a[:, 0], a[:, 1])),
            float(np.interp(t_node, a[:, 0], a[:, 2])),
            float(np.interp(t_node, a[:, 0], a[:, 3])))
