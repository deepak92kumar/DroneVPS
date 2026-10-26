#!/usr/bin/env python3
"""Finding the logs on this machine and reading them once.

There are two logs per flight and they are not interchangeable:

  flight_logs/NNNNNNNN.BIN        the flight controller's own log, 10-26 MB,
                                  half a million messages. This is the only
                                  place the modes, the EKF variances, the
                                  vibration, the RC switch and the raw
                                  receiver live - and the node's own status
                                  lines, which it sends as MAVLink MSGs, land
                                  in here too.

  flight_logs/vps/vps_NNNNN/      the Jetson's own log: one row per iteration,
                                  80 columns, everything the node believed and
                                  everything it sent.

They have separate clocks. The .BIN counts microseconds since the flight
controller booted; the CSV counts seconds since the node started. Nothing in
either file states the offset, so `align()` recovers it by correlating the two
altitude traces, reports the correlation it achieved, and leaves the caller to
decide whether to trust it. Every finding that can be made from one file alone
is made from one file alone, so a failed alignment costs the combined plots
and nothing else.

Parsing a .BIN costs about 10 s on the Jetson, so it is done once and cached
as an .npz beside it. The cache key is the file's size and mtime: a log that
is still being written gets re-read, and a log that is finished never is.
"""
from __future__ import annotations

import csv
import glob
import io
import json
import os
import time

import numpy as np

ROOT = "/data/DRONE"
BIN_DIR = os.path.join(ROOT, "flight_logs")
VPS_DIR = os.path.join(ROOT, "flight_logs", "vps")
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

# What is pulled out of a .BIN, and nothing else. Reading every field of every
# message would be 500 000 dicts and most of a gigabyte; these are the ones
# some finding or some plot actually reads.
WANTED = {
    "ATT":  ("DesRoll", "Roll", "DesPitch", "Pitch", "DesYaw", "Yaw"),
    "RATE": ("RDes", "R", "PDes", "P", "YDes", "Y"),
    "XKF1": ("C", "Roll", "Pitch", "Yaw", "VN", "VE", "VD", "PN", "PE", "PD", "OH"),
    "XKF4": ("C", "SV", "SP", "SH", "SM", "SVT", "FS", "SS"),
    "GPS":  ("I", "Status", "NSats", "HDop", "Lat", "Lng", "Alt", "Spd", "GCrs",
             "GMS", "GWk"),
    "GPA":  ("I", "HAcc", "VAcc", "SAcc"),
    "RFND": ("Instance", "Dist", "Stat", "Quality"),
    "BARO": ("I", "Alt", "AltAMSL", "CRt", "Offset"),
    "VIBE": ("IMU", "VibeX", "VibeY", "VibeZ", "Clip"),
    "CTUN": ("ThO", "ThH", "DAlt", "Alt", "BAlt", "DCRt", "CRt"),
    "PSCN": ("TPN", "PN", "TVN", "VN", "TAN", "AN"),
    "PSCE": ("TPE", "PE", "TVE", "VE", "TAE", "AE"),
    "POS":  ("Lat", "Lng", "Alt", "RelHomeAlt"),
    "RCIN": ("C5", "C6", "C7", "C8"),
    "PM":   ("NLon", "MaxT", "Load", "Mem"),
    "MAG":  ("I", "MagX", "MagY", "MagZ", "Health"),
}
# These carry text or codes and are kept as records, not arrays.
RECORDS = ("MODE", "MSG", "ERR", "EV")

MODE_NAMES = {0: "STABILIZE", 1: "ACRO", 2: "ALT_HOLD", 3: "AUTO", 4: "GUIDED",
              5: "LOITER", 6: "RTL", 7: "CIRCLE", 9: "LAND", 11: "DRIFT",
              13: "SPORT", 14: "FLIP", 15: "AUTOTUNE", 16: "POSHOLD",
              17: "BRAKE", 18: "THROW", 19: "AVOID_ADSB", 20: "GUIDED_NOGPS",
              21: "SMART_RTL", 22: "FLOWHOLD", 23: "FOLLOW", 24: "ZIGZAG",
              27: "AUTO_RTL"}
# ArduPilot's own reason codes for a mode change. "4" is the one that matters
# here: the aircraft changed mode by itself because a failsafe fired.
MODE_REASON = {0: "unknown", 1: "RC command", 2: "GCS command",
               3: "radio failsafe", 4: "battery failsafe",
               5: "GCS failsafe", 6: "EKF failsafe", 7: "GPS glitch",
               8: "mission end", 9: "throttle land escape",
               10: "fence breach", 11: "terrain failsafe",
               12: "brake timeout", 13: "flip complete", 14: "avoidance",
               15: "avoidance recovery", 16: "throw complete",
               17: "terminate", 18: "toy mode", 19: "crash failsafe",
               25: "initialised", 26: "surface complete"}
# Anything but these two was the aircraft's decision, not the pilot's.
REASON_COMMANDED = {"RC command", "GCS command", "initialised", "unknown"}
EV_NAMES = {10: "ARMED", 11: "DISARMED", 15: "AUTO_ARMED",
            17: "LAND_COMPLETE_MAYBE", 18: "LAND_COMPLETE", 19: "LOST_GPS",
            25: "SET_HOME", 28: "NOT_LANDED", 60: "EKF_ALT_RESET",
            61: "LAND_CANCELLED_BY_PILOT", 62: "EKF_YAW_RESET"}
# ArduPilot's LogErrorSubsystem, which is NOT the same numbering as anything
# else in the log. Only the ones that are named here are named; an unknown
# code is printed as its number rather than guessed at, because a wrong
# subsystem name in a review is worse than a number the reader can look up.
ERR_SUBSYS = {1: "main", 2: "radio", 3: "compass", 4: "optical flow",
              5: "radio failsafe", 6: "battery failsafe", 7: "GPS failsafe",
              8: "GCS failsafe", 9: "fence failsafe", 10: "flight mode",
              11: "GPS", 12: "crash check", 13: "flip", 14: "autotune",
              15: "parachute", 16: "EKF check", 17: "EKF failsafe",
              18: "barometer", 19: "CPU", 20: "ADSB failsafe", 21: "terrain",
              22: "navigation", 23: "terrain failsafe", 24: "EKF primary",
              25: "thrust loss", 26: "sensor failsafe", 27: "leak failsafe",
              28: "pilot input", 29: "vibration failsafe",
              30: "internal error", 31: "dead reckoning failsafe"}
# The subsystems whose entries mean something is wrong with THIS aircraft's
# navigation. "pilot input" pairs are the sticks leaving and returning to
# centre; "EKF primary" is a lane change, which is worth seeing but is the
# EKF doing its job.
ERR_SERIOUS = {11, 12, 16, 17, 18, 22, 25, 26, 29, 30, 31}
# The ECode means different things per subsystem - for an EKF primary change
# it is not an error code at all, it is the core that took over. Reading it
# as a generic error is how "EKF primary / failed to initialise" appears in a
# report 25 times for something that is a lane change.
ERR_CODE = {0: "resolved", 1: "occurred", 4: "unhealthy"}
ERR_CODE_BY_SUBSYS = {
    11: {0: "glitch cleared", 2: "GPS glitch"},
    16: {0: "variances back within limits", 2: "bad variance"},
    17: {0: "cleared", 1: "EKF failsafe fired"},
    29: {0: "cleared", 1: "vibration failsafe fired"},
    12: {0: "cleared", 1: "crash detected", 2: "loss of control"},
    18: {0: "healthy again", 4: "barometer unhealthy"},
    2:  {0: "radio back", 2: "late frame"},
}

# --------------------------------------------------------------------------- #
#  what is on the disk
# --------------------------------------------------------------------------- #
def _stamp(path):
    st = os.stat(path)
    return f"{st.st_size}-{int(st.st_mtime)}"


def discover():
    """Every FC log and every node log on this machine, newest first."""
    fc = []
    for p in sorted(glob.glob(os.path.join(BIN_DIR, "*.BIN"))):
        st = os.stat(p)
        fc.append(dict(kind="fc", id=os.path.basename(p)[:-4], path=p,
                       name=os.path.basename(p), mb=st.st_size / 1e6,
                       mtime=st.st_mtime,
                       when=time.strftime("%d %b %H:%M", time.localtime(st.st_mtime)),
                       cached=os.path.exists(_cache_path(p))))
    node = []
    seen = set()
    for d in sorted(glob.glob(os.path.join(VPS_DIR, "vps_*"))):
        if not os.path.isdir(d):
            continue
        csvs = sorted(glob.glob(os.path.join(d, "vps_*.csv")))
        csvs = [c for c in csvs if not c.endswith("_imu.csv")]
        if not csvs:
            continue
        st = os.stat(csvs[0])
        seen.add(os.path.basename(d))
        node.append(dict(kind="node", id=os.path.basename(d), path=d,
                         csv=csvs[0], name=os.path.basename(d),
                         mb=st.st_size / 1e6, mtime=st.st_mtime,
                         when=time.strftime("%d %b %H:%M", time.localtime(st.st_mtime)),
                         frames=len(glob.glob(os.path.join(d, "*.jpg")))
                         + len(glob.glob(os.path.join(d, "mono", "*.png")))))
    # The older recorder wrote a bare CSV with no folder around it.
    for p in sorted(glob.glob(os.path.join(VPS_DIR, "vps_*.csv"))):
        if p.endswith("_imu.csv"):
            continue
        stem = os.path.basename(p)[:-4]
        if stem in seen:
            continue
        st = os.stat(p)
        node.append(dict(kind="node", id=stem, path=os.path.dirname(p), csv=p,
                         name=stem, mb=st.st_size / 1e6, mtime=st.st_mtime,
                         when=time.strftime("%d %b %H:%M", time.localtime(st.st_mtime)),
                         frames=0))
    fc.sort(key=lambda r: -r["mtime"])
    node.sort(key=lambda r: -r["mtime"])
    # A guess at which node log belongs to which FC log, by wall clock. Both
    # files are closed within a minute or two of the same landing, and the
    # numbering is independent (245 against vps_00165), so there is nothing
    # better to go on until both are open. It is offered, never assumed.
    for f in fc:
        best = min(node, key=lambda n: abs(n["mtime"] - f["mtime"]), default=None)
        if best and abs(best["mtime"] - f["mtime"]) < 3600:
            f["suggest"] = best["id"]
            f["suggest_gap_min"] = abs(best["mtime"] - f["mtime"]) / 60.0
    return dict(fc=fc, node=node)


# --------------------------------------------------------------------------- #
#  the flight controller's log
# --------------------------------------------------------------------------- #
def _cache_path(path):
    base = os.path.basename(path) + "." + _stamp(path) + ".npz"
    return os.path.join(CACHE, base)


def read_bin(path: str, progress=None):
    """Every wanted field of a .BIN as float arrays, cached as an .npz.

    Arrays are named "TYPE.FIELD" and every type also gets "TYPE.t", seconds
    since the flight controller booted. Records (modes, messages, errors,
    events) and the parameter table come back beside them.
    """
    path = os.path.abspath(path)
    os.makedirs(CACHE, exist_ok=True)
    cp = _cache_path(path)
    if os.path.exists(cp):
        z = np.load(cp, allow_pickle=False)
        arrays = {k: z[k] for k in z.files}
        meta = json.load(open(cp + ".json"))
        return arrays, meta

    from pymavlink import mavutil
    mlog = mavutil.mavlink_connection(path)
    cols = {}
    recs = {k: [] for k in RECORDS}
    parm = {}
    t0 = time.time()
    n = 0
    while True:
        try:
            msg = mlog.recv_match()
        except Exception:
            break
        if msg is None:
            break
        n += 1
        if progress and n % 100000 == 0:
            progress(n)
        typ = msg.get_type()
        if typ == "PARM":
            parm[str(msg.Name)] = float(msg.Value)
            continue
        if typ in RECORDS:
            t = getattr(msg, "TimeUS", 0) / 1e6
            if typ == "MODE":
                recs[typ].append(dict(t=t, mode=int(getattr(msg, "ModeNum", -1)),
                                      reason=int(getattr(msg, "Rsn", -1))))
            elif typ == "MSG":
                recs[typ].append(dict(t=t, text=str(msg.Message)))
            elif typ == "ERR":
                recs[typ].append(dict(t=t, subsys=int(msg.Subsys),
                                      code=int(msg.ECode)))
            else:
                recs[typ].append(dict(t=t, id=int(msg.Id)))
            continue
        fields = WANTED.get(typ)
        if not fields:
            continue
        d = cols.setdefault(typ, {f: [] for f in ("t",) + fields})
        d["t"].append(getattr(msg, "TimeUS", 0) / 1e6)
        for f in fields:
            v = getattr(msg, f, None)
            try:
                d[f].append(float(v) if v is not None else np.nan)
            except (TypeError, ValueError):
                d[f].append(np.nan)

    arrays = {}
    for typ, d in cols.items():
        for f, vals in d.items():
            arrays[f"{typ}.{f}"] = np.asarray(vals, dtype=np.float64)
    meta = dict(path=path, messages=n, parse_s=round(time.time() - t0, 1),
                records=recs, params=parm,
                types=sorted(cols.keys()))
    np.savez_compressed(cp, **arrays)
    json.dump(meta, open(cp + ".json", "w"))
    return arrays, meta


def series(arrays, typ, field, where=None):
    """(t, v) for one field, with NaNs dropped - and optionally one instance.

    `where` is (field, value), which is how a single IMU, GPS or rangefinder
    instance is picked out of a log that interleaves all of them. Forgetting
    it is how you get a trace that alternates between two sensors.
    """
    tk, vk = f"{typ}.t", f"{typ}.{field}"
    if tk not in arrays or vk not in arrays:
        return np.array([]), np.array([])
    t, v = arrays[tk], arrays[vk]
    m = np.isfinite(t) & np.isfinite(v)
    if where:
        wk = f"{typ}.{where[0]}"
        if wk in arrays:
            m &= arrays[wk] == where[1]
    return t[m], v[m]


def modes(meta):
    out = []
    for r in meta["records"].get("MODE", []):
        out.append(dict(t=r["t"], mode=r["mode"],
                        name=MODE_NAMES.get(r["mode"], f"mode {r['mode']}"),
                        reason=MODE_REASON.get(r["reason"],
                                               f"reason {r['reason']}")))
    return out


def events(meta):
    out = []
    for r in meta["records"].get("EV", []):
        out.append(dict(t=r["t"], id=r["id"],
                        name=EV_NAMES.get(r["id"], f"event {r['id']}"),
                        known=r["id"] in EV_NAMES))
    return out


def errors(meta):
    out = []
    for r in meta["records"].get("ERR", []):
        out.append(dict(t=r["t"], n=r["subsys"],
                        subsys=ERR_SUBSYS.get(r["subsys"],
                                              f"subsystem {r['subsys']}"),
                        code=r["code"],
                        code_name=(f"primary core {r['code']}"
                                   if r["subsys"] == 24 else
                                   ERR_CODE_BY_SUBSYS.get(r["subsys"], {}).get(
                                       r["code"],
                                       ERR_CODE.get(r["code"], str(r["code"])))),
                        serious=r["subsys"] in ERR_SERIOUS))
    return out


def node_messages(meta):
    """The Jetson's own status lines, which it sends to the FC as MAVLink MSGs.

    They are the only record of what the node was saying at a given flight
    controller timestamp, which makes them the bridge between the two logs
    even when the altitude correlation fails. The node prefixes them
    "SRC=sys/comp:VPS ".
    """
    out = []
    for r in meta["records"].get("MSG", []):
        txt = r["text"]
        if ":VPS " in txt:
            out.append(dict(t=r["t"], text=txt.split(":VPS ", 1)[1]))
    return out


def all_messages(meta):
    return [dict(t=r["t"], text=r["text"]) for r in meta["records"].get("MSG", [])]


# --------------------------------------------------------------------------- #
#  the node's own log
# --------------------------------------------------------------------------- #
def read_csv(path: str, want=None):
    """The node's CSV as a dict of float arrays, plus the text columns.

    NUL padding is stripped the way analyse_flight.py strips it: either
    recorder can be killed mid-write, and a file that ends in NULs is not a
    corrupt flight, it is a flight that was still being written when the power
    went off.
    """
    raw = open(path, "rb").read().replace(b"\x00", b"")
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf8", "replace"))))
    if not rows:
        raise RuntimeError(f"{path} has a header and no rows")
    keys = list(rows[0].keys())
    num, txt = {}, {}
    # Pairing a .BIN against eight candidate node logs only needs the clock
    # and the altitude; converting all 80 columns of each candidate is most of
    # a minute of the pilot's time for nothing.
    for k in (keys if want is None else [k for k in keys if k in want]):
        vals = [r.get(k, "") for r in rows]
        try:
            num[k] = np.array([float(v) if v not in ("", None) else np.nan
                               for v in vals])
        except ValueError:
            txt[k] = vals
    return dict(n=len(rows), num=num, txt=txt, columns=keys, path=path)


def origin_of(csvd):
    """The node's origin, recovered from any row that carries both frames.

    Every row's lat/lon IS the origin plus that row's own north/east, so one
    row with all four gives the origin back. This is analyse_flight.py's trick
    and it is exact - the origin is never written to the CSV.
    """
    num = csvd["num"]
    for k in ("lat", "lon", "north_m", "east_m"):
        if k not in num:
            return None
    lat, lon, n, e = num["lat"], num["lon"], num["north_m"], num["east_m"]
    ok = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(n) & np.isfinite(e)
    ok &= (lat != 0) & (lon != 0)
    if not ok.any():
        return None
    i = int(np.argmax(ok))
    olat = lat[i] - n[i] / 111_320.0
    mlon = 111_320.0 * np.cos(np.radians(olat))
    return float(lon[i] - e[i] / mlon), float(olat)


def align(csvd, arrays):
    """The offset between the node's clock and the flight controller's.

    Both logs carry an altitude at a few hertz. Resample each to 1 Hz, try
    every whole-second lag, and keep the one with the highest correlation.
    Returns the lag to ADD to the node's `t` to land on the FC's clock, the
    correlation it reached, and how much overlap it had - so a caller can
    refuse a weak alignment instead of plotting a lie.
    """
    num = csvd["num"]
    if "t" not in num or "alt_m" not in num:
        return None
    ct, ca = num["t"], num["alt_m"]
    ft, fa = series(arrays, "CTUN", "Alt")
    if len(ft) < 30:
        ft, fa = series(arrays, "BARO", "Alt", where=("I", 0))
    m = np.isfinite(ct) & np.isfinite(ca)
    ct, ca = ct[m], ca[m]
    if len(ct) < 30 or len(ft) < 30:
        return None

    def to_1hz(t, v):
        grid = np.arange(np.floor(t[0]), np.ceil(t[-1]) + 1, 1.0)
        return grid, np.interp(grid, t, v)

    cg, cv = to_1hz(ct, ca)
    fg, fv = to_1hz(ft, fa)
    best = None
    # The node is started by hand some time after the FC boots, so the lag is
    # positive and usually under a few minutes. The whole FC log is searched
    # anyway: it costs milliseconds at 1 Hz.
    for lag in range(int(fg[0] - cg[-1]), int(fg[-1] - cg[0]) + 1):
        a_t = cg + lag
        lo, hi = max(a_t[0], fg[0]), min(a_t[-1], fg[-1])
        if hi - lo < 30:
            continue
        grid = np.arange(lo, hi, 1.0)
        x = np.interp(grid, a_t, cv)
        y = np.interp(grid, fg, fv)
        if np.std(x) < 0.5 or np.std(y) < 0.5:
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        if best is None or r > best["corr"]:
            best = dict(lag_s=float(lag), corr=r, overlap_s=float(hi - lo))
    if best:
        best["trust"] = bool(best["corr"] > 0.9 and best["overlap_s"] > 60)
    return best
