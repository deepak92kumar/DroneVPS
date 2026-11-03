#!/usr/bin/env python3
"""Replay a flight through the REAL fuser, with the map fixes it actually had.

My first replay re-integrated the flow steps and nothing else. That is not
what the aircraft did: 406 applied 79 map fixes, 411 applied 139 and 418
applied 59, and leaving them out means the comparison that said "418 got 87%
worse" was measuring something the aircraft never ran. fix_lon and fix_lat
are in the CSV for 189, 195 and 107 rows respectively, so there was never a
reason to drop them.

This drives vps_vio.map_fuse.MapFuser with the recorded steps and the
recorded fixes, at the rows they actually arrived, so the only difference
between the two runs is the yaw applied to the flow.

Scored against gps1_north_m / gps1_east_m, which is the only ground truth
here - never against the node's own fused output.
"""
import csv, math, sys
import numpy as np

sys.path.insert(0, '/data/DRONE/vps_vio')
from map_fuse import MapFuser
import config as C

cfg = C.VpsConfig()


def run(path, yaw_off, use_fixes=True):
    R = list(csv.DictReader(open(path)))

    def F(k):
        o = []
        for r in R:
            try:
                o.append(float(r[k]))
            except (TypeError, ValueError):
                o.append(float('nan'))
        return np.array(o)

    t, dn, de = F('t'), F('step_dn'), F('step_de')
    dt = F('step_dt_s')
    fl, fla, fu = F('fix_lon'), F('fix_lat'), F('fix_used')
    n0, e0 = F('north_m'), F('east_m')
    gn, ge = F('gps1_north_m'), F('gps1_east_m')
    lon, lat = F('lon'), F('lat')
    src = np.array([r.get('src', '') for r in R])
    i = np.where(src == 'VPS')[0]
    if len(i) < 50:
        return None
    # local frame: the same one the CSV's north_m/east_m live in
    k0 = next(k for k in range(len(R))
              if not math.isnan(lon[k]) and not math.isnan(n0[k]))
    lat0 = lat[k0] - n0[k0] / 110574.0
    mlon = 111320.0 * math.cos(math.radians(lat[k0]))
    lon0 = lon[k0] - e0[k0] / mlon

    fuser = MapFuser()
    fuser.pos = np.array([n0[i[0]], e0[i[0]]], float)
    c, s = math.cos(math.radians(yaw_off)), math.sin(math.radians(yaw_off))
    nfix = 0
    for k in i:
        if not math.isnan(dn[k]):
            a, b = dn[k], de[k]
            fuser.predict(c * a - s * b, s * a + c * b,
                          dt=(None if math.isnan(dt[k]) else dt[k]))
        if use_fixes and fu[k] == 1 and not math.isnan(fl[k]):
            fn = (fla[k] - lat0) * 110574.0
            fe = (fl[k] - lon0) * mlon
            if fuser.update(fn, fe, t=t[k]):
                nfix += 1
    end = fuser.pos
    err = math.hypot(end[0] - gn[i[-1]], end[1] - ge[i[-1]])
    return err, nfix, len(i)


if __name__ == "__main__":
    print("  %-11s %9s %10s %10s %9s" % ("run", "fixes", "yaw 0 (old)",
                                         "yaw -4.42", "fark"))
    for run_ in sys.argv[1:] or ['vps_00406', 'vps_00411', 'vps_00418']:
        p = f'/data/DRONE/flight_logs/vps/{run_}/{run_}.csv'
        a = run(p, 0.0)
        b = run(p, -4.42)
        if a is None or b is None:
            print(f"  {run_}: kam data")
            continue
        print("  %-11s %9d %9.1f m %9.1f m %+8.1f m" % (
            run_, a[1], a[0], b[0], b[0] - a[0]))
