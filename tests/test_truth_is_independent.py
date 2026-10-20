"""What a VPS flight's error is measured AGAINST, and why it was not independent.

`err_m` and `gps_north_m`/`gps_east_m` read like the GPS. They are not. They
come from GLOBAL_POSITION_INT, which is the EKF's FUSED position - and
`EK3_SRC1_POSXY` is 3, GPS, with our own GPS_INPUT injected as GPS2. So the
moment `GPS_PRIMARY` is 1 and the aircraft is flying ON the VPS, the EKF is
being driven by what we send, and comparing our position against it is
comparing the VPS with itself.

It had never mattered, which is why it survived. Every recording with real
travel so far ran with `src = GPS1`:

    vps_00016   18799 rows, all GPS1   (1339 above 60 m)
    vps_00019   10048 rows GPS1, 4 unknown

and on GPS1 the source gate blocks our injection entirely, so the EKF really
was GPS-driven and the comparison really was independent. The next flight is
the one that breaks it: GPS take-off, VPS at altitude, GPS back before the
descent. Half of it would have been scored against itself.

GPS_RAW_INT is the receiver before the EKF touches it, and it is the only
independent truth this aircraft carries. It now has its own fields and its own
columns, and match_flight prefers them and says which it used.

One honest limit, unchanged: the receiver itself wandered 2.4 to 27.4 m on the
ground (10 Oct), and the surveyed point asked for on three separate days still
has not been taken. Independent is not the same as correct.
"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "vps_vio"))

from match_flight import gps_lonlat, truth_source  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        fails.append(name)


live = open(os.path.join(HERE, "vps_vio", "live_node.py")).read()
rec = open(os.path.join(HERE, "vps_vio", "record_flight.py")).read()

# ------------------------------------------------- it is captured at all

print("the raw receiver is read, separately from the EKF")
check("GPS_RAW_INT's own lat/lon are captured",
      "gps1_lat=lat1, gps1_lon=lon1" in rec)
check("only with a 3D fix, and not at the null island",
      "msg.fix_type >= 3 and msg.lat != 0" in rec,
      "lat 0 is what the receiver reports with nothing")
check("the frame dict carries it apart from the fused position",
      'gps1_raw=((s["gps1_lon"], s["gps1_lat"])' in live)
check("and it is NOT confused with f[\"gps\"]",
      'f.get("gps1_raw")' in live and 'if f["gps"]:' in live,
      "two sources, two comparisons, both logged")

print("\nits own columns, and its own error")
for col in ("gps1_north_m", "gps1_east_m", "gps1_err_m"):
    check(f"{col} is a column", f'"{col}"' in live)
check("the error against the raw receiver is computed",
      'est["g1_err"] = float(np.hypot(' in live)
check("the older pair is still logged, not replaced",
      '"gps_north_m", "gps_east_m", "err_m"' in live,
      "the pre-21-Oct recordings are read with it")

# ------------------------------------------------- the preference, driven

print("\ngps_lonlat prefers the raw receiver - driven, not grepped")
base = dict(north_m="10.0", east_m="20.0", lon="74.6131820", lat="30.8967418")

both = dict(base, gps_north_m="110.0", gps_east_m="20.0",
            gps1_north_m="12.0", gps1_east_m="20.0")
lon_r, lat_r = gps_lonlat(both)
# The raw row is 2 m north of the estimate; the fused row is 100 m north. Which
# one came back is readable straight off the latitude.
dn_raw = (lat_r - float(base["lat"])) * 111320.0
check("with both present it uses the RAW receiver", abs(dn_raw - 2.0) < 0.1,
      f"came back {dn_raw:.2f} m north of the estimate; raw is 2 m, fused is 100 m")

lon_f, lat_f = gps_lonlat(both, prefer_raw=False)
dn_fused = (lat_f - float(base["lat"])) * 111320.0
check("and prefer_raw=False still gives the old answer",
      abs(dn_fused - 100.0) < 0.1, f"{dn_fused:.2f} m")

only_old = dict(base, gps_north_m="110.0", gps_east_m="20.0")
check("a pre-21-Oct row still works, through the fallback",
      gps_lonlat(only_old) is not None)
lon_o, lat_o = gps_lonlat(only_old)
check("and gives the EKF's position, since that is all it has",
      abs((lat_o - float(base["lat"])) * 111320.0 - 100.0) < 0.1)

blank = dict(base, gps_north_m="", gps_east_m="", gps1_north_m="", gps1_east_m="")
check("a row with neither returns None", gps_lonlat(blank) is None)
check("a row with no estimate returns None",
      gps_lonlat(dict(gps1_north_m="1", gps1_east_m="1")) is None)

# A raw fix that drops out mid-flight must fall back per row, not per file.
patchy = dict(base, gps_north_m="110.0", gps_east_m="20.0",
              gps1_north_m="", gps1_east_m="")
check("a row whose raw fix dropped falls back on its own",
      abs((gps_lonlat(patchy)[1] - float(base["lat"])) * 111320.0 - 100.0) < 0.1,
      "the receiver loses lock mid-flight; that is one row, not the whole run")

# ------------------------------------------------- and it says which

print("\nevery offline run states what it measured against")
check("raw is named when present",
      "raw receiver" in truth_source([dict(gps1_north_m="1", gps_north_m="1")]))
check("the fused fallback is named AND qualified",
      "FUSED" in truth_source([dict(gps_north_m="1")])
      and "GPS1" in truth_source([dict(gps_north_m="1")]),
      "a reader has to know it is only independent where src was GPS1")
check("and nothing is named when there is nothing",
      "nothing" in truth_source([dict(north_m="1")]))
check("the matcher prints it on every run",
      "truth for d_gps_m" in open(os.path.join(HERE, "vps_vio",
                                               "match_flight.py")).read())

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
