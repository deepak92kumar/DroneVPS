"""Put SCR_USER1/2 at where the receiver says the aircraft is standing.

SCR_USER1 is latitude and SCR_USER2 is longitude, both in degrees - that is
the order preflight prints and the order live_node reads them. They are
float32, so about 0.2 m of resolution at this latitude, which is well under
the receiver's own error.

The position is averaged over several seconds of GPS_RAW_INT, not taken from
one sample and not from GLOBAL_POSITION_INT: the raw receiver is what the
origin is supposed to mean, and one sample of it moves by the drift this
aircraft has been measuring all day.
"""
import sys, time
import numpy as np
from pymavlink import mavutil

APPLY = "--apply" in sys.argv
m = mavutil.mavlink_connection("udpout:127.0.0.1:14650",
                               source_system=250, source_component=191)
for _ in range(6):
    m.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS,
                         mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
    time.sleep(0.3)
if m.wait_heartbeat(timeout=20) is None:
    sys.exit("NO HEARTBEAT - kuch nahi badla")
print("FC sys %d comp %d" % (m.target_system, m.target_component), flush=True)

lat, lon, sats, hdop = [], [], [], []
t0 = time.time()
while time.time() - t0 < 12.0:
    r = m.recv_match(type="GPS_RAW_INT", blocking=True, timeout=2)
    if r is None:
        m.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS,
                             mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
        continue
    if r.fix_type < 3:
        continue
    lat.append(r.lat / 1e7); lon.append(r.lon / 1e7)
    sats.append(r.satellites_visible); hdop.append(r.eph / 100.0)
if len(lat) < 10:
    sys.exit("GPS_RAW_INT ke %d 3D samples - bahut kam, kuch nahi badla" % len(lat))
LAT, LON = float(np.mean(lat)), float(np.mean(lon))
spread = float(np.max(np.hypot((np.array(lat) - LAT) * 111320.0,
                               (np.array(lon) - LON) * 111320.0
                               * np.cos(np.radians(LAT)))))
print("  %d samples, %d sats, HDop %.2f" % (len(lat), int(np.median(sats)),
                                            float(np.median(hdop))))
print("  mean  %.7f N  %.7f E   (samples %.2f m ke andar bikhre)" % (LAT, LON, spread))

def read(name):
    for _ in range(5):
        m.mav.param_request_read_send(m.target_system, m.target_component,
                                      name.encode(), -1)
        t = time.time()
        while time.time() - t < 2.0:
            r = m.recv_match(type="PARAM_VALUE", blocking=True, timeout=2)
            if r and r.param_id.strip("\x00") == name:
                return float(r.param_value)
    return None

for name, want in (("SCR_USER1", LAT), ("SCR_USER2", LON)):
    before = read(name)
    if before is None:
        print("  %s padha nahi gaya - chhoda" % name); continue
    moved = abs(want - before) * 111320.0
    print("  %s  %.7f -> %.7f   (%.0f m)" % (name, before, want, moved))
    if not APPLY:
        continue
    m.mav.param_set_send(m.target_system, m.target_component, name.encode(),
                         want, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    time.sleep(0.8)
    after = read(name)
    err = abs(after - want) * 111320.0 if after is not None else 9e9
    print("     wapas padha: %.7f   %s" % (after if after is not None else float("nan"),
          "OK (%.2f m ke andar)" % err if err < 1.0 else "*** FAIL ***"))
print("\n%s" % ("likh diya" if APPLY else "dry run - kuch nahi likha, --apply se lagega"))
