"""The TF03's analog span is what sets its noise, and the frames must be exact.

The TF03 on this aircraft is the 4-20 mA variant on the X7's ADC pin through
150 Ohms, so the FC computes distance = (volts - OFFSET) * SCALING. It ships
with its analog span at the full 180 m: 16 mA across 180 m through 150 Ohms is
2.4 V across 180 m, so ONE MILLIVOLT IS 7.26 CM. Every millivolt of drift
anywhere in that chain is seven centimetres of height.

Measured 26 Oct: 6.94 m with the lens 0.26 m above the floor, wandering
0.65 -> 2.9 m in thirty minutes untouched. Not the rail - Vcc sd 0.0058 V,
correlation -0.110. Not the resistor - the pilot's two-distance test gave
+0.24 m of error at 0.26 m and +0.25 m at 1.03 m, which is a constant offset,
where a resistor fault would scale with distance.

The manual's section 4.1 says the span is settable and Table 7 gives the
command. This aircraft never needs the lidar above 50 m, so 180 m of span is
130 m of nothing bought at a 3x noise penalty.

The frames are checked against the ones the manual prints IN FULL, because the
Len byte is the length of the whole frame and not of the payload, and writing
it the other way produces frames the sensor silently ignores - which on a
bench looks exactly like bad wiring.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "vps_vio"))

import tf03_range as T

FAILED = []


def check(label, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}   {detail}")
    if not ok:
        FAILED.append(label)


print(__doc__)

# Frames the manual prints in full, Table 7.
for name, got, want in (
        ("Save settings", T.save(), "5A04116F"),
        ("System reset", T.frame(0x02), "5A040260"),
        ("Obtain firmware version", T.frame(0x01), "5A04015F"),
        ("Restore default settings", T.frame(0x10), "5A04106E")):
    check(f"{name} matches the manual", got.hex().upper() == want,
          f"{got.hex(' ').upper()} vs {want}")

# The one we actually send.
u = T.set_upper_limit(6000)
check("upper limit 60 m is 5A 06 8C 70 17 73",
      u.hex().upper() == "5A068C701773", u.hex(" ").upper())
check("its Len byte is the whole frame, not the payload", u[1] == len(u),
      f"Len {u[1]}, frame {len(u)} bytes")
check("its checksum is the low byte of the sum of the rest",
      u[-1] == (sum(u[:-1]) & 0xFF))
check("the distance is little-endian centimetres",
      u[3] | (u[4] << 8) == 6000, f"{u[3] | (u[4] << 8)} cm")

# The scaling that has to follow it, or the aircraft reads nonsense.
print()
for span, want in ((180.0, 75.00), (60.0, 25.00), (30.0, 12.50)):
    sc = T.scaling_for(span)
    print(f"        span {span:5.0f} m  ->  SCALING {sc:6.2f} m/V  "
          f"(1 mV = {sc * 0.1:.2f} cm)")
    check(f"span {span:.0f} m gives SCALING {want}", abs(sc - want) < 0.01,
          f"{sc:.2f}")

check("60 m is 2.9x finer than the 180 m it ships with",
      abs(T.scaling_for(180.0) / T.scaling_for(60.0) - 3.0) < 0.01)

print(f"\n{'FAILED: ' + ', '.join(FAILED) if FAILED else 'all checks passed'}")
sys.exit(1 if FAILED else 0)
