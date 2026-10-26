"""The lidar zeroes itself on the ground, and stops the moment the aircraft arms.

The TF03 is the 4-20 mA variant read on an ADC pin: the FC computes
dist = (volts - RNGFND1_OFFSET) * RNGFND1_SCALING. SCALING is 72.61 m/V and is
right - one millivolt is 7.3 cm, the 20 Oct hover-minus-ground test gave 4.47 m
against the EKF's 4.45, and on log 246 the lidar fits 0.898 * GPSalt - 0.04 m,
through zero. What moves is the OFFSET. The pin sat at 0.768 V when logs
246/247 were flown and at 0.855 V six days later: 87 mV, which is 6.3 m. On
26 Oct the rangefinder read 6.94 m with the lens 0.26 m above the floor.

Zeroing RNGFND1_OFFSET by hand before every flight was tried and the pilot
rejected it - three terminal commands in a field is not a procedure, and a
parameter rewritten every boot is a worse thing to own than a number computed
here. So the FC parameter is left alone and the zero is measured in the node.

Two properties this locks:

ONE: it tracks right up to arming and freezes there. The zero wanders both
ways - 0.298 -> 0.391 m in 65 s on 26 Oct, then down to 0.1-0.2 over the next
few minutes - so what matters is its value at lift-off, not at boot.

TWO: a stale barometric bias cannot survive on the ground. ALT_BIAS stops
updating below ALT_BIAS_LIDAR_MIN_M but keeps its last value, and the node was
found holding -5.81 m and reporting 5.5 m of altitude while the aircraft sat on
the floor. Sitting disarmed on the ground now clears it.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "vps_vio"))

FAILED = []


def check(label, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}   {detail}")
    if not ok:
        FAILED.append(label)


print(__doc__)


class Node:
    """Just the zeroing, lifted off VpsNode so the test needs no camera."""
    import collections as _c
    from live_node import VpsNode as _V
    RNGFND_CLEARANCE_M = _V.RNGFND_CLEARANCE_M
    RNGFND_ZERO_N = _V.RNGFND_ZERO_N
    RNGFND_GROUND_ALT_M = _V.RNGFND_GROUND_ALT_M
    _rngfnd_zero = _V._rngfnd_zero

    def __init__(self):
        import collections
        self._rz_hist = collections.deque(maxlen=self.RNGFND_ZERO_N)
        self._rngfnd_zero_m = 0.0
        self._alt_bias_m = 0.0
        self._alt_bias_t = None


# The reading seen on 26 Oct with the lens 0.26 m above the floor.
GROUND_READING = 6.94
TRUE = 0.26

n = Node()
for _ in range(20):
    z = n._rngfnd_zero(dict(rngfnd=GROUND_READING, alt=0.0, armed=False))
print(f"  sitting disarmed, lidar reads {GROUND_READING} m, truth {TRUE} m")
check("the zero it learns is the whole error",
      abs(z - (GROUND_READING - TRUE)) < 1e-6, f"{z:.3f} m")
check("so the corrected reading is the truth",
      abs((GROUND_READING - z) - TRUE) < 1e-6, f"{GROUND_READING - z:.3f} m")

# It tracks while still on the ground: the pin wanders and the zero follows.
for _ in range(40):
    z2 = n._rngfnd_zero(dict(rngfnd=7.40, alt=0.0, armed=False))
check("it keeps tracking while disarmed on the ground",
      abs(z2 - (7.40 - TRUE)) < 1e-6, f"{z2:.3f} m after the pin moved 0.46 m")

# And freezes the moment it arms, however wrong the air reading looks.
frozen = z2
for _ in range(40):
    z3 = n._rngfnd_zero(dict(rngfnd=55.0, alt=60.0, armed=True))
check("armed and airborne, the zero is frozen", z3 == frozen, f"{z3:.3f} m")

# Airborne but somehow disarmed must not re-learn either: the altitude gate.
for _ in range(40):
    z4 = n._rngfnd_zero(dict(rngfnd=55.0, alt=60.0, armed=False))
check("altitude alone stops it, even if the armed flag is wrong",
      z4 == frozen, f"{z4:.3f} m")

# A MISSING armed flag must not read as disarmed.
n_m = Node()
for _ in range(20):
    zm = n_m._rngfnd_zero(dict(rngfnd=6.94, alt=0.0))      # no 'armed' key
check("a missing armed flag does not count as disarmed", zm == 0.0, f"{zm}")

# A stale barometric bias cannot survive sitting on the ground.
n2 = Node()
n2._alt_bias_m = -5.81          # what the node was found holding
n2._alt_bias_t = 123.0
n2._rngfnd_zero(dict(rngfnd=0.30, alt=0.0, armed=False))
check("sitting on the ground clears a stale barometric bias",
      n2._alt_bias_m == 0.0 and n2._alt_bias_t is None,
      f"bias {n2._alt_bias_m}")

# Nothing is learned when the lidar says nothing.
n3 = Node()
for _ in range(10):
    z5 = n3._rngfnd_zero(dict(rngfnd=None, alt=0.0, armed=False))
check("no lidar means no correction, not a wrong one", z5 == 0.0, f"{z5}")

print(f"\n{'FAILED: ' + ', '.join(FAILED) if FAILED else 'all checks passed'}")
sys.exit(1 if FAILED else 0)
