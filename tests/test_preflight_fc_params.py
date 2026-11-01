"""The parameters preflight insists on, and why each one is in the list.

A flight controller swap resets everything silently. The X7 became an X7+ on
31 Oct, the new board came up on ArduPilot defaults, and on 1 Nov - two
flights later - a full read showed what had quietly gone:

    EK3_SRC1_VELZ       3        should be 0 (NONE)
    INS_HNTCH_ENABLE    0        the harmonic notch, off
    RNGFND1_TYPE        0        no rangefinder configured at all
    ATC_RAT_RLL_P/I     0.135    the tuning that flew was 0.09
    ATC_RAT_RLL_D       0.0036   flew 0.0030
    DISARM_DELAY        10       flew 2

None of it announced itself. flight_logs/00000111.BIN has zero RFND rows and
every rngfnd_m in every run since the swap is NaN, and nothing said so.

EXPECT already covered the tuning. What it could not do is name a CAUSE:
when INS_HNTCH_ENABLE and RNGFND1_TYPE read 0, ArduPilot does not create
INS_HNTCH_FREQ or RNGFND1_RMETRIC at all, so those checks report "absent" and
the pilot is told a symptom. Both parents are now checked by name.

FC_WANTS is OUR list, not a vendor's. Where it differs from the Stride
Dynamics integration guide the difference is deliberate - GPS_AUTO_SWITCH
above all, where their UseBest would let the flight controller choose the
receiver and ours must obey the pilot's RC6 and nothing else.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vps_vio"))
import preflight  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}   {detail}")
    if not ok:
        fails.append(name)


W = {name: (want, blocker, why) for name, want, blocker, why in preflight.FC_WANTS}

print("the six that a board swap takes, and did")
for name, want in (("EK3_SRC1_VELZ", 0), ("INS_HNTCH_ENABLE", 1),
                   ("RNGFND1_TYPE", 1)):
    check(f"{name} is checked, wanting {want}",
          name in W and abs(W[name][0] - want) < 1e-9,
          f"found {W[name][0]}" if name in W else "NOT CHECKED")

print("\nthe ones that cost the flight are blockers, not warnings")
for name in ("GPS2_TYPE", "EK3_SRC1_POSXY", "EK3_SRC1_VELXY", "EK3_SRC1_VELZ",
             "EK3_SRC1_POSZ", "EK3_SRC1_YAW", "GPS_AUTO_SWITCH",
             "INS_HNTCH_ENABLE", "RNGFND1_TYPE"):
    check(f"{name} blocks", name in W and W[name][1] is True)

print("\nand the ones that are judgement are warnings")
for name in ("ATC_ANGLE_MAX", "GPS_BLEND_MASK", "RC6_OPTION"):
    check(f"{name} warns", name in W and W[name][1] is False)

print("\nwhere we differ from the vendor, the reason is written down")
want, blocker, why = W["GPS_AUTO_SWITCH"]
check("GPS_AUTO_SWITCH wants 0, not UseBest", abs(want - 0) < 1e-9)
check("and says why in its own text",
      "switch means switch" in why and "GPS_PRIMARY" in why,
      "a value that disagrees with a vendor guide needs its reason beside it")

print("\nEK3_SRC1_VELZ's reason names the code that makes it true")
check("it points at IGN_VEL_VERT", "IGN_VEL_VERT" in W["EK3_SRC1_VELZ"][2])
src = open(os.path.join(os.path.dirname(__file__), "..", "vps_vio",
                        "live_node.py"), encoding="utf-8").read()
check("and live_node really does set it on every packet",
      "ign = self.IGN_VEL_VERT" in src,
      "if this ever stops being true, VELZ = NONE stops being right")

print("\nevery entry has a reason, and it is a sentence")
for name, want, blocker, why in preflight.FC_WANTS:
    check(f"{name} explains itself", len(why) > 40, f"{len(why)} chars")

print("\nthe checker runs against a stub FC without touching anything")


class FakeFC:
    def __init__(self, vals):
        self.vals = vals
        self.asked = []

    def get(self, name, timeout=None):
        self.asked.append(name)
        return self.vals.get(name)


before = len(preflight.blockers)
fc = FakeFC({n: w for n, w, _, _ in preflight.FC_WANTS})
preflight.fc_parameters(fc)
check("a correct FC adds no blocker", len(preflight.blockers) == before)
check("and it asked for every one", len(fc.asked) == len(preflight.FC_WANTS),
      f"{len(fc.asked)} of {len(preflight.FC_WANTS)}")

bad = FakeFC({n: (w + 1) for n, w, _, _ in preflight.FC_WANTS})
preflight.fc_parameters(bad)
nblock = sum(1 for _, _, b, _ in preflight.FC_WANTS if b)
check("a wrong FC blocks on every blocker",
      len(preflight.blockers) - before == nblock,
      f"{len(preflight.blockers) - before} of {nblock}")

print()
if fails:
    print(f"{len(fails)} FAILED: " + ", ".join(fails))
    sys.exit(1)
print("ALL PASS")
