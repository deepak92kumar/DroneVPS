"""Every place that names a map names maps/active.tif, and nothing else.

studio/flightsheet.py repoints that symlink when the pilot loads a sheet:
"Point maps/active.tif at this sheet". It is therefore the only name that
follows the aircraft to a new site. A sheet named in the source, or in the
unit file, is a name that was right on the day it was typed.

1 NOV, FLIGHT 378, IS WHY THIS TEST EXISTS. The pilot loaded bam1.tif in
Studio at 15:36 and Studio did its job - maps/active.tif pointed at
demo/bam1.tif, which covers the site. The systemd unit still passed
--map /data/DRONE/demo_bam1.tif, a symlink to an older sheet whose bounds
(lon 74.0945-74.1195) stop short of the take-off point at 74.1254. So:

  * MapFixWorker.covers() was false for all 764 s,
  * the matcher never ran once - every fix_*/try_* column in all 18225 rows
    of vps_00378.csv is null, not merely zero,
  * the radio read f0 for the whole flight,
  * and both VPS legs flew on flow alone.

Nothing in the logs said "wrong map". The failure looks exactly like a
matcher that cannot match, which is why it cost a flight to find. The two
misleading symlinks - demo_bam1.tif and demo_loc.tif, both pointing at the
old sheet under names that read like the new one - are deleted.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACTIVE = "/data/DRONE/maps/active.tif"

fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}   {detail}")
    if not ok:
        fails.append(name)


def read(rel):
    with open(os.path.join(HERE, rel), encoding="utf-8") as fh:
        return fh.read()


print("the three places that choose a map for a flight")
live = read("vps_vio/live_node.py")
check("live_node's DEFAULT_MAP is the symlink",
      f'DEFAULT_MAP = "{ACTIVE}"' in live,
      "it was bhh.tif, 36 km from where this aircraft flies")
check("preflight checks that same path",
      f'def sheet_covers(lon, lat, path="{ACTIVE}")' in read("vps_vio/preflight.py"),
      "a preflight that reads a different sheet than the node opens is a "
      "check that passes while the flight gets no fixes")
check("the unit passes it",
      f"--map {ACTIVE}" in read("systemd/drone-vps.service"),
      "this is the line that cost flight 378")

print("\nand analysis scores against the sheet that flew")
check("analyse_flight's MAP", f'MAP = "{ACTIVE}"' in read("vps_vio/analyse_flight.py"))

print("\nno source file still names a specific sheet for flying")
pat = re.compile(r"^(?!\s*#).*?[\"'](/data/DRONE/(?!maps/active\.tif)[^\"']*\.tif)[\"']")
for rel in ("vps_vio/live_node.py", "vps_vio/preflight.py",
            "vps_vio/analyse_flight.py", "systemd/drone-vps.service"):
    hits = [m.group(1) for m in (pat.match(l) for l in read(rel).splitlines()) if m]
    check(f"{rel} names no other sheet in code", not hits, str(hits))

print("\nthe deleted symlinks stay deleted")
for gone in ("demo_bam1.tif", "demo_loc.tif"):
    check(f"{gone} is gone", not os.path.lexists(os.path.join(HERE, gone)),
          "a name that reads like the right map and is not")

print("\nthe symlink itself")
if not os.path.lexists(ACTIVE):
    print(f"  SKIP  {ACTIVE} not present on this machine")
else:
    check("it is a symlink", os.path.islink(ACTIVE),
          "Studio replaces it with os.symlink; a real file here would be "
          "overwritten by the next load")
    check("and it resolves to a sheet that exists", os.path.exists(ACTIVE),
          os.path.realpath(ACTIVE))

print()
if fails:
    print(f"{len(fails)} FAILED: " + ", ".join(fails))
    sys.exit(1)
print("ALL PASS")
