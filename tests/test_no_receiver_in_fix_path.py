"""The map fix path never reads the receiver. The pilot's standing rule.

"vps ke time par kuch bhi ho jaye GPS ka data nahi aana chiye."

The M9N appears in three places and none of them is an input to the estimate:

  * the ORIGIN, at the moment the pilot hands over to the VPS. Where the GPS
    is, that is where the VPS starts - and after that the origin does not move.
  * gps1_hacc, as that origin's declared accuracy, at the same moment.
  * g1_north / g1_east / g1_err in the LOG, which nothing reads back. That is
    a recording so a flight can be scored on the ground afterwards.

This matters for a second reason the pilot asked about: who tells each matcher
how wrong it is. Nothing does, in flight - the aircraft has no truth there.
Every error figure behind the trust order and the altitude bands was measured
offline, on the ground, against the M9N in a recorded log. So those thresholds
are FIXED CONSTANTS chosen from replay, never learned or adjusted in the air,
and this file is what stops the receiver quietly becoming an input to them.
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vps_vio"))

fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}   {detail}")
    if not ok:
        fails.append(name)


HERE = os.path.dirname(__file__)
LN = open(os.path.join(HERE, "..", "vps_vio", "live_node.py")).read()
WK = open(os.path.join(HERE, "..", "vps_vio", "map_fix_worker.py")).read()
MF = open(os.path.join(HERE, "..", "vps_vio", "map_fuse.py")).read()
MX = open(os.path.join(HERE, "..", "vps_vio", "map_match_xfeat.py")).read()
NC = open(os.path.join(HERE, "..", "vps_vio", "map_match_ncc.py")).read()


def strip_comments(src):
    out = []
    for ln in src.split("\n"):
        s = ln.split("#", 1)[0]
        out.append(s)
    return "\n".join(out)


print("the matchers and the filter cannot see the receiver at all")
for name, src in (("map_fix_worker", WK), ("map_fuse", MF),
                  ("map_match_xfeat", MX), ("map_match_ncc", NC)):
    code = strip_comments(src)
    hits = sorted(set(re.findall(r"\bgps\w*", code)))
    check(f"{name} names no gps symbol", not hits, str(hits))

print("\nthe fix path in live_node does not read the receiver")
# From the fix attempt to the row that records it.
m = re.search(r"fix_used = False(.*?)lon, lat = self\.to_lonlat", LN, re.S)
check("the fix path was found", m is not None)
if m:
    body = strip_comments(m.group(1))
    hits = sorted(set(re.findall(r"\bgps\w*", body)))
    check("and it names no gps symbol", not hits, str(hits))

print("\nthe declared accuracy does not come from the receiver either")
asig = re.search(r"def absolute_sigma_m.*?\n        return sigma", LN, re.S)
check("absolute_sigma_m was found", asig is not None)
if asig:
    body = strip_comments(asig.group(0))
    hits = sorted(set(re.findall(r"\bgps\w*", body)))
    check("and it names no gps symbol", not hits, str(hits))

print("\nwhere the receiver IS used, and that it is only those places")
code = strip_comments(LN)
check("the origin is set from the handover position",
      "node.set_origin(*f[\"gps\"]" in code,
      "where the GPS is, that is where the VPS starts")
# Every line that mentions it, named, so a new use has to be added here
# deliberately instead of slipping past a count.
_hacc = [l.strip() for l in code.split("\n") if "gps1_hacc" in l]
_ok = ('gps1_hacc=s.get("gps1_hacc"),',                         # into the frame
       'boot_sigma = (f.get("gps1_hacc")',                      # origin at boot
       'new_sigma = f.get("gps1_hacc")',                        # re-origin
       'node.set_origin(*f["gps"], sigma_m=f.get("gps1_hacc"))')  # at handover
check("gps1_hacc is the origin's accuracy and nothing else",
      all(l in _ok for l in _hacc) and len(_hacc) == 4,
      "; ".join(l for l in _hacc if l not in _ok) or f"{len(_hacc)} known uses")
check("gps1_raw only reaches the log",
      'est["g1_north"]' in code and "f[\"gps1_raw\"]" in code
      and "fuser.update(" not in re.sub(r'.*gps1_raw.*', '',
                                        code.split('est["g1_north"]')[0][-400:]),
      "it is written for scoring on the ground, not read back")

print("\nthe trust order and the bands are constants, not learned in flight")
for pat, what in ((r"self\._trust\s*=", "the trust list"),
                  (r"ncc_below_m", "NCC's altitude band"),
                  (r"min_alt_m", "LighterGlue's floor")):
    check(f"{what} exists", re.search(pat, WK) or re.search(pat, MX))
check("nothing scores a matcher in flight",
      "err_m" not in strip_comments(WK) and "true_lat" not in strip_comments(WK),
      "the aircraft has no truth in flight; every figure came from replay")

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
