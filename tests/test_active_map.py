"""The sheet the node flies with comes from one place, and says where from.

Written with the studio's "load this sheet" button, which writes the pointer.
The failure this guards against is the silent one: a pointer that names a
sheet which is not on the machine, resolving quietly to the sheet that has
always flown, and the flight taking off over ground the sheet does not cover.
A take-off point off the sheet means no map fixes for the whole flight and
there is no sign of it in the air except that no fix ever lands.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "vps_vio"))
import active_map  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        fails.append(name)


tmp = tempfile.mkdtemp()
active_map.MAPS_DIR = tmp
active_map.POINTER = os.path.join(tmp, "ACTIVE_MAP")
active_map.LINK = os.path.join(tmp, "active.tif")
sheet = os.path.join(tmp, "a_sheet.tif")
open(sheet, "wb").write(b"not really a tif, but it exists")

print("with no pointer at all")
p, src, note = active_map.resolve()
check("falls back to the sheet that has always flown", p == active_map.FALLBACK, p)
check("and says it fell back", src == "fallback", note)

print("\nafter the studio points it at a cut sheet")
active_map.set_active(sheet)
p, src, note = active_map.resolve()
check("the pointer is used", p == sheet, p)
check("and says so", src == "pointer", note)
check("the symlink points at the same file",
      os.path.realpath(active_map.LINK) == os.path.realpath(sheet))

print("\nwhen the pointer names a sheet that is not on this machine")
os.unlink(sheet)
p, src, note = active_map.resolve()
check("falls back rather than opening nothing", p == active_map.FALLBACK, p)
check("and the reason names the missing file", "not on this machine" in note, note)

print("\nan empty pointer")
open(active_map.POINTER, "w").write("\n")
p, src, note = active_map.resolve()
check("falls back", p == active_map.FALLBACK)
check("and says the pointer is empty", "empty" in note, note)

print("\nclearing it")
active_map.clear()
check("the pointer is gone", not os.path.exists(active_map.POINTER))
check("and so is the symlink", not os.path.lexists(active_map.LINK))

print()
print(f"{'FAILED: ' + ', '.join(fails) if fails else 'all passed'}")
sys.exit(1 if fails else 0)
