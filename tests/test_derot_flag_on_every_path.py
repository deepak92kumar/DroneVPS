"""The derotation flag must be set on EVERY path through step(), not one.

29 Oct. `_use_pp` decides whether the gyro's rotation is handed to the
trackers. It used to be assigned inside the `else` of the frame-age branch:

    if _age > frame_age_max_s:      # stale frame
        d, ntrk = None, 0           # _use_pp never assigned
    else:
        _use_pp = ...
        d, ntrk = self.tracker.step(..., derot_deg=... if _use_pp else None)

    if tracker_b is not None and f.get("img_partner") is not None:
        db, ntrk_b = tracker_b.step(..., derot_deg=... if _use_pp else None)

The second camera's call sits OUTSIDE that branch and read the flag anyway, so
a stale frame arriving while the partner had an image raised

    UnboundLocalError: local variable '_use_pp' referenced before assignment

and the process exited. systemd restarted it five seconds later, which on the
ground is noise and in the air is the camera init and the XFeat load as well -
twenty to thirty seconds with no injection, which is an ALT_HOLD.

It was not rare. vps_stderr.log holds 109 crashes and the last 25 are all this
one; five of those are the build that was flying. It fires when the camera or
the USB link falls behind by more than frame_age_max_s, which is what happens
when the machine is loaded - on 29 Oct the node ran 65 minutes clean and then
crashed four times in eleven while a 50 MB sheet was being written and read.
frame_age_s reached 0.1135 against a 0.12 threshold in the run that died.

So this checks the shape rather than the symptom: the names the partner's call
depends on are assigned at the top level of step(), where every path runs.
A test that drove a real frame through would need a camera; this needs the
source, and it fails for exactly the reason the bug existed.
"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NODE = os.path.join(os.path.dirname(HERE), "vps_vio", "live_node.py")

fails = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name
          + (f"   {detail}" if detail else ""))
    if not cond:
        fails.append(name)


tree = ast.parse(open(NODE).read())
step = next((n for n in ast.walk(tree)
             if isinstance(n, ast.FunctionDef) and n.name == "step"), None)
check("step() is in live_node", step is not None)

# Assigned where every path runs = a statement directly in the function body,
# not buried in an If/For/Try that some path can skip.
top = set()
for stmt in step.body:
    for n in ast.walk(stmt) if isinstance(stmt, ast.Assign) else ():
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            top.add(n.id)

print("\nthe names the second camera's call depends on")
for name in ("_use_pp", "_gr", "_gp"):
    reads = [n.lineno for n in ast.walk(step)
             if isinstance(n, ast.Name) and n.id == name
             and isinstance(n.ctx, ast.Load)]
    stores = [n.lineno for n in ast.walk(step)
              if isinstance(n, ast.Name) and n.id == name
              and isinstance(n.ctx, ast.Store)]
    check(f"{name} is assigned at the top level of step()", name in top,
          f"assigned at {stores}, read at {reads}")
    check(f"{name} is read only after it is assigned",
          bool(stores) and bool(reads) and min(reads) > min(stores),
          f"first store {min(stores) if stores else None}, "
          f"first read {min(reads) if reads else None}")

print("\nand the partner's call still reads it, so the check means something")
src = open(NODE).read()
check("two trackers are given derot_deg=... if _use_pp else None",
      src.count("derot_deg=((_gr, _gp) if _use_pp else None))") == 2,
      str(src.count("derot_deg=((_gr, _gp) if _use_pp else None))")))

print("\nALL PASS" if not fails else f"\n{len(fails)} FAILED: {', '.join(fails)}")
sys.exit(1 if fails else 0)
