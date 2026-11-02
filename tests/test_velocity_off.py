"""Velocity is off in GPS_INPUT, and the config is what turns it off.

Log 116, 2 Nov. GPS_INPUT carries a speed accuracy and live_node declares it
honestly - 0.6 + 0.048 x altitude + 2.0 x age - which is 1.42 m/s at 17 m
and 4.56 at 83 m. EKF3's calcGpsGoodForFlight() marks a GPS bad once its
low-passed speed accuracy passes about 1.0 m/s, so every altitude above
about 8 m flags our injection as a glitching GPS. On that flight the glitch
outlived the switch back to GPS by 16.6 s, the EKF trusted no position, and
FS_EKF_ACTION=2 took ALT_HOLD five times.

Two things are held here: the setting is OFF, and the node actually READS
it. The second is the one that has gone wrong before - map_window_alt_frac
and map_query_shape were both dead for weeks because the node never passed
them and the worker's own default won.
"""
import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "vps_vio"))

fails = []


def check(what, ok, detail=""):
    print(("  ok   " if ok else "  FAIL ") + what
          + (f"   [{detail}]" if detail else ""))
    if not ok:
        fails.append(what)


import config as C  # noqa: E402

cfg = C.VpsConfig()
print("the setting")
check("send_velocity is OFF", cfg.send_velocity is False,
      str(cfg.send_velocity))

print("\nthe node reads it")
src = open(os.path.join(ROOT, "vps_vio", "live_node.py"),
           encoding="utf-8").read()
check("GpsInputSender is given the config, not just the CLI flag",
      'getattr(_vps_config(), "send_velocity", True)' in src)
check("the CLI --no-send-vel can still force it off",
      "bool(a.send_vel) and bool(" in src)
check("what it is doing is printed at start", 'VELOCITY: ' in src)

print("\nthe sender honours it")
tree = ast.parse(src)
vf = next(n for n in ast.walk(tree)
          if isinstance(n, ast.FunctionDef) and n.name == "velocity_fields")
# The docstring and `ign = self.IGN_VEL_VERT` come first; what matters is
# that the FIRST branch in the function is the send_velocity refusal, so
# nothing else can decide to send a velocity before it is consulted.
first_if = next((n for n in vf.body if isinstance(n, ast.If)), None)
check("the FIRST branch in velocity_fields is the send_velocity refusal",
      first_if is not None and "send_velocity" in ast.dump(first_if.test))
check("and it sets both ignore flags when it does",
      "IGN_VEL_HORIZ | self.IGN_SPEED_ACC" in src)

print("\nposition is untouched - that is the whole point")
check("horiz_accuracy still carries the fuser's sigma",
      "float(sigma_m), self.VERT_ACC_M" in src)
check("fix_type is still 3", "week_ms, week, 3," in src)

print("\nALL PASS" if not fails else f"\n{len(fails)} FAILED: {', '.join(fails)}")
sys.exit(1 if fails else 0)
