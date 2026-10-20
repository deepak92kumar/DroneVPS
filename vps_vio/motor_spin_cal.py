#!/usr/bin/env python3
"""Find MOT_SPIN_ARM and MOT_SPIN_MIN by spinning each motor. PROPS OFF.

*** THIS SPINS THE MOTORS. TAKE THE PROPELLERS OFF FIRST. ***

Why. Both values came across from the quadcopter this flight controller used
to fly and were never measured on the Tarot MT4008 340 kV motors with 16x5
props: SPIN_ARM 0.10 and SPIN_MIN 0.15 are ArduPilot's generic defaults, and
the 15 Oct quad parameter file has exactly the same two numbers.

What they mean, from ArduPilot's own parameter text: both are "a number from
0 to 1 in the entire output range", and the output range is MOT_PWM_MIN to
MOT_PWM_MAX. On this aircraft that is 1000-2000, so

    MOT_SPIN_ARM = (pwm - 1000) / 1000

which is why this tool sweeps PWM directly rather than percent - percent
would need an assumption about what it is a percent of, and a wrong
assumption would be written into the aircraft without anyone noticing.

  SPIN_ARM is where the motors just start when armed. Too low and a motor
  sits silent while the others turn, which on take-off is a flip. Too high
  and the aircraft is lively the instant it arms.

  SPIN_MIN is where thrust actually begins and the motor responds. Too low
  and the mixer has no authority at the bottom of its range - that is the
  one that bites in a fast descent, when the controller asks a motor to go
  quieter than it can and loses the axis.

Method: at each PWM step every motor is run in turn for a moment, and YOU say
whether all six did what was asked. The flight controller cannot see a motor
that did not turn; only the person standing there can.

    python3 motor_spin_cal.py            # measure, then offer to write
    python3 motor_spin_cal.py --dry-run  # show what is set now and stop

Goes through the ROUTER, never /dev/ttyTHS1. Refuses while armed.
"""
from __future__ import annotations
import argparse
import sys
import time

from pymavlink import mavutil

ROUTER = "udpout:127.0.0.1:14650"

# ArduPilot's hexa-X test order (AP_MotorsMatrix::setup_hexa_matrix): the
# trailing field of each add_motor is the TEST order, which is what
# MAV_CMD_DO_MOTOR_TEST counts in - not the output number. Printed so the
# operator can see which arm should be turning and catch a miswired ESC.
TEST_ORDER = {1: ("motor 5", "front-right"), 2: ("motor 1", "right"),
              3: ("motor 4", "rear-right"), 4: ("motor 6", "rear-left"),
              5: ("motor 2", "left"), 6: ("motor 3", "front-left")}

MOTOR_TEST_THROTTLE_PWM = 1


class FC:
    def __init__(self, port):
        self.m = mavutil.mavlink_connection(port, source_system=200)
        self.m.mav.heartbeat_send(6, 8, 0, 0, 0)
        if self.m.wait_heartbeat(timeout=15) is None:
            sys.exit(f"no heartbeat on {port} - FC powered? router running?")
        if bool(self.m.motors_armed()):
            sys.exit("*** ARMED - ABORT. Motor test is for a disarmed "
                     "aircraft with the props off. ***")

    def beat(self):
        self.m.mav.heartbeat_send(6, 8, 0, 0, 0)

    def get(self, name, timeout=6.0):
        self.m.mav.param_request_read_send(
            self.m.target_system, self.m.target_component, name.encode(), -1)
        t0 = time.time()
        while time.time() - t0 < timeout:
            msg = self.m.recv_match(type="PARAM_VALUE", blocking=True, timeout=1)
            if msg is not None and msg.param_id.strip("\x00") == name:
                return float(msg.param_value)
            self.beat()
        return None

    def set(self, name, value, timeout=6.0):
        self.m.mav.param_set_send(
            self.m.target_system, self.m.target_component, name.encode(),
            float(value), mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
        t0 = time.time()
        while time.time() - t0 < timeout:
            msg = self.m.recv_match(type="PARAM_VALUE", blocking=True, timeout=1)
            if msg is not None and msg.param_id.strip("\x00") == name:
                return float(msg.param_value)
            self.beat()
        return None

    def spin(self, which, pwm, seconds):
        """One motor, one PWM, for a moment. Returns the FC's own reply."""
        self.m.mav.command_long_send(
            self.m.target_system, self.m.target_component,
            mavutil.mavlink.MAV_CMD_DO_MOTOR_TEST, 0,
            which, MOTOR_TEST_THROTTLE_PWM, pwm, seconds, 0, 0, 0)
        t0 = time.time()
        while time.time() - t0 < 3:
            msg = self.m.recv_match(type=["COMMAND_ACK", "STATUSTEXT"],
                                    blocking=True, timeout=1)
            if msg is None:
                self.beat()
                continue
            if msg.get_type() == "COMMAND_ACK" and \
                    msg.command == mavutil.mavlink.MAV_CMD_DO_MOTOR_TEST:
                return msg.result
            if msg.get_type() == "STATUSTEXT" and not msg.text.startswith("SRC="):
                print(f"        FC: {msg.text}")
        return None


def ask(q):
    while True:
        a = input(q).strip().lower()
        if a in ("y", "yes"):
            return True
        if a in ("n", "no"):
            return False
        if a in ("q", "quit", "stop"):
            raise KeyboardInterrupt


def sweep(fc, pwm_lo, pwm_hi, step, hold, label, question):
    """Walk PWM up until the operator says every motor did the thing."""
    pwm = pwm_lo
    while pwm <= pwm_hi:
        # Printed in Mission Planner's units too. MP's motor test asks for a
        # PERCENT, and ArduCopter turns it into pwm_min + (pwm_max-pwm_min) *
        # pct/100 (motor_test.cpp) - the same span MOT_SPIN_ARM is a fraction
        # of. So MP's "14%" is PWM 1140 is SPIN_ARM 0.14, and the pilot who
        # has already found the number in Mission Planner can say so directly.
        print(f"\n  --- {label}: PWM {pwm}  =  {(pwm - 1000) / 10.0:.0f}% in "
              f"Mission Planner  =  SPIN value {(pwm - 1000) / 1000.0:.3f} ---")
        for which in range(1, 7):
            out, where = TEST_ORDER[which]
            print(f"      {which}/6  {out:<8} {where:<12}", end="", flush=True)
            r = fc.spin(which, pwm, hold)
            # None is "no acknowledgement seen", not "refused". The FC
            # interleaves its own chatter with the ack and the first version
            # printed "FC refused" over tests that had plainly run - the
            # operator could see the motor turn while the screen said it had
            # not. Only a non-zero result is a refusal.
            print(" ok" if r in (0, None) else f" FC REFUSED (result {r})")
            if r not in (0, None):
                print("\n      The flight controller refused the motor test. "
                      "Usual reasons: the\n      safety switch is still on, "
                      "or something else has the aircraft armed.")
                return None
            time.sleep(hold + 0.4)
        if ask(f"      {question} [y/n/q] "):
            return pwm
        pwm += step
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=ROUTER)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--start", type=int, default=1100,
                    help="first PWM to try. 1100 = 10%% in Mission Planner; "
                         "below that almost nothing turns on a 4008 with a "
                         "16in prop, and sweeping from 1030 only spends the "
                         "operator's patience.")
    ap.add_argument("--stop", type=int, default=1260, help="give up above this")
    ap.add_argument("--step", type=int, default=15)
    ap.add_argument("--hold", type=float, default=1.2,
                    help="seconds each motor runs")
    a = ap.parse_args()

    fc = FC(a.port)
    pmin = fc.get("MOT_PWM_MIN") or 1000.0
    pmax = fc.get("MOT_PWM_MAX") or 2000.0
    arm_now = fc.get("MOT_SPIN_ARM")
    min_now = fc.get("MOT_SPIN_MIN")
    print("=" * 68)
    print("MOT_SPIN_ARM / MOT_SPIN_MIN  -  by spinning the motors")
    print("=" * 68)
    print(f"  output range MOT_PWM_MIN {pmin:.0f} .. MOT_PWM_MAX {pmax:.0f}")
    print(f"  MOT_SPIN_ARM {arm_now:.3f}  = PWM {pmin + arm_now * (pmax - pmin):.0f}")
    print(f"  MOT_SPIN_MIN {min_now:.3f}  = PWM {pmin + min_now * (pmax - pmin):.0f}")
    print("  (both inherited from the quadcopter; never measured on 4008s "
          "with 16in props)")
    if a.dry_run:
        print("\n  --dry-run, nothing spun.")
        return 0

    print("\n  *** PROPELLERS OFF. All six. Check each arm by hand. ***")
    print("  Each motor will run for about a second at a time, one after the")
    print("  other, front-right first and then clockwise. Nothing arms the")
    print("  aircraft and the throttle stick does nothing here.")
    if not ask("\n  Are all six propellers OFF? [y/n] "):
        print("  stopped, nothing sent.")
        return 1

    try:
        arm_pwm = sweep(fc, a.start, a.stop, a.step, a.hold,
                        "looking for SPIN_ARM",
                        "did ALL SIX motors start turning?")
        if arm_pwm is None:
            print("\n  No PWM in that range started all six. Either the sweep "
                  "needs to go\n  higher (--stop) or a motor or ESC is not "
                  "responding. Nothing written.")
            return 1
        print(f"\n  all six start at PWM {arm_pwm}")

        min_pwm = sweep(fc, arm_pwm + a.step, a.stop, a.step, a.hold,
                        "looking for SPIN_MIN",
                        "are all six turning STEADILY and smoothly (no "
                        "stutter)?")
        if min_pwm is None:
            min_pwm = arm_pwm + 30
            print(f"\n  nothing in range read as steady; falling back to "
                  f"SPIN_ARM + 30 = {min_pwm}")
    except KeyboardInterrupt:
        print("\n\n  stopped by you. Nothing written.")
        return 1

    new_arm = (arm_pwm - pmin) / (pmax - pmin)
    new_min = (min_pwm - pmin) / (pmax - pmin)
    print("\n" + "=" * 68)
    print(f"  measured:  SPIN_ARM {new_arm:.3f} (PWM {arm_pwm})"
          f"   SPIN_MIN {new_min:.3f} (PWM {min_pwm})")
    print(f"  currently: SPIN_ARM {arm_now:.3f}         SPIN_MIN {min_now:.3f}")
    if new_min <= new_arm:
        print("  *** SPIN_MIN must be above SPIN_ARM. Nothing written. ***")
        return 1
    if not (0.0 <= new_arm <= 0.2 and 0.0 <= new_min <= 0.25):
        print("  *** outside ArduPilot's documented ranges (ARM 0-0.2, "
              "MIN 0-0.25). Nothing written. ***")
        return 1
    if not ask("\n  Write these to the aircraft? [y/n] "):
        print("  nothing written.")
        return 0
    ok = True
    for name, val in (("MOT_SPIN_ARM", new_arm), ("MOT_SPIN_MIN", new_min)):
        got = fc.set(name, round(val, 3))
        print(f"  {name}  ->  {got}" if got is not None
              else f"  *** {name} write not confirmed ***")
        ok = ok and got is not None
    print("\n  done." if ok else "\n  check them in Mission Planner before flying.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
