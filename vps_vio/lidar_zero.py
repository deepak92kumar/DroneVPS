#!/usr/bin/env python3
"""Zero the analog rangefinder on the ground. GROUND ONLY, DISARMED ONLY.

Why this exists, and why it has to be run again every session
-------------------------------------------------------------
The TF03-180 on this aircraft is the 4-20 mA variant: the distance arrives as
a current, a 150 Ohm resistor turns it into 0.6-3.0 V, and the FC reads that on
an ADC pin as `dist = (volts - RNGFND1_OFFSET) * RNGFND1_SCALING`. With 180 m
of range across 2.4 V, **one millivolt is seven centimetres**. So the SCALE is
fine - 20 Oct measured hover minus ground as 4.47 m against the EKF's 4.45 -
and the ZERO is not. Three readings on the same ground on 20 Oct:

    log 00000215   0.82 m      board Vcc 5.088
    log 00000216   2.53 m      board Vcc 5.101
    live, 12:45    0.96 m      board Vcc 5.094

True clearance is 0.26 m (RNGFND1_GNDCLR). Vcc barely moves, so the 5 V rail is
not the cause; this is the sensor's own 4 mA zero plus the ADC path, and
+-1-2 m is inside what this chain can do. The UART on this TF03 variant carries
configuration only, not data, so there is no digital reading to switch to.

What a wrong zero actually costs
--------------------------------
Altitude is fine: EK3 uses the rangefinder relative to the take-off reference,
so a constant offset cancels. What does NOT cancel is ArduPilot's land
detector, which needs `rangefinder < 2 m` in absolute terms before it will
declare the aircraft landed:

    motor_at_lower_limit && throttle_mix_at_min && !large_angle_request
    && !large_angle_error && accel_stationary && descent_rate_low
    && rangefinder_check && WoW

On 20 Oct the zero sat at 2.53 m, `rangefinder_check` never passed, LDET.Count
never left 0, `land_complete` never became true, DISARM_DELAY never fired and
the battery had to be pulled - while LOITER's position controller wound its
integrator up to 13.9 degrees of lean and pushed the motors from 1150 to 1491
against the ground for 23 seconds. That is what this script prevents.

Usage
-----
    python3 lidar_zero.py                  # just report, change nothing
    python3 lidar_zero.py --apply          # write the new OFFSET
    python3 lidar_zero.py --apply --fix-rmetric
    python3 lidar_zero.py --true-m 0.26    # if it is not on its legs

The aircraft must be sitting on its landing legs on the ground, disarmed. If it
is on a bench or a stand, pass the real sensor-to-floor distance with --true-m,
or the wrong number gets written into the aircraft.

Everything goes through the ROUTER, never /dev/ttyTHS1. The router is the one
owner of that device; a second opener splits the stream with it.
"""
from __future__ import annotations
import argparse
import sys
import time

import numpy as np
from pymavlink import mavutil

ROUTER = "udpout:127.0.0.1:14650"
# A zero this far out is not a calibration, it is a different problem - the
# aircraft is not on the ground, or the sensor is unplugged. Refuse rather than
# write it.
MAX_SANE_READING_M = 8.0
OFFSET_VOLT_RANGE = (0.30, 1.50)


class FC:
    def __init__(self, port: str, timeout: float = 15.0):
        self.m = mavutil.mavlink_connection(port, source_system=200)
        self.m.mav.heartbeat_send(6, 8, 0, 0, 0)
        if self.m.wait_heartbeat(timeout=timeout) is None:
            sys.exit(f"no heartbeat on {port} - is the FC powered and "
                     "drone-router running?")
        if bool(self.m.motors_armed()):
            sys.exit("*** ARMED - ABORT. Zero the lidar disarmed, on the "
                     "ground. ***")
        self.m.mav.request_data_stream_send(
            self.m.target_system, self.m.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1)

    def _keepalive(self):
        # A receive-only client is dropped by the router after 30 s and then
        # goes silently deaf. Everything that listens here heartbeats.
        self.m.mav.heartbeat_send(6, 8, 0, 0, 0)

    def get(self, name: str, timeout: float = 5.0):
        self.m.mav.param_request_read_send(
            self.m.target_system, self.m.target_component, name.encode(), -1)
        t0 = time.time()
        while time.time() - t0 < timeout:
            msg = self.m.recv_match(type="PARAM_VALUE", blocking=True,
                                    timeout=1)
            if msg is not None and msg.param_id.strip("\x00") == name:
                return float(msg.param_value)
            self._keepalive()
        return None

    def set(self, name: str, value: float, timeout: float = 5.0):
        """Write, then read back. A write that is not read back is a guess."""
        self.m.mav.param_set_send(
            self.m.target_system, self.m.target_component, name.encode(),
            float(value), mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
        t0 = time.time()
        while time.time() - t0 < timeout:
            msg = self.m.recv_match(type="PARAM_VALUE", blocking=True,
                                    timeout=1)
            if msg is not None and msg.param_id.strip("\x00") == name:
                return float(msg.param_value)
            self._keepalive()
        return None

    def read_distance(self, seconds: float = 6.0):
        """Mean, spread and sample count of what the FC reports."""
        d, vcc = [], []
        t0 = time.time()
        while time.time() - t0 < seconds:
            msg = self.m.recv_match(
                type=["RANGEFINDER", "DISTANCE_SENSOR", "POWER_STATUS"],
                blocking=True, timeout=1)
            if msg is None:
                self._keepalive()
                continue
            t = msg.get_type()
            if t == "RANGEFINDER":
                d.append(msg.distance)
            elif t == "DISTANCE_SENSOR":
                d.append(msg.current_distance / 100.0)
            elif t == "POWER_STATUS":
                vcc.append(msg.Vcc / 1000.0)
        if not d:
            sys.exit("no rangefinder messages - RNGFND1_TYPE off, or the "
                     "stream was not granted")
        a = np.array(d, dtype=float)
        return a, (float(np.mean(vcc)) if vcc else float("nan"))


# A clamped reading is exactly 0.000, because ArduPilot floors a negative
# analog distance. That is the whole test. An earlier version of this also
# demanded NOISE, and the aircraft disproved it on 21 Oct: the pin sat at
# 0.68194 V steady to a tenth of a millivolt, the search below moved the
# reading 0.640 -> 18.790 m on a straight line of slope 72.60 m/V, and sd read
# 0.000 at every single step - because 72.61 m/V makes one millivolt 7.3 cm and
# the sensor is quieter than one ADC code. Requiring noise rejected a sensor
# that was working perfectly.
CLAMP_MEAN_M = 0.02


def _unclamp(fc, offset, scaling, seconds, step_mv=25.0, tries=12):
    """Walk RNGFND1_OFFSET down until the reading comes off zero.

    Returns (samples, offset_used), or (None, _) with the original offset put
    back. 25 mV is 1.8 m at this scaling - coarse on purpose, because the point
    is to find the voltage, not to be gentle, and the zero is measured properly
    once a reading exists.
    """
    original = offset
    for i in range(1, tries + 1):
        trial = original - i * step_mv / 1000.0
        if trial < OFFSET_VOLT_RANGE[0]:
            break
        got = fc.set("RNGFND1_OFFSET", trial)
        if got is None:
            break
        time.sleep(0.8)
        d, _ = fc.read_distance(max(2.0, seconds / 3.0))
        print(f"    offset {trial:.5f} V  ->  mean {d.mean():.3f} m  "
              f"sd {d.std():.4f}")
        if d.mean() > CLAMP_MEAN_M:
            return d, trial
    fc.set("RNGFND1_OFFSET", original)
    return None, original


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=ROUTER,
                    help="MAVLink endpoint. Default is the router, which is "
                         "the only correct answer on the aircraft.")
    ap.add_argument("--true-m", type=float, default=None,
                    help="real sensor-to-floor distance. Default: "
                         "RNGFND1_GNDCLR read off the aircraft.")
    ap.add_argument("--seconds", type=float, default=6.0,
                    help="how long to average before deciding.")
    ap.add_argument("--apply", action="store_true",
                    help="write the new RNGFND1_OFFSET. Without this the "
                         "script only reports.")
    ap.add_argument("--fix-rmetric", action="store_true",
                    help="also set RNGFND1_RMETRIC=0. A 4-20 mA loop into a "
                         "resistor does not scale with the 5 V rail, so "
                         "dividing by that rail only feeds its noise into the "
                         "height. Changing it MOVES the reading, so the zero "
                         "is measured again afterwards.")
    a = ap.parse_args()

    fc = FC(a.port)
    scaling = fc.get("RNGFND1_SCALING")
    offset = fc.get("RNGFND1_OFFSET")
    gndclr = fc.get("RNGFND1_GNDCLR")
    rmetric = fc.get("RNGFND1_RMETRIC")
    if None in (scaling, offset):
        sys.exit("could not read RNGFND1_SCALING / RNGFND1_OFFSET")
    true_m = a.true_m if a.true_m is not None else (gndclr or 0.26)

    print(f"  RNGFND1_SCALING = {scaling:.3f} m/V   "
          f"({1000.0 / scaling:.1f} mV per metre - a millivolt is "
          f"{scaling / 10.0:.1f} cm)")
    print(f"  RNGFND1_OFFSET  = {offset:.5f} V")
    print(f"  RNGFND1_RMETRIC = {rmetric:.0f}")
    print(f"  true distance   = {true_m:.3f} m "
          f"({'given' if a.true_m is not None else 'RNGFND1_GNDCLR'})")

    if a.fix_rmetric and rmetric and rmetric != 0:
        if not a.apply:
            print("\n  --fix-rmetric needs --apply; reporting only.")
        else:
            got = fc.set("RNGFND1_RMETRIC", 0.0)
            print(f"\n  RNGFND1_RMETRIC {rmetric:.0f} -> {got:.0f}"
                  if got is not None else
                  "\n  *** RNGFND1_RMETRIC write not confirmed ***")
            if got is None or got != 0.0:
                sys.exit("RMETRIC not accepted - stopping before the zero")
            time.sleep(1.0)

    d, vcc = fc.read_distance(a.seconds)
    print(f"\n  reported: n={len(d)}  mean {d.mean():.3f} m  "
          f"sd {d.std():.3f}  min {d.min():.2f}  max {d.max():.2f}"
          f"   board Vcc {vcc:.3f}")

    # IS THE READING CLAMPED? This has to be asked before anything is said
    # about the zero, and until 21 Oct it was not.
    #
    # ArduPilot clamps a negative analog distance to zero. So when the offset
    # has drifted ABOVE the sensor's actual voltage, every reading comes back a
    # dead-flat 0.000 m with sd 0.000 - and the arithmetic below then reads
    # that as "the sensor says 0 m", computes an error of exactly the ground
    # clearance, and writes a correction of a few millivolts. Measured on the
    # bench that evening: true 0.260 m against a clamped 0.000 gives -3.6 mV,
    # when the real voltage could be anywhere below the offset and the error
    # could be a hundred millivolts. Applying it leaves the reading still
    # clamped, and the pilot repeats the whole procedure at the field.
    #
    # A live sensor is noisy: sd 0.121 m with RNGFND1_RMETRIC 1 and 0.027 m
    # after it was set to 0, both far above zero. And RNGFND1_SCALING 72.61
    # makes one millivolt 7.3 cm, so a live sensor resting on a single ADC code
    # could read a constant - but it would read that code's value, not zero.
    # Both tests are needed.
    clamped = float(d.mean()) <= CLAMP_MEAN_M
    if clamped:
        print(f"  *** CLAMPED at {d.mean():.3f} m. "
              f"ArduPilot floors a negative "
              f"analog distance at zero, so RNGFND1_OFFSET {offset:.5f} V is "
              f"ABOVE the sensor's actual voltage and the true error is unknown "
              f"from here, not {1000.0 * (float(d.mean()) - true_m) / scaling:+.1f} mV. ***")
        if not a.apply:
            print("\n  Pass --apply and this will walk the offset DOWN until a "
                  "reading appears, then measure the zero properly.")
            return 0
        d, offset = _unclamp(fc, offset, scaling, a.seconds)
        if d is None:
            sys.exit("\n*** the reading never came off zero. That is not a "
                     "drifted zero - check the sensor's power and wiring. "
                     "RNGFND1_OFFSET has been put back. ***")
        print(f"\n  off the clamp: mean {d.mean():.3f} m  sd {d.std():.3f} "
              f"at RNGFND1_OFFSET {offset:.5f} V")
    else:
        print(f"  noise floor: sd {d.std():.3f} m = "
              f"{1000.0 * d.std() / scaling:.1f} mV of ADC. This is the sensor, "
              f"not the aircraft.")

    err = float(d.mean()) - true_m
    new_offset = offset + err / scaling
    print(f"\n  zero error      = {err:+.3f} m  "
          f"({1000.0 * err / scaling:+.1f} mV)")
    print(f"  RNGFND1_OFFSET    {offset:.5f}  ->  {new_offset:.5f} V")

    if abs(float(d.mean())) > MAX_SANE_READING_M:
        sys.exit(f"\n*** {d.mean():.2f} m is too far from {true_m:.2f} m to be "
                 "a zero error. Is the aircraft on the ground? Is the sensor "
                 "plugged in? Nothing written. ***")
    if not (OFFSET_VOLT_RANGE[0] <= new_offset <= OFFSET_VOLT_RANGE[1]):
        sys.exit(f"\n*** computed offset {new_offset:.4f} V is outside "
                 f"{OFFSET_VOLT_RANGE} - that is not a calibration. Nothing "
                 "written. ***")

    if not a.apply:
        print("\n  (report only - pass --apply to write it)")
        return 0

    got = fc.set("RNGFND1_OFFSET", new_offset)
    if got is None:
        sys.exit("*** write not confirmed by the FC - check it in Mission "
                 "Planner before flying ***")
    print(f"  written and read back: {got:.5f} V")

    time.sleep(1.0)
    d2, _ = fc.read_distance(max(3.0, a.seconds / 2))
    print(f"\n  after: mean {d2.mean():.3f} m (wanted {true_m:.3f})  "
          f"sd {d2.std():.3f}")
    if abs(float(d2.mean()) - true_m) > 0.30:
        print("  *** still off by more than 30 cm - do not trust the land "
              "detector; check the wiring ***")
        return 1
    if float(d2.mean()) >= 2.0:
        print("  *** still 2 m or more: ArduPilot will NOT declare the "
              "aircraft landed and it will not auto-disarm ***")
        return 1
    print("  OK - under 2 m, so the land detector's rangefinder check can "
          "pass and DISARM_DELAY can fire.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
