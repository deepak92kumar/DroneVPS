#!/usr/bin/env python3
"""Set the TF03's analog output span, which is what makes its noise small.

WHY THIS EXISTS

The TF03 on this aircraft is the 4-20 mA variant read on the X7's ADC pin
through a 150 Ohm resistor, so the flight controller computes

    distance = (volts - RNGFND1_OFFSET) * RNGFND1_SCALING

The sensor ships with its analog span set to its FULL range, 180 m. Sixteen
milliamps across 180 m through 150 Ohms is 2.4 V across 180 m, so one
millivolt is 7.26 cm and every millivolt of drift anywhere in that chain -
the sensor's own 4 mA point, the resistor, the ADC - is seven centimetres of
reported height. Measured on 26 Oct: the reading sat at 6.94 m with the lens
0.26 m above the floor, and wandered 0.65 -> 2.9 m in thirty minutes while
nothing moved. The board's 5 V rail was ruled out by measurement (sd 0.0058 V,
correlation with the reading -0.110), and the pilot's two-distance test ruled
out the resistor: the error was +0.24 m at 0.26 m and +0.25 m at 1.03 m, so
it is a constant offset and not a scale.

The manual's section 4.1 says the span is settable, and Table 7 gives the
command. This aircraft never needs the lidar above 50 m - that is what
VpsNode.ALT_BIAS_LIDAR_MAX_M says - so 180 m of span is 130 m of nothing,
bought with a 3x noise penalty.

    span 180 m    1 mV = 7.26 cm    SCALING 72.61 (what flew until now)
    span  60 m    1 mV = 2.50 cm    SCALING 25.00
    span  30 m    1 mV = 1.25 cm    SCALING 12.50

WIRING, AND WHY THIS IS NOT AUTOMATIC

"The two interfaces, Analog Signal and UART, cannot work simultaneously.
Normally, the UART interface can only be used to configure TF03." So this
needs a USB-TTL adapter on the TF03's own UART - 115200 8N1 - and it is a
one-off bench job. The analog wiring stays exactly as it is.

AFTER RUNNING IT, set RNGFND1_SCALING to the figure this prints. The span and
the scaling are two halves of one number and the aircraft reads nonsense if
only one of them changes.

    python3 tf03_range.py --port /dev/ttyUSB0 --span-m 60          # show
    python3 tf03_range.py --port /dev/ttyUSB0 --span-m 60 --apply  # write
"""
import argparse
import sys

RESISTOR_OHM = 150.0        # the sampling resistor on this airframe
MA_LOW, MA_HIGH = 4.0, 20.0  # forward mode: 4 mA is 0 m, 20 mA is the span


def frame(cmd_id: int, payload: bytes = b"") -> bytes:
    """A TF03 command frame: 0x5A, Len, ID, payload, checksum.

    Len is the length of the WHOLE frame, not of the payload - header, Len,
    ID, payload and checksum, so 4 + len(payload). Checked against the two
    frames the manual prints in full: 'Save settings' is 5A 04 11 6F and
    'System reset' is 5A 04 02 60. Writing it the other way produces frames
    the sensor ignores, which on a one-wire bench setup looks exactly like
    bad wiring.
    """
    body = bytes([0x5A, 4 + len(payload), cmd_id]) + payload
    return body + bytes([sum(body) & 0xFF])


def set_upper_limit(cm: int) -> bytes:
    """Table 7, 'Set the upper limit distance of analog signal'. Unit cm."""
    return frame(0x8C, bytes([cm & 0xFF, (cm >> 8) & 0xFF]))


def save() -> bytes:
    """Table 7, 'Save settings'. Without it the span is lost on power-off."""
    return frame(0x11)


def _self_test():
    """The manual's own frames, as printed, so a typo cannot ship."""
    assert save() == bytes.fromhex("5A04116F"), save().hex()
    assert frame(0x02) == bytes.fromhex("5A040260"), frame(0x02).hex()
    assert frame(0x01) == bytes.fromhex("5A04015F"), frame(0x01).hex()
    assert frame(0x10) == bytes.fromhex("5A04106E"), frame(0x10).hex()


def scaling_for(span_m: float) -> float:
    """The RNGFND1_SCALING that matches a span, in metres per volt."""
    volts = (MA_HIGH - MA_LOW) / 1000.0 * RESISTOR_OHM
    return span_m / volts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyUSB0",
                    help="the USB-TTL adapter on the TF03's UART")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--span-m", type=float, default=60.0,
                    help="what 20 mA should mean, in metres (default 60)")
    ap.add_argument("--apply", action="store_true",
                    help="actually write it. Without this, nothing is sent.")
    a = ap.parse_args(argv)

    cm = int(round(a.span_m * 100))
    if not (10 <= cm <= 18000):
        print(f"span {a.span_m} m is outside the TF03's 0.1-180 m range")
        return 1
    cmd, sv = set_upper_limit(cm), save()
    sc = scaling_for(a.span_m)
    mv_cm = sc * 0.001 * 100

    print(f"  span          {a.span_m:.0f} m  ({cm} cm)")
    print(f"  upper limit   {' '.join(f'{b:02X}' for b in cmd)}")
    print(f"  save settings {' '.join(f'{b:02X}' for b in sv)}")
    print(f"  one millivolt is now {mv_cm:.2f} cm, against 7.26 cm at 180 m")
    print(f"\n  AFTERWARDS set RNGFND1_SCALING = {sc:.2f}")
    print(f"  (and re-zero: python3 vps_vio/lidar_zero.py --true-m 0.26 --apply)")

    if not a.apply:
        print("\n  report only - pass --apply to send it")
        return 0
    try:
        import serial
    except ImportError:
        print("\n  pyserial is not installed, so this cannot send. The two "
              "frames above are correct; send them with any terminal at "
              f"{a.baud} 8N1.")
        return 2
    try:
        with serial.Serial(a.port, a.baud, timeout=1.0) as s:
            for label, b in (("upper limit", cmd), ("save", sv)):
                s.write(b)
                r = s.read(16)
                print(f"  sent {label}: {' '.join(f'{x:02X}' for x in b)}  "
                      f"-> {' '.join(f'{x:02X}' for x in r) if r else 'no reply'}")
    except Exception as e:
        print(f"\n  could not open {a.port}: {e}")
        return 2
    print("\n  written. Now set RNGFND1_SCALING and re-zero.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
