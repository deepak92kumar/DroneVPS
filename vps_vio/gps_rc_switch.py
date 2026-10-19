#!/usr/bin/env python3
"""Flip between the real GPS and the VPS from a switch on the transmitter.

ArduPilot has no auxiliary function for choosing the primary GPS, so this
watches an RC channel and sets GPS_PRIMARY itself. A Lua script could do the
same on the flight controller, but SCR_ENABLE is deliberately off and the
companion is already reading this link.

    ch low  (~1500)  ->  GPS_PRIMARY = 0   real GPS
    ch high (~1950)  ->  GPS_PRIMARY = 1   VPS

Three guards, because handing a switch this much authority deserves them.

It refuses a channel that already carries an ArduPilot auxiliary function. The
first channel picked here was RC11, and RC11_OPTION is 18 - Land. Flipping it
to change GPS source would have landed the aircraft. Nothing about reading a
channel tells you what else is listening to it, so the parameter is checked.

It refuses to move onto the VPS unless GPS2 is actually reporting a 3D fix.
The flight controller will not check that for you - on the GPS_INPUT path it
trusts whatever arrives - so a switch flipped while the VPS is unhealthy would
hand navigation to nothing at all. The refusal is announced on the GCS.

And it refuses to do anything while armed unless --allow-armed is passed. The
standing rule on this aircraft is that parameters change on the ground; flying
switches break that rule on purpose, so it has to be asked for explicitly, and
only after the ground and props-off tests have passed.
"""
from __future__ import annotations
import argparse, sys, time
from pymavlink import mavutil

sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from fc_port import exclusive_fc

# Defaults are the direct serial line, which is what this did before the
# router existed. The systemd unit passes the router's UDP endpoint instead:
# with one owner of the port there is no contention to work around.
PORT, BAUD = "/dev/ttyTHS1", 921600


class RcGpsSwitch:
    def __init__(self, channel=11, high_us=1750, allow_armed=False,
                 require_vps_fix=True, stable_n=4, port: str | None = None):
        self.ch = int(channel)
        self.high = int(high_us)
        self.allow_armed = bool(allow_armed)
        self.require_fix = bool(require_vps_fix)
        self.stable_n = int(stable_n)

        self.m = mavutil.mavlink_connection(port or PORT, baud=BAUD,
                                            source_system=1,
                                            source_component=221)
        if self.m.wait_heartbeat(timeout=20) is None:
            sys.exit("FC se heartbeat nahi aaya")
        self.m.mav.request_data_stream_send(self.m.target_system,
                                            self.m.target_component,
                                            mavutil.mavlink.MAV_DATA_STREAM_ALL, 5, 1)
        self.armed = False
        self.gps2_fix = 0
        self.gps2_sats = 0
        self.want = None
        self.stable = 0
        self.applied = None
        self.switches = 0
        self.refusals = 0

    # -- helpers ----------------------------------------------------------- #
    def say(self, text):
        """Tell the pilot, on the GCS, not just the log."""
        print(f"  {text}", flush=True)
        try:
            self.m.mav.statustext_send(mavutil.mavlink.MAV_SEVERITY_NOTICE,
                                       text.encode()[:50])
        except Exception:
            pass

    def get_param(self, name, timeout=4.0):
        for _ in range(3):
            self.m.mav.param_request_read_send(self.m.target_system,
                                               self.m.target_component,
                                               name.encode(), -1)
            t0 = time.time()
            while time.time() - t0 < timeout:
                msg = self.m.recv_match(type="PARAM_VALUE", blocking=True, timeout=1)
                if msg and msg.param_id == name:
                    return msg.param_value
        return None

    def set_primary(self, idx):
        self.m.mav.param_set_send(self.m.target_system, self.m.target_component,
                                  b"GPS_PRIMARY", float(idx),
                                  mavutil.mavlink.MAV_PARAM_TYPE_REAL32)

    # -- main -------------------------------------------------------------- #
    def run(self):
        # The channel must not already carry an ArduPilot auxiliary function.
        # The first channel picked for this was RC11, which is RC11_OPTION=18,
        # Land - flipping it to change GPS would have put the aircraft down.
        opt = self.get_param(f"RC{self.ch}_OPTION")
        if opt is None:
            sys.exit(f"RC{self.ch}_OPTION padha nahi ja saka")
        if int(opt) != 0:
            sys.exit(f"*** RC{self.ch} par pehle se option {int(opt)} laga hai "
                     f"(0 hona chahiye). Doosra channel chuno. ***")
        auto = self.get_param("GPS_AUTO_SWITCH")
        if auto is None or int(auto) != 0:
            print(f"GPS_AUTO_SWITCH={auto} - manual ke liye 0 kar raha hoon")
            self.m.mav.param_set_send(self.m.target_system, self.m.target_component,
                                      b"GPS_AUTO_SWITCH", 0.0,
                                      mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
            time.sleep(1.0)
        cur = self.get_param("GPS_PRIMARY")
        self.applied = None if cur is None else int(cur)
        t2 = self.get_param("GPS2_TYPE")
        print(f"ch{self.ch}: <{self.high} = asli GPS | >{self.high} = VPS")
        print(f"abhi GPS_PRIMARY={self.applied}  GPS2_TYPE={t2}  "
              f"armed-switching={'ON' if self.allow_armed else 'OFF (zameen par hi)'}\n",
              flush=True)

        while True:
            msg = self.m.recv_match(
                type=["RC_CHANNELS", "HEARTBEAT", "GPS2_RAW"],
                blocking=True, timeout=2)
            if msg is None:
                continue
            t = msg.get_type()
            if t == "HEARTBEAT":
                self.armed = bool(msg.base_mode &
                                  mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                continue
            if t == "GPS2_RAW":
                self.gps2_fix = msg.fix_type
                self.gps2_sats = msg.satellites_visible
                continue

            v = getattr(msg, f"chan{self.ch}_raw", 0)
            if not v:
                continue
            want = 1 if v >= self.high else 0
            if want == self.want:
                self.stable += 1
            else:
                self.want, self.stable = want, 1
            if self.stable != self.stable_n or want == self.applied:
                continue

            name = "VPS (GPS2)" if want else "asli GPS (GPS1)"
            if self.armed and not self.allow_armed:
                self.say(f"GPS switch mana: armed hai ({name})")
                self.refusals += 1
                self.applied = want      # do not nag on every sample
                continue
            if want == 1 and self.require_fix and self.gps2_fix < 3:
                self.say(f"VPS switch MANA: GPS2 fix={self.gps2_fix} sats={self.gps2_sats}")
                self.refusals += 1
                self.stable = 0
                continue

            self.set_primary(want)
            self.switches += 1
            self.say(f"GPS -> {name}")
            time.sleep(0.8)
            got = self.get_param("GPS_PRIMARY")
            self.applied = want if got is None else int(got)
            print(f"     (GPS_PRIMARY ab {got})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", type=int, default=11)
    ap.add_argument("--high-us", type=int, default=1750)
    ap.add_argument("--allow-armed", action="store_true",
                    help="hawa mein bhi switch karne do (pehle zameen par test karo)")
    ap.add_argument("--no-fix-check", action="store_true",
                    help="VPS ka fix check mat karo (SAFAI NAHI - test ke liye)")
    ap.add_argument("--port", default=PORT,
                    help="serial device, or a pymavlink endpoint such as "
                         "udpout:127.0.0.1:14650 to go through the router")
    a = ap.parse_args()
    sw = RcGpsSwitch(a.channel, a.high_us, a.allow_armed, not a.no_fix_check,
                     port=a.port)
    try:
        sw.run()
    except KeyboardInterrupt:
        print(f"\nband. {sw.switches} switch hue, {sw.refusals} mana kiye gaye.")


if __name__ == "__main__":
    # exclusive_fc stops the recorder so two readers do not split one serial
    # stream. Through the router there is only ever one reader of the port, so
    # pausing anything would be pointless - and on a service that must stay up
    # it would be harmful.
    if any(a.startswith(("udp", "tcp")) for a in sys.argv[1:]):
        main()
    else:
        with exclusive_fc():
            main()
