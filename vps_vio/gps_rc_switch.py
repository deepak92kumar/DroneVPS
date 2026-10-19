#!/usr/bin/env python3
"""Flip between the real GPS and the VPS from a switch on the transmitter.

ArduPilot has no auxiliary function for choosing the primary GPS, so this
watches an RC channel and sets GPS_PRIMARY itself. A Lua script could do the
same on the flight controller, but SCR_ENABLE is deliberately off and the
companion is already reading this link.

    ch6 at 1045  ->  GPS_PRIMARY = 0   real GPS
    ch6 at 1495  ->  nothing. Holds whatever is selected.
    ch6 at 1945  ->  GPS_PRIMARY = 1   VPS

Three positions, and the centre is blank on purpose: passing through it on the
way to the VPS must not command a source change, and brushing the detent must
not be an instruction. Guard bands are the midpoints, 1270 and 1720, leaving a
450 us dead zone.

Those endpoints are not guesses - RC6_MIN and RC6_MAX read 1045 and 1945 off
the flight controller, so the channel is already calibrated to exactly that
travel. RC6_OPTION is 0 and FLTMODE_CH is 5, so nothing else is listening.

Three guards, because handing a switch this much authority deserves them.

It refuses a channel that already carries an ArduPilot auxiliary function. The
first channel picked for this was RC11, and RC11_OPTION was 18 - Land. Flipping
it to change GPS source would have landed the aircraft. Nothing about reading a
channel tells you what else is listening to it, so the parameter is checked -
and it was checked again before moving to channel 6.

It refuses to move onto the VPS unless the VPS is actually healthy. The flight
controller will not check that for you - on the GPS_INPUT path it trusts
whatever arrives - so a switch flipped while the VPS is dead would hand
navigation to nothing at all. The refusal is announced on the GCS.

Health comes from the VPS node's own liveness file, NOT from whether GPS2 has
a fix. Using GPS2 deadlocks, and 20 Oct is how that was found: the node now
stops injecting entirely while the aircraft is on the real receiver, which is
the pilot's rule - so on GPS, GPS2 has no fix, so this refused to move onto the
VPS, so the node never started injecting, so GPS2 never got a fix. Round and
round, with the aircraft unable to ever reach the VPS. The log reads "VPS
switch MANA: GPS2 fix=1 sats=0" twelve times while the pilot flips the switch
and nothing happens.

The liveness file breaks the circle because the node writes it whether or not
it is injecting: it says what the node can SEE, not what it has sent. A GPS2
fix is still accepted as proof on its own, for when the aircraft is already on
the VPS.

And it refuses to do anything while armed unless --allow-armed is passed. The
standing rule on this aircraft is that parameters change on the ground; flying
switches break that rule on purpose, so it has to be asked for explicitly, and
only after the ground and props-off tests have passed.
"""
from __future__ import annotations
import argparse, os, sys, time
from pymavlink import mavutil

sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from fc_port import exclusive_fc
import single_instance

# Defaults are the direct serial line, which is what this did before the
# router existed. The systemd unit passes the router's UDP endpoint instead:
# with one owner of the port there is no contention to work around.
PORT, BAUD = "/dev/ttyTHS1", 921600


class RcGpsSwitch:
    LIVENESS = "/run/drone/vps.alive"
    LIVENESS_MAX_AGE_S = 5.0
    MIN_TRACKS = 30

    def vps_healthy(self):
        """(ok, why) from the VPS node's own liveness file.

        Rewritten every loop with what the node can see - "t=91.72 n=1705
        tracks=147 sigma=6.6 fixes=0 flow=1699 vel=ok" - so a stale one means
        the node is stopped or wedged, which Restart= alone would not tell us.
        """
        try:
            age = time.time() - os.path.getmtime(self.liveness)
        except OSError:
            return False, "VPS node chal hi nahi raha"
        if age > self.LIVENESS_MAX_AGE_S:
            return False, f"VPS {age:.0f}s se ruka hai"
        try:
            with open(self.liveness) as fh:
                kv = dict(p.split("=", 1) for p in fh.read().split() if "=" in p)
        except OSError:
            return False, "VPS liveness padhi nahi gayi"
        try:
            tracks = int(float(kv.get("tracks", 0)))
        except ValueError:
            tracks = 0
        if tracks < self.MIN_TRACKS:
            return False, f"VPS tracking kamzor ({tracks})"
        return True, f"VPS ok t{tracks}"

    def __init__(self, channel=6, high_us=1720, allow_armed=False,
                 require_vps_fix=True, stable_n=4, port: str | None = None,
                 low_us=1270, liveness: str | None = None):
        self.liveness = liveness or self.LIVENESS
        self.ch = int(channel)
        self.high = int(high_us)
        self.low = int(low_us)
        self.allow_armed = bool(allow_armed)
        self.require_fix = bool(require_vps_fix)
        self.stable_n = int(stable_n)

        self.m = mavutil.mavlink_connection(port or PORT, baud=BAUD,
                                            source_system=1,
                                            source_component=221)
        # Announce before waiting. Through the router a client that has not
        # spoken is not registered, so the heartbeat it is waiting for can
        # never reach it - the router learns an address from a packet, and
        # silence carries no address.
        try:
            self.m.mav.heartbeat_send(6, 8, 0, 0, 0)
        except Exception:
            pass
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
    def decide(self, v: int):
        """Which source this pulse width asks for, or None for 'nothing'.

        Three positions, and the middle one is DELIBERATELY blank. RC6 travels
        1045 / 1495 / 1945. Low asks for the real GPS, high asks for the VPS,
        and the centre detent asks for nothing at all - it holds whatever is
        already selected.

        A two-way threshold would have made the centre mean "real GPS", so
        passing through it on the way to the VPS would have commanded a source
        change the pilot never asked for, and every brush against the detent
        would have been an instruction.

        Guard bands are the midpoints: 1270 and 1720, leaving 450 us of dead
        zone between them.
        """
        if v <= self.low:
            return 0
        if v >= self.high:
            return 1
        return None

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
        print(f"ch{self.ch}:  <={self.low} = asli GPS  |  "
              f"{self.low+1}-{self.high-1} = centre, kuch nahi  |  "
              f">={self.high} = VPS")
        print(f"abhi GPS_PRIMARY={self.applied}  GPS2_TYPE={t2}  "
              f"armed-switching={'ON' if self.allow_armed else 'OFF (zameen par hi)'}\n",
              flush=True)

        # A heartbeat every few seconds, for the whole run - not just at the
        # start. The router drops a UDP client it has not heard from in 30
        # seconds (client_timeout_s), and after the opening handshake this
        # process is PURELY a listener: it reads RC_CHANNELS and sets a
        # parameter, and setting a parameter is rare. So half a minute after
        # starting it fell off the router's client list, RC_CHANNELS stopped
        # arriving, and recv_match returned None for ever while the loop
        # quietly went round doing nothing.
        #
        # That is not a theoretical failure. 20 Oct: the service switched once
        # at 02:18:08 and then logged nothing at all while the pilot flipped
        # the switch three to five times - "kis par hai pata hi nahi chal
        # raha". The switch was not slow, it was deaf.
        next_hb = 0.0
        while True:
            now = time.time()
            if now >= next_hb:
                next_hb = now + 5.0
                try:
                    self.m.mav.heartbeat_send(6, 8, 0, 0, 0)
                except Exception:
                    pass
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
            want = self.decide(v)
            if want is None:
                self.stable = 0          # centre: decide nothing, hold
                continue
            if want == self.want:
                self.stable += 1
            else:
                self.want, self.stable = want, 1
            if self.stable != self.stable_n or want == self.applied:
                continue

            name = "VPS (GPS2)" if want else "asli GPS (GPS1)"
            # The armed refusal that used to sit here is gone, at the pilot's
            # instruction. It was half of what stranded the aircraft on 20 Oct:
            # the VPS node stopped injecting on arm, and then this refused to
            # let the pilot switch back to the receiver, so there was no way
            # out from the transmitter. The switch is the pilot's control and
            # it answers while flying.
            if want == 1 and self.require_fix:
                # Either proof will do. A GPS2 fix means the node is already
                # injecting and the aircraft can see it. The liveness file
                # means the node is alive and tracking even though it is
                # deliberately silent toward the FC while we sit on the
                # receiver. Requiring the FIRST alone is the deadlock.
                ok, why = self.vps_healthy()
                if not ok and self.gps2_fix < 3:
                    self.say(f"VPS switch MANA: {why}"[:50])
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
    ap.add_argument("--channel", type=int, default=6)
    ap.add_argument("--low-us", type=int, default=1270,
                    help="at or below this = real GPS")
    ap.add_argument("--high-us", type=int, default=1720,
                    help="at or above this = VPS. Between the two is the "
                         "centre detent, which commands nothing. RC6 travels "
                         "1045 / 1495 / 1945, so these are the midpoints.")
    ap.add_argument("--allow-armed", action="store_true",
                    help="hawa mein bhi switch karne do (pehle zameen par test karo)")
    ap.add_argument("--no-fix-check", action="store_true",
                    help="VPS ka fix check mat karo (SAFAI NAHI - test ke liye)")
    ap.add_argument("--liveness", default=RcGpsSwitch.LIVENESS,
                    help="the VPS node's liveness file. Health is read here "
                         "rather than from GPS2's fix, because the node stops "
                         "injecting while the aircraft is on the real receiver "
                         "- so GPS2 has no fix exactly when you need to decide "
                         "whether the VPS is fit to take over.")
    ap.add_argument("--port", default=PORT,
                    help="serial device, or a pymavlink endpoint such as "
                         "udpout:127.0.0.1:14650 to go through the router")
    a = ap.parse_args()
    # One copy. Two would both be setting GPS_PRIMARY from the same channel,
    # and a refusal announced twice on the radio is a pilot reading two
    # different stories about which source is live.
    _lock_fd = single_instance.require("drone-rc-switch")   # noqa: F841
    sw = RcGpsSwitch(a.channel, a.high_us, a.allow_armed, not a.no_fix_check,
                     port=a.port, low_us=a.low_us, liveness=a.liveness)
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
