"""Horizontal velocity for the flight controller: flow for truth, accel for the gaps.

This replaces the AccelBridge that was lost with the SSD. It is written against
that module's measured failure rather than its code, which is gone.

WHAT WENT WRONG BEFORE, from the 8-11 Oct reports:

    raw flow at 6+ m/s      9.93 m/s against a 9.72 GPS1 truth   = 102%
    AccelBridge output      3.90 m/s                             =  40%
    sent to the FC          3.73 m/s                 the send path lost 0.18

Six metres per second disappeared inside one module. "Smoothing, the deadband
and the send gate are all innocent." It read 89% of truth on 9 Oct, 75% then
63% on 10 Oct with nothing changed in it, and 40% at sustained speed - worse
the faster and longer the flight. And: "replaying the flight's own logged
inputs through the real module gives about 9 m/s where the live run recorded
1.10 - so the collapse lives in state the recording did not carry."

What it cost: the aircraft flew 6.78 m/s on a 4.0 m/s commanded mission while
the EKF believed 3.02, missed waypoint 4 because the waypoint was always still
ahead from its own point of view, and produced thirty-two EKF resets on one
flight - with XKF4.SP at 0.21 and SM at 0.06, both far inside the 1.0 gate,
while SV reached 1.63-1.75. The position was fine; the velocity was not.

WHY THIS DESIGN CANNOT DO THAT

The loss was an ATTENUATION - a fixed fraction of truth - not an offset. That
matters, because it rules things out. A constant accelerometer bias in a
complementary filter produces a steady OFFSET (the fixed point moves to
v_flow + bias*dt/K), never a fraction of the input. An attenuation needs a term
that pulls the state toward zero, and its steady-state gain is then
K/(K+leak) < 1: invisible at a hover where the truth is near zero, and exactly
a fraction of truth at sustained speed. A decay-toward-zero is also the usual
way people stop a hover estimate from wandering, and it hides in an offline
replay that feeds flow on every step while the live run did not.

So there is NO leak here. The velocity state is never pulled toward zero.
When flow stops arriving, the estimate HOLDS and its uncertainty grows, and
past a timeout it is declared invalid and not sent at all. Holding a stale
velocity is wrong; silently reporting a fraction of the real one is worse,
because the aircraft keeps flying on it and nothing aboard can tell.

The second rule follows from the same report: every piece of internal state is
returned with the estimate, so the recorder carries it. The old fault could not
be diagnosed from the logs because the state that caused it was not written
down - five columns had to be added after the fact. Nothing here is private.

UNVERIFIED: the gains below are reasoned and unit-tested, not flown. tau_s in
particular wants a sweep against a recording. What IS pinned by the tests is
the property that matters: steady-state unity gain to flow, no attenuation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class VelocityState:
    """Everything the estimate is made of, returned with it every time.

    The 8 Oct fault was undiagnosable from the recording. This is the fix for
    that, not a convenience: the recorder writes these columns.
    """
    vn: float = 0.0              # m/s north, the estimate
    ve: float = 0.0              # m/s east
    valid: bool = False          # may this be sent to the FC at all
    reason: str = "no data"      # if not, why
    sent_vn: float = 0.0         # what the FC would be given (deadband applied)
    sent_ve: float = 0.0
    deadbanded: bool = False     # the deadband zeroed a real motion
    deadband_mps: float = 0.0    # the threshold in force at this height
    speed_mps: float = 0.0       # |v| of the estimate, before the deadband
    age_s: float = 0.0           # since the last flow update
    altitude_m: float = 0.0      # what the deadband and the sent accuracy
                                 # were computed at. Carried because the flow
                                 # -> metres scale IS altitude, so anything
                                 # downstream that weighs this estimate needs
                                 # to know the height it was made at.
    rejected_jump: int = 0       # flow samples refused as physically impossible
    slewed: bool = False         # the OUTPUT slew limited this step
    capped: bool = False         # the magnitude ceiling cut this step
    cap_mps: float = 0.0         # the ceiling in force, carried because the
                                 # sender has to declare an honest accuracy
                                 # for a capped sample and cannot reach the
                                 # bridge to ask what the ceiling was
    cap_count: int = 0           # how many times it has, this run
    slew_count: int = 0          # how many times it has, this run
    flow_updates: int = 0
    accel_steps: int = 0
    bias_n: float = 0.0          # accel bias estimate, north
    bias_e: float = 0.0
    bias_frozen: bool = True     # was bias adaptation allowed this step
    # The INPUTS, so that a flight can be replayed at a different tau_s.
    # Log 225 could not be: it showed that the accel blend was inflating the
    # speed but carried neither term, so the new gain had to be reasoned
    # instead of swept. Same lesson as the five columns added after 8 Oct.
    flow_vn: float = 0.0         # last accepted flow velocity, NED m/s
    flow_ve: float = 0.0
    accel_n: float = 0.0         # last horizontal accel pushed, NED m/s^2
    accel_e: float = 0.0


class VelocityBridge:
    """Fuse flow velocity with camera linear acceleration, in NED horizontal.

    Flow is the authority for magnitude: it measured 98-108% of truth across
    flights and 102% at the speed where the old bridge collapsed. Acceleration
    is used only to carry the estimate BETWEEN flow updates, where flow has
    nothing to say.

    tau_s is how quickly the estimate is pulled onto flow. It is a time
    constant, not a leak: it pulls toward the flow measurement, never toward
    zero, so its steady-state gain to a constant flow input is exactly 1.

    tau_s IS NOT SETTLED AND LOG 225 COULD NOT SETTLE IT. Recorded here
    because it was briefly changed to 0.05 on the strength of a measurement
    that was wrong, and the mistake is worth keeping.

    The claim was that the accel blend inflated the speed: flow-only read
    p50 0.53 m/s and this module's output 1.04, against a receiver truth of
    0.17. But the flow-only figure came from the position path smoothed over
    30 samples while the output was read raw. Smoothed the same way at every
    width, on log 225 / vps_00059:

        width   flow-only   output   truth    corr N flow / output
          1       2.13       1.04     n/a        -0.093 / +0.266
          3       1.69       1.04     n/a        +0.072 / +0.263
         10       0.90       1.00     0.15       +0.235 / +0.253
         30       0.53       0.75     0.17       +0.412 / +0.234

    So the blend HALVES the raw flow velocity rather than doubling it, and at
    high frequency it correlates better with truth than flow does. Lowering
    tau_s would have made the output follow that raw signal more closely and
    driven XKF4.SV - which is what tripped ALT_HOLD at 166.4 s - the wrong
    way.

    What the table actually indicts is the flow velocity itself: p50 2.13 m/s
    raw while the aircraft moved at 0.17. That is not a surprise, it is the
    number already written down a few lines below - noise sigma = 0.0867 x
    altitude, so 2.69 m/s at the 31 m this flight flew. The blend exists to
    filter exactly that, and SV is high because 2.7 m/s of input noise is
    more than any single-pole filter can hide.

    The honest state: this gain wants a sweep, the sweep wants the module's
    own inputs, and log 225 carried neither. flow_vn / flow_ve / accel_n /
    accel_e below were added for that reason and nothing else.
    """

    def __init__(self, *, tau_s: float = 0.35, flow_timeout_s: float = 0.6,
                 deadband_base_mps: float = 0.35,
                 # 0.005 -> 0.010 ON 22 OCT, measured on log 227, which
                 # would not hold in any of four VPS windows.
                 #
                 # The deadband's job is to suppress a phantom speed that the
                 # aircraft is not doing, and the phantom grows with height
                 # because velocity is pixels x altitude / focal. At 83 m the
                 # old threshold was 0.35 + 0.005*83 = 0.77 m/s while the
                 # sustained phantom measured over 10 s windows ran p50 0.17
                 # and p90 0.91 m/s - so the worst ones passed straight
                 # through, and a sustained false velocity is the one thing
                 # LOITER cannot settle against. With no stick input it can
                 # only cancel a velocity error by flying, which is exactly
                 # what log 227 did: in every window the aircraft moved
                 # OPPOSITE to the phantom, 19.8 m in 7.7 s in the worst.
                 #
                 # Swept against the raw receiver over 12606 samples above
                 # 5 m, counting rows the deadband would zero:
                 #
                 #   base  per_m   31m    83m   phantom killed   real motion
                 #                                              wrongly killed
                 #   0.35  0.005  0.51   0.77      19.2%            1.4%
                 #   0.35  0.010  0.66   1.18      26.9%            3.2%
                 #   0.35  0.015  0.81   1.59      32.7%            5.2%
                 #   0.35  0.020  0.97   2.01      37.1%            8.4%
                 #
                 # 0.010 puts the threshold above the p90 phantom and buys
                 # 7.7 points of suppression for 1.8 points of real motion,
                 # a ratio of 4.3:1. 0.015 falls to 3.6:1 and zeroes 47% of a
                 # VPS window outright, which approaches withholding the
                 # velocity - and the archived reports measured what that
                 # costs: "the EKF differentiates our position and believed
                 # 0.21 m/s while the aircraft did 1.09".
                 #
                 # A zeroed velocity is still SENT. The estimate stays valid
                 # and the FC is told zero, which for a hovering aircraft is
                 # the truth; deadbanded is recorded so a log can show it.
                 deadband_per_m: float = 0.010,
                 max_accel_mps2: float = 8.0,
                 slew_mps2: float = 5.0,
                 max_sent_mps: float = 6.0,
                 slew_hold_s: float = 2.0,
                 jump_escape_n: int = 5,
                 bias_tau_s: float = 20.0,
                 bias_adapt_below_mps: float = 0.25,
                 bias_adapt_after_s: float = 2.0,
                 max_speed_mps: float = 30.0):
        self.tau_s = float(tau_s)
        self.flow_timeout_s = float(flow_timeout_s)
        self.deadband_base_mps = float(deadband_base_mps)
        self.deadband_per_m = float(deadband_per_m)
        # The aircraft cannot change velocity faster than it can accelerate,
        # and that is a HARD physical bound, not a tuning choice. ATC_ANGLE_MAX
        # is 30 deg, so 9.81*tan(30) = 5.66 m/s2; 8.0 leaves room for a gust
        # and for the estimate's own lag without admitting anything real.
        #
        # Why this gate had to exist, measured over three flights on 20 Oct:
        # the flow's error in PIXELS is roughly constant with height (p99 about
        # 6-7 px at every altitude band), but velocity is pixels x altitude /
        # focal - so the same tracking artefact reads 0.6 m/s at 10 m and
        # 19.4 m/s at 70 m. Frame-to-frame change in the reported speed came
        # out at p99 1.696 and max 22.74 m/s against a physical ceiling of
        # 0.189 m/s per frame. Nine times, and a hundred and twenty times.
        #
        # The flow was NOT stale when this happened - age averaged 0.014 s
        # across the 668 spike samples - and the accel bias was tiny and
        # correctly frozen. So this is not integration drift; it is the flow
        # itself, amplified by height.
        self.max_accel_mps2 = float(max_accel_mps2)
        # An estimate that has gone wrong would otherwise reject every good
        # sample for ever. After this many refusals in a row, believe the
        # measurement instead of the state.
        self.jump_escape_n = int(jump_escape_n)
        self.rejected_jump = 0
        self._jump_run = 0
        self._vn_flow = self._ve_flow = None
        self._an = self._ae = 0.0
        self._last_alt_m = 0.0
        # The output slew, from the 9 Oct report. See value().
        self.slew_mps2 = float(slew_mps2)
        self.max_sent_mps = float(max_sent_mps)
        self.slew_hold_s = float(slew_hold_s)
        self.slewed = 0
        self._sent_n = self._sent_e = 0.0
        self._t_sent = None
        self._slew_since = -1e9
        self.bias_tau_s = float(bias_tau_s)
        self.bias_adapt_below_mps = float(bias_adapt_below_mps)
        self.bias_adapt_after_s = float(bias_adapt_after_s)
        self.max_speed_mps = float(max_speed_mps)

        self.vn = self.ve = 0.0
        self.bias_n = self.bias_e = 0.0
        self._t_accel: float | None = None
        self._t_flow: float | None = None
        self._slow_since: float | None = None
        self._started = False
        self.flow_updates = 0
        self.accel_steps = 0
        self.rejected_flow = 0
        self.capped = 0

    # -- inputs ----------------------------------------------------------
    def reset_state(self, why: str = "") -> None:
        """Throw away everything learned or integrated. Keep the configuration.

        AT THE PILOT'S INSTRUCTION, 23 Oct: "gps se vps par switch ke waqt
        purana koi bhi data" must not cross the handover - where the GPS is,
        that is where the VPS starts.

        The handover already did half of this. live_node re-takes the origin
        from the flight controller's own position and calls
        MapFuser.reset_displacement(), which zeroes the accumulated
        displacement AND puts the variance back to init_sigma_m. Measured over
        ten switches on flight 00000233, displacement went from as much as
        74.5 m down to 0.4-2.0 m every time. POSITION was never the thing
        being carried.

        This bridge was. Nothing has ever reset it - there was no method to
        call - so the integrated velocity, the learned accelerometer bias, the
        last flow and accel samples, the slew reference and every timer ran
        continuously from node start, through however long the aircraft sat on
        GPS, and straight into the first GPS_INPUT packet after the switch.

        WHAT IT COSTS, honestly: bias_n/bias_e are relearned from zero over
        bias_tau_s, which is 20 s. The bias was measured at 0.09-0.31 m/s^2 on
        the three 23 Oct flights and it only has to be right across the ~33 ms
        between flow frames, so the arithmetic says this is small - but it IS
        a real cost and it is the reason this is a deliberate instruction
        rather than an obvious cleanup.

        The cumulative counters are NOT reset. They are what a flight is
        audited with afterwards, and zeroing them mid-flight would hide the
        first half of it.
        """
        self.vn = self.ve = 0.0
        self.bias_n = self.bias_e = 0.0
        self._vn_flow = self._ve_flow = None
        self._an = self._ae = 0.0
        self._sent_n = self._sent_e = 0.0
        self._t_sent = None
        self._t_flow = None
        self._t_accel = None
        self._slew_since = -1e9
        self._slow_since = None
        self._jump_run = 0
        self._started = False
        self._last_alt_m = 0.0
        self.reset_why = str(why)
        self.n_resets = getattr(self, "n_resets", 0) + 1

    def push_accel(self, t: float, an: float, ae: float) -> None:
        """Horizontal acceleration in NED, m/s^2, gravity already removed.

        LINEAR_ACCELERATION from the camera IMU is what this expects - that is
        why the daemon asks for that report rather than ACCELEROMETER_RAW.
        """
        if self._t_accel is None:
            self._t_accel = t
            return
        dt = t - self._t_accel
        self._t_accel = t
        # A dt that is negative, zero or absurd means the clock jumped or a
        # sample arrived out of order. Integrating it would inject a step.
        if not (0.0 < dt < 0.25):
            return
        if not self._started:
            return                      # no flow yet: nothing to carry
        self._an, self._ae = float(an), float(ae)
        self.vn += (an - self.bias_n) * dt
        self.ve += (ae - self.bias_e) * dt
        self.accel_steps += 1
        self._clamp()

    def push_flow(self, t: float, vn: float, ve: float) -> bool:
        """Absolute horizontal velocity from optical flow, NED m/s.

        Returns False if the sample was refused. A flow velocity beyond
        max_speed_mps is not a fast aircraft, it is a tracking collapse - the
        old flow code returned 630 m/s when tracking failed - and must not be
        allowed to define the estimate.
        """
        speed = math.hypot(vn, ve)
        if not (speed < self.max_speed_mps) or not all(map(math.isfinite, (vn, ve))):
            self.rejected_flow += 1
            return False

        if not self._started:
            # First flow sample IS the estimate. Starting from zero and easing
            # onto it would under-report for the first few hundred ms of every
            # flight, which is exactly the error this module exists to avoid.
            self.vn, self.ve = vn, ve
            self._started = True
        else:
            dt = 0.0 if self._t_flow is None else max(0.0, t - self._t_flow)
            # The physical gate, against the PREVIOUS FLOW SAMPLE - not
            # against the estimate. Comparing with the estimate was the first
            # attempt and it was wrong: the complementary filter lags by
            # roughly a*tau, so a genuine 5 m/s2 acceleration puts 1.75 m/s
            # between flow and estimate, against a per-frame allowance of
            # 0.27. It refused real acceleration. Two consecutive flow
            # samples, though, are two measurements of the same velocity, and
            # the airframe's acceleration really does bound the difference.
            #
            # The allowance is that physical bound PLUS three sigma of the
            # flow's own noise, because a gate tighter than the measurement
            # noise rejects good data: at 5 m the noise alone is 0.43 m/s
            # per sample against a physical 0.27. Noise grows with height
            # (sigma = 0.0867 x altitude, measured), so the allowance does
            # too - which means at 70 m this gate is deliberately loose and
            # the protection there comes from the speed accuracy we report,
            # not from here. This catches the gross artefact: 22.74 m/s in
            # one frame at low level, which is 120x the physical ceiling.
            if dt > 0.0 and self._vn_flow is not None:
                jump = math.hypot(vn - self._vn_flow, ve - self._ve_flow)
                allowed = (self.max_accel_mps2 * dt
                           + 3.0 * self.ACC_PER_M * max(0.0, self._last_alt_m))
                if jump > allowed and self._jump_run < self.jump_escape_n:
                    self._jump_run += 1
                    self.rejected_jump += 1
                    # Deliberately do NOT advance _t_flow: the estimate is now
                    # running on the accelerometer alone, and the staleness
                    # clock has to keep ticking so that a run of refusals ends
                    # in "invalid, do not send" rather than in a stale number
                    # being presented as fresh.
                    return False
                self._jump_run = 0
            # Pull toward flow with a time constant. alpha -> 1 as dt grows,
            # so a long gap snaps onto the new measurement rather than
            # creeping; a short gap trusts the integrated state more.
            alpha = 1.0 - math.exp(-dt / self.tau_s) if dt > 0 else 0.5
            self.vn += alpha * (vn - self.vn)
            self.ve += alpha * (ve - self.ve)

        self._vn_flow, self._ve_flow = vn, ve
        self._adapt_bias(t, speed)
        self._t_flow = t
        self.flow_updates += 1
        self._clamp()
        return True

    # -- bias ------------------------------------------------------------
    def _adapt_bias(self, t: float, flow_speed: float) -> None:
        """Learn the accelerometer bias, but ONLY while barely moving.

        This is the trap the old module most plausibly fell into. A bias
        estimator that adapts while the aircraft is moving cannot tell a real
        sustained acceleration from an offset, so it learns the motion as bias
        and then subtracts it - and the faster and longer the flight, the more
        of the true speed it removes. That matches the record exactly: 89% of
        truth on a day of Loiter flights, 63% on mission days, 40% at sustained
        6+ m/s, "getting worse on its own" as the missions got longer.

        So adaptation is gated on the aircraft being slow for a sustained
        period, and bias_frozen is reported every step so a log can show
        whether it was ever allowed to move.
        """
        if flow_speed > self.bias_adapt_below_mps:
            self._slow_since = None
            return
        if self._slow_since is None:
            self._slow_since = t
            return
        if t - self._slow_since < self.bias_adapt_after_s:
            return
        # Stationary by flow, so whatever velocity we integrated is bias.
        dt = 0.0 if self._t_flow is None else max(0.0, t - self._t_flow)
        if dt <= 0.0:
            return
        k = 1.0 - math.exp(-dt / self.bias_tau_s)
        self.bias_n += k * (self.vn / max(dt, 1e-3))
        self.bias_e += k * (self.ve / max(dt, 1e-3))

    def _bias_frozen(self, t: float) -> bool:
        return (self._slow_since is None
                or (t - self._slow_since) < self.bias_adapt_after_s)

    def _clamp(self) -> None:
        s = math.hypot(self.vn, self.ve)
        if s > self.max_speed_mps:
            k = self.max_speed_mps / s
            self.vn *= k
            self.ve *= k

    # -- output ----------------------------------------------------------
    def deadband_for(self, altitude_m: float) -> float:
        """0.35 + 0.005 x altitude m/s. Flown 4 Oct, kept.

        It was flat 0.5 before that, and the phantom speed it suppresses grows
        with height, so the threshold has to as well. KNOWN COST, unsolved:
        below 25 m this discards 49% of genuine motion, which is why the
        aircraft wanders under 10 m while its POSITION there is good to 2 cm.
        That loss is reported, not hidden - see VelocityState.deadbanded.
        """
        return self.deadband_base_mps + self.deadband_per_m * max(0.0, altitude_m)

    #: m/s of flow velocity noise per metre of altitude, fitted to the three
    #: flights of 20 Oct. Used here to keep the jump gate from fighting the
    #: measurement's own scatter; live_node uses the same number to tell the
    #: flight controller how far to trust the velocity.
    ACC_PER_M = 0.0867

    def value(self, t: float, altitude_m: float) -> VelocityState:
        self._last_alt_m = float(altitude_m)
        st = VelocityState(
            altitude_m=float(altitude_m), rejected_jump=self.rejected_jump,
            vn=self.vn, ve=self.ve,
            flow_updates=self.flow_updates, accel_steps=self.accel_steps,
            bias_n=self.bias_n, bias_e=self.bias_e,
            bias_frozen=self._bias_frozen(t),
            deadband_mps=self.deadband_for(altitude_m),
            speed_mps=math.hypot(self.vn, self.ve),
            flow_vn=0.0 if self._vn_flow is None else self._vn_flow,
            flow_ve=0.0 if self._ve_flow is None else self._ve_flow,
            accel_n=self._an, accel_e=self._ae,
        )
        if not self._started or self._t_flow is None:
            st.reason = "no flow yet"
            return st
        st.age_s = max(0.0, t - self._t_flow)
        if st.age_s > self.flow_timeout_s:
            # HOLD the value, do not decay it, and refuse to send. The estimate
            # stays readable so a log shows what it was; valid=False is what
            # stops it reaching the aircraft.
            st.reason = f"flow {st.age_s:.2f}s stale (> {self.flow_timeout_s})"
            return st
        st.valid = True
        st.reason = "ok"
        # SOFT, not a cliff. The threshold is SUBTRACTED rather than used to
        # zero everything below it, so the output is continuous: at exactly
        # the deadband it is 0, just above it it is small, and a real 3 m/s
        # still arrives as 3 minus the threshold.
        #
        # Flight 245 is why. The phantom finally shrank enough to sit ON the
        # deadband instead of well above it, and the hard version then
        # chattered across it: the sent speed crossed zero 307 times in
        # 634 s, 0.48 times a second, with 39.9% of rows zeroed. Every
        # crossing is a step from 0 to the full value and back, and the
        # flight controller answered with tilt commands that saturated at
        # ATC_ANGLE_MAX and rates to 56.9 deg/s. That is the jerking the
        # pilot reported, and map fixes were ruled out as the cause - 16% of
        # the >1 m/s jumps land within 0.3 s of a fix against a 15% base
        # rate.
        #
        # What this does NOT change: the same speeds are still suppressed at
        # a hover, because at and below the threshold the output is still
        # zero. It removes the edge, not the deadband. deadbanded still
        # reports whenever the threshold took anything, so the log reads the
        # same way.
        # A RAMP, not a cliff - and not a permanent subtraction either.
        # Zero at and below the threshold, the full velocity at 1.5x it,
        # straight line between. Continuous at both ends.
        #
        # The ramp is HALF a deadband wide and no wider, because everything
        # inside it is under-reported and under-reporting real motion is the
        # fault that flew away on log 229. At 82 m that band is 1.17-1.76 m/s;
        # above 1.76 the velocity goes out in full. It is wide enough to
        # cover the chatter, which sat on the threshold itself.
        #
        # The first attempt here subtracted the threshold at every speed,
        # which removes the step but then under-reports real motion for
        # ever: a genuine 3 m/s went out as 2.34. That is precisely the
        # fault log 229 flew - an aircraft told it is slower than it is does
        # not correct, and it ran 11.1 m in 24 s. Two of the bridge's own
        # tests caught it, which is what they are for.
        if st.speed_mps < st.deadband_mps:
            st.deadbanded = True
            want_n = want_e = 0.0
        else:
            k = (st.speed_mps - st.deadband_mps) / (0.5 * st.deadband_mps)
            if k >= 1.0:
                want_n, want_e = self.vn, self.ve
            else:
                st.deadbanded = True
                want_n, want_e = self.vn * k, self.ve * k

        # THE OUTPUT SLEW. Restored from the 9 Oct 2026 status report, where
        # it is recorded as flown and kept:
        #
        #   "Velocity bound on the OUTPUT: input gate only -> 5 m/s^2 slew,
        #    2 s hold. Peak sent velocity 16.99 -> 2.37 m/s on the same
        #    inputs. This is what let the mission finish."
        #
        # Read the first three words. An input gate ALONE was already tried
        # and was not enough - which matters here, because the gate added to
        # push_flow on 20 Oct is an input gate. What the aircraft actually
        # flew on was a limit on what LEAVES this module, and the two are not
        # the same: a sequence of individually plausible steps still adds up
        # to an implausible speed, and only the output side sees the sum.
        #
        # 5 m/s2 is the number that flew. It sits just under the airframe's
        # own ceiling - ATC_ANGLE_MAX 30 deg gives 9.81*tan(30) = 5.66 - so
        # it cannot stop the aircraft reporting anything it can really do.
        #
        # The 2 s hold is the other half: once the slew has had to bite, the
        # limit stays engaged for that long rather than releasing on the next
        # sample, so a burst of outliers cannot walk the output up one
        # allowed step at a time.
        dt = 0.0 if self._t_sent is None else max(0.0, t - self._t_sent)
        if self._t_sent is not None and dt > 0.0:
            if t - self._slew_since < self.slew_hold_s:
                allowed = self.slew_mps2 * dt
            else:
                allowed = self.slew_mps2 * dt
            step = math.hypot(want_n - self._sent_n, want_e - self._sent_e)
            if step > allowed:
                k = allowed / step
                want_n = self._sent_n + k * (want_n - self._sent_n)
                want_e = self._sent_e + k * (want_e - self._sent_e)
                self.slewed += 1
                self._slew_since = t
                st.slewed = True
        # A HARD CEILING ON THE MAGNITUDE, and the slew is not one.
        #
        # The slew bounds the rate of CHANGE. At 5 m/s^2 it will happily walk
        # the output to 25 m/s over five seconds and call every step legal,
        # and that is exactly what flew: on logs 228 and 230 the speed sent to
        # the flight controller reached 25.86 and 25.47 m/s, with p99 at 15.5
        # and 15.1, and the bridge's pre-slew maximum was 26.25 and 25.48 - so
        # the slew removed essentially nothing.
        #
        # Against what the aircraft ACTUALLY did in those VPS windows, from
        # the receiver: p50 0.54-0.86, p95 1.39-2.22, p99 1.90-3.58, maximum
        # 5.04 m/s. LOIT_SPEED_MS is 3.0 and the VPS is only ever the source
        # in LOITER, so the position controller cannot command more than that,
        # and ATC_ANGLE_MAX of 15 degrees bounds the acceleration at about
        # 2.6 m/s^2. Nothing above 6 m/s is a measurement; it is the flow's
        # pixel noise multiplied by height.
        #
        # Those spikes are what tripped the EKF. XKF4.SV sat at p50 0.22-0.61
        # against a 1.2 gate all flight and reached 13.77 - the failsafe fires
        # on the spikes, not on a sustained bad state, and ALT_HOLD followed
        # four times on log 230. At 40-50 m, 48% of sent samples were above
        # 4 m/s.
        #
        # 6.0 clips above the fastest the aircraft has been measured doing on
        # the VPS (5.04 m/s) with margin, so no real motion is lost, and
        # bounds the noise at something the airframe could at least attempt.
        # Direction is kept; only the length is cut. A bounded wrong velocity
        # is recoverable, 25 m/s is not.
        cap = self.max_sent_mps
        if cap > 0.0:
            mag = math.hypot(want_n, want_e)
            if mag > cap:
                k = cap / mag
                want_n, want_e = want_n * k, want_e * k
                self.capped += 1
                st.capped = True

        self._sent_n, self._sent_e = want_n, want_e
        self._t_sent = t
        st.sent_vn, st.sent_ve = want_n, want_e
        st.slew_count = self.slewed
        st.cap_count = self.capped
        st.cap_mps = self.max_sent_mps
        return st


# --------------------------------------------------------------------------- #
# WHY THE VELOCITY IS SENT AT ALL, and why switching it off is not a fallback.
#
# Recorded 22 Oct 2026 from the archived reports of the pre-SSD project, which
# tried it and measured the result:
#
#     "send-veloff -> ON. Without it the EKF differentiates our position and
#      believed 0.21 m/s while the aircraft did 1.09."
#
# That is the 8 Oct failure mode reached by a different route: a fifth of the
# true speed, with nothing aboard able to tell. The EKF's own differentiation
# of a GPS_INPUT position is heavily filtered, so it lags and attenuates - the
# very thing this module exists to prevent.
#
# It matters because the obvious response to a high XKF4.SV is to stop sending
# velocity and let the position carry the flight, and on 22 Oct that is what
# was about to be proposed. The position IS the good signal here (SP never
# left 0.56 against a 1.2 gate on log 226, and the GPS->VPS handover lands
# within a metre) which makes the idea look safe. It is not: the EKF would
# then fly on a velocity five times too small rather than one that is noisy.
#
# The answer to a noisy velocity is to measure it better - a longer baseline,
# VpsConfig.flow_vel_frames - and to declare its accuracy honestly, which is
# VEL_ACC_PER_M in live_node. Not to withhold it.
