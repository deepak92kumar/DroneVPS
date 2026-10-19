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
    flow_updates: int = 0
    accel_steps: int = 0
    bias_n: float = 0.0          # accel bias estimate, north
    bias_e: float = 0.0
    bias_frozen: bool = True     # was bias adaptation allowed this step


class VelocityBridge:
    """Fuse flow velocity with camera linear acceleration, in NED horizontal.

    Flow is the authority for magnitude: it measured 98-108% of truth across
    flights and 102% at the speed where the old bridge collapsed. Acceleration
    is used only to carry the estimate BETWEEN flow updates, where flow has
    nothing to say.

    tau_s is how quickly the estimate is pulled onto flow. It is a time
    constant, not a leak: it pulls toward the flow measurement, never toward
    zero, so its steady-state gain to a constant flow input is exactly 1.
    """

    def __init__(self, *, tau_s: float = 0.35, flow_timeout_s: float = 0.6,
                 deadband_base_mps: float = 0.35,
                 deadband_per_m: float = 0.005,
                 bias_tau_s: float = 20.0,
                 bias_adapt_below_mps: float = 0.25,
                 bias_adapt_after_s: float = 2.0,
                 max_speed_mps: float = 30.0):
        self.tau_s = float(tau_s)
        self.flow_timeout_s = float(flow_timeout_s)
        self.deadband_base_mps = float(deadband_base_mps)
        self.deadband_per_m = float(deadband_per_m)
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

    # -- inputs ----------------------------------------------------------
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
            # Pull toward flow with a time constant. alpha -> 1 as dt grows,
            # so a long gap snaps onto the new measurement rather than
            # creeping; a short gap trusts the integrated state more.
            alpha = 1.0 - math.exp(-dt / self.tau_s) if dt > 0 else 0.5
            self.vn += alpha * (vn - self.vn)
            self.ve += alpha * (ve - self.ve)

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

    def value(self, t: float, altitude_m: float) -> VelocityState:
        st = VelocityState(
            vn=self.vn, ve=self.ve,
            flow_updates=self.flow_updates, accel_steps=self.accel_steps,
            bias_n=self.bias_n, bias_e=self.bias_e,
            bias_frozen=self._bias_frozen(t),
            deadband_mps=self.deadband_for(altitude_m),
            speed_mps=math.hypot(self.vn, self.ve),
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
        if st.speed_mps < st.deadband_mps:
            st.deadbanded = True
            st.sent_vn = st.sent_ve = 0.0
        else:
            st.sent_vn, st.sent_ve = self.vn, self.ve
        return st
