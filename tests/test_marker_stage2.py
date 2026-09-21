"""Marker map stage 2: the marker holds the horizontal position sent to the EKF."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flight_daemon import (MARKER_CORR_RATE_MS, MARKER_REANCHOR_M, marker_fix_step,
                           marker_fix_usable)

DT = 0.04          # 25 Hz


def run(st, flow, mk):
    return marker_fix_step(st, flow[0], flow[1], mk, DT)


# --- engaging must never make the position jump -------------------------------------------

def test_without_a_marker_it_is_exactly_the_flow():
    st = {}
    for f in [(0, 0), (0.1, 0.0), (0.3, -0.2), (1.0, 0.5)]:
        n, e, how = run(st, f, None)
        assert (round(n, 9), round(e, 9)) == (float(f[0]), float(f[1])) and how == "flow"


def test_the_first_fix_does_not_move_the_position():
    """Flight 37 had the flow at +15 m north when the marker locked. Sending the marker's own
    frame there would be a 15 m jump; the anchor must absorb it."""
    st = {}
    run(st, (15.0, 1.4), None)
    n, e, how = run(st, (15.0, 1.4), (0.1, -0.2))
    assert how == "anchor" and (n, e) == (15.0, 1.4)


# --- once anchored, the marker removes drift -------------------------------------------------

def test_flow_drift_is_pulled_back_to_the_marker():
    """The drone hovers still over the marker while the flow creeps north 2 m."""
    st = {}
    run(st, (0.0, 0.0), (0.0, 0.0))                     # anchor
    for i in range(1, 400):                              # 16 s
        flow = (2.0 * i / 399, 0.0)                      # flow drifts 2 m
        n, e, _ = run(st, flow, (0.0, 0.0))              # marker: drone has not moved
    assert abs(n) < 0.05 and abs(e) < 0.05


def test_real_motion_seen_by_both_is_kept():
    st = {}
    run(st, (0.0, 0.0), (0.0, 0.0))
    for i in range(1, 50):
        d = 0.02 * i
        n, e, _ = run(st, (d, 0.0), (d, 0.0))
    assert abs(n - 0.98) < 1e-6


def test_one_bad_detection_moves_it_only_centimetres():
    st = {}
    run(st, (0.0, 0.0), (0.0, 0.0))
    n, e, _ = run(st, (0.0, 0.0), (3.0, 0.0))          # a wild fix, 3 m off
    assert math.hypot(n, e) <= MARKER_CORR_RATE_MS * DT + 1e-9    # 16 cm, not 3 m


# --- the awkward cases ----------------------------------------------------------------------

def test_a_flow_reset_is_absorbed():
    """The daemon zeroes its flow integral in several places; the sent position must not follow."""
    st = {}
    run(st, (12.0, 3.0), (0.0, 0.0))
    n0, e0, _ = run(st, (12.0, 3.0), (0.0, 0.0))
    n1, e1, _ = run(st, (0.0, 0.0), (0.0, 0.0))        # flow snaps back to zero
    assert math.hypot(n1 - n0, e1 - e0) < 1e-6


def test_a_huge_disagreement_is_reanchored_not_chased():
    """After a long loss the flow may have drifted far; do not drag the drone across the field."""
    st = {}
    run(st, (0.0, 0.0), (0.0, 0.0))
    for _ in range(10):
        run(st, (0.0, 0.0), None)
    n, e, how = run(st, (0.0, 0.0), (MARKER_REANCHOR_M + 2.0, 0.0))
    assert how == "reanchor" and math.hypot(n, e) < 1e-9


def test_losing_the_marker_carries_on_with_the_flow_from_where_it_was():
    st = {}
    run(st, (0.0, 0.0), (0.0, 0.0))
    for i in range(1, 200):
        run(st, (1.0 * i / 199, 0.0), (0.0, 0.0))      # marker holds it at 0 while flow drifts 1 m
    n_lock, _, _ = run(st, (1.0, 0.0), (0.0, 0.0))
    n_lost, _, how = run(st, (1.3, 0.0), None)          # marker gone, flow moves 0.3 m
    assert how == "flow" and abs((n_lost - n_lock) - 0.3) < 1e-6


# --- gates -----------------------------------------------------------------------------------

def good(**kw):
    a = dict(mid=45, height_m=10.0, marker_down_m=10.1, roll_deg=3.0, pitch_deg=-2.0,
             confirmed=True, airborne=True, armed=True)
    a.update(kw)
    return marker_fix_usable(**a)


def test_a_good_fix_passes():
    assert good()[0]


def test_only_the_centre_target_is_used():
    assert good(mid=16)[0] and good(mid=0)[0]
    assert not good(mid=6)[0]           # a distributed marker, somewhere else on the field
    assert not good(mid=15)[0]          # the separate pad


def test_the_height_window():
    assert not good(height_m=1.5, marker_down_m=1.5)[0]
    assert not good(height_m=110.0, marker_down_m=110.0)[0]


def test_a_pose_that_disagrees_with_the_lidar_is_refused():
    """A wrong size or a mis-decode shows up as a height the lidar does not agree with."""
    assert not good(height_m=10.0, marker_down_m=6.0)[0]
    assert not good(height_m=10.0, marker_down_m=14.0)[0]


def test_unconfirmed_or_not_flying_is_refused():
    assert not good(confirmed=False)[0]
    assert not good(airborne=False)[0]
    assert not good(armed=False)[0]


def test_steep_tilt_is_refused():
    assert not good(roll_deg=30.0)[0]


def test_before_the_marker_locks_nothing_changes_even_through_a_flow_reset():
    """With --marker-pos on and no marker ever seen, the daemon must behave exactly as before."""
    st = {}
    for f in [(0.0, 0.0), (5.0, 1.0), (9.0, 2.0), (0.0, 0.0), (0.4, 0.1)]:   # includes a reset to 0
        n, e, how = run(st, f, None)
        assert (n, e) == (float(f[0]), float(f[1])) and how == "flow"


# --- stage 3: above 15 m -----------------------------------------------------------------------

from flight_daemon import MARKER_POS_ERR_M, marker_pos_err_m


def test_stage_3_uses_the_marker_up_to_100_m():
    """Flights 38/39 locked ID 45 continuously at 40-55 m; that is where the flow drifts most."""
    for h in (20.0, 50.0, 95.0):
        assert good(height_m=h, marker_down_m=h)[0]
    assert not good(height_m=110.0, marker_down_m=110.0)[0]


def test_declared_error_matches_what_the_marker_measures():
    """Flight 64 measured the marker's own per-fix scatter from consecutive fixes 0.15 s apart:
    0.22 m at 40-60 m, 0.25 at 60-80, 0.23 at 80-95 - flat, not growing with height. What we
    declare has to stay in the same country as that: a few times the jitter for the slow errors
    the jitter cannot see, never the 1.7 m we used to claim at 85 m."""
    assert marker_pos_err_m(5.0) == MARKER_POS_ERR_M and marker_pos_err_m(25.0) == MARKER_POS_ERR_M
    for h, jitter in ((50.0, 0.22), (70.0, 0.25), (85.0, 0.23), (100.0, 0.48)):
        declared = marker_pos_err_m(h)
        assert declared > jitter                      # never flatter than the truth
        assert declared < 6.0 * jitter                # nor absurdly humble (was 7.6x at 85 m)


def test_a_jittery_far_marker_still_holds_against_a_drifting_flow():
    """50 m: the marker jitters 15 cm per frame, the flow drifts 0.8 m/s. The sent position must
    stay near the marker, not follow the flow."""
    import random
    rnd = random.Random(3)
    st = {}
    run(st, (0.0, 0.0), (0.0, 0.0))
    for i in range(1, 500):                               # 20 s
        flow = (0.8 * i * DT, 0.0)                        # 16 m of flow drift
        n, e, _ = run(st, flow, (rnd.gauss(0, 0.15), rnd.gauss(0, 0.15)))
    assert math.hypot(n, e) < 0.6


# --- gentler corrections up high (20 Sep: every blow-up reached 25-30 deg of lean) ------------

from flight_daemon import MARKER_CORR_RATE_MS, marker_corr_rate_ms, marker_pos_err_m


def test_the_correction_rate_does_not_fade_with_height():
    """Flight 51: fading it to 1 m/s above 60 m saturated the limiter - the flow drifts against
    the marker at 3-4.8 m/s up there - and the hover broke above 50 m."""
    for h in (5.0, 35.0, 60.0, 90.0):
        assert marker_corr_rate_ms(h) == MARKER_CORR_RATE_MS


def test_the_marker_can_outrun_the_flow_drift_up_high():
    """The flow drifted against the marker at up to 4.8 m/s at 60-80 m in flight 51."""
    assert marker_corr_rate_ms(70.0) > 3.0


def test_the_declared_error_never_falls_under_the_ekf_floor():
    """EK3_POSNE_M_NSE is 0.5 m on this airframe, so claiming less buys nothing and only removes
    our own margin. The floor stays where stage 2 put it."""
    for h in (0.0, 30.0, 62.0):
        assert marker_pos_err_m(h) == MARKER_POS_ERR_M
    assert marker_pos_err_m(100.0) > MARKER_POS_ERR_M   # above ~62 m it does open up again


def test_the_reanchor_threshold_is_flat_at_every_height():
    """Flight 60: max(6 m, 0.15*h) let the correction drag the sent position 9.6 m in five seconds
    at 70-95 m (4.8 m before). The EKF read that sustained one-way pull as a velocity its IMU could
    not see - test ratio 0.11 -> 1.07 - and failsafed to ALT_HOLD at 72 m. Re-anchoring is what
    stops the pull; it must not be relaxed with height."""
    st = {}
    run(st, (0.0, 0.0), (0.0, 0.0))
    flow = 0.0
    hows = []
    for _ in range(8):
        flow += 1.5
        _, _, how = run(st, (flow, 0.0), (0.0, 0.0))
        hows.append(how)
    assert "reanchor" in hows            # the pull is cut off, at every height
