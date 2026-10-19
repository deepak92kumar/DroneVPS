"""Above 35 m the flow velocity is withheld while the marker holds the position (flights 41-43)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flight_daemon import (FLOW_VEL_CALM_S, FLOW_VEL_GAP_LOCKOUT_S, FLOW_VEL_ON_H_M,
                           FLOW_VEL_OFF_H_M, FLOW_VEL_SP_ON, flow_vel_gap_limit, flow_velocity_wanted)

DT = 0.04


def run(st, t0, secs, h, holds=True, gap=(0.0, 0.0), sp=0.1):
    """Feed `secs` seconds at 25 Hz; return (last answer, time after)."""
    n = int(round(secs / DT))
    w = True
    for i in range(n):
        w = flow_velocity_wanted(st, t0 + i * DT, h, holds, gap, sp)
    return w, t0 + n * DT


def test_below_35_m_nothing_changes():
    w, _ = run({}, 0.0, 20.0, 34.0)
    assert w is True


def test_above_35_m_with_the_marker_and_a_steady_ekf_it_stops():
    w, _ = run({}, 0.0, FLOW_VEL_CALM_S + 0.5, 40.0)
    assert w is False


def test_it_waits_for_the_ekf_to_be_steady():
    w, _ = run({}, 0.0, FLOW_VEL_CALM_S - 0.5, 40.0)
    assert w is True


def test_it_does_not_stop_while_the_ekf_is_uneasy():
    w, _ = run({}, 0.0, 20.0, 40.0, sp=0.6)
    assert w is True


def test_without_the_marker_it_is_always_sent():
    w, _ = run({}, 0.0, 10.0, 50.0, holds=False)
    assert w is True


def test_without_the_ekf_status_it_is_always_sent():
    """If the EKF is not reporting there is nothing to catch a divergence: keep the velocity."""
    st = {}
    for i in range(250):
        assert flow_velocity_wanted(st, i * DT, 50.0, True, (0.0, 0.0), None) is True
        assert flow_velocity_wanted(st, i * DT, 50.0, True, None, 0.1) is True


def test_losing_the_marker_brings_it_back_at_once():
    st = {}
    _, t = run(st, 0.0, 5.0, 45.0)
    assert flow_velocity_wanted(st, t, 45.0, False, (0.0, 0.0), 0.1) is True


def test_a_constant_offset_between_the_frames_is_ignored():
    w, _ = run({}, 0.0, 10.0, 45.0, gap=(7.0, -3.0))
    assert w is False


def test_flight_42a_the_ekf_test_ratio_brings_the_velocity_back():
    """42A: the test ratio reached 0.8 at 240.8 s, before the EKF began rejecting the marker."""
    st = {}
    _, t = run(st, 0.0, 5.0, 36.0)
    assert flow_velocity_wanted(st, t, 36.0, True, (0.0, 0.0), 0.6) is False
    assert flow_velocity_wanted(st, t + DT, 36.0, True, (0.0, 0.0), FLOW_VEL_SP_ON) is True
    assert "test ratio" in st["why"]


def test_flight_43_a_2_m_gap_on_a_climb_no_longer_brings_it_back():
    """43: climbing at 46 m the gap reached 2.1 m with the test ratio at 0.35; the old 1.5 m
    limit brought back a flow velocity that is 2 m/s wrong at that height."""
    st = {}
    _, t = run(st, 0.0, 5.0, 46.0)
    assert flow_velocity_wanted(st, t, 46.0, True, (2.1, 0.0), 0.35) is False


def test_a_huge_gap_still_brings_it_back():
    st = {}
    _, t = run(st, 0.0, 5.0, 45.0)
    assert flow_velocity_wanted(st, t, 45.0, True, (flow_vel_gap_limit(45.0) + 0.1, 0.0), 0.3) is True
    assert "off by" in st["why"]


def test_the_backstop_grows_with_height():
    assert flow_vel_gap_limit(40.0) == 6.0
    assert abs(flow_vel_gap_limit(90.0) - 9.0) < 1e-9


def test_an_ordinary_climb_gap_no_longer_brings_it_back():
    """Flights 44/46/47 all resumed the velocity on a 3 m gap at 43-51 m while the EKF was fine."""
    st = {}
    _, t = run(st, 0.0, 5.0, 45.0)
    assert flow_velocity_wanted(st, t, 45.0, True, (3.1, 0.0), 0.35) is False


def test_after_trouble_it_stays_on_for_a_while():
    st = {}
    _, t = run(st, 0.0, 5.0, 45.0)
    assert flow_velocity_wanted(st, t, 45.0, True, (0.0, 0.0), 0.9) is True
    w, t2 = run(st, t + DT, FLOW_VEL_GAP_LOCKOUT_S - 1.0, 45.0)
    assert w is True                                     # calm again, but still locked out
    w, _ = run(st, t2, 3.5, 45.0)
    assert w is False


def test_hysteresis_on_the_way_down():
    st = {}
    _, t = run(st, 0.0, 5.0, 40.0)
    assert flow_velocity_wanted(st, t, 34.0, True, (0.0, 0.0), 0.1) is False     # between 33 and 35
    assert flow_velocity_wanted(st, t + DT, FLOW_VEL_ON_H_M - 0.1, True, (0.0, 0.0), 0.1) is True
    assert flow_velocity_wanted(st, t + 2 * DT, 34.0, True, (0.0, 0.0), 0.1) is True
    assert FLOW_VEL_ON_H_M < FLOW_VEL_OFF_H_M


def test_hand_flying_stops_the_flow_velocity():
    """Flight 67: an EKF failsafe put the drone in ALT_HOLD at 56 m. Both lanes' POSITION test
    ratios were back under 0.5 within five seconds and the drone was calm, but the VELOCITY ratio
    kept bouncing 1.03-1.37 against FS_EKF_THRESH of 1.20 - the marker was lost, the guard had
    brought the flow velocity back, and up there it is 1.4-2.3 m/s wrong. The failsafe therefore
    never cleared in the air: it cleared at 0.3 m, forty-eight seconds later. Until it clears
    ArduPilot refuses LOITER, so the pilot cannot get back into it without landing."""
    from flight_daemon import flow_velocity_blocked_by_mode

    for mode in ("ALT_HOLD", "alt_hold", " STABILIZE ", "ACRO", "SPORT", "DRIFT"):
        assert flow_velocity_blocked_by_mode(mode, True) is True


def test_a_position_mode_is_never_starved_of_velocity():
    """The whole point is that this only acts when no position controller is running. LOITER and
    the autonomous modes must be untouched, or this becomes the sixth change to break a flight."""
    from flight_daemon import flow_velocity_blocked_by_mode

    for mode in ("LOITER", "POSHOLD", "AUTO", "GUIDED", "RTL", "LAND", "CIRCLE", "BRAKE"):
        assert flow_velocity_blocked_by_mode(mode, True) is False


def test_on_the_ground_it_changes_nothing():
    """Not airborne is not hand flying, and the takeoff path is left exactly as it was."""
    from flight_daemon import flow_velocity_blocked_by_mode

    assert flow_velocity_blocked_by_mode("ALT_HOLD", False) is False
    assert flow_velocity_blocked_by_mode("", True) is False
    assert flow_velocity_blocked_by_mode(None, True) is False
