"""Above 35 m the flow velocity is withheld while the marker holds the position (flights 41/42)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flight_daemon import (FLOW_VEL_CALM_S, FLOW_VEL_GAP_LOCKOUT_S, FLOW_VEL_ON_H_M,
                           FLOW_VEL_OFF_H_M, flow_vel_gap_limit, flow_velocity_wanted)

DT = 0.04


def run(st, t0, secs, h, holds=True, gap=(0.0, 0.0)):
    """Feed `secs` seconds at 25 Hz; return (last answer, time after)."""
    n = int(round(secs / DT))
    w = True
    for i in range(n):
        w = flow_velocity_wanted(st, t0 + i * DT, h, holds, gap)
    return w, t0 + n * DT


def test_below_35_m_nothing_changes():
    st = {}
    w, _ = run(st, 0.0, 20.0, 34.0)
    assert w is True


def test_above_35_m_with_the_marker_and_a_steady_ekf_it_stops():
    st = {}
    w, _ = run(st, 0.0, FLOW_VEL_CALM_S + 0.5, 40.0)
    assert w is False


def test_it_waits_for_the_ekf_to_be_steady():
    st = {}
    w, _ = run(st, 0.0, FLOW_VEL_CALM_S - 0.5, 40.0)
    assert w is True


def test_without_the_marker_it_is_always_sent():
    st = {}
    w, _ = run(st, 0.0, 10.0, 50.0, holds=False)
    assert w is True


def test_without_the_ekf_position_it_is_always_sent():
    """If the EKF position is not arriving there is nothing to catch a divergence: keep the velocity."""
    st = {}
    for i in range(250):
        assert flow_velocity_wanted(st, i * DT, 50.0, True, None) is True


def test_losing_the_marker_brings_it_back_at_once():
    st = {}
    _, t = run(st, 0.0, 5.0, 45.0)
    assert flow_velocity_wanted(st, t, 45.0, False, (0.0, 0.0)) is True


def test_a_constant_offset_between_the_frames_is_ignored():
    st = {}
    w, _ = run(st, 0.0, 10.0, 45.0, gap=(7.0, -3.0))
    assert w is False


def test_flight_42a_the_ekf_drifting_away_brings_the_velocity_back():
    """42A: the EKF fell 2.9 m behind the marker position 8 s before it reset itself."""
    st = {}
    _, t = run(st, 0.0, 5.0, 36.0)
    assert flow_velocity_wanted(st, t, 36.0, True, (1.0, 0.0)) is False     # under 1.5 m: stays off
    assert flow_velocity_wanted(st, t + DT, 36.0, True, (2.9, 0.0)) is True
    assert "EKF" in st["why"]


def test_the_limit_grows_with_height():
    assert flow_vel_gap_limit(36.0) == 1.5
    assert abs(flow_vel_gap_limit(65.0) - 2.6) < 1e-9


def test_after_a_divergence_it_stays_on_for_a_while():
    st = {}
    _, t = run(st, 0.0, 5.0, 45.0)
    assert flow_velocity_wanted(st, t, 45.0, True, (5.0, 0.0)) is True
    w, t2 = run(st, t + DT, FLOW_VEL_GAP_LOCKOUT_S - 1.0, 45.0, gap=(5.0, 0.0))
    assert w is True                                     # steady again, but still locked out
    w, _ = run(st, t2, 3.0, 45.0, gap=(5.0, 0.0))
    assert w is False


def test_hysteresis_on_the_way_down():
    st = {}
    _, t = run(st, 0.0, 5.0, 40.0)
    assert flow_velocity_wanted(st, t, 34.0, True, (0.0, 0.0)) is False     # between 33 and 35
    assert flow_velocity_wanted(st, t + DT, FLOW_VEL_ON_H_M - 0.1, True, (0.0, 0.0)) is True
    assert flow_velocity_wanted(st, t + 2 * DT, 34.0, True, (0.0, 0.0)) is True


def test_an_off_height_below_the_on_height_is_not_needed():
    assert FLOW_VEL_ON_H_M < FLOW_VEL_OFF_H_M
