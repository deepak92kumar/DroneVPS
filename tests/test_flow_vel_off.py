"""Above 35 m the flow velocity is withheld while the marker holds the position (flight 41)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flight_daemon import FLOW_VEL_OFF_H_M, FLOW_VEL_ON_H_M, FLOW_VEL_OFF_HOLD_S, flow_velocity_wanted


def climb(st, heights, holds=True, t0=0.0, dt=0.04):
    out = []
    for i, h in enumerate(heights):
        out.append(flow_velocity_wanted(st, t0 + i * dt, h, holds))
    return out


def test_below_35_m_nothing_changes():
    st = {}
    assert all(climb(st, [h * 0.1 for h in range(0, 340)]))


def test_above_35_m_with_the_marker_it_stops():
    st = {}
    climb(st, [30.0] * 50)                               # marker has held for 2 s already
    assert flow_velocity_wanted(st, 10.0, FLOW_VEL_OFF_H_M + 0.1, True) is False


def test_without_the_marker_it_is_always_sent():
    """Flight 41 ended with the marker out of view; the flow is then the only velocity there is."""
    st = {}
    assert all(climb(st, [50.0] * 100, holds=False))


def test_losing_the_marker_brings_it_back_at_once():
    st = {}
    climb(st, [45.0] * 50)
    assert flow_velocity_wanted(st, 5.0, 45.0, True) is False
    assert flow_velocity_wanted(st, 5.04, 45.0, False) is True


def test_the_marker_must_hold_a_second_before_it_stops_again():
    """An intermittent marker at 45 m must not flap the velocity on and off every frame."""
    st = {}
    climb(st, [45.0] * 50)
    flow_velocity_wanted(st, 5.0, 45.0, False)             # lost: velocity back
    assert flow_velocity_wanted(st, 5.1, 45.0, True) is True           # just back: still sending
    assert flow_velocity_wanted(st, 5.1 + FLOW_VEL_OFF_HOLD_S, 45.0, True) is False


def test_hysteresis_on_the_way_down():
    st = {}
    climb(st, [40.0] * 50)
    assert flow_velocity_wanted(st, 3.0, 34.0, True) is False          # between 33 and 35: stays off
    assert flow_velocity_wanted(st, 3.04, FLOW_VEL_ON_H_M - 0.1, True) is True
    assert flow_velocity_wanted(st, 3.08, 34.0, True) is True          # and stays on until 35 again
