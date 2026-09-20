"""Flight 46: the velocity came back at 82 m after a half-second marker gap, and the marker fixes
were fed to the EKF 174 ms late."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flight_daemon import (FLOW_VEL_OFF_H_M, MARKER_FRESH_S, MARKER_HOLD_FAR_S,
                           MARKER_HOLD_FULL_H_M, flow_shift_since, marker_hold_window_s)


# --- how long a quiet marker still counts as holding the position -----------------------------

def test_low_down_the_window_is_unchanged():
    assert marker_hold_window_s(10.0) == MARKER_FRESH_S
    assert marker_hold_window_s(FLOW_VEL_OFF_H_M) == MARKER_FRESH_S


def test_high_up_the_daemon_waits_much_longer():
    """At 82 m flight 46 gave up after 0.5 s and the flow velocity wrecked the hover."""
    assert marker_hold_window_s(82.0) == MARKER_HOLD_FAR_S
    assert marker_hold_window_s(MARKER_HOLD_FULL_H_M) == MARKER_HOLD_FAR_S


def test_the_window_grows_smoothly_in_between():
    a, b, c = (marker_hold_window_s(h) for h in (40.0, 50.0, 60.0))
    assert MARKER_FRESH_S < a < b < c < MARKER_HOLD_FAR_S


# --- carrying a fix forward from its own frame ------------------------------------------------

HIST = [(100.0, 0.0, 0.0), (100.1, 0.1, -0.05), (100.2, 0.25, -0.10), (100.3, 0.40, -0.12)]


def test_the_flow_movement_since_the_frame_is_returned():
    dn, de = flow_shift_since(HIST, 100.1)
    assert abs(dn - 0.30) < 1e-9 and abs(de + 0.07) < 1e-9


def test_it_interpolates_between_samples():
    dn, de = flow_shift_since(HIST, 100.15)
    assert abs(dn - (0.40 - 0.175)) < 1e-9


def test_nothing_to_carry_when_the_frame_is_the_newest_sample():
    assert flow_shift_since(HIST, 100.3) == (0.0, 0.0)


def test_a_frame_older_than_the_history_is_refused():
    assert flow_shift_since(HIST, 99.0) is None
    assert flow_shift_since([], 100.0) is None
    assert flow_shift_since(HIST, None) is None


def test_a_frame_from_the_future_is_refused():
    assert flow_shift_since(HIST, 100.5) is None


def test_a_stale_frame_is_refused():
    """A second of flow movement is not a lag correction any more."""
    assert flow_shift_since([(100.0, 0.0, 0.0), (102.0, 1.0, 0.0)], 100.0, max_age_s=1.0) is None


def test_a_flow_reset_is_not_mistaken_for_movement():
    """The daemon zeroes its flow integral in several places; that is not 5 m of motion."""
    assert flow_shift_since([(100.0, 5.0, 0.0), (100.1, 0.0, 0.0)], 100.0) is None
