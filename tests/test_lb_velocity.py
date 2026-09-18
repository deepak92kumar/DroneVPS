"""Long-baseline flow velocity: where it is used and how it is mixed in."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flight_daemon import LB_FULL_M, LB_NONE_M, blend_velocity, long_baseline_weight


def test_it_is_used_where_it_was_measured_to_help():
    """Below 30 m the 87 ms baseline measured 1.7-2.3x less noise."""
    assert long_baseline_weight(5.0) == 1.0
    assert long_baseline_weight(20.0) == 1.0
    assert long_baseline_weight(LB_FULL_M) == 1.0


def test_it_is_off_where_it_measured_worse():
    """Above 40 m the gain was 1.09x, and by 50-60 m it was 0.94x - worse than one frame pair."""
    assert long_baseline_weight(LB_NONE_M) == 0.0
    assert long_baseline_weight(60.0) == 0.0
    assert long_baseline_weight(120.0) == 0.0


def test_the_changeover_has_no_step_in_it():
    prev = 1.0
    h = LB_FULL_M
    while h <= LB_NONE_M:
        w = long_baseline_weight(h)
        assert w <= prev + 1e-9
        assert abs(w - prev) < 0.06
        prev = w
        h += 0.5
    assert abs(long_baseline_weight(35.0) - 0.5) < 1e-9


def test_full_weight_gives_the_long_baseline_estimate():
    vn, ve, w = blend_velocity(1.0, 2.0, 0.4, 0.6, 1.0)
    assert (vn, ve, w) == (0.4, 0.6, 1.0)


def test_half_weight_mixes_them():
    vn, ve, w = blend_velocity(1.0, 2.0, 0.0, 0.0, 0.5)
    assert abs(vn - 0.5) < 1e-9 and abs(ve - 1.0) < 1e-9 and w == 0.5


def test_a_missing_long_baseline_sample_falls_back_to_the_live_flow():
    """A gap in the tracker must never blank the velocity sent to the autopilot."""
    assert blend_velocity(1.0, 2.0, None, None, 1.0) == (1.0, 2.0, 0.0)
    assert blend_velocity(1.0, 2.0, math.nan, 0.5, 1.0) == (1.0, 2.0, 0.0)
    assert blend_velocity(1.0, 2.0, 0.5, math.inf, 1.0) == (1.0, 2.0, 0.0)


def test_zero_weight_leaves_the_live_flow_untouched():
    assert blend_velocity(1.5, -2.5, 9.9, 9.9, 0.0) == (1.5, -2.5, 0.0)


def test_the_weight_is_clamped():
    vn, ve, w = blend_velocity(1.0, 1.0, 0.0, 0.0, 5.0)
    assert w == 1.0 and vn == 0.0
    vn, ve, w = blend_velocity(1.0, 1.0, 0.0, 0.0, -3.0)
    assert w == 0.0 and vn == 1.0
