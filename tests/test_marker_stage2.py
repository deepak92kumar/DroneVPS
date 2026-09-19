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
    assert not good(height_m=20.0, marker_down_m=20.0)[0]


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
