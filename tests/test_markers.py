"""Which ArUco marker the daemon picks, and at what size it reads it."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flight_daemon import MARKER_MIN_PX, MARKER_SIZES_M, marker_candidates, pick_marker


def square(cx, cy, side):
    h = side / 2.0
    return np.array([[cx - h, cy - h], [cx + h, cy - h], [cx + h, cy + h], [cx - h, cy + h]],
                    dtype=np.float32)


def test_the_field_and_the_pad_are_all_known():
    for mid in (45, 16, 0, 10, 18, 22, 1, 2, 3, 4, 5, 6, 11, 12, 13):
        assert mid in MARKER_SIZES_M
    assert MARKER_SIZES_M[45] == 3.60
    assert MARKER_SIZES_M[16] == 0.72
    assert MARKER_SIZES_M[0] == 0.12
    assert MARKER_SIZES_M[10] == 1.20


def test_the_pad_is_no_longer_id_0():
    """ID 0 is the 0.12 m marker in the field target now; the 1.13 m pad is ID 15."""
    assert MARKER_SIZES_M[15] == 1.13
    assert MARKER_SIZES_M[0] != 1.13


def test_unknown_ids_are_ignored():
    ids = np.array([[7], [45]])
    corners = [square(300, 200, 40), square(320, 210, 90)]
    got = marker_candidates(ids, corners)
    assert [c[0] for c in got] == [45]


def test_candidates_come_back_biggest_first():
    ids = np.array([[0], [45], [16]])
    corners = [square(320, 200, 12), square(320, 200, 300), square(320, 200, 60)]
    got = marker_candidates(ids, corners)
    assert [c[0] for c in got] == [45, 16, 0]


def test_the_biggest_whole_marker_is_used():
    """At 20 m over the field target, ID 45 is large and ID 16 small: take ID 45."""
    ids = np.array([[45], [16]])
    corners = [square(320, 200, 71), square(320, 200, 14)]
    assert pick_marker(marker_candidates(ids, corners))[0] == 45


def test_the_next_marker_down_takes_over_when_the_outer_one_leaves_the_frame():
    """Close to the ground ID 45 is wider than the sensor, so the detector never returns it."""
    ids = np.array([[16], [0]])                       # ID 45 absent: it did not decode
    corners = [square(320, 200, 114), square(320, 200, 19)]
    assert pick_marker(marker_candidates(ids, corners))[0] == 16


def test_a_large_detected_marker_is_not_thrown_away():
    """It was detected, so its four corners were in frame - taking it is right."""
    ids = np.array([[45], [16]])
    corners = [square(320, 200, 407), square(320, 200, 66)]
    assert pick_marker(marker_candidates(ids, corners))[0] == 45


def test_markers_below_the_decode_floor_are_not_used():
    ids = np.array([[0]])
    corners = [square(320, 200, 4)]
    assert pick_marker(marker_candidates(ids, corners)) is None


def test_the_decode_floor_is_the_measured_one():
    assert MARKER_MIN_PX == 11.0
    ids = np.array([[16]])
    assert pick_marker(marker_candidates(ids, [square(320, 200, 11.5)]))[0] == 16
    assert pick_marker(marker_candidates(ids, [square(320, 200, 10.0)])) is None


def test_nothing_detected_is_handled():
    assert marker_candidates(None, None) == []
    assert pick_marker([]) is None


def test_the_pad_can_still_be_chosen_on_its_own():
    ids = np.array([[15]])
    assert pick_marker(marker_candidates(ids, [square(320, 200, 120)]))[0] == 15


def test_a_size_override_reaches_the_table():
    sizes = dict(MARKER_SIZES_M)
    sizes[0] = 1.13                      # the old pad, before it is reprinted as ID 15
    cands = marker_candidates(np.array([[0]]), [square(320, 200, 90)], sizes)
    assert cands[0][0] == 0
    assert sizes[0] == 1.13
