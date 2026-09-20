"""Marker camera (RGB crop): intrinsics, frame change, frame-time attitude, and source switching."""
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flight_daemon import (MARKER_CAM_FRESH_S, MARKER_MIN_PX, attitude_at_time, centre_marker_pick,
                           crop_intrinsics, marker_fix_step, marker_offset_ned, to_flow_camera_frame)

# Measured on this unit (19 Sep): CAM_A at 2560x1440 and the extrinsics CAM_A -> CAM_B.
K_ISP = np.array([[1943.4, 0.0, 1301.4], [0.0, 1941.6, 653.1], [0.0, 0.0, 1.0]])
R_AB = np.array([[0.999945, 0.008817, 0.005667],
                 [-0.00878, 0.999941, -0.006446],
                 [-0.005724, 0.006396, 0.999963]])
T_AB = np.array([0.03742, -0.00061, -0.00023])


def test_crop_moves_only_the_principal_point():
    K = crop_intrinsics(K_ISP, 640, 320)
    assert K[0, 0] == K_ISP[0, 0] and K[1, 1] == K_ISP[1, 1]
    assert abs(K[0, 2] - 661.4) < 1e-9 and abs(K[1, 2] - 333.1) < 1e-9


def test_frame_change_is_rotation_then_offset():
    p = np.array([1.0, 2.0, 95.0])
    got = to_flow_camera_frame(p, R_AB, T_AB)
    assert np.allclose(got, R_AB @ p + T_AB)


def _project_and_solve(p_a, size=3.6):
    """Render a marker's corners seen by the crop camera at p_a (its frame) and solve the pose."""
    K = crop_intrinsics(K_ISP, 640, 320)
    h = size / 2.0
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32)
    img, _ = cv2.projectPoints(obj, np.zeros(3), np.asarray(p_a, dtype=np.float64), K, None)
    ok, _, tv = cv2.solvePnP(obj, img.reshape(-1, 2).astype(np.float32), K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    assert ok
    side = float(np.mean([np.linalg.norm(img[i, 0] - img[(i + 1) % 4, 0]) for i in range(4)]))
    return tv.ravel(), side


def test_at_95_m_the_marker_is_big_and_lands_where_it_is():
    """95 m straight down, drone level, marker 2 m north of the drone in the flow camera's frame."""
    p_b = np.array([0.0, -2.0, 95.0])                   # camera +y is the tail: marker ahead = -y
    p_a = R_AB.T @ (p_b - T_AB)                          # the same point in the RGB camera's frame
    tv, side = _project_and_solve(p_a)
    assert side > 70.0                                   # 14 px on the 640x400 mono
    got = to_flow_camera_frame(tv, R_AB, T_AB)
    n, e = marker_offset_ned(got, 0.0, 0.0, 0.0)
    assert abs(n - (-2.0)) < 0.01 and abs(e) < 0.01      # drone is 2 m south of the marker


def test_skipping_the_camera_tilt_would_cost_most_of_a_metre():
    """0.49 deg of tilt between the cameras is 0.8 m at 95 m."""
    p_b = np.array([0.0, 0.0, 95.0])
    tv, _ = _project_and_solve(R_AB.T @ (p_b - T_AB))
    wrong = marker_offset_ned(tv, 0.0, 0.0, 0.0)
    assert 0.7 < math.hypot(*wrong) < 0.9


HIST = [(10.00, 1.0, -2.0, 350.0), (10.05, 3.0, -1.0, 355.0), (10.10, 5.0, 0.0, 5.0)]


def test_attitude_is_interpolated_at_the_frame_time():
    r, p, y = attitude_at_time(HIST, 10.025)
    assert abs(r - 2.0) < 1e-9 and abs(p + 1.5) < 1e-9 and abs(y - 352.5) < 1e-9


def test_yaw_is_interpolated_across_north():
    r, p, y = attitude_at_time(HIST, 10.075)
    assert abs(((y + 180.0) % 360.0) - 180.0) < 1e-9    # 355 -> 5 passes through 0, not 180


def test_no_history_near_the_frame_gives_none():
    assert attitude_at_time([], 10.0) is None
    assert attitude_at_time(HIST, 9.0) is None
    assert attitude_at_time(HIST, 11.0) is None
    assert attitude_at_time([(10.0, 0, 0, 0), (11.0, 5, 5, 5)], 10.5) is None   # a 1 s hole


def test_only_centre_markers_are_picked():
    c = np.zeros((4, 2))
    cands = [(6, 80.0, c), (45, 60.0, c), (16, 20.0, c)]
    assert centre_marker_pick(cands)[0] == 45
    assert centre_marker_pick([(45, MARKER_MIN_PX - 1, c)]) is None


def test_switching_camera_does_not_move_the_position():
    """The two cameras' fixes differ by their calibration (0.11-0.21 m on 20 Sep); do not chase it."""
    st = {}
    marker_fix_step(st, 0.0, 0.0, (0.0, 0.0), 0.04, source="lo")
    for _ in range(20):
        n0, e0, _ = marker_fix_step(st, 0.0, 0.0, (0.0, 0.0), 0.04, source="lo")
    n1, e1, how = marker_fix_step(st, 0.0, 0.0, (0.4, -0.3), 0.04, source="hi")
    assert how == "source" and (n1, e1) == (n0, e0)


def test_swapping_cameras_over_and_over_does_not_drift():
    """Flight 45 swapped source 20 times in five minutes. Each camera keeps its own anchor, so
    the position must come out exactly where it went in."""
    st = {}
    marker_fix_step(st, 0.0, 0.0, (0.0, 0.0), 0.04, source="lo")
    n0, e0, _ = marker_fix_step(st, 0.0, 0.0, (0.0, 0.0), 0.04, source="lo")
    for i in range(40):
        src, mk = ("hi", (0.4, -0.3)) if i % 2 else ("lo", (0.0, 0.0))
        n, e, _ = marker_fix_step(st, 0.0, 0.0, mk, 0.04, source=src)
    assert abs(n - n0) < 1e-9 and abs(e - e0) < 1e-9


def test_each_camera_still_corrects_drift_on_its_own_anchor():
    st = {}
    marker_fix_step(st, 0.0, 0.0, (0.0, 0.0), 0.04, source="lo")
    for i in range(1, 300):
        n, e, _ = marker_fix_step(st, 2.0 * i / 299, 0.0, (0.4, -0.3), 0.04, source="hi")
    assert abs(n - 0.0) < 0.05 and abs(e - 0.0) < 0.05        # held at the hi anchor, flow drift gone


def test_the_flow_camera_is_not_locked_out_while_the_marker_camera_is_quiet():
    """Flight 48: the marker camera went quiet for 0.6-5 s at a time and the flow camera saw the
    marker throughout. A long preference window threw those fixes away and the hover broke."""
    assert MARKER_CAM_FRESH_S <= 0.3
