"""Drone position relative to a marker, from the PnP translation and the attitude."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flight_daemon import marker_offset_ned, marker_offset_yaw_only


def r_nb(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch),
                              math.cos(yaw), math.sin(yaw))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def tvec_for(north, east, height, roll, pitch, yaw):
    """What the camera would measure for a drone `north`/`east` of the marker, `height` above it."""
    m_ned = np.array([-north, -east, height])            # marker relative to the drone
    m_body = r_nb(roll, pitch, yaw).T @ m_ned             # forward, right, down
    return np.array([m_body[1], -m_body[0], m_body[2]])   # camera: right, tail, down


def close(a, b, tol=1e-6):
    return abs(a[0] - b[0]) < tol and abs(a[1] - b[1]) < tol


def test_level_matches_the_code_that_has_been_flying():
    """Level, the new maths must give exactly what the flown yaw-only code gives."""
    for yaw_deg in (0, 37, 90, -120, 179):
        tv = tvec_for(3.0, -2.0, 20.0, 0.0, 0.0, math.radians(yaw_deg))
        assert close(marker_offset_ned(tv, 0.0, 0.0, math.radians(yaw_deg)),
                     marker_offset_yaw_only(tv, math.radians(yaw_deg)))


def test_the_true_position_comes_back_at_any_attitude():
    for north, east, h, r, p, y in [(0, 0, 20, 10, 0, 0), (4, -3, 26, -8, 6, 45),
                                    (-7, 2, 60, 12, -9, -100), (1.5, 1.5, 5, 3, 3, 170)]:
        tv = tvec_for(north, east, h, math.radians(r), math.radians(p), math.radians(y))
        got = marker_offset_ned(tv, math.radians(r), math.radians(p), math.radians(y))
        assert close(got, (north, east), 1e-6), (north, east, got)


def test_a_tilt_is_not_mistaken_for_movement():
    """Flight 36 swung with 3-13 degrees of roll at 26 m. Directly over the marker, a 10 degree roll
    makes yaw-only maths report the drone h*sin(10 deg) = 4.5 m away; the full attitude reports it
    where it is."""
    tv = tvec_for(0.0, 0.0, 26.0, math.radians(10), 0.0, 0.0)
    full = marker_offset_ned(tv, math.radians(10), 0.0, 0.0)
    old = marker_offset_yaw_only(tv, 0.0)
    assert math.hypot(*full) < 1e-6
    assert abs(math.hypot(*old) - 26.0 * math.sin(math.radians(10))) < 0.01    # 4.5 m of error


def test_the_error_it_removes_grows_with_height():
    errs = []
    for h in (5.0, 26.0, 60.0):
        tv = tvec_for(0.0, 0.0, h, math.radians(8), math.radians(5), math.radians(30))
        errs.append(math.hypot(*marker_offset_yaw_only(tv, math.radians(30))))
    assert errs[0] < errs[1] < errs[2]
    assert errs[2] > 9.0                                   # about 10 m of false offset at 60 m
