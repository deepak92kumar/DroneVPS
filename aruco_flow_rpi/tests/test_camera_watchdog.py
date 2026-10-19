"""Camera watchdog: tell the pilot within a second when frames stop."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flight_daemon import CAMERA_REPEAT_S, CAMERA_STALL_S, camera_watch_step


def test_frames_flowing_says_nothing():
    st = {}
    assert camera_watch_step(st, now=100.0, last_frame_t=99.99, armed=True) is None


def test_a_stall_in_flight_is_reported_within_a_second():
    """Flight 36: the camera left the USB bus at 25 m and nobody knew for nine seconds."""
    st = {}
    assert camera_watch_step(st, 100.5, 100.0, armed=True) is None           # 0.5 s: fine
    got = camera_watch_step(st, 100.0 + CAMERA_STALL_S, 100.0, armed=True)
    assert got is not None and got[0] == "lost"


def test_a_stall_on_the_ground_is_not_reported():
    st = {}
    assert camera_watch_step(st, 110.0, 100.0, armed=False) is None


def test_the_warning_is_repeated_but_not_spammed():
    st = {}
    assert camera_watch_step(st, 101.0, 100.0, armed=True)[0] == "lost"
    for t in (101.2, 102.0, 104.0, 105.9):
        assert camera_watch_step(st, t, 100.0, armed=True) is None        # inside the repeat gap
    assert camera_watch_step(st, 101.0 + CAMERA_REPEAT_S, 100.0, armed=True)[0] == "lost"


def test_recovery_is_announced_once():
    st = {}
    camera_watch_step(st, 101.5, 100.0, armed=True)                        # lost
    got = camera_watch_step(st, 102.0, 101.95, armed=True)                 # a frame arrived
    assert got is not None and got[0] == "back"
    assert camera_watch_step(st, 102.1, 102.05, armed=True) is None        # and not again


def test_no_frame_yet_says_nothing():
    """At start-up there has been no frame at all; that is the camera connecting, not a stall."""
    assert camera_watch_step({}, 500.0, None, armed=True) is None


def test_the_reported_age_is_how_long_the_camera_has_been_silent():
    st = {}
    kind, age = camera_watch_step(st, 103.0, 100.0, armed=True)
    assert kind == "lost" and abs(age - 3.0) < 1e-9
