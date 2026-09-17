"""Flow height above the old 50 m ceiling: lidar -> barometer handover."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flight_daemon import (LIDAR_FADE_TOP_M, LIDAR_FULL_TRUST_M, MAX_FLOW_ALT_M,
                           lidar_agrees_with_baro, lidar_weight, track_baro_offset)


def test_lidar_carries_the_height_below_the_fade():
    assert lidar_weight(5.0) == 1.0
    assert lidar_weight(30.0) == 1.0
    assert lidar_weight(LIDAR_FULL_TRUST_M) == 1.0


def test_barometer_carries_the_height_above_the_fade():
    assert lidar_weight(LIDAR_FADE_TOP_M) == 0.0
    assert lidar_weight(100.0) == 0.0


def test_the_handover_has_no_step_in_it():
    mid = 0.5 * (LIDAR_FULL_TRUST_M + LIDAR_FADE_TOP_M)
    assert abs(lidar_weight(mid) - 0.5) < 1e-9
    prev = 1.0
    h = LIDAR_FULL_TRUST_M
    while h <= LIDAR_FADE_TOP_M:
        w = lidar_weight(h)
        assert w <= prev + 1e-9              # never rises
        assert abs(w - prev) < 0.05          # and never jumps
        prev = w
        h += 0.5


def test_61_metres_is_no_longer_reported_as_50():
    """17 Sep: the drone reached 61.9 m and the flow was scaled as if it were at 50 m."""
    height = min(61.85, MAX_FLOW_ALT_M)
    assert height == 61.85
    assert MAX_FLOW_ALT_M >= 120.0


def test_offset_carries_the_lidar_reference_onto_the_barometer():
    # Ground has fallen away 3 m since takeoff: lidar reads 40, barometer 37.
    offset = None
    for _ in range(600):
        offset = track_baro_offset(offset, 40.0, 37.0, trusted=True)
    assert abs(offset - 3.0) < 0.05
    # At the top of the fade the barometer alone must reproduce the same AGL.
    assert abs((72.0 + offset) - 75.0) < 0.05


def test_offset_is_not_moved_by_an_untrusted_sample():
    offset = track_baro_offset(None, 40.0, 37.0, trusted=True)
    same = track_baro_offset(offset, 900.0, 37.0, trusted=False)
    assert same == offset


def test_offset_barely_moves_for_one_bad_trusted_sample():
    offset = 3.0
    moved = track_baro_offset(offset, 48.0, 37.0, trusted=True)   # 8 m of lidar error
    assert abs(moved - offset) < 0.05


def test_a_lidar_spike_is_rejected_against_the_barometer():
    """Log 29 carried an 81.7 m reading while the drone hovered at about 4 m."""
    assert not lidar_agrees_with_baro(81.68, 4.0)
    assert not lidar_agrees_with_baro(40.15, 4.0)


def test_ordinary_lidar_noise_is_accepted():
    assert lidar_agrees_with_baro(4.6, 4.0)
    assert lidar_agrees_with_baro(31.0, 30.0)
    assert lidar_agrees_with_baro(58.0, 52.0)     # 6 m at 52 m is within tolerance


def test_no_barometer_reference_yet_means_trust_the_lidar():
    assert lidar_agrees_with_baro(20.0, None)


# --- learned from flight 33 (17 Sep 2026) -------------------------------------------------

from flight_daemon import baro_offset_trusted, height_source_changed


def test_offset_is_not_learned_while_climbing():
    """Flight 33: the offset swung 0.26-1.60 m in fifteen seconds of climb, because the
    autopilot's height lags the lidar and the difference becomes the climb rate."""
    climbing = baro_offset_trusted(lidar_agl_m=57.0, lidar_prev_m=55.0, dt_s=0.5,
                                   weight=1.0, airborne=True, armed=True)   # 4 m/s
    assert climbing is False


def test_offset_is_learned_while_holding_altitude():
    holding = baro_offset_trusted(lidar_agl_m=55.1, lidar_prev_m=55.0, dt_s=0.5,
                                  weight=1.0, airborne=True, armed=True)    # 0.2 m/s
    assert holding is True


def test_offset_is_not_learned_on_the_ground_or_inside_the_fade():
    assert not baro_offset_trusted(55.0, 55.0, 0.5, 1.0, airborne=False, armed=True)
    assert not baro_offset_trusted(55.0, 55.0, 0.5, 1.0, airborne=True, armed=False)
    assert not baro_offset_trusted(60.0, 60.0, 0.5, 0.6, airborne=True, armed=True)


def test_offset_needs_a_previous_sample():
    assert not baro_offset_trusted(55.0, None, 0.5, 1.0, True, True)
    assert not baro_offset_trusted(55.0, 55.0, 0.0, 1.0, True, True)


def test_the_55_m_boundary_no_longer_fills_the_log():
    """Flight 33 wrote twenty lines flip-flopping between TF03 and TF03+BARO(100%)."""
    assert not height_source_changed("TF03+BARO", "TF03", 1.0, 1.0)
    assert not height_source_changed("TF03", "TF03+BARO", 1.0, 1.0)
    assert not height_source_changed("TF03+BARO", "TF03", 0.99, 1.0)


def test_a_real_handover_is_still_reported():
    assert height_source_changed("TF03+BARO", "TF03", 0.70, 1.0)
    assert height_source_changed("BARO_AGL", "TF03+BARO", 0.0, 0.5)


def test_the_first_sample_is_reported():
    assert height_source_changed("TF03", "", 1.0, None)
