"""Unit tests for AnchorFuser (VIO + ArUco fusion). No hardware."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from anchor import AnchorFuser  # noqa: E402


def test_starts_at_zero_offset():
    a = AnchorFuser()
    np.testing.assert_allclose(a.fused([1, 2, 3]), [1, 2, 3])


def test_first_small_fix_slews_not_snaps():
    a = AnchorFuser(corr_rate_mps=4.0, reanchor_m=6.0)
    # marker says we are 1 m North of where VIO+offset thinks -> residual (1,0,0)
    a.update_marker(vio_ned=[0, 0, 0], marker_world_ned=[1, 0, 0], dt_s=0.1)
    # step limited to 4.0 * 0.1 = 0.4 m
    np.testing.assert_allclose(a.offset, [0.4, 0, 0], atol=1e-9)
    assert a.reanchor_count == 0


def test_large_residual_reanchors():
    a = AnchorFuser(corr_rate_mps=4.0, reanchor_m=6.0)
    a.update_marker(vio_ned=[0, 0, 0], marker_world_ned=[10, 0, 0], dt_s=0.1)
    # 10 m > 6 m -> snap the whole residual
    np.testing.assert_allclose(a.offset, [10, 0, 0], atol=1e-9)
    assert a.reanchor_count == 1


def test_slew_converges_over_many_fixes():
    a = AnchorFuser(corr_rate_mps=4.0, reanchor_m=6.0)
    # constant true offset of 2 m East, VIO fixed at origin, 10 Hz marker for 3 s
    for _ in range(30):
        a.update_marker(vio_ned=[0, 0, 0], marker_world_ned=[0, 2, 0], dt_s=0.1)
    np.testing.assert_allclose(a.offset, [0, 2, 0], atol=1e-6)
    assert a.reanchor_count == 0


def test_step_never_exceeds_rate_limit():
    a = AnchorFuser(corr_rate_mps=4.0, reanchor_m=100.0)  # high reanchor so it always slews
    prev = a.offset.copy()
    a.update_marker(vio_ned=[0, 0, 0], marker_world_ned=[5, 5, 0], dt_s=0.1)
    step = np.linalg.norm(a.offset - prev)
    assert step <= 4.0 * 0.1 + 1e-9


def test_reanchor_is_flat_not_height_scaled():
    # Same horizontal residual must behave identically regardless of Z (height).
    a_low = AnchorFuser()
    a_high = AnchorFuser()
    r_low = a_low.update_marker([0, 0, -10], [7, 0, -10], 0.1)   # 10 m alt
    r_high = a_high.update_marker([0, 0, -110], [7, 0, -110], 0.1)  # 110 m alt
    assert a_low.reanchor_count == a_high.reanchor_count == 1
    np.testing.assert_allclose(a_low.offset[:2], a_high.offset[:2])


def test_fused_tracks_vio_between_fixes():
    a = AnchorFuser()
    a.update_marker([0, 0, 0], [3, 0, 0], 10.0)  # big dt -> offset reaches 3 (rate*dt huge)
    # VIO now reports drift to (0.5, 0, 0); world should be 3.5
    np.testing.assert_allclose(a.fused([0.5, 0, 0]), [3.5, 0, 0], atol=1e-9)


def test_invalid_config_rejected():
    for kw in ({"corr_rate_mps": 0}, {"reanchor_m": -1}):
        try:
            AnchorFuser(**kw); assert False, "should have raised"
        except ValueError:
            pass


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn(); passed += 1; print(f"  PASS  {fn.__name__}")
        except Exception:
            print(f"  FAIL  {fn.__name__}"); traceback.print_exc()
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)
