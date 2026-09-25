"""Unit tests for the VIO->NED transform. No hardware, no FC, no camera."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vio_frame import VioToNed, quat_to_yaw, yaw_rot, wrap_pi  # noqa: E402


def test_identity_passthrough():
    t = VioToNed()  # R = I, no heading align, no origin
    np.testing.assert_allclose(t.to_ned([1, 2, 3]), [1, 2, 3])


def test_origin_subtraction():
    t = VioToNed()
    t.set_origin([10, 20, 5])
    np.testing.assert_allclose(t.to_ned([12, 23, 9]), [2, 3, 4])
    # at the origin itself the output is exactly zero
    np.testing.assert_allclose(t.to_ned([10, 20, 5]), [0, 0, 0], atol=1e-12)


def test_heading_align_90deg():
    # VIO reports 1 m along its x. With a +90 deg alignment, North should map to East.
    t = VioToNed(heading_align_rad=np.pi / 2)
    ned = t.to_ned([1, 0, 0])
    np.testing.assert_allclose(ned, [0, 1, 0], atol=1e-9)


def test_axis_permutation_matrix():
    # Example VIO frame: x=East, y=North, z=Up  ->  NED
    R = np.array([[0, 1, 0],    # N =  vio.y
                  [1, 0, 0],    # E =  vio.x
                  [0, 0, -1]])  # D = -vio.z
    t = VioToNed(R_vio_to_ned=R)
    # a step of +1 East, +2 North, +3 Up in VIO (x,y,z)=(1,2,3)
    np.testing.assert_allclose(t.to_ned([1, 2, 3]), [2, 1, -3])


def test_rigid_transform_preserves_distance():
    rng = np.random.default_rng(0)
    R = yaw_rot(0.7)  # any rotation about down
    t = VioToNed(R_vio_to_ned=R, heading_align_rad=-0.3)
    a, b = rng.normal(size=3), rng.normal(size=3)
    d_in = np.linalg.norm(a - b)
    d_out = np.linalg.norm(t.to_ned(a) - t.to_ned(b))
    assert abs(d_in - d_out) < 1e-9


def test_from_compass_alignment():
    # VIO thinks it faces yaw=0, compass says the nose is at 30 deg from North.
    t = VioToNed.from_compass(compass_yaw_rad=np.radians(30), vio_yaw_at_init_rad=0.0)
    assert abs(t.yaw_to_ned(0.0) - np.radians(30)) < 1e-9


def test_quat_to_yaw_known():
    # 90 deg yaw quaternion about z: (w,x,y,z) = (cos45, 0, 0, sin45)
    q = (np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4))
    assert abs(quat_to_yaw(*q) - np.pi / 2) < 1e-9


def test_wrap_pi():
    # 3*pi and -3*pi are both the +/-pi heading (same direction); check magnitude.
    assert abs(abs(wrap_pi(3 * np.pi)) - np.pi) < 1e-9
    assert abs(abs(wrap_pi(-3 * np.pi)) - np.pi) < 1e-9
    assert abs(wrap_pi(0.0)) < 1e-12
    assert abs(wrap_pi(np.radians(200)) - np.radians(200 - 360)) < 1e-9


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
