"""End-to-end pipeline test with synthetic VIO — no hardware, no FC."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VpsConfig       # noqa: E402
from vio_source import SyntheticSource, VioPose  # noqa: E402
from vps_node import VpsNode       # noqa: E402


def test_dryrun_wiring_runs():
    node = VpsNode(VpsConfig(), SyntheticSource(), bridge=None)
    p = node.source.read()
    world = node.step(p)
    assert world is not None and world.shape == (3,)
    assert node._origin_set


def test_origin_makes_first_output_near_zero():
    node = VpsNode(VpsConfig(), SyntheticSource(radius=2.0), bridge=None)
    p = node.source.read()
    world = node.step(p)
    # first sample is the origin -> world ~ 0
    np.testing.assert_allclose(world, [0, 0, 0], atol=1e-6)


def test_marker_corrects_vio_drift():
    cfg = VpsConfig()
    # strong constant drift so VIO-only would run away
    src = SyntheticSource(radius=0.0, drift_mps=(0.5, 0.0, 0.0), rate_hz=20.0)
    node = VpsNode(cfg, src, bridge=None)

    # truth: aircraft is stationary at world origin (0,0). Marker keeps saying so.
    # Check the magnitude, not one axis: which body axis the VIO drift lands on
    # depends on the measured mounting rotation R, so an axis-specific assertion
    # would silently pass for the wrong reason if R changes.
    max_err_with_marker = 0.0
    for i in range(400):                       # 20 s at 20 Hz
        p = src.read()
        world = node.step(p, marker_world_ned=[0.0, 0.0, 0.0])
        if i > 100:                            # after convergence
            max_err_with_marker = max(max_err_with_marker, float(np.linalg.norm(world)))
    # with the marker anchoring, the reported world position stays bounded despite
    # VIO drifting ~0.5 m/s. Without correction it would be ~10 m by now.
    assert max_err_with_marker < 1.0, max_err_with_marker


def test_no_marker_lets_drift_through():
    # sanity: without marker fixes the VIO drift is NOT corrected (proves the
    # marker is what bounds it, not some accidental clamp).
    src = SyntheticSource(radius=0.0, drift_mps=(0.5, 0.0, 0.0), rate_hz=20.0)
    node = VpsNode(VpsConfig(), src, bridge=None)
    world = None
    for _ in range(400):
        world = node.step(src.read())          # no marker
    # magnitude, not a fixed axis — see note in test_marker_corrects_vio_drift
    assert float(np.linalg.norm(world)) > 5.0   # ~10 m of uncorrected drift


def test_untracked_pose_ignored():
    node = VpsNode(VpsConfig(), SyntheticSource(), bridge=None)
    bad = VioPose(t=0.0, pos=np.zeros(3), quat=np.array([1, 0, 0, 0.0]), tracking=False)
    assert node.step(bad) is None
    assert not node._origin_set


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
