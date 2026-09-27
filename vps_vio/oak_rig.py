"""The one verified OAK-D Pro + cuVSLAM setup. Single source of truth.

Every constant here was established by measurement on this aircraft during the
27 Sep bring-up, not guessed. The rig config was duplicated across four scripts
during debugging; it lives here now so it cannot drift apart.

Measured facts behind these values:
  * Distortion: Polynomial with getDistortionCoefficients()[:8], camera order
    CAM_B,CAM_C, right camera placed by getCameraExtrinsics(CAM_C, CAM_B).
    The opposite extrinsics direction gives 206 m of stationary drift.
  * IMU noise x10: datasheet-class values make cuVSLAM over-trust the BNO086 and
    diverge (a real 0.5 m move read 5.07 m). x10 tracks stereo-only almost
    exactly. The BNO086's jittery timestamps have to be absorbed somewhere.
  * Exposure 4000 us / ISO 800: no blown highlights, matched left/right means,
    lowest stationary drift (0.8 cm at ~1 m over a textured scene). Shorter
    exposure went too dark and diverged; longer blew out and unbalanced L/R.
"""
from __future__ import annotations
import numpy as np
import depthai as dai
import cuvslam as vslam
from scipy.spatial.transform import Rotation

W, H = 640, 400
FPS = 30
EXPOSURE_US = 4000
ISO = 800
IMU_HZ = 200
BORDER_PX = 24          # mask distorted edges; cuVSLAM ignores features here

# IMU noise, tuned by measurement (sweep_imu.py). See module docstring.
IMU_NOISE_SCALE = 10.0
IMU_ACC_NOISE_DENSITY = 2.8e-3 * IMU_NOISE_SCALE
IMU_ACC_RANDOM_WALK = 8.6e-4 * IMU_NOISE_SCALE
IMU_GYR_NOISE_DENSITY = 1.6e-4 * IMU_NOISE_SCALE
IMU_GYR_RANDOM_WALK = 2.2e-5 * IMU_NOISE_SCALE

_LEFT = dai.CameraBoardSocket.CAM_B
_RIGHT = dai.CameraBoardSocket.CAM_C


def transform_to_pose(mat) -> vslam.Pose:
    """OAK 4x4 extrinsics (translation in cm) -> cuVSLAM Pose (metres, quat xyzw)."""
    m = np.asarray(mat, dtype=float)
    return vslam.Pose(rotation=Rotation.from_matrix(m[:3, :3]).as_quat(),
                      translation=m[:3, 3] / 100.0)


def build_rig(calib, with_imu: bool) -> vslam.Rig:
    """Build the verified cuVSLAM Rig from the device's own calibration."""
    cams = []
    for i, sock in enumerate((_LEFT, _RIGHT)):
        K = np.array(calib.getCameraIntrinsics(sock, W, H))
        c = vslam.Camera()
        c.size = (W, H)
        c.focal = (K[0][0], K[1][1])
        c.principal = (K[0][2], K[1][2])
        c.distortion = vslam.Distortion(vslam.Distortion.Model.Polynomial,
                                        list(calib.getDistortionCoefficients(sock))[:8])
        c.rig_from_camera = (
            vslam.Pose(rotation=[0, 0, 0, 1], translation=[0, 0, 0]) if i == 0
            else transform_to_pose(calib.getCameraExtrinsics(_RIGHT, _LEFT)))
        c.border_top = c.border_bottom = c.border_left = c.border_right = BORDER_PX
        cams.append(c)

    rig = vslam.Rig()
    rig.cameras = cams
    if with_imu:
        imu = vslam.ImuCalibration()
        imu.rig_from_imu = transform_to_pose(calib.getImuToCameraExtrinsics(_LEFT))
        imu.frequency = float(IMU_HZ)
        imu.accelerometer_noise_density = IMU_ACC_NOISE_DENSITY
        imu.accelerometer_random_walk = IMU_ACC_RANDOM_WALK
        imu.gyroscope_noise_density = IMU_GYR_NOISE_DENSITY
        imu.gyroscope_random_walk = IMU_GYR_RANDOM_WALK
        rig.imus = [imu]
    else:
        rig.imus = []
    return rig


def build_pipeline(with_imu: bool, fps: int = FPS,
                   exposure_us: int = EXPOSURE_US, iso: int = ISO) -> dai.Pipeline:
    """DepthAI v2 pipeline: synced mono pair (+ raw IMU). Manual exposure on
    purpose — auto-exposure flicker and motion blur are what break tracking."""
    p = dai.Pipeline()
    for name, sock in (("left", _LEFT), ("right", _RIGHT)):
        m = p.create(dai.node.MonoCamera)
        m.setBoardSocket(sock)
        m.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        m.setFps(fps)
        m.initialControl.setManualExposure(exposure_us, iso)
        x = p.create(dai.node.XLinkOut)
        x.setStreamName(name)
        m.out.link(x.input)
    if with_imu:
        imu = p.create(dai.node.IMU)
        imu.enableIMUSensor([dai.IMUSensor.ACCELEROMETER_RAW,
                             dai.IMUSensor.GYROSCOPE_RAW], IMU_HZ)
        imu.setBatchReportThreshold(1)
        imu.setMaxBatchReports(20)
        xi = p.create(dai.node.XLinkOut)
        xi.setStreamName("imu")
        imu.out.link(xi.input)
    return p


def make_tracker(rig: vslam.Rig, inertial: bool) -> vslam.Tracker:
    cfg = vslam.Tracker.OdometryConfig(
        odometry_mode=(vslam.Tracker.OdometryMode.Inertial if inertial
                       else vslam.Tracker.OdometryMode.Multicamera),
        use_gpu=True,                    # run on the Orin GPU
        rectified_stereo_camera=False,   # we feed raw mono + the distortion model
        async_sba=False,
        use_motion_model=True,
        use_denoising=False,
    )
    return vslam.Tracker(rig, cfg)


def td_ns(td) -> int:
    """depthai timedelta -> integer nanoseconds on the device clock."""
    return int(td.total_seconds() * 1e9)
