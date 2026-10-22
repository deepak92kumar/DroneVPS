"""All VPS/VIO tunables in one place. NOT active until Phase 1 (after demo).

Defaults mirror the proven marker-map behaviour from the live daemon. The two
correction constants MUST stay flat (never scale with height): scaling them cost
two flights and was reverted (see memory flow-velocity-above-40m / d75a29f).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class VpsConfig:
    # --- FC link (reuses the existing bridge; same values the daemon uses) ---
    fc_port: str = "/dev/ttyTHS1"
    # 921600, NOT 115200. Read from the FC itself in log 00000195.BIN:
    #   SERIAL2_BAUD 921   SERIAL2_PROTOCOL 2 (MAVLink2)
    # TELEM2 is where the companion sits, and flight_daemon.py's own comment
    # records why it moved: at 115200 "the link was near full and the FC
    # forwarded it to the RC link", which made parameter download crawl.
    fc_baud: int = 921600

    # --- VIO -> BODY axis mapping (MEASURED 27 Sep, no longer a placeholder) ---
    # Derived from geometry, not from drift-prone motion tests:
    #   * cam_z is DOWN  - the accelerometer's gravity vector, rotated into the
    #     camera frame by the IMU->CAM_B extrinsics, lands on cam_z at 99.8%.
    #   * the drone's nose points at the TOP of the image, and cam_y is image
    #     DOWN, so body forward = -vio_y.
    #   * body right = down x forward = +vio_x (right-handed, no measurement needed).
    # Cross-checked against the one clean motion run: a forward move gave
    # vio_y = -0.225 with 100% on that axis, exactly as this matrix predicts.
    # Rows map a VIO vector to [forward, right, down] in the body frame; the
    # compass supplies body->NED heading at runtime (VioToNed.heading_align_rad).
    R_vio_to_ned: list = field(default_factory=lambda: [[0.0, -1.0, 0.0],
                                                        [1.0,  0.0, 0.0],
                                                        [0.0,  0.0, 1.0]])

    # --- ArUco-as-anchor fusion (mirror of marker_fix_step; keep FLAT) ---
    corr_rate_mps: float = 4.0     # max slew speed of the offset correction, m/s. FLAT.
    reanchor_m: float = 6.0        # horizontal residual above which we snap. FLAT.
    marker_timeout_s: float = 2.0  # ignore marker fixes older than this

    # --- covariance sent to EKF (quality-dependent, like the daemon) ---
    pos_cov_good: float = 0.05     # m^2 when VIO tracking is healthy
    pos_cov_poor: float = 0.50     # m^2 when tracking is weak

    # --- VIO health gating ---
    min_vio_rate_hz: float = 15.0  # below this we stop trusting X/Y
    send_rate_hz: float = 20.0     # how often we push VISION_POSITION_ESTIMATE

    # ---------------------------------------------------------------------
    # Values recovered from the 28 Sep - 13 Oct status reports after the SSD
    # that held the live code was lost. Each one carries the date it was
    # changed and whether it had flown, because that is the only provenance
    # left for them - the commits themselves are gone.
    # ---------------------------------------------------------------------

    # --- map matcher ---
    matcher: str = "xfeat"         # 4 Oct: NCC -> XFeat. Flown, kept.
                                   # Map matching went 45% -> 75% on the swap.
    map_min_alt_m: float = 60.0    # 3 Oct: 80 -> 60 m. Flown, kept. At 80 m the
                                   # map never started. The map runs ABOVE this
                                   # height and NOT below it, which is why the
                                   # descent is uncorrected: error went 4.13 ->
                                   # 26.0 m in the 12 s of a 6 m/s descent.
    map_window_alt_frac: float = 0.60   # 4 Oct: window was a fixed 100 m, which
                                   # had the matcher guessing outside what the
                                   # camera could see. Now 0.60 x altitude.
    fix_period_s: float = 2.0      # 5 Oct: 0.4 -> 2.0 s. At 0.4 s the map pushed
                                   # a STATIONARY aircraft at 1.18 m/s above 80 m
                                   # while the drift it exists to catch is
                                   # 0.15-0.37 m/s. At 2 s it pushes 0.26, and the
                                   # worker drops from 75% of a core to 17%.

    # --- recorder ---
    image_hz: float = 2.0          # 5 Oct: JPEGs gated separately from CSV rows
                                   # at 4 Hz; 8 Oct: halved to 2 Hz. index.csv
                                   # still logs EVERY frame - the rows carry the
                                   # conclusions, the JPEGs only answer the few
                                   # questions that need pixels back.

    # --- velocity sent to the FC ---
    # 4 Oct: the hover deadband was flat 0.5 m/s and became 0.35 + 0.005 x alt.
    # Flown, kept. The phantom speed it suppresses grows with height, so the
    # threshold has to as well. KNOWN COST, not yet solved: below 25 m this
    # discards 49% of genuine motion, which is why the aircraft wanders under
    # 10 m while its POSITION there is accurate to 2 cm.
    hover_deadband_base_mps: float = 0.35
    # 0.005 -> 0.010, 22 Oct. See VelocityBridge's deadband_per_m: at 83 m
    # the old threshold was 0.77 m/s against a sustained phantom whose p90 was
    # 0.91, so the worst ones passed and LOITER flew them out.
    hover_deadband_per_m: float = 0.010

    # --- camera (OAK-D Pro mono pair) ---
    # Measured on this device 19 Oct: THE_800_P gives 1280x800 and 29.8-30.7 fps
    # on both sockets over a SUPER (USB3) link, mono focal fx 791.7 / 795.1.
    # 791.72 is what bench_workload.py and the matcher already use - do NOT
    # confuse it with the 1943 px in map_db.py's docstring, which is the RGB
    # sensor and is only an example there.
    mono_fps: int = 30

    # Camera IMU report rate, Hz, for all three enabled sensors - they share
    # one packet, so one number sets them together.
    #
    # 100 WAS CALLED A CEILING AND IS NOT ONE. Measured on this device
    # 21 Oct, IMU alone: asking 400 delivered 357.9 Hz on BOTH
    # GYROSCOPE_CALIBRATED and LINEAR_ACCELERATION, against Luxonis' table
    # giving the calibrated gyro a maximum of 100.
    #
    # Why it matters, from log 225's own camera recording in flight: the
    # linear acceleration put 94% of its energy above 10 Hz with every top
    # peak at 49.8 Hz - hard against Nyquist for a 100 Hz sample. At rest the
    # same channel spreads its energy (59.8% below 10 Hz on X), and its sd is
    # 0.030 m/s2 against 1.30 in flight, so what piles at Nyquist is real
    # vibration above 50 Hz folding down, not sensor noise. A hex at hover
    # puts its prop fundamental near 65-85 Hz, which is exactly there. The
    # velocity bridge integrates that aliased signal.
    #
    # 3 Oct recorded the opposite result - "asking for 400 on the gyro dragged
    # the whole stream down rather than speeding anything up" - so the
    # comparison was run again 21 Oct on the FULL pipeline, both mono streams
    # and stereo depth up, 800p at 30 fps, all three IMU sensors:
    #
    #     requested   frames    IMU packets   accel/gyro/quat
    #       100 Hz    30.0 Hz       22006         99.9 Hz
    #       400 Hz    30.0 Hz       36601        165.9 Hz
    #
    # The frame rate does not move. 400 does not deliver 400 with the camera
    # streams competing for the link - it delivers 165.9 - but that is 1.66x
    # the reports and, more to the point, it moves Nyquist from 50 Hz to 83.
    # The same comparison at rest shows the fold directly: at 100 Hz the accel
    # peaks sat at 36.0-48.7 Hz, and at 400 Hz the same content appears at
    # 59.0-66.8 Hz. It was never at 36 Hz; it was folding there.
    #
    # 83 Hz still only just covers a hex's hover fundamental, so this is an
    # improvement rather than a solution. What it buys is the first honest
    # look at where the vibration actually is.
    imu_rate: int = 400

    # How many frames the flow VELOCITY is measured over. Position is
    # unaffected - it integrates every frame as before.
    #
    # It was one frame until 22 Oct, which is the worst possible choice,
    # because velocity = pixels x altitude / focal: at 80 m one pixel is
    # 80/791.72 = 0.101 m, and at 30 fps that is 3.03 m/s PER PIXEL of
    # frame-to-frame jitter. Measured on log 226, per-axis error against the
    # raw receiver, by baseline length:
    #
    #     N frames   lag     5-25m   25-45m   45-70m   70-85m
    #        1      0.02 s    1.79     8.45     9.67    17.79 m/s
    #       20      0.33 s    0.25     1.76     1.31     3.75
    #       30      0.50 s    0.15     1.31     0.83     2.61
    #
    # Summing N displacements and dividing by their span divides coherent
    # motion by N but white jitter only by sqrt(N), so the noise falls as
    # 1/sqrt(N) while real motion is preserved. The cost is lag: the estimate
    # is the average over the span, so it is centred N/2 frames in the past.
    #
    # 20 and not 30. The lag is NOT compensated - GPS2_DELAY_MS stays 0 -
    # because one GPS delay applies to the position too, and the position is
    # the part that works (SP never left 0.56 against a 1.2 gate on log 226).
    # Delaying it to help the velocity would trade the good signal for the
    # bad one. The lag penalty is therefore already inside the numbers above:
    # they are measured against the receiver's speed AT THE SAME INSTANT, not
    # against a shifted copy.
    #
    # The displacements are the SAME ones the fuser consumed, so position and
    # velocity still cannot disagree about what the camera saw - which is what
    # the EKF reads as a velocity innovation.
    # 20 -> 40 ON 22 OCT, after log 227 would not hold in any VPS window.
    #
    # The pilot's own observation settled where to look: on log 225 the
    # aircraft anchored itself as long as no stick was given and ran away the
    # moment one was. Measured below 60 m across logs 225/226/227, where the
    # map never runs so this is flow alone, the velocity error against the raw
    # receiver by the aircraft's TRUE speed:
    #
    #     N    lag    hover 0-0.5   0.5-1    1-2     2-3    3-5 m/s
    #     20  0.33 s     0.50        0.90    1.22    2.44   5.70
    #     30  0.50 s     0.29        0.58    0.68    2.23   4.65
    #     40  0.67 s     0.24        0.51    0.60    1.97   3.98
    #     60  1.00 s     0.17        0.36    0.49    1.54   2.56
    #     90  1.50 s     0.16        0.43    0.61    1.42   2.06
    #
    # At N=20 the hover error (0.50) sits under the 31 m deadband of 0.66 and
    # is suppressed - which IS the anchoring the pilot saw. The moment the
    # aircraft moves at 0.5 m/s the error is 0.90, above the deadband, so it
    # reaches the flight controller; and a sustained false velocity is the one
    # thing LOITER cannot settle against, because with no stick it can only
    # cancel a velocity error by flying. Hence the runaway on stick.
    #
    # N=40 puts 0.5-2 m/s at 0.51-0.60, BELOW that same deadband, so the band
    # LOIT_SPEED_MS=3.0 actually commands is suppressed too. 60 is better
    # still and costs 1.0 s of lag, which a position controller acting on
    # stale velocity can turn into the oscillation this is meant to remove -
    # not taken without a flight behind it.
    #
    # NOT a scale error and not derotation, both tested and both ruled out:
    # correcting the measured scale (forward 0.903, sideways 0.873 on each
    # axis's own dominant samples) made the velocity error WORSE by 9-13% and
    # did nothing for position drift in either direction; and holding speed at
    # 0.5-2 m/s, per-frame tilt of 0-0.10 deg and 0.2+ deg give the same error
    # (1.42 vs 1.44), with corr(error, tilt) +0.109 against corr(error, speed)
    # +0.662. What is left is the flow's own pixel noise: 1.4 m/s at 31 m is
    # 1.2 px per frame, and a longer baseline is the only thing that touches it.
    flow_vel_frames: int = 40

    # Stereo depth: ON, and for VIO ONLY.
    #
    # It is cheap where it was feared to be expensive. Measured 19 Oct on this
    # camera: adding StereoDepth left left/right/depth all at 30.7 fps and the
    # IMU at 104 Hz, for +2 C of chip temperature - because the depth is
    # computed on the OAK's own Myriad X VPU, not on the Jetson. The 9 Oct
    # figures (fps 25 -> 12.5, camera +12 C, main loop stalled 44 s and 30 s)
    # were host-side contention with the flow loop, map worker and FC bridge
    # all running, not the camera's own cost. One real warning did show up
    # even alone: the worst frame gap went 35.2 -> 47.2 ms, already past one
    # frame period at 30 fps, which is how those stalls begin.
    #
    # What it is NOT: a height source. That is the part 9 Oct reverted, and the
    # report's own verdict was that it came out "worse than the baro it was
    # meant to help". Height stays TF03 below lidar_max_alt_m, barometer above.
    # --- the aircraft's own leg, in the left camera's frame ---
    # Measured 20 Oct off CAM_B at 1280x800. The left landing leg occludes the
    # left edge: columns 0-63 fall below 60% of the frame median (70-90 against
    # 147.5) and the darkening tapers out by x~120. The RIGHT camera is flat
    # across every column, 153-157, which is what proves it is an object in
    # front of CAM_B and not lens vignetting - both sensors are the same part.
    #
    # It matters because it is RIGIDLY ATTACHED. 32 of 300 detected corners sat
    # inside it, and a corner on the aircraft reports zero motion in every
    # frame while the ground moves. RANSAC does not save you from that: those
    # tracks agree perfectly with each other on a single rigid motion of zero,
    # so a tenth of the evidence is pulling the estimate towards standing still.
    #
    # Cropped, not masked: a masked block is a uniform patch that the NCC
    # matcher would happily correlate against. Cropping the column off takes
    # the leg out of flow, the map matcher and the recorded frames at once, and
    # cx moves with it.
    crop_left_px: int = 130       # 120 where the taper ends, plus margin

    stereo_depth: bool = True
    stereo_depth_as_height: bool = False   # keep False. See above.

    # --- camera IMU (OAK-D Pro) ---
    # 5 Oct config, with gyro at 100 Hz per the pilot's instruction rather than
    # the 200 that report lists. 3 Oct measured why over-asking is not neutral:
    # all three reports share one packet, so gyro at 400 dragged the whole
    # stream to 57.8 Hz, and a calibrated gyro requested at 200 against a
    # documented 100 Hz maximum delivered exactly 100 anyway.
    imu_gyro_hz: int = 100
    imu_rotation_vector_hz: int = 100     # GAME_ROTATION_VECTOR
    imu_linear_accel_hz: int = 100        # LINEAR_ACCELERATION
    imu_batch_reports: int = 1     # 3 Oct: batching 5 left 34.2% of frames with
                                   # NO IMU sample at all; at 1 it is 1.1%, same
                                   # camera rate, same gaps.
    imu_magnetometer: bool = False  # 5 Oct: it measured clean, but
                                   # ROTATION_VECTOR shares the rotation pipe and
                                   # taking the magnetometer cost derotation and
                                   # two flights. Stays off.

    # --- height source ---
    # 9 Oct: the TF03 went from not fitted to PRIMARY below ~56 m. Flown.
    # 94% usable below 58 m (stuck 0.7%, out of range 5.3%, spikes 0.2%).
    #
    # READ FROM THE FLIGHT CONTROLLER, log 00000195.BIN (15 Oct, the last
    # flight on the card and later than every status report):
    #   RNGFND1_MAX     60      <- so 60, not the report's approximate 56
    #   RNGFND1_TYPE    1       = ANALOG
    #   RNGFND1_PIN     4
    #   RNGFND1_SCALING 72.61   RNGFND1_OFFSET 0.606977  RNGFND1_RMETRIC 1
    #   RNGFND1_GNDCLR  0.13    RNGFND1_ORIENT 25 (down)
    #
    # GNDCLR is now 0.26, read back off the FC on 19 Oct after the move to the
    # ZD850 hexa - its 350 mm landing gear doubles the lidar's height above the
    # ground. The matching code-side constants are in cuav_mavlink_bridge
    # (lidar 0.26, camera 0.17), and on this airframe the camera sits BELOW the
    # lidar rather than above it.
    #
    # Note what that says: it is STILL on an analog pin. 5 Oct blamed the
    # lidar's 253 cm reading against a true 86 cm on "a serial TF03 wired to an
    # analog pin", so it is tempting to assume it was rewired to serial. It was
    # not. What fixed it was calibrating the analog path - that SCALING and
    # OFFSET - not changing the wiring. Moving it to a serial port is still
    # available and still unmeasured.
    lidar_max_alt_m: float = 60.0  # above this, height comes from the barometer

    # --- heading ---
    # NOT the camera. 4 Oct tested GAME_ROTATION_VECTOR's yaw for exactly this
    # and it cannot do it. Heading is still the FC compass, which is the last
    # hard dependency: the whole position solution rotates with it.
    heading_from_camera: bool = False

    def R(self) -> np.ndarray:
        return np.asarray(self.R_vio_to_ned, float)
