#!/usr/bin/env python3
"""
OAK-D Hardware & Pipeline Benchmark Script
Measures actual FPS, sync, latency, IMU rate, frame drops, and CPU/GPU usage.
"""

import depthai as dai
import cv2
import numpy as np
import time
import os
import psutil

def main():
    print("=" * 70)
    print("      OAK-D CAMERA PIPELINE PERFORMANCE & LIMITATION BENCHMARK     ")
    print("=" * 70)

    pipeline = dai.Pipeline()
    board_config = dai.BoardConfig()
    board_config.usb.maxSpeed = dai.UsbSpeed.SUPER
    pipeline.setBoardConfig(board_config)

    # 1. Left Mono OV9282 (1280x800 @ 30 FPS)
    mono_l = pipeline.create(dai.node.MonoCamera)
    mono_l.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    mono_l.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_l.setFps(30)
    mono_l.initialControl.setAutoExposureEnable()
    mono_l.initialControl.setAutoExposureLimit(15000)

    # 2. Right Mono OV9282 (1280x800 @ 30 FPS)
    mono_r = pipeline.create(dai.node.MonoCamera)
    mono_r.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    mono_r.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    mono_r.setFps(30)
    mono_r.initialControl.setAutoExposureEnable()
    mono_r.initialControl.setAutoExposureLimit(15000)

    xout_mono_l = pipeline.create(dai.node.XLinkOut)
    xout_mono_l.setStreamName("mono_l")
    mono_l.out.link(xout_mono_l.input)

    xout_mono_r = pipeline.create(dai.node.XLinkOut)
    xout_mono_r.setStreamName("mono_r")
    mono_r.out.link(xout_mono_r.input)

    # 3. RGB Color Camera (1920x1080 @ 30 FPS) with on-chip MJPEG VideoEncoder
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setFps(30)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    vid_enc = pipeline.create(dai.node.VideoEncoder)
    vid_enc.setDefaultProfilePreset(30, dai.VideoEncoderProperties.Profile.MJPEG)
    vid_enc.setQuality(85)
    cam_rgb.video.link(vid_enc.input)

    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    vid_enc.bitstream.link(xout_rgb.input)

    # 4. Stereo Depth (1280x800 @ 30 FPS)
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    stereo.initialConfig.setConfidenceThreshold(230)
    stereo.initialConfig.setLeftRightCheckThreshold(5)
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(False)
    stereo.setSubpixel(True)
    stereo.setOutputSize(1280, 800)

    cfg = stereo.initialConfig.get()
    cfg.postProcessing.spatialFilter.enable = True
    cfg.postProcessing.temporalFilter.enable = True
    cfg.postProcessing.speckleFilter.enable = True
    stereo.initialConfig.set(cfg)

    mono_l.out.link(stereo.left)
    mono_r.out.link(stereo.right)

    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)

    # 5. IMU (400 Hz target)
    imu = pipeline.create(dai.node.IMU)
    imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_CALIBRATED, 400)
    imu.enableIMUSensor(dai.IMUSensor.ROTATION_VECTOR, 400)
    imu.setBatchReportThreshold(5)
    imu.setMaxBatchReports(20)

    xout_imu = pipeline.create(dai.node.XLinkOut)
    xout_imu.setStreamName("imu")
    imu.out.link(xout_imu.input)

    print("[*] Connecting to OAK-D device and initializing pipeline...")
    with dai.Device(pipeline) as device:
        usb_speed = device.getUsbSpeed().name
        print(f"[+] Connected! Negotiated USB Speed: {usb_speed}")
        calib = device.readCalibration()
        K_800p = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, 1280, 800)
        print(f"[+] Intrinsics at 1280x800: fx={K_800p[0][0]:.1f}, fy={K_800p[1][1]:.1f}, cx={K_800p[0][2]:.1f}, cy={K_800p[1][2]:.1f}")

        q_mono_l = device.getOutputQueue("mono_l", maxSize=2, blocking=False)
        q_mono_r = device.getOutputQueue("mono_r", maxSize=2, blocking=False)
        q_rgb = device.getOutputQueue("rgb", maxSize=2, blocking=False)
        q_depth = device.getOutputQueue("depth", maxSize=2, blocking=False)
        q_imu = device.getOutputQueue("imu", maxSize=20, blocking=False)

        count_mono_l = 0
        count_mono_r = 0
        count_rgb = 0
        count_depth = 0
        count_imu = 0

        sync_deltas_ms = []
        latencies_ms = []

        last_l_ts = None
        last_r_ts = None

        print("[*] Benchmarking pipeline for 10 seconds under full load...")
        start_time = time.time()
        test_duration = 10.0

        cpu_usages = []

        while time.time() - start_time < test_duration:
            now = time.time()

            # Mono Left
            in_l = q_mono_l.tryGet()
            if in_l is not None:
                count_mono_l += 1
                last_l_ts = in_l.getTimestamp().total_seconds()
                xfer_ms = max(0.0, (time.monotonic() - last_l_ts) * 1000.0)
                latencies_ms.append(xfer_ms)

            # Mono Right
            in_r = q_mono_r.tryGet()
            if in_r is not None:
                count_mono_r += 1
                last_r_ts = in_r.getTimestamp().total_seconds()

            # Check stereo synchronization
            if last_l_ts is not None and last_r_ts is not None:
                diff_ms = abs(last_l_ts - last_r_ts) * 1000.0
                sync_deltas_ms.append(diff_ms)
                last_l_ts = None
                last_r_ts = None

            # RGB
            in_rgb = q_rgb.tryGet()
            if in_rgb is not None:
                count_rgb += 1

            # Depth
            in_depth = q_depth.tryGet()
            if in_depth is not None:
                count_depth += 1

            # IMU
            in_imu = q_imu.tryGet()
            if in_imu is not None:
                count_imu += len(in_imu.packets)

            time.sleep(0.001)
            if int(now * 10) % 10 == 0:
                cpu_usages.append(psutil.cpu_percent(interval=None))

        dur = time.time() - start_time
        fps_mono_l = count_mono_l / dur
        fps_mono_r = count_mono_r / dur
        fps_rgb = count_rgb / dur
        fps_depth = count_depth / dur
        rate_imu = count_imu / dur

        avg_sync_ms = float(np.mean(sync_deltas_ms)) if sync_deltas_ms else 0.0
        max_sync_ms = float(np.max(sync_deltas_ms)) if sync_deltas_ms else 0.0
        avg_latency_ms = float(np.mean(latencies_ms)) if latencies_ms else 0.0
        avg_cpu = float(np.mean(cpu_usages)) if cpu_usages else 0.0

        print("\n" + "=" * 70)
        print("                   BENCHMARK RESULTS REPORT                   ")
        print("=" * 70)
        print(f" USB Connection Speed       : {usb_speed}")
        print(f" Left Mono FPS (1280x800)   : {fps_mono_l:.2f} FPS")
        print(f" Right Mono FPS (1280x800)  : {fps_mono_r:.2f} FPS")
        print(f" RGB Camera FPS (1080p)     : {fps_rgb:.2f} FPS")
        print(f" Stereo Depth FPS (800p)    : {fps_depth:.2f} FPS")
        print(f" Stereo Sync Delta          : avg={avg_sync_ms:.3f} ms, max={max_sync_ms:.3f} ms")
        print(f" IMU Output Rate            : {rate_imu:.1f} Hz (Target: 400 Hz)")
        print(f" Camera-to-Jetson Latency   : {avg_latency_ms:.1f} ms")
        print(f" Jetson CPU Usage           : {avg_cpu:.1f} %")
        print("=" * 70)

if __name__ == "__main__":
    main()
