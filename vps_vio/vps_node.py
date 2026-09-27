#!/usr/bin/env python3
"""VPS node — the full VIO -> FC pipeline, as a STANDALONE program.

    VIO source -> vio_frame (NED + heading align) -> AnchorFuser (ArUco drift
    correction) -> cuav_mavlink_bridge.send_vision_estimate (X/Y only)

This is the whole Phase-1 application, written now but NOT run until after the
demo. It is deliberately a separate process from flight_daemon.py:

    * It does NOT import or modify flight_daemon.py.
    * It reuses cuav_mavlink_bridge.CuavMavlinkClient unchanged.
    * It refuses to start while the drone-vio daemon is active (they would fight
      over the OAK-D and the FC serial port).
    * Z is never sent from VIO — the FC keeps barometer/lidar for height, exactly
      like Stride's VPS.

Two run modes:
    --dry-run   : SyntheticSource, no camera, no FC. Prints the fused NED it
                  WOULD send. Safe to run anywhere, anytime — proves the wiring.
    (default)   : real OAK-D VIO + real FC. Bench/flight only, after the demo.

Marker fixes: for the dry-run we synthesise none; in the real integration the
ArUco fix will come from the same detector the daemon uses. The hook is
`fuser.update_marker(...)`; wiring the live marker feed is the last Phase-1 step
and is marked TODO below (needs the daemon's detector, done after demo).
"""
from __future__ import annotations
import argparse
import logging
import subprocess
import sys
import time

import numpy as np

from config import VpsConfig
from vio_frame import VioToNed, quat_to_yaw
from anchor import AnchorFuser
from vio_source import SyntheticSource, CuVslamSource

log = logging.getLogger("vps_node")


def daemon_active() -> bool:
    try:
        r = subprocess.run(["systemctl", "is-active", "drone-vio"],
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip() == "active"
    except Exception:
        return False


class VpsNode:
    def __init__(self, cfg: VpsConfig, source, bridge=None):
        self.cfg = cfg
        self.source = source
        self.bridge = bridge                      # None in dry-run
        self.frame = VioToNed(R_vio_to_ned=cfg.R())
        self.fuser = AnchorFuser(cfg.corr_rate_mps, cfg.reanchor_m)
        self._origin_set = False
        self._last_t = None
        self._rate = 0.0

    def _origin(self, pose):
        # latch takeoff as world origin and align VIO heading to compass.
        self.frame.set_origin(pose.pos)
        # heading align comes from the compass at init; in the real integration
        # read the FC's yaw here. Dry-run uses VIO yaw as-is (align = 0).
        self._origin_set = True
        log.info("origin latched at VIO pos %s", np.round(pose.pos, 3))

    def step(self, pose, marker_world_ned=None):
        """Process one VIO sample (+ optional marker fix). Returns fused NED or None."""
        if not pose.tracking:
            return None
        if not self._origin_set:
            self._origin(pose)
        # rate tracking
        if self._last_t is not None:
            dt = max(pose.t - self._last_t, 1e-6)
            self._rate = 0.9 * self._rate + 0.1 * (1.0 / dt)
        self._last_t = pose.t

        vio_ned = self.frame.to_ned(pose.pos)
        if marker_world_ned is not None:
            dt_m = 1.0 / max(self.cfg.send_rate_hz, 1.0)
            self.fuser.update_marker(vio_ned, marker_world_ned, dt_m)
        world = self.fuser.fused(vio_ned)
        return world

    def _send(self, world_ned, pose):
        if self.bridge is None:
            return
        good = self._rate >= self.cfg.min_vio_rate_hz
        cov = self.cfg.pos_cov_good if good else self.cfg.pos_cov_poor
        yaw = self.frame.yaw_to_ned(quat_to_yaw(*pose.quat))
        # X/Y from VIO; Z is left to the FC baro — send 0.0 and let EK3_SRC1_POSZ=BARO.
        self.bridge.send_vision_estimate(
            float(world_ned[0]), float(world_ned[1]), 0.0,
            0.0, 0.0, float(yaw), pos_cov=cov)

    def run(self, seconds: float, dry_run: bool):
        t0 = time.monotonic()
        interval = 1.0 / max(self.cfg.send_rate_hz, 1.0)
        next_send = 0.0
        n = 0
        while time.monotonic() - t0 < seconds:
            pose = self.source.read()
            if pose is None:                 # source timed out (no frames)
                continue
            world = self.step(pose)          # TODO: pass the live ArUco fix here
            if world is None:
                continue
            now = time.monotonic() - t0
            if now >= next_send:
                next_send = now + interval
                n += 1
                if dry_run:
                    if n % 20 == 0:
                        log.info("world NED x=%+.2f y=%+.2f  rate=%.1fHz  reanchors=%d",
                                 world[0], world[1], self._rate, self.fuser.reanchor_count)
                else:
                    self._send(world, pose)
                    if n % 30 == 0:
                        log.info("body x=%+.2f y=%+.2f  rate=%.1fHz  reanchors=%d  drops=%d",
                                 world[0], world[1], self._rate,
                                 self.fuser.reanchor_count, getattr(self.source, "dropped", 0))
            if dry_run:
                time.sleep(max(0.0, interval * 0.1))  # don't spin the CPU in dry-run
        log.info("done: %d sends, final offset=%s", n, np.round(self.fuser.offset, 3))


def main():
    ap = argparse.ArgumentParser(description="VPS node (VIO->FC). Standalone, post-demo.")
    ap.add_argument("--dry-run", action="store_true",
                    help="synthetic VIO, no camera, no FC — safe anywhere")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--force", action="store_true", help="run even if daemon looks active")
    ap.add_argument("--inertial", action="store_true", help="stereo+IMU (default: stereo only)")
    ap.add_argument("--send-fc", action="store_true",
                    help="actually transmit VISION_POSITION_ESTIMATE to the flight controller")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    cfg = VpsConfig()

    if a.dry_run:
        node = VpsNode(cfg, SyntheticSource(drift_mps=(0.02, -0.01, 0.0)), bridge=None)
        log.info("DRY RUN — synthetic VIO, no hardware. Nothing is sent to any FC.")
        node.run(a.seconds, dry_run=True)
        return

    # --- real hardware path (post-demo, daemon stopped) ---
    if daemon_active() and not a.force:
        sys.exit("REFUSING: drone-vio daemon is active (owns OAK-D + FC serial). "
                 "Stop it first: sudo systemctl stop drone-vio")
    from cuav_mavlink_bridge import CuavMavlinkClient  # imported only on the real path
    bridge = CuavMavlinkClient(preferred_port=cfg.fc_port, baud=cfg.fc_baud) if a.send_fc else None
    if bridge is None:
        log.info("FC sending DISABLED (no --send-fc): pose is computed but nothing is transmitted")
    src = CuVslamSource(inertial=a.inertial); src.open()
    node = VpsNode(cfg, src, bridge=bridge)
    try:
        node.run(a.seconds, dry_run=False)
    finally:
        src.close()


if __name__ == "__main__":
    # allow running from repo root or from vps_vio/
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (here, os.path.dirname(here)):
        if p not in sys.path:
            sys.path.insert(0, p)
    main()
