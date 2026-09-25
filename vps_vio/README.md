# vps_vio — VPS/VIO upgrade (work in progress, NOT wired into flight)

Goal: give this drone a real **Visual-Inertial Odometry** layer so it can hold
position **without an ArUco marker** at altitude — the way Stride Dynamics' VPS
does. Full rationale, phases and timeline are in
`../markers/VPS_Upgrade_Plan.pdf`.

## Status: build started, implementation AFTER the demo

Nothing in this folder runs in flight or touches the live system. It is isolated
on purpose:

- **No existing file is modified** — not `flight_daemon.py`, not
  `cuav_mavlink_bridge.py`, not any FC parameter.
- `phase0_vio_eval.py` **refuses to run while the `drone-vio` daemon is active**
  (the daemon owns the OAK-D). Bench only.
- The demo configuration stays frozen at `d75a6be`.

## Files

| File | What | Runs now? |
|------|------|-----------|
| `vio_frame.py` | Pure math: VIO world pose → ArduPilot NED, + compass heading align. The one piece buildable with no hardware. | yes (pure) |
| `tests/test_vio_frame.py` | Unit tests for the transform (8, all pass). | yes |
| `phase0_vio_eval.py` | Bench harness: run VIO on OAK-D, log pose, measure rate + cam-IMU latency. | after demo, daemon stopped |

Run the tests:
```
python3 vps_vio/tests/test_vio_frame.py
```

## How it will plug in (Phase 1, later)

The VIO pose feeds the FC through the **existing** bridge method — no new MAVLink
behaviour:

```
cuav_mavlink_bridge.CuavMavlinkClient.send_vision_estimate(x, y, z, roll, pitch, yaw, pos_cov)
   # x,y,z are NED metres from vio_frame.VioToNed.to_ned(...)
   # ArduPilot already set to EK3_SRC1_POSXY = 6 (ExternalNav) — same as today
```

Height (Z) stays with the FC's barometer / TF03 lidar — VIO provides **X/Y only**,
exactly like Stride's VPS. ArUco stays as the absolute anchor that corrects VIO
drift when a marker is visible.

## Phase 0 exit criteria (before writing any Phase 1 code)

1. VIO output rate steady, ≥ ~20 Hz.
2. Camera–IMU timing sane (low, stable `|wall - vio_time|`).
3. Axis mapping measured: move the aircraft in known directions, read the CSV,
   and replace the **placeholder identity `R`** in `vio_frame.py` with the real
   VIO→NED permutation. (This is flagged in the code — do not skip it.)

If SpectacularAI's cam-IMU sync on the OAK-D is not good enough, the backup is
OpenVINS / VINS-Fusion (same `vio_frame` transform, different engine).
