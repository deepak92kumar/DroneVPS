# GNSS-Denied Quadcopter — Vision-Based Navigation

**CUAV X7 Pro · NVIDIA Jetson Orin Nano · Luxonis OAK-D Pro · ArduCopter 4.7**

A ~3.75 kg quad-X UAV that holds position and lands precisely **without GPS or
RTK**, using a downward camera locked to a ground ArUco marker plus optical flow,
fused into ArduPilot's EKF3 as an external navigation source. Vision, lidar and
barometer only — no satellite fix.

> Scope note: this README describes **this aircraft** — the OAK-D quad in this
> repo. A separate tethered hexacopter and a Stride-Dynamics VPS unit referenced
> elsewhere are different platforms and are not covered here.

---

## 1. Platform

| Item | Spec |
|------|------|
| Airframe | Quad-X, carbon frame |
| All-up weight | ~3.75 kg |
| Motors | 4× T-Motor MN5008 KV340 |
| Propellers | 15×5 carbon |
| Autopilot | CUAV X7 Pro, ArduCopter 4.7 |
| Companion computer | NVIDIA Jetson Orin Nano |
| Camera | Luxonis OAK-D Pro |
| Rangefinder | Benewake TF03 lidar (up to ~180 m) |
| Compass | External M9N (magnetometer only) |
| Positioning | Vision (ArUco + optical flow), **no GNSS/RTK** |

### Camera use

The OAK-D Pro serves two roles from one device:

- **Mono global-shutter (OV9282, 640×400, high frame rate)** — optical flow and
  close-range marker tracking. Global shutter avoids the rolling-shutter jello a
  moving multirotor would otherwise inject.
- **RGB 12 MP (IMX378), cropped** — reads the ArUco marker from higher altitude,
  where the small mono frame can no longer resolve it.

---

## 2. How it navigates without GPS

```
OAK-D Pro ──▶ Jetson Orin Nano ──▶ VISION_POSITION_ESTIMATE / VISION_SPEED ──▶ CUAV X7 Pro (EKF3)
  marker + flow      detection + pose               MAVLink 2                    EK3_SRC1_POSXY = 6
```

- **ArUco marker = absolute position.** A nested 3.6 m ground marker
  (ID 45 / ID 16 / ID 0, coarse-to-fine) gives the aircraft an absolute X/Y fix.
  The nesting lets it stay locked from low hover up to high altitude as the outer
  then inner tags fall in and out of the frame.
- **Optical flow = velocity aid** at low altitude (it is unreliable above ~40 m,
  so it is gated off above that height).
- **EKF3 external nav.** Position comes in as `VISION_POSITION_ESTIMATE` with a
  quality-dependent covariance; `EK3_SRC1_POSXY = 6` (ExternalNav). Height (Z)
  stays on the **barometer and TF03 lidar** — vision provides X/Y only.
- **Heading.** An external M9N magnetometer with `EK3_MAG_CAL = 2` (simple
  heading fusion, no in-flight bias learning) — this fixed a yaw/EKF lane
  disagreement and raised the stable ceiling.

The companion software (`flight_daemon.py`) runs the detector, aligns its clock
to the FC log via `VISP.RTimeUS`, integrates the sent position, and manages the
marker-map anchoring and failover.

---

## 3. What has been demonstrated

- Marker-locked **LOITER position hold past 100 m** AGL (best flight ~112 m by
  barometer), GPS disabled.
- Re-engageable position modes after manual ALT_HOLD, and a marker-distance
  warning with direction-to-home.
- Precision landing onto the nested marker.

Altitudes quoted are **flight-demonstrated results from onboard logs**, not rated
specifications.

---

## 4. Key ArduPilot parameters

| Parameter | Value | Why |
|-----------|-------|-----|
| `EK3_SRC1_POSXY` | 6 (ExternalNav) | use the vision position |
| `EK3_SRC1_POSZ` | Baro | height from barometer, not vision |
| `EK3_MAG_CAL` | 2 | simple heading fusion; fixed the yaw drift |
| `EK3_POS_I_GATE` | 500 | opened back to default; stopped EKF failsafes |
| `PSC_NE_POS_P` | 1.0 | position controller (tuned per flight) |
| `PSC_ANGLE_MAX` | 15 | lean-angle limit for gentle vision holds |

---

## 5. In development — `vps_vio/`

A marker-free **Visual-Inertial Odometry** layer, so the aircraft can hold
position at altitude even when no marker is visible — the way a dedicated VPS
does. It reuses the same FC path (`VISION_POSITION_ESTIMATE`, X/Y only, baro
keeps Z) and keeps the ArUco marker as an absolute drift anchor. The pipeline is
written and unit-tested but **not yet wired into flight**; see
[`vps_vio/README.md`](vps_vio/README.md) for the architecture and the
post-integration plan. Nothing in `vps_vio/` runs during normal operation.

---

## Repository layout (selected)

| Path | What |
|------|------|
| `flight_daemon.py` | Main companion daemon — detection, pose, marker map, failover |
| `cuav_mavlink_bridge.py` | MAVLink 2 link to the FC (vision estimate / speed / params) |
| `vps_vio/` | Marker-free VIO upgrade (in development, isolated) |
| `markers/` | ArUco marker artwork and print files |
| `tests/` | Daemon test suite |

## Safety

Position modes are only engaged with a healthy vision solution; loss of tracking
falls back to ALT_HOLD/STABILIZE rather than fusing bad data. FC parameter
changes are made on the ground, disarmed, and verified by flight before being
trusted.
