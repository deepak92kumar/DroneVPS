# vps_vio — VPS/VIO upgrade (code complete, NOT implemented until after demo)

Goal: give this drone a real **Visual-Inertial Odometry** layer so it can hold
position **without an ArUco marker** at altitude — the way Stride Dynamics' VPS
does. Full rationale, phases and timeline: `../markers/VPS_Upgrade_Plan.pdf`.

## Status

The whole pipeline is **written and unit-tested**, but nothing is wired into the
live system and nothing runs in flight yet. Implementation (bench bring-up, then
flights) starts **after the demo**.

Isolation guarantees:
- **No existing file is modified** — not `flight_daemon.py`, not
  `cuav_mavlink_bridge.py` (only imported, unchanged), no FC parameter.
- The real-hardware entry points **refuse to run while `drone-vio` is active**
  (they would fight over the OAK-D and the FC serial port).
- Demo config stays frozen at `d75a6be` on `fix/vio-loiter-runaway`.

## Architecture

```
VIO source ──▶ vio_frame (VIO world → NED + compass heading align)
                    │
                    ▼
              AnchorFuser  ◀── ArUco fix (absolute, corrects VIO drift)
                    │
                    ▼
     cuav_mavlink_bridge.send_vision_estimate(x, y, 0, …)   [X/Y only]
                    │
                    ▼
        ArduPilot EKF3   (EK3_SRC1_POSXY=6, Z stays on BARO/lidar)
```
Z is never sent from VIO — the FC keeps barometer/TF03 for height, exactly like
Stride's VPS. ArUco is the absolute anchor that cancels VIO drift when a marker
is visible; the two correction constants stay **flat** (never height-scaled —
that cost two flights, see memory `d75a29f`).

## Files

| File | What | Runs now? |
|------|------|-----------|
| `config.py` | All tunables (FC port, axis map, fusion rates, covariances). | pure |
| `vio_frame.py` | VIO world pose → NED, compass heading align. | pure |
| `anchor.py` | `AnchorFuser` — VIO + ArUco drift correction (slew / re-anchor, flat). | pure |
| `vio_source.py` | `SpectacularAiSource` (OAK-D) + `SyntheticSource` (for tests). | mixed |
| `vps_node.py` | Full standalone node tying it together; `--dry-run` needs no hardware. | dry-run: yes |
| `phase0_vio_eval.py` | Bench VIO eval: rate + camera-IMU latency check. | after demo |
| `tests/` | 21 unit tests (frame 8, anchor 8, node 5), all passing. | yes |

## Run

```
# all tests (no hardware)
for t in vps_vio/tests/test_*.py; do python3 "$t"; done

# prove the full wiring with synthetic VIO — no camera, no FC, nothing sent
python3 vps_vio/vps_node.py --dry-run --seconds 6
```

## After the demo — bring-up order

1. `sudo systemctl stop drone-vio`  (frees the OAK-D).
2. `python3 vps_vio/phase0_vio_eval.py` — confirm VIO rate ≥ ~20 Hz and stable
   camera-IMU latency. **Exit criterion for Phase 0.**
3. From the eval CSV, measure the VIO→NED axis mapping and replace the
   **placeholder identity `R`** in `config.py` / `vio_frame.py`.
4. Confirm SpectacularAI attribute names in `vio_source.py` against the installed
   version (flagged in code).
5. Bench `vps_node.py` against the FC (props off), watch VISION_POSITION_ESTIMATE
   accepted; then low-alt flights, then climb.
6. Wire the **live ArUco fix** into `node.step(..., marker_world_ned=…)` — the
   only TODO left; it needs the daemon's detector and is done at bench, post-demo.

## Known must-verify constants (flagged in code, do not skip)

- `config.R_vio_to_ned` — bench-measured axis map (identity is a placeholder).
- Heading alignment source — read the FC compass yaw at origin (dry-run uses 0).
- SpectacularAI pose field names — confirm against installed SDK.
