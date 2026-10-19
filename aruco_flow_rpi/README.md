# ArUco + Optical Flow — Raspberry Pi port

Copied from the Jetson project at `/data/DRONE` on 2026-10-19. This is the
pre-26-Sep marker + optical-flow navigation work, the part that sits **outside**
`vps_vio/`. Target for this copy: a Raspberry Pi with a camera and a hardware
optical flow sensor, to stabilise a second drone on an ArUco marker alone.

**Nothing was moved.** Every file here is a copy; the Jetson project is unchanged
and still runs.

## What is here

| File | Lines | Notes |
|---|---|---|
| `flight_daemon.py` | 3218 | The marker + flow daemon. 123 ArUco references. Uses OAK-D via `depthai`. |
| `cuav_mavlink_bridge.py` | 1142 | FC link over MAVLink. Shared with `vps_vio` in the original — copied so this folder stands alone. |
| `live_preview.py` | 680 | Live marker view. Uses `depthai`. |
| `c12_precision_landing.py` | 418 | Precision landing on the pad. No `depthai`. |
| `generate_a3_aruco_pads.py` | 160 | Pad generator. No Jetson or OAK-D dependency. |
| `flow_shadow.py` | 145 | Optical flow: `LongBaselineTracks`, `JetsonLKWorker`, `px_to_ned_velocity`. |
| `generate_landing_pad.py` | 144 | Pad generator. No Jetson or OAK-D dependency. |
| `markers/` | 32 files | Printable ArUco pads: 5 ft, 10 ft nested, 36 ft in 9 ft panels. |
| `tests/` | 10 files | The marker and flow tests. All import `flight_daemon`. |
| `drone-vio.service` | — | The systemd unit that ran this daemon on the Jetson. |

## Marker spec (keep this consistent or nothing will detect)

- Dictionary: `DICT_4X4_50`
- Target marker ID: `0`
- Physical pad: `MARKER_SIZE_M = 0.20` (20 cm), set in `c12_precision_landing.py`
  and matched in `generate_landing_pad.py`
- Larger pads for altitude are in `markers/` (up to 36 ft, printed as panels)

## What has to change for the Pi

**1. Camera — the real work.** `flight_daemon.py` and `live_preview.py` build an
OAK-D pipeline (`dai.Pipeline()`, `dai.node.MonoCamera` at 400p, `CAM_B`).
A Pi camera needs `picamera2` or `cv2.VideoCapture` instead.

The harder part: the OAK-D also runs `dai.node.FeatureTracker` **on the camera**,
doing Harris corner detection in hardware. A Pi has no equivalent, so that work
moves to the CPU.

**2. Optical flow — probably delete most of it.** Despite its name,
`JetsonLKWorker` uses plain `cv2.calcOpticalFlowPyrLK` (CPU OpenCV, no CUDA), so
it would run on a Pi, just slower. But with a hardware flow sensor you do not
need it at all. The part worth keeping is `px_to_ned_velocity()`, which turns
pixel motion into NED velocity using height — the same maths applies to a flow
sensor's output.

**3. Serial port.** `cuav_mavlink_bridge.py` defaults to `/dev/ttyTHS1` at 115200
— that is the Jetson's UART. On a Pi use `/dev/serial0` (or `/dev/ttyAMA0`), and
free it from the console first.

**4. Compute budget.** `flight_daemon.py` is 3218 lines tuned for an Orin Nano.
A Pi has far less headroom, so expect to cut resolution or frame rate, and to
measure before trusting any loop rate.

**5. Runs on a Pi unchanged:** both pad generators — no `depthai`, no Jetson
dependency.

## Tests

The 10 tests here import `flight_daemon` and insert the parent directory on
`sys.path`:

```python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

Since everything was copied together, that still resolves inside this folder.
They were written against the OAK-D daemon, so they will need updating as the
camera layer is replaced.
