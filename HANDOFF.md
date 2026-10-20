# DroneVPS — handoff

Written 20 Oct 2026 for someone (or something) starting from zero. It assumes
no knowledge of this project, this aircraft, or ArduPilot. Everything here was
read off the hardware or measured on it; where a number is a guess it says so.

---

## 1. What this is

A quadcopter — now a hexacopter — that can hold position **without GPS**, by
looking at the ground with a downward camera and matching what it sees against
a satellite map. The companion computer works out where the aircraft is and
feeds that position to the flight controller **disguised as a second GPS
receiver**. The flight controller does not know the difference.

The pilot chooses which source is live with a switch on the transmitter.

### The hardware

| Part | What it is | Notes |
|---|---|---|
| CUAV X7 Pro | Flight controller | ArduCopter 4.7.1, hexa-X frame |
| Jetson Orin | Companion computer | 6 cores, runs everything below. Root on eMMC, project on an NVMe at `/data` |
| OAK-D Pro | Camera | Two mono sensors (1280×800, 30 fps) + a BNO086 IMU, pointing **down** |
| TF03-180 | Lidar | 4–20 mA through a 150 Ω resistor; the FC reads it directly |
| u-blox M9N | The real GPS | On GPS1. Often has no fix indoors |
| MK15E | Radio | Transmitter and video link |
| Tarot 4008 330 kV ×6, Hobbywing 40 A ESCs | Motors | ZD850 airframe |

**The Jetson has no battery-backed clock.** It boots with a stale time and
jumps forward when NTP lands, so early-boot log lines can be dated days
earlier. This is why logs are *numbered*, never timestamped.

---

## 2. How a position gets made

```
  OAK-D mono camera (30 fps)
        │
        ├─ left 130 px cropped off        ← the aircraft's own landing leg
        │                                   sits there; see §6
        ├─ optical flow (Lucas-Kanade + RANSAC affine)
        │     → pixel displacement between frames
        │     → metres, by dividing by (altitude / focal length)
        │
        ├─ camera IMU at 100 Hz
        │     gyro  → removes the apparent shift caused by TILT
        │     accel → fills the gap BETWEEN frames (~60 ms at 17 Hz)
        │
        └─ stereo depth (computed on the camera's own chip)
              → drops tracks that are not on the same surface as the rest

  position so far = dead reckoning. It drifts.

  every 0.5 s, above 60 m:
        XFeat (a small neural net) matches the frame against a satellite
        GeoTIFF and returns a real lon/lat → corrects the drift

  the corrected position is sent to the flight controller as GPS_INPUT,
  which arrives as GPS2.
```

**Why "disguised as a GPS" and not the obvious thing.** ArduPilot has a proper
interface for exactly this (`VISION_POSITION_ESTIMATE` with
`EK3_SRC1_POSXY = 6`). This project used it and moved away. The GPS path is
better understood by the EKF, needs no source switching in flight, and lets the
pilot flip between the real receiver and this one with a switch. So
`GPS2_TYPE = 14` (MAV), `EK3_SRC1_POSXY = 3` (GPS).

**A dead end, so nobody repeats it.** Visual-inertial odometry with the OAK-D's
stereo pair was tried and rejected on 28 Sep. The baseline is 7.5 cm, which
gives under 0.3 px of disparity above 25 m — useless at altitude. Stereo is
still used, but only for the plane-rejection above, never for height. Height
from stereo was also tried, on 9 Oct, and came out worse than the barometer.

---

## 3. What runs, and where

Three systemd services, all enabled and starting at boot:

| Service | Program | Job |
|---|---|---|
| `drone-router` | `mavlink_router.py` | **Sole owner** of `/dev/ttyTHS1` (the FC serial line). Everything else is a UDP client of it on `127.0.0.1:14650` |
| `drone-vps` | `vps_vio/live_node.py` | The camera, the flow, the map matching, the GPS_INPUT injection |
| `drone-rc-switch` | `vps_vio/gps_rc_switch.py` | Watches RC channel 6 and sets `GPS_PRIMARY` |

Plus `drone-watchdog.timer`, which restarts `drone-vps` if its liveness file
goes stale (a crash is caught by `Restart=`; a loop *stuck on a camera frame*
is not, and that is what the liveness file is for).

One command tells you whether the aircraft is ready:

```bash
/data/DRONE/systemd/after_reboot_check.sh
```

It prints enabled/active/restarts per service, any systemd ordering cycle this
boot, the liveness file and its age, and whether the journal is persistent.

### Things that will bite you

**One owner per device.** The FC serial line belongs to the router; open it
directly and you will split the stream with it. The camera opens exclusively:
`live_node` and `record_flight` cannot both run. Each program takes a file lock
(`vps_vio/single_instance.py`) and a second copy exits with code 3.

**A MAVLink client that only listens gets dropped.** The router forgets a UDP
client it has not heard from in 30 s, and the client then silently stops
receiving — from its own side that is indistinguishable from a dead link. Every
listener here heartbeats every 5 s. If you add one, make it do the same.

**No password is needed for the services.** `/etc/sudoers.d/dronevps` allows
start/stop/restart of the drone units without one, because the aircraft is
flown somewhere without a keyboard. Installing a *new* unit file still needs a
password, deliberately — that is workshop work, not field work.

---

## 4. The switch, and the rule behind it

RC channel 6, three positions:

| Pulse | Means |
|---|---|
| 1045 | Real GPS. `GPS_PRIMARY = 0` |
| 1495 | Centre — **does nothing**, holds whatever is selected |
| 1945 | VPS. `GPS_PRIMARY = 1` |

The centre is blank on purpose: passing through it on the way must not command
a change, and brushing the detent must not be an instruction.

**The pilot's rule, in their words:** *"vps ke time par kuch bhi ho jaye GPS ka
data nahi aana chiye"* — when the aircraft is on the VPS, the origin comes from
the point the pilot typed (`SCR_USER1/2`) and never from the receiver, even if
the receiver has a perfectly good fix sitting on the bench. And the other way:
when the switch is on the real GPS, **nothing from the VPS reaches the flight
controller at all**. Not "sent and ignored" — not sent.

`live_node` follows `GPS_PRIMARY` while it runs (`SourceWatcher`) and stops
injecting on 0. If the parameter cannot be read it keeps sending, deliberately:
injecting into GPS2 while the FC navigates on GPS1 is ignored, whereas going
silent while actually *on* the VPS leaves the aircraft with no position at all.

**The deadlock this created, and the fix.** The switch refuses to hand
navigation to an unhealthy VPS. Its original test was "does GPS2 have a fix?",
which can never be true while we are on the receiver and deliberately silent.
So it refused forever and the VPS was unreachable from the transmitter. It now
reads the node's own liveness file instead: *is the VPS fit to take over* is
not the same question as *is the VPS currently sending*.

---

## 5. Recording

`drone-vps` writes one numbered CSV per run to `flight_logs/vps/vps_NNNNN.csv`
— 39 columns: position, sigma, track count, every piece of velocity-bridge
state, both derotation estimates, the IMU sample counts, the stereo depth, the
lidar, and which source was live.

It records **whatever the switch is set to**. On a GPS flight you still get the
full VPS estimate alongside, with `src = GPS1`, which is exactly how you compare
the two without trusting the VPS with the aircraft.

Why this matters more than it sounds: on 20 Oct the camera accelerometer was
found to have never worked. The code read correctly, its tests passed — they
exercise the bridge directly, not the camera — and on the aircraft it did
nothing, because `IMUPacket` has no field called `linearAcceleration` (every
accelerometer report arrives in `acceleroMeter`). `getattr` returned `None` on
every packet. **The CSV is what said so**, and the CSV existed only because
recording had been switched on an hour earlier.

---

## 6. Measurements that explain the design

Everything here was measured on this hardware. Do not "optimise" any of it
without re-measuring.

**One torch thread, not six.** XFeat on this Orin: 1 thread 261 ms, 2 → 484,
3 → 510, 4 → 646, 6 (the default) → 1233. Monotonic, no sweet spot. It is a
small sparse model and torch's thread pool costs more to coordinate than the
work it divides. At six threads it could not finish inside its own fix period
*and* it occupied all six cores against a flow loop that needs one every 59 ms.
That — not stereo depth — was the "main loop stalled 44 s" contention the 9 Oct
report blamed on the camera.

**The camera's full intrinsics matter.** `cx = 624.89`, `cy = 391.80`, which is
15.1 px and 8.2 px from the geometric centre the motion used to be evaluated
at. The fitted affine carries rotation, so evaluating it off-centre turns
aircraft *rotation* into apparent *travel*: 0.298 px at 1°/frame, against
0.006 px once corrected. At 100 m and 17 Hz that is 0.64 m/s of phantom speed
removed, and the project's own unexplained hover phantom was 0.36–0.43 m/s.
Distortion is the 8-coefficient rational model with large coefficients and
moves an edge point 2–5 px.

**The IMU is 180° off the camera frame.** `getImuToCameraExtrinsics` reads
179.63°, essentially `diag(1,-1,-1)`. Using the documented body mapping on raw
IMU vectors would have had forward and back exactly backwards. Always read the
extrinsics.

**The map is anisotropic.** A geographic GeoTIFF has equal pixel size in
*degrees*, so in metres it is stretched by 1/cos(latitude) — 16.53% at 30.89 N.
Through XFeat the correction is not an optimisation: **9 of 9 points match with
it and 0 of 9 without**, median error 0.57 m. This is the project's long-standing
"16 m that has never been explained".

**The aircraft's leg is in the left camera's frame.** Columns 0–63 sit at 70–90
against a frame median of 147.5, tapering out by x≈120; the *right* camera is
flat across every column, which is what proves it is an object and not lens
vignetting. It held 32 of 300 tracked corners — rigid features that report zero
motion while the ground moves, and RANSAC cannot save you from that, because
they agree perfectly with each other. Cropped at the source.

**Camera IMU rates are at the hardware ceiling.** Luxonis caps
`GYROSCOPE_CALIBRATED` at 100 Hz on the BNO086. `LINEAR_ACCELERATION` could go
to 400 but is deliberately matched to 100: the reports share one packet, asking
for more was measured dragging the whole stream down, and six samples per 60 ms
gap already integrate to below the sensor's own noise.

---

## 7. The flight controller, as configured

Read off the aircraft 20 Oct. ArduCopter **4.7 renamed parameters to SI units**
— `LOIT_SPEED` became `LOIT_SPEED_MS` in m/s, `WPNAV_SPEED` became `WP_SPD`,
`ANGLE_MAX` became `PSC_ANGLE_MAX`. If you look for an old name and find
nothing, that is why.

| Parameter | Value | Meaning |
|---|---|---|
| `FRAME_CLASS` / `FRAME_TYPE` | 2 / 1 | Hexa, X |
| `GPS2_TYPE` | 14 | MAV — our injection arrives here |
| `GPS_PRIMARY` | 0 or 1 | Which source navigates. The switch sets it |
| `GPS_AUTO_SWITCH` | 0 | The FC will **not** change source by itself |
| `EK3_SRC1_POSXY` / `VELXY` | 3 / 3 | Position **and velocity** from GPS |
| `EK3_SRC1_POSZ` | 2 | Height from the **rangefinder** |
| `EK3_RNG_USE_HGT` | 70 | …below 70% of `RNGFND1_MAX` (60 m) = **42 m**. Barometer above |
| `EK3_GPS_CHECK` | 0 | GPS quality gates **off** — which is why our fix is accepted at all |
| `DISARM_DELAY` | 2 | Disarms 2 s after landing, in any mode |
| `LOIT_SPEED_MS` | 5 m/s | Reduced from the 12.5 default |
| `WP_SPD` | 4 m/s | Reduced from the 10 default |
| `PSC_ANGLE_MAX` | 15° | Lean limit |
| `MOT_SPIN_ARM` / `MIN` | 0.10 / 0.15 | **Set, not measured** |
| `FENCE_ENABLE` | 0 | Fence configured (100 m, 300 m) but **off** |
| `RNGFND1_OFFSET` / `SCALING` | 0.66537 / 72.61 | Calibrated 19 Oct against a measured 0.26 m |
| `INS_USE3` / `INS_ACC3_ID` | 1 / 0 | Third IMU **enabled but physically absent** |

IMU bench test, aircraft stationary, 20 Oct:

| | \|accel\| | noise | \|gyro\| |
|---|---|---|---|
| IMU1 | 9.809 m/s² | 0.002 | 0.0013 rad/s |
| IMU2 | 9.812 m/s² | 0.006 | 0.0108 rad/s |
| IMU3 | absent | | |

Both present IMUs are healthy.

---

## 8. What is NOT true yet

Be honest about this with the pilot; they ask, and they are right to.

- **Nothing in §6 has flown.** All of it is bench-verified only.
- **Velocity is the live risk.** `EK3_SRC1_VELXY = 3` means the EKF *fuses* the
  velocity we send, and sending it is new. This is the channel that reported
  47.7 m/s during a 106 m hover on 28 Sep and caused four EKF failsafes.
  `vps_vio/velocity_bridge.py` and its 23 tests exist for exactly that, but a
  test is not a flight.
- **The accelerometer's body mapping is unverified on this airframe.** It comes
  from a 27 Sep measurement and the camera has been remounted since. Flow stays
  the authority, so a wrong sign degrades what is carried between frames rather
  than the answer — but it wants a deliberate push along one axis to settle.
- **`DR_TRUST = 3.0`** (how fast the estimate doubts itself while dead
  reckoning) and the **0.5 m cap** on each map correction are chosen, not swept.
- **XFeat's thresholds** (`min_score` 0.25, `min_inliers` 12) have never been
  scored against real recorded frames.
- **The lidar spike filter is not in the flight path.** `EK3_SRC1_POSZ = 2`
  means the FC reads the TF03 directly on its own input.
  `vps_vio/lidar_filter.py` cleans what the *companion* records and uses; a
  lidar spike below 42 m still reaches the height estimate.
- **The router resyncs on about 0.71% of frames.** Low, not zero, unexplained.
- **The camera cannot be shared.** `live_node` and `record_flight` both open it
  exclusively, so saving images during a flight is not possible yet.


---

## 9. The code, file by file

Twelve cuVSLAM-era modules were deleted on 20 Oct — that path was rejected on
28 Sep and the files had been dead ever since, importing only each other and
tested only by their own small test directory. What is left is what runs.

```
/data/DRONE
├── mavlink_router.py        sole owner of /dev/ttyTHS1, UDP fan-out on :14650
├── drone_watchdog.py        restarts drone-vps when its liveness file goes stale
├── flight_daemon.py         the ArUco precision-landing daemon (separate feature)
├── cuav_mavlink_bridge.py   FC telemetry bridge; holds the lidar/camera mount offsets
├── HANDOFF.md               this file
│
├── systemd/
│   ├── drone-router.service      ─┐
│   ├── drone-vps.service          ├ enabled, start at boot
│   ├── drone-rc-switch.service   ─┘
│   ├── drone-watchdog.service / .timer
│   ├── install.sh                install / enable / start / status
│   ├── field_setup.sh            run ONCE as root: passwordless control +
│   │                             persistent journal + reinstall the units
│   └── after_reboot_check.sh     one screen: is the aircraft ready?
│
├── vps_vio/                 everything the VPS node is made of
│   │
│   │   ── the flying path ──
│   ├── live_node.py         THE node. Camera, flow, fusion, GPS_INPUT, the
│   │                        notifier, SourceWatcher, AmslRef, the CSV
│   ├── flow_odom.py         FlowTracker (LK + RANSAC affine, undistortion,
│   │                        the stereo plane gate) and shift_to_ne
│   ├── velocity_bridge.py   flow + accelerometer -> velocity, with the bias
│   │                        rule that caused the 8 Oct collapse
│   ├── map_fuse.py          the position filter: predict, update, outlier
│   │                        gate, and the 0.5 m cap on a correction
│   ├── map_fix_worker.py    runs the matcher on its own thread, refuses
│   │                        below 60 m
│   ├── map_match_xfeat.py   XFeat. Sets torch to ONE thread (see §6)
│   ├── map_query.py         rescale to the map's GSD and rotate north-up,
│   │                        including the east-west anisotropy correction
│   ├── map_cache.py         the GeoTIFF in RAM, and windows out of it
│   ├── map_db.py            tiling and geo-referencing helpers
│   ├── config.py            every measured constant, with the measurement
│   │                        written beside it. READ THIS ONE.
│   ├── lidar_filter.py      TF03 spike filter (companion-side only — §8)
│   ├── single_instance.py   the file lock that stops two copies
│   ├── fc_port.py           serial-port arbitration from before the router
│   ├── gps_rc_switch.py     channel 6 -> GPS_PRIMARY, with the health check
│   ├── record_flight.py     bench recorder: frames + 56-column telemetry.
│   │                        Also holds Telemetry, which live_node uses
│   ├── frame_writer.py      [IN PROGRESS] save frames from inside the
│   │                        navigating process — see below
│   │
│   │   ── tools, not in the flight path ──
│   ├── analyse_flight.py    map-matching accuracy of a recorded flight
│   ├── bench_workload.py    time every stage, then all of them together
│   ├── lidar_cal.py         calibrate the rangefinder against known heights
│   ├── gps_switch.py        pick a GPS by hand and watch the other
│   ├── rc_watch.py          which RC channel moves when you flip a switch
│   └── map_match.py         older matcher, kept for offline comparison
│
├── tests/                   32 suites. Each prints PASS/FAIL, exits non-zero
│   │                        on failure, and its docstring names the real
│   │                        failure it pins, with dates and numbers.
│   ├── test_vio_gap_fill.py       the accelerometer filling the frame gap
│   ├── test_camera_k.py           rotation-as-translation, undistortion
│   ├── test_stereo_gate.py        the plane gate, and when it refuses
│   ├── test_source_gate.py        nothing reaches the FC while on GPS
│   ├── test_switch_deadlock.py    the switch can still reach a silent VPS
│   ├── test_client_registration.py  a listener that stops talking goes deaf
│   ├── test_systemd_units.py      every ExecStart flag is real; no cycles
│   ├── test_single_copy.py        one copy, and a kill -9 frees the lock
│   ├── test_csv_columns.py        the log header matches its rows
│   ├── test_lidar_filter.py       spikes up and down
│   ├── test_map_anisotropy.py     9/9 with the correction, 0/9 without
│   ├── test_velocity_bridge.py    the 8 Oct collapse, pinned
│   └── …                          router, notifier, origin, markers, watchdog
│
├── flight_logs/
│   ├── vps/vps_NNNNN.csv    one numbered file per run of the node
│   ├── vps_stdout.log       the node's own output
│   └── *.BIN                flight controller dataflash
│
├── third_party/accelerated_features/   XFeat checkout + weights
└── bhh.tif                  the satellite sheet (gitignored, 91 MB)
```

### The camera-sharing work, in progress

`live_node` and `record_flight` both call `dai.Device()`, and the OAK-D permits
one owner — so on a flight, where navigation has to win, no images are ever
recorded and there is nothing for offline map matching to work on. That is
being fixed now, in `vps_vio/frame_writer.py`, and the shape is worth knowing
because the constraint it works under applies to anything else added to that
loop.

The process that already holds the camera is the one that writes the frames;
no second process, no socket, no shared memory — a channel between processes is
one more thing that can block the flow loop. The rule the module is built
around is that **a saved frame must never cost a flow update**:

- `offer()` does no encoding, no disk work, and never waits. It hands a
  reference to a bounded queue and returns.
- The queue is small and uses `put_nowait`. When the disk or the encoder falls
  behind, the frame is **dropped and counted**. There is no `put()` with a
  timeout anywhere in the file, because a timeout is just a smaller stall.
- Encoding runs on a daemon worker thread.

The numbers behind that: the flow loop has ~59 ms a frame at 17 Hz and spends
22 ms of it; JPEG encoding costs 9.5 ms a frame and PNG 30 ms, so encoding on
the loop thread would eat a sixth to a half of the budget keeping the
aircraft's position alive.

Frames are named by the **CSV row index**, not by the clock — `offer()` takes
the index the caller is about to write and returns the filename it chose, which
the caller then puts in that row's `frame_file` column. One row, one image, and
nothing that a jumping clock can corrupt.

---

## 10. Running it

```bash
# Is the aircraft ready?
/data/DRONE/systemd/after_reboot_check.sh

# Every test suite (32 of them). Each prints PASS/FAIL and exits non-zero.
cd /data/DRONE && for t in tests/test_*.py; do python3 "$t"; done

# Stop the service to free the camera for bench work, then put it back
sudo systemctl stop drone-vps     # no password needed
sudo systemctl start drone-vps

# What the node is doing right now
tail -f /data/DRONE/flight_logs/vps_stdout.log
cat /run/drone/vps.alive
journalctl -u drone-vps -f
```

Pinned versions, each learned the hard way:

- `depthai==2.33.0.0` — **not** v3, which has no `XLinkOut` and breaks everything
- `torch==2.3.1` CPU build — not newer (2.14 fails with `undefined symbol: sbgemm_`)
- XFeat is a checkout under `third_party/`, not a package. Weights are present.

The repository is `github.com/deepak92kumar/DroneVPS` (private), working branch
`recover/post-ssd-loss-19oct`. **Push over SSH, not HTTPS** — pasting into the
terminal does not work on this machine, which cost five failed attempts before
anyone noticed.

---

## 11. How to work on this

The commit history is the real documentation. Each commit explains what was
broken, what the measurement was, and why the fix is the fix. Read a few before
changing anything.

Three habits this project runs on:

**Measure before you believe.** Several things here looked right in the source,
passed their tests, and did nothing on the aircraft. The accelerometer field
name, the systemd ordering cycle, the RC switch going deaf, the `--gps2` flag
that never existed — none of those were visible from reading the code.

**A test pins a real failure.** Every test file here opens with the failure it
exists to prevent, with the date and the measured numbers. Write them that way;
a test that cannot fail is worse than none, and two of them in this project
were caught doing exactly that and rewritten.

**Say what is not verified.** The pilot flies this aircraft. A guess presented
as a measurement is the one thing that can get somebody hurt.

One more, from the pilot directly: *"koi bhi guard likhne ya lagne se pahle
pucho"* — ask before adding a safety guard. Three guards once combined to strand
the aircraft with no way out from the transmitter. They know when it is safe to
fit props; the software does not.
