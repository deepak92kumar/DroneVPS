# DroneVPS — where it stands and what is left

Written 20 Oct 2026, after a day of flights and after reading the fifteen
pre-SSD-loss status reports (`~/DRONE_fc_logs_backup/old_status_reports/`) and
the flight controller's own logs from when the VPS was working.

Three sources are used throughout and they are not equal. **A number from a
flight log beats a number from a report, and a report beats reasoning.** Where
something has not flown, it says so.

---

## The audit: what the rebuild changed without meaning to

The NVMe took the code; what came back was rewritten from the reports. The
logic mostly survived. **The numbers did not** — and every one that drifted
made the VPS look more trustworthy than it is.

### Found and put back on 20 Oct

| | was (rebuild) | now | evidence |
|---|---|---|---|
| GPS_INPUT rate | 30 Hz | **4.3 Hz** | logs 191/194 carried 4.28 and 4.31 Hz |
| the rate gate's own arithmetic | `= t + period`, 3.913 Hz | **`+= period`, 4.299 Hz live** | replayed on all thirteen recordings' real timestamps |
| horizontal accuracy | the drift alone | **origin error AND drift** | the three 20 Oct flights declared 14 m against a true 151 m, all of it the origin's |
| `fix_sigma_m` in the flying code | 12 | **20** | `live_node` overrode MapFuser's default, its docstring and the test, all of which said 20 |
| speed accuracy | `0.0867 x alt`, fitted on RAW flow | **`0.6 + 0.025 x alt`** | re-fitted on what leaves the SLEW; the old one over-stated by 3x |
| velocity output slew | none | **5 m/s², 2 s hold** | 9 Oct: *"input gate only → … 16.99 to 2.37 m/s. This is what let the mission finish"* |
| vertical accuracy sent | sigma × 2 (~42 m) | **3.0 m constant** | `GPA.VAcc` is exactly 3.00 through both logs |
| speed accuracy sent | flat 0.4 m/s | **0.0867 × altitude** | flights sent 0.93–1.55; measured σ = 0.0867·alt over three flights |
| `fix_sigma_m` | 30 | **20** | 9 Oct: *"the code's fix_sigma_m = 20 was already right"* |
| origin precedence | SCR_USER always | **receiver on GPS1, SCR_USER on VPS** | 13 Oct: *"On the VPS it now wins; on GPS1 the receiver still does"* |
| armed flag source | any heartbeat | **system 1 component 1 only** | 9 Oct: *"Armed state read only from system 1, component …"* |

The last two had each been decided before the loss and came back wrong. The
origin one was quietly wrong for three flights, each with the origin 147–157 m
from the aircraft.

### Checked and correct — no action

| | |
|---|---|
| `min_alt_m` 60, map window `0.60 × altitude`, `fix_period` 2.0 s | match 3–4 Oct |
| IMU batch threshold **1** | 3 Oct: batching 5 left 34.2% of frames with no IMU sample; 1 took it to 1.1% |
| hover deadband `0.35 + 0.005 × alt` | 4 Oct, flown and kept |
| GPS_INPUT fix type 3, 12 satellites | matches both working logs |
| horizontal accuracy = the fuser's own sigma | p50 21.28 / 21.36 m in the logs |
| watchdog stands down while armed | 9 Oct fix, survived |
| `FRAME_CLASS/TYPE` 2/1, `SERVO1..6` = motors 1–6 | hexa-X, mixer proven by a 0.0% residual fit |
| `INS_GYRO_FILTER` 28 Hz | exactly the 16" interpolation |

### Flags that vanished, and why that is fine

The 4 Oct command line was `--flow-scale --accel-vel --send-vel --xfeat
--usb2`. None of those flags exist now, but four became the behaviour rather
than an option: XFeat is the only matcher left, the accelerometer is always in
the bridge, sending velocity is the default (`--no-send-vel` is the way off),
and flow scale always uses the FC's height. `--usb2` was never a fix — 30 Sep:
*"an unbooted MyriadX always enumerates as USB2. It proved nothing"*, and the
camera drops were cured by a soft cable.

### Not yet checked

- `--image-hz` is now `--save-frames`; the reports left it at "NOT YET" twice
  (4 Hz, then 4 → 2). It flies at 4 Hz today, armed-only.
- The flow tracker's own constants (1200 corners, quality 0.01, min_dist 8,
  winSize 31, maxLevel 4) appear in no report, so there is nothing to compare
  them against. They have flown all week.

---

## Camera and flow, as they stand

Measured on 20 Oct, against the reports where there is a comparison:

| | today | the reports |
|---|---|---|
| loop rate | **30.0 Hz**, unchanged with frames saving | daylight 12–22 Hz (9 Oct) |
| tracks, daylight | 250–290 | 241–290 (9 Oct) |
| tracks at 60–80 m | 211–228, min 69–120 | — |
| FC vibration, airborne | VibeZ p50 **1.72**, max 2.98, zero clipping | 9.62 → 5.13 after the arm screw (13 Oct) |
| camera IMU, hover | accel Z **2.46 m/s²**, gyro X 4.72 °/s | camera 6.55 → 3.74 (13 Oct) |
| camera vs FC, accel Z | camera 1.44× the FC | — |
| camera IMU noise floor | gyro 0.0002 rad/s, accel 0.017 m/s² | — |
| exposure cap | `setAutoExposureLimit(8000)` µs | 9 Oct: *"Exposure caps the sensor"* at night |

The camera is in better shape than it was: more frames, the same track count,
and a quarter of the vibration. Nothing here needs fixing.

---

# TODO

Ordered by what blocks what. Nothing below has flown unless it says so.

## 1 — Before the next flight (ground, minutes)

| | what | why |
|---|---|---|
| ☐ | **Zero the lidar at the field, after the final power cycle** | the zero moved 0.66537 → 0.69663 → 0.70449 → 0.72316 V in one afternoon. Over 2 m and the aircraft will not auto-disarm. `lidar_zero.py --apply` |
| ☐ | **Check whether companion messages reach the handset** | 13 Oct: *"Companion status lines still do not reach the radio link."* Every "you will see it on the radio" in this project rests on this being true, and it is unverified |
| ☐ | **Set `SCR_USER1/2` at the actual take-off point** | still the VPS origin when on the VPS. The arm-time re-take covers the GPS case but not this one |

## 2 — The next flight (GPS only, no VPS switch)

| | what | why |
|---|---|---|
| ☐ | **100–120 m, not 70** | the matcher's own error is 17.81 m at 95–110 m and **68.36 m at 60–70 m** (12 Oct). Both 20 Oct attempts were at 70–72 m |
| ☐ | **Before noon** | 30 Sep: *"the map has a time of day"* — residual 4.61 m at 14:46, 15.81 m at 17:35, with contrast 43.8 → 32.1 |
| ☐ | **30 s hands-off hover at 15–20 m as well** | the only way to compare the tuning against log 217 like for like. Wind and altitude confounded the 218 comparison |
| ☐ | Confirm it auto-disarms on landing | the lidar fix has not been seen to work end to end |

What that flight settles: whether map fixes appear at all at the documented
altitude. Everything about flying on the VPS waits on that answer.

## 3 — Then, and only then: the first VPS segment

| | what | why |
|---|---|---|
| ☑ | **The handover re-anchors now** (21 Oct, not flown) | switching to the VPS in flight re-takes the origin from where the FC says it is and drops the accumulated VIO displacement, so the first VPS position equals what the EKF already holds. Before this it sent `take-off origin + every metre of drift from the GPS phase` — measured 5.24 and 5.73 m on two flights and a 20.55 m single-sample jump on a third. The pilot's rule: GPS→VPS must behave the way VPS→GPS already does, which injects its current position rather than flying off the error |
| ☐ | GPS take-off → climb to 110 m → VPS for 60 s → **back to GPS before the descent** | 13 Oct: error went **4.13 → 26.0 m in twelve seconds** of a 6 m/s descent below 60 m, and *"it is the one number that has not moved all week"* |
| ☐ | Mode switch in hand, ALT_HOLD on positions 2 and 4 | it removes the position controller entirely, inside the FC's own loop, with no companion in the path |

## 4 — Open faults, in the order the reports ranked them

| | what | evidence |
|---|---|---|
| ☐ | **Velocity to the FC** — 54% on 13 Oct, *"still the largest open fault"* | the slew and an altitude-scaled accuracy are restored but **not flown**. Replayed through the restored path the 20 Oct flight's worst is 9.77 m/s against log 194's 8.30, and the declared accuracy now tracks the measured output (1.62 m/s p50 against 194's 0.93). At 60–200 m, 78.6% of messages carry a velocity against 194's 60% |
| ☐ | **The descent below 60 m** | no absolute reference exists down there; the step cap is 0.5 m per fix, so 5 s buys 2.5 m against a 29 m error |
| ☐ | **Heading / yaw** — 55%, *"compass calibration is still the pilot's next step"* | measured 20 Oct: field 509/518 mG, offsets under 50, the two compasses agree to 1.8°, and the EKF's yaw resets moved yaw by −1.2° and −0.02°. **So the compass is healthy and this may now be closed** — but it was listed open |
| ☐ | **Map coverage** — 64% | `bhh.tif` is 3.31 × 2.14 km. A 3–5 km mission leaves it |
| ☐ | **The matcher's position output at 61–72 m is not usable** | measured offline 20–21 Oct on 299 saved frames, prior at the true GPS. Handed a window 600 m from where the aircraft was, it still reports a position 16 m from THAT window's centre — so the 9.5 m error against GPS was measuring the window, not a match. `min_score` 0.25 is unreachable (max 0.140 on any frame) and is doing the job it was added for. Yield carries signal (46/99 fits on the true place against 6–14/99 off it); position does not. **Nothing here speaks to 80–120 m — there are no saved frames above 72 m.** See FROM_THE_OLD_REPORTS.md §9 |
| ☐ | **Distance** — 22%, the row that *is* the mission | 383 m is the whole evidence base. 500 m, then 1 km |

## 5 — Mechanical, measured and still open

| | what | evidence |
|---|---|---|
| ☐ | **Yaw share of the hover imbalance** — 50–57% across three flights, halved by the 20 Oct work but not removed | 13 Oct: **one missing arm screw** took the yaw differential from −95 to −12.5 and the vibration with it. Check every arm screw |
| ☐ | Re-run the propeller swap test if it persists | motors 3 and 4 are both CW, so swapping them is valid: torque follows the blade → propeller; stays with the corner → motor or arm. One flight settles it, and the log has no ESC telemetry so nothing else can |
| ☐ | **CoG** — 5.3 mm left / 8.7 mm forward after the last adjustment | 11 Oct: the lidar's cantilever alone was +41 to +51 µs of left-right imbalance before it was removed |
| ☐ | Re-check fasteners between flights | 10 Oct: motor spread 99.7 → 117.9 µs between two flights of the same day with nothing touched, *"the second time in three days"* |

## 6 — Tuning, after the next hover

| | what | why |
|---|---|---|
| ☐ | Read the phase lag again | 20 Oct took it −65° → **−55°** and the gain 0.90 → 0.68. If it is still there, `ATC_ANG_RLL_P`/`PIT_P` 4.5 → 3.5 is the next lever |
| ☐ | Then Autotune | it measures the response instead of reasoning about it. `AUTOTUNE_AXES=7`, `AGGR=0.075` are already set |
| ☐ | `MOT_THST_EXPO` 0.67 → ~0.71 | ArduPilot's table interpolates 0.71 for 16"; it should really come off a thrust stand |

## 6a — Settled on 20 Oct by measurement, not yet flown

| | what | evidence |
|---|---|---|
| ☑ | The rate gate holds 4.3 Hz | 4.299 Hz measured on the live bus over 90 s, no gaps. Was 3.913 |
| ☑ | The declared horizontal accuracy carries the origin's error | 21.08 m live against log 194's 21.36. The drift alone reads 3.80 m |
| ☑ | `fix_sigma_m` is 20 in the flying code too | it said 12; no effect on the 20 Oct flights, which took no fixes at all |
| ☑ | The speed accuracy is fitted on the bridge's OUTPUT | `0.6 + 0.025 x alt`, never more than 0.044 m/s optimistic across 5–70 m |
| ☐ | **None of the four has flown.** The next flight is what tests them | |

## 7 — Things that need no flight and keep being deferred

| | what | evidence |
|---|---|---|
| ☐ | **A surveyed ground point** | asked for on 10, 12 **and** 13 Oct and never done. *"GPS1 itself wandered 2.4 to 27.4 m this morning while the aircraft stood still. Whether the sheet or the receiver is wrong about the ground cannot be settled from the aircraft."* Every error figure in this project — including 20 Oct's judgement that the camera frame matches `bhh.tif` — is measured against that receiver |
| ☐ | **Battery current sensor** | reports 17.3 A with the motors merely ticking over. Until it is calibrated every mAh figure is fiction, and `BATT_FS_VOLTSRC = 1` (sag-compensated failsafe, the correct fix for the voltage question) cannot be used |
| ☐ | **Decide `ARMING_SKIPCHK = 6156`** | compass, GPS lock, hardware safety switch and GPS configuration pre-arm checks are all skipped |
| ☐ | **Decide `FS_EKF_THRESH = 1.2`** | ArduPilot documents 0.6 strict, 0.8 default, 1.0 relaxed. 1.2 is looser than the loosest |
| ☐ | **Move the lidar to serial** — no longer a judgement call, 21 Oct | 5 Oct: *"The cause is the wiring: RNGFND1_TYPE=1 on analog pin 4, when the TF03 is a serial device."* By 9 Oct it was refitted and 94% usable below 58 m; this aircraft is back on analog. **Measured 21 Oct with the pack on:** the TF03 reaches 180 m, accurate to ~70, which is why `MAX` is 60 — and 180 m over `SCALING` 72.61 m/V is 2.48 V of full scale, so 1 mV = 7.26 cm is inherent to the analog interface, not a bad setting. The whole 0–2 m band the land detector needs is **27 mV**, while the zero moved **58 mV** in one afternoon (0.66537 → 0.72316 V). The drift is twice the range auto-disarm depends on, so re-zeroing more carefully cannot fix it — only serial can. Pointed at a wall the sensor read 0.93 m p50 and 57 of 89 samples non-zero — it works; the ground reading of a flat 0.000 m is `RNGFND1_OFFSET 0.723159` clamping 0.26 m to nothing |
| ☑ | ~~CLAHE in the matcher?~~ **Already there** | `clipLimit` 3.0, `tileGridSize` 8×8, on both the query and the map window, `map_match_xfeat.py:242`. Not an open question |

## 8 — Known and deliberately not acted on

- **Emergency stop switch** (`RCx_OPTION = 31`). Offered, not taken. The pilot's
  rule is to ask before fitting a guard.
- **Derotation runs off the FC attitude**, which is exactly 0 on 87% of frames
  (it updates at ~4 Hz against a 30 Hz loop), and the camera gyro — sampled
  properly at 100 Hz — is **sign-inverted** against the FC on both axes
  (cumulative r = −0.80). Switching is right for the timing and wrong until
  the sign is fixed.
- **Stereo depth reads 2.62 m at a true 4.5 m.** Only used for the plane gate.
- **`EK3_SRC1_VELXY = 3`** — the EKF still fuses the velocity we send. The
  restored slew and accuracy are what make that safe; neither has flown.
- **The deadband sends nothing below `0.35 + 0.005 x alt`.** Measured over all
  thirteen recordings it suppresses 98.5% of messages — but 44,539 of the
  77,540 are at 0–2 m, the aircraft standing on the bench. At 60–200 m it
  suppresses 21.4%. Working as designed; left alone.
