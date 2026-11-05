# DroneVPS — Map matching: what the 3–4 November flights showed, and what was changed

**Aircraft:** ZD850 hexacopter, CUAV X7+ (ArduCopter 4.7.2), Jetson Orin Nano,
OAK-D Pro downward camera, u-blox M9N receiver (reference only).
**Repository:** `recover/post-ssd-loss-19oct`, mirrored to `feat/vio-vps`, head `76d2544`.
**Map:** `bhh.tif`, 7569 × 4200 px, 0.44 × 0.51 m/px, 3307 × 2138 m = 7.07 km²,
2022 satellite imagery.
**Status of every change in this report: UNFLOWN.** Offline replay only.

---

## 1. Summary

The 4 November 12:30 waypoint flight hit four waypoints correctly and then
drifted away from truth, reaching 124.7 m of position error while the aircraft
was declaring 8.5 m of accuracy to the flight controller. The pilot switched to
GPS and recovered the aircraft.

Replaying that flight and the good one from the same morning established five
things, each measured against the M9N receiver, which is the only position on
board that is not partly our own injection returning:

1. **The flight controller was being told the wrong matcher's answer.** Of 153
   map fixes published on that flight, every single one came from NCC and none
   from XFeat, because of a result-selection defect. NCC's fixes reached 180.74 m
   of error. XFeat succeeded 51 times and published nothing.
2. **The runaway was not caused by bad fixes.** It is optical-flow drift. With
   the map removed entirely in replay, the estimate still ran away.
3. **The aircraft had no way to say it was lost.** The declared accuracy
   understated the real error by a factor of fifteen, and grew more confident as
   the aircraft grew more lost.
4. **XFeat's matching step, not its tuning, was the limit.** Eighteen tuning
   configurations were measured and all were flat. Replacing the matching step
   with a learned matcher (LighterGlue) raised the fix rate from 50% to 68% and
   worked in the exact conditions where XFeat fails.
5. **A lost frame can be found again on the map.** From 51 to 131 m of position
   error, LighterGlue with a tiled search recovered 3 of 4 frames to 3.6–8.3 m.
   The matcher that flies today found none of four.

Five software changes were made. No flight-controller parameter was touched.

---

## 2. Flight record, 3–4 November

Flights reaching operating height. Fix counts are from the on-board estimate log.

| Run | Date, time | Duration | Max alt | Map fixes | Frames saved |
|---|---|---|---|---|---|
| vps_00431 | 3 Nov 09:07 | 229 s | 38.1 m | 0 | 561 |
| vps_00432 | 3 Nov 09:12 | 214 s | 36.4 m | 0 | 673 |
| vps_00436 | 3 Nov 11:12 | 663 s | 72.3 m | 86 | 2294 |
| vps_00445 | 3 Nov 13:11 | 633 s | 80.5 m | 19 | 1127 |
| vps_00448 | 3 Nov 14:07 | 521 s | 79.9 m | 43 | 1193 |
| vps_00451 | 3 Nov 15:52 | 519 s | 81.3 m | 56 | 1718 |
| vps_00455 | 3 Nov 16:53 | 679 s | 71.7 m | 76 | 1950 |
| vps_00468 | 3 Nov 21:06 | 544 s | 70.2 m | 79 | 1458 |
| **vps_00476** | **4 Nov 10:36** | **584 s** | **112.0 m** | **141** | **1611** |
| **vps_00478** | **4 Nov 12:30** | **830 s** | **110.7 m** | **84** | **2103** |

The two 4 November flights are the ones analysed in depth here, because they are
the first at 110 m with frames saved and because one succeeded and one did not,
on the same site, with the same software, 1 h 54 min apart. The 3 November
flights were flown below 82 m. Two of them, vps_00436 and vps_00455, were
re-measured for this report because they are the evidence for what the map
does at 70–82 m; the rest were analysed on the day for separate matters
(the declared speed-accuracy path, the ignore-flag behaviour, and the return of
the aircraft to its 27 October software and parameter state); they were not
re-analysed for this report.

Flight-controller logs were matched to the on-board logs by cross-correlating
yaw **and** barometric altitude and requiring both to agree on the clock offset.
This was necessary because the log filename carries the flight controller's own
clock and identifies nothing: the log of the 12:30 flight is named `12-54-13`.
Both flights matched at r = 0.999–1.000 on both channels with the two offsets
agreeing to 0.00 s.

---

## 3. What happened on the 4 November 12:30 flight

Position error against the M9N, by phase:

| Phase | Prior error (median) | Map accept rate | Camera/window overlap |
|---|---|---|---|
| t 200–300 s | 6.1 m | 59% | 75% |
| t 300–400 s | 11.1 m | 83% | 74% |
| t 400–500 s | 42.2 m | 8% | 61% |
| t 500–600 s | 124.0 m | 0% | 23% |
| t 600–900 s | 120.1 m | 0% | 10% |

The map's search window is centred on the aircraft's own estimate. Once that
estimate had drifted past about 20 m, the window no longer contained the ground
the camera was looking at, so the matcher's later failure is a **consequence** of
the drift rather than its cause.

**What drifted was the optical flow.** Net displacement over 20-second windows,
compared with the M9N, gives a scale of 0.907 at the median but swinging from
0.38 to 1.70 — so the position random-walks rather than carrying a steady bias.
The drift rate reached 0.546 m/s from t = 400 s and 0.625 m/s from t = 500 s.

**The map could not keep up, and raising its authority does not help.** A fix is
capped at 0.5 m every 2 s, which is 0.250 m/s — less than half the drift rate.
But replaying the flight closed-loop, with the real filter and the matchers
re-run inside the loop so the window follows the replayed estimate:

| Policy | t 300–400 | t 400–500 | t 500–600 | Final error |
|---|---|---|---|---|
| no map at all | 20.6 m | 25.3 m | 107.3 m | 91.9 m |
| map, cap 0.5 m | 10.6 m | 44.8 m | 125.6 m | 109.3 m |
| map, cap 2 m | 9.7 m | 38.4 m | 119.7 m | 103.9 m |
| map, cap 5 m | 9.7 m | 36.6 m | 116.3 m | 100.6 m |
| map, cap 20 m | 9.7 m | 36.6 m | 116.3 m | 100.6 m |

Caps of 5 m and 20 m are identical, so beyond 5 m the cap does not bind at all,
and in the late phase every map policy is worse than no map. **The map is a
drift-bounder, not a recovery mechanism.** Fixing the runaway means fixing the
flow, or recovering the position outright — not tuning the map.

---

## 4. The five findings

### 4.1 The flight controller was hearing the wrong matcher

`map_fix_worker` ran both matchers and kept the first result unless a later one
succeeded where the earlier had not. The pair is ordered `["ncc", "xfeat"]`, so
**whenever NCC succeeded its answer was published and XFeat's was discarded.**

This was only discoverable because `fix_inliers` read zero on all 153 fix rows
of the flight — NCC does not produce inliers. An inference from a side effect
is not a record, so the matcher now names itself in the log (`fix_source`).

Measured on 675 frames from both flights, both matchers, tilt-corrected, against
the M9N:

| | n | NCC error p50 / p90 / max | XFeat error p50 / p90 / max |
|---|---|---|---|
| both accepted | 339 | 3.80 / 8.09 / **85.69** | 3.55 / 6.49 / **11.36** |
| only NCC accepted | 226 | 25.01 / 146.55 / **180.74** | (XFeat declined) |
| only XFeat accepted | 9 | (NCC declined) | 5.18 / 7.37 / 9.00 |
| neither | 101 | both ≈ 104 m | |

**XFeat's gate is a self-assessment and NCC's is not.** By how wrong the
estimate was when the frame was handed over:

| Prior error | XFeat accept / worst | NCC accept / worst |
|---|---|---|
| 0–5 m | 72% / 7.88 m | 100% / 68.75 m |
| 5–10 m | 81% / 9.46 m | 97% / 93.05 m |
| 10–20 m | 76% / 11.26 m | 90% / 68.70 m |
| 20–50 m | 26% / 11.36 m | 94% / 66.69 m |
| 50–200 m | 1% / 11.07 m | 55% / **180.74 m** |

XFeat declines when it should: it accepts 1% of frames when the estimate is
50–200 m out. NCC accepts 55% of them with a median error of 96.88 m.

Taking XFeat's answer and publishing nothing when it declines costs a third of
the fixes and removes **every** fix over 20 m:

| | Fix rate | p50 | p90 | Worst | Fixes > 50 m |
|---|---|---|---|---|---|
| NCC wins (as flown) | 574/675 (85%) | 4.91 m | 89.83 m | 180.74 m | **109** |
| XFeat trusted | 348/675 (52%) | 3.57 m | 6.59 m | 11.36 m | **0** |

On the good flight the rate falls 98% → 79% and the worst error 85.69 → 9.16 m.
The good flight was not safe either: it simply did not drift far enough for its
eleven fixes over 50 m to matter.


**Below 90 m the same comparison comes out differently, and that is why NCC is
still used there.** Measured on the two 3 November flights that cruised at
70–82 m, 197 attempts:

| | Coverage | Error p50 | Worst | Fixes > 20 m |
|---|---|---|---|---|
| NCC alone | 172/197 (87%) | 11.78 m | 68.92 m | 65 |
| XFeat alone | 94/197 (48%) | 4.80 m | 21.53 m | 1 |
| **XFeat, else NCC** | **186/197 (94%)** | 6.64 m | 45.81 m | 47 |

NCC is not more accurate down there — 11.78 m against 4.80 m, and it produced a
68.9 m fix on a frame whose estimate was under 10 m out, because its gate does
not self-assess at any height. What it is, is **available**: 87% against 48%,
with LighterGlue not asked below 90 m at all. Trusting XFeat first and falling
back to NCC keeps the whole 94% and takes the worst case from 68.92 m to
45.81 m, because wherever XFeat speaks it is believed instead.

### 4.2 The aircraft could not say it was lost

| Phase | Real error | Declared accuracy |
|---|---|---|
| t 200–300 s | 2.9 m | 2.60 m |
| t 300–400 s | 10.9 m | 5.10 m |
| t 400–500 s | 41.2 m | 6.20 m |
| t 500–600 s | **124.7 m** | **8.50 m** |

`corr(declared, real) = +0.940` — the direction is right, the magnitude is out
by fifteen. Across the 4582 samples where the error passed 50 m, the declared
accuracy never reached 9.60 m, and `EK3_GLITCH_RAD` is 25 m, so the flight
controller had no reason to doubt the position.

The mechanism is circular: every accepted fix pulls the filter's sigma down, and
the fixes were being accepted **because** they agreed with a drifting prior. The
aircraft grew more confident as it grew more lost.

A detector was built from signals the aircraft already has at the moment of a
fix attempt — no truth, no hindsight. Candidates scored two ways: how much error
the rule lets through on the flight that ran away, against how often it fires on
the flight that never got lost (whose error peaked at 26.4 m, so anything there
is a false alarm):

| Rule | Fires at | False alarms |
|---|---|---|
| **5 refusals, score < 0.05, above 75 m** | **24.7 m** | **0 of 299** |
| 40 s since the last fix | 69.2 m | 0 |
| 5 refusals, above 75 m | 24.7 m | 1 |
| accept rate over 10 ≤ 30% | 4.0 m | 35 |

A refused score far *below* the gate means "there is nothing in this window",
not "I nearly had it" — that distinction is what separates the working rule from
the one that false-alarms. It fires before the run from 25 m to 124 m.

### 4.3 XFeat's tuning was not the problem

Eighteen configurations, 20 frames, every other variable held:

| Change | Error p50 | Worst |
|---|---|---|
| as flown (top_k 1024, threshold 0.05, mutual NN, median fit, tol 4) | 9.3 m | 22.2 m |
| mutual-NN similarity floor 0.5 / 0.7 | 9.3 m | 22.2 m |
| similarity floor 0.82 | 10.0 m | 27.5 m |
| top_k 2048 / 3072 / 4096 | 9.3 m | 22.0 m |
| detection threshold 0.03 / 0.07 / 0.10 | 9.8–10.0 m | 22.2 m |
| RANSAC instead of median-of-offsets | 9.9 m | **30.4 m** |
| XFeat\* semi-dense | 10.8 m | 25.1 m, 1.9× the cost |

A flat sweep says the knob is not the thing. Two further hypotheses were
measured and rejected: **widening the search window** (0.60 → 1.40 × altitude
changed the answer by under 2 m on 17 of 20 frames and collapsed the accept rate
from 75% to 15%) and **interpolating attitude to the frame's instant** (at the
10 Hz the attitude arrives, 100 ms of staleness costs nothing; 400 ms doubles
the error, so only the *age* needs recording).

Separately, the inlier ratio was found to carry no information about whether a
fix is right — `corr(ratio, error) = +0.052` over 275 would-be-accepted results,
with the weakest quartile holding the best median error. The gate is a threshold
on a number that does not know the answer.

### 4.4 The matching step was the thing

`xfeat-lighterglue.pt` was already in the repository, unused. It replaces the
mutual-nearest-neighbour step — which the flight path ran with no
descriptor-similarity floor at all — with a learned matcher that reasons over
both feature sets together.

Measured on 338 frames, gates applied the same way:

| | Fix rate | p50 | p90 | Worst | > 20 m | Cost |
|---|---|---|---|---|---|---|
| mutual NN (as flown) | 168/338 (50%) | 3.53 m | 6.41 m | 10.95 m | 0 | 520 ms |
| LighterGlue | 213/338 (63%) | 3.94 m | 7.79 m | 14.79 m | 0 | 2222 ms |
| **mutual NN, else LighterGlue** | **236/338 (70%)** | 3.72 m | 7.96 m | 14.79 m | **0** | 1237 ms |

The accuracy is not the gain — the **coverage** is, and it arrives exactly where
the aircraft needs it:

| Prior error | mutual NN | LighterGlue | Worst |
|---|---|---|---|
| 0–10 m | 78% | 75% | 8.9 m |
| 10–20 m | 70% | 86% | 8.3 m |
| **20–50 m** | **22%** | **94%** | 14.8 m |
| **50 m +** | **0%** | **15%** | 9.3 m |

One number decides whether LighterGlue is safe: its inlier floor. At XFeat's
value of 12 it produced six fixes over 20 m and a worst of 152.67 m, every one
with 12–19 inliers. At 20 all six disappear and the worst becomes 14.79 m. The
old floor failed because it was written for a different distribution —
LighterGlue's inlier ratio runs 0.248 against mutual NN's 0.053.

LighterGlue is **worse** below 85 m (19% accept against 46%) and better above
100 m (83% against 52%), so it is not asked below 90 m.

### 4.5 A lost frame can be found again

Widening the window does not work. The prior was moved D metres away in eight
bearings with the window widened to contain the truth anyway:

| Prior error | Recovered | False fix | Declined | Cost |
|---|---|---|---|---|
| 50 m | 2% | 0% | 98% | 649 ms |
| 100 m | **0%** | 0% | 100% | 944 ms |
| 200 m | **0%** | 0% | 100% | 1685 ms |
| 500 m | **0%** | 0% | 100% | 5576 ms |

Tiling it does. The region around the estimate was cut into overlapping tiles of
the size the matcher already works on, and every tile tried, on the lost
flight's own frames:

| Frame | Position error at the time | mutual NN | LighterGlue |
|---|---|---|---|
| f010095 | 51.4 m | found nothing | **8.28 m** (55 points) |
| f011327 | 102.1 m | found nothing | **3.64 m** (24 points) |
| f012693 | 131.4 m | found nothing | **5.24 m** (24 points) |
| f014185 (58 m altitude) | 115.2 m | found nothing | found nothing |

Three of four, to 3.6–8.3 m. The fourth was at 58 m, where LighterGlue is
measured weak. **Neither matcher ever named a wrong tile** — false fix was 0%
here and 0% in the widening test. That is what makes a recovery mode safe to
contemplate: the search goes quiet rather than inventing a position, so being
told "you are 300 m that way" and then "800 m" cannot come from the search
itself.

Cost: 1482 ms a tile, so a 7 × 7 square is about 30 s and a ring about 20 s.
This is not a routine path; the aircraft must be holding while it runs.

---

## 5. What was changed

Five changes, all software on the companion computer. **No flight-controller
parameter was modified.** All configurable, all reversible from
`vps_vio/config.py`.

| Change | Setting | Measured effect |
|---|---|---|
| Only a trusted matcher's answer may be published | `map_trust` | 109 fixes over 50 m → 0; worst 180.74 → 11.36 m |
| LighterGlue runs on the frames XFeat refuses, above 90 m | `matcher = "xfeat,xfeat_lg,ncc"` | fix rate 50% → 68%; its own fixes worst 10.11 m |
| NCC is trusted again **below 90 m**, as the last resort | `map_ncc_below_m = 90.0` | coverage at 70–82 m 48% → 94%; worst there 68.92 → 45.81 m |
| The aircraft declares when it is lost | `map_lost_gap/score/alt_m/sigma_m` | fires at 24.7 m of error, 0 false alarms |
| The attitude behind the tilt correction carries its age | `map_att_age_max_s` | correction withheld past 0.30 s, where the cost starts |
| Every match attempt records a reason, and which matcher answered | log columns | refusals were previously invisible |

A sixth was built and deliberately **not** connected: `MapFixWorker.relocate()`,
the tiled search of §4.5. Connecting it requires confirm-before-commit logic —
a recovered position must be confirmed against a second answer after known
motion before anything is reset — and that has not been built or tested.

Test suite: **68 of 68 pass.** End to end through the real worker on 190
attempts from both flights: fix rate 50% → 68%, LighterGlue produced 34 fixes
and not one of them on a frame mutual NN had also matched, their error p50
4.57 m and worst 10.11 m, their prior error p50 23.3 m.

---

## 6. What is not verified

1. **kornia** — the library LighterGlue needs — was installed today and has
   never flown. It is the largest new component. Installed without touching the
   existing PyTorch build, which was verified unchanged.
2. **The path from "lost" to ALT_HOLD is designed, not measured.** The detector
   declares 50 m of accuracy; the flight controller decides what to do with it.
   How many seconds that takes on this aircraft needs a simulator run or a
   flight.
3. **24% of fix attempts exceed the 2-second period** when LighterGlue runs. The
   worker drops a job while one is in flight, so this costs attempts rather than
   stalling the loop, but the resulting fix rate in flight has not been measured.
4. **The inlier floor of 20 was chosen from the same data it is evaluated on.**
   It needs confirming on other flights.
5. **Below 90 m the aircraft depends on NCC, whose gate does not self-assess.**
   NCC is trusted there because nothing else is available, not because it is
   accurate: measured on the two 3 November flights that actually cruised at
   70–82 m, NCC's own fixes run 11.78 m median against XFeat's 4.80 m, and it
   produced a 68.9 m fix on a frame whose estimate was under 10 m out. It is
   the last resort in the list, so XFeat is believed wherever it speaks, which
   is what takes the worst case from 68.92 m to 45.81 m. What keeps this
   survivable is downstream and unchanged: the filter rejects a residual past
   60.3 m and caps an accepted one at 0.5 m. Those flights held with NCC
   publishing 65 fixes over 20 m.
6. **The recovery of §4.5 rests on four frames.**

---

## 7. Recommended next steps, in order

1. **A short validation flight** — known area, 90–110 m, 5–8 minutes, simple
   pattern, not a waypoint mission. The question is only whether the position
   holds with the new matcher chain. Metrics: how often `fix_source` reads
   `xfeat_lg`, whether `lost` ever asserts, and the error against the M9N.
2. **The optical flow's scale stability.** This is the actual cause of the
   runaway and nothing in this report addresses it. A scale swinging 0.38–1.70
   over 20 s is what no map authority can keep up with.
3. **Confirm the recovery on 20–30 lost frames** before connecting
   `relocate()`, and build the confirm-before-commit logic with it.
4. **Run LighterGlue on its own core.** Its 1482 ms would overlap rather than
   follow mutual NN's 520 ms, removing the cost objection. Six cores are
   available and one worker thread is in use.
5. **Newer imagery.** The sheet is from 2022. The working altitude band for
   camera-to-sheet matching has been measured at 60–80 m previously, and
   everything in this report is built on top of a map that is three farming
   seasons out of date.

---

## 8. Claims made during this work and later corrected

Recorded because the measurements that overturned them are the useful part.

| Claimed | Corrected to |
|---|---|
| Bad map fixes drifted the estimate, which moved the window, which produced worse fixes | The runaway happens with no map at all. It is flow drift. |
| The 0.5 m correction cap was the limiting factor | Caps of 5 m and 20 m are identical; beyond 5 m it does not bind |
| XFeat's error is bounded at about 11 m | That is a sample, not a guarantee — a differently spaced 190-frame sample produced a 126.02 m fix. What is bounded is the *system*: the filter rejects a residual past 60.3 m and caps an accepted one at 0.5 m |
| 153 fixes were applied with zero inliers, so a gate was bypassed | No gate was bypassed. NCC does not produce inliers, which is how it emerged that NCC had published all 153 |
| The missing descriptor-similarity floor was the fault | Floors of 0.5 and 0.7 give identical results; 0.82 is worse |
| 80–90 m breaks the matcher | The 110 m flight contradicted it: 3.88 m at 100–115 m |
| Attitude should be interpolated to the frame's instant | At 10 Hz, 100 ms of staleness costs nothing. Only its age needed recording |
| Wider map windows would help | 17 of 20 frames changed by under 2 m, and the accept rate collapsed |
| NCC should be switched off at every height | The measurement behind that was 675 frames of which 497 were at 100–118 m, and I applied its conclusion everywhere. Below 90 m XFeat accepts 48% and LighterGlue is not asked at all, so removing NCC halves the coverage in the one band the project had already recorded it as necessary for. It is trusted again below 90 m, last in the list. |
| NCC at 70–85 m has a worst error of 7.6 m | That figure came from the 4 November flights passing through that band on climb and descent (56 frames). On the 3 November flights that actually cruised there, the same matcher's worst was 68.92 m. |

---

*Prepared 4 November 2026. All figures and the scripts that produced them are in
`bench/xfeat_audit/`; every table above can be regenerated from the saved
flight frames.*
