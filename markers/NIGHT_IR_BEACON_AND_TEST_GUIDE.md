# Night IR beacon + ground/flight test guide

Drone: OAK-D-PRO-AF (IMX378 12MP RGB + 2x OV9282 global-shutter mono, IR-sensitive),
CUAV X7 Pro ArduCopter 4.7.1. Landing pad: `nested_pad_10ft_ID45_ID0.svg`.

> Status: the daemon today detects only the small ArUco ID 0 on the 640x400 mono camera.
> Outer marker ID 45, higher camera resolution and IR-beacon detection are NOT in the
> code yet (future changes H/I). This guide covers hardware setup and the tests that
> can be run now.

---

## Part A: Tests you can run now

### A1. Gyro tilt-validation hand test (props OFF)

Purpose: the daemon turns gyro derotation on only after gyro and rotation-vector tilt
agree on BOTH axes (>= 15 moving 0.4 s windows, |corr| >= 0.85). This must be proven
with real motion.

1. Remove all propellers. Power FC + Jetson. Wait for RC to show `EKF3 ... using external nav data`.
2. Hold the drone level about 1 m above textured ground (grass/soil, no shadow of people).
   Do NOT arm.
3. Tilt nose up / nose down about +/-10-15 deg, 1-2 tilts per second, for 20 s.
4. Tilt left / right the same way for 20 s.
5. Mixed circular wobble for 20 s. Keep the drone over the same spot.
6. Expected on RC: `>>> TILT: GYRO DEROT ON`.
   Daemon log: `[+] Gyro tilt derotation ON: roll slope ... corr ... | pitch slope ... corr ...`.
7. Pass criteria: |corr| >= 0.85 on both axes, |slope| between 0.7 and 2.0.
8. If it does not turn on in 60 s: note the clock time and send it; the debug log keeps
   `gdr/gdp` (gyro) and `droll/dpitch` (rotation vector) for every 10th frame.

Note: validation restarts every daemon start. If you skip this test, the daemon keeps the
older rotation-vector derotation (the one that flew stably), so it is safe either way.

### A2. Precision Loiter over the small 24.2 cm pad (ID 0)

Usable height with the small pad: about 1-3 m (reliable detection ~3 m).

Setup:
1. Lay the pad flat on open ground. Tape all edges / weigh the corners: propwash lifts paper.
2. No shadows or people near the pad. Keep the rope away.

Flight:
1. Arm in ALT_HOLD next to the pad, climb to 2-2.5 m above it.
2. Wait for `>>> VIO TRUST ON | LOITER OK`, switch to LOITER, sticks centred.
3. When the pad is seen 8 times in 1 s the RC shows `>>> PREC LOITER ON | PAD #0`
   and the drone should move over the pad centre and hold.
4. Hold 20-30 s. Then land in ALT_HOLD.
5. Abort to ALT_HOLD immediately if the drone moves away from the pad, oscillates
   around it with growing swings, or you see `EKF variance` / `EKF Failsafe`.
6. Stick input > about 0.5 m/s^2 of demand makes ArduPilot fall back to normal LOITER.
   Leaving LOITER, landing, or losing the pad for 3 s sends `>>> PREC LOITER OFF`.
7. Send afterwards: SD card BIN (PL messages show target acquired / position) + daemon log.

---

## Part B: Nested 10 ft pad construction

Files:
- `nested_pad_10ft_ID45_ID0.svg`: true scale in millimetres (open in Inkscape / send to flex printer).
- `nested_pad_10ft_fabrication_map.md`: every black/white cell with mm coordinates (for painting).
- `nested_pad_10ft_layout_preview.png`: dimensions + IR LED positions.

Sizes:
| Part | Size |
|---|---|
| Whole white sheet | 4.064 x 4.064 m (13.33 ft) |
| White quiet zone around outer marker | 508 mm on every side (do not skip) |
| Outer marker ID 45 black edge | 3.048 m (10 ft), cell 508 mm |
| Inner marker ID 0 black edge | 450 mm, cell 75 mm, exactly centred |

Material and build rules:
1. Matte, non-glossy: matte flex banner or white tarpaulin with matte black paint.
   Gloss reflects the sun and turns black cells white at some angles.
2. Pure black and white only; no logos, text, grid lines or crosshairs anywhere on the sheet
   (the quiet zone must stay plain white).
3. Keep every cell square within +/-1 % (outer +/-5 mm, inner +/-1 mm). Paint with a
   chalk-line grid, then remove the chalk.
4. Mark which edge is TOP (row 1) on the back side only.
5. Stake every corner and every ~1 m along edges (eyelets + ground pegs). A 4 m sheet
   lifts easily in propwash and wind.
6. Keep it flat: wrinkles bend the cell edges and break decoding at long range.

Measured in simulation (flat pad; real range will be lower, field-test it):
| Camera | Outer ID 45 detected | Inner ID 0 detected |
|---|---|---|
| mono 640x400 (current) | ~5-80/100 m | 0.8-10 m |
| mono 1280x800 | ~5-120 m | 0.8-20 m |
| RGB 4K | ~20-120 m | up to ~40 m |

Stress results (same simulation): sun glare, 15-20 deg tilt and 3 px motion blur cost only
some range (mono 800 still ~100-120 m). **A dark shadow lying across the pad broke detection
at almost every height.** At 100 m the drone's own shadow is blurred away by the sun's size,
but during the final descent it can cover the pad. Plan approaches so the sun is not directly
behind the drone over the pad (early morning / late afternoon: shadow falls to the side), and
keep people and poles away from the pad.

---

## Part C: Night IR LED beacon

Why: printed markers are invisible at night; glow stickers are far too dim for a
1.5 ms camera exposure. The OAK-D Pro mono cameras have no IR-cut filter and see
850/940 nm light. (The IMX378 RGB camera normally has an IR-cut filter: do not rely on
it for the beacon.)

### C1. Parts (per pad)
| Item | Qty | Spec |
|---|---|---|
| IR LED emitter | 5 | 940 nm (invisible) or 850 nm (faint red glow, ~2x more signal on the OV9282). 3 W high-power emitter on star board, or a 12 V CCTV IR illuminator. Wide beam 90-120 deg. |
| Constant-current driver | 5 | 700 mA (for 3 W emitters). **DC output, no PWM dimming** (PWM makes the LED dark in some 1.5 ms exposures). |
| Heatsink + IP65 box | 5 | Clear polycarbonate or glass window, LED pointing straight UP. |
| Battery | 1 | 12 V LiFePO4 20-30 Ah (5 x 3 W x 8 h ~ 120 Wh). |
| Cable | - | Outdoor 2-core, buried or taped flat; fuse at the battery. |
| Tilt/level | - | Small bubble level for each LED housing. |

CCTV illuminators with a built-in photocell switch on automatically at dusk.

### C2. Layout (pattern must be asymmetric so the drone can tell the heading)
Place LEDs at the white sheet corners (see `nested_pad_10ft_layout_preview.png`):
- L1 top-left corner, L2 top-right, L3 bottom-right, L4 bottom-left (4.064 m square).
- L5 on the TOP edge, 3/4 of the way from left to right (3.048 m from the left corner).
- LEDs sit on the ground just outside/at the sheet corners, never on black/white cells.
- Spacing check: at 120 m one mono-800 pixel covers ~15 cm, so 4 m spacing = ~27 px;
  the closest pair (L5-L2, 1.0 m) = ~7 px, still separable.

### C3. Installation steps
1. Lay and stake the pad first (Part B).
2. Put each LED housing at its position, level it (beam straight up), stake it.
3. Run cables along the sheet edge, not across it; bury or tape flat.
4. Connect drivers, fuse and battery. Check every LED with a phone camera (most phone
   front cameras show 850/940 nm as purple/white).
5. Eye safety: never look into a lit high-power IR LED from close range; it is invisible
   but can still harm the eye. Switch on only when needed.

### C4. Night checks on the ground (no flight)
1. Hold the drone (props off) 2-3 m above the pad at night; record flight images.
   The five LEDs should appear as bright round blobs in the mono frames.
2. The small inner marker at low height is lit only by the drone's own IR floodlight
   (currently 700 mA). Check the mono images at 1-4 m: the ID 0 marker must be readable.
   If it is too dark, add a low ground IR illuminator aimed at the pad from the side.
3. Send the images/time so the exposure/gain settings can be set for beacon detection.

### C5. What is still needed in software before night/long-range flights
- H: detect outer ID 45 with its own size (3.048 m) and share the landing target with ID 0.
- Camera mode change (mono 1280x800 or RGB 4K ROI) for >80 m.
- I: IR beacon blob detection (fixed low exposure/gain, 5-point pattern match) -> LANDING_TARGET.
- PLND_ALT_MIN = 0 and PLND_ALT_MAX = 0 on the FC before flying precision loiter above 8 m.
- Before any 100-120 m flight: TF03 range over grass at night/day, VIO behaviour at that
  height, and local airspace rules.
