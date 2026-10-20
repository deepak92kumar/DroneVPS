# What the pre-SSD-loss reports already knew

The pilot kept fifteen status reports from 27 Sep to 13 Oct 2026. They are the
only surviving record of the work that was lost with the NVMe, and they answer
several questions this branch had been treating as open. Text copies live in
`~/DRONE_fc_logs_backup/old_status_reports/`.

**There is nothing after 13 October.** That report is the last, and its "Next"
list is the project's to-do at the moment everything was lost.

---

## 1. The map matcher's altitude band — measured twice, a fortnight apart

This is the answer to "why did the matcher find nothing on 20 Oct".

| | |
|---|---|
| 29 Sep | map matching at **80–120 m: 70% of frames**, 11.9 m |
| 29 Sep | map matching at **20–50 m: 5% of frames** |
| 30 Sep | *"Works above 80 m and not below: 349 map pixels of patch at 110 m against 95 at 30 m"* |
| 12 Oct | **17.81 m of matcher error at 95–110 m against 68.36 m at 60–70 m** |
| 12 Oct | *"Keep every mission waypoint at 100 m or above, including the one before LAND"* |
| 4 Oct | *"XFeat flying, with the window scaled to height: **724 fixes in one flight with NONE rejected**, median residual 10.2 m"* |

**The 20 Oct flights were at 70 and 72 m.** Below the floor, by a little,
twice. At 60–70 m the matcher's own error was measured at 68 m — it is not
that it nearly works down there, it is that it does not work down there.

The 349-pixel figure reproduces exactly: 1280 px (uncropped) × 110 m / 791.7 px
focal ÷ 0.509 m/px = 349. Today's frames are cropped to 1150 px for the landing
leg, so the same patch size needs about **122 m**.

**Inlier counts, on real imagery:**

| | |
|---|---|
| 5 Oct | *"inliers above 140 m (14 to 32)"* |
| 20 Oct, this branch | 9–11 at 72 m, against a gate of 12 |

Consistent: inliers grow with height, and 72 m sits just under the gate.

**But do not reach for the gate.** 7 Oct measured *"Inliers vs fix error:
correlation −0.073 — an inlier gate is closed as a route"*. The inlier count
does not predict how wrong a fix is. Lowering `min_inliers` would buy fixes of
unknown error, not good fixes. And 12 Oct: `MAP_MIN_SCORE 0.25` was introduced
because *"189 fixes were accepted on one flight including ones 85.66 m out"*.

**The map has a time of day.** 30 Sep: residual 4.61 m at 14:46 against
15.81 m at 17:35, with image contrast 43.8 falling to 32.1 and sharpness 489
to 381. The 20 Oct flights were at 13:49, 15:15 and 16:56. The CLAHE
experiment on this branch — frames clearing the gate going from 1 of 12 to 4
of 12 — is the same effect seen from the other side.

---

## 2. The origin rule was already decided, and the rebuild lost it

13 Oct, under "Actually flying on the VPS":

> *"On the VPS the origin now comes from SCR_USER1/2, not from the live
> receiver. The pilot found this himself: he typed coordinates for one site,
> drove to a second, retyped nothing, and the aircraft picked up the new
> location by itself."*

and in the summary: **"On the VPS it now wins; on GPS1 the receiver still
does."**

That is exactly the rule restored on this branch on 20 Oct. It is not a new
idea — it is the 13 Oct decision, which came back from the rebuild as
"SCR_USER always wins" and was quietly wrong for three flights, each of which
had the origin 147–157 m from the aircraft.

13 Oct also records *"Every boot now prints which receiver the switch is on
and which point the origin came from, so a stale SCR_USER is visible before
the aircraft leaves the ground"* — the preflight announcement this branch was
about to propose already existed.

---

## 3. The descent below 60 m — the one number that never moved

| | |
|---|---|
| 13 Oct | error **4.13 → 26.0 m in the twelve seconds** of a 6 m/s descent below 60 m |
| 13 Oct | came within **1.6 m** of the LAND point, touched down **15.2 m** from it |
| 12 Oct | *"Below MAP_MIN_ALT_M = 60 there is no absolute reference at all and whatever error exists is carried"* |
| 12 Oct | the step cap is 0.500 m per fix at 1 fix/s, *"so 5 s buys 2.5 m and a 29 m error would need 58 s"* |
| 13 Oct | *"It is the one number that has not moved all week"* |

So the planned GPS → VPS → GPS flight should expect the descent to be the
worst phase, and the hand-back to GPS should happen **before** it, not during.

---

## 4. Mechanical: one missing arm screw was two faults

| | |
|---|---|
| 13 Oct | motor yaw differential **−95 → −12.5** with one arm screw replaced |
| 13 Oct | yaw authority held continuously **7–13% → ~1%** |
| 13 Oct | FC vibration above 80 m **9.62 → 5.13**; camera **6.55 → 3.74**, VibeY 5.25 → 0.99 |
| 13 Oct | *"Heading error was never the symptom — it stayed 1.65–2.47 deg throughout — the cost was headroom"* |

**The diagnostic that found it was the pilot's**, and it is reusable: both
blades on motors 3 and 4 are CW, so swapping them is a valid test. *"If the
torque followed the blade it was the propeller; if it stayed with the corner
it was the motor or the arm. It stayed."* One flight settled what the logs
could not — and the log has no ESC telemetry, so motor RPM cannot be read at
all.

This matters on 20 Oct, where `hover_balance.py` put **yaw at 50–57% of the
imbalance** across three flights and the mechanical work halved it without
removing it.

And 10 Oct, on why it keeps coming back:

> *"Motor spread went 99.7 to 117.9 µs and the roll rate error 7.05 to 10.33
> deg/s between two flights of the same day, with nothing touched. That is the
> second time in three days that something has loosened inside a day."*

---

## 5. The velocity fault was already the largest open item

13 Oct: **velocity to the flight controller, 54% — "still the largest open
fault"**, and the lowest-scoring rows gating the mission were
`3-5 km 22%, velocity 54%, heading/yaw 55%, map coverage 64%`.

The old AccelBridge's failure is worth re-reading because `velocity_bridge.py`
was written against it from these reports rather than from its code:

> 10 Oct: *"89% on 9 October, 75% then 63% on 10 October, nothing changed in
> it. The flow into it reads 98–108% of truth, so the loss is inside."*

The 20 Oct finding is a different fault in the same channel — not attenuation
but amplification, the flow's constant pixel error times altitude — and the
two defences added (an honest altitude-scaled accuracy, and a physical
acceleration gate) do not address the old one, which the no-leak design
already does.

---

## 6. What the reports say cannot be settled from the aircraft

Asked for on **three separate days** (10, 12 and 13 Oct) and never done:

> *"A surveyed ground point. Every error figure in this report is against
> GPS1, and GPS1 itself wandered 2.4 to 27.4 m this morning while the aircraft
> stood still. Whether the sheet or the receiver is wrong about the ground
> cannot be settled from the aircraft. It needs no flight."*

This bears directly on 20 Oct's map work. When the camera frame was put beside
`bhh.tif` at "the real GPS position" and the two were judged to be the same
place, the reference for "real" was GPS1 — which may itself be metres out. The
matcher's documented *"own error of 11–12 m"* may be partly the receiver's.

---

## 7. Two things to stop assuming

**Companion status lines may not reach the pilot.** 13 Oct, MAVLink routing
94%: *"Companion status lines still do not reach the radio link."* They do
reach the FC and land in the dataflash as `MSG`, which is where this branch
has been reading them — but "the pilot will see it on the radio" is
unverified. Everything on this branch that depends on a radio announcement
reaching the handset (`LIDAR ZERO`, `FRAMES ON/OFF`, the origin line) needs
that checked before it is relied on.

**The target altitude was always 100–150 m.** The 13 Oct header: *"Target:
100–150 m AGL, 3–5 km out-and-back"*, and the 1077 m autonomous mission was
flown at 110 m. The map matcher's band and the mission profile were designed
together. Flying the VPS at 20–70 m is outside what any of this was built for.

---

## 8. Numbers worth keeping as a baseline

| | | |
|---|---|---|
| best position error above 60 m | 5.57 m mission, 4.41 m Loiter | 13 Oct |
| longest autonomous mission | 1077 m path, 383 m from home, at 110 m | 13 Oct |
| real distance from each waypoint | 2.3, 9.1, 6.7, 8.6 m | FC believed 0.3 m |
| map offset after the bias was removed | N −1.69 / E −0.22 m | 13 Oct |
| matcher's own error | 11–12 m | unchanged all week |
| automated tests | 222 passing | 13 Oct |
| airframe | 3.9 kg **quad** | the hexa is new since |
