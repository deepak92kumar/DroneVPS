"""All VPS/VIO tunables in one place. NOT active until Phase 1 (after demo).

Defaults mirror the proven marker-map behaviour from the live daemon. The two
correction constants MUST stay flat (never scale with height): scaling them cost
two flights and was reverted (see memory flow-velocity-above-40m / d75a29f).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class VpsConfig:
    # --- FC link (reuses the existing bridge; same values the daemon uses) ---
    fc_port: str = "/dev/ttyTHS1"
    # 921600, NOT 115200. Read from the FC itself in log 00000195.BIN:
    #   SERIAL2_BAUD 921   SERIAL2_PROTOCOL 2 (MAVLink2)
    # TELEM2 is where the companion sits, and flight_daemon.py's own comment
    # records why it moved: at 115200 "the link was near full and the FC
    # forwarded it to the RC link", which made parameter download crawl.
    fc_baud: int = 921600

    # --- VIO -> BODY axis mapping (MEASURED 27 Sep, no longer a placeholder) ---
    # Derived from geometry, not from drift-prone motion tests:
    #   * cam_z is DOWN  - the accelerometer's gravity vector, rotated into the
    #     camera frame by the IMU->CAM_B extrinsics, lands on cam_z at 99.8%.
    #   * the drone's nose points at the TOP of the image, and cam_y is image
    #     DOWN, so body forward = -vio_y.
    #   * body right = down x forward = +vio_x (right-handed, no measurement needed).
    # Cross-checked against the one clean motion run: a forward move gave
    # vio_y = -0.225 with 100% on that axis, exactly as this matrix predicts.
    # Rows map a VIO vector to [forward, right, down] in the body frame; the
    # compass supplies body->NED heading at runtime (VioToNed.heading_align_rad).
    #
    # ALSO DEAD, AS OF 22 OCT: nothing in the flight path reads this. It was
    # the cuVSLAM VIO's axis mapping, and cuVSLAM was rejected on 28 Sep - the
    # OAK-D's 7.5 cm baseline gives under 0.3 px of disparity above ~25 m
    # against a 100-150 m mission, and it contributed exactly 0 over 1680
    # frames. vio_source.py and oak_rig.py went with it. The flow path's own
    # axis mapping lives in flow_odom.shift_to_ne, and the founding brief's
    # note that this matrix was "an identity placeholder" is the same class of
    # fault that caused the 21 Oct runaway: an unmeasured sign.
    R_vio_to_ned: list = field(default_factory=lambda: [[0.0, -1.0, 0.0],
                                                        [1.0,  0.0, 0.0],
                                                        [0.0,  0.0, 1.0]])

    # --- ArUco-as-anchor fusion (mirror of marker_fix_step; keep FLAT) ---
    #
    # DEAD, AS OF 22 OCT: nothing reads either of these. They belong to the
    # AnchorFuser in vps_vio/anchor.py, which went with the SSD along with the
    # cuVSLAM path; MapFuser replaced it and uses max_correction_m and
    # gate_sigmas instead. They are left here, labelled, because the project's
    # founding brief singles them out - "keep corr_rate_mps and reanchor_m
    # FLAT - never scale with height. (Scaling them previously cost two
    # flights and was reverted.)" - so a reader who finds them will reasonably
    # think they fly. Tuning them changes nothing.
    #
    # The warning itself has NOT expired, and it is the nearest thing this
    # project has to a rule about height-scaled gains.
    corr_rate_mps: float = 4.0     # max slew speed of the offset correction, m/s. FLAT.
    reanchor_m: float = 6.0        # horizontal residual above which we snap. FLAT.
    marker_timeout_s: float = 2.0  # ignore marker fixes older than this

    # --- covariance sent to EKF (quality-dependent, like the daemon) ---
    pos_cov_good: float = 0.05     # m^2 when VIO tracking is healthy
    pos_cov_poor: float = 0.50     # m^2 when tracking is weak

    # --- VIO health gating ---
    min_vio_rate_hz: float = 15.0  # below this we stop trusting X/Y
    send_rate_hz: float = 20.0     # how often we push VISION_POSITION_ESTIMATE

    # ---------------------------------------------------------------------
    # Values recovered from the 28 Sep - 13 Oct status reports after the SSD
    # that held the live code was lost. Each one carries the date it was
    # changed and whether it had flown, because that is the only provenance
    # left for them - the commits themselves are gone.
    # ---------------------------------------------------------------------

    # --- map matcher ---
    # 4 NOV: "xfeat,xfeat_lg". Mutual NN first, LighterGlue only on the
    # frames it refuses, and only above 90 m where LighterGlue is measured
    # better. This is what the pilot asked for in words - "agar xfeat ko
    # points na mile tho LG uski help kare" - and it measures, end to end
    # through the real worker on 190 attempts: fix rate 50% -> 68%,
    # LighterGlue's own 34 fixes at error p50 4.57 and worst 10.11 m, their
    # prior error p50 23.3 m, and NOT ONE of them on a frame mutual NN had
    # also matched.
    #
    # WHAT THIS DROPS: NCC no longer runs, so fix_disagree_m is empty. That
    # costs nothing measured - NCC was never trusted after the 4 Nov finding,
    # so its only remaining job was that number - but it does remove the
    # shadow insurance of flight 160, where XFeat managed 2 fixes of 40 at
    # 15:20 and NCC managed 26. On a day like that, set this back to "both".
    # "ncc,xfeat,xfeat_lg" runs all three and needs map_trust set explicitly.
    matcher: str = "xfeat,xfeat_lg,ncc"
    # NCC IS TRUSTED AGAIN BELOW THIS HEIGHT, and it took the pilot pointing
    # at the 3 Nov flights to find it. The measurement that removed NCC was
    # 675 frames of which 497 were at 100-118 m, and I applied its conclusion
    # at every height. Below 90 m the aircraft has nothing else: XFeat
    # accepts 48% there and LighterGlue is not asked below 90 m at all.
    #
    # Measured on the two 3 Nov flights that actually flew at 70-82 m, 197
    # attempts, against the M9N:
    #
    #                      coverage   err p50   WORST   >20 m
    #     NCC alone        172 (87%)   11.78    68.92     65
    #     XFeat alone       94 (48%)    4.80    21.53      1
    #     XFeat, else NCC  186 (94%)    6.64    45.81     47
    #
    # NCC is NOT more accurate down there - 11.78 m against 4.80 - and it
    # produced a 68.9 m fix on a frame whose prior was under 10 m, because
    # its gate does not self-assess at any height. What it is, is AVAILABLE.
    # Trusting XFeat first and falling back to NCC keeps the whole 94% and
    # takes the worst case from 68.92 to 45.81 m.
    #
    # Those flights held with NCC publishing 65 fixes over 20 m, and what
    # held them is downstream and unchanged: map_fuse rejects a residual past
    # 3 sigma (60.3 m at nominal) and caps an accepted one at 0.5 m.
    #
    # 80, NOT 90: the pilot flies this aircraft and has watched NCC across
    # the band - "ncc 80mtr tak hi aacha hai, best 70 mtr par". The replay
    # agrees with the shape of that: NCC's own error at 70-85 m is 1.95 m p50
    # and at 85-100 m it is 6.19 m, so it degrades through exactly there.
    #
    # XFeat still goes FIRST below 80 m even though NCC covers more, because
    # where XFeat speaks it is more accurate - 4.80 m against 11.78 m on the
    # 3 Nov flights - and putting NCC first would take the worst case from
    # 45.81 m back to 68.92 m. NCC catches what XFeat drops; it does not lead.
    #
    # 0.0 disables the fallback. UNFLOWN.
    map_ncc_below_m: float = 80.0
    # HOW FAST THE DECLARED ACCURACY IS ALLOWED TO GROW once the aircraft
    # knows it is lost, in metres per second. The first version of this
    # stepped straight to map_lost_sigma_m, and the pilot stopped it: a step
    # to 50 m hands EKF3 a position it must reject, which is a glitch, which
    # is ALT_HOLD - and ALT_HOLD is not a solution, it is the aircraft
    # drifting on the wind with nobody flying it.
    #
    # So it ramps. 0.6 m/s is not a guess: it is the rate the position
    # ACTUALLY diverged on flight 478, measured at 0.546 m/s from t=400 and
    # 0.625 m/s from t=500. Declaring uncertainty that grows at the rate the
    # error is really accumulating is the honest number, and it means the
    # flight controller de-weights us progressively instead of being handed a
    # reason to stop. From a typical 8.5 m to 50 m takes about 69 s at this
    # rate.
    #
    # It is a de-weighting, not a veto: a larger horiz_accuracy makes EKF3
    # lean on our position less, it cannot make the aircraft jump. 0.0 goes
    # back to the step. UNFLOWN.
    map_lost_ramp_mps: float = 0.6
    # HOW FAR ONE MAP FIX MAY MOVE THE ESTIMATE. 0.5 -> 1.0 on 5 Nov, after
    # the first flight of the new matcher chain, and the pilot's call: "1
    # karo, agar faida hua tho aur badhayenge."
    #
    # The 5 Nov flight held - no runaway, error p50 3.41 m against the M9N,
    # the lost detector never fired - but the aircraft wandered about 10 m
    # back and forth, 22.85 m peak to peak at 96 m with periods of 23-92 s.
    # Measured, that is the cap:
    #
    #     fix interval                       2.03 s
    #     flow moves the estimate            2.572 m between fixes
    #     the fix pulls back                 0.383 m   - seven times weaker
    #     at 90-100 m the correction's
    #       median was 0.489 m against the 0.5 cap - saturated
    #     the cap bound 35% of all corrections
    #
    # EK3_SRC1_POSXY is 3 and VELXY is 0, so the flight controller navigates
    # on OUR POSITION. The aircraft holds where we say it is, so our
    # estimate's wander becomes real aircraft motion, and the position
    # controller commanded up to 14.9 degrees of pitch chasing it. The
    # attitude loop is not at fault: velocity target against achieved ran
    # 0.09-0.48 m/s RMS.
    #
    # WHY THIS IS SAFE NOW AND WAS NOT BEFORE. A bigger cap applies to a bad
    # fix as readily as a good one, and until 4 Nov NCC was publishing fixes
    # up to 180.74 m wrong - which is why 0.7 was tried earlier and reverted.
    # On 5 Nov all 144 published fixes were within 7.0 m of the M9N:
    #
    #     xfeat     n=118   p50 1.89   p90 4.90   worst 6.40 m
    #     ncc       n= 20   p50 2.13   p90 3.21   worst 5.78 m
    #     xfeat_lg  n=  6   p50 4.58   p90 6.77   worst 7.00 m
    #     none over 20 m
    #
    # At 1.0 m the map's authority goes from 0.19 to about 0.49 m/s, against
    # a flow that moves the estimate at roughly 1.0-1.5 m/s. That is not
    # enough to win, which is the point of going to 1.0 rather than 2.0: it
    # is one measurable step, and map_fuse's 3-sigma gate at about 60 m still
    # rejects anything wild. UNFLOWN.
    map_max_correction_m: float = 1.0
    # OVER HOW LONG A FIX'S CORRECTION IS APPLIED, seconds. 0 is the old
    # one-sample step. Set to the fix period, so a correction finishes
    # arriving just as the next fix is computed.
    #
    # The 5 Nov flights put this here. A fix moved the estimate up to
    # max_correction_m in ONE sample, and the flight controller navigates on
    # that position (EK3_SRC1_POSXY = 3), so it saw a metre of error appear
    # instantly: DesPitch p90 9.52 deg, max 14.75, against ATC_ANGLE_MAX of
    # 15. The pilot felt it as a jerk, "lagta hai jaise drone ko palat dega",
    # and 14% of corrections were hitting the 1.0 m cap. Raising the cap
    # makes a step worse; ramping makes the cap safe to raise.
    #
    # It is an exponential approach, so one fix delivers 63% of its metres
    # inside one ramp and about 95% in three - 0.636 m of a 1.0 m correction
    # in 2 s, measured. So the authority is 0.63 x cap / fix_period, which at
    # cap 1.0 and period 2 s is 0.31 m/s rather than 0.50. Against a flow
    # that moves the estimate at roughly 1.3 m/s that is still short, and the
    # gain change below is what makes up the difference.
    # UNFLOWN.
    map_correction_ramp_s: float = 2.0
    # WHAT THE FILTER IS TOLD A MAP FIX IS WORTH, in metres. 20 -> 5 on
    # 5 Nov, because 20 was simply not true.
    #
    # It sets two things at once:
    #     gain = var / (var + fix_sigma^2)       how much a fix moves us
    #     gate = 3 x sqrt(var + fix_sigma^2)     how wrong a fix may be
    #
    # At 20 the gain measured 0.061 on the 16:25 flight - a 6 m residual
    # became a 0.37 m correction, which matches the 0.392 m median measured
    # in flight - so the cap at 1.0 was binding only 14% of the time and the
    # GAIN was the limiter, not the cap. Meanwhile the published fixes
    # themselves measured 1.89 m p50 and 6.40 m worst (xfeat, 5 Nov 08:56),
    # 2.68 m p50 and 10.42 m worst (5 Nov 10:57). The filter was being told
    # the map is three to ten times worse than it is.
    #
    # At 5 the gain becomes about 0.51 and the gate narrows from 60 m to
    # 17 m. Both directions are right: the map corrects properly AND a wild
    # fix is rejected sooner. 5 and not 3, because the worst measured fix was
    # 10.42 m and a gate of 3 sigma on 3 m would start throwing away honest
    # ones.
    #
    # This is only safe WITH map_correction_ramp_s: a higher gain means
    # bigger corrections, and as a step those are the jerk the pilot felt.
    # UNFLOWN.
    map_fix_sigma_m: float = 5.0
    # The previous setting and its reasoning, kept because it is still the
    # answer on a day when the shadows have moved:
    # matcher = "both"             # 25 OCT: BOTH. NCC came back, and the reason is in
                                   # map_match_ncc's docstring. XFeat accepted
                                   # 78% of frames at 11:38 and 0 of 40 at
                                   # 15:20 on the same site with sharper
                                   # imagery - the shadows had swung 100 deg
                                   # from the map's. Replayed on both flights
                                   # at the radius NCC needs (150 m):
                                   #            XFeat          NCC
                                   #   11:38    78%, 2.13 m    97%, 1.55 m
                                   #   15:20     0%,     -     70%, 5.01 m
                                   # and 0.106 s a fix against 0.345. The
                                   # system error is per-fix/sqrt(N), so the
                                   # accept rate is what it buys.
                                   #
                                   # "both" because neither covers the day.
                                   # Fixes produced on 45 frames from each:
                                   #            NCC  XFeat  either
                                   #   11:38     45    34     45
                                   #   15:20     26     2     26
                                   # And when both fire they check each
                                   # other: corr(their disagreement, their
                                   # error) = +0.928, against -0.10 for
                                   # score and +0.20 for inlier count. That
                                   # disagreement is LOGGED, not enforced -
                                   # a veto on it is a new gate. UNFLOWN.
                                   # Map matching went 45% -> 75% on the swap.
    # WHOSE ANSWER MAY BECOME A FIX, when `matcher` runs more than one. Both
    # still run - the disagreement above needs two answers - but only a
    # trusted matcher's position is published. Comma-separated, in order of
    # preference; None means "xfeat" when both are running.
    #
    # THIS IS THE 4 NOV FINDING. The worker kept the first result unless a
    # later one was ok where the earlier was not, and the pair reads
    # ["ncc", "xfeat"], so NCC won every frame it accepted. On flight 478
    # that was all 153 fixes; XFeat succeeded 51 times and published nothing.
    # Measured on 675 frames of two flights, both matchers, nadir corrected,
    # against the M9N:
    #                       n    NCC p50/p90/max      XFeat p50/p90/max
    #   both accepted     339    3.80  8.09  85.69     3.55  6.49  11.36
    #   only NCC          226   25.01 146.55 180.74   (XFeat declined)
    #   only XFeat          9   (NCC declined)         5.18  7.37   9.00
    #   neither           101   both about 104 m
    #
    # XFeat's error is bounded at 11.36 m in every prior-error band and every
    # tilt band, including prior 50-200 m where it accepts 1% of frames. NCC
    # accepts 94-100% whatever the prior, and 55% of the prior 50-200 m
    # frames with a median error of 96.88 m. One gate is a self-assessment;
    # the other, at min_score 0.05 and min_margin 0.01, is not.
    #
    #                    fix rate   p50    p90      max    >20 m  >50 m
    #   NCC wins (old)   574/675    4.91  89.83  180.74     132    109
    #   XFeat trusted    348/675    3.57   6.59   11.36       0      0
    #
    # A third of the fixes go and every fix over 20 m goes with them; the
    # lost ones had a median error of 25 m. Set this to "ncc,xfeat" to get
    # the old behaviour back, or to "ncc" on a day like 15:20 above when the
    # shadows have moved and XFeat produces nothing - on such a day the
    # aircraft otherwise gets no map fixes and ALT_HOLD is the answer, which
    # is the standing rule for the VPS stopping. UNFLOWN.
    # EXPLICIT, not None. With matcher="xfeat,xfeat_lg,ncc" a None trust
    # would trust all three at every height, which is the behaviour this
    # whole day's work removed. NCC comes back only below
    # map_ncc_below_m, and only as the last resort there.
    map_trust: str | None = "xfeat,xfeat_lg"
    # How old the roll and pitch behind the nadir correction may be before
    # the correction is withheld. The matcher says "the middle of this frame
    # is here" and the aircraft is alt*tan(tilt) from that, so the shift is
    # only as good as the attitude it is built from - and nothing recorded
    # that attitude's age until 4 Nov.
    #
    # Measured on 451 frames of two flights, accepted fixes, error
    # p50/p90/max against how stale the attitude was:
    #     0 ms  3.48/ 6.18/11.26     100 ms  3.48/ 7.68/11.60
    #   200 ms  4.65/ 9.97/18.55     400 ms  6.32/15.13/29.41
    # so 100 ms is free and 400 ms doubles it. ATTITUDE arrives at 10 Hz, so
    # the normal case sits inside the free band; 0.30 s is set where the cost
    # starts and exists for the stream STALLING, which this project has
    # already done once when a second MAVLink client changed the shared
    # channel's stream rates. UNFLOWN.
    map_att_age_max_s: float = 0.30
    # THE LOST DETECTOR. Until 4 Nov nothing on the aircraft could say it had
    # stopped knowing where it was: flight 478 declared 8.5 m of accuracy
    # while 124.7 m wrong, and over the 4582 samples where the error passed
    # 50 m the declared sigma never reached 9.60 m. The fuser's sigma is
    # honest about DRIFT and blind to being lost, because every accepted fix
    # pulls it down and the fixes were being accepted for agreeing with a
    # drifting prior - so the aircraft grew more confident as it grew more
    # lost.
    #
    # The rule: this many refusals in a row, each with a score far BELOW the
    # gate rather than near it ("nothing in this window", not "nearly had
    # it"), above this height. Scored on the two 4 Nov flights - how much
    # error it lets through on 478, which ran 25 -> 124 m, against how often
    # it fires on 476, which never got lost:
    #
    #     gap>=5 & score<0.05 & alt>75    fires at 24.7 m    0 of 299 false
    #     secs >= 40                      fires at 69.2 m    0
    #     gap>=5 & alt>75                 fires at 24.7 m    1
    #     rate10 <= 0.3                   fires at  4.0 m   35
    #
    # The height floor is not caution, it is necessary: the 60 m matcher
    # floor depresses the accept rate on its own, which is exactly what makes
    # a rule on the accept rate alone false-alarm.
    #
    # What it does is declare map_lost_sigma_m as horiz_accuracy, which is
    # the one number the flight controller weighs our position by. It
    # commands no mode. The pilot's instruction stands: the aircraft holds in
    # ALT_HOLD rather than landing, because a blind descent has no idea what
    # is under it. Set map_lost_gap to 0 to switch the detector off. UNFLOWN.
    map_lost_gap: int = 5
    map_lost_score: float = 0.05
    map_lost_alt_m: float = 75.0
    map_lost_sigma_m: float = 50.0
    # 60 -> 50 ON 26 OCT, at the pilot's instruction and on their reasoning:
    # the gate reads the BAROMETER, which runs 5-9 m high, so a gate of 50
    # stops the map at about 44 m of real height. Below that the flow, the
    # VIO and the stereo depth carry it.
    #
    # It is NCC that makes this possible, and I had said the opposite earlier
    # the same day. Tested on 156 frames from the two 25 Oct flights, each
    # matcher with the window and gate the worker gives it:
    #
    #     matcher   30-45 m   45-60 m   60-80 m
    #     NCC          43%       53%       73%
    #     XFeat         0%        0%        9%
    #
    # XFeat dies down there because the query is rescaled to the map's
    # 0.509 m/px and runs out of pixels - 56x56 at 30 m, 9 raw matches
    # against a gate of 12. NCC slides a template in a much larger window and
    # does not care.
    #
    # THE COST, so nobody is surprised by it: NCC's fixes below 60 m are
    # 13.07-13.39 m p50 with a p90 past 50 m, against the 2.89 m the pair
    # manage at 60-80 m. The system settles near the map's own accuracy, so
    # expect roughly 13 m down there. The case for it is the alternative -
    # this file's own note below records the flow going 4.13 -> 26.0 m in the
    # 12 s of a 6 m/s descent. 13 beats 26. UNFLOWN.
    # 50 -> 60 ON 5 NOV, at the pilot's instruction, the same day the 50 was
    # found never to have reached the worker. MapFixWorker's own default was
    # 60 and won, so 60 is what every flight has actually used; writing 60
    # here keeps the aircraft exactly where it has been while the setting is
    # finally live. There is also no measurement below 58 m - the replay
    # frames start there - so 50 would put NCC on ground this project has
    # never scored a matcher against. "60 hi rakho, 50 baad mein naapenge."
    map_min_alt_m: float = 60.0    # 3 Oct: 80 -> 60 m. Flown, kept. At 80 m the
                                   # map never started. The map runs ABOVE this
                                   # height and NOT below it, which is why the
                                   # descent is uncorrected: error went 4.13 ->
                                   # 26.0 m in the 12 s of a 6 m/s descent.
    map_window_alt_frac: float = 0.60   # 4 Oct: a fixed 100 m -> 0.60 x
                                   # altitude, because the matcher was guessing
                                   # outside what the camera could see.
                                   #
                                   # 23 Oct: 0.90 and 0.805 were both tried and
                                   # both are WORSE. Widening the window is not
                                   # what the frame needed; the QUERY was the
                                   # thing being thrown away. Measured on 40
                                   # frames of vps_00084 at 85 m, scored against
                                   # that run's own receiver with a PER-ROW
                                   # origin:
                                   #
                                   #   full + 0.60    p50 10.28  p90 23.86
                                   #   full + 0.805   p50 11.23  p90 38.98
                                   #
                                   # Until 23 Oct this constant was DEAD - it
                                   # was defined here and nothing read it,
                                   # because VpsNode built MapFixWorker without
                                   # passing it and the worker's own default
                                   # won. It is wired now, so changing it here
                                   # changes what flies.
    map_query_shape: str = "full"  # What of the frame reaches the matcher.
                                   # "square" is a centred square of 0.95 x the
                                   # SHORT side - 760x760 on a 1280x800 frame,
                                   # so 44% of the frame never arrived. "full"
                                   # keeps the whole rotated frame.
                                   #
                                   # The two sizes are one decision. At 60-83 m
                                   # the camera sees 117 x 81 m; a square query
                                   # is 77 x 77 m, and in a 0.60 window (97 m
                                   # across) it can slide only +-9.7 m while our
                                   # own prior is 10.6 m out at p50 and 20.5 at
                                   # p90. The truth was outside the searchable
                                   # range on 52.5% of attempts - geometry
                                   # refusing the fix before any matching.
                                   #
                                   # 301 frames of vps_00076, scored against
                                   # that run's own GPS:
                                   #   square 0.60   85/301 (28.2%)  14.17 m
                                   #   square 0.90   63/301 (20.9%)  14.70 m
                                   #   full   0.60  100/301 (33.2%)  12.81 m
                                   #   full   0.90  147/301 (48.8%)  13.59 m
                                   # Nearly double the fix RATE at the same
                                   # per-fix error, and the rate is what sets
                                   # the system error: 0 fixes gave 38-68 m,
                                   # 19 gave 4.50 m.
                                   #
                                   # The square rows are why "bigger window" sat
                                   # on the do-not-retry list. With the query
                                   # held at 77 m more map is only more chance
                                   # to match wrong. The window was never the
                                   # mistake on its own; the RATIO was.
    # 1.0 AT THE PILOT'S INSTRUCTION, 23 Oct evening, to be flown tomorrow.
    # They were told the measurement disagrees and said do it anyway: "error
    # badhe tab bhi karo".
    #
    # WHAT THE MEASUREMENT SAID. Swept on flight 232's own saved frames and
    # scored against GPS_RAW_INT - the receiver before the EKF, the only truth
    # here that is independent of what we inject - VPS rows only:
    #
    #     2.0 s   all VPS 1.35 m   above 100 m  2.76 m
    #     1.0 s   all VPS 5.33 m   above 100 m 11.16 m
    #
    # on 683 and 684 rows above 100 m. The earlier sweep that FAVOURED 1.0 was
    # scored against the map's own fixes, which is circular, and that is the one
    # that got reverted.
    #
    # WHY IT IS WORTH FLYING ANYWAY, and this is not nothing: those two numbers
    # come from two different flights with different profiles, not from one
    # flight replayed twice. Nothing has flown 1.0 since the matcher stopped
    # stalling the flow loop at 60 m, since derotation moved to the camera gyro,
    # or since the second camera doubled the track count. The 0.5 m cap per fix
    # is unchanged, so 1.0 doubles the correction the map can apply - 0.24 m/s
    # against 0.12 - and flight 237's first ALT_HOLD was the map losing a race
    # at exactly that rate: it could pull 0.12 m/s while the error grew 0.14.
                                   #
                                   # 24 OCT: BACK TO 2.0, AND THIS TIME IT WAS
                                   # FLOWN. The pilot asked for 1.0 knowing the
                                   # replay disagreed - "error badhe tab bhi
                                   # karo" - and flight 00000238 settled it.
                                   # Against GPS_RAW_INT, VPS rows, band by
                                   # band:
                                   #
                                   #            237 (2.0)   238 (1.0)
                                   #   60-80 m    4.08 m      8.79 m
                                   #   > 80 m     6.16 m     10.00 m
                                   #
                                   # Fixes went 39 -> 113 and the position got
                                   # worse, which is the same direction the
                                   # replay gave. More correction towards a map
                                   # whose own error is 2.6-7.5 m is not better
                                   # correction. The plumbing stays.
    # THE CAMERA'S YAW ON THE AIRFRAME, degrees, added to the FC heading
    # before the map query is rotated north-up. The mount was moved on
    # 23 Oct and this is the calibration that was asked for then.
    #
    # Measured two ways on 24 Oct. Fitting a similarity to the translation
    # inliers gives a residual rotation of -2.03 deg on log 240 (spread 0.85)
    # and -1.87 deg on log 242 (spread 0.83), over different yaw ranges and
    # with corr against yaw of only -0.25 and +0.19 - a constant offset, not
    # a heading-scale error.
    #
    # Then checked by replaying locate() on real frames against the raw
    # receiver, which is what decided the SIGN - the first attempt tested
    # +1.95 only and read it as harmful:
    #
    #     log 240:  -1.95 -> 23 fixes, 11.22 m | 0 -> 19, 11.38 | +1.95 -> 12, 12.58
    #     log 242:  -1.95 -> 16 fixes,  9.41 m | 0 -> 14,  9.70 | +1.95 -> 10,  8.19
    #
    # Negative gains fixes and loses nothing on both flights; positive halves
    # the fix count on both. The error gain is small (1-3%); the fix-rate gain
    # is 14-21%, which is what the map is short of.
    #
    # NOT applied to the flow's own yaw rotation. The same offset should
    # physically appear there too, but measuring it from the flow direction
    # against the receiver gave +3.05, +6.21, -1.19, +5.43 deg across three
    # flights with an interquartile spread of 15-49 deg - the phantom
    # velocity swamps it. One place at a time, where it is measured.
    # 3 NOV, at the pilot's word: -1.95 -> -4.42. The mount was rebuilt after
    # 27 Oct, so -1.95 describes a mount that no longer exists. -4.42 was
    # measured three separate ways on the mount as it stands.
    #
    # ON THIS CODE IT REACHES ONE CONSUMER ONLY. Of the three places a
    # camera-frame measurement is rotated into north-east, only the map
    # matcher adds it (worker.submit, "f[yaw] + self.cam_yaw_offset_deg").
    # shift_to_ne, which turns the flow into the position AND the velocity,
    # and accel_to_ne, which dead-reckons between flow updates, both still
    # pass the raw f["yaw"]. So this changes where the matcher looks on the
    # sheet and nothing else.
    camera_yaw_offset_deg: float = -4.42

    fix_period_s: float = 2.0      # 5 Oct: 0.4 -> 2.0 s. At 0.4 s the map pushed
                                   # a STATIONARY aircraft at 1.18 m/s above 80 m
                                   # while the drift it exists to catch is
                                   # 0.15-0.37 m/s. At 2 s it pushes 0.26, and the
                                   # worker drops from 75% of a core to 17%.

    # --- recorder ---
    image_hz: float = 2.0          # 5 Oct: JPEGs gated separately from CSV rows
                                   # at 4 Hz; 8 Oct: halved to 2 Hz. index.csv
                                   # still logs EVERY frame - the rows carry the
                                   # conclusions, the JPEGs only answer the few
                                   # questions that need pixels back.

    # --- velocity sent to the FC ---
    # 4 Oct: the hover deadband was flat 0.5 m/s and became 0.35 + 0.005 x alt.
    # Flown, kept. The phantom speed it suppresses grows with height, so the
    # threshold has to as well. KNOWN COST, not yet solved: below 25 m this
    # discards 49% of genuine motion, which is why the aircraft wanders under
    # 10 m while its POSITION there is accurate to 2 cm.
    hover_deadband_base_mps: float = 0.35
    # 0.005 -> 0.010, 22 Oct. See VelocityBridge's deadband_per_m: at 83 m
    # the old threshold was 0.77 m/s against a sustained phantom whose p90 was
    # 0.91, so the worst ones passed and LOITER flew them out.
    hover_deadband_per_m: float = 0.010

    # --- camera (OAK-D Pro mono pair) ---
    # Measured on this device 19 Oct: THE_800_P gives 1280x800 and 29.8-30.7 fps
    # on both sockets over a SUPER (USB3) link, mono focal fx 791.7 / 795.1.
    # 791.72 is what bench_workload.py and the matcher already use - do NOT
    # confuse it with the 1943 px in map_db.py's docstring, which is the RGB
    # sensor and is only an example there.
    # 30 -> 25 on 23 Oct, at the pilot's instruction, and it sets BOTH monos.
    #
    # With the partner camera streaming as well, 30 fps asked for delivered
    # 28.5 and 22% of frames arrived late - against 7% with one camera - so
    # the link was being asked for more than it could hold evenly. The loop
    # rate was never the symptom; the JITTER was, and a dropped frame costs
    # flow a whole interval of tracking.
    mono_fps: int = 25

    # Camera IMU report rate, Hz, for all three enabled sensors - they share
    # one packet, so one number sets them together.
    #
    # 100 WAS CALLED A CEILING AND IS NOT ONE. Measured on this device
    # 21 Oct, IMU alone: asking 400 delivered 357.9 Hz on BOTH
    # GYROSCOPE_CALIBRATED and LINEAR_ACCELERATION, against Luxonis' table
    # giving the calibrated gyro a maximum of 100.
    #
    # Why it matters, from log 225's own camera recording in flight: the
    # linear acceleration put 94% of its energy above 10 Hz with every top
    # peak at 49.8 Hz - hard against Nyquist for a 100 Hz sample. At rest the
    # same channel spreads its energy (59.8% below 10 Hz on X), and its sd is
    # 0.030 m/s2 against 1.30 in flight, so what piles at Nyquist is real
    # vibration above 50 Hz folding down, not sensor noise. A hex at hover
    # puts its prop fundamental near 65-85 Hz, which is exactly there. The
    # velocity bridge integrates that aliased signal.
    #
    # 3 Oct recorded the opposite result - "asking for 400 on the gyro dragged
    # the whole stream down rather than speeding anything up" - so the
    # comparison was run again 21 Oct on the FULL pipeline, both mono streams
    # and stereo depth up, 800p at 30 fps, all three IMU sensors:
    #
    #     requested   frames    IMU packets   accel/gyro/quat
    #       100 Hz    30.0 Hz       22006         99.9 Hz
    #       400 Hz    30.0 Hz       36601        165.9 Hz
    #
    # The frame rate does not move. 400 does not deliver 400 with the camera
    # streams competing for the link - it delivers 165.9 - but that is 1.66x
    # the reports and, more to the point, it moves Nyquist from 50 Hz to 83.
    # The same comparison at rest shows the fold directly: at 100 Hz the accel
    # peaks sat at 36.0-48.7 Hz, and at 400 Hz the same content appears at
    # 59.0-66.8 Hz. It was never at 36 Hz; it was folding there.
    #
    # 83 Hz still only just covers a hex's hover fundamental, so this is an
    # improvement rather than a solution. What it buys is the first honest
    # look at where the vibration actually is.
    imu_rate: int = 400

    # How many frames the flow VELOCITY is measured over. Position is
    # unaffected - it integrates every frame as before.
    #
    # It was one frame until 22 Oct, which is the worst possible choice,
    # because velocity = pixels x altitude / focal: at 80 m one pixel is
    # 80/791.72 = 0.101 m, and at 30 fps that is 3.03 m/s PER PIXEL of
    # frame-to-frame jitter. Measured on log 226, per-axis error against the
    # raw receiver, by baseline length:
    #
    #     N frames   lag     5-25m   25-45m   45-70m   70-85m
    #        1      0.02 s    1.79     8.45     9.67    17.79 m/s
    #       20      0.33 s    0.25     1.76     1.31     3.75
    #       30      0.50 s    0.15     1.31     0.83     2.61
    #
    # Summing N displacements and dividing by their span divides coherent
    # motion by N but white jitter only by sqrt(N), so the noise falls as
    # 1/sqrt(N) while real motion is preserved. The cost is lag: the estimate
    # is the average over the span, so it is centred N/2 frames in the past.
    #
    # 20 and not 30. The lag is NOT compensated - GPS2_DELAY_MS stays 0 -
    # because one GPS delay applies to the position too, and the position is
    # the part that works (SP never left 0.56 against a 1.2 gate on log 226).
    # Delaying it to help the velocity would trade the good signal for the
    # bad one. The lag penalty is therefore already inside the numbers above:
    # they are measured against the receiver's speed AT THE SAME INSTANT, not
    # against a shifted copy.
    #
    # The displacements are the SAME ones the fuser consumed, so position and
    # velocity still cannot disagree about what the camera saw - which is what
    # the EKF reads as a velocity innovation.
    # 20 -> 40 ON 22 OCT, after log 227 would not hold in any VPS window.
    #
    # The pilot's own observation settled where to look: on log 225 the
    # aircraft anchored itself as long as no stick was given and ran away the
    # moment one was. Measured below 60 m across logs 225/226/227, where the
    # map never runs so this is flow alone, the velocity error against the raw
    # receiver by the aircraft's TRUE speed:
    #
    #     N    lag    hover 0-0.5   0.5-1    1-2     2-3    3-5 m/s
    #     20  0.33 s     0.50        0.90    1.22    2.44   5.70
    #     30  0.50 s     0.29        0.58    0.68    2.23   4.65
    #     40  0.67 s     0.24        0.51    0.60    1.97   3.98
    #     60  1.00 s     0.17        0.36    0.49    1.54   2.56
    #     90  1.50 s     0.16        0.43    0.61    1.42   2.06
    #
    # At N=20 the hover error (0.50) sits under the 31 m deadband of 0.66 and
    # is suppressed - which IS the anchoring the pilot saw. The moment the
    # aircraft moves at 0.5 m/s the error is 0.90, above the deadband, so it
    # reaches the flight controller; and a sustained false velocity is the one
    # thing LOITER cannot settle against, because with no stick it can only
    # cancel a velocity error by flying. Hence the runaway on stick.
    #
    # N=40 puts 0.5-2 m/s at 0.51-0.60, BELOW that same deadband, so the band
    # LOIT_SPEED_MS=3.0 actually commands is suppressed too. 60 is better
    # still and costs 1.0 s of lag, which a position controller acting on
    # stale velocity can turn into the oscillation this is meant to remove -
    # not taken without a flight behind it.
    #
    # NOT a scale error and not derotation, both tested and both ruled out:
    # correcting the measured scale (forward 0.903, sideways 0.873 on each
    # axis's own dominant samples) made the velocity error WORSE by 9-13% and
    # did nothing for position drift in either direction; and holding speed at
    # 0.5-2 m/s, per-frame tilt of 0-0.10 deg and 0.2+ deg give the same error
    # (1.42 vs 1.44), with corr(error, tilt) +0.109 against corr(error, speed)
    # +0.662. What is left is the flow's own pixel noise: 1.4 m/s at 31 m is
    # 1.2 px per frame, and a longer baseline is the only thing that touches it.
    flow_vel_frames: int = 40

    # Above flow_vel_alt_m the baseline lengthens to flow_vel_frames_high.
    #
    # Measured per altitude band across logs 225/226/227/228, velocity error
    # p90 against the raw receiver (p90, not p50, because the spikes are what
    # trip FS_EKF_THRESH):
    #
    #     N    lag    5-15m   15-25m   25-40m   40-60m   60-90m
    #     20  0.33 s   0.80    1.31     3.38     3.55     7.79
    #     40  0.67 s   0.43    0.65     2.31     1.89     5.31
    #     60  1.00 s   0.42    0.59     1.98     1.49     4.18
    #     90  1.50 s   0.53    0.73     1.75     1.39     3.60
    #
    # Below 25 m the gain stops at about 40-60 and 90 is WORSE (0.43 -> 0.53 at
    # 5-15 m): real motion over 1.5 s starts being averaged away. Above 25 m it
    # keeps improving, so 60 is taken there - 14-21% better than 40 - and 90 is
    # not, because its 1.5 s of lag is a cost this measurement cannot see. A
    # position controller acting on stale velocity oscillates, which is the
    # thing being removed.
    #
    # The archived pre-SSD engine built the same mechanism and reached the
    # OPPOSITE conclusion, on 216856 airborne samples of 17-18 Sep:
    #
    #     height    live    long baseline   gain
    #      5-15 m   1.21       0.57         2.14x
    #     20-30 m   2.43       1.25         1.95x
    #     30-40 m   2.82       1.62         1.74x
    #     40-50 m   3.09       2.82         1.09x
    #     50-60 m   3.22       3.43         0.94x  - worse
    #
    # with long_baseline_weight ramping the long baseline OUT between 30 and
    # 40 m because "above 40 m it stops helping and then hurts - up there the
    # error is dominated by rotation and timing, which do not shrink with the
    # baseline". That does not reproduce here: the baseline keeps helping to
    # 90 m, and above 25 m corr(error, tilt) is +0.093 and corr(error, yaw
    # rate) +0.044 against corr(error, speed) +0.629. Holding speed at
    # 0.5-2 m/s above 25 m, tilt of 0-0.2 deg gives error p50 1.06 and 0.5+ deg
    # gives 1.21 - no rotation signal. Different airframe, different camera,
    # and the direction is reversed, so their ramp is deliberately not copied.
    # REVERTED ON 22 OCT, THE SAME DAY, BY FLIGHT: both are set to the
    # flat value so the baseline no longer changes with height.
    #
    # The sweep that justified 60 above 25 m stands - velocity error p90 at
    # 25-40 m really is 2.31 at N=40 and 1.98 at N=60 - but log 229 flew it
    # and the aircraft ran 11.1 m in 24 s at 31.6 m (N about 47 on the ramp)
    # against 0.9 m in 23.2 s at 31.5 m on log 228 with a flat 40. A longer
    # baseline reports less of the real motion, and at 17.2 m our speed read
    # p50 0.12 m/s with 77% deadbanded while the aircraft actually moved at
    # 0.65 m/s - the flight controller was told it was standing still, so it
    # did not correct.
    #
    # The founding brief says it plainly: "keep corr_rate_mps and reanchor_m
    # FLAT - never scale with height. (Scaling them previously cost two
    # flights and was reverted.)" Those two constants are dead, but the rule
    # was not about them. This is the third height-scaled gain this project
    # has flown and reverted.
    #
    # The machinery stays, so a future attempt needs a flight and not a
    # rewrite: set these to differ and _vel_baseline_frames ramps again.
    #
    # 24 OCT, FLIGHT 239, AND IT STAYS FLAT. This is the first sustained
    # flight above 70 m - 86 m median, 110.9 m max, 795 s - and up there a
    # standing aircraft (true speed under 0.5 m/s against the raw receiver) is
    # reported moving at:
    #
    #     25-40 m   0.00 m/s median      the deadband absorbs it
    #     40-55 m   0.89
    #     70-85 m   2.40           p90 5.56
    #     85-100 m  2.53           p90 5.89
    #
    # against a deadband of 0.35 + 0.010 x altitude, so 1.45 m/s at 110 m - the
    # phantom clears it and reaches the flight controller. corr(flow, truth) is
    # 0.19-0.30 up there, so this is not a scale error on a signal that tracks;
    # the signal barely tracks. EKF SV ran p50 0.73 against 0.040 on the
    # receiver, 22-67% of samples over FS_EKF_THRESH 1.2, and the aircraft fell
    # to ALT_HOLD on the EKF velocity failsafe twice, at t 270 and t 361, both
    # at 110 m.
    #
    # A longer baseline is the obvious answer and it is NOT taken, because the
    # row above says log 229 already flew it: 11.1 m in 24 s at 31.6 m against
    # 0.9 m in 23.2 s on a flat 40, and the aircraft told it was standing still
    # does not correct. The sweep that would settle it at 85-110 m - the band
    # none of 225/226/227/228 reached - CANNOT BE RUN ON WHAT IS LOGGED:
    # differencing the fused north_m/east_m reproduces the node's own vel_flow
    # at corr 0.374 (the fuser is not a raw integrator), and frames land every
    # 3-8 so no consecutive pair survives to re-track. step_dn/step_de/
    # step_dt_s were added the same day for exactly this, so the NEXT flight's
    # CSV answers it as arithmetic.
    #
    # IT DID, THE SAME DAY - FLIGHT 240 - AND THE ANSWER IS NO. The sweep ran
    # on that flight's own step_dn/step_de, reconstruction validated against
    # the node's own vel_flow at corr 0.9990 and a median difference of
    # 0.0021 m/s, with the 6.9% of frames that produced no shift masked and
    # the window reset at each of them exactly as _vel_hist.clear() does:
    #
    #     N    lag     phantom 35-45m  70-95m  95-112m | real motion the
    #                  (reported while truly under 0.5)| deadband zeroes
    #     20   0.40 s      0.61         1.27    1.96   |      33.3%
    #     40   0.80 s      0.52         1.40    1.67   |      38.6%
    #     60   1.20 s      0.52         1.40    1.46   |      43.6%
    #     90   1.80 s      0.49         1.40    1.41   |      49.6%
    #     120  2.40 s      0.41         1.40    1.29   |      51.7%
    #     160  3.20 s      0.32         1.40    1.26   |      52.2%
    #
    # At 70-95 m the phantom is FLAT at 1.40 from N=40 onward - the baseline
    # does nothing there at any length. At 95-112 m it falls 25% between N=40
    # and N=160, for 3.2 s of lag, while the share of real motion the deadband
    # zeroes goes 38.6% -> 52.2%: that second column is log 229's runaway,
    # measured rather than remembered. corr(reported, truth) sits at 0.23-0.25
    # at EVERY N, so length does not make the signal track either.
    #
    # White noise halves in power when the window doubles. This does not move.
    # The phantom is a CORRELATED error that persists over seconds, so no
    # amount of averaging will reach it, and the whole longer-baseline family
    # is now closed by this aircraft's own data instead of by recollection.
    # Flat, and not to be reopened without a different mechanism.
    flow_vel_frames_high: int = 40
    flow_vel_alt_m: float = 25.0

    # Stereo depth: ON, and for VIO ONLY.
    #
    # It is cheap where it was feared to be expensive. Measured 19 Oct on this
    # camera: adding StereoDepth left left/right/depth all at 30.7 fps and the
    # IMU at 104 Hz, for +2 C of chip temperature - because the depth is
    # computed on the OAK's own Myriad X VPU, not on the Jetson. The 9 Oct
    # figures (fps 25 -> 12.5, camera +12 C, main loop stalled 44 s and 30 s)
    # were host-side contention with the flow loop, map worker and FC bridge
    # all running, not the camera's own cost. One real warning did show up
    # even alone: the worst frame gap went 35.2 -> 47.2 ms, already past one
    # frame period at 30 fps, which is how those stalls begin.
    #
    # What it is NOT: a height source. That is the part 9 Oct reverted, and the
    # report's own verdict was that it came out "worse than the baro it was
    # meant to help". Height stays TF03 below lidar_max_alt_m, barometer above.
    # --- the aircraft's own leg, in the left camera's frame ---
    # Measured 20 Oct off CAM_B at 1280x800. The left landing leg occludes the
    # left edge: columns 0-63 fall below 60% of the frame median (70-90 against
    # 147.5) and the darkening tapers out by x~120. The RIGHT camera is flat
    # across every column, 153-157, which is what proves it is an object in
    # front of CAM_B and not lens vignetting - both sensors are the same part.
    #
    # It matters because it is RIGIDLY ATTACHED. 32 of 300 detected corners sat
    # inside it, and a corner on the aircraft reports zero motion in every
    # frame while the ground moves. RANSAC does not save you from that: those
    # tracks agree perfectly with each other on a single rigid motion of zero,
    # so a tenth of the evidence is pulling the estimate towards standing still.
    #
    # 130 -> 0 on 22 Oct, because the primary camera moved to CAM_C and the
    # leg is CAM_B's. See live_node.PRIMARY_SOCKET_NAME.
    #
    # The leg was photographed that day with the airframe lifted: on CAM_B it
    # and its foam grip fill x 0-75 and a bright strut reaches about x 165,
    # while CAM_C's left edge is clear ground. So on CAM_C there is nothing to
    # crop, and this buys back three things the crop was costing - 10% of the
    # frame's width for flow to find corners in, a cx only 4 px off the image
    # centre instead of 80, and one less transform to carry a pixel through.
    #
    # It is kept as a setting rather than deleted: it is the only lever if the
    # leg ever shows up in the primary's frame, and the 130 px figure is the
    # measured one for CAM_B (120 px to where the taper ends, plus margin).
    # Cropped, not masked, if it is ever needed again: a masked block is a
    # uniform patch that a correlator would happily match against.
    # BOTH MONO CAMERAS RUN OPTICAL FLOW, and their estimates are averaged.
    #
    # At the pilot's instruction, 23 Oct, after the camera mount was lowered
    # 2-3 cm and the landing leg left CAM_B's frame - confirmed in the air and
    # from a ground photograph, where the leg used to sit at x 112-179,
    # y 728-800 and is now gone.
    #
    # WHAT IT BUYS, and it is reliability rather than accuracy. Tracks fall
    # with height on this camera - measured across flight 00000237, altitude
    # against track count correlates -0.63, from 279 at 17 m to 215-230 at
    # 88 m - and that flight's first EKF failsafe came when the count
    # collapsed 231 -> 80 at 88 m. Two cameras roughly double the features, so
    # a dropout needs both to fail at once.
    #
    # WHAT IT DOES NOT BUY: the dominant flow errors are SHARED between the
    # two. Both use the same altitude for the pixel-to-metre scale, the same
    # IMU for derotation and the same yaw, so averaging cannot touch either -
    # and those are what the measurements say dominate (slope against the
    # receiver swings 0.51 to 2.68 with per-frame rotation). What averaging
    # removes is the part that is independent between the sensors, which is
    # the feature-localisation noise, and two of those average down by
    # sqrt(2).
    #
    # WHAT IT COSTS: CAM_B's frames did not cross XLink before - only the
    # primary left the device - so this adds a second 1280x800 mono stream at
    # the mono rate, and a second LK pass on the Jetson, which was already
    # running at about 56% of six cores.
    # THE TWO CAMERAS' DISAGREEMENT IS A VETO, and it is the only thing in
    # this project that can tell a broken tracker from a bad estimate.
    #
    # Measured on flight 00000238, 5216 VPS frames with both cameras and a
    # receiver truth: healthy frames agree to 0.03-0.08 px p50 and 0.28 px
    # p90 even at the highest rotation, and the frames where the flow bursts
    # disagree by 30-36 px. That is a hundredfold separation, not a margin.
    #
    # It matters because those rare frames are what ends a window. At t 331
    # the flow reported 23.71 m/s while the aircraft did 1.70, the bridge
    # smoothed it to 11.16 and the cap held it at 6.00 - and XKF4.SV went to
    # 3.71 against a 1.2 gate. Derotation was NOT the cause: it removes 92.7%
    # of the rotation effect and the residual is flat at 0.48-0.71 m/s. The
    # tracker simply failed on those frames, independently in each camera.
    #
    # Thresholds, scored against the receiver on that flight:
    #
    #     threshold   rejected   flow err p50 / p90 / p99
    #      (none)        0.0%        0.58 / 2.08 / 9.31
    #      0.5 px        3.1%        0.56 / 1.92 / 5.96
    #      2.0 px        2.7%        0.56 / 1.93 / 6.20
    #
    # 0.5 px takes the tail - p99 9.31 to 5.96 m/s - and leaves the middle
    # alone, which is the whole point: p50 and p90 barely move. A rejected
    # frame sends IGNORE, not zero. "I do not know" is a thing the EKF can
    # ride out on the IMU; a 4 m/s lie is not.
    #
    # It only works while both cameras are running. With one, there is
    # nothing to compare and the veto cannot fire - see dual_flow.
    flow_disagree_max_px: float = 0.5
    # How old the frame the loop acted on may be. The queue is drained to the
    # newest now, so this is a backstop rather than the mechanism: if the
    # newest frame available is itself this old, the camera or the link is in
    # trouble and the position is stale enough to matter. 3 m/s for 0.12 s is
    # 0.36 m, and the flight controller reads a stale position as motion.
    # 0.12 -> 0.28 ON 5 NOV, at the pilot's instruction, after the 10:57
    # flight threw away 42% of its frames.
    #
    # A frame older than this is REFUSED and ntrk is set to 0, which is what
    # the pilot heard as "VPS TRACK LOST 0" over and over - not the tracker
    # failing, the node declining the frame. Measured on that flight,
    # frame_age p50 0.1028 and p90 0.2519 against a 0.12 limit, so 42% of
    # frames were dropped, the flow ran at 58% of its rate and the error came
    # out at 6.46 m p50 against the 08:56 flight's 3.41 m.
    #
    # Why accepting them is sound: the flow measures displacement between two
    # consecutive frames and divides by THEIR OWN dt, so a late frame is a
    # valid measurement that arrived late, not a wrong one. And the FC is not
    # given our velocity at all (EK3_SRC1_VELXY = 0), so the latency cannot
    # corrupt its velocity. What it costs is position lag: up to 0.28 s, which
    # at 2 m/s is 0.56 m - against the 6.46 m that dropping the frames bought.
    #
    # The latency is a STARTUP TRANSIENT, not thermal and not the frame
    # saving. Both flights started high and settled - 489 from p50 0.2096 in
    # its first half to 0.0526 in its second, 486 from 0.0413 to 0.0000 - and
    # saving a frame changed the next samples' age by +0.008 s on 486 and
    # -0.002 s on 489, which is nothing. 489's VPS leg (t=106-349 of 522 s)
    # was flown entirely inside its unsettled period.
    # 0.28 -> 0.12 on 6 Nov morning, then BACK TO 0.28 the same afternoon.
    # The revert was wrong and the reasoning behind it was wrong: it read
    # "p50 0.0000, 0.0% over 0.12" off a GROUND run, where nothing is late,
    # and concluded 0.28 was buying nothing. Two flights that afternoon, both
    # at 28 m, gave three airborne "VPS TRACK LOST 0" at frame ages 0.1206,
    # 0.1267 and 0.1331 - every one of them a frame this limit threw away at
    # a hundredth of a second over, with the tracker working fine. Run 504's
    # normal rows sit at p95 0.119, a millisecond under, so it tips over
    # whenever the pipeline breathes.
    #
    # Raising it is not the cure. Run 504's load log says the lateness is NOT
    # load - corr(frame_age, rq_wait_pct) = -0.22 and corr with cpu_pct
    # = -0.40, both NEGATIVE, and the late seconds carried LESS CPU than the
    # clean ones. The row rate held 21.6-24.7 Hz throughout, so nothing was
    # dropped; the frames arrived on time and old. It is one continuous block,
    # t=145-237, rising 0.073 -> 0.116 and plateauing there - about three
    # frames at 24 Hz - then falling back as the aircraft descended. A
    # plateau is a queue that filled and stayed, not a clock that drifted.
    # Until that queue is found, this limit is the difference between
    # dead-reckoning quietly and shouting TRACK LOST at the pilot.
    frame_age_max_s: float = 0.28
    dual_flow: bool = True
    crop_left_px: int = 0

    stereo_depth: bool = True
    stereo_depth_as_height: bool = False   # keep False. See above.

    # --- camera IMU (OAK-D Pro) ---
    # 5 Oct config, with gyro at 100 Hz per the pilot's instruction rather than
    # the 200 that report lists. 3 Oct measured why over-asking is not neutral:
    # all three reports share one packet, so gyro at 400 dragged the whole
    # stream to 57.8 Hz, and a calibrated gyro requested at 200 against a
    # documented 100 Hz maximum delivered exactly 100 anyway.
    imu_gyro_hz: int = 100
    imu_rotation_vector_hz: int = 100     # GAME_ROTATION_VECTOR
    imu_linear_accel_hz: int = 100        # LINEAR_ACCELERATION
    imu_batch_reports: int = 1     # 3 Oct: batching 5 left 34.2% of frames with
                                   # NO IMU sample at all; at 1 it is 1.1%, same
                                   # camera rate, same gaps.
    imu_magnetometer: bool = False  # 5 Oct: it measured clean, but
                                   # ROTATION_VECTOR shares the rotation pipe and
                                   # taking the magnetometer cost derotation and
                                   # two flights. Stays off.

    # --- height source ---
    # 9 Oct: the TF03 went from not fitted to PRIMARY below ~56 m. Flown.
    # 94% usable below 58 m (stuck 0.7%, out of range 5.3%, spikes 0.2%).
    #
    # READ FROM THE FLIGHT CONTROLLER, log 00000195.BIN (15 Oct, the last
    # flight on the card and later than every status report):
    #   RNGFND1_MAX     60      <- so 60, not the report's approximate 56
    #   RNGFND1_TYPE    1       = ANALOG
    #   RNGFND1_PIN     4
    #   RNGFND1_SCALING 72.61   RNGFND1_OFFSET 0.606977  RNGFND1_RMETRIC 1
    #   RNGFND1_GNDCLR  0.13    RNGFND1_ORIENT 25 (down)
    #
    # GNDCLR is now 0.26, read back off the FC on 19 Oct after the move to the
    # ZD850 hexa - its 350 mm landing gear doubles the lidar's height above the
    # ground. The matching code-side constants are in cuav_mavlink_bridge
    # (lidar 0.26, camera 0.17), and on this airframe the camera sits BELOW the
    # lidar rather than above it.
    #
    # Note what that says: it is STILL on an analog pin. 5 Oct blamed the
    # lidar's 253 cm reading against a true 86 cm on "a serial TF03 wired to an
    # analog pin", so it is tempting to assume it was rewired to serial. It was
    # not. What fixed it was calibrating the analog path - that SCALING and
    # OFFSET - not changing the wiring. Moving it to a serial port is still
    # available and still unmeasured.
    lidar_max_alt_m: float = 60.0  # above this, height comes from the barometer

    # --- heading ---
    # NOT the camera. 4 Oct tested GAME_ROTATION_VECTOR's yaw for exactly this
    # and it cannot do it. Heading is still the FC compass, which is the last
    # hard dependency: the whole position solution rotates with it.
    heading_from_camera: bool = False

    def R(self) -> np.ndarray:
        return np.asarray(self.R_vio_to_ned, float)
