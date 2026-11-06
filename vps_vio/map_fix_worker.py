"""Run the map fix on its own core so it never stalls the flow loop.

A fix costs 68 ms. The flow loop's budget is 33 ms at 30 Hz or 100 ms at 10 Hz,
so a fix taken inline blocks one frame outright - measured as 24 overruns in
244 frames at 10 Hz, and almost every frame at 30 Hz. The Orin has six cores
and was using roughly one.

Threads are enough here, despite the GIL: OpenCV releases it around the heavy
calls, so matchTemplate genuinely runs on another core. No second process, no
shared-memory plumbing, and the map stays in one place.

BOTH NUMBERS ABOVE ARE THE NCC MATCHER'S, AND THE MATCHER IS NOW XFEAT.
Measured 24 Oct on this Orin with the node live, locate() on a real 110 m
frame against this map, the window radius the code actually picks:

    altitude   radius   map window   locate()
      42 m     40.0 m    156x156      0.098 s
      60 m     40.0 m    156x156      0.145 s
      87 m     52.2 m    204x204      0.270 s
     110 m     66.0 m    258x258      0.319 s

The radius is clip(window_alt_frac x altitude, 40, radius_m), so the cost
grows with height and 68 ms is a floor that only holds near the clip. XFeat is
not matchTemplate either - its keypoint handling, matching and RANSAC are
Python-level and do NOT release the GIL, so a 0.319 s fix is largely a 0.319 s
stall of the flow loop.

Flight 239 shows it: loop iterations slower than 60 ms went 3.6% at 37 m to
47.7% at 110 m, and the node threw away 544 camera frames a minute up there
against 3 a minute at 37 m. The slow fraction tracks ALTITUDE, not the number
of fixes - 300-360 s gave 11 fixes at 47.7% slow while 540-600 s gave 24 fixes
at 11.6% - which is what a per-attempt cost that scales with the window looks
like, and not what a busy matcher looks like.

The matcher accounts for roughly half of it: 0.319 s every fix_period_s of
2.0 s is 16% of wall time, about 4 of the 9 frames dropped per second. The
rest is the base load of two FlowTrackers at 25 fps. A second PROCESS is the
fix for the matcher's share and is not written yet - the trade against a
smaller window is unmeasured, because a 40 m radius at 110 m may not contain
the true position once the drift is 20 m, which is why the radius scales in
the first place.

The queue holds exactly one job. If a fix is already running, a new request is
DROPPED rather than queued - a fix computed against a prior that is two seconds
stale is worse than no fix at all, and the flow keeps the position meanwhile.

Two rates matter and they are not the same. Dropping-while-busy is about not
queueing stale work; fix_period_s is about not asking too often in the first
place. At 0.4 s the map pushed a stationary aircraft at 1.18 m/s above 80 m
against a real drift of 0.15-0.37, and the worker ate 75% of a core. At 2.0 s
it pushes 0.26 and costs 17%. The period is enforced here as well as by the
caller, so a caller that forgets cannot put that behaviour back.
"""
from __future__ import annotations
import math
import threading
import time
from dataclasses import dataclass

import numpy as np

# Default thresholds per matcher. They are not interchangeable: the NCC margin
# is read off a coarse correlation surface and peaks around 0.1-0.2 on
# four-year-old imagery, while the XFeat numbers are an inlier ratio and a
# cluster tightness. Swapping one matcher's gate onto the other would either
# pass everything or nothing.
_GATES = {
    "ncc":   dict(min_score=0.05, min_margin=0.01),   # swept: 50 frames, +-150 m
    # 0.10, NOT 0.25, and this is now the swept value rather than the guess.
    #
    # 12 Oct introduced 0.25 for a real reason: the gate "had never been
    # applied on the XFeat path at all - that branch hard-coded ok=True, so 189
    # of 189 fixes were accepted including ones 85.66 m out". That is an
    # argument for HAVING a gate, which this still is, not for 0.25
    # specifically - and the same report doubted the number in the same
    # paragraph: "Its rejections track our own error... because the matcher
    # takes its window from our estimate, so the gate closes exactly when it
    # needs to be open."
    #
    # Swept 21 Oct on the first frames this project has ever had above 72 m -
    # 601 above the 60 m floor from the 95 m flight, matched against the raw
    # receiver:
    #
    #     90-100 m, 477 matches      score p50 0.077, p95 0.187, MAX 0.248
    #     at min_score 0.25            0 of 477 pass. Not one, all flight.
    #     at 0.10, min_inliers 12    176 pass, error p50 8.3  p95 12.9  max 24.8 m
    #     at 0.00, min_inliers 12    228 pass, error p50 8.3  p95 12.7  max 24.8 m
    #
    # 8.3 m is better than the fuser's own fix_sigma_m of 20, and nothing like
    # 12 Oct's 85.66 m outliers. The score does not predict the error at all
    # there - r = -0.118 against it, inliers -0.158, margin +0.034 - so the gate
    # buys accuracy with yield and gets none back, which is why it goes to the
    # bottom of the swept range rather than to zero: it still catches the
    # degenerate case that 12 Oct found, and costs 52 of 228 fixes to do it.
    #
    # What bounds the damage is downstream and unchanged: map_fuse drops a fix
    # beyond 3 sigma outright, and max_correction_m caps an accepted one at
    # 0.5 m however confident it was.
    # 0.10 -> 0.07 ON 22 OCT. The pilot's point: this matcher gave 4-6 m at
    # 110 m and the number did not move wherever it flew, so it is not
    # mis-tuned - it is under-flown and under-fed with fixes.
    #
    # Both halves of that turned out right. Our own flights, position error
    # above 60 m against the receiver, by how often a fix landed:
    #
    #     run        fixes   rate     error p50
    #     vps_00038    0    0.00/s     39.69 m
    #     vps_00047    0    0.00/s     38.40 m
    #     vps_00054    5    0.05/s     38.07 m
    #     vps_00066   19    0.27/s      4.50 m   <- the pilot's 4-6 m
    #     vps_00076   98    0.40/s      9.77 m
    #     vps_00091    2    0.04/s      8.51 m
    #
    # So the system error is set by the FIX RATE, not by the accuracy of any
    # one fix. A single fix is 18-22 m and that is random, not biased - mean
    # offset 15.1 m against a spread of 19.9 m - so averaging N of them falls
    # as 1/sqrt(N) and 25 fixes reach 4 m. Tuning this matcher means raising
    # the accept rate, and nothing else.
    #
    # Swept on 78 frames across three altitude bands, holding min_inliers 12
    # and min_margin 0.20:
    #
    #     band      min_score   accept   per-fix err   system err
    #     75-90 m     0.14       15.4%      19.7 m       9.19 m
    #                 0.10       34.6%      22.3 m       6.92 m
    #                 0.07       42.3%      22.3 m       6.26 m
    #     90-100 m    0.10       15.4%      18.2 m       8.48 m
    #                 0.07       19.2%      18.1 m       7.54 m
    #
    # The per-fix error does NOT degrade - 22.3 to 22.3 and 18.2 to 18.1 - so
    # the fixes admitted between 0.07 and 0.10 are as good as the ones already
    # passing. Below 0.07 nothing more changes, because min_inliers 12 becomes
    # the binding gate: at the 124 matches this matcher typically returns, a
    # ratio of 0.10 and a count of 12 are the same constraint, which is why
    # lowering min_inliers alone did nothing at all.
    "xfeat": dict(min_score=0.07, min_margin=0.20),
    # SAME score and margin, and a different min_inliers - which locate_lg
    # carries itself, because 20 against 12 is the whole difference between a
    # LighterGlue that is bounded at 14.79 m and one that reaches 152.67 m.
    # Its inlier ratio runs 0.248 against mutual NN's 0.053, so a score gate
    # written for one barely gates the other, and the floor is what bites.
    "xfeat_lg": dict(min_score=0.07, min_margin=0.20),
}


@dataclass
class FixResult:
    lon: float
    lat: float
    score: float
    margin: float
    ok: bool
    age_s: float          # how long the fix took, i.e. how stale it is on arrival
    prior_lon: float      # the prior it was computed against, so the caller can
    prior_lat: float      # apply it as a correction to that, not to "now"
    disagree_m: float | None = None   # how far the two matchers were apart,
                          # when both produced a fix. None if only one ran or
                          # only one succeeded. The measured predictor.
    inliers: int = 0      # RANSAC inliers behind the fix
    total_matches: int = 0
    reason: str = ""      # why a rejected fix was rejected, for the recorder
    # WHICH MATCHER THIS POSITION CAME FROM. Until 4 Nov nothing recorded it
    # and the answer could only be inferred from fix_inliers being zero,
    # because NCC does not produce inliers. That inference is how it came out
    # that NCC had published every fix of flight 478 and XFeat none of them.
    # Named now, so the log says it outright.
    source: str = ""
    # How stale the roll and pitch behind the nadir correction were. See
    # MapFixWorker.submit: the correction moves the answer from the middle of
    # the frame to under the aircraft, and it is only as good as the attitude
    # it is given. Measured 4 Nov on 451 frames of two flights, accepted
    # fixes, error p50/p90/max against attitude age:
    #     0 ms  3.48/ 6.18/11.26    100 ms  3.48/ 7.68/11.60
    #   200 ms  4.65/ 9.97/18.55    400 ms  6.32/15.13/29.41
    # so 100 ms costs nothing and 400 ms doubles it. ATTITUDE arrives at
    # 10 Hz, which is inside the harmless band - the risk is not the rate but
    # the stream stalling without anyone noticing, which is exactly what
    # nothing could see before this field existed.
    att_age_s: float | None = None
    # The altitude the MATCH was made at, not the altitude now. Added 22 Oct:
    # the matcher's error depends strongly on height, because the search
    # window is window_alt_frac x altitude and a smaller window holds fewer
    # features - and until now the fix's own position was never written down,
    # so that dependence could only be taken from an archived report of a
    # configuration this code does not share. It is measurable here instead.
    altitude_m: float = 0.0


class MapFixWorker:
    def __init__(self, cache, *, focal_px: float, matcher: str = "xfeat",
                 radius_m: float = 150.0, window_alt_frac: float = 0.60,
                 query_shape: str = "full",
                 min_alt_m: float = 60.0, fix_period_s: float = 2.0,
                 min_score: float | None = None,
                 min_margin: float | None = None,
                 trust: str | None = None,
                 ncc_below_m: float = 0.0,
                 att_age_max_s: float = 0.30,
                 principal_xy=None):
        self.cache = cache
        self.focal_px = float(focal_px)
        # (cx, cy) of the frames being handed in, when the caller knows them.
        # A fix is consumed as the AIRCRAFT's position, and the pixel that
        # corresponds to is the one the optical axis pierces, not the middle
        # of the image - 80 px apart on this camera's leg-cropped frames,
        # which is 11 m of ground at 110 m, in the same direction every frame.
        # None keeps the old geometric-centre behaviour for a caller that has
        # no K: wrong by that offset, but not silently different.
        self.principal_xy = (None if principal_xy is None
                             else (float(principal_xy[0]), float(principal_xy[1])))
        self.matcher = matcher.lower()
        self.radius_m = float(radius_m)
        self.window_alt_frac = float(window_alt_frac)
        # How much of the frame reaches the matcher. Paired with
        # window_alt_frac above and meaningless apart from it - see
        # VpsConfig.map_query_shape for the measurement.
        self.query_shape = str(query_shape)
        self.min_alt_m = float(min_alt_m)
        # NCC only: how far beyond the frame's own footprint the search window
        # reaches. This IS the search range - the prior has to be within it.
        self.ncc_search_m = 45.0
        self.last_tilt_shift_m = 0.0
        self.frame_w_px = 1280.0
        self.fix_period_s = float(fix_period_s)

        # BOTH MATCHERS AGAIN, as of 25 Oct. NCC was removed on 20 Oct as dead
        # weight; it is back because flight 160 showed what it is for. XFeat
        # accepted 78% of frames at 11:38 and 0 of 40 at 15:20 on the same
        # site with sharper imagery - the shadows had swung a hundred degrees
        # from the map's. Descriptors move with the shadow; correlation of
        # gradient magnitude does not. Neither is strictly better, so the
        # choice is a setting (VpsConfig.matcher) and not a code change.
        #
        # They do NOT share a gate: see _GATES above. NCC's margin is read off
        # a coarse correlation surface and XFeat's is a cluster tightness.
        from map_match_xfeat import locate as _loc_xfeat
        from map_match_xfeat import locate_lg as _loc_lg
        from map_match_ncc import locate as _loc_ncc
        _ALL = {"xfeat": _loc_xfeat, "ncc": _loc_ncc, "xfeat_lg": _loc_lg}
        if self.matcher == "both":
            # BOTH, because neither covers the day on its own. Measured
            # 25 Oct on 45 frames from each of two flights, fixes produced:
            #
            #                   NCC   XFeat   either
            #     11:38          45     34      45   (100%)
            #     15:20          26      2      26   ( 58%)
            #
            # XFeat alone is 76% and 4%. The union is what the system error
            # actually rides on, because that error is per-fix/sqrt(N).
            #
            # And when both fire they cross-check each other: corr between
            # how far apart they are and how wrong they are is +0.928, the
            # strongest predictor of a bad fix this project has found -
            # score and inlier count both fail at it (-0.10 and +0.20).
            # The disagreement is CARRIED OUT, not acted on: refusing a fix
            # on it would be a new gate and that is the pilot's call.
            self._pair = ["ncc", "xfeat"]
        elif self.matcher in _ALL:
            self._pair = [self.matcher]
        else:
            # A COMMA LIST, so LighterGlue can be added as XFeat's fallback
            # without a new keyword for every combination. "xfeat,xfeat_lg" is
            # the measured one: mutual NN first at 520-670 ms, LighterGlue
            # only on the frames it refuses, which is where the coverage is -
            # fix rate 50% -> 70% on 338 frames, worst error 10.95 -> 14.79 m,
            # and still no fix over 20 m. See map_match_xfeat._match_lg.
            names = [s.strip().lower() for s in self.matcher.split(",")
                     if s.strip()]
            if names and all(n in _ALL for n in names):
                self._pair = names
            else:
                raise ValueError(
                    f"unknown matcher {matcher!r}, expected 'xfeat', 'ncc', "
                    f"'xfeat_lg', 'both', or a comma list of those")
        self._all_locate = _ALL
        self._window_kind = "grey"
        self._locate = _ALL[self._pair[0]]

        # ONE GATE PER MATCHER. They are not interchangeable - see _GATES -
        # and running both made that concrete rather than a comment: NCC's
        # margin comes off a coarse correlation surface, XFeat's is a cluster
        # tightness. An explicit min_score/min_margin overrides both, which
        # is what the offline sweeps pass.
        self._gate = {}
        for mname in self._pair:
            g = _GATES[mname]
            self._gate[mname] = (
                g["min_score"] if min_score is None else float(min_score),
                g["min_margin"] if min_margin is None else float(min_margin))
        self.min_score, self.min_margin = self._gate[self._pair[0]]

        # WHOSE ANSWER MAY BECOME A FIX. Both matchers still run - the
        # disagreement is the project's best predictor of a bad fix and it
        # needs two answers - but only a trusted one is published.
        #
        # Until 4 Nov there was no such idea. The loop kept the FIRST result
        # unless a later one was ok where the earlier was not, and _pair reads
        # ["ncc", "xfeat"], so NCC won every frame it accepted and XFeat's
        # answer was computed, used for the disagreement number, and thrown
        # away. On flight 478 that meant XFeat succeeded 51 times and was
        # never once the fix.
        #
        # Measured 4 Nov, 675 frames of two flights, both matchers, nadir
        # correction applied, scored against the M9N:
        #
        #                         n    NCC err p50/p90/max   XFeat err p50/p90/max
        #     both accepted     339     3.80  8.09  85.69     3.55  6.49  11.36
        #     only NCC          226    25.01 146.55 180.74   (XFeat declined)
        #     only XFeat          9    (NCC declined)         5.18  7.37   9.00
        #     neither           101     both about 104 m, both declined
        #
        # XFeat's error is bounded at 11.36 m in EVERY prior-error band and
        # EVERY tilt band, including prior 50-200 m where it accepts 1% of
        # frames, and including tilt 10-20 deg. NCC accepts 94-100% of frames
        # whatever the prior, and at prior 50-200 m it accepts 55% of them
        # with a median error of 96.88 m. XFeat's gate is a self-assessment;
        # NCC's, at min_score 0.05 and min_margin 0.01, is not.
        #
        # Taking XFeat's answer and publishing nothing when it declines:
        #
        #                      fix rate   err p50   p90     max    >20 m  >50 m
        #     NCC wins (old)   574/675     4.91    89.83  180.74    132    109
        #     XFeat trusted    348/675     3.57     6.59   11.36      0      0
        #
        # A third of the fixes go, and every fix over 20 m goes with them. On
        # the good flight the rate falls 98% -> 79% and the worst error 85.69
        # -> 9.16 m. The lost fixes were not help: their median error was
        # 25 m.
        #
        # NCC is NOT removed and NOT distrusted as a matcher. Flight 160 is
        # still on the record - XFeat 2 of 40 at 15:20 against NCC's 26, when
        # the shadows had swung away from the map's - and on a day like that
        # this setting is how NCC is believed again, without a code change.
        # On such a day the aircraft gets no map fixes and ALT_HOLD is the
        # answer, which is the standing rule for the VPS stopping.
        self._trust = [m for m in (self._pair if trust is None
                                   else [s.strip().lower()
                                         for s in str(trust).split(",")])
                       if m in self._pair]
        if trust is None and self._pair == ["ncc", "xfeat"]:
            self._trust = ["xfeat"]
        elif trust is None:
            # Everything running is trusted unless the pair is the measured
            # NCC/XFeat one above. In particular "xfeat,xfeat_lg" trusts both,
            # in that order, which is what makes LighterGlue a fallback rather
            # than a replacement: mutual NN answers when it can, and only the
            # frames it refuses cost the extra 1482 ms.
            self._trust = list(self._pair)
        if not self._trust:
            raise ValueError(f"trust={trust!r} names no matcher that is "
                             f"running (matcher={self.matcher!r})")
        # NCC IS TRUSTED AGAIN BELOW THIS HEIGHT, and the pilot was right to
        # insist. The 4 Nov measurement that removed NCC was taken on 675
        # frames of which 497 were at 100-118 m, and the conclusion was then
        # applied at every height. Below 90 m the aircraft has nothing else:
        # XFeat accepts 48% there and LighterGlue is not asked at all, so
        # removing NCC halves the coverage in exactly the band this project
        # already recorded it as necessary for - "It is NCC that makes this
        # possible", 26 Oct, tested on 156 frames, in VpsConfig.min_alt_m.
        #
        # Measured on the two 3 Nov flights that flew at 70-82 m, 197
        # attempts, against the M9N:
        #
        #                      coverage   err p50   WORST   >20 m
        #     NCC alone        172 (87%)   11.78    68.92     65
        #     XFeat alone       94 (48%)    4.80    21.53      1
        #     XFeat, else NCC  186 (94%)    6.64    45.81     47
        #
        # So NCC is NOT more accurate down there - its own fixes run 11.78 m
        # against XFeat's 4.80, and it produced a 68.9 m fix on a frame whose
        # prior was under 10 m, because its gate does not self-assess at any
        # height. What it is, is AVAILABLE: 87% against 48%. Trusting XFeat
        # first and falling back to NCC keeps the whole 94% and takes the
        # worst case from 68.92 to 45.81 m, because wherever XFeat speaks it
        # is believed instead.
        #
        # What makes the fallback survivable is downstream and unchanged:
        # map_fuse rejects a residual past 3 sigma, which is 60.3 m at
        # nominal, and caps an accepted one at max_correction_m = 0.5 m. That
        # is why those flights held with NCC publishing 65 fixes over 20 m.
        #
        # 0.0 disables the fallback entirely. NCC must also be in `matcher`
        # or there is nothing to fall back to.
        self.ncc_below_m = float(ncc_below_m)
        self.n_ncc_low = 0
        # Answers suppressed because the matcher that produced them is not
        # trusted. A fix rate that falls has to be visible as a number, not
        # inferred from the absence of fixes.
        self.n_untrusted = 0
        # Fixes whose nadir correction rested on attitude older than this are
        # published with the correction NOT applied and the age recorded, so
        # a stalled ATTITUDE stream shows up as a worse fix with a reason
        # rather than as a confident wrong one. See FixResult.att_age_s for
        # the measurement behind 0.30 s.
        self.att_age_max_s = float(att_age_max_s)
        self.n_att_stale = 0

        self._job = None
        self._result: FixResult | None = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._run = True
        self._last_submit = 0.0
        self.submitted = self.dropped = self.completed = 0
        self.skipped_low = self.skipped_period = 0
        # Runs made below min_alt_m whose RESULT is thrown away. They exist to
        # keep the path warm and to prove in the air, before 60 m, that the
        # matcher is alive. See submit().
        self.warm_runs = 0
        # Has the model finished loading, and how long it took. The pilot sees
        # this on the radio before arming.
        self.model_ready = False
        self.model_load_s = 0.0
        # Named so the load log can say "map-fix used 190% of the 273%"
        # instead of "python3".
        self._thread = threading.Thread(target=self._loop, name="map-fix",
                                        daemon=True)
        self._thread.start()

    def _warm(self) -> None:
        """Build the XFeat model NOW, in this thread, before any fix is asked for.

        THIS IS THE 60 METRE STALL. map_match_xfeat._load() is deliberately
        lazy - so that an aircraft without torch installed loses the map path
        instead of the whole node - and the first call to it therefore landed
        on the first frame above min_alt_m. Measured on this machine the load
        is 2.35 s warm and 4.95 s cold, and `import torch` holds the GIL for
        most of it, so this thread being separate bought nothing.

        It showed up in every flight, identically, and nobody had placed it:

            flight 232   56 fixes   one loop gap, 2.67 s, at t 218.0
            flight 233   93 fixes   one loop gap, 2.65 s, at t 257.2
            flight 236    1 fix     one loop gap, 2.62 s, at t 321.0

        and in all three the aircraft first crossed 60 m within 0.1 s of that
        gap. 56 and 93 fixes with no further stalls is also the proof that the
        matching itself is fine - it is only the load.

        On flight 236 the 2.62 s cost 2.6 s of optical flow, the tracker came
        back with 135 features instead of 205, the accumulated position jump
        read as 4.25 m/s of motion, and XKF4.SV went 0.16 to 2.88 against a
        1.2 gate. That was the ALT_HOLD.

        Failure here is not fatal, for the same reason the lazy load existed.
        """
        import time as _t
        t0 = _t.monotonic()
        if "xfeat" not in self._pair:
            # Nothing to warm. NCC is cv2.matchTemplate and has no model, so
            # an NCC-only worker must not drag torch in - that import is the
            # 2.35-4.95 s this method exists to move off the flying path.
            self.model_ready = True
            self.model_load_s = 0.0
            return
        try:
            from map_match_xfeat import _load
            _load()
            self.model_ready = True
        except Exception as exc:
            self.model_load_s = _t.monotonic() - t0
            print(f"MAP matcher model did NOT load ({exc}); the map path is "
                  f"off and the rest of the node is unaffected", flush=True)
            return
        self.model_load_s = _t.monotonic() - t0
        print(f"MAP matcher model ready in {self.model_load_s:.2f} s, on the "
              f"ground - this is the 2.6 s that used to be spent crossing 60 m",
              flush=True)

    @staticmethod
    def nadir_offset_m(altitude_m, roll_deg, pitch_deg, yaw_deg):
        """Where the image CENTRE lands on the ground, relative to nadir.

        A tilted camera does not look straight down, so the point under the
        middle of the frame is not the point under the aircraft - it is
        alt*tan(tilt) away, in the direction of the tilt. The matcher reports
        that centre as the aircraft's position, and nothing corrected it.

        The pilot named this before it was measured: "moving ke time drone
        tilt hota hai tho ncc bhagayega". It is geometry, not noise, and it
        bites hardest exactly when the aircraft is moving:

            tilt    @38 m   @76 m   @110 m
             2 deg   1.3 m   2.7 m    3.8 m
             5 deg   3.3 m   6.6 m    9.6 m
            10 deg   6.7 m  13.4 m   19.4 m

        Measured on the fixes of two flights: corr(tilt, fix error) is +0.484
        and corr(alt*tan(tilt), fix error) +0.506, with the error running
        8.85 m below 2 deg, 10.68 m at 2-5 and 14.33 m above 5.

        THE SIGNS ARE MEASURED, NOT DERIVED. Both were tried against the raw
        receiver on every fix of both flights, because an unmeasured sign is
        what caused the 21 Oct runaway:

            flight 240 (70 fixes)   no correction 11.18 m -> 8.69 m, p90 21.73 -> 13.45
            flight 242 (17 fixes)   no correction  9.86 m -> 6.83 m, p90 16.61 ->  8.75

        and the other three sign pairs all made it worse on both.
        """
        if not (altitude_m and altitude_m > 0.0):
            return 0.0, 0.0
        r = math.radians(float(roll_deg or 0.0))
        p = math.radians(float(pitch_deg or 0.0))
        y = math.radians(float(yaw_deg or 0.0))
        fwd = altitude_m * math.tan(p)          # + forward
        rgt = -altitude_m * math.tan(r)         # - right: the measured pair
        return (fwd * math.cos(y) - rgt * math.sin(y),
                fwd * math.sin(y) + rgt * math.cos(y))

    def submit(self, img: np.ndarray, *, prior_lon: float, prior_lat: float,
               altitude_m: float, yaw_deg: float,
               roll_deg: float = 0.0, pitch_deg: float = 0.0,
               att_age_s: float | None = None) -> bool:
        """Queue a fix. False if it was refused, for any of three reasons.

        Below min_alt_m the map does not run at all. That is deliberate and it
        is also the project's largest single error: on a 6 m/s descent the
        estimate went 4.13 to 26.0 m in twelve seconds because nothing absolute
        corrects below this height. Raising the floor is not the fix - the map
        was measured as not starting at 80 m, which is why the floor came DOWN
        to 60 - so whatever closes the descent has to come from elsewhere.
        """
        now = time.monotonic()
        # BELOW min_alt_m THE MATCH STILL RUNS AND THE RESULT IS THROWN AWAY,
        # at the pilot's instruction, 23 Oct.
        #
        # It used to return here without running. That is still what happens
        # to the ANSWER - nothing below min_alt_m reaches the fuser, which is
        # unchanged - but the work now happens anyway, for one reason: a
        # matcher that has never run is a matcher nobody can see. Until 60 m
        # there was no way to tell from the air whether it was alive, and on
        # flight 236 the first thing it ever did was stall the flow loop.
        #
        # The cost is small and measured on this machine: one match is 0 ms at
        # 20 m (the window is 24 px, too small to attempt), 81 ms at 40 m, and
        # 48 ms at 60 m. At a 2 s period that is about 4% of one core.
        #
        # The model load is NOT why this is here - _warm() handles that at
        # startup and does not depend on altitude or on the map covering the
        # take-off point.
        discard = altitude_m < self.min_alt_m
        if discard:
            self.skipped_low += 1
        if now - self._last_submit < self.fix_period_s:
            self.skipped_period += 1
            return False
        with self._lock:
            if self._job is not None:
                self.dropped += 1
                return False
            self._job = (img, prior_lon, prior_lat, altitude_m, yaw_deg,
                         float(roll_deg or 0.0), float(pitch_deg or 0.0),
                         None if att_age_s is None else float(att_age_s),
                         now, discard)
            self.submitted += 1
            self._last_submit = now
        self._wake.set()
        return True

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._job is not None

    def poll(self) -> FixResult | None:
        """Take the most recent finished fix, if there is one. Returns it once."""
        with self._lock:
            r, self._result = self._result, None
        return r

    def _radius_for(self, altitude_m: float, matcher: str | None = None) -> float:
        """Search window scaled to what the camera can actually see.

        It was a fixed 100 m until 4 Oct, which had the matcher guessing
        outside the frame's own footprint at low altitude and cropping the
        search too tight at high. 0.60 x altitude, floored so a window is
        never smaller than the query, and capped at radius_m so one bad
        altitude cannot ask for the whole map.
        """
        r = self.window_alt_frac * altitude_m
        if (matcher or self.matcher) == "ncc":
            # NCC SLIDES THE TEMPLATE INSIDE THE WINDOW, so the window has to
            # be bigger than the frame's own footprint or matchTemplate has
            # nowhere to search and every rotation is skipped. XFeat matches
            # features and is happy with partial overlap, which is why the
            # same radius rule serves one and starves the other: restored on
            # 25 Oct, NCC scored 0 of 40 on both flights until this was found,
            # with "template does not fit the search window" every time.
            #
            # The footprint is frame_width x altitude / focal; half of that
            # plus room to search is the floor. Measured on 30 frames at
            # 60-110 m, accept rate against radius: 150 m -> 97%, 200 -> 77%,
            # 260 -> 53%, so bigger is NOT better once it fits - a wider
            # window offers more wrong places to peak.
            half = 0.5 * self.frame_w_px * altitude_m / self.focal_px
            r = max(r, half + self.ncc_search_m)
        return float(np.clip(r, 40.0, self.radius_m))

    def _loop(self):
        self._warm()
        while self._run:
            self._wake.wait(0.2)
            self._wake.clear()
            with self._lock:
                job = self._job
            if job is None:
                continue
            img, plon, plat, alt, yaw, roll, pitch, att_age, t0, discard = job
            res = None
            got = {}      # matcher -> result, accepted only
            tried = {}    # matcher -> result, whatever it said
            try:
                if self.cache.covers(plon, plat):
                    for mname in self._pair:
                        # STOP ONCE A TRUSTED MATCHER HAS ANSWERED. This is
                        # what makes LighterGlue affordable: it costs 1482 ms
                        # against mutual NN's 520-670, and it is only needed
                        # on the frames mutual NN refuses. Running both every
                        # time measured a median 2587 ms, past the 2 s fix
                        # period; running LG only on refusals measured 1237 ms
                        # median over 338 frames.
                        #
                        # It does NOT apply to the NCC/XFeat pair, where the
                        # second answer is the disagreement number and the
                        # whole point is to have both. There, NCC is not
                        # trusted, so this never triggers.
                        if any(m in got for m in self._trusted_at(alt)):
                            break
                        # Each matcher gets the window IT needs: NCC slides
                        # the template inside the window and starves on
                        # XFeat's radius. See _radius_for.
                        win, wg = self.cache.window(
                            plon, plat, self._radius_for(alt, mname),
                            kind=self._window_kind)
                        ms, mm = self._gate[mname]
                        r_ = self._all_locate[mname](
                            img, win, wg, altitude_m=alt, focal_px=self.focal_px,
                            yaw_deg=yaw, map_gsd_m=self.cache.gsd,
                            map_gsd_x_m=getattr(self.cache, "gsd_x", None),
                            min_score=ms, min_margin=mm,
                            window_is_edges=(self._window_kind == "edges"),
                            principal_xy=self.principal_xy,
                            query_shape=self.query_shape)
                        if r_ is not None:
                            tried[mname] = r_
                        if r_ is not None and r_.ok:
                            got[mname] = r_
            except Exception:
                pass
            # WHOSE ANSWER IS PUBLISHED. See self._trust for the measurement.
            # Trust order first, so a trusted matcher that accepted wins over
            # one that did not; then, if no trusted matcher accepted, the best
            # available result is still carried through with ok=False so the
            # recorder gets a position and a reason - but it can never become
            # a fix, because live_node only folds in a result whose ok is set.
            trust_now = self._trusted_at(alt)
            res = next((got[m] for m in trust_now if m in got), None)
            if res is not None and res is got.get("ncc") and "ncc" not in self._trust:
                self.n_ncc_low += 1
            if res is None:
                untrusted = [m for m in got if m not in trust_now]
                if untrusted:
                    self.n_untrusted += 1
                res = next((tried[m] for m in trust_now if m in tried),
                           next(iter(tried.values())) if tried else None)
                if res is not None and res.ok:
                    # Can only happen if _trust and the loop disagree; make it
                    # loud rather than letting an untrusted answer through.
                    res.ok = False
                    res.reason = (res.reason or "") + " [not a trusted matcher]"
                if untrusted and res is not None:
                    # APPENDED, not substituted. The trusted matcher's own
                    # reason says why it declined, which is the diagnosis;
                    # this says a fix was available and was suppressed, which
                    # is what a falling fix rate has to be explainable by.
                    res.reason = ((res.reason + "; ") if res.reason else "") + (
                        f"{'/'.join(sorted(untrusted))} accepted, not trusted")
            # How far apart the two answers are, when both produced one. The
            # ONLY measured predictor of a bad fix: corr +0.928 against the
            # error, where score manages -0.10 and inlier count +0.20.
            # Logged, not enforced.
            # Both matchers get the SAME tilt shift, so it cancels out of
            # their disagreement entirely - this number is unaffected by the
            # correction above and still means what it meant.
            disagree = None
            if len(got) == 2:
                # Keyed off got, not _pair: _pair can hold three matchers now
                # ("ncc,xfeat,xfeat_lg"), and indexing _pair's names into got
                # would raise on whichever of them declined.
                a, b = (got[k] for k in self._pair if k in got)
                disagree = float(math.hypot(
                    (a.lat - b.lat) * 111320.0,
                    (a.lon - b.lon) * 111320.0
                    * math.cos(math.radians(a.lat))))
            # THE TILT CORRECTION. The matcher answers "the middle of this
            # frame is here"; the aircraft is nadir of that, not at it. Both
            # matchers report the same centre, so this is applied once, here,
            # rather than inside either of them.
            #
            # AND IT IS ONLY AS GOOD AS THE ATTITUDE IT IS GIVEN. roll and
            # pitch come from the last ATTITUDE message to arrive, so the
            # shift is computed from the aircraft's attitude at some earlier
            # moment than the frame. At 10 Hz that is 0-100 ms and measures as
            # free; a stalled stream is what hurts, and until att_age_s
            # existed nothing could see one. Beyond att_age_max_s the
            # correction is NOT applied: an uncorrected fix is wrong by
            # alt*tan(tilt) in a known direction, while one corrected from
            # attitude of unknown age is wrong by an unknown amount. The
            # reason is recorded either way.
            dN, dE = self.nadir_offset_m(alt, roll, pitch, yaw)
            att_stale = (att_age is not None and att_age > self.att_age_max_s)
            if att_stale:
                self.n_att_stale += 1
                dN = dE = 0.0
            self.last_tilt_shift_m = float(math.hypot(dN, dE))
            if res is not None and res.ok and (dN or dE):
                res.lat = res.lat - dN / 111320.0
                res.lon = res.lon - dE / (111320.0
                                          * math.cos(math.radians(res.lat)))
            out = None
            if res is not None:
                reason = getattr(res, "reason", "")
                if att_stale:
                    reason = ((reason + "; ") if reason else "") + (
                        f"attitude {att_age:.2f} s old, nadir correction "
                        f"withheld")
                out = FixResult(res.lon, res.lat, res.score, res.margin, res.ok,
                                time.monotonic() - t0, plon, plat,
                                disagree,
                                getattr(res, "inliers", 0),
                                getattr(res, "total_matches", 0),
                                reason,
                                # By keyword from here on: source and
                                # att_age_s sit between reason and
                                # altitude_m in the dataclass, so a
                                # positional altitude would land on source.
                                altitude_m=float(alt),
                                # tried, not got: a rejected answer still has
                                # to say which matcher it came from, or the
                                # recorder cannot tell "XFeat declined" from
                                # "XFeat never ran".
                                source=next((m for m, v in tried.items()
                                             if v is res), ""),
                                att_age_s=att_age)
            with self._lock:
                if discard:
                    # Ran to keep the path warm and visible. The answer does
                    # not go to the fuser - nothing below min_alt_m ever has.
                    self.warm_runs += 1
                else:
                    self._result = out
                    self.completed += 1
                self._job = None

    def _trusted_at(self, altitude_m):
        """Whose answer may be published at this height.

        One list, plus NCC appended below ncc_below_m. See that attribute for
        the measurement: NCC is not more accurate down there, it is the only
        thing available - 87% against XFeat's 48% at 70-82 m, with
        LighterGlue not asked at all below 90 m. Appended LAST, so XFeat is
        still believed wherever it speaks, which is what takes the worst case
        from 68.92 m to 45.81 m.
        """
        if (self.ncc_below_m > 0.0 and "ncc" in self._pair
                and "ncc" not in self._trust
                and altitude_m < self.ncc_below_m):
            return self._trust + ["ncc"]
        return self._trust

    def relocate(self, img, *, prior_lon, prior_lat, altitude_m, yaw_deg,
                 roll_deg=0.0, pitch_deg=0.0, search_m=250.0, tile_m=150.0,
                 step_m=75.0, ring_min_m=0.0, matcher="xfeat_lg",
                 budget_s=45.0):
        """Find this frame on the map when the estimate is no longer near it.

        THE WINDOW CANNOT JUST BE WIDENED. Measured 4 Nov, the prior moved D
        metres away in eight bearings with the window widened to contain the
        truth anyway:

            prior   window    n  | recovered  false fix  declined  cost
              0 m   528 px   12  |  17%        0%         83%      495 ms
             50 m   724 px   96  |   2%        0%         98%      649 ms
            100 m   920 px   96  |   0%        0%        100%      944 ms
            200 m  1312 px   96  |   0%        0%        100%     1685 ms
            500 m  2492 px   96  |   0%        0%        100%     5576 ms

        Nothing recovers past 100 m, and the D=0 control shows why: its
        window is already 528 px against the flown ~300, and that alone takes
        the accept rate from about 78% to 17%. Width is what kills it.

        So the region is TILED instead, at the size the matcher already works
        on, overlapping so nothing falls between tiles. On flight 478's own
        lost frames, 49 tiles each:

            lost 51.4 m   mutual NN found nothing   LighterGlue 8.28 m
            lost 102.1 m  mutual NN found nothing   LighterGlue 3.64 m
            lost 131.4 m  mutual NN found nothing   LighterGlue 5.24 m
            lost 115.2 m  mutual NN found nothing   LighterGlue found nothing

        Three of four, 51 to 131 m out, to 3.6-8.3 m. The fourth was at 58 m
        of altitude, where LighterGlue is measured weak. Mutual NN found none
        of four. Neither EVER named a wrong tile - false fix 0% here and 0%
        in the widening test above - which is what makes this safe to run at
        all: it goes quiet rather than inventing a position, so the failure
        the pilot named, being told "300 m that way" and then "800", cannot
        come from the search itself.

        Cost: 1482 ms a tile at top_k 1024, so a 7x7 square is about 30 s and
        a ring about 20. That is why this is not the routine path and why the
        aircraft must be holding while it runs. budget_s stops it rather than
        letting it run forever.

        Returns the best candidate as a FixResult with ok set, or None. It is
        a HYPOTHESIS, not a position: the caller must confirm it against a
        second answer after known motion before resetting anything. See
        RelocateCandidate for why one answer is not enough.
        """
        import numpy as _np
        locate = self._all_locate[matcher]
        ms, mm = _GATES[matcher]["min_score"], _GATES[matcher]["min_margin"]
        mlat = 111320.0
        mlon = 111320.0 * math.cos(math.radians(prior_lat))
        dN, dE = self.nadir_offset_m(altitude_m, roll_deg, pitch_deg, yaw_deg)
        grid = _np.arange(-search_m, search_m + 1e-6, step_m)
        t0 = time.monotonic()
        best = None
        self.reloc_tiles = 0
        for dn in grid:
            for de in grid:
                if time.monotonic() - t0 > budget_s:
                    self.reloc_timeout = True
                    break
                r = math.hypot(dn, de)
                if r > search_m or r < ring_min_m:
                    # A RING, not a square, when the caller knows roughly how
                    # far it has gone and only not where. Dead reckoning gives
                    # the distance to about 40% - net displacement over 20 s
                    # windows measured 0.907 of truth at the median, swinging
                    # 0.38 to 1.70 - so the plausible region is an annulus and
                    # the middle of it is already known to be wrong.
                    continue
                tlat = prior_lat + dn / mlat
                tlon = prior_lon + de / mlon
                if not self.cache.covers(tlon, tlat):
                    continue
                try:
                    win, wg = self.cache.window(tlon, tlat, tile_m / 2.0,
                                                kind=self._window_kind)
                    res = locate(img, win, wg, altitude_m=altitude_m,
                                 focal_px=self.focal_px, yaw_deg=yaw_deg,
                                 map_gsd_m=self.cache.gsd,
                                 map_gsd_x_m=getattr(self.cache, "gsd_x", None),
                                 min_score=ms, min_margin=mm,
                                 principal_xy=self.principal_xy,
                                 query_shape=self.query_shape)
                except Exception:
                    continue
                self.reloc_tiles += 1
                if res is None or not res.ok:
                    continue
                # The winner is the tile with the MOST agreeing points, not
                # the best score. The score does not predict the error:
                # measured corr(inlier ratio, error) = +0.052 over 275
                # would-be-accepted results, with the weakest quartile
                # holding the best median error of the four.
                if best is None or res.inliers > best.inliers:
                    res.lat = res.lat - dN / mlat
                    res.lon = res.lon - dE / (111320.0
                                              * math.cos(math.radians(res.lat)))
                    best = res
            else:
                continue
            break
        self.reloc_secs = time.monotonic() - t0
        if best is None:
            return None
        return FixResult(best.lon, best.lat, best.score, best.margin, True,
                         self.reloc_secs, prior_lon, prior_lat, None,
                         getattr(best, "inliers", 0),
                         getattr(best, "total_matches", 0),
                         f"relocated on {self.reloc_tiles} tiles in "
                         f"{self.reloc_secs:.0f} s",
                         altitude_m=float(altitude_m), source=matcher)

    def stop(self):
        self._run = False
        self._wake.set()
        # Long enough to outlast _warm(). The model load is 2.35 s warm and
        # 4.95 s cold, so a 2 s join returned while the thread was still
        # inside torch - and an abandoned torch thread aborts the interpreter
        # at exit ("terminate called without an active exception"). Seen
        # 25 Oct when a test built several workers in a row.
        self._thread.join(timeout=8.0)
