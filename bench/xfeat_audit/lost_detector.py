#!/usr/bin/env python3
"""Can the aircraft tell that it is lost, using only what it knows in flight?

    python3 lost_detector.py

Flight 478 declared 8.5 m of accuracy while being 124.7 m wrong, and the
declared sigma never exceeded 9.60 m on 4582 samples where the error was over
50 m. corr(sigma, error) is +0.940, so the direction is right and the
magnitude is useless. A relocalisation can never fire, because nothing ever
says the aircraft is lost.

Every candidate signal here is one the aircraft HAS at the moment of a fix
attempt - no truth, no hindsight:

    accept          did the trusted matcher accept this attempt
    rate_N          share of the last N attempts it accepted
    gap             attempts since the last acceptance
    secs            seconds since the last acceptance
    alt             altitude, because the accept rate also falls near the
                    60 m floor and on poor terrain, which is how a detector
                    on the accept rate alone false-alarms
    score           the matcher's own score on a DECLINED attempt - near the
                    gate means "almost", far below means "nothing there"

Scored two ways, because a detector has two failure modes and they trade off:

    DELAY       on flight 478, the error when the rule first fires. The
                runaway took the error 25 -> 124 m between t=400 and t=600,
                so firing at 25 m prevents most of the excursion.
    FALSE ALARM on flight 476, which never got lost - its prior error peaked
                at 26.4 m - so anything that fires there is a false alarm,
                and a false alarm costs the mission.
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "arbitrate.csv")
LOST_M = 25.0          # EK3_GLITCH_RAD: beyond this the FC itself objects


def load():
    rows = list(csv.DictReader(open(SRC)))
    for x in rows:
        for k in ("t", "alt", "prior_err", "xfeat_score", "disagree_m"):
            try:
                x[k] = float(x.get(k, ""))
            except Exception:
                x[k] = float("nan")
        x["acc"] = x.get("xfeat_ok") == "1"
    out = {}
    for x in rows:
        out.setdefault(x["run"], []).append(x)
    for run in out:
        out[run].sort(key=lambda q: q["t"])
    return out


def features(seq):
    """What the aircraft would know, attempt by attempt."""
    last_t, gap = None, 0
    hist = []
    for x in seq:
        x["gap"] = gap
        x["secs"] = float("nan") if last_t is None else x["t"] - last_t
        for N in (5, 10, 20):
            h = hist[-N:]
            x[f"rate{N}"] = (sum(h) / len(h)) if h else float("nan")
        hist.append(1 if x["acc"] else 0)
        if x["acc"]:
            last_t, gap = x["t"], 0
        else:
            gap += 1
    return seq


RULES = {
    "rate10 <= 0.3":            lambda x: x["rate10"] <= 0.30,
    "rate10 <= 0.2":            lambda x: x["rate10"] <= 0.20,
    "rate20 <= 0.3":            lambda x: x["rate20"] <= 0.30,
    "gap >= 5":                 lambda x: x["gap"] >= 5,
    "gap >= 8":                 lambda x: x["gap"] >= 8,
    "secs >= 20":               lambda x: x["secs"] >= 20,
    "secs >= 40":               lambda x: x["secs"] >= 40,
    # Altitude guards the two signals that the 60 m floor also depresses.
    "rate10<=0.3 & alt>75":     lambda x: x["rate10"] <= 0.30 and x["alt"] > 75,
    "gap>=5 & alt>75":          lambda x: x["gap"] >= 5 and x["alt"] > 75,
    "gap>=5 & alt>90":          lambda x: x["gap"] >= 5 and x["alt"] > 90,
    # "almost matched" vs "nothing there". A declined attempt whose score is
    # far below the gate means the window holds no recognisable ground, which
    # is a different statement from a marginal match.
    "gap>=4 & score<0.04":      lambda x: x["gap"] >= 4 and x["xfeat_score"] < 0.04,
    "gap>=5 & score<0.05 & alt>75":
        lambda x: (x["gap"] >= 5 and x["xfeat_score"] < 0.05 and x["alt"] > 75),
}


def main():
    data = load()
    for run in data:
        features(data[run])

    print("what the flights actually did")
    for run, seq in data.items():
        pe = np.array([x["prior_err"] for x in seq])
        lost = pe > LOST_M
        first = next((x["t"] for x in seq if x["prior_err"] > LOST_M), None)
        print("  %-10s %3d attempts | prior err p50 %6.1f max %6.1f | "
              "over %.0f m on %3d of them%s" % (
                  run, len(seq), np.median(pe), pe.max(), LOST_M, lost.sum(),
                  f", first at t={first:.0f} s" if first else " - never lost"))

    print(f"\n{'=' * 92}")
    print("  %-30s %-34s %s" % ("rule", "FLIGHT 478 (got lost)",
                                "FLIGHT 476 (did not)"))
    print("  %-30s %-34s %s" % ("", "fires at t= / error then / late by",
                                "false alarms"))
    print(f"{'=' * 92}")
    best = []
    for name, fn in RULES.items():
        line = "  %-30s" % name
        verdict = {}
        for run in ("vps_00478", "vps_00476"):
            seq = data[run]
            fires = [x for x in seq if _safe(fn, x)]
            truly = next((x["t"] for x in seq if x["prior_err"] > LOST_M), None)
            if run == "vps_00478":
                if not fires:
                    line += " %-34s" % "NEVER FIRES"
                    verdict["delay"] = None
                else:
                    f0 = fires[0]
                    late = (f0["t"] - truly) if truly else float("nan")
                    line += " t=%3.0f  err %5.1f m  %+5.0f s     " % (
                        f0["t"], f0["prior_err"], late)
                    verdict["delay"] = f0["prior_err"]
            else:
                # Anything at all on 476 is a false alarm: it never got lost.
                fa = [x for x in fires if x["prior_err"] <= LOST_M]
                line += " %3d of %3d attempts" % (len(fa), len(seq))
                verdict["fa"] = len(fa)
        print(line)
        if verdict.get("delay") is not None:
            best.append((verdict["fa"], verdict["delay"], name))
    best.sort()
    print(f"\n  ranked on false alarms first, then on how much error it lets "
          f"through:")
    for fa, d, name in best[:5]:
        print("    %-32s %2d false alarms, fires at %5.1f m of error"
              % (name, fa, d))
    print("\n  A rule that fires at 25-30 m of error with no false alarm is "
          "what is wanted:\n  25 m is where the FC itself starts objecting, "
          "and the runaway is after that.")


def _safe(fn, x):
    try:
        v = fn(x)
    except Exception:
        return False
    return bool(v) and v == v


if __name__ == "__main__":
    main()
