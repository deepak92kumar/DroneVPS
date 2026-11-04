#!/usr/bin/env python3
"""Judge LighterGlue on the error, not on the match count.

The pilot's own constraint for this whole audit: "Do not claim that XFeat is
fixed merely because the number of matches increases." LighterGlue changes
BOTH sides of the inlier ratio - it returns fewer candidate matches and more
of them agree - so the ratio rising says nothing on its own. What matters:

  coverage   frames each one turns into a fix
  accuracy   the error on the fixes it publishes, and the WORST of them
  overlap    what each does on the frames the other refuses - the only place
             a second matcher can actually add anything
"""
import csv
import os
import sys

import numpy as np

P = "/data/DRONE/bench/xfeat_audit/lighterglue.csv"
rows = list(csv.DictReader(open(P)))
for x in rows:
    for k in list(x):
        if k in ("run", "mnn_why", "lg_why"):
            continue
        try:
            x[k] = float(x[k])
        except Exception:
            x[k] = float("nan")
    x["mok"] = x.get("mnn_ok") == 1
    x["lok"] = x.get("lg_ok") == 1


def stats(s, tag):
    e = np.array([x[f"{tag}_err"] for x in s
                  if x.get(f"{tag}_ok") == 1 and np.isfinite(x[f"{tag}_err"])])
    return e


print(f"{len(rows)} frames, two flights\n")
print("  %-22s %-24s %-30s %s" % ("", "coverage", "error of what it published",
                                  "cost"))
for tag, name in (("mnn", "XFeat + mutual NN"), ("lg", "XFeat + LighterGlue")):
    e = stats(rows, tag)
    n_tot = np.nanmedian([x[f"{tag}_n"] for x in rows])
    n_in = np.nanmedian([x[f"{tag}_in"] for x in rows])
    ms = np.nanmedian([x[f"{tag}_ms"] for x in rows])
    print("  %-22s %4d/%4d (%3.0f%%)        p50 %5.2f  p90 %5.2f  WORST %6.2f"
          "   >20 m %2d   %5.0f ms" % (
              name, len(e), len(rows), 100 * len(e) / len(rows),
              np.median(e) if len(e) else np.nan,
              np.percentile(e, 90) if len(e) else np.nan,
              e.max() if len(e) else np.nan,
              int((e > 20).sum()) if len(e) else 0, ms))
    print("  %-22s matches p50 %4.0f   inliers p50 %4.0f   ratio %.3f"
          % ("", n_tot, n_in, n_in / max(n_tot, 1)))

print("\nwhere they differ - the only place a second matcher earns its place")
both = [x for x in rows if x["mok"] and x["lok"]]
monly = [x for x in rows if x["mok"] and not x["lok"]]
lonly = [x for x in rows if x["lok"] and not x["mok"]]
neither = [x for x in rows if not x["mok"] and not x["lok"]]
print("  both accepted    %3d | MNN err p50 %5.2f max %6.2f | LG err p50 %5.2f max %6.2f"
      % (len(both),
         np.median([x["mnn_err"] for x in both]) if both else np.nan,
         max([x["mnn_err"] for x in both]) if both else np.nan,
         np.median([x["lg_err"] for x in both]) if both else np.nan,
         max([x["lg_err"] for x in both]) if both else np.nan))
for nm, s, tag in (("only MNN", monly, "mnn"), ("only LG", lonly, "lg")):
    e = [x[f"{tag}_err"] for x in s if np.isfinite(x[f"{tag}_err"])]
    print("  %-16s %3d | err p50 %5.2f  max %6.2f" % (
        nm, len(s), np.median(e) if e else np.nan, max(e) if e else np.nan))
print("  neither          %3d" % len(neither))
if both:
    w = sum(1 for x in both if x["lg_err"] < x["mnn_err"])
    print("  head to head on the %d both accepted: LG closer %d (%.0f%%)"
          % (len(both), w, 100 * w / len(both)))

print("\nthe union - what trusting either would publish, worst case first")
for nm, pick in (
        ("MNN only", lambda x: x["mnn_err"] if x["mok"] else None),
        ("LG only", lambda x: x["lg_err"] if x["lok"] else None),
        ("LG, else MNN", lambda x: (x["lg_err"] if x["lok"]
                                    else (x["mnn_err"] if x["mok"] else None))),
        ("MNN, else LG", lambda x: (x["mnn_err"] if x["mok"]
                                    else (x["lg_err"] if x["lok"] else None))),
        ("both must agree",
         lambda x: (min(x["lg_err"], x["mnn_err"])
                    if (x["mok"] and x["lok"]
                        and abs(x["lg_err"] - x["mnn_err"]) < 10) else None)),
):
    e = [pick(x) for x in rows]
    e = np.array([v for v in e if v is not None and np.isfinite(v)])
    if not len(e):
        continue
    print("  %-18s fix %4d/%4d (%3.0f%%) | p50 %5.2f p90 %5.2f WORST %6.2f"
          " | >20 m %2d" % (nm, len(e), len(rows), 100 * len(e) / len(rows),
                            np.median(e), np.percentile(e, 90), e.max(),
                            int((e > 20).sum())))

print("\nby altitude")
for lo, hi in ((55, 85), (85, 100), (100, 118)):
    s = [x for x in rows if lo <= x["alt"] < hi]
    if not s:
        continue
    line = "  alt %3d-%3d m n=%3d |" % (lo, hi, len(s))
    for tag in ("mnn", "lg"):
        e = stats(s, tag)
        line += "  %-3s %3.0f%% / p50 %5s / worst %6s" % (
            tag, 100 * len(e) / len(s),
            "%.2f" % np.median(e) if len(e) else "-",
            "%.2f" % e.max() if len(e) else "-")
    print(line)
