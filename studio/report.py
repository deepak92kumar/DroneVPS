#!/usr/bin/env python3
"""The review as a file, so a flight can be filed without a browser open.

Same findings, same wording, in the shape this project's own handoffs use -
headline first, then how it was measured, then the flight that set the
threshold. It is written to be read a month later by someone who was not
there, which is the test every report in `reports/` had to pass.
"""
from __future__ import annotations

import os
import time

import analysis

MARK = {"alarm": "ALARM", "warn": "WARN ", "ok": "OK   ", "info": "INFO "}


def name(b):
    bits = []
    if b.get("fc"):
        bits.append(os.path.basename(b["fc"]["path"]).replace(".BIN", ""))
    if b.get("node"):
        bits.append(b["node"]["name"])
    return "review_" + "_".join(bits or ["flight"]) + ".md"


def markdown(b):
    s = analysis.summary(b)
    fs = analysis.findings(b)
    L = []
    L.append(f"# Flight review - {', '.join(filter(None, [s.get('fc'), s.get('node')]))}")
    L.append("")
    L.append(f"Written by DroneVPS Studio on "
             f"{time.strftime('%d %B %Y %H:%M')}. Every threshold below comes "
             f"from a flight in this repository and the flight is named with "
             f"it; nothing here is a default from somewhere else.")
    L.append("")
    L.append("## The flight")
    L.append("")
    if s.get("node"):
        L.append(f"* `{s['node']}` - {s['rows']} rows, {s['duration_s']:.0f} s, "
                 f"{s['vps_s']:.0f} s of it on the VPS")
        L.append(f"* height: median {s['alt_med']:.1f} m, max {s['alt_max']:.1f} m")
        if s.get("pos_p50") is not None:
            L.append(f"* position against the raw receiver: p50 "
                     f"{s['pos_p50']:.2f} m")
        L.append(f"* map fixes attempted: {s.get('fixes', 0)}")
    if s.get("fc"):
        L.append(f"* `{s['fc']}` - modes {', '.join(s.get('modes') or []) or 'none logged'}"
                 f"; {s.get('althold', 0)} drop(s) to ALT_HOLD")
    if s.get("align"):
        a = s["align"]
        L.append(f"* clocks: node t + {a['lag_s']:.0f} s = flight controller t "
                 f"(correlation {a['corr']:.4f})")
    L.append("")

    for level in ("alarm", "warn", "ok", "info"):
        group = [f for f in fs if f["level"] == level]
        if not group:
            continue
        heading = {"alarm": "What is wrong", "warn": "What to watch",
                   "ok": "What is working", "info": "For the record"}[level]
        L.append(f"## {heading}")
        L.append("")
        for f in group:
            L.append(f"### {f['title']}")
            L.append("")
            L.append(f"**{f['headline']}**")
            L.append("")
            L.append(f["detail"])
            if f.get("basis"):
                L.append("")
                L.append(f"*Threshold from: {f['basis']}*")
            if f.get("rows"):
                L.append("")
                head = f["rows"][0]
                L.append("| " + " | ".join(head) + " |")
                L.append("|" + "|".join(["---"] * len(head)) + "|")
                for r in f["rows"][1:25]:
                    L.append("| " + " | ".join(r) + " |")
                if len(f["rows"]) > 25:
                    L.append(f"")
                    L.append(f"*{len(f['rows']) - 25} further rows omitted.*")
            L.append("")
    return "\n".join(L)
