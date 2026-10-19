#!/usr/bin/env python3
"""Spike filter for the TF03, in both directions.

The pilot's report: "tf03 bich bich mein jump bhi karta hai... uppar aur niche
bhi". Log 214 shows it on a stationary aircraft - RFND ranges 0.05 to 0.77 m
with a median of 0.41, which is 70 cm of spread on a drone that never left the
floor.

There was already a median of 5 in cuav_mavlink_bridge. A median only rejects a
spike that is in the minority of its window, so three bad samples in a row walk
straight through it, and it has no notion of how fast the ground can actually
approach. This adds the missing half.

Two gates, in order:

  1. RANGE. NaN, infinity, and anything outside [min_m, max_m] never enters the
     window at all. A 4-20 mA sensor reads something even when it has nothing.

  2. RATE, with confirmation. A reading that implies the ground moved faster
     than max_rate_mps is not accepted on the spot - but it is not discarded
     for ever either, because flying over a wall IS a real step change. It has
     to repeat consistently confirm_n times before the filter concedes the
     level has genuinely moved. Until then the last accepted value is HELD.

Holding, not interpolating. A held value is the last thing that was actually
measured; an extrapolated one is a guess that looks like a measurement.

Spikes are counted up and down separately, because they do not mean the same
thing: a jump DOWN is usually a return off something closer than the ground
(a leg, a wire, dust), and a jump UP is usually a dropout reading as max range.
"""
from __future__ import annotations

import math
import statistics
from collections import deque


class LidarFilter:
    def __init__(self, window: int = 5, max_rate_mps: float = 8.0,
                 confirm_n: int = 3, min_m: float = 0.05, max_m: float = 180.0,
                 confirm_tol_m: float = 0.5, spike_report_m: float = 0.5):
        self.window = int(window)
        self.max_rate = float(max_rate_mps)
        self.confirm_n = int(confirm_n)
        self.min_m, self.max_m = float(min_m), float(max_m)
        self.confirm_tol = float(confirm_tol_m)
        self.spike_report_m = float(spike_report_m)
        self._hist: deque[float] = deque(maxlen=self.window)
        self._pending: list[float] = []
        self.value: float | None = None
        self._t: float | None = None
        self.n_in = self.n_out_of_range = 0
        self.spikes_up = self.spikes_down = 0
        self.rate_refusals = 0
        self.held = 0
        self.steps = 0

    def push(self, t: float, raw_m) -> float | None:
        """One reading in, the accepted height out (or None before the first).

        The returned value is always something the sensor actually reported at
        some point - never an average with a rejected sample in it, and never
        an extrapolation.
        """
        self.n_in += 1
        try:
            v = float(raw_m)
        except (TypeError, ValueError):
            self.n_out_of_range += 1
            return self.value
        if not math.isfinite(v) or not (self.min_m <= v <= self.max_m):
            self.n_out_of_range += 1
            return self.value

        # Count the spike on the RAW sample, before the median hides it. A
        # median of 5 swallows a single outlier completely - which is what it
        # is for - so the rate gate below never sees one and would report a
        # sensor that jumps every second as perfectly calm. The pilot asked
        # about the jumps, so the jumps have to be counted where they happen.
        if self.value is not None:
            dev = v - self.value
            if abs(dev) > self.spike_report_m:
                if dev > 0:
                    self.spikes_up += 1
                else:
                    self.spikes_down += 1

        self._hist.append(v)
        med = float(statistics.median(self._hist))

        if self.value is None:                 # first real reading defines the level
            self.value, self._t = med, t
            return self.value

        dt = max(1e-3, t - (self._t if self._t is not None else t))
        allowed = self.max_rate * dt
        delta = med - self.value

        if abs(delta) <= allowed:
            self.value, self._t = med, t
            self._pending.clear()
            return self.value

        # Too fast to be the ground. Could still be a real step - a wall, a
        # roof, a vehicle - so make it prove itself before believing it.
        self.rate_refusals += 1
        if self._pending and abs(med - self._pending[-1]) > self.confirm_tol:
            self._pending.clear()              # not even consistent with itself
        self._pending.append(med)
        if len(self._pending) >= self.confirm_n:
            self.value = float(statistics.median(self._pending))
            self._t = t
            self._pending.clear()
            self.steps += 1
            return self.value

        self.held += 1
        self._t = t
        return self.value                      # HOLD the last measured value

    def stats(self) -> dict:
        return dict(n_in=self.n_in, out_of_range=self.n_out_of_range,
                    spikes_up=self.spikes_up, spikes_down=self.spikes_down,
                    rate_refusals=self.rate_refusals,
                    held=self.held, steps=self.steps, value=self.value)
