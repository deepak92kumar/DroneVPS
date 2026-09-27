#!/usr/bin/env python3
"""Part 0b - measure every stage of the flight workload, then run them together.

The plan asks for a table of where each piece runs, what it costs, and whether
a GPU version helps. Stage costs are measured on real flight frames so the
numbers mean something; the combined run then holds the whole pipeline at
flight rate while tegrastats watches CPU, GPU, RAM and temperature.
"""
from __future__ import annotations
import argparse, csv, io, os, subprocess, sys, threading, time
import numpy as np, cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flow_odom import FlowTracker, frame_shift_px, shift_to_ne, attitude_shift_px
from map_cache import MapCache
from map_match_ncc import locate

FLIGHT = "/mnt/ssd/DRONE/flights/auto_20260927_181924"
MAP = "/home/abc/SSD/DRONE/ap/site_2km.tif"
FOCAL = 791.72


def load(n=80):
    raw = open(os.path.join(FLIGHT, "index.csv"), "rb").read().replace(b"\x00", b"")
    rows = [r for r in csv.DictReader(io.StringIO(raw.decode("utf8", "replace")))
            if r.get("alt_rel_m") and r.get("lat") not in (None, "", "0.0")
            and r.get("yaw_deg") and r.get("roll_deg")
            and float(r["alt_rel_m"]) > 80]
    rows = rows[:n]
    return [(cv2.imread(os.path.join(FLIGHT, "mono", r["mono_file"]), cv2.IMREAD_GRAYSCALE), r)
            for r in rows]


class Tegra:
    """Sample tegrastats while something runs."""

    def __init__(self):
        self.lines = []
        self.p = None

    def __enter__(self):
        try:
            self.p = subprocess.Popen(["tegrastats", "--interval", "500"],
                                      stdout=subprocess.PIPE, text=True)
            threading.Thread(target=self._read, daemon=True).start()
        except Exception as e:
            print(f"  (tegrastats nahi chala: {e})")
        return self

    def _read(self):
        for line in self.p.stdout:
            self.lines.append(line)

    def __exit__(self, *a):
        if self.p:
            self.p.terminate()

    def summary(self):
        cpu, gpu, ram, temp = [], [], [], []
        for ln in self.lines:
            for tok in ln.split():
                if tok.startswith("CPU"):
                    inner = ln.split("CPU [")[1].split("]")[0] if "CPU [" in ln else ""
                    vals = [c.split("%")[0] for c in inner.split(",") if "%" in c]
                    if vals:
                        try: cpu.append([float(v) for v in vals])
                        except ValueError: pass
            if "GR3D_FREQ" in ln:
                try: gpu.append(float(ln.split("GR3D_FREQ")[1].split("%")[0].strip()))
                except (ValueError, IndexError): pass
            if "RAM" in ln:
                try: ram.append(float(ln.split("RAM ")[1].split("/")[0]))
                except (ValueError, IndexError): pass
            if "cpu@" in ln:
                try: temp.append(float(ln.split("cpu@")[1].split("C")[0]))
                except (ValueError, IndexError): pass
        out = {}
        if cpu:
            a = np.array(cpu)
            out["cpu_total_pct"] = float(a.sum(axis=1).mean())
            out["cpu_cores"] = a.shape[1]
            out["cpu_per_core_max"] = float(a.mean(axis=0).max())
        if gpu: out["gpu_pct"] = float(np.mean(gpu))
        if ram: out["ram_mb"] = float(np.mean(ram))
        if temp: out["temp_c"] = float(np.mean(temp))
        out["samples"] = len(self.lines)
        return out


def timeit(fn, n=20):
    fn()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--flow-hz", type=float, default=30)
    ap.add_argument("--fix-hz", type=float, default=1.0)
    a = ap.parse_args()

    print("frames load kar raha hoon...", flush=True)
    data = load()
    M = MapCache(MAP)
    print(f"{len(data)} frames, {M.summary()}\n")

    att = lambda r: (float(r["roll_deg"]), float(r["pitch_deg"]), float(r["yaw_deg"]))
    (i0, r0), (i1, r1) = data[0], data[1]
    lat, lon = float(r1["lat"]), float(r1["lon"])
    alt, yaw = float(r1["alt_rel_m"]), float(r1["yaw_deg"])
    win, wg = M.window(lon, lat, 150)
    _tr = FlowTracker(); _tr.step(i0)

    print("=== STAGE COSTS (ek frame ka kaam) ===")
    stages = [
        ("JPEG encode 1280x800 (recording)",
         lambda: cv2.imencode(".jpg", i1, [cv2.IMWRITE_JPEG_QUALITY, 92])),
        ("flow OLD: corners har frame",
         lambda: frame_shift_px(i0, i1)),
        ("flow NEW: FlowTracker (reuse)",
         lambda: _tr.step(i1)),
        ("map fix: +-150 m, 5 rotations",
         lambda: locate(i1, win, wg, altitude_m=alt, focal_px=FOCAL, yaw_deg=yaw,
                        map_gsd_m=M.gsd, window_is_edges=True)),
    ]
    costs = {}
    for name, fn in stages:
        ms = timeit(fn, 10 if "map" in name else 20)
        costs[name] = ms
        print(f"  {name:38s} {ms:8.1f} ms")

    budget_flow = costs["flow NEW: FlowTracker (reuse)"] * a.flow_hz / 10.0
    budget_jpeg = costs["JPEG encode 1280x800 (recording)"] * a.flow_hz / 10.0
    budget_fix = costs["map fix: +-150 m, 5 rotations"] * a.fix_hz / 10.0
    print(f"\n  ek core ka kitna hissa (%):")
    print(f"    flow @ {a.flow_hz:.0f} Hz        {budget_flow:6.1f} %")
    print(f"    JPEG @ {a.flow_hz:.0f} Hz        {budget_jpeg:6.1f} %")
    print(f"    map fix @ {a.fix_hz:.1f} Hz     {budget_fix:6.1f} %")
    print(f"    ------------------------------")
    print(f"    TOTAL              {budget_flow+budget_jpeg+budget_fix:6.1f} % of ONE core")

    print(f"\n=== SAB EK SAATH, {a.seconds:.0f} s ===", flush=True)
    stop = time.time() + a.seconds
    n_flow = n_fix = 0
    next_fix = time.time()
    interval = 1.0 / a.flow_hz
    tracker = FlowTracker()
    lat_flow = []
    with Tegra() as teg:
        t_next = time.time()
        k = 0
        prev_r = None
        while time.time() < stop:
            img, r = data[k % len(data)]
            k += 1
            t0 = time.perf_counter()
            d, nn = tracker.step(img)
            if d is not None and prev_r is not None:
                d = d - attitude_shift_px(att(prev_r), att(r))
                shift_to_ne(d, altitude_m=float(r["alt_rel_m"]), focal_px=FOCAL,
                            yaw_deg=float(r["yaw_deg"]))
            n_flow += 1
            cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            prev_r = r
            if time.time() >= next_fix:
                locate(img, win, wg, altitude_m=float(r["alt_rel_m"]), focal_px=FOCAL,
                       yaw_deg=float(r["yaw_deg"]), map_gsd_m=M.gsd, window_is_edges=True)
                n_fix += 1
                next_fix = time.time() + 1.0 / a.fix_hz
            lat_flow.append((time.perf_counter() - t0) * 1000)
            t_next += interval
            slack = t_next - time.time()
            if slack > 0:
                time.sleep(slack)
            else:
                t_next = time.time()
        s = teg.summary()
    lat = np.array(lat_flow)
    print(f"  flow steps  : {n_flow}  ({n_flow/a.seconds:.1f} Hz achieved, target {a.flow_hz:.0f})")
    print(f"  map fixes   : {n_fix}   ({n_fix/a.seconds:.2f} Hz)")
    print(f"  per-frame   : median {np.median(lat):.1f} ms, p95 {np.percentile(lat,95):.1f} ms, "
          f"budget {1000/a.flow_hz:.1f} ms")
    print(f"  overruns    : {int((lat > 1000/a.flow_hz).sum())} / {len(lat)}")
    for k2, v in s.items():
        print(f"  {k2:16s} {v}")


if __name__ == "__main__":
    main()
