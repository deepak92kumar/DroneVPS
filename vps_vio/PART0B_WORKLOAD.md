# Part 0b — workload audit (27 Sept 2026)

Measured on the Orin Nano with real flight frames from `auto_20260927_181924`
(117 m, 30 Hz) and the 2.33 × 2.00 km site map. Every number here is measured,
not estimated.

## Where each stage runs, and what it costs

| Stage | Runs on | Cost per call | At flight rate | GPU version? | Verdict |
|---|---|---|---|---|---|
| Camera capture + ISP | OAK-D Myriad-X VPU | free to us | — | already off-CPU | leave it |
| JPEG encode (recording) | Jetson CPU | 8.0 ms | 24 % of a core @ 30 Hz | NVENC / nvjpeg exists | not worth it yet |
| Corner detection | Jetson CPU (GFTT) | 30 ms | was every frame | VPI Harris — **rejected** | reuse corners instead |
| Sparse LK optical flow | Jetson CPU | 24–48 ms | dominant cost | VPI CUDA PyrLK — **works** | worth taking later |
| Flow step (corners+LK+RANSAC) | Jetson CPU | **24.4 ms** (was 52.7) | 73 % of a core @ 30 Hz | partial | fixed on CPU |
| Map fix (±150 m, 5 rotations) | Jetson CPU | **68 ms** | 7 % of a core @ 1 Hz | VPI CUDA — **rejected** | fine on CPU |
| cuVSLAM VIO | Jetson **GPU** | 15.1 ms | 29.9 Hz stereo | already GPU | already right |

## What the measurements changed

**Corner detection was the biggest waste.** Re-detecting every frame cost 30 ms
of a 53 ms flow step, and it bought nothing: tracked across real flight frames,
707 of 714 corners were still alive after 59 frames — two seconds at 30 Hz.
`FlowTracker` seeds once and re-seeds only when the population thins. Flow step:
**52.7 ms → 24.4 ms**, with drift unchanged (median 13.8 m → 14.4 m over 2618
frames, final 13.1 m → 11.8 m).

**Map matching was 97 % matchTemplate**, not GDAL and not features. Coarse-to-fine
correlation (half resolution, then a 12-px box at full resolution) returns the
identical peak — zero pixels of difference — for a quarter of the time, and a
±150 m window matches as accurately as ±300 m. **1291 ms → 68 ms per fix.**

## GPU offloads: what was tried and rejected

The plan named optical flow and map-matching pre-processing as the candidates.
Both were measured. Only one of the four is worth taking, and it is not the one
that mattered most.

| Candidate | Measured | Kept? |
|---|---|---|
| VPI CUDA `templateMatching` | 68 ms vs CPU pyramid's 61 ms (±300 m); 15.8 vs 10.0 (±150 m) | **No** — the CPU pyramid is faster |
| VPI CUDA Harris corners | returned **3** corners where OpenCV GFTT returned 714 | **No** — not usable as configured |
| VPI CUDA PyrLK optical flow | 16.6 ms vs OpenCV's ~24 ms forward, agreeing to 0.23 px median | **Later** — real but smaller than the algorithmic fix |
| OpenCV CUDA anything | `cv2.cuda.getCudaEnabledDeviceCount()` = **0** | **No** — JetPack's OpenCV has no CUDA |

`GR3D_FREQ` reads 56–58 % on an idle desktop, so a GPU figure only means
something as a delta against that baseline. None of this pipeline moved it.

## Sustained run, everything together

25–30 s with flow, JPEG encode and a map fix every second, on the current
(desktop-loaded) machine:

| Target flow rate | Achieved | Overruns | CPU total | Verdict |
|---|---|---|---|---|
| 30 Hz | 18.0 Hz | 526 / 540 | 463 % of 600 % | does not fit single-threaded |
| 15 Hz | 14.2 Hz | 38 / 354 | 394 % | fits |
| **10 Hz** | **9.8 Hz** | **24 / 244** | **339 %** | **comfortable** |

Idle baseline on this machine is about 150 % CPU (GNOME desktop), so the
pipeline's own share at 10 Hz is roughly 190 % spread over 6 cores.

The remaining overruns are the map-fix frames: a 68 ms fix blocks a 100 ms
budget on a single thread. Moving the fix to its own thread would remove them,
and there are six cores. That is the next offload — to another **core**, not to
the GPU.

## Does 30 Hz matter?

No. Subsampling the same flight: median drift is 21.2 m at 2 Hz, 19.7 at 6,
19.6 at 10, 18.1 at 30. The endpoint improves sharply up to about 6 Hz and the
median barely moves after. **10 Hz flow with a 1 Hz map fix is the configuration
to fly**, and it fits on one core with room to spare.
