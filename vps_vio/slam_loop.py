#!/usr/bin/env python3
"""Loop-closure test: does revisiting a place pull the drift back?

This is the free half of Part 3. Satellite map matching needs licensed imagery
(1 m/px or better, which is a procurement problem), but the mission profile is
out-and-back: the return leg flies over the outbound path. SLAM loop closure
recognises those places and corrects the accumulated drift — no imagery, no
licence, no cost, and cuVSLAM already implements it.

What it measures:
  * whether loop closure actually fires (lc_status / get_loop_closure_poses)
  * odometry return error   — where track() thinks we ended up
  * SLAM return error       — where the optimised pose graph says we ended up
The second should be much smaller than the first if loop closure is working.

Honest limit: this bounds error on the RETURN leg, not at the far point of the
outbound leg. If the mission needs accuracy AT the target, a map is still required.

Protocol (camera needs a textured scene ~1 m away, as always):
    still -> move away ~1 m (or walk it around) -> come back to the SAME spot -> still
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import depthai as dai            # noqa: E402
import cuvslam as vslam          # noqa: E402
import oak_rig                   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--inertial", action="store_true")
    ap.add_argument("--save-map", default="", help="folder to save the SLAM map into")
    ap.add_argument("--csv", default=os.path.join(os.path.dirname(__file__), "logs", "slam_loop.csv"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.csv), exist_ok=True)

    vslam.warm_up_gpu()
    with dai.Device() as dev:
        calib = dev.readCalibration()
        rig = oak_rig.build_rig(calib, with_imu=a.inertial)
        tracker = oak_rig.make_tracker(rig, inertial=a.inertial,
                                       slam_config=oak_rig.make_slam_config())
        print(f"SLAM tracker ready (inertial={a.inertial}, GPU)")
        dev.startPipeline(oak_rig.build_pipeline(with_imu=a.inertial))
        qL = dev.getOutputQueue("left", 8, False)
        qR = dev.getOutputQueue("right", 8, False)
        qI = dev.getOutputQueue("imu", 300, False) if a.inertial else None

        bufL, bufR, pend = {}, {}, []
        last_imu = None
        T, P = [], []
        lc_seen, lc_status_hist = 0, {}
        gpu_ms, nframes = 0.0, 0
        t0 = time.monotonic()
        while time.monotonic() - t0 < a.seconds:
            if qI is not None:
                while qI.has():
                    for pk in qI.get().packets:
                        acc, gyr = pk.acceleroMeter, pk.gyroscope
                        pend.append((oak_rig.td_ns(acc.getTimestampDevice()),
                                     [acc.x, acc.y, acc.z], [gyr.x, gyr.y, gyr.z]))
            while qL.has():
                f = qL.get(); bufL[f.getSequenceNum()] = f
            while qR.has():
                f = qR.get(); bufR[f.getSequenceNum()] = f
            for seq in sorted(set(bufL) & set(bufR)):
                fl, fr = bufL.pop(seq), bufR.pop(seq)
                ts = oak_rig.td_ns(fl.getTimestampDevice())
                if a.inertial:
                    while pend and pend[0][0] <= ts:
                        its, la, av = pend.pop(0)
                        if last_imu is not None and its < last_imu:
                            continue
                        m = vslam.ImuMeasurement()
                        m.timestamp_ns = its; m.linear_accelerations = la
                        m.angular_velocities = av
                        tracker.register_imu_measurement(0, m); last_imu = its
                tc = time.monotonic()
                est, _ = tracker.track(ts, [fl.getCvFrame(), fr.getCvFrame()])
                gpu_ms += (time.monotonic() - tc) * 1e3; nframes += 1
                q = getattr(est, "world_from_rig", None) if est is not None else None
                q = getattr(q, "pose", q)
                if q is not None and np.all(np.isfinite(q.translation)):
                    T.append(time.monotonic() - t0)
                    P.append(np.array(q.translation))
                # loop-closure telemetry
                mt = tracker.get_slam_metrics()
                if mt is not None:
                    st = str(getattr(mt, "lc_status", "?"))
                    lc_status_hist[st] = lc_status_hist.get(st, 0) + 1
                lcp = tracker.get_loop_closure_poses()
                if lcp is not None and len(lcp) > lc_seen:
                    print(f"  [t={time.monotonic()-t0:5.1f}s] LOOP CLOSURE #{len(lcp)}")
                    lc_seen = len(lcp)
            time.sleep(0.002)

        # --- results -------------------------------------------------------
        P = np.stack(P); T = np.array(T)
        odo_start, odo_end = P[:30].mean(0), P[-30:].mean(0)
        odo_err = float(np.linalg.norm(odo_end - odo_start))
        far = float(np.linalg.norm(P - P[:30].mean(0), axis=1).max())

        slam = tracker.get_all_slam_poses(0)
        slam_err = None
        if slam:
            S = np.array([[p.pose.translation[0], p.pose.translation[1],
                           p.pose.translation[2]] for p in slam])
            slam_err = float(np.linalg.norm(S[-30:].mean(0) - S[:30].mean(0)))

        np.savetxt(a.csv, np.column_stack([T, P]), delimiter=",",
                   header="t,x,y,z", comments="")
        print("\n---- loop closure result ----")
        print(f"frames {nframes}  ({nframes/max(T[-1],1e-6):.1f} Hz)  "
              f"GPU {gpu_ms/max(nframes,1):.1f} ms/frame")
        print(f"furthest excursion from start : {far:.3f} m")
        print(f"ODOMETRY return error         : {odo_err:.3f} m")
        if slam_err is not None:
            print(f"SLAM     return error         : {slam_err:.3f} m   ({len(slam)} optimised poses)")
        print(f"loop closures detected        : {lc_seen}")
        print(f"lc_status histogram           : {lc_status_hist}")
        if lc_seen == 0:
            print("\nNo loop closure fired. Either the return did not revisit the same view,")
            print("or the excursion was too small to be a meaningful loop.")
        if a.save_map:
            done = {}
            tracker.save_map(a.save_map, lambda ok: done.setdefault("ok", ok))
            t = time.monotonic()
            while "ok" not in done and time.monotonic() - t < 30:
                time.sleep(0.2)
            print(f"map saved to {a.save_map}: {done.get('ok')}")


if __name__ == "__main__":
    main()
