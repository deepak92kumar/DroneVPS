#!/usr/bin/env python3
"""The camera mount's yaw offset, measured from a slide on the ground.

The mount was rebuilt after 27 Oct, so camera_yaw_offset_deg = -1.95 is the
OLD mount's number. Two attempts failed to replace it:

  flow direction against GPS over whole flights - the estimate moved from
  +4.3 to +1.7 on one flight and +4.4 to +14.7 on another as the sample
  filter changed, so it was reading motion-correlated flow noise.

  the camera's own absolute heading against the FC's - it gave a clean and
  repeatable relation (camera = -(FC) + 42 deg, holding to 2.9 deg across an
  87 deg turn) but that is the BNO086's frame, not the image's, and flow and
  the matcher both work in the image.

A slide measures the image frame directly: push the aircraft along a known
line on the ground and compare the direction the flow reports against the
direction it actually went. Nothing else is involved - no IMU mapping, no
magnetometer, no GPS, no altitude.

Out and back gives two legs whose directions must be 180 deg apart. If they
are not, the measurement is not trustworthy and says so.
"""
import csv
import math
import sys

import numpy as np


def legs(path, t_from=0.0, min_m=0.30, gap_s=1.0):
    """Find stretches where the flow says the aircraft was moving."""
    R = list(csv.DictReader(open(path)))

    def F(k):
        o = []
        for r in R:
            try:
                o.append(float(r[k]))
            except (TypeError, ValueError):
                o.append(float('nan'))
        return np.array(o)

    t, dn, de = F('t'), F('step_dn'), F('step_de')
    yaw = F('yaw_deg')
    ok = (~np.isnan(dn)) & (~np.isnan(de)) & (t >= t_from)
    if ok.sum() < 50:
        return [], t, dn, de, yaw
    mag = np.hypot(np.nan_to_num(dn), np.nan_to_num(de))
    # A slide is many consecutive frames with a real step, not one spike.
    k = 9
    sm = np.convolve(mag * ok, np.ones(k) / k, mode='same')
    thr = max(0.02, 4.0 * np.median(sm[ok]))
    moving = ok & (sm > thr)
    out, i = [], 0
    while i < len(moving):
        if not moving[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(moving) and (moving[j + 1] or
                                       (t[min(j + 1, len(t) - 1)] - t[j]) < gap_s):
            j += 1
            if not moving[j] and (t[j] - t[i]) > 0 and \
                    not moving[min(j + 5, len(moving) - 1)]:
                break
        s = slice(i, j + 1)
        tot = np.hypot(np.nansum(dn[s]), np.nansum(de[s]))
        if tot >= min_m:
            out.append((s, tot))
        i = j + 1
    return out, t, dn, de, yaw


def main(path, t_from=0.0, truth=None):
    L, t, dn, de, yaw = legs(path, t_from)
    print(f"{path}")
    print(f"  {len(L)} chalne wale hisse mile (t >= {t_from:.0f} s)")
    if not L:
        print("  koi harkat nahi mili - flow ne kuch nahi dekha")
        return 1
    dirs = []
    for n, (s, tot) in enumerate(L, 1):
        N = np.nansum(dn[s])
        E = np.nansum(de[s])
        head = math.degrees(math.atan2(E, N)) % 360.0
        ym = np.nanmedian(yaw[s])
        print(f"  leg {n}: t {t[s][0]:7.1f}-{t[s][-1]:7.1f} s  "
              f"chala {tot:5.2f} m  N {N:+6.2f} E {E:+6.2f}  "
              f"flow ki disha {head:6.1f} deg  (yaw {ym:+7.1f})")
        dirs.append((head, tot))
    if len(dirs) >= 2:
        a, b = dirs[0][0], dirs[1][0]
        d = abs(((a - b + 180.0) % 360.0) - 180.0)
        print(f"\n  pehle do leg ke beech: {d:.1f} deg "
              f"(180 hona chahiye agar wapas laaye)")
        if abs(d - 180.0) > 25.0:
            print("  >> 180 se bahut door. Ya to seedhi lakeer nahi thi, ya "
                  "drone ghuma, ya flow bharose ke layak nahi. DOBARA karo.")
        else:
            print("  >> dono leg sehmat hai, naap bharose layak hai")
    if truth is not None:
        for n, (head, tot) in enumerate(dirs, 1):
            exp = truth if n % 2 == 1 else (truth + 180.0) % 360.0
            off = ((head - exp + 180.0) % 360.0) - 180.0
            print(f"  leg {n}: asli disha {exp:5.1f}, flow ne kaha {head:5.1f}"
                  f"  ->  OFFSET {off:+6.2f} deg")
    return 0


if __name__ == "__main__":
    p = sys.argv[1]
    tf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    tr = float(sys.argv[3]) if len(sys.argv) > 3 else None
    sys.exit(main(p, tf, tr))
