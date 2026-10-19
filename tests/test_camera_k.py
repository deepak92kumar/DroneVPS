"""The camera's own K and D, and what ignoring them cost.

live_node read the full intrinsic matrix and then kept one number out of it -
fx - and threw away fy, the principal point and the distortion. Two of those
three turn aircraft ROTATION into phantom TRANSLATION, which is the error this
project has been chasing at a hover.

Measured off this OAK-D on 20 Oct at 1280x800:
    fx 791.723  fy 791.583  cx 624.889  cy 391.803
    D (8-coefficient rational): k1 24.09  k2 -39.14  k3 18.88
cx/cy sit 15.11 px and 8.20 px from the geometric centre the motion used to be
evaluated at. The fitted affine carries rotation, so evaluating it 17 px off
centre reads a 1 deg/frame roll as 0.32 px of travel - 0.64 m/s at 100 m and
17 Hz, against the project's own unexplained hover phantom of 0.36-0.43 m/s.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "vps_vio"))
from flow_odom import FlowTracker, shift_to_ne  # noqa: E402

fails = []
K = np.array([[791.723, 0.0, 624.889], [0.0, 791.583, 391.803], [0.0, 0.0, 1.0]])
D = np.array([24.08736, -39.14004, 0.00036, 0.00026, 18.88474, 23.47907,
              -37.42698, 17.61689])
CX, CY = 624.889, 391.803


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        fails.append(name)


def texture(seed=7, w=1280, h=800):
    """Something with corners everywhere, so tracking is not the variable."""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (h // 8, w // 8), dtype=np.uint8)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(img, (3, 3), 0)


def rotate_about(img, deg, cx, cy):
    M = cv2.getRotationMatrix2D((cx, cy), deg, 1.0)
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


print("a pure rotation about the OPTICAL centre is not travel")
base = texture()
rot = rotate_about(base, 1.0, CX, CY)

off = FlowTracker(max_corners=600)                       # geometric centre, raw px
on = FlowTracker(max_corners=600, K=K, D=D)              # optical centre, undistorted
for tr in (off, on):
    tr.step(base)
d_off, n_off = off.step(rot)
d_on, n_on = on.step(rot)
check("both tracked the pair", d_off is not None and d_on is not None,
      f"{n_off} / {n_on} tracks")
if d_off is not None and d_on is not None:
    m_off, m_on = float(np.linalg.norm(d_off)), float(np.linalg.norm(d_on))
    print(f"      geometric centre: {m_off:.3f} px    optical centre + undistort: {m_on:.3f} px")
    check("using K reduces the phantom shift", m_on < m_off, f"{m_on:.3f} < {m_off:.3f}")
    # 1 deg about a point 17.1 px away is 0.30 px of apparent travel.
    check("the old error is the predicted size", 0.1 < m_off < 1.2, f"{m_off:.3f} px")
    for alt in (50.0, 100.0):
        ph = m_off * alt / 791.723 * 17.0
        print(f"      at {alt:.0f} m, 17 Hz, that was {ph:.2f} m/s of phantom speed")

print("\na rotation about the GEOMETRIC centre is the control")
# Rotating about (640,400) should fool the optical-centre tracker by the same
# 17 px lever arm, in reverse. If it does not, the centre is not being used.
rot_g = rotate_about(base, 1.0, 640.0, 400.0)
off2 = FlowTracker(max_corners=600)
on2 = FlowTracker(max_corners=600, K=K, D=D)
for tr, im in ((off2, base), (on2, base)):
    tr.step(im)
d_off2, _ = off2.step(rot_g)
d_on2, _ = on2.step(rot_g)
if d_off2 is not None and d_on2 is not None:
    m_off2, m_on2 = float(np.linalg.norm(d_off2)), float(np.linalg.norm(d_on2))
    print(f"      geometric centre: {m_off2:.3f} px    optical centre: {m_on2:.3f} px")
    check("the two centres genuinely disagree", abs(m_off2 - m_on2) > 0.05,
          f"|{m_off2:.3f} - {m_on2:.3f}|")

print("\npure translation still measures as itself")
shifted = np.roll(base, 20, axis=1)          # 20 px right
t1 = FlowTracker(max_corners=600, K=K, D=D)
t1.step(base)
d, n = t1.step(shifted)
check("20 px of translation is recovered", d is not None and abs(d[0] - 20.0) < 1.5,
      f"{d} ({n} tracks)")

print("\nundistortion actually does something")
pts = np.array([[[50.0, 50.0]], [[1230.0, 750.0]], [[CX, CY]]], np.float32)
u = FlowTracker(max_corners=10, K=K, D=D)._undistort(pts)
moved = [float(np.linalg.norm(u[i][0] - pts[i][0])) for i in range(3)]
print(f"      corner {moved[0]:.2f} px, far corner {moved[1]:.2f} px, "
      f"optical centre {moved[2]:.4f} px")
check("edges move, the optical centre does not",
      moved[0] > 1.0 and moved[1] > 1.0 and moved[2] < 0.01, str([round(m, 3) for m in moved]))
check("without K it is a no-op",
      np.allclose(FlowTracker(max_corners=10)._undistort(pts), pts))

print("\nfy is used for the y axis")
n1, e1 = shift_to_ne(np.array([0.0, 100.0]), altitude_m=100.0,
                     focal_px=791.723, yaw_deg=0.0, focal_y_px=791.583)
n2, e2 = shift_to_ne(np.array([0.0, 100.0]), altitude_m=100.0,
                     focal_px=791.723, yaw_deg=0.0)
check("fy changes the north answer", n1 != n2, f"{n1:.6f} vs {n2:.6f}")
check("but only by the 0.018% it should", abs(n1 - n2) / abs(n2) < 5e-4,
      f"{100 * abs(n1 - n2) / abs(n2):.4f}%")
# x must still use fx, so a pure-x shift must be untouched by focal_y.
a, _ = shift_to_ne(np.array([100.0, 0.0]), altitude_m=100.0, focal_px=791.723,
                   yaw_deg=0.0, focal_y_px=1.0)
b, _ = shift_to_ne(np.array([100.0, 0.0]), altitude_m=100.0, focal_px=791.723,
                   yaw_deg=0.0)
check("the x axis still uses fx alone", abs(a - b) < 1e-12, f"{a} vs {b}")

print("\nno K is still a working tracker, as every flight before 20 Oct ran")
t3 = FlowTracker(max_corners=600)
t3.step(base)
d3, n3 = t3.step(np.roll(base, 15, axis=1))
check("it tracks without intrinsics", d3 is not None and abs(d3[0] - 15.0) < 1.5,
      f"{d3} ({n3} tracks)")

print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
