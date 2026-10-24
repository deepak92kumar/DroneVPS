"""One socket decides every camera-frame measurement, and stereo's eyes do not move.

The primary camera became CAM_C on 22 Oct 2026, at the pilot's instruction.
The reason is the airframe's landing leg: photographed that day with the
aircraft lifted, it fills CAM_B's x 0-75 with its foam grip and sends a bright
strut to about x 165, while CAM_C's left edge is clear ground. CAM_B needed
config.crop_left_px = 130 to cut that off, which cost flow 10% of the frame's
width and moved cx 80 px off the image centre - 11 m of ground at 110 m - to be
carried by hand through the whole map-query chain. CAM_C needs no crop and its
cx is 4 px off centre.

Two things can go silently wrong when the primary moves, and this is here for
both of them.

ONE: a camera-frame quantity read from the OTHER socket. The intrinsics, the
distortion, the IMU->camera rotation the accelerometer is turned by, the pixel
a map fix is reported for and the frame the depth gate indexes must all come
from one socket, because they are combined with each other. The two sockets
genuinely differ on this device - fx 791.72 against 795.10, cx 624.89 against
644.13, k1 24.09 against 26.05 - so a mixed pair is wrong, not merely untidy.

TWO: stereo's left and right. They are PHYSICAL eyes: CAM_B is the left on this
board, CAM_C the right. They must stay keyed to the socket, not to which camera
is primary, because feeding them the other way round flips the sign of every
disparity and depth comes back as noise - with no error, no exception and
nothing in the log to show for it.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "vps_vio"))

from config import VpsConfig

FAILED = []


def check(label, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}   {detail}")
    if not ok:
        FAILED.append(label)


print(__doc__)

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = open(os.path.join(root, "vps_vio", "live_node.py")).read()

primary = re.search(r'PRIMARY_SOCKET_NAME\s*=\s*"(\w+)"', src)
partner = re.search(r'STEREO_PARTNER_NAME\s*=\s*"(\w+)"', src)
check("the primary socket is named once, at module level", primary is not None)
check("the stereo partner is named once, at module level", partner is not None)
if primary and partner:
    print(f"        primary {primary.group(1)}   partner {partner.group(1)}")
    check("the two are different sockets", primary.group(1) != partner.group(1))
    check("both are mono sockets, not the RGB CAM_A",
          {primary.group(1), partner.group(1)} == {"CAM_B", "CAM_C"},
          f"{primary.group(1)}/{partner.group(1)}")

# Every per-socket calibration read takes the primary, never a bare socket.
PER_SOCKET = ("getCameraIntrinsics", "getDistortionCoefficients",
              "getImuToCameraExtrinsics", "setDepthAlign")
print("\nthe per-socket calls as written")
for fn in PER_SOCKET:
    calls = re.findall(re.escape(fn) + r"\(\s*([^,)]+)", src)
    for a in calls:
        print(f"        {fn:28s} <- {a.strip()}")
    # 23 Oct: a per-socket call may now name EITHER socket, because the
    # partner camera got its own tracker and must be given its OWN
    # intrinsics and distortion - CAM_B reads cx 624.89 / k1 24.09 against
    # CAM_C's 644.13 / 26.05, and feeding the primary's to both would put
    # the primary's distortion error into the very average that exists to
    # cancel it. What must never happen is a call with a HARD-CODED socket,
    # which is what this file was written to catch.
    named = [a for a in calls
             if "PRIMARY_SOCKET_NAME" in a or "STEREO_PARTNER_NAME" in a]
    check(f"{fn} is given a socket by NAME, never hard-coded",
          bool(calls) and len(named) == len(calls),
          ", ".join(a.strip() for a in calls) or "no call found")
    check(f"{fn} is given the primary at least once",
          any("PRIMARY_SOCKET_NAME" in a for a in calls))

# Stereo's eyes are keyed to the socket NAME, not to the primary/partner role.
left = re.search(r'by_socket\[\s*"(\w+)"\s*\]\.out\.link\(\s*st\.left\s*\)', src)
right = re.search(r'by_socket\[\s*"(\w+)"\s*\]\.out\.link\(\s*st\.right\s*\)', src)
print("\nstereo wiring as written")
print(f"        st.left  <- {left.group(1) if left else '??'}")
print(f"        st.right <- {right.group(1) if right else '??'}")
check("stereo's left eye is CAM_B, by socket name",
      left is not None and left.group(1) == "CAM_B")
check("stereo's right eye is CAM_C, by socket name",
      right is not None and right.group(1) == "CAM_C")
check("neither eye is wired through the primary/partner role",
      not re.search(r"mono\[\s*\"mono(_partner)?\"\s*\]\.out\.link\(\s*st\.(left|right)",
                    src))

# The crop existed for CAM_B's leg. On CAM_C there is nothing to cut.
crop = VpsConfig().crop_left_px
check("the leg crop is off while CAM_C is primary", crop == 0, f"crop_left_px = {crop}")
check("the crop is still a setting, not deleted",
      "crop_left_px" in open(os.path.join(root, "vps_vio", "config.py")).read())

# The depth gate must not keep a focal baked in for the other socket.
flow = open(os.path.join(root, "vps_vio", "flow_odom.py")).read()
check("the depth gate takes fx from this camera's own K",
      re.search(r"fx\s*=\s*791\.723\s+if\s+self\.K\s+is\s+None\s+else", flow) is not None)

print(f"\n{'FAILED: ' + ', '.join(FAILED) if FAILED else 'all checks passed'}")
sys.exit(1 if FAILED else 0)
