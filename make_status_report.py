#!/usr/bin/env python3
"""Build the DroneVPS project status PDF.

No dependencies. reportlab and fpdf are not installed on the Jetson and
pip on a flying aircraft's computer is not a thing anyone should need, so
this writes the PDF itself. Helvetica is one of the PDF base-14 fonts, so
nothing is embedded and the file stays small.

Edit REPORT below and run it. The layout is the one the 22-25 October
reports used, because the pilot reads these side by side.
"""
import sys, zlib

PAGE_W, PAGE_H = 595.28, 841.89          # A4 in points
M = 56.0                                  # margin

# --- Helvetica widths, /1000 em, ASCII 32..126 -------------------------
_HW = ("278 278 355 556 556 889 667 191 333 333 389 584 278 333 278 278 "
       "556 556 556 556 556 556 556 556 556 556 278 278 584 584 584 556 "
       "1015 667 667 722 722 667 611 778 722 278 500 667 556 833 722 778 "
       "667 778 722 667 611 722 667 944 667 667 611 278 278 278 469 556 "
       "333 556 556 500 556 556 278 556 556 222 222 500 222 833 556 556 "
       "556 556 333 500 278 556 500 722 500 500 500 334 260 334 584")
_HBW = ("278 333 474 556 556 889 722 238 333 333 389 584 278 333 278 278 "
        "556 556 556 556 556 556 556 556 556 556 333 333 584 584 584 611 "
        "975 722 722 722 722 667 611 778 722 278 556 722 611 833 722 778 "
        "667 778 722 667 611 722 667 944 667 667 611 333 278 333 584 556 "
        "333 556 611 556 611 556 333 611 611 278 278 556 278 889 611 611 "
        "611 611 389 556 333 611 556 778 556 556 500 389 280 389 584")
HW = [int(x) for x in _HW.split()]
HBW = [int(x) for x in _HBW.split()]


def width(txt, size, bold=False):
    tab = HBW if bold else HW
    w = 0
    for ch in txt:
        o = ord(ch)
        w += tab[o - 32] if 32 <= o <= 126 else 556
    return w * size / 1000.0


def wrap(txt, size, maxw, bold=False):
    out, line = [], ""
    for word in txt.split():
        trial = word if not line else line + " " + word
        if width(trial, size, bold) <= maxw:
            line = trial
        else:
            if line:
                out.append(line)
            line = word
    if line:
        out.append(line)
    return out


def esc(s):
    return s.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


class Page:
    def __init__(self):
        self.ops = []

    def rect(self, x, y, w, h, col):
        r, g, b = col
        self.ops.append(f"{r:.3f} {g:.3f} {b:.3f} rg {x:.2f} {y:.2f} "
                        f"{w:.2f} {h:.2f} re f")

    def text(self, x, y, s, size, bold=False, col=(0, 0, 0)):
        r, g, b = col
        f = "F2" if bold else "F1"
        self.ops.append(f"BT {r:.3f} {g:.3f} {b:.3f} rg /{f} {size:.2f} Tf "
                        f"{x:.2f} {y:.2f} Td ({esc(s)}) Tj ET")

    def stream(self):
        return "\n".join(self.ops).encode("latin-1", "replace")


def build(pages, path):
    objs, out = [], bytearray(b"%PDF-1.4\n")
    n_pages = len(pages)
    # 1 catalog, 2 pages, 3..  page objs, then contents, then fonts
    page_ids = [3 + i for i in range(n_pages)]
    cont_ids = [3 + n_pages + i for i in range(n_pages)]
    f1, f2 = 3 + 2 * n_pages, 4 + 2 * n_pages
    objs.append((1, f"<< /Type /Catalog /Pages 2 0 R >>".encode()))
    kids = " ".join(f"{i} 0 R" for i in page_ids)
    objs.append((2, f"<< /Type /Pages /Count {n_pages} /Kids [{kids}] >>".encode()))
    for i, p in enumerate(pages):
        objs.append((page_ids[i], (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_W} {PAGE_H}] "
            f"/Resources << /Font << /F1 {f1} 0 R /F2 {f2} 0 R >> >> "
            f"/Contents {cont_ids[i]} 0 R >>").encode()))
    for i, p in enumerate(pages):
        raw = p.stream()
        comp = zlib.compress(raw)
        objs.append((cont_ids[i], b"<< /Length " + str(len(comp)).encode() +
                     b" /Filter /FlateDecode >>\nstream\n" + comp + b"\nendstream"))
    objs.append((f1, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
                     b"/Encoding /WinAnsiEncoding >>"))
    objs.append((f2, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold "
                     b"/Encoding /WinAnsiEncoding >>"))
    offs = {}
    for num, body in sorted(objs):
        offs[num] = len(out)
        out += f"{num} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    hi = max(offs) + 1
    out += f"xref\n0 {hi}\n".encode() + b"0000000000 65535 f \n"
    for i in range(1, hi):
        out += f"{offs.get(i, 0):010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {hi} /Root 1 0 R >>\nstartxref\n{xref}\n"
            "%%EOF\n").encode()
    open(path, "wb").write(out)
    return len(out)


# ===================== the report ====================================
DATE = "26 October 2026"
NAVY = (0.106, 0.216, 0.369)
GREY = (0.45, 0.45, 0.45)
LIGHT = (0.863, 0.863, 0.863)
RULE = (0.75, 0.75, 0.75)

INTRO = ("Marker-free GNSS-denied navigation on a 4.0 kg hexacopter. Target: "
         "100-150 m AGL, 3-5 km out-and-back, no GNSS, no ground marker. "
         "Two flights since the last report, 246 and 247. Flight 247 flew a "
         "full waypoint mission on the VPS from takeoff and touched every "
         "waypoint - a first - then lost its hold below 60 m and took an EKF "
         "failsafe at 377 s.")

OVERALL = 79
OVER_NOTE = ("78% on 25 October. The rise is the first autonomous mission and "
             "the altitude fault finally being measured rather than argued. "
             "Nothing new has flown, so nothing new is above 95%.")

HILITE = [
    ("The barometer is 5.5 m high, and now it is proven.",
     "Against the GPS altitude on log 246, which is independent of both: "
     "lidar = 0.898*GPSalt - 0.04 m at corr 0.980, baro = 0.889*GPSalt "
     "+5.49 m at corr 0.989. Same slope - that is the GPS - and the offsets "
     "are the finding. Velocity is pixels x altitude / focal, so +5.5 m is "
     "+7% of velocity at 100 m and +70% at 10 m."),
    ("The map cannot work below 60 m with this sheet.",
     "276 frames at 28-62 m, four window floors - 40 m as flown, 25, 15 and "
     "none - accepted 0 of 276 in every one. The query is rescaled down to "
     "the map's 0.509 m/px, so at 30 m it is 56x56 px and XFeat finds 9 "
     "matches against a gate of 12. At 80 m it is 150x150 px and 145."),
    ("The lidar's zero wanders metres, and the node now handles it.",
     "6.94 m read with the lens 0.26 m above the floor; 0.65 -> 2.9 m in "
     "30 minutes untouched. Not the supply: Vcc sd 0.0058 V, corr with the "
     "reading -0.110. The scale is right. The node re-learns the zero on the "
     "ground until the moment of arming."),
    ("The handover is clean. The glitch is not the handover.",
     "Measured on both switches into the VPS on log 246: the first injected "
     "position was 0.01 m from what the EKF held, after dropping 31 m and "
     "33 m of accumulated drift. The GPS-glitch and EKF messages arrive 0.4 "
     "to 3.4 s LATER."),
]

SECTIONS = [
 ("Platform and plumbing", [
  ("Airframe, compute, sensors", 95,
   "ZD850 hexa, CUAV X7 Pro, Jetson Orin, OAK-D Pro. The pilot re-measured "
   "the stack: FC 6 cm above the CG, camera 22 cm below. GPS2_POS_Z "
   "corrected 0.14 -> 0.22 on the aircraft."),
  ("Camera and frame rate", 98,
   "Unchanged. Both monos at 25 fps, CAM_C primary, no crop."),
  ("Vibration", 97,
   "Re-measured across five flights at the pilot's challenge: X 0.89-1.35, "
   "Y 0.91-1.37, Z 1.37-1.95 m/s2 median, zero clipping on all of them. "
   "ArduPilot's limit is 30. Not a fault and removed from the list."),
  ("MAVLink routing", 94,
   "Unchanged. Parameters were set live through the router today - rangefinder "
   "zero and GPS2_POS_Z - with the node running and its rangefinder stream "
   "intact afterwards."),
  ("Services and auto-start", 100, "Unchanged."),
  ("Flight data recording", 96,
   "80-column CSV, frames at 4 Hz. Down from 97: flight 247 produced NO "
   "node-side CSV at all, so that flight can only be read from the FC log."),
  ("Preflight diagnostics", 94,
   "The lidar zero no longer needs a preflight command - the node measures it "
   "itself while disarmed and on the ground."),
 ]),
 ("Position estimation", [
  ("Optical flow", 92, "Unchanged. Per-track derotation since 25 October."),
  ("Dead reckoning over distance", 85, "Unchanged method."),
  ("Gyro derotation", 90, "Unchanged. The model is right; the gain never was the problem."),
  ("Camera / IMU configuration", 95, "Unchanged."),
  ("Mount alignment", 90,
   "Camera yaw offset -1.95 deg. The ROLL mount offset is still unset while "
   "pitch is -1.13; roll MAD was 2.43 at 110 m, which is +-4.6 m of scatter."),
  ("Map matching", 78,
   "Down from 80, on knowledge rather than performance. Above 60 m it is "
   "unchanged and good - 5.64 m p50, 2.89 m at 60-80. Below 60 m it is now "
   "known to be impossible with this sheet, so lowering map_min_alt_m is "
   "closed. A finer map is the only route left, and only for low altitude."),
  ("Sensor fusion", 72,
   "Unchanged. fix_sigma_m is still 20 while the map now measures 5.64 m, so "
   "the Kalman gain sits at 0.068. The correction written for this was "
   "reverted and has not been brought back."),
  ("Velocity to the FC", 55,
   "At 40-60 m on flight 246 the sent speed was 2.03 m/s where the receiver "
   "said 0.94. The barometric offset explains about 0.09 of that 1.09 m/s. "
   "The rest is the phantom and is still open."),
  ("Heading / yaw", 55, "Unchanged. The compass is still the only yaw source."),
  ("Altitude", 60,
   "Up from 45. The fault is measured and a fix is written and unflown: the "
   "lidar corrects the barometer's offset without replacing it, and the "
   "lidar's own zero is re-learned on the ground to the moment of arming. "
   "EK3_SRC1_POSZ stays 1 and EK3_RNG_USE_HGT stays -1 - this sensor is fit "
   "to measure a slow offset, not to carry height."),
  ("Position injection as GPS2", 92,
   "GPS2_POS_Z now 0.22 m, the camera's true offset below the CG."),
 ]),
 ("Flight controller integration", [
  ("EKF origin and home", 100,
   "From SCR_USER1/2 at arm on both flights, which is what a jammed "
   "environment needs. The pilot has closed the question of taking it from "
   "the receiver and is right to."),
  ("Manual RC GPS1/GPS2 switch", 100,
   "Measured on both switches into the VPS on log 246: 0.01 m from the EKF's "
   "own position, 31 m and 33 m of drift dropped. Nothing to fix."),
  ("Actually flying on the VPS", 80,
   "Down from 85. Flight 246 held - 3.95 m at 40-60 m against the receiver. "
   "Flight 247 did not: 27.23 m at 40-60 and 24.83 at 60-80, and an EKF "
   "failsafe to ALT_HOLD at 377 s."),
  ("Smoothness", 35,
   "Unchanged and still the lowest line on the sheet. The deadband is a hard "
   "edge; the ramp written for it is in history and not brought back."),
  ("Autonomous mission", 70,
   "Up from 40. Flight 247 flew the whole waypoint mission on the VPS from "
   "takeoff and touched every waypoint. It lost the hold on descent, which is "
   "why this is not higher."),
  ("Mission speed ceiling", 70, "Unchanged."),
  ("Behaviour on RC loss", 100, "Unchanged."),
 ]),
]

MOVED_UP = [
 ("Autonomous mission", 40, 70, "247 flew every waypoint on the VPS"),
 ("Altitude", 45, 60, "+5.5 m measured against GPS altitude; fix written"),
 ("Vibration", 96, 97, "re-measured on five flights, zero clipping"),
 ("Preflight diagnostics", 92, 94, "the lidar zeroes itself now"),
 ("Position injection as GPS2", 90, 92, "GPS2_POS_Z 0.14 -> 0.22"),
]
MOVED_DOWN = [
 ("Actually flying on the VPS", 85, 80, "247 took an EKF failsafe below 60 m"),
 ("Map matching", 80, 78, "below 60 m is now known to be impossible here"),
 ("Flight data recording", 97, 96, "247 produced no node CSV at all"),
]

CORRECTIONS = [
 ("I called the 40 m window floor the direct answer to the descent problem.",
  "It was arithmetic. Measured on 276 real frames, all four window floors "
  "accepted 0 of 276. The fault is the query's pixel count, not the window."),
 ("I measured a 43 m step at the handover into the VPS.",
  "My own artefact: injection stops while the switch is on the receiver, so "
  "the nearest GPS2 sample was minutes stale. The real step is 0.01 m."),
 ("I told the pilot to apply the 20-25 September rangefinder configuration.",
  "Applied and measured: the reading went 6.94 m -> 19.72 m. Restored. Lower "
  "scalings were then swept at the pilot's instruction - 65, 50, 30, 10 - and "
  "all of them shrink the reading rather than zero it."),
 ("I started to doubt that the barometer reads high.",
  "The GPS-altitude check confirmed the original finding. The premise the "
  "altitude fix rests on holds."),
 ("I put vibration on the to-do list without measuring it first.",
  "The pilot asked why it had never been reported. Measured: fine on every "
  "flight. Removed."),
 ("I wrote a three-command field procedure for the lidar zero.",
  "The pilot rejected it on design, not convenience, and was right. The node "
  "does it now and there is nothing to run."),
]

OPEN = [
 ("Velocity at 40-60 m", "2.03 m/s sent where the receiver said 0.94. The "
  "barometric offset explains 0.09 of it. This is the largest open fault."),
 ("Smoothness", "307 zero crossings in 634 s. The deadband ramp is written "
  "and waiting on the pilot, because it changes a guard."),
 ("The lidar's zero", "4.5 m/hour at a fixed distance today. Not the supply. "
  "A two-distance test an hour apart separates a drifting sensor zero from a "
  "drifting resistor; only the second is fixed by a 25 ppm/C metal-film part."),
 ("Roll mount offset", "Unset. Pitch is -1.13. Roll MAD 2.43 at 110 m."),
 ("fix_sigma_m 20", "The map now measures 5.64 m. The gain is 0.068."),
 ("A finer map, below 60 m only", "The only route left to map fixes during "
  "the descent. Above 60 m a finer sheet gives nothing - 9.0 m against 8.9."),
]


# ===================== layout =========================================
class Doc:
    def __init__(self):
        self.pages = [Page()]
        self.y = PAGE_H - M
        self._header()

    @property
    def p(self):
        return self.pages[-1]

    def _header(self):
        p = self.p
        p.text(M, PAGE_H - 44, f"DroneVPS   ·   Project Status   ·   {DATE}",
               8, False, GREY)
        p.rect(M, PAGE_H - 54, PAGE_W - 2 * M, 0.6, RULE)
        self.y = PAGE_H - 78

    def need(self, h):
        if self.y - h < M + 24:
            self.pages.append(Page())
            self._header()

    def para(self, txt, size=9, bold=False, col=(0, 0, 0), lead=None,
             x=M, w=None, gap=0):
        w = w or (PAGE_W - 2 * M)
        lead = lead or size * 1.35
        for ln in wrap(txt, size, w, bold):
            self.need(lead)
            self.p.text(x, self.y, ln, size, bold, col)
            self.y -= lead
        self.y -= gap

    def bar(self, x, y, w, h, pct, col=NAVY):
        self.p.rect(x, y, w, h, LIGHT)
        self.p.rect(x, y, w * max(0.0, min(pct, 100)) / 100.0, h, col)

    def section(self, num, title, rows):
        self.need(74)
        self.y -= 10
        self.p.text(M, self.y, f"{num}.", 15, True, (0, 0, 0))
        self.p.text(M + 24, self.y, title, 15, True, (0, 0, 0))
        self.y -= 19
        self.p.text(M, self.y, "Item", 7.5, True, GREY)
        self.p.text(M + 236, self.y, "Status / evidence", 7.5, True, GREY)
        self.y -= 5
        self.p.rect(M, self.y, PAGE_W - 2 * M, 0.6, RULE)
        self.y -= 15
        for name, pct, note in rows:
            lines = wrap(note, 8, PAGE_W - M - (M + 236))
            h = max(26.0, len(lines) * 10.4 + 10)
            self.need(h)
            top = self.y
            self.p.text(M, top, name, 8.5, False, (0, 0, 0))
            self.bar(M + 140, top - 1.5, 58, 6.5, pct)
            self.p.text(M + 228 - width(f"{pct}%", 8.5, True), top,
                        f"{pct}%", 8.5, True, (0, 0, 0))
            yy = top
            for ln in lines:
                self.p.text(M + 236, yy, ln, 8, False, (0.2, 0.2, 0.2))
                yy -= 10.4
            self.y = min(top - 26, yy) - 6

    def movetable(self, title, rows, up=True):
        self.need(50)
        self.y -= 6
        self.p.text(M, self.y, title, 10.5, True, (0, 0, 0))
        self.y -= 15
        for name, a, b, why in rows:
            self.need(14)
            self.p.text(M, self.y, name, 8.5, False, (0, 0, 0))
            self.p.text(M + 150, self.y, f"{a}%", 8.5, False, GREY)
            self.p.text(M + 176, self.y, "->", 8.5, False, GREY)
            self.p.text(M + 190, self.y, f"{b}%", 8.5, True,
                        NAVY if up else (0.55, 0.12, 0.12))
            self.p.text(M + 236, self.y, why, 8, False, (0.2, 0.2, 0.2))
            self.y -= 13.5
        self.y -= 4


def main(path):
    d = Doc()
    d.p.text(M, d.y, "DroneVPS - Project Status", 24, True, (0, 0, 0))
    d.y -= 26
    d.para(INTRO, 9, False, (0.1, 0.1, 0.1), 12.5, M, PAGE_W - 2 * M, gap=14)

    top = d.y
    d.p.text(M + 14, top, "Overall", 10, False, (0.3, 0.3, 0.3))
    d.p.text(M + 14, top - 42, f"{OVERALL}%", 36, True, (0, 0, 0))
    d.bar(M + 14, top - 62, 270, 9, OVERALL)
    yy = top - 80
    for ln in wrap(OVER_NOTE, 7.5, 262):
        d.p.text(M + 14, yy, ln, 7.5, False, GREY)
        yy -= 10
    hy = top
    for h, b in HILITE:
        for ln in wrap(h, 9, PAGE_W - M - (M + 300), True):
            d.p.text(M + 300, hy, ln, 9, True, (0, 0, 0))
            hy -= 11.5
        hy -= 2
        for ln in wrap(b, 8, PAGE_W - M - (M + 300)):
            d.p.text(M + 300, hy, ln, 8, False, (0.2, 0.2, 0.2))
            hy -= 10
        hy -= 10
    d.y = min(yy, hy) - 10
    d.p.text(M, d.y, "An item is 100% only when it has been verified in flight.",
             7.5, False, GREY)
    d.y -= 22

    for i, (title, rows) in enumerate(SECTIONS, 1):
        d.section(i, title, rows)

    d.need(120)
    d.y -= 8
    d.p.text(M, d.y, "4.", 15, True, (0, 0, 0))
    d.p.text(M + 24, d.y, "What moved since 25 October", 15, True, (0, 0, 0))
    d.y -= 22
    d.movetable("Up", MOVED_UP, True)
    d.movetable("Down", MOVED_DOWN, False)

    d.need(90)
    d.y -= 8
    d.p.text(M, d.y, "5.", 15, True, (0, 0, 0))
    d.p.text(M + 24, d.y, "Corrections - all of these are mine", 15, True, (0, 0, 0))
    d.y -= 22
    for head, body in CORRECTIONS:
        d.need(38)
        d.para(head, 8.5, True, (0, 0, 0), 11)
        d.para(body, 8, False, (0.2, 0.2, 0.2), 10, gap=7)

    d.need(90)
    d.y -= 8
    d.p.text(M, d.y, "6.", 15, True, (0, 0, 0))
    d.p.text(M + 24, d.y, "Open, in the order I would take them", 15, True, (0, 0, 0))
    d.y -= 22
    for i, (head, body) in enumerate(OPEN, 1):
        d.need(34)
        lines = wrap(body, 8, PAGE_W - M - (M + 150))
        top = d.y
        d.p.text(M, top, f"{i}.  {head}", 8.5, True, (0, 0, 0))
        yy = top
        for ln in lines:
            d.p.text(M + 150, yy, ln, 8, False, (0.2, 0.2, 0.2))
            yy -= 10
        d.y = min(top - 12, yy) - 6

    for i, p in enumerate(d.pages, 1):
        p.text(PAGE_W - M - width(str(i), 8), M - 18, str(i), 8, False, GREY)
    n = build(d.pages, path)
    print(f"{path}  {len(d.pages)} pages  {n/1024:.1f} KB")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "/data/DRONE/DroneVPS_Status_2026-10-26.pdf")
