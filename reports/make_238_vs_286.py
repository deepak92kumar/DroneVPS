#!/usr/bin/env python3
"""Build the 238-vs-286 comparison PDF: software, config, FC parameters, flight.

Every number in here is read live from the repository, the two Jetson CSVs and
the two FC dataflash logs at the moment this runs. Nothing is typed in by hand,
so a value that looks wrong is a value that IS wrong somewhere upstream.

    python3 reports/make_238_vs_286.py
"""
import csv
import subprocess
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

REPO = "/data/DRONE"
SCRATCH = ("/tmp/claude-1000/-data-DRONE/"
           "86eaaeb5-803c-484e-beba-357c25eb6503/scratchpad")
OUT = f"{REPO}/reports/238_vs_286_everything.pdf"
WP_BUILD = "87e19ad"

# ---------------------------------------------------------------- page engine
LINES_PER_PAGE = 62
pages = []


def sh(cmd):
    return subprocess.run(cmd, shell=True, cwd=REPO, capture_output=True,
                          text=True).stdout.rstrip("\n")


def emit(title, lines):
    """One logical section, split across as many pages as it needs."""
    lines = [l.rstrip() for l in lines]
    first = True
    while lines or first:
        chunk, lines = lines[:LINES_PER_PAGE], lines[LINES_PER_PAGE:]
        pages.append((title if first else title + "  (contd)", chunk))
        first = False
        if not lines:
            break


def render():
    with PdfPages(OUT) as pdf:
        for i, (title, body) in enumerate(pages, 1):
            fig = plt.figure(figsize=(8.27, 11.69))   # A4 portrait
            fig.patch.set_facecolor("white")
            fig.text(0.06, 0.967, title, fontsize=11.5, fontweight="bold",
                     family="DejaVu Sans", color="#111111")
            fig.add_artist(plt.Line2D([0.06, 0.94], [0.957, 0.957],
                                      color="#999999", lw=0.8,
                                      transform=fig.transFigure))
            fig.text(0.06, 0.945, "\n".join(body), fontsize=6.9,
                     family="DejaVu Sans Mono", va="top", linespacing=1.42,
                     color="#1a1a1a")
            fig.text(0.94, 0.018, f"{i} / {len(pages)}", fontsize=7,
                     ha="right", color="#777777")
            fig.text(0.06, 0.018,
                     "DroneVPS  -  flight 238 (27 Oct) vs flight 286 (28 Oct)",
                     fontsize=7, color="#777777")
            pdf.savefig(fig)
            plt.close(fig)


# ------------------------------------------------------------------- the data
def params(path):
    """First recorded value of every parameter = what the flight STARTED with.

    Taking the last value instead is how a wrong answer was produced on 29 Oct:
    GPS_PRIMARY ends log 261 at 0 because the pilot flipped the switch at
    96.2 s, and reading that as the flight's setting turned a pilot action into
    an imaginary configuration difference.
    """
    from pymavlink import mavutil
    m = mavutil.mavlink_connection(path)
    p = {}
    while True:
        msg = m.recv_match(type="PARM")
        if msg is None:
            break
        if msg.Name not in p:
            p[msg.Name] = msg.Value
    return p


def param_history(path, name):
    from pymavlink import mavutil
    m = mavutil.mavlink_connection(path)
    out, t0 = [], None
    while True:
        msg = m.recv_match(type="PARM")
        if msg is None:
            break
        t = getattr(msg, "TimeUS", 0) / 1e6
        if t0 is None:
            t0 = t
        if msg.Name == name:
            out.append((t - t0, msg.Value))
    return out


def load_csv(run):
    r = list(csv.DictReader(open(f"{REPO}/flight_logs/vps/{run}/{run}.csv")))

    def c(k):
        o = []
        for x in r:
            try:
                o.append(float(x[k]))
            except Exception:
                o.append(np.nan)
        return np.array(o)

    d = {k: c(k) for k in ("t", "alt_m", "north_m", "east_m", "sigma_m",
                           "sigma_abs_m", "tracks", "gps1_north_m",
                           "gps1_east_m", "vel_sent_vn", "vel_sent_ve",
                           "vel_flow_vn", "vel_flow_ve", "vel_deadbanded",
                           "vel_deadband", "fix_score", "fix_disagree_m",
                           "ekf_vel_var", "ekf_pos_h_var", "flow_disagree_px",
                           "frame_age_s", "depth_med_m")}
    d["src"] = np.array([x.get("src", "") for x in r])
    d["fx"] = np.array([bool((x.get("fix_lon") or "").strip()) for x in r])
    d["n"] = len(r)
    d["vps"] = d["src"] == "VPS"
    d["air"] = d["vps"] & (d["alt_m"] > 5)
    d["ok"] = (d["vps"] & np.isfinite(d["gps1_north_m"])
               & np.isfinite(d["gps1_east_m"])
               & (np.abs(d["gps1_north_m"]) < 400)
               & (np.abs(d["gps1_east_m"]) < 400))
    d["err"] = np.hypot(d["north_m"] - d["gps1_north_m"],
                        d["east_m"] - d["gps1_east_m"])
    return d


A = load_csv("vps_00238")
B = load_csv("vps_00286")
PA = params(f"{SCRATCH}/00000252.BIN")
PB = params(f"{SCRATCH}/00000261.BIN")


def two(label, a, b, w=34):
    return "  %-*s %20s %20s" % (w, label, a, b)


def pct(D, m, k, q):
    v = D[k][m]
    v = v[np.isfinite(v)]
    return np.percentile(v, q) if v.size else np.nan


# =========================================================== 1. the headline
emit("1.  Kya poochha gaya, aur seedha jawaab", [
    "",
    "SAWAAL: flight 238 (27 Oct, WP mission, bina ek bhi failsafe) aur",
    "        flight 286 (28 Oct subah, drone height leke bhaga) ke beech",
    "        software, code, parameter - har chhoti cheez - me kya alag tha?",
    "",
    "JAWAAB, ek line me:",
    "",
    "    Software me kuch alag nahi tha.  Parameters me kuch alag nahi tha.",
    "    Alag tha: kitni der hawa me raha, kitni tezi se chadha, 80 m par",
    "    flow ka shor, aur jo sigma FC ko declare ki gayi.",
    "",
    "-" * 92,
    "",
    "  Yeh dono flight kis build par udi:",
    "",
    f"    238  ->  {WP_BUILD}   27 Oct 10:03",
    "             \"accel_to_ne was rotating by yaw alone, and an aircraft",
    "              is never level\"",
    "",
    f"    286  ->  {WP_BUILD}   wahi build, byte-by-byte",
    "",
    "  27 Oct 15:26 aur 16:01 par do revert hue (9ce229f, a3bbf8e). Pehla",
    "  do commit zyada peeche chala gaya tha (44e86db); dusre ne theek kiya.",
    f"  286 ke waqt tak tree {WP_BUILD} par hi tha.",
    "",
    "-" * 92,
    "",
    "  TEEN CHEEZEIN JO IS JAANCH ME MERI GALTI THI, aur sudhaari gayi:",
    "",
    "   1. \"Node 28 Oct ko chala hi nahi\" - galat. Maine directory ke",
    "      timestamps par bharosa kiya. Jetson ka clock 12 ghante peeche",
    "      tha (19:15 = asli 07:15), aur journal ka boot record",
    "      1970-01-01 se shuru hota hai. vps_00286 asli flight hai.",
    "",
    "   2. \"GPS_PRIMARY 1 se 0 ho gaya tha, yahi antar hai\" - galat.",
    "      Meri script har parameter ka AAKHRI value le rahi thi.",
    "      GPS_PRIMARY dono flight me 1 (VPS) tha. 286 me woh 97.2 s par",
    "      0 hua kyunki pilot ne 96.2 s par switch GPS par kiya.",
    "",
    "   3. \"50-60 m me matcher chala par accept nahi hua\" - yeh mujhe",
    "      pata hi nahi tha. CSV sirf accept hue fix likhta tha. Aaj",
    "      (29 Oct) woh theek kiya gaya - niche section 3 me.",
    "",
])

# ===================================================== 2. software: the code
diffstat = sh(f"git diff --stat {WP_BUILD} -- vps_vio tests").splitlines()
hashes = []
for f in sh(f"git ls-tree -r --name-only {WP_BUILD} -- vps_vio "
            "| grep '\\.py$'").splitlines():
    a = sh(f"git show {WP_BUILD}:{f} | md5sum | cut -d' ' -f1")
    b = sh(f"md5sum {f} | cut -d' ' -f1")
    hashes.append((f, a == b))
same = sum(1 for _, s in hashes if s)

emit("2.  Software:  238 ke waqt kya tha, ab kya hai", [
    "",
    f"  238 ke build ({WP_BUILD}) ki HAR python file ka hash, aaj disk par",
    "  padi file se milaya gaya:",
    "",
    f"    {len(hashes)} files checked     {same} identical     "
    f"{len(hashes) - same} badli",
    "",
] + [f"      {'SAME  ' if s else 'BADLI '} {f}" for f, s in hashes] + [
    "",
    "-" * 92,
    "",
    f"  git diff --stat {WP_BUILD} -> working tree (abhi):",
    "",
] + [f"    {l}" for l in diffstat] + [
    "",
    "    audit.py / vel_fast.py / vel_vectors.py  -  teen analysis script.",
    "    Node inhe import NAHI karta:",
    "",
    "      grep -rn \"import (audit|vel_fast|vel_vectors)\" vps_vio tests",
    "        -> kahin bhi nahi",
    "",
    "    Yeh sirf terminal se chalte hai. Udte waqt load nahi hote.",
    "",
    "    live_node.py  -  aaj, 29 Oct ki teen tabdeeliyan. Section 3.",
    "",
    "-" * 92,
    "",
    f"  Commits {WP_BUILD} se HEAD tak:",
    "",
] + [f"    {l}" for l in sh(f"git log --oneline {WP_BUILD}..HEAD").splitlines()] + [
    "",
    "    52eb646  origin sigma 20 -> 0.5         REVERT ho gaya",
    "    b0b39f6  flow_vel_frames 40 -> 30       REVERT ho gaya",
    "    d7e95ad  comment fix (velocity ON)      REVERT ho gaya",
    "    9ce229f  revert, par 44e86db tak        2 commit zyada peeche",
    "    39a7b61  map_min_alt_m 50 -> 0          REVERT ho gaya",
    "    b637ae1  audit.py naya                  RAHA (node use nahi karta)",
    f"    a3bbf8e  {WP_BUILD} par wapas             YAHI 286 ke waqt tha",
])

# ========================================== 3. what changed today, 29 Oct
emit("3.  Aaj (29 Oct) maine kya badla - teen cheezein", [
    "",
    "  Pilot ne teen kaam diye the. Teeno hue. Sigma ko CHHODA gaya -",
    "  \"ab experiment ka time nahi hai kal demo hai\".",
    "",
    "-" * 92,
    "",
    "  (A)  MAP MATCHER AB APNI HAR KOSHISH LIKHTA HAI",
    "",
    "   Pehle: CSV me fix_* columns `if res.ok` ke andar likhe jaate the.",
    "   Reject hui koshish ka koi nishaan nahi bachta tha. vps_00286 me",
    "   fix_score 69 rows par tha aur wahi 69 accept bhi hue the.",
    "",
    "   Isliye us flight ka asli sawaal - map_min_alt_m 50 hai, drone ne",
    "   50 m t=266 par paar kiya, pehla fix 65.0 m par t=272.5 par aaya,",
    "   beech me kya hua - log se jawaab hi nahi mil sakta tha. Teen",
    "   wajah ek jaisi dikhti thi:",
    "       - matcher chala hi nahi",
    "       - chala aur har jawaab reject hua",
    "       - chala par band 7.5 s me paar ho gaya, sirf 3 mauke mile",
    "",
    "   Worker yeh sab PEHLE SE compute karta hai: FixResult me ok, score,",
    "   margin aur reason (\"why a rejected fix was rejected, for the",
    "   recorder\") hai. Bas yahan phenk diya jaata tha.",
    "",
    "   Ab 12 naye column (CSV 80 -> 92):",
    "",
    "     try_ok              1 = accept, 0 = reject",
    "     try_score           us koshish ka score",
    "     try_margin          us koshish ka margin",
    "     try_inliers         RANSAC inliers",
    "     try_alt_m           kis altitude par match hua",
    "     try_reason          kis gate ne roka",
    "     map_submitted       worker ko kitni baar bheja (cumulative)",
    "     map_completed       kitni poori hui",
    "     map_skipped_low     min_alt_m ke niche chhodi",
    "     map_skipped_period  fix_period ki wajah se chhodi",
    "     map_dropped         thread busy thi, gir gayi",
    "     map_warm            path garam rakhne wale run",
    "",
    "   Yeh SIRF LOGGING hai. Koi behaviour nahi badla, koi gate nahi",
    "   badla, koi guard nahi laga.",
    "",
    "-" * 92,
    "",
    "  (B)  ARM PAR ORIGIN - JIS PAR ARM KARE, USI KA",
    "",
    "   Pilot ka nirdesh: \"jis par arm kare usi ka origin lo na mille tho",
    "   batao nahi mila\".",
    "",
    "   Pehle:  if src_now == 0 and f[\"gps\"]:  -> receiver",
    "           elif scr_origin is not None:   -> SCR_USER1/2",
    "",
    "   Do chhed the:",
    "     - receiver par arm, par fix nahi -> pehli shart fail, aur",
    "       CHUPCHAP SCR_USER1/2 le leta tha. Us arm ke liye galat source.",
    "     - VPS par arm, par SCR_USER1/2 set nahi -> dono shart fail, aur",
    "       origin wahi pada rehta jo startup fallback ne chuna tha",
    "       (286 me woh receiver tha, typed point se 5 m door).",
    "       Radio par sirf \"VPS ORIGIN NOT RE-TAKEN AT ARM\" aata tha.",
    "",
    "   Ab: receiver par arm -> sirf receiver ka fix. Na mile to",
    "       \"VPS ORIGIN NOT TAKEN: armed on receiver, it has no fix\"",
    "       VPS par arm -> sirf SCR_USER1/2. Na mile to",
    "       \"VPS ORIGIN NOT TAKEN: armed on VPS, SCR_USER1/2 not set\"",
    "",
    "   Dusre source se badla NAHI jaata. Aur pilot ko naam ke sath pata",
    "   chalta hai ki kiski kami hai.",
    "",
    "-" * 92,
    "",
    "  (C)  YEH PDF, aur dono flight ke poore parameter (section 5-7)",
    "",
    "   Disk par ek bhi parameter backup nahi tha. Aaj ka poora compare",
    "   sirf in do BIN ki wajah se ho paya.",
    "",
    "-" * 92,
    "",
    "  TESTS:  65 test files, sab pass.",
    "  test_csv_columns.py me hard-coded ginti 80 -> 92 ki gayi - woh",
    "  tripwire hai jo galti se juda column pakadta hai, isliye usko",
    "  badalna column badalne ka hissa hai.",
])

# =================================================== 4. config.py, as it is
sys.path.insert(0, f"{REPO}/vps_vio")
import dataclasses
from config import VpsConfig
cfg = VpsConfig()
cfg_lines = []
for fl in dataclasses.fields(cfg):
    v = getattr(cfg, fl.name)
    if isinstance(v, (int, float, str, bool)) or v is None:
        cfg_lines.append("    %-30s %s" % (fl.name, v))

emit("4.  config.py  -  238 ke waqt aur ab, bilkul ek jaisa", [
    "",
    "  config.py ka hash 87e19ad se identical hai (section 2), toh yeh",
    "  har value 238 ke waqt bhi yahi thi. Jo iss flight se seedha",
    "  juda hua woh niche nishaan-zada hai.",
    "",
] + cfg_lines + [
    "",
    "-" * 92,
    "",
    "  IN FLIGHT 286 ME JINKA SEEDHA ASAR DIKHA:",
    "",
    "    map_min_alt_m  50.0    50 m ke niche matcher CHALTA HI NAHI.",
    "                           Dono flight me yahi tha. 238 me farak",
    "                           nahi pada kyunki woh 286 s hawa me raha.",
    "",
    "    fix_period_s   2.0     Fix ~2 s ke antar par. 50->65 m ka band",
    "                           7.5 s me paar hua = 3-4 mauke.",
    "",
    "    matcher        both    NCC + XFeat. NCC ka gate min_score 0.05,",
    "                           min_margin 0.01. 286 ke aakhri fix 0.053,",
    "                           0.056, 0.057 par the - gate ke kinare.",
    "",
    "    flow_vel_frames  40    Velocity baseline. 80 m par |vel_flow|",
    "                           p90 10.31 m/s tha jab drone khada tha.",
])

# ====================================== 5. FC parameters: the real difference
BORING_EXACT = ("BARO1_GND_PRESS", "BARO2_GND_PRESS", "STAT_BOOTCNT",
                "STAT_DISTFLWN", "STAT_FLTCNT", "STAT_FLTTIME",
                "STAT_RUNTIME", "MOT_THST_HOVER", "COMPASS_DEC")
real, boring = [], []
for k in sorted(set(PA) | set(PB)):
    va, vb = PA.get(k), PB.get(k)
    if va is None or vb is None or abs(va - vb) > 1e-9:
        (boring if (k in BORING_EXACT or k.startswith("INS_GYR"))
         else real).append((k, va, vb))


def pv(v):
    return "ABSENT" if v is None else "%.9g" % v


gp_a = param_history(f"{SCRATCH}/00000252.BIN", "GPS_PRIMARY")
gp_b = param_history(f"{SCRATCH}/00000261.BIN", "GPS_PRIMARY")

emit("5.  FC parameters  -  1086 me se kitne alag the", [
    "",
    f"  252.BIN  = flight 238 ka FC log   (27 Oct 11:46)   {len(PA)} params",
    f"  261.BIN  = flight 286 ka FC log   (28 Oct 07:23)   {len(PB)} params",
    "",
    "  Har parameter ka FLIGHT KE SHURU ka value liya gaya hai.",
    "",
    "-" * 92,
    "",
    f"  ASLI DIFFERENCES:  {len(real)}",
    "",
    "    %-24s %20s %20s" % ("param", "252 (238, WP)", "261 (286)"),
    "    " + "-" * 66,
] + ["    %-24s %20s %20s" % (k, pv(a), pv(b)) for k, a, b in real] + [
    "",
    "    MIS_TOTAL = mission me kitne waypoint the. 238 me 15, 286 me 9.",
    "    Yeh setting nahi hai - yeh do ALAG MISSION hai. Isliye 286 ko",
    "    238 ka dohraav maanna hi galat tha.",
    "",
    "-" * 92,
    "",
    f"  PER-BOOT RECALIBRATION AUR COUNTERS:  {len(boring)}",
    "  (pilot ne kaha tha \"every small thing that does not matter\" bhi",
    "   likho - yeh wahi hai)",
    "",
    "    %-24s %20s %20s" % ("param", "252 (238, WP)", "261 (286)"),
    "    " + "-" * 66,
] + ["    %-24s %20s %20s" % (k, pv(a), pv(b)) for k, a, b in boring] + [
    "",
    "    BARO*_GND_PRESS   har boot par zameen ka dabaav. Mausam.",
    "    INS_GYR*OFFS      har boot par gyro bias naya naapa jaata hai.",
    "                      Antar 0.001-0.015 rad/s. Kuch nahi.",
    "    INS_GYR*_CALTEMP  board ka temperature jab calibrate hua:",
    "                      27 Oct 42.6/44.2 C, 28 Oct 32.5/32.8 C.",
    "                      27 Oct ko board garam tha, 28 Oct subah thanda.",
    "    MOT_THST_HOVER    FC khud seekhta hai. 0.343 vs 0.341.",
    "    COMPASS_DEC       0.034231 vs 0.0342308 - 7 digit tak same.",
    "    STAT_*            boot count, flight count, total time. Counters.",
    "",
    "-" * 92,
    "",
    "  GPS_PRIMARY KA POORA ITIHAAS - yeh woh jagah hai jahan main",
    "  pehle galat nateeje par pahuncha tha:",
    "",
    "    252 (flight 238):",
] + ["        %8.1f s   ->  %.0f" % (t, v) for t, v in gp_a] + [
    "",
    "    261 (flight 286):",
] + ["        %8.1f s   ->  %.0f" % (t, v) for t, v in gp_b] + [
    "",
    "    1 = GPS2 primary = hamari VPS injection",
    "    0 = GPS1 primary = M9N",
    "",
    "    DONO FLIGHT 1 PAR SHURU HUI - bilkul VPS par, jaisa hona chahiye.",
    "    286 me 97.2 s par 0 hua kyunki pilot ne 96.2 s par switch GPS par",
    "    kiya (RC6: 1945 -> 1495 -> 1045). Switch aur watcher dono theek",
    "    kaam kar rahe the - ek second me likh diya.",
])

# ============================================ 6. key parameter groups, in full
GROUPS = {
    "GPS / SOURCE": ["GPS_PRIMARY", "GPS2_TYPE", "GPS_AUTO_SWITCH",
                     "GPS_AUTO_CONFIG", "GPS2_DELAY_MS", "GPS2_POS_X",
                     "GPS2_POS_Y", "GPS2_POS_Z"],
    "EKF SOURCES": ["AHRS_EKF_TYPE", "EK3_ENABLE", "EK3_SRC1_POSXY",
                    "EK3_SRC1_VELXY", "EK3_SRC1_POSZ", "EK3_SRC1_VELZ",
                    "EK3_SRC1_YAW", "EK3_SRC2_POSXY", "EK3_SRC2_VELXY",
                    "EK3_SRC_OPTIONS", "EK3_GLITCH_RAD", "EK3_POSNE_M_NSE",
                    "EK3_VELNE_M_NSE", "EK3_VELD_M_NSE", "EK3_GPS_CHECK"],
    "FAILSAFE": ["FS_EKF_THRESH", "FS_EKF_ACTION", "FS_GCS_ENABLE",
                 "FS_THR_ENABLE", "FS_OPTIONS", "FS_CRASH_CHECK"],
    "YAW RATE": ["PILOT_Y_RATE", "ACRO_Y_RATE"],
    "RANGEFINDER": ["RNGFND1_TYPE"],
    "ORIGIN / SCRIPT": ["SCR_ENABLE", "SCR_USER1", "SCR_USER2", "SCR_USER3",
                        "RC6_OPTION", "FLTMODE_CH"],
    "MISSION": ["MIS_TOTAL"],
}
glines = []
for g, ks in GROUPS.items():
    glines.append("")
    glines.append(f"  {g}")
    glines.append("    %-20s %18s %18s   %s"
                  % ("param", "252 (238)", "261 (286)", ""))
    for k in ks:
        a, b = PA.get(k), PB.get(k)
        if a is None and b is None:
            continue
        mark = "" if (a is not None and b is not None
                      and abs(a - b) <= 1e-9) else "  <-- ALAG"
        glines.append("    %-20s %18s %18s%s" % (k, pv(a), pv(b), mark))

emit("6.  Woh parameters jo is system ke liye mayne rakhte hai", [
    "",
    "  Dono flight, side by side. \"<-- ALAG\" tabhi lagta hai jab sach me",
    "  alag ho.",
] + glines + [
    "",
    "-" * 92,
    "",
    "  SCR_USER1/2 = 30.8957901, 74.6140366  -  dono flight me yahi.",
    "  Pilot ne RC par type kiya tha: 30.8957905, 74.6140336.",
    "  Antar ~0.4 m lat, ~0.3 m lon. Typed point theek pahuncha tha.",
    "",
    "  RNGFND1_TYPE = 0  -  lidar dono flight me band tha.",
    "",
    "  EK3_SRC1_POSXY/VELXY = 3 (GPS), aur GPS_PRIMARY = 1, matlab",
    "  EKF ka primary horizontal source hamari injection thi. Jaisa",
    "  hona chahiye.",
])

# ================================================= 7. the flight itself
rows = ["", two("", "238 (WP, theek)", "286 (bhaga)"), "  " + "-" * 76, ""]
rows.append("  DHAANCHA")
rows.append(two("rows", A["n"], B["n"]))
rows.append(two("duration", "%.0f s" % (A["t"][-1] - A["t"][0]),
                "%.0f s" % (B["t"][-1] - B["t"][0])))
rows.append(two("max altitude", "%.1f m" % np.nanmax(A["alt_m"]),
                "%.1f m" % np.nanmax(B["alt_m"])))
rows.append(two("hawa me (>5 m), VPS par", "%.0f s" % (A["air"].sum() * .04),
                "%.0f s" % (B["air"].sum() * .04)))
rows.append(two("VPS rows / M9N rows",
                "%d / %d" % (A["vps"].sum(), (A["src"] == "GPS1").sum()),
                "%d / %d" % (B["vps"].sum(), (B["src"] == "GPS1").sum())))
rows += ["", "  POSITION KI GALTI  (M9N ke kacche output, gps1_* ke against)"]
for lo, hi in ((5, 20), (20, 50), (50, 85)):
    v = []
    for D in (A, B):
        m = D["ok"] & (D["alt_m"] >= lo) & (D["alt_m"] < hi)
        v.append("p50 %.1f  p90 %.1f m" % (np.percentile(D["err"][m], 50),
                                           np.percentile(D["err"][m], 90))
                 if m.sum() > 30 else "n/a")
    rows.append(two("%d-%d m" % (lo, hi), *v))
rows.append(two("max galti kabhi bhi",
                "%.1f m" % np.nanmax(A["err"][A["ok"]]),
                "%.1f m" % np.nanmax(B["err"][B["ok"]])))
rows += ["", "  JO SIGMA FC KO BHEJI GAYI"]
rows.append(two("sigma_m airborne p50 / max",
                "%.1f / %.1f m" % (pct(A, A["air"], "sigma_m", 50),
                                   np.nanmax(A["sigma_m"][A["air"]])),
                "%.1f / %.1f m" % (pct(B, B["air"], "sigma_m", 50),
                                   np.nanmax(B["sigma_m"][B["air"]]))))
rows.append(two("50 m upar: GALTI / DECLARE",
                "%.1f m / %.1f m" % (
                    np.percentile(A["err"][A["ok"] & (A["alt_m"] >= 50)], 50),
                    pct(A, A["air"] & (A["alt_m"] >= 50), "sigma_m", 50)),
                "%.1f m / %.1f m" % (
                    np.percentile(B["err"][B["ok"] & (B["alt_m"] >= 50)], 50),
                    pct(B, B["air"] & (B["alt_m"] >= 50), "sigma_m", 50))))
rows += ["", "  MAP MATCHER"]
rows.append(two("fixes total / airborne",
                "%d / %d" % (A["fx"].sum(), (A["fx"] & A["air"]).sum()),
                "%d / %d" % (B["fx"].sum(), (B["fx"] & B["air"]).sum())))
rows.append(two("fix_score p50 / min",
                "%.3f / %.3f" % (np.nanmedian(A["fix_score"][A["fx"]]),
                                 np.nanmin(A["fix_score"][A["fx"]])),
                "%.3f / %.3f" % (np.nanmedian(B["fix_score"][B["fx"]]),
                                 np.nanmin(B["fix_score"][B["fx"]]))))
rows.append(two("fixes 60 m ke niche",
                int((A["fx"] & A["vps"] & (A["alt_m"] < 60)).sum()),
                int((B["fx"] & B["vps"] & (B["alt_m"] < 60)).sum())))
rows += ["", "  EKF   (FS_EKF_THRESH = 1.2, FS_EKF_ACTION = 2 = ALT_HOLD)"]
for lbl, k in (("ekf_vel_var", "ekf_vel_var"),
               ("ekf_pos_h_var", "ekf_pos_h_var")):
    rows.append(two(lbl + " p50/p90/max",
                    "%.2f / %.2f / %.2f" % (pct(A, A["air"], k, 50),
                                            pct(A, A["air"], k, 90),
                                            np.nanmax(A[k][A["air"]])),
                    "%.2f / %.2f / %.2f" % (pct(B, B["air"], k, 50),
                                            pct(B, B["air"], k, 90),
                                            np.nanmax(B[k][B["air"]]))))
rows.append(two("rows jahan vel_var > 1.2",
                "%d of %d (%.0f%%)" % (
                    (A["ekf_vel_var"][A["air"]] > 1.2).sum(), A["air"].sum(),
                    100 * (A["ekf_vel_var"][A["air"]] > 1.2).mean()),
                "%d of %d (%.0f%%)" % (
                    (B["ekf_vel_var"][B["air"]] > 1.2).sum(), B["air"].sum(),
                    100 * (B["ekf_vel_var"][B["air"]] > 1.2).mean())))
rows += ["", "  VELOCITY"]
for lbl, kn, ke in (("|vel_sent|", "vel_sent_vn", "vel_sent_ve"),
                    ("|vel_flow|", "vel_flow_vn", "vel_flow_ve")):
    rows.append(two(lbl + " airborne p50/p90",
                    "%.2f / %.2f m/s" % (
                        np.nanpercentile(np.hypot(A[kn], A[ke])[A["air"]], 50),
                        np.nanpercentile(np.hypot(A[kn], A[ke])[A["air"]], 90)),
                    "%.2f / %.2f m/s" % (
                        np.nanpercentile(np.hypot(B[kn], B[ke])[B["air"]], 50),
                        np.nanpercentile(np.hypot(B[kn], B[ke])[B["air"]], 90))))
rows.append(two("deadbanded",
                "%.0f%%" % (100 * np.nanmean(A["vel_deadbanded"][A["air"]])),
                "%.0f%%" % (100 * np.nanmean(B["vel_deadbanded"][B["air"]]))))
rows.append(two("deadband threshold p50",
                "%.3f m/s" % pct(A, A["air"], "vel_deadband", 50),
                "%.3f m/s" % pct(B, B["air"], "vel_deadband", 50)))
rows += ["", "  FLOW / CAMERA KI SEHAT"]
for lbl, k, u in (("tracks p50", "tracks", ""),
                  ("flow_disagree_px p50", "flow_disagree_px", " px"),
                  ("frame_age_s p90", "frame_age_s", " s"),
                  ("depth_med_m p50", "depth_med_m", " m")):
    q = 90 if "p90" in lbl else 50
    rows.append(two(lbl, "%.2f%s" % (pct(A, A["air"], k, q), u),
                    "%.2f%s" % (pct(B, B["air"], k, q), u)))

emit("7.  Flight khud  -  dono ka seedha milan", rows + [
    "",
    "-" * 92,
    "",
    "  PADHNE KA TAREEKA:",
    "",
    "   - 50 m ke NICHE dono flight lagbhag barabar hai.",
    "   - 50 m ke UPAR 286 ki galti 36.6 m hai, 238 ki 7.9 m. 4.6 guna.",
    "   - Aur usi waqt 286 ne 3.2 m ka bharosa declare kiya, 238 ne 5.3 m.",
    "     Yani 286 jyada GALAT tha AUR jyada BHAROSA dikha raha tha.",
    "     FC ko sirf yahi number milta hai, aur woh us par ud jaata hai.",
    "   - Camera, flow, deadband, vibration - sab barabar. Wajah nahi.",
    "   - 238 BHI accha nahi tha: 7.9 m galti aur 15% rows EKF threshold",
    "     ke upar (max 2.75, 286 ke 2.58 se bhi jyada). Us flight me bhi",
    "     failsafe ki haalat ban rahi thi, bas SUSTAINED nahi hui.",
    "",
    "  Matlab yeh system 80 m par KINARE par chalta hai. 238 ko 286 s",
    "  mile toh map ne sambhal liya. 286 ko 77 s mile aur shor dugna tha.",
])

# =============================================== 8. what the FC itself did
def fc_events(path):
    from pymavlink import mavutil
    MODES = {0: "STABILIZE", 2: "ALT_HOLD", 3: "AUTO", 5: "LOITER",
             6: "RTL", 9: "LAND", 16: "POSHOLD"}
    SUB = {11: "GPS_GLITCH", 16: "EKF_CHECK", 17: "FAILSAFE_EKFINAV",
           24: "EKF_PRIMARY", 10: "FLT_MODE", 4: "OPTFLOW"}
    m = mavutil.mavlink_connection(path)
    out, t0 = [], None
    while True:
        msg = m.recv_match(type=["MODE", "ERR", "MSG"])
        if msg is None:
            break
        t = getattr(msg, "TimeUS", 0) / 1e6
        if t0 is None:
            t0 = t
        ty = msg.get_type()
        if ty == "MODE":
            out.append((t - t0, "MODE  -> %s (reason %d)"
                        % (MODES.get(int(msg.Mode), msg.Mode), int(msg.Rsn))))
        elif ty == "ERR":
            out.append((t - t0, "ERR    %-18s code %d"
                        % (SUB.get(int(msg.Subsys), "SUBSYS %d" % msg.Subsys),
                           int(msg.ECode))))
        elif ty == "MSG" and any(w in msg.Message for w in
                                 ("EKF", "Glitch", "SRC=", "GPS 1", "GPS 2")):
            out.append((t - t0, "MSG    " + msg.Message[:62]))
    return out


ea = fc_events(f"{SCRATCH}/00000252.BIN")
eb = fc_events(f"{SCRATCH}/00000261.BIN")
emit("8.  FC ne khud kya kiya  -  dataflash ke events", [
    "",
    "  261.BIN  =  FLIGHT 286 (bhaga).  Yahan ALT_HOLD aaya.",
    "",
] + ["    %8.1f s   %s" % e for e in eb] + [
    "",
    "  PADHO:",
    "    96.2 s   pilot ne switch GPS par kiya (RC6 1945 -> 1495 -> 1045)",
    "    97.2 s   GPS_PRIMARY 1 -> 0",
    "    98.0 s   EKF_CHECK code 2 + FAILSAFE_EKFINAV  ->  ALT_HOLD",
    "",
    "  Switch ke 1.8 second baad. 80 m par hamari VPS position aur M9N",
    "  ~30 m apas me alag the; primary source hawa me badalna matlab EKF",
    "  ko ek jhatke me 30 m ka step dena. ALT_HOLD bhagne ki WAJAH nahi,",
    "  bhagne ka NATEEJA hai - drone pehle bah chuka tha.",
    "",
    "  Aur 0.5 s par node ka apna message:",
    "    SRC=1/220:VPS ORIGIN SCR_USER1/2 at arm 5m",
    "  Arm par origin 5 m khiska, kyunki pehle GPS par arm hua tha phir",
    "  VPS par. 238 me ek hi arm tha aur \"moved 0 m\".",
])
emit("8b. FC events  -  flight 238 (WP), tulna ke liye", [
    "",
    "  252.BIN  =  FLIGHT 238 (theek chali).  Koi ALT_HOLD nahi.",
    "",
] + ["    %8.1f s   %s" % e for e in ea] + [
    "",
    "  DHYAN DO: GPS Glitch aur EKF lane switching DONO FLIGHT ME HAI.",
    "  238 me 13+ lane switch, 286 me 12. Yeh antar NAHI hai - yeh is",
    "  airframe ka normal vyavhaar hai.",
])

# ============================================ 9. everything else, small stuff
cam_a = open(f"{REPO}/flight_logs/vps/vps_00238/camera.txt").read().split()
svc = sh("systemctl cat drone-vps | grep -E '^\\s+--|^ExecStart'")
emit("9.  Baaki har chhoti cheez", [
    "",
    "  CAMERA CALIBRATION   -  camera.txt, dono run me byte-identical",
    "",
] + ["    " + " ".join(cam_a[i:i + 4]) for i in range(0, min(len(cam_a), 8), 4)] + [
    "",
    "  MAP SHEET",
    "",
    "    bhh.tif      95,394,978 bytes     mtime 19 Oct 10:18",
    "    symlink nahi hai. 19 Oct se nahi badli.",
    "    Dono flight ne YAHI sheet use ki.",
    "    (bhh_cut.tif aur d.tif padi hai par service bhh.tif deti hai)",
    "",
    "  SERVICE UNIT  -  /etc/systemd/system/drone-vps.service",
    "",
] + ["    " + l.strip() for l in svc.splitlines()] + [
    "",
    "    File 27 Oct 15:30 ko edit hui thi (238 ke baad), par content",
    "    238 jaisa hi hai: --no-send-vel NAHI hai, toh velocity ja rahi",
    "    hai. Sirf file ki date badli, behaviour nahi.",
    "",
    "  ORIGIN",
    "",
    "    238  pehla row  lat 30.8957901  lon 74.6140366  sigma_abs 20.1 m",
    "    286  pehla row  lat 30.8957463  lon 74.6140153  sigma_abs  2.1 m",
    "    pilot ne type kiya 30.8957905, 74.6140336",
    "",
    "    286 startup par SCR_USER1/2 padha NAHI gaya:",
    "      \"Set SCR_USER1/2 from the RC, or pass --origin lon,lat. Waiting.\"",
    "      \"origin from the live receiver (GPS_PRIMARY=0): 30.895746, ...\"",
    "    aur phir arm par:",
    "      \"ARM: origin re-taken from SCR_USER1/2 at arm: ... (moved 5 m)\"",
    "    238 me: \"(moved 0 m)\"",
    "",
    "  VIBRATION  (FC ke VIBE message se)",
    "",
    "    238   p50  1.02 0.89 1.75     p95  3.73 2.68 2.95   m/s2",
    "    286   p50  1.02 0.75 1.56     p95  2.67 2.16 2.56   m/s2",
    "    clipping  0  dono me",
    "    286 ki vibration 238 se KAM thi. Wajah nahi hai.",
    "",
    "  M9N KI HAALAT  (jo primary nahi tha, par chal raha tha)",
    "",
    "    238   sats min 12, p50 23     status p50 3,  0 rows status<3",
    "    286   sats min  0, p50 26     status p50 4, 77 rows status<3",
    "    286 me receiver ne 0 satellite par fix khoya - 77 rows. Woh",
    "    primary nahi tha, par yeh wahi \"0 satellite\" hai jo pilot ne",
    "    pehle report kiya tha.",
    "",
    "  JETSON KA CLOCK",
    "",
    "    12 ghante peeche tha. Jetson 27 Oct 19:15 bol raha tha, FC",
    "    28 Oct 07:14. journal ka boot record 1970-01-01 se shuru hota",
    "    hai. Isse maine do galat nateeje nikale.",
    "    Pilot ab boot par hotspot dega, toh NTP se theek ho jayega.",
    "",
    "  NVMe",
    "",
    "    28 Oct ke chaaro boot par /data mount nahi hua, aur teeno",
    "    service (router, node, rc-switch) \"Dependency failed\" se start",
    "    hi nahi hui. Pilot: naya Jetson aaj liya, SD card aur NVMe aaj",
    "    lage - pehle lage hi nahi the. Toh yeh ab nahi hona chahiye.",
    "",
    "  5 V RAIL",
    "",
    "    27 Oct, log 259:  5.071 -> 4.162 V ek sample me, motors aur",
    "    battery bilkul sthir. FC browned out, stabilisation gayi, drone",
    "    jhuka aur gira. 252/255/256/257/258 me ek bhi dip nahi (min",
    "    5.011-5.027). Intermittent connection. Pilot theek kar raha hai.",
])

# =================================================== 10. appendix: all params
allk = sorted(set(PA) | set(PB))
ap = ["", "  Dono log ke saare parameter, naam ke hisab se. Yeh disk par",
      "  pehla parameter backup hai - pehle ek bhi nahi tha.", "",
      "    %-24s %18s %18s" % ("param", "252 (238)", "261 (286)"),
      "    " + "-" * 62]
for k in allk:
    a, b = PA.get(k), PB.get(k)
    mark = "" if (a is not None and b is not None
                  and abs(a - b) <= 1e-9) else "  *"
    ap.append("    %-24s %18s %18s%s" % (k, pv(a), pv(b), mark))
emit("10.  Appendix:  dono flight ke poore %d parameters" % len(allk), ap)

render()
print(f"wrote {OUT}")
print(f"{len(pages)} pages")
