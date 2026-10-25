"""The estimate log's header and its rows must be the same width.

20 Oct: seven columns were added to the ROW writer and not to the header. The
file opened perfectly well. Every data row was 39 fields wide under a 32-field
header, so everything past the 32nd column was attributed to the wrong name -
1162 rows of it were written on the aircraft before anyone counted.

That is a worse failure than a file which refuses to open, because nothing
complains and the numbers look plausible. It had already happened once in the
same week (56 against 30) and was caught then by chance.

So the header and the row are defined together in live_node and this runs both.
"""
import csv
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "vps_vio"))
from live_node import CSV_COLUMNS, csv_row  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        fails.append(name)


class Vel:
    vn = ve = speed_mps = sent_vn = sent_ve = 0.0
    deadband_mps = 0.35
    age_s = 0.0
    bias_n = bias_e = 0.0
    flow_updates = 7
    valid = True
    deadbanded = False
    bias_frozen = True
    flow_vn = flow_ve = accel_n = accel_e = 0.0
    reason = "ok"


def est(**kw):
    d = dict(t=12.34, north=1.0, east=-2.0, lon=74.613182, lat=30.896742,
             sigma=4.5, sigma_abs=20.5, tracks=250, fix_used=False, alt=110.0, yaw=3.2,
             fc_d_roll=0.01, fc_d_pitch=-0.02, gyro_d_roll=0.011,
             gyro_d_pitch=-0.019, gyro_n=500, accel_n=480,
             depth_med_m=12.0, depth_dropped=3, n_dr=1, src="VPS")
    d.update(kw)
    return d


print("the header and a full row are the same width")
row = csv_row(est(), dict(rngfnd=0.26, alt_amsl=300.7), Vel(), 1.0, 2.0, 0.5)
# Added 24 Oct, after three columns went into the header after quat_n and
# into csv_row after gyro_d_roll. The widths still matched, so the check
# below passed while every value from gyro_d_pitch onward was written under
# the wrong heading. Equal length is not alignment. Each field here gets a
# value unlike all the others, and is then looked for under its own column.
_sent = dict(fc_d_roll=0.1111, fc_d_pitch=0.2222, gyro_d_roll=0.3333,
             gyro_d_pitch=0.4444, gyro_n=55, quat_d_roll=0.6666,
             quat_d_pitch=0.7777, quat_n=88, gyro_drain_roll=0.9999,
             gyro_drain_pitch=1.1111, gyro_win_s=0.0404)
_row = csv_row(est(**_sent), dict(rngfnd=0.26, alt_amsl=300.7), Vel(), 1.0, 2.0, 0.5)
_bad = []
for _c, _want in _sent.items():
    _got = _row[CSV_COLUMNS.index(_c)]
    _ok = (str(_got) == str(_want) if isinstance(_want, int)
           else abs(float(_got) - _want) < 1e-6 if _got else False)
    if not _ok:
        _bad.append(f"{_c}: wanted {_want}, column holds {_got!r}")
check("each derotation value lands under its own heading", not _bad,
      "; ".join(_bad) if _bad else f"{len(_sent)} fields, all aligned")

check("widths match", len(row) == len(CSV_COLUMNS),
      f"header {len(CSV_COLUMNS)}, row {len(row)}")
# 39 until 20 Oct, when frame_file was added: live_node now saves camera
# frames during a live run (the OAK-D has one owner, so the only process that
# can record them is the one navigating), and that column is what ties a saved
# image to the row holding the heading and height it was taken at.
#
# 41 since sigma_abs_m joined it the same day. sigma_m is the fuser's own
# number - the uncertainty of the DISPLACEMENT - and sigma_abs_m is what the
# flight controller is actually told, with the origin's own error folded in.
# Logging both is the only way a flight can show which of the two was wrong:
# on the three stale-origin flights of 20 Oct the drift figure read 14 m
# against a true error of 151 m, all of it the origin's.
#
# 44 since 21 Oct, when the RAW receiver got its own three. gps_north_m /
# gps_east_m / err_m are not the GPS - they come from GLOBAL_POSITION_INT,
# the EKF's fused position, which on a VPS flight is partly our own injection
# returning. gps1_* are GPS_RAW_INT, the receiver before the EKF touches it,
# and they are the only independent truth in the file.
#
# 45 once n_step_rej joined them: position steps refused as physically
# impossible, cumulative. The position is ACCUMULATED, so one bad flow frame is
# a permanent offset rather than one bad reading - vps_00016 carries a
# single-sample 20.55 m step to the end of the flight.
#
# 49 once the EKF's own innovation test ratios joined them. The recorder had
# read them since it was written and nothing logged them, so FS_EKF_THRESH -
# 1.2 on this aircraft against ArduPilot's documented 0.8 default - had been
# argued about rather than measured.
#
# 52 once GAME_ROTATION_VECTOR's per-frame delta joined them. Derotation has
# three possible sources now and the CSV carries all three side by side -
# fc_d_*, gyro_d_*, quat_d_* - so one flight settles which to fly on.
#
# 55 once derotation started saying which of its three sources ran on each
# frame, and how far the two independent ones disagreed when both were there.
#
# 59 once the velocity bridge's two INPUTS joined them - the flow velocity it
# was handed and the acceleration it integrated. Log 225 showed the blend was
# doubling the reported speed and halving its correlation with truth, and then
# could not be replayed at a different tau_s because neither input had been
# written down. The gain had to be reasoned instead of swept. That is the
# second time a velocity fault has been undiagnosable from the recording.
# 64 once the map fix's OWN position joined them. fix_used recorded only that
# a fix had been applied; where the matcher actually put the aircraft was
# never written down. max_correction_m caps every fix at 0.5 m, so a fix 40 m
# out and a fix 2 m out leave the same trace in the fused position - which
# meant the matcher's error by height could only be taken from an archived
# report whose configuration carried an 8.61 m east bias this code never had.
# 66 once the step gate's INPUT joined them. n_step_rej counted the bites and
# nothing recorded what was bitten: a refused step enters the position as zero
# and a clipped one at exactly the ceiling, so the log could not tell a
# justified refusal from a lost real step. Post-gate the accepted steps run
# p50 3.4-4.9 px against a 50 px ceiling, and at 85 m a real 3 m/s is 0.93 px
# per frame, so the question is whether the top few percent were noise at all.
# 68 -> 70 on 24 Oct: frame_age_s and frames_skipped, when the reader started
# draining the queue to the newest frame instead of taking the oldest.
# 66 -> 68 on 23 Oct: flow_disagree_px and flow_dual, when the partner camera
# got its own tracker. The count is pinned on purpose - this file exists
# because seven columns were once added to the ROW and not to the header, and
# 1162 rows were written with everything past the 32nd attributed to the wrong
# name. Adding a column here is meant to be a decision, not an accident.
check("and there are the columns we expect", len(CSV_COLUMNS) == 79,
      str(len(CSV_COLUMNS)))
i = CSV_COLUMNS.index("fc_d_roll")
check("the three derotation opinions sit together",
      CSV_COLUMNS[i:i + 8] == ["fc_d_roll", "fc_d_pitch",
                               "gyro_d_roll", "gyro_d_pitch", "gyro_n",
                               "quat_d_roll", "quat_d_pitch", "quat_n"],
      str(CSV_COLUMNS[i:i + 8]))
i = CSV_COLUMNS.index("gps1_north_m")
check("the raw receiver's three are together, wherever they sit",
      CSV_COLUMNS[i:i + 3] == ["gps1_north_m", "gps1_east_m", "gps1_err_m"],
      str(CSV_COLUMNS[i:i + 3]))

print("\nand with every optional value missing, which is the usual case")
e = est(fc_d_roll=None, fc_d_pitch=None, gyro_d_roll=None, gyro_d_pitch=None,
        depth_med_m=None, src=None)
row = csv_row(e, {}, Vel(), None, None, None)
check("widths still match", len(row) == len(CSV_COLUMNS),
      f"header {len(CSV_COLUMNS)}, row {len(row)}")
check("missing values are empty, not the string None",
      "None" not in [str(x) for x in row], str([x for x in row if str(x) == "None"]))

print("\nthe names are usable as column names")
check("no duplicates", len(set(CSV_COLUMNS)) == len(CSV_COLUMNS),
      str([c for c in CSV_COLUMNS if CSV_COLUMNS.count(c) > 1]))
bad = [c for c in CSV_COLUMNS if not c or " " in c or "," in c]
check("no spaces or commas in a name", not bad, str(bad))

print("\nthe columns that were missed are actually present")
for c in ("accel_n", "depth_med_m", "depth_dropped", "n_dr", "src",
          "rngfnd_m", "alt_amsl_m", "frame_file"):
    check(f"{c} is in the header", c in CSV_COLUMNS)

print("\na written file parses back at a single width")
buf = io.StringIO()
w = csv.writer(buf)
w.writerow(CSV_COLUMNS)
for i in range(5):
    w.writerow(csv_row(est(t=i), dict(rngfnd=0.26), Vel(), None, None, None))
rows = list(csv.reader(io.StringIO(buf.getvalue())))
widths = set(len(r) for r in rows)
check("every line has the same number of fields", len(widths) == 1, str(widths))
check("and it round-trips into a dict reader",
      list(csv.DictReader(io.StringIO(buf.getvalue())))[0]["src"] == "VPS")

print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
