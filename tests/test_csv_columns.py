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
check("and there are the columns we expect", len(CSV_COLUMNS) == 44,
      str(len(CSV_COLUMNS)))
check("the raw receiver's three are the last three",
      CSV_COLUMNS[-3:] == ["gps1_north_m", "gps1_east_m", "gps1_err_m"],
      str(CSV_COLUMNS[-3:]))

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
