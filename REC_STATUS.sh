#!/bin/bash
# Pre-flight gate for the map-matching recording. Prints one clear verdict.
# A wasted flight cost us a whole evening once: frames were never written and
# nobody checked. So this refuses to say "go" unless ALL THREE are true:
# frames growing, a real 3D GPS fix, and altitude arriving from the FC.
D=$(ls -dt /mnt/ssd/DRONE/flights/auto_* 2>/dev/null | head -1)
[ -z "$D" ] && { echo "!! koi recording folder nahi - service band hai?"; systemctl is-active drone-record; exit 1; }
A=$(ls "$D/mono" 2>/dev/null | wc -l); sleep 4; B=$(ls "$D/mono" 2>/dev/null | wc -l)
L=$(tail -1 "$D/index.csv" 2>/dev/null)
LAT=$(echo "$L" | cut -d, -f6); ALT=$(echo "$L" | cut -d, -f8)
FIX=$(echo "$L" | cut -d, -f14); SATS=$(echo "$L" | cut -d, -f15)
echo "folder  : $D"
echo "frames  : $B  (4 sec mein +$((B-A)))"
echo "service : $(systemctl is-active drone-record)"
echo
OK=1
if [ $((B-A)) -lt 2 ]; then echo "  FRAMES   : RUKE HUE HAIN  X"; OK=0
else echo "  FRAMES   : badh rahe hain  OK"; fi

if [ -z "$LAT" ]; then
  echo "  TELEMETRY: NAHI AA RAHI  X   <- drone ko power do / cable check"; OK=0
elif [ "$LAT" = "0.0" ] || [ -z "$SATS" ] || [ "${SATS:-0}" -lt 6 ] || [ "${FIX:-0}" -lt 3 ]; then
  echo "  GPS      : FIX NAHI MILA  X   sats=${SATS:-0} fix=${FIX:-0}  <- khule aasman mein 1-2 min ruko"; OK=0
else
  echo "  GPS      : lat=$LAT sats=$SATS fix=$FIX  OK"
fi

if [ -z "$ALT" ]; then echo "  ALTITUDE : nahi aa rahi  X"; OK=0
else echo "  ALTITUDE : $ALT m  OK"; fi

# The first flight logged 545 frames that were 100% white: indoor exposure used
# in daylight. Frames existing is not the same as frames being usable, so the
# newest one is inspected here. Run this OUTSIDE, where the flight will happen.
IMG=$(ls -t "$D"/mono/*.png "$D"/mono/*.jpg 2>/dev/null | head -1)
if [ -z "$IMG" ]; then echo "  IMAGE    : koi frame nahi  X"; OK=0
else
  R=$(/mnt/ssd/DRONE/vps_vio/.venv/bin/python - "$IMG" <<'PY2'
import cv2, sys
im = cv2.imread(sys.argv[1], 0)
if im is None:
    print("BAD 0 0"); raise SystemExit
print(f"{'OK' if 100*(im>250).mean()<40 and im.std()>10 else 'BAD'} "
      f"{100*(im>250).mean():.0f} {im.std():.0f}")
PY2
)
  set -- $R
  if [ "$1" = "OK" ]; then echo "  IMAGE    : white=$2% detail=$3  OK"
  else echo "  IMAGE    : KHARAB  X   white=$2% detail=$3  <- jali hui / kaali hai"; OK=0; fi
fi
echo
if [ $OK -eq 1 ]; then echo "########  SAB THEEK - UDA SAKTE HO  ########"
else echo "########  ABHI MAT UDAO  ########"; fi
