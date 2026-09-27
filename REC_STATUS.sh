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
echo
if [ $OK -eq 1 ]; then echo "########  SAB THEEK - UDA SAKTE HO  ########"
else echo "########  ABHI MAT UDAO  ########"; fi
