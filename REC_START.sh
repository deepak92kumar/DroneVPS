#!/bin/bash
# Start flight recording in the background and PROVE it is working before takeoff.
cd /mnt/ssd/DRONE
NAME="${1:-flight$(date +%H%M)}"
OUT="/mnt/ssd/DRONE/flights/$NAME"
LOG="/mnt/ssd/DRONE/flights/$NAME.log"
if pgrep -f "venv/bin/python.*[r]ecord_flight" >/dev/null; then
  echo "!! recording pehle se chal rahi hai — pehle ./REC_STOP.sh chalao"; exit 1
fi
mkdir -p /mnt/ssd/DRONE/flights
nohup vps_vio/.venv/bin/python vps_vio/record_flight.py --out "$OUT" --seconds 3600 > "$LOG" 2>&1 &
PID=$!
echo "starting (pid $PID) ... 12 sec wait"
sleep 12
N=$(ls "$OUT/mono" 2>/dev/null | wc -l)
GPS=$(tail -5 "$LOG" 2>/dev/null | grep -o 'sats=[0-9]*' | tail -1)
echo
if ! kill -0 $PID 2>/dev/null; then
  echo "########################################"
  echo "   RECORDING FAIL — MAT UDAO"
  echo "########################################"
  tail -15 "$LOG"; exit 1
fi
if [ "$N" -lt 3 ]; then
  echo "########################################"
  echo "   FRAMES NAHI BAN RAHE ($N) — MAT UDAO"
  echo "########################################"
  tail -15 "$LOG"; exit 1
fi
echo "########################################"
echo "   RECORDING CHAL RAHI HAI  ✓"
echo "   frames : $N  (aur badh rahe hain)"
echo "   folder : $OUT"
echo "   GPS    : ${GPS:-telemetry aa rahi hai}"
echo "   AB UDA SAKTE HO"
echo "########################################"
