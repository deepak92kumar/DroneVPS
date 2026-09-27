#!/bin/bash
OUT=$(pgrep -af "venv/bin/python.*[r]ecord_flight" | grep -o '\-\-out [^ ]*' | awk '{print $2}' | head -1)
pkill -INT -f "venv/bin/python.*[r]ecord_flight" 2>/dev/null; sleep 3; pkill -f "venv/bin/python.*[r]ecord_flight" 2>/dev/null
if [ -n "$OUT" ] && [ -d "$OUT" ]; then
  N=$(ls "$OUT/mono" 2>/dev/null | wc -l)
  ALT=$(awk -F, 'NR>1 && $8!="" {if($8+0>m) m=$8+0} END{printf "%.1f",m}' "$OUT/index.csv" 2>/dev/null)
  echo "RECORDING BAND"
  echo "  frames   : $N"
  echo "  max alt  : ${ALT:-?} m"
  echo "  folder   : $OUT"
else
  echo "koi recording chal nahi rahi thi"
fi
