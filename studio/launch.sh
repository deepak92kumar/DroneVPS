#!/usr/bin/env bash
# Open DroneVPS Studio, starting it first if it is not already running.
#
# This is what the desktop icon runs. It is deliberately idempotent: double
# clicking the icon twice must open a second tab, not a second server fighting
# for the port.
set -u
PORT="${DRONEVPS_STUDIO_PORT:-8099}"
DIR="/data/DRONE"
LOG="$DIR/studio/studio.log"
URL="http://127.0.0.1:$PORT/"

up() { curl -fsS -o /dev/null --max-time 2 "http://127.0.0.1:$PORT/api/sheets"; }

if ! up; then
  cd "$DIR" || exit 1
  echo "$(date '+%F %T') starting the studio on port $PORT" >> "$LOG"
  nohup python3 studio/server.py --port "$PORT" --host 0.0.0.0 >> "$LOG" 2>&1 &
  for _ in $(seq 1 60); do
    up && break
    sleep 0.25
  done
fi

if ! up; then
  # Say so on the screen rather than failing silently behind an icon.
  msg="DroneVPS Studio did not start. The last lines of $LOG:

$(tail -n 15 "$LOG" 2>/dev/null)"
  if command -v zenity >/dev/null; then zenity --error --width=600 --text="$msg"
  else echo "$msg" >&2; fi
  exit 1
fi

xdg-open "$URL" >/dev/null 2>&1 || python3 -m webbrowser "$URL" >/dev/null 2>&1 &
exit 0
