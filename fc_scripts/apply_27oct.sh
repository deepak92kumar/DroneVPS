#!/usr/bin/env bash
# Put the 27 October parameters on the FC, in the one order that is safe here.
#
# The FC is not on USB - only /dev/ttyTHS1, which drone-router owns - so the
# writes have to go through the router at 127.0.0.1:14650, the same channel the
# VPS node talks on. That channel is exactly what the pilot told me not to
# probe while the node runs: a previous probe of mine left the node's
# rangefinder dead. So the node is stopped first. It has to be restarted
# anyway to pick up the reverted code.
#
# restore_params.py sends only PARAM_REQUEST_READ and PARAM_SET - it never
# asks the FC to change a stream rate, which is what caused that damage - and
# it reads every value back, retrying three times, because a write that is not
# read back is not a write.
#
# THE AIRCRAFT MUST BE DISARMED. ArduPilot refuses many of these while armed
# and a half-written set is worse than an unwritten one.
#
#   sudo bash /data/DRONE/fc_scripts/apply_27oct.sh
#
set -u
LOG=/data/DRONE/reports/apply_27oct_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
echo "=== log: $LOG"

echo "=== 1/4  VPS node band kar raha hoon (channel free karne ke liye)"
systemctl stop drone-vps
sleep 2

echo
echo "=== 2/4  dry run - kuch likhega nahi, sirf dikhayega"
python3 /data/DRONE/fc_scripts/restore_params.py \
    --file /data/DRONE/reports/restore_to_27oct.parm \
    --port udpout:127.0.0.1:14650
echo
read -r -p "Upar sahi lag raha hai? likhne ke liye 'haan' type karo: " ans
if [ "$ans" != "haan" ]; then
  echo "kuch nahi likha. node wapas chaalu kar raha hoon."
else
  echo
  echo "=== 3/4  likh raha hoon, har ek read-back ke saath"
  python3 /data/DRONE/fc_scripts/restore_params.py --apply \
      --file /data/DRONE/reports/restore_to_27oct.parm \
      --port udpout:127.0.0.1:14650
fi

echo
echo "=== 4/4  camera ko bootloader state me aane do, phir node chaalu"
# Never a plain restart: stop, wait for 03e7:2485, start. A restart catches
# the camera mid-teardown and startPipeline then fails with an I/O error.
for i in $(seq 1 40); do
  lsusb -d 03e7:2485 >/dev/null 2>&1 && { echo "camera bootloader state mila (${i}s)"; break; }
  sleep 1
done
systemctl start drone-vps
sleep 20
echo "drone-vps: $(systemctl is-active drone-vps)"
ls -l --time-style=+%H:%M:%S /run/drone/vps.alive 2>/dev/null
echo "abhi: $(date +%H:%M:%S)"
chmod 644 "$LOG" 2>/dev/null
echo
echo "FC reboot karna hai in sab ke lagne ke liye (EK3_*, SERIAL*, PLND_*)."
echo "Aur 27 Oct ka WP mission alag se upload karna hai - MIS_TOTAL haath se nahi likhte."
