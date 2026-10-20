#!/bin/bash
# What to run after a reboot to know, in one screen, whether the aircraft is
# ready - without a keyboard, a password, or reading a journal.
#
# This exists because "systemctl is-enabled" said enabled on 20 Oct while the
# services were never started at all: an ordering cycle had systemd delete
# their start jobs before it tried. Enabled is not running, and running is not
# working.
echo "uptime:  $(uptime -p)   (booted $(uptime -s))"
echo
printf '%-24s %-10s %-10s %s\n' UNIT ENABLED ACTIVE RESTARTS
for u in drone-router drone-vps drone-rc-switch; do
    printf '%-24s %-10s %-10s %s\n' "$u" \
        "$(systemctl is-enabled $u 2>&1)" "$(systemctl is-active $u 2>&1)" \
        "$(systemctl show $u -p NRestarts --value 2>/dev/null)"
done
echo
echo "ordering cycles this boot (must be empty):"
CYC=$(journalctl -b --no-pager 2>/dev/null | grep -i "ordering cycle")
if [ -n "$CYC" ]; then
    echo "$CYC" | sed 's/^/   /'
    echo "   ^ systemd DELETED start jobs. Services will not come up."
else
    echo "   none"
fi
echo
echo "VPS liveness:"
if [ -f /run/drone/vps.alive ]; then
    echo "   $(cat /run/drone/vps.alive)"
    echo "   age: $(( $(date +%s) - $(stat -c %Y /run/drone/vps.alive) ))s  (under 5s = alive)"
else
    echo "   /run/drone/vps.alive MISSING - the node is not running"
fi
echo
echo "last lines from the VPS:"
tail -4 /data/DRONE/flight_logs/vps_stdout.log 2>/dev/null | sed 's/^/   /'
echo
echo "journal is persistent: $([ -d /var/log/journal ] && echo yes || echo 'NO - logs die at reboot')"
# Whether sudo ASKED, not whether the command succeeded: is-active exits 3 for
# a stopped unit, which an exit code alone reads as a refusal.
if sudo -n systemctl is-active drone-vps 2>&1 | grep -q "password is required"; then
    echo "passwordless control:  NO - run systemd/field_setup.sh"
else
    echo "passwordless control:  yes"
fi
