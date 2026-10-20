#!/bin/bash
# One-time root setup so the aircraft needs no keyboard in the field.
#
# Three things, all of which bit us on 20 Oct:
#
#   1. A sudoers rule so the drone services can be started, stopped and
#      restarted without typing a password. Scoped to those units and those
#      verbs - NOT a blanket NOPASSWD. Validated with visudo before it is
#      installed, because a broken sudoers file locks sudo out entirely.
#
#   2. A persistent journal. /var/log/journal did not exist, so the journal
#      lived in RAM and every reboot threw away the evidence. After the field
#      reboot there was no record of why anything did or did not start, and
#      "journalctl --list-boots" showed one boot stretching back eight days.
#
#   3. Reinstall the units and reload, because the ones in /etc are the old
#      ones with the ordering cycle that stopped drone-vps and drone-rc-switch
#      from EVER starting at boot.
set -uo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"
USER_NAME="${SUDO_USER:-abc}"
SYSTEMCTL=/usr/bin/systemctl
UNITS=(drone-router.service drone-vps.service drone-rc-switch.service
       drone-watchdog.service drone-watchdog.timer)

[ "$(id -u)" -eq 0 ] || { echo "run with sudo: sudo $0" >&2; exit 1; }

echo "== 1. passwordless control of the drone services for '$USER_NAME'"
TMP=$(mktemp)
{
  echo "# DroneVPS: let $USER_NAME control the flight services without a"
  echo "# password, so a reboot in the field needs no keyboard. Scoped to"
  echo "# these units and these verbs only."
  for u in "${UNITS[@]}"; do
    for verb in start stop restart status enable disable is-active is-enabled; do
      echo "$USER_NAME ALL=(root) NOPASSWD: $SYSTEMCTL $verb $u"
    done
  done
  echo "$USER_NAME ALL=(root) NOPASSWD: $SYSTEMCTL daemon-reload"
  echo "$USER_NAME ALL=(root) NOPASSWD: /usr/bin/journalctl -u drone-*"
} > "$TMP"
if visudo -c -q -f "$TMP"; then
    install -m 0440 -o root -g root "$TMP" /etc/sudoers.d/dronevps
    echo "   installed /etc/sudoers.d/dronevps ($(wc -l < "$TMP") rules)"
else
    echo "   REFUSED: the generated sudoers file did not validate" >&2
    rm -f "$TMP"; exit 1
fi
rm -f "$TMP"

echo "== 2. persistent journal"
if [ -d /var/log/journal ]; then
    echo "   already persistent"
else
    mkdir -p /var/log/journal
    systemd-tmpfiles --create --prefix /var/log/journal 2>/dev/null
    systemctl restart systemd-journald
    echo "   /var/log/journal created - logs now survive a reboot"
fi

echo "== 3. reinstall the units (the installed ones have the ordering cycle)"
for u in "${UNITS[@]}"; do
    install -m 0644 "$SRC/$u" "/etc/systemd/system/$u" || exit 1
done
systemctl daemon-reload
systemctl enable drone-router.service drone-vps.service \
                 drone-rc-switch.service drone-watchdog.timer >/dev/null 2>&1
echo "   reinstalled and enabled"

echo
echo "== check: nothing may be ordered after a target that wants it"
for u in drone-router drone-vps drone-rc-switch; do
    printf '   %-22s ' "$u"
    systemctl show "$u" -p After --value | tr ' ' '\n' | grep -qx multi-user.target \
        && echo "STILL IN A CYCLE" || echo "ok"
done
echo
echo "Now reboot to prove it: sudo reboot"
echo "After it comes back:    systemctl is-active drone-router drone-vps drone-rc-switch"
