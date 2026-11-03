#!/bin/bash
# Install the four units (plus the watchdog's oneshot) into systemd.
#
# Replaces install_autostart.sh, which installed one unit back when one process
# owned the serial port. The order matters now: the router owns /dev/ttyTHS1
# and the others are its clients.
#
#   ./install.sh install   copy the units and reload systemd. Nothing starts.
#   ./install.sh enable    enable on boot (router, vps, rc-switch, watchdog timer)
#   ./install.sh start     start now
#   ./install.sh status    what is installed, enabled and running
#   ./install.sh disable   disable on boot and stop
#
# install and enable are separate on purpose. These start on battery power with
# no one at a keyboard, so turning that on should be its own decision.
set -uo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)"
DEST=/etc/systemd/system
UNITS=(drone-router.service drone-vps.service drone-rc-switch.service
        drone-watchdog.service drone-watchdog.timer)
# The oneshot is pulled by the timer, so it is not enabled on its own.
ENABLE=(drone-router.service drone-vps.service drone-rc-switch.service
        drone-watchdog.timer)

die(){ echo "ERROR: $*" >&2; exit 1; }

case "${1:-}" in
install)
    for u in "${UNITS[@]}"; do
        [ -f "$SRC/$u" ] || die "$SRC/$u missing"
    done
    for u in "${UNITS[@]}"; do
        sudo install -m 0644 "$SRC/$u" "$DEST/$u" || die "copy $u failed"
        echo "  installed $u"
    done
    sudo systemctl daemon-reload
    echo
    echo "Installed, nothing started. './install.sh enable' to run on boot."
    echo
    echo "Before enabling, check the map: live_node.py needs a GeoTIFF that"
    echo "  COVERS THE TAKE-OFF POINT. /data/DRONE/bhh.tif spans"
    echo "  30.88411-30.90332 N, 74.59762-74.63224 E. SCR_USER1/2 on the FC"
    echo "  must be inside that, or there are no map fixes for the whole flight."
    ;;
enable)
    sudo systemctl enable "${ENABLE[@]}" || die "enable failed"
    echo "enabled on boot: ${ENABLE[*]}"
    ;;
start)
    # Router first and on its own: if it cannot open the port, starting the
    # clients only produces three confusing failures instead of one clear one.
    sudo systemctl start drone-router.service || die "router did not start"
    sleep 2
    systemctl is-active --quiet drone-router.service \
        || die "router started then exited - check: journalctl -u drone-router -n 30"
    sudo systemctl start drone-vps.service drone-rc-switch.service drone-watchdog.timer
    echo "started. './install.sh status' to see what took."
    ;;
status)
    for u in "${UNITS[@]}"; do
        # is-enabled and is-active both PRINT a word and return non-zero for
        # a unit that is off, so "|| echo" would append a second line. Take
        # the output and only substitute when it is empty.
        en=$(systemctl is-enabled "$u" 2>/dev/null); en=${en:--}
        ac=$(systemctl is-active "$u" 2>/dev/null); ac=${ac:--}
        printf '%-28s installed:%-4s enabled:%-9s active:%s\n' "$u" \
            "$([ -f "$DEST/$u" ] && echo yes || echo no)" "$en" "$ac"
    done
    echo
    if [ -f /run/drone/vps.alive ]; then
        echo "liveness: $(( $(date +%s) - $(stat -c %Y /run/drone/vps.alive) ))s old"
        echo "          $(cat /run/drone/vps.alive)"
    else
        echo "liveness: /run/drone/vps.alive not present"
    fi
    ;;
disable)
    sudo systemctl disable --now "${ENABLE[@]}" 2>/dev/null
    echo "disabled and stopped"
    ;;
*)
    sed -n '2,20p' "$0" | sed 's/^# \?//'
    exit 1
    ;;
esac
