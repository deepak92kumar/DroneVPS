#!/usr/bin/env bash
# Put the DroneVPS Studio icon on the desktop and in the applications menu.
#
#     bash /data/DRONE/studio/install-icon.sh
#
# GNOME refuses to run a .desktop file on the desktop until it is marked
# trusted, which is why the gio line is here - without it the icon appears
# with a "Untrusted application launcher" warning and does nothing.
set -eu
SRC="/data/DRONE/studio/DroneVPS-Studio.desktop"
APPS="$HOME/.local/share/applications"
DESKTOP="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"

chmod +x /data/DRONE/studio/launch.sh
mkdir -p "$APPS" "$DESKTOP"

install -m 755 "$SRC" "$APPS/DroneVPS-Studio.desktop"
install -m 755 "$SRC" "$DESKTOP/DroneVPS-Studio.desktop"
gio set "$DESKTOP/DroneVPS-Studio.desktop" metadata::trusted true 2>/dev/null || true
update-desktop-database "$APPS" 2>/dev/null || true

echo "Installed:"
echo "  $DESKTOP/DroneVPS-Studio.desktop   (double-click this)"
echo "  $APPS/DroneVPS-Studio.desktop      (Activities -> DroneVPS Studio)"
echo
echo "From another machine on the same network, no icon is needed:"
ip=$(hostname -I 2>/dev/null | awk '{print $1}')
echo "  http://${ip:-<jetson-ip>}:8099"
