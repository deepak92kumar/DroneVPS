#!/bin/sh
# Mount a freshly inserted removable partition at /media/<user>/<LABEL>.
#
# WHY NOT udisksctl. The first attempt had udev call udisksctl, and udisksctl
# asks polkit, and polkit says no: this Jetson is driven over VNC, whose
# systemd session has NO SEAT (loginctl shows c1 for abc with an empty seat
# column while gdm's c2 holds seat0), so udisks treats the session as
# inactive and filesystem-mount falls through to auth_admin with no desktop
# to prompt in. A .pkla granting the plugdev group did not change it either.
#
# udev's RUN+= already runs as root, so the mount needs no authorisation at
# all. That is the whole reason this script exists.
#
#   $1 = kernel device name, e.g. sda1
DEV="/dev/$1"
USER_NAME="abc"
UID_N=$(id -u "$USER_NAME" 2>/dev/null) || exit 0
GID_N=$(id -g "$USER_NAME" 2>/dev/null) || exit 0

# Already mounted somewhere? Leave it alone.
/bin/findmnt -S "$DEV" >/dev/null 2>&1 && exit 0

LABEL=$(/sbin/blkid -o value -s LABEL "$DEV" 2>/dev/null)
TYPE=$(/sbin/blkid -o value -s TYPE "$DEV" 2>/dev/null)
[ -n "$TYPE" ] || exit 0
case "$TYPE" in
    vfat|exfat|ntfs|ext2|ext3|ext4) ;;
    *) exit 0 ;;
esac
[ -n "$LABEL" ] || LABEL="$1"

DIR="/media/$USER_NAME/$LABEL"
/bin/mkdir -p "$DIR" || exit 0

case "$TYPE" in
    vfat|exfat|ntfs)
        # These have no Unix ownership of their own, so it is handed over at
        # mount time - otherwise the card lands as root and the flight logs
        # cannot be read without sudo, which is the thing being fixed.
        /bin/mount -t "$TYPE" -o "rw,uid=$UID_N,gid=$GID_N,umask=002,flush" "$DEV" "$DIR" ;;
    *)
        /bin/mount "$DEV" "$DIR" && /bin/chown "$UID_N:$GID_N" "$DIR" ;;
esac || { /bin/rmdir "$DIR" 2>/dev/null; exit 0; }
exit 0
