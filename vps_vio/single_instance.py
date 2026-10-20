#!/usr/bin/env python3
"""One copy at a time, whoever started it.

The check this replaces asked systemctl whether "drone-vio" was active. There
is no unit by that name - the units are drone-router, drone-vps, drone-rc-switch
and drone-watchdog - so it was always inactive and the guard never once fired.
It read like a safety check and was dead code.

That matters because of what a second copy actually does. The OAK-D opens
exclusively, so the second one cannot have the camera; with Restart=always in
front of it, it fails, restarts five seconds later, fails again, and sits there
doing that for the whole flight. Worse is the variant where the service is the
one that loses: the pilot's own copy holds the camera, the service crash-loops,
and the GPS_INPUT stream has a hole in it every time - which is the gap that
takes the EKF's position away and drops the aircraft into ALT_HOLD. Log 214 has
exactly that at 8.6 s.

A file lock, not a PID file and not a process scan:

  * the kernel drops it when the holder dies, however it dies, so a crash
    never leaves a stale lock that keeps the next start out;
  * it does not care whether the holder was started by systemd, by hand over
    ssh, or by the watchdog;
  * and it is held by an OPEN FILE DESCRIPTOR for the life of the process, so
    it cannot be released early by accident. The caller keeps the fd.

EXIT_BUSY is its own exit code so a unit can carry RestartPreventExitStatus and
step aside instead of looping.
"""
from __future__ import annotations

import fcntl
import os

EXIT_BUSY = 3
DEFAULT_DIR = "/run/lock" if os.path.isdir("/run/lock") else "/tmp"


def lock_path(name: str) -> str:
    return os.path.join(DEFAULT_DIR, f"{name}.lock")


def acquire(name: str):
    """Returns an open fd holding the lock, or None if someone else has it."""
    path = lock_path(name)
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
    except OSError:
        return None
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    os.ftruncate(fd, 0)
    os.write(fd, f"{os.getpid()}\n".encode())
    try:
        os.fsync(fd)
    except OSError:
        pass
    return fd


def holder(name: str) -> str:
    """The PID written by whoever holds it, for the message the loser prints."""
    try:
        with open(lock_path(name)) as fh:
            return fh.read().strip() or "?"
    except OSError:
        return "?"


def holder_cmdline(pid: str) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            return fh.read().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        return ""


def require(name: str):
    """Take the lock or leave. Returns the fd; the caller must keep it alive."""
    import sys
    fd = acquire(name)
    if fd is not None:
        return fd
    pid = holder(name)
    cmd = holder_cmdline(pid)
    sys.stderr.write(
        f"*** {name} is already running (PID {pid}). Two copies cannot run - "
        f"the camera opens exclusively and both would inject GPS_INPUT. ***\n")
    if cmd:
        sys.stderr.write(f"    running: {cmd}\n")
    sys.stderr.write(f"    stop it with:  sudo systemctl stop {name}   "
                     f"(or  kill {pid})\n")
    raise SystemExit(EXIT_BUSY)
