#!/usr/bin/env python3
"""Which satellite sheet the node flies with, and one place that decides it.

Until now the answer was a constant in live_node.py, and cutting a sheet for a
new site meant editing the flying node by hand on the day of the flight. The
studio cuts sheets now, so the choice needs somewhere to live that is not
source code:

    /data/DRONE/maps/ACTIVE_MAP     one line, the absolute path of the sheet
    /data/DRONE/maps/active.tif     a symlink to the same file, for humans

Both are written together; the text file is what is read. If it is missing,
empty, or names a file that is not there, the answer falls back to the sheet
that has always flown and SAYS it fell back - a silent fallback to the wrong
site is exactly the failure this is meant to prevent, because a take-off point
off the sheet means no map fixes for the whole flight.

Nothing here rejects, clamps or vetoes anything: it reports a path and the
reason for it. `--map` on the command line still overrides everything.
"""
from __future__ import annotations

import os

MAPS_DIR = "/data/DRONE/maps"
POINTER = os.path.join(MAPS_DIR, "ACTIVE_MAP")
LINK = os.path.join(MAPS_DIR, "active.tif")
# The sheet that flew every flight to date. It does NOT cover every site - see
# the note at live_node.DEFAULT_MAP - which is the whole reason the pointer
# exists.
FALLBACK = "/data/DRONE/bhh.tif"


def resolve():
    """(path, source, note) - the sheet to fly and why it is that one."""
    try:
        with open(POINTER) as f:
            p = f.read().strip()
    except OSError:
        return FALLBACK, "fallback", f"no {POINTER}, so the sheet that has always flown"
    if not p:
        return FALLBACK, "fallback", f"{POINTER} is empty"
    if not os.path.exists(p):
        return (FALLBACK, "fallback",
                f"{POINTER} names {p}, which is not on this machine")
    return p, "pointer", f"set in {POINTER}"


def active_map() -> str:
    return resolve()[0]


def set_active(path: str) -> str:
    """Point the node at `path`. Returns the absolute path written."""
    p = os.path.abspath(path)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    os.makedirs(MAPS_DIR, exist_ok=True)
    tmp = POINTER + ".tmp"
    with open(tmp, "w") as f:
        f.write(p + "\n")
    os.replace(tmp, POINTER)            # so a half-written pointer never exists
    try:
        if os.path.islink(LINK) or os.path.exists(LINK):
            os.unlink(LINK)
        os.symlink(p, LINK)
    except OSError:
        pass                            # the symlink is a convenience, not the record
    return p


def clear():
    """Back to the fallback sheet."""
    for f in (POINTER, LINK):
        try:
            os.unlink(f)
        except OSError:
            pass


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(set_active(sys.argv[1]))
    else:
        p, src, note = resolve()
        print(f"{p}\n  ({src}: {note})")
