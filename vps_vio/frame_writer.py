#!/usr/bin/env python3
"""Save camera frames from the navigating process, without ever stalling it.

Why this exists at all, and why it is not a second process: the OAK-D permits
ONE owner. live_node's LiveSource does dai.Device() and record_flight.py does
dai.Device(), so on an aircraft that has to navigate, record_flight can never
run and no images are ever recorded - which is why the offline map-matching
work the pilot asked for has had nothing to work on. The process that already
holds the camera is the one that can write frames, so the frames are written
here, inside it. The alternative - handing frames between two processes over a
socket or shared memory - adds a channel that can block the flow loop, to
solve a problem that only exists because two processes both wanted the device.

The single rule everything here follows: A SAVED FRAME MUST NEVER COST A FLOW
UPDATE. The flow loop has about 59 ms a frame at 17 Hz and spends 22 ms of it
(measured 20 Oct). JPEG encoding is 9.5 ms a frame and PNG 30 ms, so encoding
on the loop thread would eat a sixth to a half of the budget that is keeping
the aircraft's position alive. Frames are for afterwards; navigation is for
now. So:

  * offer() does no encoding, no disk work and no waiting. It hands a
    reference to a bounded queue and returns.
  * The queue is bounded and put_nowait: when the disk or the encoder falls
    behind, the frame is DROPPED and counted. Dropping is correct. Blocking
    the loop is not, which is why there is no put() with a timeout anywhere in
    this file - a timeout is just a smaller stall.
  * Encoding runs on a daemon worker thread, so it cannot hold the loop up and
    cannot keep the process alive at shutdown.

Naming ties a frame to the estimate CSV. offer() takes the row index the
caller is about to write and returns the file name it chose, so the caller can
put that name in the row (live_node's frame_file column). One row, one name,
no clock involved - this Jetson has no battery-backed RTC and its clock jumps
at boot, so an index is the only thing that cannot lie.
"""
from __future__ import annotations
import os
import queue
import threading
import time


def _default_encode(path, img, params):
    """Write one frame, and make sure it is really on the disk.

    TWO FAULTS LIVED IN THE THREE LINES THIS REPLACES, and the 20 Oct 16:38
    flight shows both. 86 of its 1151 saved frames are ZERO BYTES - not
    truncated, empty - they are the last 86, consecutive, spanning the final
    23 seconds, and the last readable frame carries the same timestamp as the
    first empty one. The writer reported every one of them as written.

    cv2.imwrite RETURNS FALSE on failure; it does not raise. So the worker's
    `except Exception` could never fire, `failed` stayed at 0, and a row's
    frame_file named a file no one could open. Any real write failure - a full
    disk, a read-only filesystem, a bad path - was invisible by construction.
    Checking the return value is the whole fix for that, and it is checked
    whether or not the other fault is what bit here.

    THE DATA WAS NEVER FLUSHED. ext4 delays allocation, so a written file's
    bytes can sit in the page cache for up to 30 s with only its name on the
    disk. The battery was pulled at the end of that flight. 23 seconds of
    unflushed frames is exactly what that leaves behind, and it is the same
    failure the estimate CSVs showed as NUL tails on the same day - those got a
    per-second fsync on 20 Oct and the frames did not.

    So the bytes are encoded in memory and written through a file this function
    owns, with fsync before it closes. The cost is one fsync per frame at 4 Hz,
    which this Jetson's NVMe does in well under the 250 ms budget; the
    alternative is losing the end of every flight that ends with the battery.
    """
    import cv2
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".jpg", img, params)
    if not ok:
        raise OSError(f"could not encode {path}")
    with open(path, "wb") as fh:
        fh.write(buf.tobytes())
        fh.flush()
        os.fsync(fh.fileno())


class FrameWriter:
    """Rate-limited, drop-on-backlog frame saver with its own worker thread.

    hz        frames a second to attempt. The flow loop runs faster than this;
              frames in between are skipped by the rate gate, which costs one
              float comparison on the loop thread.
    maxsize   queued frames allowed to be in flight. Deliberately small: a
              long queue does not rescue a disk that cannot keep up, it only
              delays the moment we notice, and every queued 1150x800 frame is
              ~0.9 MB of RAM held away from the map cache.
    encode    injectable so a test can make encoding arbitrarily slow and
              prove the loop still is not held up.
    clock     injectable for the same reason.
    """

    def __init__(self, folder: str, hz: float = 2.0, *, quality: int = 90,
                 maxsize: int = 8, ext: str = "jpg",
                 encode=_default_encode, clock=time.monotonic):
        self.folder = folder
        os.makedirs(folder, exist_ok=True)
        self.hz = float(hz)
        self.interval = 1.0 / self.hz if self.hz > 0 else 0.0
        self.ext = ext
        self.maxsize = int(maxsize)
        # JPEG, not PNG. PNG is 30 ms a frame against JPEG's 9.5 (measured
        # 20 Oct), and while that cost is on the worker thread rather than the
        # loop, it is still three times the CPU taken from a machine that is
        # also running flow and XFeat. record_flight switches to JPEG above
        # 5 Hz for the same reason.
        self.params = []
        if ext == "jpg":
            try:
                import cv2
                self.params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
            except ImportError:
                # A test with its own encode() does not need OpenCV present
                # just to construct the writer.
                pass
        self._encode = encode
        self._clock = clock
        self._q: "queue.Queue" = queue.Queue(maxsize=self.maxsize)
        self.saved = 0        # handed to the worker and written
        self.dropped = 0      # offered, wanted, and thrown away: queue full
        self.failed = 0       # reached the worker and the write did not land
        self.skipped = 0      # not wanted: the rate gate said not yet
        # Frames that were accepted - so their names ARE in the CSV - and then
        # lost because the disk was still behind when close() ran out of time.
        # Counted apart from the others because this is the one case that can
        # leave a row naming a file that does not exist, and a report that
        # hides it would send someone map-matching against a missing image.
        self.lost_at_close = 0
        self._next = None
        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._run, name="frame-writer",
                                     daemon=True)
        self._thr.start()

    # ----------------------------------------------------------------- loop #
    def due(self, now: float | None = None) -> bool:
        """Is a frame wanted at this instant? One comparison, no side effects."""
        if self.interval <= 0:
            return False
        now = self._clock() if now is None else now
        return self._next is None or now >= self._next

    def offer(self, img, index: int, now: float | None = None):
        """Offer one frame for frame index `index`. Returns the file name it
        will be saved as, or None if this frame is not being saved.

        Never blocks, never encodes, never touches the disk. The ONLY work on
        the caller's thread is a clock read, a float compare and a queue put.
        """
        now = self._clock() if now is None else now
        if not self.due(now):
            self.skipped += 1
            return None
        # The next deadline is set from NOW, not by adding to the last one. A
        # writer that fell behind must not then try to catch up by saving a
        # burst of frames - that is exactly when the loop can least afford it.
        # record_flight's save loop does the same thing for the same reason.
        self._next = now + self.interval
        name = f"f{index:06d}.{self.ext}"
        try:
            self._q.put_nowait((os.path.join(self.folder, name), img))
        except queue.Full:
            # The disk or the encoder is behind. Lose the frame, count it, and
            # return to flying. There is no version of this branch that waits.
            self.dropped += 1
            return None
        self.saved += 1
        return name

    # --------------------------------------------------------------- worker #
    def _run(self):
        while True:
            item = self._q.get()
            if item is None:
                self._q.task_done()
                return
            path, img = item
            try:
                self._encode(path, img, self.params)
            except Exception:
                # A failed write is a lost frame, never a lost aircraft: the
                # worker swallows it, counts it, and takes the next one. A
                # full disk must not end a flight's navigation.
                self.failed += 1
                # And it leaves no stub behind. A zero-byte file whose name is
                # in the CSV sends an offline run map-matching against nothing
                # and reports it as an unreadable image rather than a missing
                # one - which is how 86 lost frames read as a matcher problem.
                try:
                    if os.path.exists(path) and os.path.getsize(path) == 0:
                        os.unlink(path)
                except OSError:
                    pass
            finally:
                self._q.task_done()

    def close(self, timeout: float = 5.0):
        """Stop the worker, giving queued frames a bounded chance to land.

        Bounded, not instant, and not unlimited. This runs in live_node's
        finally block, where nothing is flying any more, so WAITING HERE IS
        ALLOWED - it is the one place in this file that may. A shutdown that
        hangs on a slow disk is a node that does not come back, so the wait
        has a deadline; but a shutdown that throws away frames already encoded
        and queued loses flight data for no reason, which is what the first
        version of this did: with the queue full it dropped every waiting
        frame on the floor and the CSV was left naming files that were never
        written. The real clock is used, never the injectable one, because a
        test's frozen clock must not be able to turn this into a hang.
        """
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            try:
                self._q.put_nowait(None)
                break
            except queue.Full:
                if time.monotonic() >= deadline:
                    # Out of time: the disk is slower than the deadline. Drop
                    # what is still waiting so the sentinel fits and the
                    # thread can end. Counted, like every other lost frame.
                    try:
                        while True:
                            self._q.get_nowait()
                            self.dropped += 1
                            self.lost_at_close += 1
                            self._q.task_done()
                    except queue.Empty:
                        pass
                else:
                    time.sleep(0.005)
        self._thr.join(max(0.0, deadline - time.monotonic()))
        return self.summary()

    def summary(self) -> str:
        on_disk = self.saved - self.lost_at_close - self.failed
        s = (f"frames: {on_disk} written, {self.dropped} dropped (writer "
             f"behind), {self.failed} failed, {self.skipped} skipped by the "
             f"{self.hz:g} Hz gate")
        # The two cases where a row names a file that is not there. Both are
        # said out loud: a silently missing image is worse than a missing
        # image, because offline matching would blame the match.
        if self.lost_at_close:
            s += (f"\n  *** {self.lost_at_close} accepted frames were still "
                  f"queued when the writer stopped and are NOT on disk - "
                  f"their rows name files that were never written ***")
        if self.failed:
            s += (f"\n  *** {self.failed} writes FAILED (disk full?) - those "
                  f"rows name files that are not there ***")
        return s


def next_run_number(out_dir: str) -> int:
    """The next run index in an estimate-log directory.

    Numbered, not timestamped, and the reason is the Jetson: it has no
    battery-backed RTC, so early boot runs on a stale clock that jumps forward
    when NTP lands. A name built from that clock can collide with an older run
    or sort before it. ArduPilot numbers its own logs for the same reason.

    Both shapes count, because a run writes one or the other: vps_NNNNN.csv
    for a run with no frames, and a vps_NNNNN/ FOLDER (holding that same CSV
    plus the images) for a run with frames. Counting only one of them would
    hand the same number to two runs and the second would overwrite the first.
    """
    n = 0
    if not os.path.isdir(out_dir):
        return 1
    for name in os.listdir(out_dir):
        stem = name[:-4] if name.endswith(".csv") else name
        if not stem.startswith("vps_"):
            continue
        try:
            n = max(n, int(stem[4:]))
        except ValueError:
            pass
    return n + 1
