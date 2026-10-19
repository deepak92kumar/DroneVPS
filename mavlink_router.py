"""One serial line to the flight controller, shared over UDP by several processes.

The problem this replaces: /dev/ttyTHS1 is a single port and several things want
it - the recorder, the VPS node, the RC switch watcher, a ground console. Two
readers on one stream each get half of every frame, which surfaces as
"device reports readiness to read but returned no data (device disconnected or
multiple access on port?)". vps_vio/fc_port.py worked around that by STOPPING
the recorder for the duration of any ground tool and starting it again
afterwards. That is safe only because those tools run on the ground; it is not
a fix, and fc_port.py's own docstring says so.

This is the fix: the router alone owns the port, and every client speaks UDP.

Two behaviours here are not obvious and both come from flights:

  * A client never receives its own bytes back. Without this the VPS read its
    own GPS_INPUT out of the stream and treated it as the aircraft's position -
    a feedback loop where the estimate confirmed itself.

  * Heartbeats are forwarded only from system 1 / component 1. Companion
    processes emit heartbeats too, and a client that accepts any heartbeat
    decides the aircraft is whatever spoke last. Only the FC is the aircraft.

Framing is parsed, not guessed: the router has to know where each frame ends
and who sent it, so it carries a MAVLink v1/v2 splitter rather than relaying
arbitrary chunks. A chunk-relay would split frames across UDP datagrams and
every client would have to resynchronise.

SAFETY. Client -> FC traffic is what writes to the aircraft. --read-only drops
it entirely and still fans out everything the FC says, which is the state to
bench in before anything is allowed to write.
"""
from __future__ import annotations

import argparse
import selectors
import socket
import sys
import time

MAVLINK1 = 0xFE
MAVLINK2 = 0xFD
HEARTBEAT = 0
FC_SYSID = 1
FC_COMPID = 1


class FrameSplitter:
    """Byte stream in, whole MAVLink frames out.

    Resynchronises by discarding bytes until a start marker, which is what
    happens after a partial frame at startup or a dropped byte on the wire.
    Length is taken from the header, so a frame is emitted only once all of it
    has arrived - never a prefix.
    """

    def __init__(self, cap: int = 8192):
        self.buf = bytearray()
        self.cap = cap
        self.resyncs = 0

    def feed(self, data: bytes):
        self.buf.extend(data)
        # A buffer that grows past the cap means the stream is not MAVLink at
        # all (wrong baud is the usual cause). Drop the oldest half rather than
        # growing without bound.
        if len(self.buf) > self.cap:
            del self.buf[: len(self.buf) // 2]
            self.resyncs += 1
        out = []
        while True:
            frame = self._take()
            if frame is None:
                return out
            out.append(frame)

    def _take(self):
        b = self.buf
        while b and b[0] not in (MAVLINK1, MAVLINK2):
            del b[0]
            self.resyncs += 1
        if not b:
            return None
        if b[0] == MAVLINK2:
            if len(b) < 10:
                return None
            payload = b[1]
            signed = bool(b[2] & 0x01)
            total = 12 + payload + (13 if signed else 0)
            sysid, compid = b[5], b[6]
            msgid = b[7] | (b[8] << 8) | (b[9] << 16)
        else:
            if len(b) < 6:
                return None
            payload = b[1]
            total = 8 + payload
            sysid, compid = b[3], b[4]
            msgid = b[5]
        if len(b) < total:
            return None
        raw = bytes(b[:total])
        del b[:total]
        return sysid, compid, msgid, raw


class Client:
    __slots__ = ("addr", "last_seen", "rx", "tx")

    def __init__(self, addr):
        self.addr = addr
        self.last_seen = time.monotonic()
        self.rx = 0          # frames this client sent us
        self.tx = 0          # frames we sent this client


class Router:
    def __init__(self, port: str, baud: int, listen: tuple[str, int], *,
                 read_only: bool = False, client_timeout_s: float = 30.0):
        import serial
        self.ser = serial.Serial(port, baud, timeout=0)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.sock.bind(listen)
        self.read_only = read_only
        self.client_timeout_s = client_timeout_s
        self.clients: dict[tuple, Client] = {}
        self.from_fc = FrameSplitter()
        self.to_fc: dict[tuple, FrameSplitter] = {}
        self.n_fc_frames = 0
        self.n_to_fc = 0
        self.n_dropped_hb = 0
        self.n_blocked = 0
        self.sel = selectors.DefaultSelector()
        self.sel.register(self.ser, selectors.EVENT_READ, "serial")
        self.sel.register(self.sock, selectors.EVENT_READ, "udp")

    # -- serial -> clients ------------------------------------------------
    def _pump_serial(self):
        data = self.ser.read(4096)
        if not data:
            return
        for sysid, compid, msgid, raw in self.from_fc.feed(data):
            self.n_fc_frames += 1
            # Only the aircraft's own heartbeat leaves the port. Anything else
            # claiming to be a heartbeat is another companion process whose
            # frame found its way onto the line.
            if msgid == HEARTBEAT and (sysid, compid) != (FC_SYSID, FC_COMPID):
                self.n_dropped_hb += 1
                continue
            for c in list(self.clients.values()):
                try:
                    self.sock.sendto(raw, c.addr)
                    c.tx += 1
                except OSError:
                    pass

    # -- clients -> serial, and clients -> other clients -------------------
    def _pump_udp(self):
        while True:
            try:
                data, addr = self.sock.recvfrom(4096)
            except BlockingIOError:
                return
            except OSError:
                return
            c = self.clients.get(addr)
            if c is None:
                c = self.clients[addr] = Client(addr)
                self.to_fc[addr] = FrameSplitter()
                print(f"client up: {addr[0]}:{addr[1]}  ({len(self.clients)} total)",
                      flush=True)
            c.last_seen = time.monotonic()

            for sysid, compid, msgid, raw in self.to_fc[addr].feed(data):
                c.rx += 1
                if self.read_only:
                    self.n_blocked += 1
                    continue
                try:
                    self.ser.write(raw)
                    self.n_to_fc += 1
                except Exception:
                    pass
                # Other clients hear it too, so a console can watch what the
                # VPS sends. The sender is skipped: see the module docstring.
                for other in list(self.clients.values()):
                    if other.addr == addr:
                        continue
                    try:
                        self.sock.sendto(raw, other.addr)
                        other.tx += 1
                    except OSError:
                        pass

    def _expire(self):
        now = time.monotonic()
        for addr, c in list(self.clients.items()):
            if now - c.last_seen > self.client_timeout_s:
                del self.clients[addr]
                self.to_fc.pop(addr, None)
                print(f"client gone: {addr[0]}:{addr[1]} "
                      f"(rx {c.rx}, tx {c.tx})", flush=True)

    def stats(self) -> str:
        return (f"fc_frames {self.n_fc_frames}  to_fc {self.n_to_fc}  "
                f"clients {len(self.clients)}  "
                f"dropped_hb {self.n_dropped_hb}  resyncs {self.from_fc.resyncs}"
                + (f"  BLOCKED(read-only) {self.n_blocked}" if self.read_only else ""))

    def run(self, report_every_s: float = 10.0):
        mode = "READ-ONLY (nothing reaches the aircraft)" if self.read_only else "read/write"
        print(f"router up on {self.ser.port} @ {self.ser.baudrate}, "
              f"udp {self.sock.getsockname()[0]}:{self.sock.getsockname()[1]}, {mode}",
              flush=True)
        last = time.monotonic()
        while True:
            for key, _ in self.sel.select(timeout=0.2):
                if key.data == "serial":
                    self._pump_serial()
                else:
                    self._pump_udp()
            now = time.monotonic()
            if now - last >= report_every_s:
                self._expire()
                print(self.stats(), flush=True)
                last = now


def _take_lock():
    """One router. Two would both own the serial port and split every stream."""
    import sys as _s, os as _o
    _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "vps_vio"))
    import single_instance
    return single_instance.require("drone-router")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", default="/dev/ttyTHS1",
                    help="FC serial port (TELEM2 is /dev/ttyTHS1; USB is usually /dev/ttyACM0)")
    ap.add_argument("--baud", type=int, default=921600,
                    help="TELEM2 flies at 921600 (SERIAL2_BAUD 921, read from "
                         "log 00000195.BIN); ignored on a USB CDC port")
    ap.add_argument("--listen", default="127.0.0.1:14650",
                    help="udp host:port clients talk to")
    ap.add_argument("--read-only", action="store_true",
                    help="drop everything clients send; bench in this mode first")
    a = ap.parse_args(argv)
    # One router. Two would both open the serial port, and every frame would
    # land in whichever read happened to get it - each copy seeing half a
    # stream and resyncing on the other half.
    _lock_fd = _take_lock()   # noqa: F841 - held for the process's life
    host, _, port = a.listen.rpartition(":")
    try:
        r = Router(a.port, a.baud, (host, int(port)), read_only=a.read_only)
    except Exception as e:
        print(f"cannot start: {e}", file=sys.stderr)
        return 2
    try:
        r.run()
    except KeyboardInterrupt:
        print("\n" + r.stats(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
