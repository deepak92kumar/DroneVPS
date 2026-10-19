"""End-to-end test of mavlink_router against a simulated FC on a pty.

A pseudo-terminal behaves like a serial port, so the router can be run for real
- its own serial reads, its own UDP socket, its own threads - with this script
playing the flight controller on the other end of the line. No hardware.

Checks, in order:
  1. framing: fragmented v1 and v2 frames are reassembled whole
  2. resync: garbage before a frame is discarded, the frame still arrives
  3. fan-out: every client gets what the FC says
  4. no self-echo: a client does NOT get its own frame back (the feedback loop
     that made the VPS read its own GPS_INPUT as the aircraft)
  5. peer delivery: the OTHER client does get it
  6. fan-in: it reaches the FC
  7. heartbeat filter: sysid 1/compid 1 passes, anything else is dropped
  8. read-only mode: nothing reaches the FC
"""
import os
import socket
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mavlink_router import FrameSplitter, Router  # noqa: E402

HEARTBEAT = 0


def v2(sysid, compid, msgid, payload=b"\x00" * 9):
    """A MAVLink v2 frame. CRC bytes are filler - the splitter frames by
    length and leaves validation to the clients, which all use pymavlink."""
    return (bytes([0xFD, len(payload), 0, 0, 0, sysid, compid,
                   msgid & 0xFF, (msgid >> 8) & 0xFF, (msgid >> 16) & 0xFF])
            + payload + b"\x00\x00")


def v1(sysid, compid, msgid, payload=b"\x00" * 9):
    return (bytes([0xFE, len(payload), 0, sysid, compid, msgid])
            + payload + b"\x00\x00")


fails = []


def check(name, got, want):
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: got {got!r} want {want!r}")
    if not ok:
        fails.append(name)


# ---------------------------------------------------------------- framing
print("framing and resync")
s = FrameSplitter()
whole = v2(1, 1, 33) + v1(1, 1, 30)
got = []
for i in range(0, len(whole), 3):            # 3 bytes at a time
    got += s.feed(whole[i:i + 3])
check("two frames from 3-byte fragments", [(g[0], g[1], g[2]) for g in got],
      [(1, 1, 33), (1, 1, 30)])

s = FrameSplitter()
got = s.feed(b"\x11\x22\x33" + v2(1, 1, 42))
check("frame after garbage", [(g[0], g[1], g[2]) for g in got], [(1, 1, 42)])
check("resyncs counted", s.resyncs > 0, True)


# ------------------------------------------------------------- the router
def start_router(read_only):
    master, slave = os.openpty()
    # Non-blocking: this script reads the master to see what reached the "FC",
    # and when nothing did, a blocking os.read would hang the test forever
    # instead of reporting an empty read.
    os.set_blocking(master, False)
    r = Router(os.ttyname(slave), 115200, ("127.0.0.1", 0), read_only=read_only)
    t = threading.Thread(target=r.run, kwargs=dict(report_every_s=1e6), daemon=True)
    t.start()
    time.sleep(0.4)
    return r, master, r.sock.getsockname()


def client(addr):
    c = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    c.settimeout(1.5)
    c.sendto(v2(200, 200, HEARTBEAT), addr)      # announce; a non-FC heartbeat
    return c


def drain(c, budget=0.6):
    out, t0 = [], time.monotonic()
    while time.monotonic() - t0 < budget:
        try:
            d = c.recv(4096)
        except socket.timeout:
            break
        s2 = FrameSplitter()
        out += [(f[0], f[1], f[2]) for f in s2.feed(d)]
    return out


print("\nrouter, read/write")
r, master, addr = start_router(False)
a, b = client(addr), client(addr)
time.sleep(0.3)
drain(a, 0.2), drain(b, 0.2)                  # clear the announce cross-talk

os.write(master, v2(1, 1, 33))                # FC says GLOBAL_POSITION_INT
time.sleep(0.3)
check("client A gets FC frame", drain(a), [(1, 1, 33)])
check("client B gets FC frame", drain(b), [(1, 1, 33)])

# Drain the FC side first: both clients' announce heartbeats were correctly
# forwarded to the FC and are still sitting in the pty, so without this the
# next check sees them too.
try:
    os.read(master, 65536)
except BlockingIOError:
    pass

a.sendto(v2(255, 190, 232), addr)             # a client sends GPS_INPUT
time.sleep(0.4)
check("sender does NOT get its own frame", drain(a), [])
check("peer client does get it", drain(b), [(255, 190, 232)])
try:
    fc_raw = os.read(master, 4096)
except BlockingIOError:
    fc_raw = b""
fc_got = FrameSplitter().feed(fc_raw)
check("frame reached the FC", [(f[0], f[1], f[2]) for f in fc_got],
      [(255, 190, 232)])

os.write(master, v2(1, 1, HEARTBEAT))
time.sleep(0.3)
check("FC heartbeat passes", drain(a), [(1, 1, HEARTBEAT)])
os.write(master, v2(42, 7, HEARTBEAT))
time.sleep(0.3)
check("foreign heartbeat dropped", drain(a), [])

print("\nrouter, read-only")
r2, master2, addr2 = start_router(True)
c = client(addr2)
time.sleep(0.3)
os.write(master2, v2(1, 1, 33))
time.sleep(0.3)
check("FC still fans out", drain(c), [(1, 1, 33)])
c.sendto(v2(255, 190, 232), addr2)
time.sleep(0.4)
try:
    leaked = os.read(master2, 4096)
except BlockingIOError:
    leaked = b""
check("nothing reached the FC", len(leaked), 0)
check("blocked counter moved", r2.n_blocked > 0, True)

print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
