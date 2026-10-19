"""A listener that stops talking stops hearing, and nothing says so.

20 Oct, on the aircraft: the RC switch service started, announced itself,
switched once at 02:18:08 - and then logged nothing at all while the pilot
flipped the transmitter three to five times. "kis par hai pata hi nahi chal
raha". The switch was not slow and it had not crashed. It was deaf.

The router drops a UDP client it has not heard from in client_timeout_s. After
its opening handshake gps_rc_switch is a pure listener: it reads RC_CHANNELS
and writes a parameter, and writing a parameter only happens when the switch
moves - which it cannot see, because it has been dropped. Thirty seconds after
starting it fell off the client list and recv_match returned None for ever.

From the client's own side this is invisible: a dropped client and a dead link
look identical, which is why it has to be printed by the router and prevented
by the client.
"""
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "vps_vio"))
from mavlink_router import Client, Router  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        fails.append(name)


class FakeRouter:
    """Just the expiry behaviour, with no serial port to open."""

    def __init__(self, timeout_s):
        self.client_timeout_s = timeout_s
        self.clients = {}
        self.to_fc = {}

    _expire = Router._expire


print("a client that keeps talking stays")
r = FakeRouter(30.0)
a = ("127.0.0.1", 5001)
r.clients[a] = Client(a)
for _ in range(5):
    r.clients[a].last_seen = time.monotonic()
    r._expire()
check("still registered", a in r.clients)

print("\na client that goes quiet is dropped - this is the bug")
r = FakeRouter(0.05)
b = ("127.0.0.1", 5002)
r.clients[b] = Client(b)
time.sleep(0.12)
r._expire()
check("silence costs it its registration", b not in r.clients,
      "this is what happened to gps_rc_switch")

print("\nand the router says so, instead of dropping it in silence")
import io  # noqa: E402
from contextlib import redirect_stdout  # noqa: E402
r = FakeRouter(0.05)
c = ("127.0.0.1", 5003)
r.clients[c] = Client(c)
time.sleep(0.12)
buf = io.StringIO()
with redirect_stdout(buf):
    r._expire()
out = buf.getvalue()
check("the drop is printed", "client gone" in out, out.strip()[:70])
check("with the address", "5003" in out, out.strip()[:70])

print("\nthe receive-only clients send a heartbeat to stay alive")
rc = open(os.path.join(HERE, "vps_vio", "gps_rc_switch.py")).read()
loop = rc[rc.index("def run("):]
check("gps_rc_switch heartbeats inside its loop",
      "heartbeat_send" in loop, "only the opening handshake would not be enough")
m = re.search(r"next_hb = now \+ ([\d.]+)", loop)
check("and does it well inside the router's timeout",
      m is not None and float(m.group(1)) < 30.0,
      f"every {m.group(1)}s vs a 30s timeout" if m else "no interval found")

print("\nthe source watcher polls fast enough for a pilot to feel it")
from live_node import SourceWatcher  # noqa: E402
import inspect  # noqa: E402
d = inspect.signature(SourceWatcher.__init__).parameters["period_s"].default
check("source is re-read at least twice a second", d <= 0.5, f"{d}s")
# The whole chain: four stable RC samples, the parameter set and its read-back,
# then this poll. Each one adds, so none of them may be lazy.
check("and that is a fraction of the chain, not the whole of it", d < 1.0, f"{d}s")

print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
