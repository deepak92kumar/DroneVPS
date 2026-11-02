"""Two things that let the node hang silently on 2 Nov, and must not again.

The loop wedged at t=300.65: the camera stopped delivering, LiveSource.read()
sat in a blocking q.get() with no timeout, and the process stayed at 216% CPU
with systemctl reporting "active" for two minutes. Restart= cannot see that,
because nothing crashed - the liveness file and drone_watchdog are what is
supposed to.

They did not. The watchdog logged STALLED seven times over 115 s and
restarted nothing, because armed_via_router() always returned None and None
means "treat as armed, do not restart". It returned None because on a udpout:
endpoint the router never learns a client that has not sent anything, so the
heartbeat wait always timed out. Measured on the live router: without an
announce, None in 3.00 s three times out of three; with one, DISARMED in 0.70
and 1.00 s.
"""
import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fails = []


def check(what, ok, detail=""):
    print(("  ok   " if ok else "  FAIL ") + what
          + (f"   [{detail}]" if detail else ""))
    if not ok:
        fails.append(what)


print("1) the camera wait is bounded")
src = open(os.path.join(ROOT, "vps_vio", "live_node.py"),
           encoding="utf-8").read()
check("no bare blocking q.get() on the frame queue",
      "fr = self.q.get()" not in src)
check("the wait has a deadline", "FRAME_WAIT_MAX_S" in src
      and "deadline = time.monotonic() + self.FRAME_WAIT_MAX_S" in src)
check("a timeout ENDS the run rather than hanging",
      "ending the run so systemd can reopen it" in src)
tree = ast.parse(src)
val = None
for n in ast.walk(tree):
    if (isinstance(n, ast.Assign) and len(n.targets) == 1
            and getattr(n.targets[0], "id", "") == "FRAME_WAIT_MAX_S"):
        val = n.value.value
check("the deadline is a real, short bound", val is not None and 0.5 <= val <= 5.0,
      f"FRAME_WAIT_MAX_S = {val}")

print("\n2) the watchdog can actually read the armed state")
wd = open(os.path.join(ROOT, "drone_watchdog.py"), encoding="utf-8").read()
fn = next(n for n in ast.walk(ast.parse(wd))
          if isinstance(n, ast.FunctionDef) and n.name == "armed_via_router")
body = ast.dump(fn)
check("armed_via_router announces itself before waiting",
      "heartbeat_send" in body)
first_hb = wd.index("heartbeat_send")
first_recv = wd.index("recv_match", wd.index("def armed_via_router"))
check("and it announces BEFORE the recv, not after", first_hb < first_recv,
      "otherwise the router still never learns the client")
check("None still means do-not-restart",
      "treating as ARMED and\n" in wd or "treating as ARMED" in wd)

print("\nALL PASS" if not fails else f"\n{len(fails)} FAILED: {', '.join(fails)}")
sys.exit(1 if fails else 0)
