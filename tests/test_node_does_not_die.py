"""The node must not die from one frame.

Two deaths in flight, both the same bug: `_use_pp` (76 crashes) and `_fe`
(4 crashes, 2 Nov, the aircraft in ALT_HOLD at 30 m). A variable assigned in
one branch of an if/else, read by code below the branch that runs on every
frame. On the ground the other branch never runs, so nothing caught either.

These tests are the two things that stop a third one:

  1. bench/unbound_locals.py over the whole package, so the SHAPE cannot be
     committed again. Verified against both real pre-fix files - it reports
     `_fe` and `_use_pp` at the right lines - because a gate that does not
     catch the bug it is for is worth nothing. pylint 4.1.3 does not catch
     either of them, nor an eight-line reproduction; that was measured.
  2. the per-frame body is inside a try that catches Exception, so a frame
     that raises costs one tick and not the process.
"""
import ast
import os
import subprocess
import sys

fails = []


def check(what, ok, detail=""):
    print(("  ok   " if ok else "  FAIL ") + what + (f"   [{detail}]" if detail else ""))
    if not ok:
        fails.append(what)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bench"))
import unbound_locals  # noqa: E402

PKG = os.path.join(ROOT, "vps_vio")


def _py(d):
    out = []
    for dirpath, _, names in os.walk(d):
        if "__pycache__" in dirpath:
            continue
        out += [os.path.join(dirpath, n) for n in sorted(names)
                if n.endswith(".py")]
    return out


print("1) koi bhi local aise raaste par nahi padha jata jahan wo bandha nahi")
bad = []
for p in _py(PKG):
    bad += unbound_locals.check(p)
check("poore package me zero unbound-local reads", not bad,
      "; ".join(f"{os.path.relpath(x[0], ROOT)}:{x[1]} '{x[2]}'" for x in bad[:6]))

print("\n2) gate un dono bug ko pakadta hai jo asli me ud kar gire")
for commit, name, line in (("83c25d1^", "_fe", 2821),
                           ("a603931^", "_use_pp", 2622)):
    src = subprocess.run(["git", "show", f"{commit}:vps_vio/live_node.py"],
                         cwd=ROOT, capture_output=True, text=True,
                         check=True).stdout
    hits = []
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            hits += [f for f in unbound_locals._Func(n, "x").run()
                     if f[1] == name]
    check(f"{name} pakda gaya (flight me crash hua tha)", bool(hits),
          str([h[0] for h in hits][:4]))
    check(f"{name} sahi line {line} par", any(h[0] == line for h in hits),
          str([h[0] for h in hits][:4]))

print("\n3) har frame ek try ke andar chalta hai jo Exception pakadta hai")
src = open(os.path.join(PKG, "live_node.py"), encoding="utf-8").read()
main = next(n for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.FunctionDef) and n.name == "main")
loops = [n for n in ast.walk(main)
         if isinstance(n, ast.While) and isinstance(n.test, ast.Constant)
         and n.test.value is True]
check("main() me frame loop mila", bool(loops))
if loops:
    loop = max(loops, key=lambda w: len(list(ast.walk(w))))
    tries = [s for s in loop.body if isinstance(s, ast.Try)]
    check("frame body SIRF ek try hai, uske bahar kuch nahi",
          len(tries) == 1 and len(loop.body) == 1,
          f"{len(loop.body)} statements, {len(tries)} try")
    if tries:
        t = tries[0]
        check("except Exception hai",
              any(getattr(h.type, "id", "") == "Exception"
                  for h in t.handlers if h.type is not None))
        check("lagatar fail hone par bare `raise` systemd ko deta hai",
              any(isinstance(n, ast.Raise) and n.exc is None
                  for h in t.handlers for n in ast.walk(h)))
check("CONSEC_MAX maujood hai", "CONSEC_MAX" in src)

print("\nALL PASS" if not fails else f"\n{len(fails)} FAILED: {', '.join(fails)}")
sys.exit(1 if fails else 0)
