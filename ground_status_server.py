#!/usr/bin/env python3
"""Live VIO status page for a second laptop after Jetson HDMI is unplugged."""
import glob
import json
import os
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOG_DIR = "/home/abc/SSD/DRONE/flight_logs"
PORTS = (8080, 8765)

LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*"
    r"Pos: N:(?P<n>[+-]?\d+\.\d+)m E:(?P<e>[+-]?\d+\.\d+)m Z:(?P<z>\d+\.\d+)m "
    r"\(Yaw:(?P<yaw>[+-]?\d+\.\d+)°\).*?"
    r"Vel: N:(?P<vn>[+-]?\d+\.\d+) E:(?P<ve>[+-]?\d+\.\d+)m/s.*?"
    r"TF03_H=(?P<tf03>[0-9.]+)m.*?"
    r"\[(?P<hsrc>[^\]]+)\].*?"
    r"VIO Sent: (?P<sent>\d+) \| CUAV: (?P<cuav>.+?) \| "
)


def latest_log():
    files = glob.glob(os.path.join(LOG_DIR, "flight_*.log"))
    return max(files, key=os.path.getmtime) if files else None


def parse_status():
    path = latest_log()
    out = {
        "ok": False,
        "age_s": None,
        "log": path,
        "ts": None,
        "n": None,
        "e": None,
        "z": None,
        "yaw": None,
        "vn": None,
        "ve": None,
        "tf03": None,
        "hsrc": None,
        "sent": None,
        "cuav": "NO LOG",
        "ready": False,
        "hint": "Waiting for VIO log after boot.",
    }
    if not path or not os.path.exists(path):
        return out
    last = None
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 65536), os.SEEK_SET)
            chunk = f.read().decode("utf-8", errors="replace")
        for line in reversed(chunk.splitlines()):
            if "[VIO " in line:
                last = line
                break
    except OSError as exc:
        out["hint"] = str(exc)
        return out
    if not last:
        out["hint"] = "Log exists but no VIO line yet. Wait for camera/CUAV."
        return out
    m = LINE_RE.search(last)
    if not m:
        out["ok"] = True
        out["raw"] = last[-240:]
        out["hint"] = "Could not parse VIO line."
        return out
    d = m.groupdict()
    out.update(
        {
            "ok": True,
            "ts": d["ts"],
            "n": float(d["n"]),
            "e": float(d["e"]),
            "z": float(d["z"]),
            "yaw": float(d["yaw"]),
            "vn": float(d["vn"]),
            "ve": float(d["ve"]),
            "tf03": float(d["tf03"]),
            "hsrc": d["hsrc"],
            "sent": int(d["sent"]),
            "cuav": d["cuav"].strip(),
        }
    )
    try:
        tlog = time.mktime(time.strptime(d["ts"], "%Y-%m-%d %H:%M:%S"))
        out["age_s"] = round(time.time() - tlog, 1)
    except ValueError:
        out["age_s"] = None
    cuav_ok = ("WAITING" not in out["cuav"]) and out["sent"] > 0
    stale = out["age_s"] is not None and out["age_s"] > 8
    tf_ok = out["tf03"] >= 0.35 and "GROUND_GATE" not in out["hsrc"]
    out["ready"] = bool(cuav_ok and not stale)
    if stale:
        out["hint"] = "Log stale. Jetson booting or VIO not running."
    elif not cuav_ok:
        out["hint"] = "CUAV not connected yet. Wait."
    elif not tf_ok:
        out["hint"] = "VIO live. Lift to ~60 cm until TF03 ~0.60, then slide."
    else:
        out["hint"] = "Height OK. Hold 3s, then LEFT 40 cm, RIGHT 40 cm."
    return out


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>VIO ground test</title>
<style>
  body { font-family: ui-sans-serif, system-ui, sans-serif; background:#0b1220; color:#e8eefc; margin:0; padding:16px; }
  .row { display:flex; flex-wrap:wrap; gap:12px; }
  .card { background:#152038; border-radius:12px; padding:16px 18px; min-width:140px; flex:1; }
  .k { font-size:13px; color:#8aa0c8; text-transform:uppercase; letter-spacing:.06em; }
  .v { font-size:42px; font-weight:700; margin-top:4px; font-variant-numeric: tabular-nums; }
  .s { font-size:22px; font-weight:600; margin-top:6px; }
  .ok { color:#4ade80; } .bad { color:#f87171; } .warn { color:#fbbf24; }
  #hint { margin-top:16px; font-size:22px; line-height:1.35; }
  #meta { margin-top:10px; color:#8aa0c8; font-size:14px; }
</style>
</head>
<body>
  <div class="row">
    <div class="card"><div class="k">Vel E</div><div id="ve" class="v">—</div></div>
    <div class="card"><div class="k">Vel N</div><div id="vn" class="v">—</div></div>
    <div class="card"><div class="k">TF03</div><div id="tf" class="v">—</div></div>
    <div class="card"><div class="k">Height src</div><div id="hs" class="s">—</div></div>
  </div>
  <div class="row" style="margin-top:12px">
    <div class="card"><div class="k">Pos E / N</div><div id="pos" class="s">—</div></div>
    <div class="card"><div class="k">Yaw</div><div id="yaw" class="s">—</div></div>
    <div class="card"><div class="k">CUAV</div><div id="cuav" class="s">—</div></div>
  </div>
  <div id="hint"></div>
  <div id="meta"></div>
<script>
async function tick(){
  try {
    const r = await fetch('/api/status', {cache:'no-store'});
    const d = await r.json();
    const f = (x,n)=> (x==null?'—':Number(x).toFixed(n));
    document.getElementById('ve').textContent = f(d.ve,2);
    document.getElementById('vn').textContent = f(d.vn,2);
    document.getElementById('tf').textContent = d.tf03==null?'—':(Number(d.tf03).toFixed(2)+' m');
    document.getElementById('hs').textContent = d.hsrc||'—';
    document.getElementById('hs').className = 's ' + ((d.hsrc||'').includes('GROUND_GATE')?'warn':'ok');
    document.getElementById('pos').textContent = 'E '+f(d.e,2)+'  N '+f(d.n,2);
    document.getElementById('yaw').textContent = f(d.yaw,1)+'°';
    document.getElementById('cuav').textContent = d.cuav||'—';
    document.getElementById('cuav').className = 's ' + (d.ready?'ok':'bad');
    document.getElementById('hint').textContent = d.hint||'';
    document.getElementById('hint').className = d.ready?'ok':'warn';
    document.getElementById('meta').textContent =
      (d.ts||'') + '  age ' + (d.age_s==null?'?':d.age_s+'s') + '  VIO Sent ' + (d.sent||0);
  } catch(e) {
    document.getElementById('hint').textContent = 'Laptop cannot reach Jetson. Same WiFi? Wait for boot.';
    document.getElementById('hint').className = 'bad';
  }
}
tick();
setInterval(tick, 500);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def _send(self, code, body, ctype):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/api/status"):
            self._send(200, json.dumps(parse_status()), "application/json")
            return
        if self.path in ("/", "/index.html"):
            self._send(200, HTML, "text/html; charset=utf-8")
            return
        self._send(404, "not found", "text/plain")


def main():
    import threading

    servers = []
    for port in PORTS:
        httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        servers.append(httpd)
        threading.Thread(target=httpd.serve_forever, daemon=True, name=f"http-{port}").start()
        print(f"ground status http://0.0.0.0:{port}/", flush=True)
    threading.Event().wait()


if __name__ == "__main__":
    main()
