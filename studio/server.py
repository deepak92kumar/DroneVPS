#!/usr/bin/env python3
"""DroneVPS Studio - the map cutter and the flight review, served to a browser.

    python3 studio/server.py                 # http://<jetson>:8099
    python3 studio/server.py --port 9000 --host 127.0.0.1

Standard library only, the same shape as `ground_status_server.py`: this has
to come up on a Jetson in a field with no network, so there is no framework
and the page loads no CDN. Heavy work (GDAL, pymavlink, matplotlib) happens
here and the browser is handed pixels and JSON.

Paths are confined to /data/DRONE. The server listens on every interface so
the pilot can open it from a laptop, and anything that will reach a laptop on
a field network should not also be an arbitrary-file reader.
"""
from __future__ import annotations

import argparse
import json
import math
import mimetypes
import os
import posixpath
import sys
import threading
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import analysis                                            # noqa: E402
import geotiff                                             # noqa: E402
import logs                                                # noqa: E402
import plots                                               # noqa: E402
import report                                              # noqa: E402

ROOT = logs.ROOT
STATIC = os.path.join(HERE, "static")
DEFAULT_OUT = os.path.join(ROOT, "maps")

def jsonable(o):
    """Python floats that JSON.parse() will accept.

    json.dumps writes NaN and Infinity by default and the browser's JSON
    parser rejects both - one missing percentile in one finding and the whole
    review fails to load with "Unexpected token N". Every measurement here can
    legitimately be NaN (a band with no rows, a log with no rangefinder), so
    they are turned into null at the edge rather than guarded one by one.
    numpy scalars go the same way: json.dumps cannot write them either.
    """
    import numpy as _np
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, _np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (_np.floating, float)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, _np.integer):
        return int(o)
    if isinstance(o, (_np.bool_, bool)):
        return bool(o)
    return o


_sheets = {}
_sheets_lock = threading.Lock()
_flights = {}
_flights_lock = threading.Lock()


def safe(path: str) -> str:
    """An absolute path inside /data/DRONE, or an error.

    Everything the browser names - a sheet to open, a log to read, a file to
    write - goes through here. The page is served to a field network.
    """
    p = os.path.abspath(os.path.expanduser(path))
    if not (p == ROOT or p.startswith(ROOT + os.sep)):
        raise PermissionError(f"{path} is outside {ROOT}")
    return p


def sheet(path: str) -> geotiff.Sheet:
    p = safe(path)
    with _sheets_lock:
        s = _sheets.get(p)
        if s is None or s.path != p:
            s = geotiff.Sheet(p)
            _sheets[p] = s
        return s


def flight(fc: str | None, node: str | None):
    """One loaded flight, kept in memory so the panels do not reparse it."""
    key = (safe(fc) if fc else None, safe(node) if node else None)
    with _flights_lock:
        b = _flights.get(key)
        if b is None:
            b = analysis.load(key[0], key[1])
            # One flight at a time: a .BIN is 500 000 rows and holding several
            # of them is how a Jetson with 8 GB starts swapping mid-review.
            _flights.clear()
            _flights[key] = b
        return b


def find_sheets():
    """Every GeoTIFF under /data/DRONE, with the big ones first."""
    out, seen = [], set()
    for dirpath, dirnames, files in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames
                       if d not in (".git", "__pycache__", "cache",
                                    "flight_logs", "flight_images", "chat_backup")]
        for f in files:
            if f.lower().endswith((".tif", ".tiff")):
                p = os.path.join(dirpath, f)
                if p in seen:
                    continue
                seen.add(p)
                st = os.stat(p)
                out.append(dict(path=p, name=f,
                                rel=os.path.relpath(p, ROOT),
                                mb=st.st_size / 1e6, mtime=st.st_mtime,
                                ovr=os.path.exists(p + ".ovr")))
    out.sort(key=lambda r: -r["mb"])
    return out


class Handler(BaseHTTPRequestHandler):
    server_version = "DroneVPSStudio/1.0"

    def log_message(self, fmt, *args):          # quiet: one line per request
        sys.stderr.write("  %s - %s\n" % (self.address_string(), fmt % args))

    # ------------------------------------------------------------------ send
    def _send(self, body, ctype="application/json", code=200, cache=False):
        if isinstance(body, (dict, list)):
            body = json.dumps(jsonable(body), allow_nan=False).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if cache:
            self.send_header("Cache-Control", "max-age=3600")
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def _fail(self, exc, code=400):
        msg = f"{type(exc).__name__}: {exc}"
        sys.stderr.write(msg + "\n" + traceback.format_exc())
        self._send(dict(error=msg), code=code)

    # ------------------------------------------------------------------- GET
    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
        try:
            if u.path == "/" or u.path == "/index.html":
                return self._static("index.html")
            if u.path.startswith("/static/"):
                return self._static(u.path[len("/static/"):])
            if u.path == "/api/sheets":
                return self._send(find_sheets())
            if u.path == "/api/sheet":
                return self._send(sheet(q["path"]).describe())
            if u.path == "/api/tile":
                s = sheet(q["path"])
                png = s.tile_png(int(q["z"]), int(q["x"]), int(q["y"]))
                if png is None:
                    return self._send(b"", "image/png", code=204)
                return self._send(png, "image/png", cache=True)
            if u.path == "/api/verify":
                takeoff = ((float(q["lon"]), float(q["lat"]))
                           if q.get("lat") and q.get("lon") else None)
                return self._send(geotiff.verify(safe(q["path"]), takeoff=takeoff))
            if u.path == "/api/logs":
                return self._send(logs.discover())
            if u.path == "/api/pair":
                return self._send(self._pair(q["fc"]))
            if u.path == "/api/flight":
                b = flight(q.get("fc") or None, q.get("node") or None)
                return self._send(dict(summary=analysis.summary(b),
                                       findings=analysis.findings(b),
                                       panels=[dict(name=k, title=v["title"],
                                                    needs=v["needs"])
                                               for k, v in plots.PANELS.items()
                                               if b.get(v["needs"])],
                                       track=analysis.track(b),
                                       frames=self._frames(q.get("node"))))
            if u.path == "/api/panel":
                b = flight(q.get("fc") or None, q.get("node") or None)
                return self._send(plots.render(b, q["name"]), "image/png")
            if u.path == "/api/report":
                b = flight(q.get("fc") or None, q.get("node") or None)
                md = report.markdown(b)
                self.send_response(200)
                self.send_header("Content-Type", "text/markdown; charset=utf-8")
                self.send_header("Content-Disposition",
                                 f'attachment; filename="{report.name(b)}"')
                body = md.encode()
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                return self.wfile.write(body)
            if u.path == "/api/frame":
                p = safe(q["path"])
                ctype = mimetypes.guess_type(p)[0] or "image/jpeg"
                return self._send(open(p, "rb").read(), ctype, cache=True)
            self._send(dict(error="no such endpoint"), code=404)
        except Exception as e:                       # noqa: BLE001
            self._fail(e)

    # ------------------------------------------------------------------ POST
    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            if u.path == "/api/cut":
                return self._send(self._cut(body))
            if u.path == "/api/overviews":
                n = geotiff.build_overviews(safe(body["path"]))
                return self._send(dict(ok=True, levels=n))
            self._send(dict(error="no such endpoint"), code=404)
        except Exception as e:                       # noqa: BLE001
            self._fail(e)

    # --------------------------------------------------------------- helpers
    def _static(self, rel):
        rel = posixpath.normpath("/" + rel).lstrip("/")
        p = os.path.join(STATIC, rel)
        if not p.startswith(STATIC) or not os.path.exists(p):
            return self._send(dict(error="not found"), code=404)
        ctype = mimetypes.guess_type(p)[0] or "application/octet-stream"
        self._send(open(p, "rb").read(), ctype)

    def _cut(self, body):
        src = sheet(body["path"])
        name = (body.get("name") or "cut").strip()
        if not name.lower().endswith((".tif", ".tiff")):
            name += ".tif"
        out_dir = safe(body.get("out_dir") or DEFAULT_OUT)
        out_path = safe(os.path.join(out_dir, os.path.basename(name)))
        kw = dict(out_path=out_path, overwrite=bool(body.get("overwrite")),
                  grey=bool(body.get("grey")),
                  resample=body.get("resample", "cubic"))
        if body.get("out_gsd"):
            kw["out_gsd_m"] = float(body["out_gsd"])
        if body.get("box"):
            kw["box_px"] = [float(v) for v in body["box"]]
        else:
            kw["centre_lonlat"] = (float(body["lon"]), float(body["lat"]))
            kw["size_m"] = (float(body["width_m"]), float(body["height_m"]))
        rep = geotiff.cut(src, **kw)
        if body.get("takeoff_lat") and body.get("takeoff_lon"):
            rep = geotiff.verify(out_path,
                                 takeoff=(float(body["takeoff_lon"]),
                                          float(body["takeoff_lat"])))
            rep["how"] = "written, then re-opened and checked"
        rep["out_path"] = out_path
        return rep

    def _pair(self, fc_path):
        """Which node log belongs to this .BIN, decided by the clocks.

        The wall-clock guess is only a shortlist: vps_00165 and vps_00167 were
        both written within six minutes of flight 245's .BIN and only one of
        them is the flight. Each candidate is aligned against the flight
        controller's altitude and ranked by the correlation that produces,
        which separates a flight from a bench run on the ground immediately.
        """
        arrays, _ = logs.read_bin(safe(fc_path))
        d = logs.discover()
        fc = next((f for f in d["fc"] if f["path"] == safe(fc_path)), None)
        cands = sorted(d["node"],
                       key=lambda nd: abs(nd["mtime"] - (fc["mtime"] if fc else 0)))[:8]
        out = []
        for c in cands:
            try:
                al = logs.align(logs.read_csv(c["csv"], want=("t", "alt_m")),
                                arrays)
            except Exception:
                al = None
            out.append(dict(id=c["id"], csv=c["csv"], when=c["when"],
                            mb=c["mb"], align=al,
                            corr=(al or {}).get("corr", -1)))
        out.sort(key=lambda r: -r["corr"])
        return out

    def _frames(self, node_csv):
        if not node_csv:
            return []
        import glob
        folder = os.path.dirname(safe(node_csv))
        files = sorted(glob.glob(os.path.join(folder, "*.jpg")))
        files += sorted(glob.glob(os.path.join(folder, "mono", "*.png")))
        step = max(1, len(files) // 60)
        return [dict(path=f, name=os.path.basename(f)) for f in files[::step]][:60]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args()
    os.makedirs(DEFAULT_OUT, exist_ok=True)
    os.makedirs(logs.CACHE, exist_ok=True)
    httpd = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"DroneVPS Studio on http://{a.host}:{a.port}  (Ctrl-C to stop)")
    print(f"  sheets and cuts under {ROOT}, cuts default to {DEFAULT_OUT}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
