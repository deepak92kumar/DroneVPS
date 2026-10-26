/* DroneVPS Studio - the browser half.
 *
 * No framework and no CDN: this page has to come up on a Jetson in a field.
 * Everything heavy (GDAL, pymavlink, matplotlib) is on the server; this file
 * draws tiles, keeps one crop box honest, and renders what the review
 * engine returns.
 */
'use strict';

const $ = s => document.querySelector(s);
const el = (tag, cls, txt) => { const n = document.createElement(tag);
  if (cls) n.className = cls; if (txt !== undefined) n.textContent = txt; return n; };
const status = t => { $('#status').textContent = t; };
const fmt = (v, n = 2) => (v === null || v === undefined || Number.isNaN(v)) ? '-' : (+v).toFixed(n);

async function api(path, opts) {
  const r = await fetch(path, opts);
  const ct = r.headers.get('content-type') || '';
  if (!ct.includes('json')) throw new Error(await r.text());
  const j = await r.json();
  if (j.error) throw new Error(j.error);
  return j;
}

/* A position typed by a human: decimal pairs, or degrees-minutes-seconds with
 * hemispheres. Returns [lat, lon] because that is the order people speak and
 * the order ArduPilot's SCR_USER1/2 are in. */
function parseLatLon(s) {
  if (!s) return null;
  s = s.trim().replace(/[,;]+/g, ' ').replace(/[°'"″′]/g, ' ').replace(/\s+/g, ' ');
  const dms = s.match(/^(-?[\d.]+) ([\d.]+) ([\d.]+) ?([NSns]) (-?[\d.]+) ([\d.]+) ([\d.]+) ?([EWew])$/);
  if (dms) {
    const d = (a, b, c, h) => (Math.abs(+a) + +b / 60 + +c / 3600) *
      (/[SsWw]/.test(h) || +a < 0 ? -1 : 1);
    return [d(dms[1], dms[2], dms[3], dms[4]), d(dms[5], dms[6], dms[7], dms[8])];
  }
  const p = s.split(' ').map(Number).filter(v => !Number.isNaN(v));
  if (p.length >= 2) return [p[0], p[1]];
  return null;
}

/* =======================================================================
 *  the sheet view
 * ===================================================================== */
class MapView {
  constructor(canvasSel, wrapSel) {
    this.cv = $(canvasSel); this.wrap = $(wrapSel);
    this.ctx = this.cv.getContext('2d');
    this.sheet = null; this.tiles = new Map(); this.pending = 0;
    this.cx = 0; this.cy = 0; this.scale = 1;
    this.box = null;          // [x0,y0,x1,y1] in SOURCE pixels
    this.marker = null;       // [lon,lat]
    this.tracks = null;       // {ours:[], truth:[], fixes:[]}
    this.onBox = null; this.onHover = null;
    this.interactive = true;
    this._bind();
    new ResizeObserver(() => this.resize()).observe(this.wrap);
  }
  resize() {
    const r = this.wrap.getBoundingClientRect(), dpr = devicePixelRatio || 1;
    this.cv.width = Math.max(1, r.width * dpr); this.cv.height = Math.max(1, r.height * dpr);
    this.W = r.width; this.H = r.height; this.dpr = dpr;
    // The fit happens HERE and not in setSheet, because a sheet can be opened
    // before the pane has been laid out - and then the height is 0, the fit
    // scale comes out negative, and the whole sheet is drawn inside out off
    // the left of the screen with no error anywhere.
    if (this.needFit && this.sheet && this.W > 50 && this.H > 50) {
      this.fit(); this.needFit = false;
    }
    this.draw();
  }
  fit() {
    const d = this.sheet;
    this.cx = d.width / 2; this.cy = d.height / 2;
    this.scale = Math.max(1e-5, Math.min((this.W - 40) / d.width,
                                         (this.H - 40) / d.height));
  }
  setSheet(d, keepView) {
    this.sheet = d; this.tiles.clear();
    if (!keepView) { this.needFit = true; this.box = null; }
    this.resize();
  }
  /* ---- coordinates. The affine is exact for any north-up geographic sheet;
   * for a projected or rotated source the four corners are interpolated and
   * the readout is marked approximate, because the real transform needs a
   * projection library that is not worth shipping to the browser. */
  approx() { const d = this.sheet; return !(d && d.geographic && d.north_up); }
  pixelToLonLat(px, py) {
    const d = this.sheet; if (!d) return [0, 0];
    if (!this.approx()) {
      const g = d.geotransform;
      return [g[0] + px * g[1] + py * g[2], g[3] + px * g[4] + py * g[5]];
    }
    const u = px / d.width, v = py / d.height, c = d.corners;
    const lerp = (a, b, t) => a + (b - a) * t;
    const top = [lerp(c[0][0], c[1][0], u), lerp(c[0][1], c[1][1], u)];
    const bot = [lerp(c[3][0], c[2][0], u), lerp(c[3][1], c[2][1], u)];
    return [lerp(top[0], bot[0], v), lerp(top[1], bot[1], v)];
  }
  lonLatToPixel(lon, lat) {
    const d = this.sheet; if (!d) return [0, 0];
    if (!this.approx()) {
      const g = d.geotransform, det = g[1] * g[5] - g[2] * g[4];
      const dx = lon - g[0], dy = lat - g[3];
      return [(dx * g[5] - dy * g[2]) / det, (dy * g[1] - dx * g[4]) / det];
    }
    const b = d.bounds;
    return [(lon - b[0]) / (b[2] - b[0]) * d.width,
            (b[3] - lat) / (b[3] - b[1]) * d.height];
  }
  toScreen(px, py) { return [(px - this.cx) * this.scale + this.W / 2,
                             (py - this.cy) * this.scale + this.H / 2]; }
  toPixel(sx, sy) { return [(sx - this.W / 2) / this.scale + this.cx,
                            (sy - this.H / 2) / this.scale + this.cy]; }
  /* ---- drawing */
  /* A redraw on the next frame, with a timer behind it.
   *
   * requestAnimationFrame alone is not enough: a browser throttles or stops
   * it entirely for a page it considers hidden - a background tab, or a
   * preview pane that is not on screen. The tiles then finish loading, ask
   * for a redraw, and nothing happens until the pointer moves, so the sheet
   * looks black for no reason. The timer fires if the frame does not. */
  requestDraw() {
    if (this._pending) return;
    this._pending = true;
    const run = () => {
      if (!this._pending) return;
      this._pending = false;
      clearTimeout(this._timer);
      cancelAnimationFrame(this._frame);
      this.draw();
    };
    this._frame = requestAnimationFrame(run);
    this._timer = setTimeout(run, 120);
  }
  draw() {
    const g = this.ctx, d = this.sheet;
    if (!g) return;
    g.setTransform(this.dpr || 1, 0, 0, this.dpr || 1, 0, 0);
    g.fillStyle = '#070a0e'; g.fillRect(0, 0, this.W, this.H);
    if (!d) return;
    if (!(this.scale > 0) || !(this.W > 0)) return;
    const level = Math.max(0, Math.min(d.max_level,
      Math.round(Math.log2(1 / this.scale)) || 0));
    const step = 1 << level, span = 256 * step;
    const [x0, y0] = this.toPixel(0, 0), [x1, y1] = this.toPixel(this.W, this.H);
    const tx0 = Math.max(0, Math.floor(x0 / span)), tx1 = Math.min(Math.ceil(d.width / span), Math.ceil(x1 / span));
    const ty0 = Math.max(0, Math.floor(y0 / span)), ty1 = Math.min(Math.ceil(d.height / span), Math.ceil(y1 / span));
    g.imageSmoothingEnabled = true;
    for (let ty = ty0; ty < ty1; ty++) for (let tx = tx0; tx < tx1; tx++) {
      const img = this.tile(level, tx, ty);
      if (!img) continue;
      const [sx, sy] = this.toScreen(tx * span, ty * span);
      const w = span * this.scale;
      g.drawImage(img, sx, sy, w + 1, w + 1);
    }
    this.drawTracks(g);
    this.drawBox(g);
    this.drawMarker(g);
  }
  tile(z, x, y) {
    const key = z + '/' + x + '/' + y;
    let t = this.tiles.get(key);
    if (t) return t.ok ? t.img : null;
    const img = new Image();
    const rec = { img, ok: false };
    this.tiles.set(key, rec);
    img.onload = () => { rec.ok = true; this.requestDraw(); };
    img.onerror = () => { rec.ok = false; };
    img.src = `/api/tile?path=${encodeURIComponent(this.sheet.path)}&z=${z}&x=${x}&y=${y}`;
    return null;
  }
  drawBox(g) {
    if (!this.box) return;
    const [a, b] = [this.toScreen(this.box[0], this.box[1]),
                    this.toScreen(this.box[2], this.box[3])];
    g.save();
    g.fillStyle = 'rgba(77,163,255,.10)';
    g.fillRect(a[0], a[1], b[0] - a[0], b[1] - a[1]);
    g.strokeStyle = '#4da3ff'; g.lineWidth = 1.4;
    g.strokeRect(a[0], a[1], b[0] - a[0], b[1] - a[1]);
    g.fillStyle = '#4da3ff';
    for (const [hx, hy] of this.handles(a, b)) g.fillRect(hx - 3.5, hy - 3.5, 7, 7);
    g.restore();
  }
  handles(a, b) {
    const mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2;
    return [[a[0], a[1]], [mx, a[1]], [b[0], a[1]], [b[0], my],
            [b[0], b[1]], [mx, b[1]], [a[0], b[1]], [a[0], my]];
  }
  drawMarker(g) {
    if (!this.marker) return;
    const [px, py] = this.lonLatToPixel(this.marker[0], this.marker[1]);
    const [sx, sy] = this.toScreen(px, py);
    g.save(); g.strokeStyle = '#ffb34d'; g.lineWidth = 1.6;
    g.beginPath(); g.arc(sx, sy, 8, 0, 7); g.stroke();
    g.beginPath(); g.moveTo(sx - 14, sy); g.lineTo(sx - 4, sy);
    g.moveTo(sx + 4, sy); g.lineTo(sx + 14, sy);
    g.moveTo(sx, sy - 14); g.lineTo(sx, sy - 4);
    g.moveTo(sx, sy + 4); g.lineTo(sx, sy + 14); g.stroke(); g.restore();
  }
  drawTracks(g) {
    const t = this.tracks; if (!t) return;
    const line = (pts, colour, width) => {
      if (!pts || !pts.length) return;
      g.save(); g.strokeStyle = colour; g.lineWidth = width; g.beginPath();
      let started = false;
      for (const p of pts) {
        const [px, py] = this.lonLatToPixel(p[0], p[1]);
        const [sx, sy] = this.toScreen(px, py);
        if (!started) { g.moveTo(sx, sy); started = true; } else g.lineTo(sx, sy);
      }
      g.stroke(); g.restore();
    };
    line(t.truth, '#ffb34d', 1.6);
    line(t.ours, '#4da3ff', 1.6);
    if (t.fixes) { g.save(); g.fillStyle = '#4ade80';
      for (const f of t.fixes) {
        const [px, py] = this.lonLatToPixel(f[0], f[1]);
        const [sx, sy] = this.toScreen(px, py);
        g.beginPath(); g.arc(sx, sy, 2.4, 0, 7); g.fill();
      } g.restore(); }
  }
  /* ---- interaction */
  _bind() {
    const cv = this.cv;
    let drag = null;
    // Right-drag pans, so the context menu must not open on top of it.
    cv.addEventListener('contextmenu', e => e.preventDefault());
    cv.addEventListener('mousedown', e => {
      if (!this.sheet || !this.interactive) return;
      const r = cv.getBoundingClientRect();
      const sx = e.clientX - r.left, sy = e.clientY - r.top;
      // The first version made a plain drag pan and reserved shift+drag for
      // the crop box, while the button above it said "Drag a box". The pilot
      // dragged, the sheet moved, and no box was ever drawn - the server log
      // has 110 requests from the laptop and not one cut. So in box mode a
      // plain drag now does what the button says, and panning moves to the
      // right button, the middle button, or shift.
      const wantPan = e.shiftKey || e.button === 1 || e.button === 2;
      const h = wantPan ? -1 : this.hitHandle(sx, sy);
      const [px, py] = this.toPixel(sx, sy);
      if (h >= 0) {
        drag = { kind: 'handle', h };
      } else if (!wantPan && this.boxTool && this.inBox(px, py)) {
        drag = { kind: 'move', px, py, box: this.box.slice() };
      } else if (!wantPan && this.boxTool) {
        this.box = [px, py, px, py];
        drag = { kind: 'handle', h: 4 };
      } else {
        drag = { kind: 'pan', sx, sy, cx: this.cx, cy: this.cy };
      }
      e.preventDefault();
    });
    addEventListener('mousemove', e => {
      if (!this.sheet) return;
      const r = cv.getBoundingClientRect();
      const sx = e.clientX - r.left, sy = e.clientY - r.top;
      if (!drag) {
        if (sx >= 0 && sy >= 0 && sx < r.width && sy < r.height && this.onHover)
          this.onHover(...this.toPixel(sx, sy));
        if (this.interactive) {
          const [px, py] = this.toPixel(sx, sy);
          cv.style.cursor = this.hitHandle(sx, sy) >= 0 ? 'crosshair'
            : (this.boxTool && this.inBox(px, py)) ? 'move'
            : this.boxTool ? 'crosshair' : 'grab';
        }
        return;
      }
      if (drag.kind === 'pan') {
        this.cx = drag.cx - (sx - drag.sx) / this.scale;
        this.cy = drag.cy - (sy - drag.sy) / this.scale;
      } else if (drag.kind === 'move') {
        const [px, py] = this.toPixel(sx, sy);
        const dx = px - drag.px, dy = py - drag.py, b0 = drag.box;
        this.box = [b0[0] + dx, b0[1] + dy, b0[2] + dx, b0[3] + dy];
        if (this.onBox) this.onBox(this.normBox());
      } else {
        const [px, py] = this.toPixel(sx, sy), b = this.box, h = drag.h;
        if ([0, 6, 7].includes(h)) b[0] = px;
        if ([2, 3, 4].includes(h)) b[2] = px;
        if ([0, 1, 2].includes(h)) b[1] = py;
        if ([4, 5, 6].includes(h)) b[3] = py;
        if (this.onBox) this.onBox(this.normBox());
      }
      this.requestDraw();
    });
    addEventListener('mouseup', () => {
      if (drag && (drag.kind === 'handle' || drag.kind === 'move')) {
        // A click with no movement in box mode should not leave a zero-sized
        // box behind, which would then swallow every later click as a "move".
        const b = this.normBox();
        this.box = (b && (b[2] - b[0] > 2) && (b[3] - b[1] > 2)) ? b : null;
        if (this.onBox) this.onBox(this.box);
      }
      drag = null;
    });
    cv.addEventListener('dblclick', () => {
      this.box = null; if (this.onBox) this.onBox(null); this.requestDraw(); });
    cv.addEventListener('wheel', e => {
      if (!this.sheet) return;
      // A map embedded in a scrolling page must not eat the page's scroll:
      // the review is a long page and the track sits in the middle of it.
      if (this.wheelNeedsModifier && !(e.ctrlKey || e.metaKey)) return;
      e.preventDefault();
      const r = cv.getBoundingClientRect();
      const sx = e.clientX - r.left, sy = e.clientY - r.top;
      const before = this.toPixel(sx, sy);
      const k = Math.exp(-e.deltaY * 0.0016);
      this.scale = Math.max(1 / (1 << (this.sheet.max_level + 1)),
                            Math.min(16, this.scale * k));
      const after = this.toPixel(sx, sy);
      this.cx += before[0] - after[0]; this.cy += before[1] - after[1];
      this.requestDraw();
    }, { passive: false });
  }
  hitHandle(sx, sy) {
    if (!this.box || !this.interactive) return -1;
    const a = this.toScreen(this.box[0], this.box[1]);
    const b = this.toScreen(this.box[2], this.box[3]);
    const hs = this.handles(a, b);
    for (let i = 0; i < hs.length; i++)
      if (Math.abs(sx - hs[i][0]) < 6 && Math.abs(sy - hs[i][1]) < 6) return i;
    return -1;
  }
  inBox(px, py) {
    const b = this.normBox();
    return !!b && px > b[0] && px < b[2] && py > b[1] && py < b[3];
  }
  normBox() {
    const b = this.box; if (!b) return null;
    return [Math.min(b[0], b[2]), Math.min(b[1], b[3]),
            Math.max(b[0], b[2]), Math.max(b[1], b[3])];
  }
  centreOn(lon, lat) {
    const [px, py] = this.lonLatToPixel(lon, lat);
    this.cx = px; this.cy = py;
    if (this.scale < 0.25) this.scale = 0.25;
    this.requestDraw();
  }
}

/* =======================================================================
 *  tab 1: the map cutter
 * ===================================================================== */
const Map1 = {
  view: null, sheets: [], sheet: null, mode: 'box',

  async init() {
    this.view = new MapView('#map-canvas', '#map-wrap');
    this.view.onHover = (px, py) => this.hover(px, py);
    this.view.onBox = b => this.boxChanged(b);
    $('#btn-goto').onclick = () => this.goto();
    $('#goto').addEventListener('keydown', e => { if (e.key === 'Enter') this.goto(); });
    $('#btn-centre-here').onclick = () => this.centreCrop();
    $('#mode-box').onclick = () => this.setMode('box');
    $('#btn-clear-box').onclick = () => this.clearBox();
    $('#mode-centre').onclick = () => this.setMode('centre');
    $('#btn-cut').onclick = () => this.cut();
    $('#btn-activate').onclick = () => this.activate(this.sheet && this.sheet.path);
    $('#btn-check').onclick = () => this.check(this.sheet && this.sheet.path);
    $('#btn-deactivate').onclick = () => this.deactivate();
    $('#btn-ovr').onclick = () => this.overviews();
    for (const id of ['#c-lat', '#c-lon', '#c-w', '#c-h'])
      $(id).addEventListener('input', () => this.centreChanged());
    this.setMode('box');
    await this.loadSheets();
    await this.loadActive();
  },
  async loadSheets(keep) {
    try {
      this.sheets = await api('/api/sheets');
      const list = $('#sheet-list'); list.innerHTML = '';
      if (!this.sheets.length) list.appendChild(el('div', 'hint',
        'No GeoTIFF found under /data/DRONE.'));
      this.sheets.forEach(s => {
        const n = el('div', 'item');
        n.appendChild(el('b', null, s.name));
        n.appendChild(el('em', null, s.rel.includes('/') ? s.rel.split('/').slice(0, -1).join('/') : ''));
        n.appendChild(el('i', null, `${s.mb.toFixed(0)} MB${s.ovr ? ' +ovr' : ''}`));
        n.onclick = () => this.open(s.path, n);
        if (keep && this.sheet && s.path === this.sheet.path) n.classList.add('on');
        list.appendChild(n);
      });
      // After a cut the list has a new entry, but re-opening the source would
      // throw away the crop box the pilot just used - and the usual next step
      // is to cut a second sheet beside the first.
      if (!keep && this.sheets.length) this.open(this.sheets[0].path, list.firstChild);
    } catch (e) { status('sheets: ' + e.message); }
  },
  async open(path, node) {
    status('opening ' + path.split('/').pop() + '...');
    try {
      const d = await api('/api/sheet?path=' + encodeURIComponent(path));
      this.sheet = d;
      document.querySelectorAll('#sheet-list .item').forEach(n => n.classList.remove('on'));
      if (node) node.classList.add('on');
      this.view.setSheet(d);
      this.showInfo(d);
      if (!$('#out-name').value)
        $('#out-name').value = d.name.replace(/\.tiff?$/i, '') + '_cut.tif';
      status('ready');
    } catch (e) { status('open: ' + e.message); }
  },
  showInfo(d) {
    $('#sheet-info').style.display = '';
    const kv = $('#sheet-kv'); kv.innerHTML = '';
    const add = (k, v) => { const tr = kv.insertRow();
      tr.insertCell().textContent = k; tr.insertCell().textContent = v; };
    add('size', `${d.width} x ${d.height} px`);
    add('ground', `${fmt(d.ground_w_km)} x ${fmt(d.ground_h_km)} km`);
    add('resolution', `${fmt(d.gsd_ns_m, 4)} N-S / ${fmt(d.gsd_ew_m, 4)} E-W m/px`);
    add('CRS', (d.epsg ? 'EPSG:' + d.epsg + ' ' : '') + d.crs);
    add('north-up', d.north_up ? 'yes' : 'NO - will be rectified');
    add('bands', `${d.bands} x ${d.dtype}`);
    add('file', `${fmt(d.file_mb, 0)} MB${d.overviews ? `, ${d.overviews} overviews` : ', no overviews'}`);
    add('as grey in RAM', `${fmt(d.grey_mb, 0)} MB`);
    add('north-west', `${fmt(d.bounds[3], 6)}, ${fmt(d.bounds[0], 6)}`);
    add('south-east', `${fmt(d.bounds[1], 6)}, ${fmt(d.bounds[2], 6)}`);
    const iss = $('#sheet-issues'); iss.innerHTML = '';
    d.issues.forEach(i => iss.appendChild(el('div', 'banner ' + (i.level === 'warn' ? 'warn' : 'ok'), i.text)));
    $('#btn-ovr').style.display = (d.overviews || d.file_mb < 40) ? 'none' : '';
  },
  hover(px, py) {
    const d = this.sheet; if (!d) return;
    const [lon, lat] = this.view.pixelToLonLat(px, py);
    const a = this.view.approx() ? '~' : '';
    $('#st-pos').innerHTML = `${a}<b>${fmt(lat, 6)}, ${fmt(lon, 6)}</b>`;
    $('#st-zoom').textContent =
      `${fmt(d.gsd_ns_m / this.view.scale, 3)} m per screen px`;
  },
  setMode(m) {
    this.mode = m;
    $('#mode-box').classList.toggle('on', m === 'box');
    $('#mode-centre').classList.toggle('on', m === 'centre');
    $('#crop-centre').style.display = m === 'centre' ? '' : 'none';
    // In box mode the canvas IS the tool: a plain drag draws the box. In
    // centre mode the box comes from the numbers, so a drag goes back to
    // panning and only the handles still edit it.
    this.view.boxTool = (m === 'box');
    $('#help-box').style.display = m === 'box' ? '' : 'none';
    $('#help-pan').style.display = m === 'box' ? 'none' : '';
    if (m === 'centre') this.centreChanged();
  },
  clearBox() {
    this.view.box = null; this.showBox(null); this.view.requestDraw();
  },
  boxChanged(b) {
    if (this.mode === 'centre' && b) this.setMode('box');
    this.showBox(b);
  },
  showBox(b) {
    const d = this.sheet, kv = $('#crop-kv'); kv.innerHTML = '';
    if (!b || !d) { $('#st-box').textContent = 'no crop box';
      $('#st-ram').textContent = ''; return; }
    const wpx = Math.round(b[2] - b[0]), hpx = Math.round(b[3] - b[1]);
    const wm = wpx * d.gsd_ew_m, hm = hpx * d.gsd_ns_m;
    const c = this.view.pixelToLonLat((b[0] + b[2]) / 2, (b[1] + b[3]) / 2);
    const add = (k, v) => { const tr = kv.insertRow();
      tr.insertCell().textContent = k; tr.insertCell().textContent = v; };
    add('centre', `${fmt(c[1], 6)}, ${fmt(c[0], 6)}`);
    add('ground', `${fmt(wm, 0)} x ${fmt(hm, 0)} m`);
    add('pixels', `${wpx} x ${hpx}`);
    add('grey in RAM', `${fmt(wpx * hpx / 1e6, 0)} MB`);
    $('#st-box').innerHTML = `box <b>${fmt(wm, 0)} x ${fmt(hm, 0)} m</b> (${wpx} x ${hpx} px)`;
    $('#st-ram').textContent = `${fmt(wpx * hpx / 1e6, 0)} MB as grey`;
  },
  centreChanged() {
    const d = this.sheet; if (!d || this.mode !== 'centre') return;
    const lat = parseFloat($('#c-lat').value), lon = parseFloat($('#c-lon').value);
    const wm = parseFloat($('#c-w').value), hm = parseFloat($('#c-h').value);
    if ([lat, lon, wm, hm].some(v => Number.isNaN(v))) return;
    const [px, py] = this.view.lonLatToPixel(lon, lat);
    const hw = wm / d.gsd_ew_m / 2, hh = hm / d.gsd_ns_m / 2;
    this.view.box = [px - hw, py - hh, px + hw, py + hh];
    this.view.marker = [lon, lat];
    this.showBox(this.view.box);
    this.view.requestDraw();
  },
  goto() {
    const p = parseLatLon($('#goto').value);
    if (!p) { status('could not read that position'); return; }
    this.view.marker = [p[1], p[0]];
    this.view.centreOn(p[1], p[0]);
    const d = this.sheet;
    const inside = d && p[1] >= d.bounds[0] && p[1] <= d.bounds[2] &&
                   p[0] >= d.bounds[1] && p[0] <= d.bounds[3];
    status(inside ? 'on this sheet' : 'OFF this sheet - no imagery there');
  },
  centreCrop() {
    const p = parseLatLon($('#goto').value); if (!p) return;
    $('#c-lat').value = p[0]; $('#c-lon').value = p[1];
    this.setMode('centre');
  },
  /* ---- which sheet the node will fly, and proving it opens ------------- */
  takeoff() {
    const t = parseLatLon($('#takeoff').value) || parseLatLon($('#goto').value);
    return t ? { lat: t[0], lon: t[1] } : null;
  },
  async loadActive() {
    try {
      const a = await api('/api/active');
      this.active = a;
      const b = $('#active-body'); b.innerHTML = '';
      if (!a.path) {
        b.appendChild(el('div', 'banner warn', 'No sheet chosen yet.'));
      } else {
        const name = a.path.split('/').pop();
        b.appendChild(el('div', 'banner ' + (a.exists ? 'ok' : 'alarm'),
          a.exists ? `active.tif points at ${name}`
                   : `${a.path} is NOT on this machine`));
        const kv = el('table', 'kv');
        const add = (k, v) => { const tr = kv.insertRow();
          tr.insertCell().textContent = k; tr.insertCell().textContent = v; };
        add('sheet', a.path);
        if (a.sheet) {
          add('ground', `${fmt(a.sheet.ground_w_km)} x ${fmt(a.sheet.ground_h_km)} km`);
          add('resolution', `${fmt(a.sheet.gsd_ns_m, 4)} m/px N-S`);
        }
        b.appendChild(kv);
        const dl = el('p', 'hint');
        dl.innerHTML = `<a href="/api/download?path=${encodeURIComponent(a.path)}" ` +
          `style="color:#4da3ff">Download this sheet</a> to take it away.`;
        b.appendChild(dl);
      }
      const cmd = el('p', 'hint');
      cmd.innerHTML = 'Fly it with <code>' + a.fly + '</code>';
      b.appendChild(cmd);
    } catch (e) { status('chosen sheet: ' + e.message); }
  },
  async activate(path) {
    if (!path) { status('pick a sheet first'); return; }
    status('pointing active.tif at ' + path.split('/').pop() + '...');
    $('#btn-activate').disabled = true;
    try {
      const body = { path };
      const t = this.takeoff();
      if (t) Object.assign(body, t);
      const r = await api('/api/activate', { method: 'POST',
        headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      this.showCheck(r);
      await this.loadActive();
      status(r.ok ? 'chosen - active.tif now points at it'
                  : 'chosen, but it will NOT work - see the checks');
    } catch (e) { status('load: ' + e.message); }
    $('#btn-activate').disabled = false;
  },
  async check(path) {
    if (!path) { status('pick a sheet first'); return; }
    status('opening it with the node\'s own MapCache...');
    $('#btn-check').disabled = true;
    try {
      const t = this.takeoff();
      const q = `path=${encodeURIComponent(path)}` +
        (t ? `&lat=${t.lat}&lon=${t.lon}` : '');
      this.showCheck(await api('/api/loadcheck?' + q));
      status('ready');
    } catch (e) { status('check: ' + e.message); }
    $('#btn-check').disabled = false;
  },
  async deactivate() {
    try {
      await api('/api/deactivate', { method: 'POST',
        headers: { 'Content-Type': 'application/json' }, body: '{}' });
      $('#check-body').innerHTML = '';
      await this.loadActive();
      status('choice cleared');
    } catch (e) { status('clear: ' + e.message); }
  },
  showCheck(r, into) {
    const b = into || $('#check-body');
    b.innerHTML = '';
    b.appendChild(el('div', 'banner ' + (r.ok ? 'ok' : 'alarm'), r.verdict));
    r.checks.forEach(c => {
      const n = el('div', 'check ' + (c.ok ? 'pass' : (c.hard ? 'fail' : 'soft')));
      n.appendChild(el('span', 'm', c.ok ? 'PASS' : (c.hard ? 'FAIL' : 'NOTE')));
      const d = el('div');
      d.appendChild(el('b', null, c.name));
      d.appendChild(el('em', null, c.detail));
      if (c.fix) { const f = el('em'); f.style.color = '#fbbf24';
        f.textContent = c.fix; d.appendChild(f); }
      n.appendChild(d); b.appendChild(n);
    });
    const t = el('p', 'hint', `checked in ${r.took_s} s with the node's own MapCache`);
    b.appendChild(t);
  },
  async overviews() {
    if (!this.sheet) return;
    status('building overviews, this takes a minute...');
    $('#btn-ovr').disabled = true;
    try {
      await api('/api/overviews', { method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: this.sheet.path }) });
      this.view.tiles.clear(); this.view.requestDraw();
      status('overviews built');
      this.open(this.sheet.path);
    } catch (e) { status('overviews: ' + e.message); }
    $('#btn-ovr').disabled = false;
  },
  async cut() {
    const d = this.sheet; if (!d) return;
    const body = { path: d.path, name: $('#out-name').value,
                   out_dir: $('#out-dir').value, overwrite: true,
                   grey: !!$('#out-grey').value,
                   out_gsd: $('#out-gsd').value || null };
    if (this.mode === 'centre') {
      const lat = parseFloat($('#c-lat').value), lon = parseFloat($('#c-lon').value);
      if (Number.isNaN(lat) || Number.isNaN(lon)) { status('type a centre first'); return; }
      Object.assign(body, { lat, lon, width_m: parseFloat($('#c-w').value),
                            height_m: parseFloat($('#c-h').value) });
    } else {
      const b = this.view.normBox();
      if (!b) { status('shift-drag a box on the sheet first'); return; }
      body.box = b;
    }
    const t = parseLatLon($('#takeoff').value);
    if (t) { body.takeoff_lat = t[0]; body.takeoff_lon = t[1]; }
    $('#btn-cut').disabled = true; status('cutting...');
    try {
      const rep = await api('/api/cut', { method: 'POST',
        headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      this.showResult(rep);
      status('written: ' + rep.out_path);
      this.loadSheets(true);
    } catch (e) { status('cut: ' + e.message);
      $('#cut-result').style.display = '';
      $('#cut-body').innerHTML = '';
      $('#cut-body').appendChild(el('div', 'banner alarm', e.message)); }
    $('#btn-cut').disabled = false;
  },
  showResult(rep) {
    $('#cut-result').style.display = '';
    const b = $('#cut-body'); b.innerHTML = '';
    b.appendChild(el('div', 'banner ' + (rep.ok ? 'ok' : 'alarm'),
      rep.ok ? 'Ready to fly: every assumption map_cache.py makes is satisfied.'
             : 'NOT usable as it stands - see the failures below.'));
    const kv = el('table', 'kv');
    const add = (k, v) => { const tr = kv.insertRow();
      tr.insertCell().textContent = k; tr.insertCell().textContent = v; };
    add('file', rep.out_path);
    add('size', `${rep.width} x ${rep.height} px, ${fmt(rep.file_mb, 1)} MB`);
    add('resolution', `${fmt(rep.gsd_ns_m, 4)} / ${fmt(rep.gsd_ew_m, 4)} m/px`);
    add('ground', `${fmt(rep.ground_w_km)} x ${fmt(rep.ground_h_km)} km`);
    add('how', rep.how || '');
    b.appendChild(kv);
    rep.checks.forEach(c => {
      const n = el('div', 'check ' + (c.ok ? 'pass' : (c.hard ? 'fail' : 'soft')));
      n.appendChild(el('span', 'm', c.ok ? 'PASS' : (c.hard ? 'FAIL' : 'NOTE')));
      const d2 = el('div');
      d2.appendChild(el('b', null, c.name));
      d2.appendChild(el('em', null, c.detail));
      n.appendChild(d2); b.appendChild(n);
    });
    const row = el('div', 'row'); row.style.marginTop = '10px';
    const load = el('button', 'b', 'Use this cut as the sheet to fly');
    load.onclick = () => this.activate(rep.out_path);
    const dl = el('button', 'b ghost', 'Download it');
    dl.onclick = () => { location.href =
      '/api/download?path=' + encodeURIComponent(rep.out_path); };
    row.appendChild(load); row.appendChild(dl);
    b.appendChild(row);
    const cmd = el('p', 'hint');
    cmd.innerHTML = 'Or fly it once with <code>python3 vps_vio/live_node.py --map ' +
      rep.out_path + '</code>';
    b.appendChild(cmd);
  },
};

/* =======================================================================
 *  tab 2: the flight review
 * ===================================================================== */
const Log = {
  fc: null, node: null, data: null, view: null,

  async init() {
    $('#btn-review').onclick = () => this.review();
    $('#btn-pair').onclick = () => this.pair();
    $('#btn-report').onclick = () => {
      location.href = `/api/report?fc=${encodeURIComponent(this.fc || '')}` +
                      `&node=${encodeURIComponent(this.node || '')}`;
    };
    await this.load();
  },
  async load() {
    $('#fc-list').innerHTML = '<div class="hint"><span class="spinner"></span> reading /data/DRONE/flight_logs</div>';
    $('#node-list').innerHTML = '<div class="hint"><span class="spinner"></span> reading the node logs</div>';
    try {
      const d = await api('/api/logs');
      this.fill('#fc-list', d.fc, r => {
        this.fc = r.path; this.suggest = r.suggest;
      }, r => [r.name, r.when, `${r.mb.toFixed(0)} MB${r.cached ? ' *' : ''}`]);
      this.fill('#node-list', d.node, r => { this.node = r.csv; },
        r => [r.id, r.when, `${r.mb.toFixed(1)} MB${r.frames ? ` ${r.frames} img` : ''}`]);
    } catch (e) { status('logs: ' + e.message); }
  },
  fill(sel, rows, pick, cols) {
    const list = $(sel); list.innerHTML = '';
    rows.forEach(r => {
      const n = el('div', 'item');
      const c = cols(r);
      n.appendChild(el('b', null, c[0]));
      n.appendChild(el('em', null, c[1]));
      n.appendChild(el('i', null, c[2]));
      n.onclick = () => {
        list.querySelectorAll('.item').forEach(x => x.classList.remove('on'));
        n.classList.add('on'); pick(r);
      };
      n.dataset.id = r.id;
      list.appendChild(n);
    });
  },
  async pair() {
    if (!this.fc) { status('pick a .BIN first'); return; }
    status('aligning candidates...');
    $('#btn-pair').disabled = true;
    try {
      const r = await api('/api/pair?fc=' + encodeURIComponent(this.fc));
      const best = r.find(x => x.corr > 0.9);
      const list = $('#node-list');
      r.forEach(c => {
        const n = list.querySelector(`[data-id="${c.id}"]`);
        if (n) n.querySelector('i').textContent =
          c.corr > 0 ? `r=${c.corr.toFixed(3)}` : 'no overlap';
      });
      if (best) {
        this.node = best.csv;
        list.querySelectorAll('.item').forEach(x =>
          x.classList.toggle('on', x.dataset.id === best.id));
        const n = list.querySelector(`[data-id="${best.id}"]`);
        if (n) n.scrollIntoView({ block: 'nearest' });
        status(`paired with ${best.id} (r=${best.corr.toFixed(4)}, ` +
               `lag ${best.align.lag_s.toFixed(0)} s)`);
      } else status('no node log correlates with this .BIN');
    } catch (e) { status('pair: ' + e.message); }
    $('#btn-pair').disabled = false;
  },
  async review() {
    if (!this.fc && !this.node) { status('pick a log first'); return; }
    const scroll = $('#log-scroll');
    scroll.innerHTML = '<div class="empty"><span class="spinner"></span> ' +
      'reading the logs - a .BIN takes about 15 s the first time</div>';
    status('reviewing...');
    try {
      const q = `fc=${encodeURIComponent(this.fc || '')}&node=${encodeURIComponent(this.node || '')}`;
      this.data = await api('/api/flight?' + q);
      this.render(q);
      $('#btn-report').disabled = false;
      status('ready');
    } catch (e) {
      scroll.innerHTML = '';
      scroll.appendChild(el('div', 'banner alarm', e.message));
      status('review: ' + e.message);
    }
  },
  render(q) {
    const d = this.data, s = d.summary, scroll = $('#log-scroll');
    scroll.innerHTML = '';

    const strip = el('div', 'strip');
    const stat = (label, value) => { const n = el('div', 'stat');
      n.appendChild(el('span', null, label)); n.appendChild(el('b', null, value));
      strip.appendChild(n); };
    if (s.duration_s !== undefined) stat('log length', `${s.duration_s.toFixed(0)} s`);
    if (s.vps_s !== undefined) stat('on the VPS', `${s.vps_s.toFixed(0)} s`);
    if (s.alt_max) stat('max height', `${s.alt_max.toFixed(0)} m`);
    if (s.pos_p50 !== undefined && s.pos_p50 !== null)
      stat('position p50', `${s.pos_p50.toFixed(2)} m`);
    if (s.fixes !== undefined) stat('fix attempts', String(s.fixes));
    if (s.althold !== undefined) stat('ALT_HOLD drops', String(s.althold));
    scroll.appendChild(strip);

    const groups = [['alarm', 'What is wrong'], ['warn', 'What to watch'],
                    ['ok', 'What is working'], ['info', 'For the record']];
    for (const [level, title] of groups) {
      const fs = d.findings.filter(f => f.level === level);
      if (!fs.length) continue;
      scroll.appendChild(el('h2', 'sec', title));
      fs.forEach(f => scroll.appendChild(this.finding(f, level === 'alarm')));
    }

    if (d.panels && d.panels.length) {
      scroll.appendChild(el('h2', 'sec', 'Panels'));
      const grid = el('div', 'panels');
      d.panels.forEach(p => {
        const card = el('div', 'panel');
        card.appendChild(el('h5', null, p.title));
        const img = new Image();
        img.loading = 'lazy';
        img.src = `/api/panel?${q}&name=${p.name}`;
        img.onclick = () => lightbox(img.src);
        card.appendChild(img); grid.appendChild(card);
      });
      scroll.appendChild(grid);
    }

    if (d.track && (d.track.ours.length || d.track.truth.length)) {
      scroll.appendChild(el('h2', 'sec', 'The track on the satellite sheet'));
      const legend = el('p', 'hint');
      legend.innerHTML = '<span style="color:#4da3ff">&#9632;</span> what the node ' +
        'believed &nbsp; <span style="color:#ffb34d">&#9632;</span> the raw receiver ' +
        '&nbsp; <span style="color:#4ade80">&#9679;</span> map fixes &nbsp;&nbsp;' +
        '<span style="opacity:.7">drag to pan, ctrl + wheel to zoom</span>';
      scroll.appendChild(legend);
      const wrap = el('div', 'panel');
      wrap.style.height = '520px'; wrap.style.position = 'relative';
      wrap.id = 'track-wrap';
      const cv = el('canvas'); cv.id = 'track-canvas';
      wrap.appendChild(cv); scroll.appendChild(wrap);
      setTimeout(() => this.drawTrack(), 30);
    }

    if (d.frames && d.frames.length) {
      scroll.appendChild(el('h2', 'sec', `Camera frames (${d.frames.length} shown)`));
      const g = el('div', 'gallery');
      d.frames.forEach(f => {
        const img = new Image(); img.loading = 'lazy';
        img.src = '/api/frame?path=' + encodeURIComponent(f.path);
        img.title = f.name;
        img.onclick = () => lightbox(img.src);
        g.appendChild(img);
      });
      scroll.appendChild(g);
    }
  },
  finding(f, open) {
    const n = el('div', 'finding ' + f.level + (open ? ' open' : ''));
    const h = el('header');
    h.appendChild(el('span', 'tag', f.level.toUpperCase()));
    h.appendChild(el('h4', null, f.title));
    h.appendChild(el('span', 'head', f.headline));
    h.onclick = () => n.classList.toggle('open');
    n.appendChild(h);
    const body = el('div', 'body');
    body.appendChild(el('p', null, f.detail));
    if (f.basis) body.appendChild(el('p', 'basis', 'Threshold from: ' + f.basis));
    if (f.rows && f.rows.length > 1) {
      const t = el('table', 'data');
      const thead = t.createTHead().insertRow();
      f.rows[0].forEach(c => { const th = document.createElement('th');
        th.textContent = c; thead.appendChild(th); });
      const tb = t.createTBody();
      f.rows.slice(1, 40).forEach(r => { const tr = tb.insertRow();
        r.forEach(c => tr.insertCell().textContent = c); });
      body.appendChild(t);
      if (f.rows.length > 40)
        body.appendChild(el('p', 'basis', `${f.rows.length - 40} further rows in the Markdown report.`));
    }
    n.appendChild(body);
    return n;
  },
  async drawTrack() {
    const t = this.data.track;
    const pts = t.truth.length ? t.truth : t.ours;
    if (!pts.length) return;
    let lon = 0, lat = 0;
    for (const p of pts) { lon += p[0]; lat += p[1]; }
    lon /= pts.length; lat /= pts.length;
    try {
      const sheets = await api('/api/sheets');
      let found = null;
      for (const s of sheets.slice(0, 12)) {
        const d = await api('/api/sheet?path=' + encodeURIComponent(s.path));
        if (lon >= d.bounds[0] && lon <= d.bounds[2] &&
            lat >= d.bounds[1] && lat <= d.bounds[3]) { found = d; break; }
      }
      if (!found) { $('#track-wrap').appendChild(el('div', 'banner warn',
        'No sheet on this machine covers where this flight actually was - ' +
        'which also means the matcher had nothing to match against.')); return; }
      this.view = new MapView('#track-canvas', '#track-wrap');
      this.view.wheelNeedsModifier = true;
      this.view.setSheet(found);
      this.view.tracks = t;
      this.view.needFit = false;          // the track picks the view, not the fit
      this.view.scale = 0.35;
      this.view.centreOn(lon, lat);
    } catch (e) { status('track: ' + e.message); }
  },
};

function lightbox(src) {
  const lb = $('#lightbox');
  lb.querySelector('img').src = src;
  lb.classList.add('on');
}
$('#lightbox').onclick = () => $('#lightbox').classList.remove('on');
addEventListener('keydown', e => { if (e.key === 'Escape') $('#lightbox').classList.remove('on'); });

$('#tab-map').onclick = () => {
  $('#tab-map').classList.add('on'); $('#tab-log').classList.remove('on');
  $('#view-map').classList.add('on'); $('#view-log').classList.remove('on');
  Map1.view && Map1.view.resize();
};
$('#tab-log').onclick = () => {
  $('#tab-log').classList.add('on'); $('#tab-map').classList.remove('on');
  $('#view-log').classList.add('on'); $('#view-map').classList.remove('on');
};

Map1.init();
Log.init();
