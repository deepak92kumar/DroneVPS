#!/usr/bin/env python3
"""Reading a GeoTIFF for the browser, and cutting a matcher-ready sheet from it.

Everything here exists to serve one consumer: `vps_vio/map_cache.py`, which is
what flies. That file makes four assumptions about the sheet it is handed, and
none of them is checked at run time - it reads the GeoTIFF's six-number
geotransform and trusts it:

    self.gsd   = abs(gt[5]) * 111_320.0
    self.gsd_x = abs(gt[1]) * 111_320.0 * cos(lat_centre)

So the sheet must be

  1. GEOGRAPHIC, degrees, EPSG:4326 - because of those 111_320 factors. A
     sheet in UTM metres would be read as if its pixel size were in degrees
     and the ground sample distance would come out 111 320 times too large.
  2. NORTH-UP - gt[2] and gt[4] exactly zero. GeoRef.from_gdal() keeps only
     gt[0], gt[1], gt[3], gt[5] and throws the two rotation terms away, so a
     rotated sheet is not detected, it is silently mis-georeferenced. This is
     the "yaw" the pilot asked about: the matcher rotates the camera frame to
     north-up by the aircraft's own heading, and if the SHEET is also rotated
     the two rotations do not cancel, they add.
  3. ORIGIN AT THE TOP-LEFT - gt[5] negative, so row increases southward.
     With a positive gt[5] every latitude the matcher reports is mirrored
     about the sheet's own origin.
  4. SMALL ENOUGH TO HOLD IN RAM as one greyscale array, because MapCache
     reads the whole sheet at startup on purpose (profiling showed windowed
     GDAL reads cost 1.4 s against 0.7 s of arithmetic).

A cut made here therefore always comes out 4326 and north-up, whatever the
source was: if the source is already both, the crop is a pixel-exact
`gdal.Translate` with no resampling at all, and if it is not, it is warped,
which is the only honest thing to do with a rotated or projected source.

The anisotropy is NOT corrected and must not be. A 4326 sheet has equal pixel
size in DEGREES, so in METRES it is 1/cos(latitude) anisotropic - 16.5% at
30.9 N. `map_query.north_up_query` already knows that and corrects the query
with `map_gsd_x_m`; a sheet "fixed" here by resampling to square metre pixels
would be corrected twice.
"""
from __future__ import annotations

import math
import os
import threading

import cv2
import numpy as np
from osgeo import gdal, osr

gdal.UseExceptions()

M_PER_DEG_LAT = 111_320.0          # the figure map_cache.py uses, kept identical
FLYING_GSD_M = 0.5092              # bhh.tif, the sheet every flight so far used
TILE = 256


def _wgs84():
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    # Without this GDAL 3 hands back (lat, lon) for 4326, because that is the
    # authority's own axis order. Every coordinate in this file is (lon, lat).
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return srs


class Sheet:
    """One open GeoTIFF, with the geometry the UI and the cutter both need.

    GDAL datasets are not thread safe and this is served from a threading HTTP
    server, so every read goes through the lock. The dataset is kept open
    because re-opening it per tile is the single most expensive thing a pan
    does on a 95 MB JPEG-compressed sheet.
    """

    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        self.lock = threading.Lock()
        self.ds = gdal.Open(self.path, gdal.GA_ReadOnly)
        if self.ds is None:
            raise RuntimeError(f"cannot open {path}")
        self.w = self.ds.RasterXSize
        self.h = self.ds.RasterYSize
        self.bands = self.ds.RasterCount
        self.gt = tuple(self.ds.GetGeoTransform())
        self.dtype = gdal.GetDataTypeName(self.ds.GetRasterBand(1).DataType)
        self.overviews = self.ds.GetRasterBand(1).GetOverviewCount()

        wkt = self.ds.GetProjection()
        self.srs = osr.SpatialReference()
        if wkt:
            self.srs.ImportFromWkt(wkt)
            self.srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        self.epsg = None
        if wkt:
            code = self.srs.GetAuthorityCode(None)
            self.epsg = int(code) if code and code.isdigit() else None
        self.geographic = bool(wkt) and bool(self.srs.IsGeographic())
        self.north_up = abs(self.gt[2]) < 1e-12 and abs(self.gt[4]) < 1e-12

        # Pixel -> lon/lat, for the cursor readout and for every lat/lon the
        # browser sends back. On a projected source this is a two-step
        # transform and the result is only north-up to the extent the
        # projection is; that is exactly why a cut from such a source is
        # warped rather than copied.
        self._to_ll = None
        if wkt and not self.geographic:
            self._to_ll = osr.CoordinateTransformation(self.srs, _wgs84())
            self._from_ll = osr.CoordinateTransformation(_wgs84(), self.srs)

    # ---------------------------------------------------------------- geometry
    def pixel_to_srs(self, col, row):
        gt = self.gt
        return (gt[0] + col * gt[1] + row * gt[2],
                gt[3] + col * gt[4] + row * gt[5])

    def srs_to_pixel(self, x, y):
        gt = self.gt
        det = gt[1] * gt[5] - gt[2] * gt[4]
        dx, dy = x - gt[0], y - gt[3]
        return ((dx * gt[5] - dy * gt[2]) / det,
                (dy * gt[1] - dx * gt[4]) / det)

    def pixel_to_lonlat(self, col, row):
        x, y = self.pixel_to_srs(col, row)
        if self._to_ll is None:
            return x, y
        lon, lat, _ = self._to_ll.TransformPoint(x, y)
        return lon, lat

    def lonlat_to_pixel(self, lon, lat):
        if self._to_ll is None:
            return self.srs_to_pixel(lon, lat)
        x, y, _ = self._from_ll.TransformPoint(lon, lat)
        return self.srs_to_pixel(x, y)

    def gsd_m(self):
        """(north-south, east-west) metres per pixel at the sheet's centre.

        Computed the way map_cache.py computes it, so what the UI reports is
        what the matcher will use - not a more correct figure from a different
        formula, which would hide a disagreement instead of showing it.
        """
        if self.geographic:
            _, lat_c = self.pixel_to_lonlat(self.w / 2, self.h / 2)
            ns = abs(self.gt[5]) * M_PER_DEG_LAT
            ew = abs(self.gt[1]) * M_PER_DEG_LAT * math.cos(math.radians(lat_c))
            return ns, ew
        # Projected: the pixel size is already in the CRS's own linear unit.
        unit = self.srs.GetLinearUnits() if self.srs.GetName() else 1.0
        return abs(self.gt[5]) * unit, abs(self.gt[1]) * unit

    def corners_lonlat(self):
        pts = [(0, 0), (self.w, 0), (self.w, self.h), (0, self.h)]
        return [self.pixel_to_lonlat(c, r) for c, r in pts]

    def bounds_lonlat(self):
        lons = [p[0] for p in self.corners_lonlat()]
        lats = [p[1] for p in self.corners_lonlat()]
        return min(lons), min(lats), max(lons), max(lats)

    # ------------------------------------------------------------------ tiles
    def read(self, col, row, cols, rows, out_w, out_h):
        """A decimated RGB window, clipped to the sheet and padded outside it.

        Padding rather than refusing: the browser asks for whole tiles and the
        last tile of a row is half off the edge. Returning None for it would
        leave a hole in the display; returning black would draw a black band
        that looks like missing imagery. The pad is transparent instead, which
        the canvas composites away.
        """
        sx0, sy0 = max(0, int(col)), max(0, int(row))
        sx1, sy1 = min(self.w, int(col + cols)), min(self.h, int(row + rows))
        if sx1 <= sx0 or sy1 <= sy0:
            return None
        bw = max(1, int(round(out_w * (sx1 - sx0) / float(cols))))
        bh = max(1, int(round(out_h * (sy1 - sy0) / float(rows))))
        with self.lock:
            arr = self.ds.ReadAsArray(sx0, sy0, sx1 - sx0, sy1 - sy0,
                                      buf_xsize=bw, buf_ysize=bh)
        if arr is None:
            return None
        arr = np.asarray(arr)
        if arr.ndim == 2:
            arr = np.dstack([arr] * 3)
        else:
            arr = np.dstack([arr[min(i, arr.shape[0] - 1)] for i in range(3)])
        if arr.dtype != np.uint8:
            # 16-bit imagery happens. Stretch 2-98% so something is visible at
            # all; this is display only and never touches an exported cut.
            a = arr.astype(np.float32)
            lo, hi = np.percentile(a, 2), np.percentile(a, 98)
            arr = np.clip((a - lo) * 255.0 / max(1e-6, hi - lo), 0, 255).astype(np.uint8)
        rgba = np.dstack([arr[:, :, 2], arr[:, :, 1], arr[:, :, 0],
                          np.full(arr.shape[:2], 255, np.uint8)])    # BGRA for cv2
        if (bw, bh) != (out_w, out_h):
            canvas = np.zeros((out_h, out_w, 4), np.uint8)
            ox = int(round((sx0 - col) * out_w / float(cols)))
            oy = int(round((sy0 - row) * out_h / float(rows)))
            ox, oy = max(0, ox), max(0, oy)
            bw, bh = min(bw, out_w - ox), min(bh, out_h - oy)
            if bw <= 0 or bh <= 0:
                return None
            canvas[oy:oy + bh, ox:ox + bw] = rgba[:bh, :bw]
            rgba = canvas
        return rgba

    def tile_png(self, z: int, tx: int, ty: int):
        """One 256x256 tile at level z: 2**z source pixels per tile pixel."""
        step = 1 << max(0, int(z))
        span = TILE * step
        rgba = self.read(tx * span, ty * span, span, span, TILE, TILE)
        if rgba is None:
            return None
        ok, buf = cv2.imencode(".png", rgba)
        return buf.tobytes() if ok else None

    def describe(self):
        ns, ew = self.gsd_m()
        lon0, lat0, lon1, lat1 = self.bounds_lonlat()
        return dict(
            path=self.path, name=os.path.basename(self.path),
            width=self.w, height=self.h, bands=self.bands, dtype=self.dtype,
            overviews=self.overviews,
            crs=(self.srs.GetName() if self.srs.GetName() else "none"),
            epsg=self.epsg, geographic=self.geographic, north_up=self.north_up,
            geotransform=list(self.gt),
            gsd_ns_m=ns, gsd_ew_m=ew,
            ground_w_km=self.w * ew / 1000.0, ground_h_km=self.h * ns / 1000.0,
            bounds=[lon0, lat0, lon1, lat1],
            corners=[list(p) for p in self.corners_lonlat()],
            grey_mb=self.w * self.h / 1e6,
            file_mb=os.path.getsize(self.path) / 1e6,
            max_level=int(max(0, math.ceil(math.log2(max(self.w, self.h) / float(TILE))))),
            issues=self.issues(),
        )

    def issues(self):
        """What is wrong with this sheet AS A SOURCE, in the UI's own words."""
        out = []
        if not self.geographic or self.epsg != 4326:
            out.append(dict(level="warn", text=(
                f"Source CRS is {self.srs.GetName() or 'unset'}"
                f"{'' if self.epsg is None else f' (EPSG:{self.epsg})'}, not "
                f"geographic WGS 84. The cut will be reprojected to EPSG:4326.")))
        if not self.north_up:
            out.append(dict(level="warn", text=(
                "Source is rotated (geotransform carries non-zero rotation "
                "terms). The cut will be rectified north-up.")))
        if self.gt[5] > 0:
            out.append(dict(level="warn", text=(
                "Source rows run south-to-north. The cut will be flipped so "
                "the origin is the top-left corner.")))
        if self.dtype != "Byte":
            out.append(dict(level="warn", text=(
                f"Source samples are {self.dtype}, not 8-bit. The cut will be "
                f"written as 8-bit; check the preview before flying it.")))
        return out


# --------------------------------------------------------------------------- #
#  cutting
# --------------------------------------------------------------------------- #
def cut(src: Sheet, *, out_path: str, box_px=None, centre_lonlat=None,
        size_m=None, out_gsd_m: float | None = None, grey: bool = False,
        resample: str = "cubic", overwrite: bool = False):
    """Write one matcher-ready sheet and return a report on what was written.

    The crop can be given either as a pixel box of the source (what dragging
    on the canvas produces) or as a centre and a ground size in metres (what
    "cut 1.5 km around the take-off point" produces). Both end up as a
    lon/lat rectangle, because that is the only frame in which both the
    pixel-exact and the warped path can be expressed.
    """
    if os.path.exists(out_path) and not overwrite:
        raise FileExistsError(out_path)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    if box_px is not None:
        x0, y0, x1, y1 = box_px
        x0, x1 = sorted((float(x0), float(x1)))
        y0, y1 = sorted((float(y0), float(y1)))
        x0, y0 = max(0.0, x0), max(0.0, y0)
        x1, y1 = min(float(src.w), x1), min(float(src.h), y1)
        if x1 - x0 < 2 or y1 - y0 < 2:
            raise ValueError("the crop box is empty")
        # All four corners, not two: on a rotated or projected source the box
        # is a quadrilateral in lon/lat and its bounding box is wider than the
        # two opposite corners suggest.
        pts = [src.pixel_to_lonlat(c, r)
               for c, r in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
        lon0 = min(p[0] for p in pts); lon1 = max(p[0] for p in pts)
        lat0 = min(p[1] for p in pts); lat1 = max(p[1] for p in pts)
        pixel_box = (x0, y0, x1, y1)
    elif centre_lonlat is not None and size_m is not None:
        lon_c, lat_c = float(centre_lonlat[0]), float(centre_lonlat[1])
        wm, hm = float(size_m[0]), float(size_m[1])
        dlat = (hm / 2.0) / M_PER_DEG_LAT
        dlon = (wm / 2.0) / (M_PER_DEG_LAT * math.cos(math.radians(lat_c)))
        lon0, lon1 = lon_c - dlon, lon_c + dlon
        lat0, lat1 = lat_c - dlat, lat_c + dlat
        pixel_box = None
    else:
        raise ValueError("give either box_px or centre_lonlat + size_m")

    ns, ew = src.gsd_m()
    # Can the crop be a copy rather than a warp? Only if the source is already
    # everything the matcher needs and no new resolution was asked for. A
    # centre-and-size request qualifies too: its lon/lat rectangle is turned
    # back into whole source pixels below, so "1.2 km around the take-off
    # point" is copied, not resampled.
    exact = (src.geographic and src.epsg == 4326 and src.north_up
             and src.gt[5] < 0 and out_gsd_m is None)
    if exact and pixel_box is None:
        c0, r0 = src.lonlat_to_pixel(lon0, lat1)        # north-west corner
        c1, r1 = src.lonlat_to_pixel(lon1, lat0)        # south-east corner
        pixel_box = (max(0.0, c0), max(0.0, r0),
                     min(float(src.w), c1), min(float(src.h), r1))
        if pixel_box[2] - pixel_box[0] < 2 or pixel_box[3] - pixel_box[1] < 2:
            raise ValueError("that centre and size fall outside the source sheet")

    opts = dict(format="GTiff",
                creationOptions=["TILED=YES", "COMPRESS=DEFLATE", "PREDICTOR=2",
                                 "BIGTIFF=IF_SAFER"])
    if grey:
        opts["bandList"] = [1]            # band 1 after the conversion below

    if exact:
        # The honest path: whole source pixels, copied. No resampling, so the
        # cut's geotransform is the source's own origin plus an integer pixel
        # offset and every pixel is bit-identical. A matcher tuned on the big
        # sheet sees exactly the same imagery in the cut.
        sx0, sy0 = int(math.floor(pixel_box[0])), int(math.floor(pixel_box[1]))
        sx1, sy1 = int(math.ceil(pixel_box[2])), int(math.ceil(pixel_box[3]))
        ds = gdal.Translate(out_path, src.ds,
                            srcWin=[sx0, sy0, sx1 - sx0, sy1 - sy0],
                            outputType=gdal.GDT_Byte,
                            bandList=[1] if grey else None,
                            **{k: v for k, v in opts.items() if k != "bandList"})
        how = (f"pixel-exact copy of source pixels "
               f"[{sx0}:{sx1}, {sy0}:{sy1}] - no resampling")
    else:
        # Warped: the source is projected, rotated, upside down, or a
        # different resolution was asked for. One warp does all of it, and the
        # result is 4326 north-up by construction.
        tr = out_gsd_m if out_gsd_m else ns
        y_res = tr / M_PER_DEG_LAT
        lat_mid = (lat0 + lat1) / 2.0
        if out_gsd_m:
            # A square metre pixel was asked for, so east-west degrees must be
            # smaller by cos(lat) to match. This is the ONE place a square
            # metre pixel is produced, and it is opt-in: map_query corrects
            # for the usual anisotropic sheet itself.
            x_res = tr / (M_PER_DEG_LAT * math.cos(math.radians(lat_mid)))
        else:
            x_res = abs(src.gt[1]) if src.geographic else y_res
        ds = gdal.Warp(out_path, src.ds, dstSRS="EPSG:4326",
                       outputBounds=[lon0, lat0, lon1, lat1],
                       xRes=x_res, yRes=y_res,
                       resampleAlg=resample, outputType=gdal.GDT_Byte,
                       multithread=True, **opts)
        why = ("the source is rotated" if not src.north_up else
               "the source is not EPSG:4326" if not (src.geographic and src.epsg == 4326)
               else "the source rows run south-to-north" if src.gt[5] > 0
               else "a different resolution was asked for")
        how = (f"warped to EPSG:4326 north-up at {tr:.4f} m/px ({resample}) "
               f"because {why}")
    if ds is None:
        raise RuntimeError("GDAL wrote nothing")
    ds.FlushCache()
    del ds
    rep = verify(out_path)
    rep["how"] = how
    rep["requested"] = dict(bounds=[lon0, lat0, lon1, lat1],
                            box_px=list(pixel_box) if pixel_box else None)
    return rep


def verify(path: str, takeoff=None, flying_gsd=FLYING_GSD_M):
    """Re-open a written sheet and check what map_cache.py will assume of it.

    This runs on the file as it lands on the disk, not on the parameters it
    was asked for, because the point is to catch the case where GDAL did
    something other than what was asked.
    """
    checks = []

    def ck(ok, name, detail, hard=True):
        checks.append(dict(ok=bool(ok), hard=hard, name=name, detail=detail))

    s = Sheet(path)
    ns, ew = s.gsd_m()
    ck(s.epsg == 4326 and s.geographic, "CRS is geographic WGS 84 (EPSG:4326)",
       f"{s.srs.GetName() or 'unset'}"
       f"{'' if s.epsg is None else f' / EPSG:{s.epsg}'}"
       + ("" if s.epsg == 4326 else
          " - map_cache multiplies pixel size by 111 320 m/deg, so a "
          "projected sheet reads 111 320x too coarse"))
    ck(s.north_up, "north-up: no rotation in the geotransform",
       f"gt[2]={s.gt[2]:.3e}, gt[4]={s.gt[4]:.3e}"
       + ("" if s.north_up else " - GeoRef.from_gdal drops these two terms, so "
                                "the error would not be detected in flight"))
    ck(s.gt[5] < 0, "origin at the top-left (row increases southward)",
       f"gt[5]={s.gt[5]:.3e} deg/px")
    ck(s.gt[1] > 0, "column increases eastward", f"gt[1]={s.gt[1]:.3e} deg/px")
    ck(s.bands in (1, 3), "1 or 3 bands, which MapCache can grey-scale",
       f"{s.bands} bands, {s.dtype}")
    ck(s.dtype == "Byte", "8-bit samples", s.dtype)
    grey_mb = s.w * s.h / 1e6
    ck(grey_mb < 600, "fits in RAM as one greyscale array",
       f"{s.w}x{s.h} = {grey_mb:.0f} MB grey (MapCache loads the whole sheet)",
       hard=False)
    ratio = ns / flying_gsd if flying_gsd else 1.0
    ck(0.5 < ratio < 2.0, "resolution close to the sheet flown so far",
       f"{ns:.4f} m/px N-S against {flying_gsd:.4f} - "
       f"{'x%.2f' % ratio}", hard=False)
    aniso = ns / ew if ew else 1.0
    checks.append(dict(ok=True, hard=False,
                       name="metre anisotropy, which map_query corrects",
                       detail=f"N-S {ns:.4f} / E-W {ew:.4f} m/px = x{aniso:.4f} "
                              f"(1/cos(lat) at this latitude)"))
    if takeoff:
        lon_t, lat_t = float(takeoff[0]), float(takeoff[1])
        c, r = s.lonlat_to_pixel(lon_t, lat_t)
        margin = 150.0 / ns
        inside = margin <= c <= s.w - margin and margin <= r <= s.h - margin
        edge_m = min(c, r, s.w - c, s.h - r) * ns
        ck(inside, "the take-off point is on the sheet with 150 m to spare",
           f"{lat_t:.6f}, {lon_t:.6f} lands at pixel {c:.0f},{r:.0f}; "
           f"{edge_m:.0f} m from the nearest edge"
           + ("" if inside else " - OFF-SHEET means no map fixes at all"))

    d = s.describe()
    d["checks"] = checks
    d["ok"] = all(c["ok"] for c in checks if c["hard"])
    return d


def build_overviews(path: str, levels=(2, 4, 8, 16, 32, 64)):
    """An external .ovr pyramid, so panning a big sheet stops decimating live.

    Read-only (`-ro` in gdaladdo terms): the pyramid goes in `<path>.ovr` and
    the source file is not touched. A sheet someone else gave us is not ours
    to rewrite.
    """
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"cannot open {path}")
    gdal.SetConfigOption("COMPRESS_OVERVIEW", "DEFLATE")
    ds.BuildOverviews("AVERAGE", list(levels))
    n = ds.GetRasterBand(1).GetOverviewCount()
    del ds
    return n
