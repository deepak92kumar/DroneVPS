"""Hold the map in RAM with its edges already computed.

Profiling the matcher showed the work was not where it looked. Reading a
window out of the GeoTIFF dominated everything - a +-300 m window cost about
1.4 s end to end against 0.7 s for the arithmetic, because GDAL decompresses
whole tiles for every read and the window moves each frame. And the map's edge
image was being recomputed on every single fix although the map never changes.

Both go away by paying once at startup: the whole map as one greyscale array
(4578x3928 is 18 MB, nothing), its gradient image beside it, and a window is
then a numpy slice. No GDAL, no CLAHE, no Sobel in the flight loop.
"""
from __future__ import annotations
import math

import numpy as np
import cv2

from map_db import GeoRef
from map_match_ncc import _edges


class MapCache:
    def __init__(self, path: str):
        from osgeo import gdal
        ds = gdal.Open(path)
        if ds is None:
            raise RuntimeError(f"cannot open {path}")
        arr = ds.ReadAsArray()
        if arr.ndim == 3:
            arr = cv2.cvtColor(np.dstack([arr[0], arr[1], arr[2]]), cv2.COLOR_RGB2GRAY)
        self.grey = np.ascontiguousarray(arr)
        self.edges = _edges(self.grey)
        gt = ds.GetGeoTransform()
        self.gt = gt
        self.geo = GeoRef.from_gdal(gt)
        # gsd is the NORTH-SOUTH figure and stays the primary one, because
        # everything already written uses it. gsd_x is the east-west figure,
        # which on a geographic raster is smaller by cos(latitude) - 16.5% at
        # 30.9 N. See north_up_query for what ignoring that cost.
        self.gsd = abs(gt[5]) * 111_320.0
        lat_c = gt[3] + (ds.RasterYSize / 2) * gt[5]
        self.gsd_x = abs(gt[1]) * 111_320.0 * math.cos(math.radians(lat_c))
        self.lat_centre = lat_c
        self.h, self.w = self.grey.shape

    def window(self, lon: float, lat: float, radius_m: float,
               kind: str = "edges"):
        """Window centred on a lon/lat, plus its own GeoRef.

        Returns (image, georef). The georef is what turns a pixel inside the
        window back into a position, so the caller never has to track offsets.

        kind picks which of the two cached arrays is sliced. "edges" is the
        gradient image the NCC matcher correlates; "grey" is the intensity the
        XFeat matcher needs, because gradient magnitude has none of the texture
        its descriptors were trained on. Both live in RAM already, so this is a
        slice either way and costs nothing to choose.
        """
        src = self.grey if kind == "grey" else self.edges
        c, r = self.geo.to_pixel(lon, lat)
        half = int(radius_m / self.gsd)
        x0 = int(np.clip(int(c) - half, 0, max(0, self.w - 2 * half)))
        y0 = int(np.clip(int(r) - half, 0, max(0, self.h - 2 * half)))
        x1 = min(self.w, x0 + 2 * half)
        y1 = min(self.h, y0 + 2 * half)
        g2 = list(self.gt)
        g2[0] = self.gt[0] + x0 * self.gt[1]
        g2[3] = self.gt[3] + y0 * self.gt[5]
        return src[y0:y1, x0:x1], GeoRef.from_gdal(g2)

    def covers(self, lon: float, lat: float, margin_px: int = 0) -> bool:
        c, r = self.geo.to_pixel(lon, lat)
        return (margin_px <= c < self.w - margin_px and
                margin_px <= r < self.h - margin_px)

    def summary(self) -> str:
        return (f"map {self.w}x{self.h} @ {self.gsd:.3f} m/px N-S, "
                f"{self.gsd_x:.3f} E-W "
                f"({self.w*self.gsd_x/1000:.2f} x {self.h*self.gsd/1000:.2f} km), "
                f"{self.grey.nbytes/1e6:.0f} MB grey + {self.edges.nbytes/1e6:.0f} MB edges")
