"""Georeferenced map database for satellite-image matching.

Part 3: gives the aircraft an ABSOLUTE position by matching what its downward
camera sees against pre-loaded satellite imagery. This is what VIO cannot do —
VIO drifts with distance, and above ~25 m the stereo baseline is useless, so at
100-150 m the only absolute reference is the map.

Scale is the thing people get wrong. The drone's ground sample distance is
    gsd_drone = altitude / focal_px
e.g. 120 m with focal 1943 px -> 6.2 cm/px, while this map is ~50 cm/px. The
query image must be resampled to the map's GSD before matching, otherwise the
descriptors describe completely different spatial frequencies.

Uses OpenCV only (AKAZE/SIFT) — no PyTorch, nothing to install. If accuracy is
not enough on real flight imagery, the next step is XFeat exported to ONNX and
run through the TensorRT that JetPack already ships.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cv2


@dataclass
class GeoRef:
    """Affine map from pixel (col,row) to lon/lat, from the GeoTIFF geotransform."""
    origin_lon: float
    origin_lat: float
    dlon_dcol: float
    dlat_drow: float

    @classmethod
    def from_gdal(cls, gt) -> "GeoRef":
        # gt = (originX, pixelWidth, 0, originY, 0, pixelHeight)
        return cls(gt[0], gt[3], gt[1], gt[5])

    def to_lonlat(self, col, row):
        return (self.origin_lon + np.asarray(col) * self.dlon_dcol,
                self.origin_lat + np.asarray(row) * self.dlat_drow)

    def to_pixel(self, lon, lat):
        return ((np.asarray(lon) - self.origin_lon) / self.dlon_dcol,
                (np.asarray(lat) - self.origin_lat) / self.dlat_drow)

    def gsd_m(self, lat_deg: float) -> float:
        """Metres per pixel at this latitude (geographic CRS)."""
        m_per_deg_lat = 111_320.0
        return abs(self.dlat_drow) * m_per_deg_lat


class MapDatabase:
    """Satellite tile + feature store for one area."""

    def __init__(self, detector: str = "AKAZE", max_features: int = 4000):
        self.detector_name = detector
        self.max_features = max_features
        self.det = self._make_detector(detector, max_features)
        self.image: np.ndarray | None = None     # greyscale map at native GSD
        self.geo: GeoRef | None = None
        self.gsd: float | None = None            # metres per pixel
        self.kp = None
        self.desc = None

    @staticmethod
    def _make_detector(name: str, n: int):
        name = name.upper()
        if name == "SIFT":
            return cv2.SIFT_create(nfeatures=n)
        if name == "ORB":
            return cv2.ORB_create(nfeatures=n)
        return cv2.AKAZE_create()            # default: fast, rotation+scale robust

    def load_geotiff(self, path: str, downscale: int = 1):
        from osgeo import gdal
        ds = gdal.Open(path)
        if ds is None:
            raise RuntimeError(f"cannot open {path}")
        W, H = ds.RasterXSize, ds.RasterYSize
        bw, bh = W // downscale, H // downscale
        arr = ds.ReadAsArray(0, 0, W, H, buf_xsize=bw, buf_ysize=bh)
        if arr.ndim == 3:                     # RGB -> grey
            rgb = np.dstack([arr[0], arr[1], arr[2]])
            img = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        else:
            img = arr
        gt = list(ds.GetGeoTransform())
        gt[1] *= downscale
        gt[5] *= downscale
        self.image = img
        self.geo = GeoRef.from_gdal(gt)
        self.gsd = self.geo.gsd_m(self.geo.origin_lat)
        return self

    def build(self):
        """Extract features over the whole map once."""
        if self.image is None:
            raise RuntimeError("load a map first")
        # CLAHE: satellite and drone imagery differ a lot in exposure/haze;
        # equalising local contrast makes descriptors far more comparable.
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        eq = clahe.apply(self.image)
        self.kp, self.desc = self.det.detectAndCompute(eq, None)
        return self

    def summary(self) -> str:
        h, w = self.image.shape[:2]
        return (f"map {w}x{h} px @ {self.gsd:.3f} m/px "
                f"({w*self.gsd/1000:.2f} x {h*self.gsd/1000:.2f} km), "
                f"{self.detector_name} features: {0 if self.kp is None else len(self.kp)}")


def prepare_query(img_grey: np.ndarray, altitude_m: float, focal_px: float,
                  map_gsd_m: float) -> tuple[np.ndarray, float]:
    """Resample a drone image to the map's ground sample distance.

    Returns the resampled image and the scale factor applied.
    """
    gsd_drone = altitude_m / float(focal_px)          # metres per pixel on the ground
    scale = gsd_drone / map_gsd_m                     # <1 means drone is finer than map
    if scale <= 0:
        raise ValueError("bad scale")
    new_w = max(16, int(round(img_grey.shape[1] * scale)))
    new_h = max(16, int(round(img_grey.shape[0] * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(img_grey, (new_w, new_h), interpolation=interp), scale
