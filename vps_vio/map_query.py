#!/usr/bin/env python3
"""Turning a camera frame into something the map can be searched with.

This is the geometry both matchers needed, kept when the NCC matcher was
removed on 20 Oct - XFeat had replaced it as the flying path and the pilot
asked for the dead one to go. It is not NCC-specific and never was: rescaling a
frame to the map's ground sample distance and rotating it north-up is what any
matcher has to do first.

The east-west correction in here is the project's own 16 m finding. A
geographic GeoTIFF has equal pixel size in DEGREES, so in METRES it is
anisotropic by 1/cos(latitude) - at 30.89 that is 1.1653, 16.5% - and the
12 Oct report blamed exactly "16 m that has never been explained" on it.
"""
from __future__ import annotations

import cv2
import numpy as np


def north_up_query(img_grey: np.ndarray, *, altitude_m: float, focal_px: float,
                   yaw_deg: float, map_gsd_m: float,
                   map_gsd_x_m: float | None = None,
                   keep: float = 0.95) -> np.ndarray:
    """Rescale to the map's GSD and rotate north-up, then keep the centre.

    Rotation leaves black corners. Cropping them away entirely (keep 0.5-0.62)
    measured far worse than tolerating them - context is what makes the peak
    unambiguous, and 12 frames went from 2/12 to 9/12 within 30 m as keep rose
    from 0.50 to 0.95. Rotating the map instead, to give a template with no
    holes at all, was worse again: it blurs the map and widens the search.
    """
    # A non-positive altitude makes the scale zero or negative and cv2.resize
    # dies on it: "inv_scale_x > 0". The barometer reads below zero on the
    # ground all the time, so this is reachable from any recording that starts
    # before takeoff - and one bad row must not take a whole offline analysis
    # down with it. Refuse the frame instead.
    if not (altitude_m > 0.0) or not (map_gsd_m > 0.0) or not (focal_px > 0.0):
        raise ValueError(f"altitude {altitude_m} / gsd {map_gsd_m} / focal "
                         f"{focal_px}: cannot scale a query from these")
    scale = (altitude_m / float(focal_px)) / map_gsd_m
    q = cv2.resize(img_grey, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = q.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -yaw_deg, 1.0)
    q = cv2.warpAffine(q, M, (w, h))

    # The map's pixels are not square IN METRES, and that is a 16.5% error here.
    #
    # A geographic (EPSG:4326) GeoTIFF has equal pixel size in DEGREES, but a
    # degree of longitude is cos(latitude) of a degree of latitude - 0.858 at
    # 30.9 N. So this map is 0.5092 m/px north-south and 0.4369 m/px east-west,
    # while MapCache.gsd carries only the north-south figure and the scale
    # above applied it to both axes.
    #
    # 1/cos(30.89) = 1.1653, which is the 16.5% the 12 Oct report found behind
    # XFEAT_ANISO and "16 m that has never been explained". Over a 178 m
    # footprint at 110 m, a 16.5% east-west scale error is about 15 m.
    #
    # The correction has to come AFTER the rotation, not before: the anisotropy
    # lives in the map's axes, not the camera's, so the frame has to be north-up
    # before x means east. With an isotropic map the order does not matter and
    # this is skipped entirely, which is why it went unnoticed.
    if map_gsd_x_m and abs(map_gsd_x_m - map_gsd_m) > 1e-9:
        # Map pixels are SMALLER in metres east-west, so the same ground
        # distance spans MORE of them: stretch x.
        q = cv2.resize(q, None, fx=map_gsd_m / float(map_gsd_x_m), fy=1.0,
                       interpolation=cv2.INTER_LINEAR)
        h, w = q.shape

    s = int(min(h, w) * keep) // 2
    return q[h // 2 - s:h // 2 + s, w // 2 - s:w // 2 + s]
