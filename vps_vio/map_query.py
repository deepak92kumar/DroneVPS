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
                   keep: float = 0.95, track_xy=None, shape: str = "square"):
    """Rescale to the map's GSD and rotate north-up, then keep the centre.

    With `track_xy` = (x, y), a pixel of the INPUT frame, this also returns
    where that pixel ended up in the query it produces: (query, (x, y)).
    Without it, just the query, as before.

    That exists for one caller and one reason. Whoever matches this query gets
    a translation, and turning that into a position means naming WHICH pixel of
    the camera frame the position belongs to. Until 20 Oct everyone took the
    geometric centre, because that is what the crop below is centred on - but
    the point under the camera is the one the optical axis pierces, which is
    the principal point (cx, cy), and this frame is the 130 px leg-crop whose
    cx sits 80 px off its middle. 80 px is 11 m of ground at 110 m. The
    transform has three stages that each move a pixel, so the only safe place
    to follow one through them is here, inside the function that applies them:
    anything that recomputed the chain outside would drift from it the next
    time one of the stages changed.

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
    # cv2's pixel centres: input u maps to (u + 0.5) * scale - 0.5. The half
    # pixel is 0.07 px here against the 80 px this is for, but writing the
    # convention down costs nothing and guessing it wrong later costs a day.
    pt = (None if track_xy is None
          else np.array([(float(track_xy[0]) + 0.5) * scale - 0.5,
                         (float(track_xy[1]) + 0.5) * scale - 0.5], float))
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -yaw_deg, 1.0)
    q = cv2.warpAffine(q, M, (w, h))
    if pt is not None:
        # The SAME M the image was warped by, so the point cannot disagree
        # with the pixels. Rotation is about the geometric centre; the
        # principal point is not at it, so it genuinely moves here.
        pt = M @ np.array([pt[0], pt[1], 1.0])

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
        kx = map_gsd_m / float(map_gsd_x_m)
        q = cv2.resize(q, None, fx=kx, fy=1.0, interpolation=cv2.INTER_LINEAR)
        h, w = q.shape
        if pt is not None:
            # AFTER the rotation, exactly as the image is - the anisotropy
            # lives in the map's axes, not the camera's. Applying it to the
            # point before the rotation would reintroduce an error of the same
            # size as the one this function already documents.
            pt = np.array([(pt[0] + 0.5) * kx - 0.5, pt[1]], float)

    # shape "square" is what flies: a centred square of 0.95 x the SHORT side.
    # On a 1280x800 frame that is 760x760, so the long axis contributes 520 px
    # that are discarded - 44% of the frame never reaches the matcher. "full"
    # keeps the whole rotated frame instead, black corners and all.
    #
    # MEASURED 22 Oct, and "full" is better. 901 recorded frames of vps_00076
    # at 60 m and above, each matched both ways against the same map window
    # and scored against the GPS truth in that run's own CSV:
    #
    #                accepted    p50 err   p90 err    mean   inliers   ratio
    #     square    272/901      13.82 m   21.16 m   14.62      17.8   0.131
    #     full      303/901      12.51 m   21.06 m   13.90      23.0   0.119
    #
    # On the 168 frames BOTH accepted, "full" is closer to truth on 103 -
    # 61%, which on a sign test is p about 0.003, so it is not sampling noise.
    # A first pass over 76 frames said the opposite (square p50 12.61 against
    # 14.01) and was simply too small: 23 and 24 accepted frames.
    #
    # What it costs: the black corners dilute the inlier RATIO, 0.131 -> 0.119,
    # which sits against min_score 0.07. The extra area more than pays for it -
    # 29% more inliers, and 31 more frames accepted, not fewer.
    #
    # Default is still "square" because this has not been flown.
    if shape == "full":
        x0 = y0 = 0
        out = q
    elif shape == "square":
        s = int(min(h, w) * keep) // 2
        x0, y0 = w // 2 - s, h // 2 - s
        out = q[y0:y0 + 2 * s, x0:x0 + 2 * s]
    else:
        raise ValueError(f"shape {shape!r}: expected 'square' or 'full'")
    if pt is None:
        return out
    # Into the cropped query's own coordinates, which is the frame the matcher
    # returns a translation in.
    return out, (float(pt[0] - x0), float(pt[1] - y0))
