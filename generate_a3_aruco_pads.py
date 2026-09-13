#!/usr/bin/env python3
"""
Generate Precision ArUco Landing Pad for A3 Paper (300 DPI)
Configured for OpenCV DICT_4X4_50, Marker ID #0.
Matching flight_daemon.py (marker_size_m = 0.20 / 20 cm).
"""

import os
import cv2
import cv2.aruco as aruco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = "/home/abc/Desktop/DRONE"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# A3 Dimensions at 300 DPI: 297 mm x 420 mm
DPI = 300
MM_TO_PX = DPI / 25.4 # ~11.811 pixels per mm

PAGE_W = int(round(297.0 * MM_TO_PX)) # 3508 px
PAGE_H = int(round(420.0 * MM_TO_PX)) # 4961 px

print(f"Generating A3 sheets at {DPI} DPI: {PAGE_W} x {PAGE_H} px")

# ArUco Setup
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# -------------------------------------------------------------
# 1. Pure High-Resolution Marker ID #0 (2400 x 2400)
# -------------------------------------------------------------
raw_marker = aruco.generateImageMarker(dictionary, 0, 2000, borderBits=1)
pure_marker = np.full((2400, 2400), 255, dtype=np.uint8)
pure_marker[200:2200, 200:2200] = raw_marker

pure_path = os.path.join(OUTPUT_DIR, "aruco_marker0_pure.png")
cv2.imwrite(pure_path, pure_marker)
print(f"[OK] Pure marker saved: {pure_path}")

# -------------------------------------------------------------
# 2. Function to build Complete Landing Pad on A3
# -------------------------------------------------------------
def build_a3_pad(marker_size_mm, output_base_name, title_sub):
    canvas = np.full((PAGE_H, PAGE_W, 3), 255, dtype=np.uint8)
    cx, cy = PAGE_W // 2, PAGE_H // 2

    # Outer border on A3 page (15 mm margin)
    margin_px = int(round(15.0 * MM_TO_PX))
    cv2.rectangle(canvas, (margin_px, margin_px), (PAGE_W - margin_px, PAGE_H - margin_px), (30, 30, 30), 4)

    # Sub-border
    sub_margin = margin_px + int(round(4.0 * MM_TO_PX))
    cv2.rectangle(canvas, (sub_margin, sub_margin), (PAGE_W - sub_margin, PAGE_H - sub_margin), (180, 180, 180), 2)

    # Target concentric circular rings around center
    radius_rings = [
        int(round((marker_size_mm/2 + 25) * MM_TO_PX)),
        int(round((marker_size_mm/2 + 50) * MM_TO_PX)),
    ]
    for r in radius_rings:
        if r < (PAGE_W // 2 - margin_px - 20):
            cv2.circle(canvas, (cx, cy), r, (210, 210, 210), 3, lineType=cv2.LINE_AA)

    # ArUco Marker exact metric sizing
    marker_px = int(round(marker_size_mm * MM_TO_PX))
    raw_m = aruco.generateImageMarker(dictionary, 0, marker_px, borderBits=1)
    raw_m_bgr = cv2.cvtColor(raw_m, cv2.COLOR_GRAY2BGR)

    x1 = cx - marker_px // 2
    y1 = cy - marker_px // 2
    x2 = x1 + marker_px
    y2 = y1 + marker_px

    # Place marker
    canvas[y1:y2, x1:x2] = raw_m_bgr

    # Black alignment frame around marker
    cv2.rectangle(canvas, (x1 - 6, y1 - 6), (x2 + 6, y2 + 6), (0, 0, 0), 4)

    # Center crosshairs (outside marker)
    cross_len = int(round(25.0 * MM_TO_PX))
    cv2.line(canvas, (cx, y1 - 10), (cx, y1 - 10 - cross_len), (0, 0, 0), 4)
    cv2.line(canvas, (cx, y2 + 10), (cx, y2 + 10 + cross_len), (0, 0, 0), 4)
    cv2.line(canvas, (x1 - 10, cy), (x1 - 10 - cross_len, cy), (0, 0, 0), 4)
    cv2.line(canvas, (x2 + 10, cy), (x2 + 10 + cross_len, cy), (0, 0, 0), 4)

    # Corner L-brackets
    corner_len = int(round(18.0 * MM_TO_PX))
    c_offset = 20
    cv2.line(canvas, (x1 - c_offset, y1 - c_offset), (x1 - c_offset + corner_len, y1 - c_offset), (0, 0, 200), 5)
    cv2.line(canvas, (x1 - c_offset, y1 - c_offset), (x1 - c_offset, y1 - c_offset + corner_len), (0, 0, 200), 5)
    cv2.line(canvas, (x2 + c_offset, y1 - c_offset), (x2 + c_offset - corner_len, y1 - c_offset), (0, 0, 200), 5)
    cv2.line(canvas, (x2 + c_offset, y1 - c_offset), (x2 + c_offset, y1 - c_offset + corner_len), (0, 0, 200), 5)
    cv2.line(canvas, (x1 - c_offset, y2 + c_offset), (x1 - c_offset + corner_len, y2 + c_offset), (0, 0, 200), 5)
    cv2.line(canvas, (x1 - c_offset, y2 + c_offset), (x1 - c_offset, y2 + c_offset - corner_len), (0, 0, 200), 5)
    cv2.line(canvas, (x2 + c_offset, y2 + c_offset), (x2 + c_offset - corner_len, y2 + c_offset), (0, 0, 200), 5)
    cv2.line(canvas, (x2 + c_offset, y2 + c_offset), (x2 + c_offset, y2 + c_offset + corner_len), (0, 0, 200), 5)

    # Typography with PIL
    pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 60)
        font_mid = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
        font_sub = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 34)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 26)
    except:
        font_large = font_mid = font_sub = font_small = ImageFont.load_default()

    # Header Titles
    draw.text((cx, margin_px + 70), "AUTONOMOUS PRECISION LANDING PAD", fill=(15, 23, 42), font=font_large, anchor="mm")
    draw.text((cx, margin_px + 140), f"STANDARD TARGET: DICT_4X4_50 | MARKER ID: #0 | {title_sub}", fill=(51, 65, 85), font=font_sub, anchor="mm")

    # Heading Arrow: NORTH / FORWARD
    arrow_y = y1 - int(round(45.0 * MM_TO_PX))
    draw.text((cx, arrow_y - 30), "^ NORTH / FORWARD ^", fill=(185, 28, 28), font=font_mid, anchor="mm")
    draw.polygon([(cx, arrow_y + 10), (cx - 30, arrow_y + 60), (cx + 30, arrow_y + 60)], fill=(185, 28, 28))

    # Metric Scale Guide
    guide_y = PAGE_H - margin_px - 220
    draw.text((cx, guide_y - 45), f"PHYSICAL RULER VERIFICATION (Marker Width = Exactly {int(marker_size_mm/10)} cm)", fill=(30, 41, 59), font=font_mid, anchor="mm")

    ruler_w_mm = min(200.0, marker_size_mm)
    ruler_px = int(round(ruler_w_mm * MM_TO_PX))
    rx1 = cx - ruler_px // 2
    rx2 = cx + ruler_px // 2
    draw.rectangle([rx1, guide_y, rx2, guide_y + 25], fill=(30, 41, 59))

    cm_steps = int(ruler_w_mm / 10)
    for cm_idx in range(cm_steps + 1):
        tick_x = rx1 + int(round(cm_idx * 10.0 * MM_TO_PX))
        draw.line([(tick_x, guide_y), (tick_x, guide_y - 25)], fill=(30, 41, 59), width=3)
        draw.text((tick_x, guide_y - 35), f"{cm_idx}", fill=(30, 41, 59), font=font_small, anchor="ms")
    draw.text((rx2 + 45, guide_y + 12), "cm", fill=(30, 41, 59), font=font_mid, anchor="lm")

    # Footer Instructions
    footer_y = PAGE_H - margin_px - 70
    draw.text((cx, footer_y - 30), "PRINT AT 100% SCALE (DO NOT FIT/SHRINK TO PAGE) ON A3 PAPER (297 x 420 mm)", fill=(220, 38, 38), font=font_sub, anchor="mm")
    draw.text((cx, footer_y + 15), "Compatible with OAK-D Pro Visual Odometry + ArUco Landing Daemon (CUAV X7 Pro)", fill=(100, 116, 139), font=font_small, anchor="mm")

    # Save PNG
    png_path = os.path.join(OUTPUT_DIR, f"{output_base_name}.png")
    pil_img.save(png_path, "PNG", dpi=(300, 300))
    print(f"[OK] PNG saved: {png_path}")

    # Save PDF
    pdf_path = os.path.join(OUTPUT_DIR, f"{output_base_name}.pdf")
    pil_img.save(pdf_path, "PDF", resolution=300.0)
    print(f"[OK] PDF saved: {pdf_path}")

    return png_path, pdf_path

# Generate 20 cm pad (Exact default in flight_daemon.py: marker_size_m = 0.20)
build_a3_pad(200.0, "aruco_landing_pad_A3_20cm", "Marker Physical Width = 200 mm (20.0 cm)")

# Generate 25 cm pad (Larger target for higher altitude detection)
build_a3_pad(250.0, "aruco_landing_pad_A3_25cm", "Marker Physical Width = 250 mm (25.0 cm)")

print("All ArUco print targets generated successfully.")
