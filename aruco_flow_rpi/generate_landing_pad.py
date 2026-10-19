#!/usr/bin/env python3
"""
ArUco Precision Landing Pad Generator
Generates high-contrast printable target pads (DICT_4X4_50, Marker ID 0)
for autonomous precision landing and return-to-home relocalization.
"""

import os
import cv2
import cv2.aruco as aruco
import numpy as np

OUTPUT_DIR = "/data/DRONE"
MARKER_ID = 0
MARKER_DICT = aruco.DICT_4X4_50
PAD_SIZE_PX = 1200 # High resolution for sharp printing

def generate_landing_pad():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    d = aruco.getPredefinedDictionary(MARKER_DICT)
    
    # Generate 700x700 marker
    marker_core = aruco.generateImageMarker(d, MARKER_ID, 650)
    
    # Create white canvas (1200 x 1200)
    canvas = np.ones((PAD_SIZE_PX, PAD_SIZE_PX), dtype=np.uint8) * 255
    cx, cy = PAD_SIZE_PX // 2, PAD_SIZE_PX // 2
    
    # 1. Concentric Target Rings (Range Guide)
    cv2.circle(canvas, (cx, cy), 570, 0, 4)
    cv2.circle(canvas, (cx, cy), 520, 180, 2)
    cv2.circle(canvas, (cx, cy), 470, 180, 2)
    
    # 2. Outer Crosshair Alignment Lines
    cv2.line(canvas, (cx, 20), (cx, cy - 350), 0, 4)
    cv2.line(canvas, (cx, cy + 350), (cx, PAD_SIZE_PX - 20), 0, 4)
    cv2.line(canvas, (20, cy), (cx - 350, cy), 0, 4)
    cv2.line(canvas, (cx + 350, cy), (PAD_SIZE_PX - 20, cy), 0, 4)
    
    # 3. Paste Central ArUco Marker
    half = 650 // 2
    canvas[cy - half : cy + half, cx - half : cx + half] = marker_core
    
    # 4. Corner Markers & Alignment Arrows
    arrow_pts_north = np.array([[cx, 35], [cx - 25, 75], [cx + 25, 75]])
    cv2.fillPoly(canvas, [arrow_pts_north], 0)
    
    # 5. Descriptive Labels
    cv2.putText(canvas, "NORTH / FRONT", (cx - 75, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.60, 0, 2)
    cv2.putText(canvas, "AI PRECISION LANDING PAD", (cx - 160, PAD_SIZE_PX - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, 0, 2)
    cv2.putText(canvas, f"ARUCO 4x4_50 | ID: #{MARKER_ID} | 20cm x 20cm TARGET", (cx - 185, PAD_SIZE_PX - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.50, 80, 1)

    # Save PNG
    png_path = os.path.join(OUTPUT_DIR, "landing_pad_marker0.png")
    cv2.imwrite(png_path, canvas)
    print(f"[+] High-Resolution Landing Pad PNG generated: {png_path}")

    # Generate Printable HTML Template
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Autonomous Drone Landing Pad (ArUco #{MARKER_ID})</title>
<style>
  @page {{ size: A4 portrait; margin: 10mm; }}
  body {{
    font-family: Arial, sans-serif;
    text-align: center;
    background-color: #f8fafc;
    margin: 0;
    padding: 20px;
  }}
  .container {{
    max-width: 750px;
    margin: 0 auto;
    background: white;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
  }}
  h1 {{ margin: 0 0 5px 0; color: #0f172a; font-size: 24px; }}
  p {{ margin: 4px 0 15px 0; color: #475569; font-size: 14px; }}
  .pad-img {{
    width: 100%;
    max-width: 580px;
    border: 3px solid #0f172a;
    border-radius: 8px;
    background: white;
  }}
  .instructions {{
    margin-top: 18px;
    text-align: left;
    background: #f1f5f9;
    padding: 15px 20px;
    border-radius: 8px;
    font-size: 13px;
    color: #334155;
    line-height: 1.6;
  }}
  .btn {{
    display: inline-block;
    background: #2563eb;
    color: white;
    padding: 10px 24px;
    border-radius: 6px;
    font-weight: bold;
    text-decoration: none;
    margin-top: 15px;
    cursor: pointer;
  }}
  @media print {{
    .instructions, .btn {{ display: none; }}
    .container {{ box-shadow: none; padding: 0; }}
    body {{ background: white; }}
  }}
</style>
</head>
<body>
<div class="container">
  <h1>Autonomous Drone Precision Landing Pad</h1>
  <p>Standardized Optical Target &bull; Dictionary: DICT_4X4_50 &bull; Marker ID: <b>#{MARKER_ID}</b></p>
  <img src="landing_pad_marker0.png" alt="ArUco Landing Pad" class="pad-img">
  
  <div class="instructions">
    <b>Print & Setup Instructions:</b>
    <ol>
      <li>Click <b>Print Landing Pad</b> below (or press Ctrl+P). Set scale to <i>Fit to Printable Area</i> (100%).</li>
      <li>For field durability, paste the printout on cardboard, sunboard, or a vinyl mat.</li>
      <li>Place this pad at your Takeoff / Home location with the <b>NORTH / FRONT</b> arrow aligned toward your takeoff heading.</li>
      <li>The OAK-D Pro downward camera will detect this marker up to 5&ndash;10 meters above ground and land directly on the center!</li>
    </ol>
  </div>
  <button class="btn" onclick="window.print()">Print Landing Pad (Ctrl+P)</button>
</div>
</body>
</html>
"""
    html_path = os.path.join(OUTPUT_DIR, "landing_pad_marker0.html")
    with open(html_path, "w") as f:
        f.write(html_content)
    print(f"[+] Printable HTML Template generated: {html_path}")

if __name__ == "__main__":
    generate_landing_pad()
