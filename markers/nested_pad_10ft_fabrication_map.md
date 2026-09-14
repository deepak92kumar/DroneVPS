# Nested landing pad: fabrication map

- Dictionary: OpenCV DICT_4X4_50. Outer marker ID 45, inner marker ID 0.
- Whole white sheet: 4064 x 4064 mm (13.33 ft).
- Outer black square: 3048 mm (10 ft), starts 508 mm from every sheet edge. Grid 6 x 6, cell 508 mm.
- Inner black square: 450 mm, centred (starts 1807 mm from every sheet edge). Grid 6 x 6, cell 75 mm.
- Row 1 is the TOP (forward) edge, column 1 is the LEFT edge. B = black, W = white. Tolerance: keep every cell within +/-1% (outer +/-5 mm, inner +/-1 mm).

## Outer ID 45 (cell 508 mm)

| row \ col | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| 1 (y 508-1016 mm) | B | B | B | B | B | B |
| 2 (y 1016-1524 mm) | B | B | B | W | B | B |
| 3 (y 1524-2032 mm) | B | W | W | W | B | B |
| 4 (y 2032-2540 mm) | B | W | W | W | B | B |
| 5 (y 2540-3048 mm) | B | W | B | W | W | B |
| 6 (y 3048-3556 mm) | B | B | B | B | B | B |

## Inner ID 0 (cell 75 mm)

| row \ col | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| 1 (y 1807-1882 mm) | B | B | B | B | B | B |
| 2 (y 1882-1957 mm) | B | W | B | W | W | B |
| 3 (y 1957-2032 mm) | B | B | W | B | W | B |
| 4 (y 2032-2107 mm) | B | B | B | W | W | B |
| 5 (y 2107-2182 mm) | B | B | B | W | B | B |
| 6 (y 2182-2257 mm) | B | B | B | B | B | B |

Column x ranges: outer col k = 1: 508-1016, 2: 1016-1524, 3: 1524-2032, 4: 2032-2540, 5: 2540-3048, 6: 3048-3556 mm
Column x ranges: inner col k = 1: 1807-1882, 2: 1882-1957, 3: 1957-2032, 4: 2032-2107, 5: 2107-2182, 6: 2182-2257 mm
