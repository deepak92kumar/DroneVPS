# DroneVPS Studio

Two tools in one page, served from the Jetson to whatever browser is on the
field network.

**On the Jetson: double-click the DroneVPS Studio icon on the desktop.** It
starts the server if it is not already running and opens the page. Install the
icon once with

    bash /data/DRONE/studio/install-icon.sh

which puts it on the desktop and in the Activities menu, and marks it trusted
(GNOME will not run an untrusted launcher).

**From a laptop on the same network** no icon is needed - the server listens on
every interface:

    http://<jetson-ip>:8099

By hand, if you want a terminal:

    python3 studio/server.py              # or studio/launch.sh
    python3 studio/server.py --port 9000 --host 127.0.0.1

Standard library, GDAL, pymavlink, numpy, matplotlib - all of which were
already on this machine. No framework, no CDN, nothing to install, and the
page works with the network unplugged.

---

## 1. Map Cutter

Cuts a flyable sheet out of a big GeoTIFF.

* Any GeoTIFF under `/data/DRONE` is listed.
* In **Draw a box** mode (the default) a plain **drag draws the crop box**,
  dragging inside it moves it, and the handles resize it. **Pan with the right
  button or shift+drag**, zoom with the wheel, double-click to clear the box.
  (The first version had it the other way round - drag panned and only
  shift+drag drew a box, under a button that said "Drag a box" - which is why
  the first session on it produced no cut at all.)
* **Go to a position** takes `30.895774, 74.614113` or
  `30 53 44.8 N 74 36 50.8 E`, drops a marker, and says whether that point is
  on the sheet at all.
* **Centre & size** cuts a box of a given ground size around a typed position
  - "1.5 km around the take-off point" - instead of dragging for it.
* **Build overview pyramid** writes `<sheet>.tif.ovr` beside a large source so
  panning stops decimating the full-resolution image on every move. The source
  file itself is never modified.

### What it guarantees about the cut

`vps_vio/map_cache.py` reads the GeoTIFF's six-number geotransform and trusts
it. It never checks any of this, so the cutter does:

| It assumes | Why it matters |
|---|---|
| EPSG:4326, degrees | It multiplies pixel size by 111 320 m/deg. A UTM sheet would read 111 320x too coarse. |
| North-up, `gt[2] = gt[4] = 0` | `GeoRef.from_gdal` throws the two rotation terms away. A rotated sheet is not detected, it is silently mis-georeferenced - and the matcher already rotates the camera frame north-up by the aircraft's heading, so a rotated sheet adds a second rotation instead of cancelling. **This is the "yaw" question.** |
| `gt[5] < 0`, origin top-left | A positive `gt[5]` mirrors every latitude the matcher reports. |
| Fits in RAM as one greyscale array | MapCache loads the whole sheet at startup on purpose: a windowed GDAL read cost 1.4 s against 0.7 s of arithmetic. |

A cut from a sheet that is already 4326 and north-up is a **pixel-exact copy**
- `gdal.Translate` on whole source pixels, no resampling, bit-identical
imagery. Anything else (UTM, rotated, flipped, or a new resolution asked for)
is warped, which is the only honest thing to do with it. The `how` line in the
result says which path ran and why.

Every cut is then **re-opened from the disk and checked**, so what is reported
is the file as it landed and not the parameters it was asked for. Give it the
take-off point and it also checks that the point is on the sheet with 150 m to
spare, which is the preflight check that decides whether there will be any map
fixes at all.

**The metre anisotropy is left alone on purpose.** A 4326 sheet has equal
pixel size in degrees, so in metres it is 1/cos(latitude) anisotropic - 16.5%
at 30.9 N, which is the "16 m that has never been explained" from the 12 Oct
report. `map_query.north_up_query` corrects for it with `map_gsd_x_m`. A sheet
resampled here to square metre pixels would be corrected twice. The one
exception is when a resolution is explicitly chosen in the dropdown: that
produces square metre pixels, which `map_query` then skips correcting, and
both are consistent.

### Loading a cut into the node, and proving it opens

Cutting a sheet is half the job; the node has to actually fly it.

**"Load the selected sheet"** writes the path into
`/data/DRONE/maps/ACTIVE_MAP` and points `/data/DRONE/maps/active.tif` at it.
`vps_vio/active_map.py` reads that pointer, `live_node.py` takes its
`DEFAULT_MAP` from it, and `preflight.py` checks coverage against the same
sheet rather than against a constant. The node's startup banner prints the
path **and why it is that one**, so a flight never begins without the sheet
being stated:

    map /data/DRONE/maps/site_a.tif  (pointer: set in /data/DRONE/maps/ACTIVE_MAP)

If the pointer is missing, empty, or names a file that is not on the machine,
it falls back to `bhh.tif` **and says that it fell back** - a silent fallback
to the wrong site is the failure this is meant to prevent. `--map` on the
command line still overrides everything, and "Back to the default sheet"
removes the pointer.

**"Check it loads"** does not re-read the header by this program's own rules.
It constructs the real `vps_vio.map_cache.MapCache` - the same class the
flying node constructs at startup - and then slices a matcher window out of it
at the take-off point. If that works here it works in the air. Every failure
carries its remedy, because "CRS is not EPSG:4326" is a puzzle and "cut it
again from the Map tab, which reprojects it" is an instruction. What it
catches:

| It fails on | and tells you |
|---|---|
| a projected or rotated sheet | why it can never match, and to re-cut |
| a truncated or half-copied file | that GDAL cannot read a block, with the byte |
| a sheet too large for RAM | that the node loads the whole thing at startup |
| the take-off point off the sheet | that there will be no map fixes for the whole flight |
| a blank window at that point | that water, cloud or a nodata margin has no texture to match |

**"Download this sheet"** hands the file back through the browser, for taking
a cut away from the Jetson.

---

## 2. Flight Review

Reads a flight controller `.BIN`, a node `vps_*` log, or both, and says what
an assistant would say about them - without one.

* **Pairing is measured, not guessed.** Two node logs are usually written
  within minutes of the same `.BIN`; file times picked `vps_00167` for flight
  245 and the flight was `vps_00165`. "Find the matching node log" aligns each
  candidate's altitude against the flight controller's and ranks by
  correlation, which separates a flight from a bench run immediately.
* A `.BIN` takes about 15 s to parse the first time and is then cached in
  `studio/cache` as an `.npz`, keyed by the file's size and mtime.
* Twelve plot panels, the track drawn on whichever satellite sheet covers it,
  and the saved camera frames.
* **Download the review as Markdown** writes the whole thing to a file, in the
  shape this project's own handoffs use.

### The rules it checks, and where each threshold comes from

| Finding | Threshold from |
|---|---|
| Position against the raw receiver, by height band | flight 245: 0.44 m at 38 m, 1.42 m at 82 m |
| Speed sent while the receiver says standing still | flights 239/240: 2.40 m/s at 70-85 m, and the N sweep that closed the longer-baseline family |
| Deadband chatter - zero crossings per second | flight 245: 307 in 634 s, 39.9% of rows zeroed |
| Map fixes: attempts against accepted, and what the 0.5 m cap costs | flight 245: 234 attempted, 159 accepted, 0.251/s; flight 239: all 68 clamped |
| Height - the barometer against the lidar | flight 245: 1.07 at 100 m rising to 1.72 at 10 m |
| Optical flow: camera disagreement, stale velocity | flight 163 (2.32 px) against flight 245 (0.050 px) |
| Loop rate by height | flight 239: 47.7% slow at 110 m, flight 240: 9.7% |
| Circling (toilet bowl) | flight 245: 754 deg in 24 s while yaw moved 9.6 |
| Mode changes the aircraft made itself | flight 245 was the first with zero ALT_HOLD |
| EKF innovations against `FS_EKF_THRESH` | reported as a failsafe question only - see below |
| EKF primary core changes | ArduPilot logs these as subsystem 24; the ECode is the new core, not an error |
| Commanded tilt against `ATC_ANGLE_MAX` | flight 245: saturated at 15 deg, rates to 56.9 deg/s |
| Vibration | 0.8 / 1.6 / 2.3 m/s2 at rest, 20 Oct |
| The switch against what the node did | 23 Oct: landed on the VPS while channel 6 asked for GPS |
| Distance run from each handover | log 228 (0.9 m in 23 s) against log 229 (11.1 m in 24 s) |
| The typed origin against where it flew | preflight checks this at 30 m |

### Four measurement rules are built in

Breaking them is how this project has repeatedly fooled itself
(`memory/dronevps-measure-fairly.md`):

1. **Truth is `gps1_north_m`/`gps1_east_m`** - GPS_RAW_INT, the receiver
   before the EKF. Never `gps_north_m`, which is GLOBAL_POSITION_INT, the
   EKF's fused output; with our GPS_INPUT injected as GPS2 that column is
   partly our own injection coming back, and scoring against it once moved a
   conclusion by a factor of four. A log without usable `gps1_*` gets a
   finding that says so and the position checks are skipped, not substituted.
2. **The gps1 columns carry corrupt rows**, up to 23 km. Anything beyond 400 m
   of the origin is dropped and the count is reported.
3. **Both sides of a comparison get the same smoothing**, named in the output.
4. **A hold is scored by how far the aircraft ran** from the handover against
   the raw receiver - not by the EKF's variance. SV is the EKF agreeing with
   itself, so anything that makes our output quieter improves it by
   construction; two changes once improved SV while making the hold twelve
   times worse. SV is still reported, under a heading that says what it does
   and does not answer.

---

## Files

    studio/server.py        routes, and a confinement to /data/DRONE
    studio/geotiff.py       sheets, tiles, the cut and its verification
    studio/logs.py          discovery, .BIN parsing and caching, clock alignment
    studio/analysis.py      the findings engine
    studio/plots.py         the twelve panels
    studio/report.py        the Markdown review
    studio/flightsheet.py   loading a sheet into the node, and the load check
    studio/launch.sh        what the desktop icon runs
    studio/install-icon.sh  puts the icon on the desktop and in the menu
    studio/icon.png         the icon
    studio/static/          one page, one stylesheet, one script
    studio/cache/           parsed .BIN caches (safe to delete)

    vps_vio/active_map.py   the pointer the node reads (tests/test_active_map.py)
