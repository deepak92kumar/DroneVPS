--[[
  gps_source_switch.lua  -  RC6 picks the position source, on the FLIGHT
  CONTROLLER, so the switch works whether or not the Jetson is alive.

  WHY THIS EXISTS. gps_rc_switch.py does the same job on the companion, and
  on 23 October it could not: the Jetson rebooted mid-session and for the
  whole of that window the transmitter could not change the source at all,
  because the thing reading the channel was not running. Worse, GPS_PRIMARY
  stays wherever it was left - if that is 1 and the VPS is dead, the aircraft
  is pinned to a source with no data.

  THE PILOT'S RULES, and this script does nothing beyond them:

    1. Whichever position the switch is in, that is the source. No exceptions.
    2. NO AUTOMATIC SOURCE CHANGE, EVER. If the VPS stops producing, the
       answer is ALT_HOLD, not a silent fall back to the receiver. Under
       spoofing the receiver is the attack, not the refuge - "samjho gps
       physically nikal gaya hai, jab tak switch na ho tab tak no gps".
    3. No health check, no liveness test, no refusal. The companion's own
       refusal was removed on 23 October for the same reason.

  So there is deliberately no timeout, no watchdog and no fallback in here.
  It reads one channel and writes one parameter.

  CHANNEL. RC6, the same one the companion uses, and the same three
  positions read off RC6_MIN/RC6_MAX as 1045 / 1495 / 1945:

      <= 1270      real GPS (GPS_PRIMARY = 0)
      1271-1719    centre detent - HOLDS, commands nothing
      >= 1720      VPS         (GPS_PRIMARY = 1)

  The centre is blank on purpose. A two-way threshold would make passing
  through the detent on the way to the VPS command a source change nobody
  asked for.

  RUNNING BOTH. This and the companion script can both be active: they read
  the same channel and write the same value, so they agree by construction.
  If only one is wanted, stop drone-rc-switch - the aircraft does not need
  the companion for this.

  INSTALL
      SCR_ENABLE = 1                     (already 1 on this aircraft)
      copy to the SD card as APM/scripts/gps_source_switch.lua
      reboot the flight controller
      RC6_OPTION must stay 0             (already 0)
--]]

local LOW_US, HIGH_US = 1270, 1720
local CH = 6
local PERIOD_MS = 200
local STABLE_N = 4          -- consecutive samples before acting, as in the
                            -- companion: four at 200 ms is 0.8 s, enough that
                            -- a brushed stick is not a command.

local want, stable, applied = nil, 0, nil

local function decide(us)
  if us <= LOW_US then return 0 end
  if us >= HIGH_US then return 1 end
  return nil                 -- centre: hold whatever is selected
end

function update()
  -- rc:get_pwm is the documented accessor and returns nil when the channel
  -- has never been seen. A missing channel is NOT a reason to change the
  -- source - it is a reason to say so and leave it alone.
  local us = rc:get_pwm(CH)
  if not us or us == 0 then
    return update, PERIOD_MS
  end

  local d = decide(us)
  if d == nil then
    stable = 0
    return update, PERIOD_MS
  end

  if d == want then stable = stable + 1 else want, stable = d, 1 end
  if stable < STABLE_N then return update, PERIOD_MS end

  -- WRITTEN EVERY TICK WHILE THE SWITCH DEMANDS IT, never skipped because we
  -- believe it is already set.
  --
  -- That belief is exactly what cost flight 233 its landing. The companion
  -- script kept a cached `applied`, took one wrong read-back at 10:31:51, and
  -- then swallowed the pilot's next command for 119 seconds - the aircraft
  -- touched down on the VPS with the transmitter asking for GPS. A cache may
  -- decide what is SAID. It may never decide whether the command is sent.
  --
  -- param:set writes RAM, not storage - set_and_save is the one that would
  -- wear flash - so doing this five times a second costs nothing and cannot
  -- be lost. GPS_PRIMARY is meant to be volatile anyway: on a reboot the
  -- aircraft should come up on whatever the switch is actually asking for,
  -- not on what it was asked for last flight.
  if not param:set("GPS_PRIMARY", d) then
    gcs:send_text(3, "GPS SRC: param:set failed")
    return update, PERIOD_MS
  end

  local got = param:get("GPS_PRIMARY")
  if got == nil or math.floor(got + 0.5) ~= d then
    -- Said every tick on purpose. A source that will not take is the one
    -- fault the pilot has to hear about immediately, and it is rare enough
    -- that it will not flood the radio.
    gcs:send_text(3, "GPS SRC: set " .. d .. " read back " ..
                     tostring(got) .. " - NOT APPLIED")
    applied = nil
  elseif applied ~= d then
    gcs:send_text(6, d == 1 and "GPS SRC -> VPS (GPS2)"
                           or  "GPS SRC -> real GPS (GPS1)")
    applied = d
  end

  return update, PERIOD_MS
end

gcs:send_text(6, "GPS SRC: RC" .. CH .. " <=" .. LOW_US ..
                 " GPS1 | >=" .. HIGH_US .. " VPS | centre holds")
return update, 1000
