Read `/data/DRONE/HANDOFF_2026-10-25.md` in full before you do anything
else, then say in two lines what you understand the current state to be.
Do not change any file until I have agreed with those two lines.

You are continuing work on DroneVPS: GNSS-denied navigation on my ZD850
hexacopter. I am the pilot. I fly it myself, several times a day, and I
read your analysis between flights.

How I want you to work:

* Ask me before you write or fit any guard — anything that rejects,
  clamps, vetoes or zeroes a value. Measurements, logging and geometric
  corrections are not guards and you can just do those.
* Never gate anything on GPS. When I am on VPS, no GPS data reaches the
  estimate. If the VPS stops, the answer is ALT_HOLD, never an automatic
  fall back. When I flip the switch to GPS, the Jetson stops injecting
  immediately.
* Measure before you claim. This project has lost whole days to
  conclusions drawn from unfair comparisons. `memory/dronevps-measure-
  fairly.md` and the handoff list exactly how. Score position only against
  the raw receiver, never against the EKF's fused output, and filter the
  corrupt rows out first.
* When I ask you a question, answer it. Do not start changing things.
* Commit after every change and push to BOTH branches
  (recover/post-ssd-loss-19oct and feat/vio-vps). I test fly and I do not
  want work lost.
* Write to me in Hinglish. Anything that goes to the radio or to a pilot
  is English and professional.
* Tell me plainly when something did not work or when you were wrong. I
  would rather hear "this failed the A/B" than a hopeful summary.
* When I tell you my theory about the hardware, measure it rather than
  argue. I have been right about the lever arm, the camera rotation, the
  tilt, the barometer and the map latency.

The immediate situation: flight 245 was the first one that worked — 634
seconds on the VPS with no failsafe. Four faults remain, listed in the
handoff. Code was written for three of them and then reverted at my
instruction, because the flight after 245 went worse and I had also
changed the origin by hand at the same time, so nothing was separable.
Those commits are in history. If you bring them back, bring them back ONE
AT A TIME so I can tell which did what.

Start by reading the handoff and the memory files it points at.
