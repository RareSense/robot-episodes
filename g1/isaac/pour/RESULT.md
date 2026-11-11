# Kettle pour. Isaac Sim PhysX, AI-operated

Status: PARTLY DONE. Take pgW4, GPU PhysX with real particle water, 600 Hz, one continuous run from t 31.5 to 45.5 (14 s).

- DONE (seen in frames): from its stance beside the mug, the robot brings the kettle over the mug, tilts it to 92 deg, pours real water into the mug and tilts back.
- FAIL: the spill. 10.1 ml left the stream around the mug during the pour, against 1.9 ml in the recording.
- NOT DONE: t 0.5 to 31.5, which is the walk to the kettle, the lift off its base and the carry walk back. The take starts from the recorded state at t 31.5 with the kettle already in the hand, then nothing writes state.
- That first walk FAILS. Take pgA1 fell at t 4.66, and pgA2 with the ankle balance fell at t 4.79. Seen in frames\stageA_fall.jpg: at t 3.5 the robot is 62 mm behind the recording as it side-steps along the counter, and at t 4.5 it tips sideways. So the kettle lift (t 18 to 22) and the carry were never reached in a take.

## Water
- Start: 380 ml in the kettle.
- End: 134.0 ml in the mug (recording 146.8), 232.2 ml in the kettle, 13.8 ml outside.
- Mug level 71.9 mm.
- Outside water, in two parts:
  - 3.7 ml left in the first seconds, before the kettle reaches the mug. It comes out of the spout, because the take starts with the kettle already tilted 41 deg and the first tilt to 55 deg sloshes it out. 23 % of it stays on the counter, the rest falls off the edge.
  - 10.1 ml during the pour, mostly t 42.5 to 45. 99 % of it is on the counter, a median 80 mm from the mug axis. This is the spill next to the mug. Cause not established.
- Earlier take pgW3 lost 20.9 ml in that first second, because the tool filled the tilted kettle as if it stood upright. The fill is now cut level with gravity at the start tilt, and starts at 380 ml, not 400.
- Early exit points in pgW3, kettle frame, t 31.7 to 32.6: x -123 to -142 mm, z 180 to 208 mm, the spout lip. Those droplets are real simulated water. They are drawn, not hidden.
- Some particles that miss the counter pass through the floor (z below 0). The floor does not stop them, so a floor puddle is not modelled.
- The kettle's rigid mass is 0.87 kg plus the 0.38 kg of particles, 1.25 kg in total as in the recording.

## Difference from the MuJoCo recording
See metrics_C.json for all numbers.
- Spout to recording during the pour, t 39 to 44: within 4 mm.
- Tilt: 92.5 deg peak, recording 91.7. After the pour the kettle comes back late: 70 deg at t 45, recording 51. The hand correction ends at t 44.5.
- Kettle origin: up to 74 mm from the recording, as the kettle sits turned in the grip. Seen in frames: the kettle body is turned compared with the clip.
- Mug not moved.

## Live fixes
1. Everything in cup\RESULT.md.
2. Takes pgW1 and pgW2 failed: the moving water turned the kettle in the grip to 104 deg and the spout moved 5 cm past the mug (58.6 and 80.3 ml in the mug, 252 and 214 ml outside). Fix: from t 38.5 to 44.5 the right hand follows the recording corrected by the measured spout and tilt error.
3. Water mass counted once (pgW1 counted it twice).

## Differs from the MuJoCo version
- In MuJoCo the water was a later pass driven by the recorded motion, so it pushed nothing. Here it is in the same run and pushes back on the kettle, which is why the hand correction is needed.
- Water is drawn as blue particles, not a smooth tea surface.
- Starts at t 31.5.

## Motors and hand
- Body joints max 0.218 rad, fingers max 0.230 rad. Left hand max 6.7 mm, right hand max 27.8 mm, mean 7.1 mm (the hand fix). Pelvis max 6.7 mm. Kettle in the hand: slip 13.5 mm over t 32 to 44.5.
- Motors at their limit: right middle_0 in 100 % of frames, right thumb_1 15.6 %, thumb_0 5.6 %, right ankle pitch 2.1 %. The drive is capped at the limit, so nothing goes over it. The grip works at the limit for the whole take.

## Frames read
- pour_C_film.mp4 at video t 3, 10.5, 11.5, 13.5 next to 3_pour_film.mp4 at the same recording times: frames\sheet4.jpg. Kettle over the mug and the stream into the mug match. Kettle body turned against the clip. At t 13.5 the kettle is still further tilted than in the clip, and drops fall in front of the mug.
- pour_C_eye.mp4 at video t 0.5, 3, 11.5, 13.5: frames\sheet5_eye.jpg. A thin trail of droplets on the counter at t 3, water in the mug, a spill patch beside the mug at t 13.5.
- pgW3 head-camera frames at video t 0.2 to 1.2: frames\sheet_eye.jpg. Water at the spout at t 0.2, the source of the early droplets.
- An eye render with no water drawn (a render fault) was replaced by this one.

## Files
pour_C_rec.npz (30 Hz), pour_C_film.mp4 (fixed camera, same view as the clip, real speed), pour_C_eye.mp4 (head camera), metrics_C.json, water_log.json (water volumes at 10 Hz). The particle positions (water.f32, 72 MB) are not in the repo.
