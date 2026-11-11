# Kettle pour. Isaac Sim PhysX, AI-operated

Status: PARTLY DONE. Take pgW4, GPU PhysX with real particle water, 600 Hz, one continuous run from t 31.5 to 45.5 (14 s).

- DONE (seen in frames): from its stance beside the mug, the robot brings the kettle over the mug, tilts it to 92 deg, pours real water into the mug and tilts back.
- FAIL: the spill. 10.1 ml left the stream around the mug during the pour, against 1.9 ml in the recording.
- NOT DONE in this take: t 0.5 to 31.5, which is the walk to the kettle, the lift off its base and the carry walk back. The lift alone, t 9 to 23, is a separate take on CPU PhysX, described at the end of this file. The take starts from the recorded state at t 31.5 with the kettle already in the hand, then nothing writes state.
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

# Kettle lift from its base. Isaac Sim PhysX, AI-operated

Status: lift DONE on CPU PhysX for this stage. The motor-limit check is NOT passed as measured: see Motors and hand. Take pcL1, CPU PhysX, 600 Hz, one continuous run from t 9.0 to 23.0 (14 s). It starts from the recorded state at t 9.0, standing at the kettle with the kettle on its base, then nothing writes state.

- DONE (seen in frames): the right hand goes to the handle, closes round it, lifts the kettle off its base and brings it in front of the body. The robot stands the whole time.
- Kettle leaves the base at t 18.33 (recording 18.2). At t 23.0 it is 184 mm above its start, at (2.566, 2.118, 1.069). Recording (2.545, 2.120, 1.076), 22 mm apart. Tilt 29.9 deg, recording 24.7.
- Kettle in the hand, t 18.4 to 23.0, against where the recording has it in the hand: 41 mm at most (t 20.9), 24 mm at the end. The kettle swings in the hook grip in the recording too (111 mm of travel in the hand frame from t 20 to 23, 95 mm here).
- Mug and jar not moved. No fall, pelvis lowest 0.788 m.
- No water in this take. The kettle is one rigid 1.25 kg mass, which includes 0.4 kg for the water.
- NOT DONE: the side-step walk to the kettle before it, t 0.5 to 9, and the carry walk back after it, t 23.5 to 31.5. Both fall on CPU PhysX as well as GPU. Tried live on CPU from the first frame: the robot fell at t 4.2; with the stance-leg pelvis servo at 0.1 and 0.2, sole friction 1.0 and 1.5, and ankle balance gains 0, 1 and 2 it fell between t 3.6 and 4.6. The friction was changed on the material while the sim was open, and I did not check that PhysX took the new value, so the friction variants prove little. The trace shows it tracking within 15 mm until t 2.7, then the pelvis falls to the left during the second left-foot swing (41 mm behind the recording at t 3.0, 122 mm at t 3.5). Seen in a frame at t 3.2 from behind: the body leans left while the right foot is in the air. Continuing this take past t 23 into the carry walk: fell at t 25.5, kettle lost.

## Differs from the MuJoCo version
- The kettle's switch lever and its light are not simulated here. The renderer draws them from the MuJoCo recording, so the lever and light in the videos are not a result of this take.
- Robot self-collision is off.
- The stage starts at t 9.0 and ends at t 23.0. It is not joined to the pour take pgW4, which starts at t 31.5 on GPU PhysX with water.

## Live fixes
1. Everything in cup\RESULT.md.
2. Stance-ankle balance, gains 1.0 / 0.1, as in the mug take (tools\fix_pour.py).

## Motors and hand
- Body joints max 0.012 rad, fingers max 0.064 rad. Left hand max 2.6 mm, right hand max 7.8 mm. Pelvis max 1.9 mm.
- The drive asked for more than the limit in 7.7 % of frames: right ankle pitch 4.4 %, right thumb_1 2.7 %, right index_0 1.3 %. The drive is capped at the limit.
- Peak projected joint force over the motor limit: right middle_1 2.53, middle_0 2.22, index_0 1.88, thumb_1 1.68, index_1 1.48. These are PhysX projected joint forces, which include joint-limit and other constraint loads as well as the motor drive, and I have not separated the parts. So the finger joints carry more load than the motors alone could hold. Read this as NOT shown to be within the motor limits.
- No contact-depth check was run.

## Frames read
- Live, rendered inside the sim process, at t 12, 17, 18.5, 19.5, 20.0, 23.0. At t 20.0 and 23.0 the kettle is off its base with the hand on the handle. A second live camera was inside a wall and showed nothing; it was not used.
- pour_L_film.mp4 at recording t 10, 12, 17.5, 19, 20.5, 23: frames\sheet_L.jpg. Hand closes on the handle at t 17.5, kettle off the base at t 19, in the air at t 20.5, in front of the body and tilted at t 23. The robot's head and shoulder cover part of this view.
- pour_L_eye.mp4 at recording t 17.6 and 20.0: frames\sheet_L_eye.jpg. Hand round the handle, then the kettle lifted.
- NOT SEEN: every frame between those times. No MuJoCo clip exists for this part, so there is no side-by-side pair.

## Files
pour_L_rec.npz (30 Hz), pour_L_film.mp4 (fixed camera chosen for this stage, real speed), pour_L_eye.mp4 (head camera), metrics_L.json.
