# Stir. Isaac Sim PhysX, AI-operated

Status: PARTLY DONE. Take sgB2, GPU PhysX, 600 Hz, one continuous run from t 15.5 to 50.3 (34.8 s).

- DONE (motion seen in frames, except the part the hands hide): spoon carried to the mug, lowered in, stirred, released leaning on the rim, right hand withdrawn. Left hand steadies the mug.
- NOT DONE: taking the spoon from the tub, t 0.5 to 15.5. The take starts from the recorded state at t 15.5 with the spoon already in the hand, then nothing writes state.
- Why the pick fails: the thumb touches the spoon first, the spoon rolls about its long axis about 10 deg more than in MuJoCo, and it escapes sideways out of the pinch. About 30 live variants on GPU and CPU (grip strength, grip timing, finger lead, hand offsets). Hand offsets of 8 to 12 mm lifted the spoon a few times, but it dropped again by t 14 and the result did not repeat from run to run.
- NO LIQUID in this take. The Isaac mug is empty, black inside. The MuJoCo clip's grey surface was a separate water pass. No water was simulated here, so the stir moves no tea.

## Difference from the MuJoCo recording
- Body joints: max 0.158 rad (at the release fix). Fingers: max 0.079 rad.
- Left hand: max 4.3 mm, mean 0.9 mm. Right hand: max 19.1 mm (the release fix), mean 1.5 mm. Pelvis: max 4.5 mm, 0.7 mm at the end.
- Spoon origin: max 14.2 mm, mean 5.6 mm. End (1.750, 2.234, 0.869), recording (1.737, 2.238, 0.869).
- Spoon end pose: leans on the same part of the rim as the clip. Handle end (1.757, 2.261, 1.006), recording (1.746, 2.274, 1.004), 17 mm apart. Direction 3.6 deg off the recording. Tilt 11.7 deg, recording 15.4.
- Mug: moved 10.5 mm against the recording, because the release fix pushes it. Relative to the mug the handle end is within 9 mm of the recording. That 10.5 mm is a cost of the fix.
- Bowl circle over t 30 to 43.5, measured the same way on both: radius 6.1 mm and 3.19 turns here, 5.8 mm and 2.21 turns in the recording.
- Motors at their limit in 15.1 % of frames, almost all left ankle pitch (15.0 %).

## Live fixes
1. Everything in cup\RESULT.md.
2. Finger torque x1.5. At x1.0 the spoon dropped straight through the grip at the start of the take.
3. Release: right hand moved +18 mm in x from t 43.2 to 44.4. In take sgB1, without it, the handle fell 34 mm along the rim from the recorded place, direction 15.8 deg off. The fall is bistable: it goes one side or the other of a corner in the mug's collider. In live variants +10 mm fell the wrong way, and +18 and +20 mm went the right way. A short +20 mm did not, so this is a narrow setting, and the take is one run of it.

## Frames read
- stir_B_film.mp4 at video t 22, 28.8, 34 (recording t 37.5, 44.3, 49.5) next to 5_stir_closeup.mp4 at the same recording times: frames\sheet2.jpg. Hands, spoon and rim lean match closely. Isaac mug empty, and it sits about 1 cm further to the left of the frame.
- From this camera the hands cover the spoon during the stir, as in the clip. The stir itself is measured, not seen: NOT SEEN in frames.
- stir_B_eye.mp4 at video t 12 and 33: hands at the mug during the stir, then the spoon left on the rim and the right hand over the tray.

## Files
stir_B_rec.npz (30 Hz), stir_B_film.mp4 (fixed camera, same view as the clip, real speed), stir_B_eye.mp4 (head camera), metrics_B.json, match_pair_B_mujoco_left_isaac_right.png. Earlier take: takes\sgB1.
