# Mug from the rack. Isaac Sim PhysX, AI-operated

Status: PARTLY DONE. GPU PhysX, 600 Hz. Two separate takes, not one episode.

- Stage A (take mgA1, t 0.5 to 14.3): walks in, reaches into the rack, grips the mug by the body, draws it out, starts the carry walk. Stopped at t 14.3 while still standing.
- Stage B (take mgB1, t 18.0 to 22.9): starts standing at the set-down stance, with the mug in the left hand at its recorded pose, and sets it down. One start state from the recording, then nothing writes state.
- NOT DONE: the carry walk from t 14.3 to 18. Run past t 14.3, the robot FALLS. Full GPU take mgF1 fell at t 16.04. Six CPU gain variants and slower replay at 0.6 to 0.8 speed all fell too (t 13.2 to 17). Falls start between t 14.5 and 15.

## Stage A numbers
- At t 14.3: pelvis 27.6 mm from the recording, mug 64.9 mm from the recording.
- Body joints max 0.314 rad (left elbow, from the hand fix). Fingers max 0.045 rad.
- Left hand max 60.8 mm, mean 14.8 mm. Right hand max 57.5 mm, mean 17.7 mm. Pelvis max 49.3 mm. Mug max 71.9 mm.
- Mug in hand t 11.5 to 14.2: slip 18.3 mm, tilt 22.4 deg. That is not a clean carry.
- Motors at their limit in 50.3 % of frames: left index_0 34.1 %, thumb_1 15.0 %, ankle pitch 13 to 15 %.

## Stage B numbers
- Mug set down at (1.727, 2.233, 0.860). Recording (1.724, 2.236, 0.860). 4.1 mm apart.
- Body joints max 0.014 rad. Fingers max 0.522 rad, which I have not explained. Left hand max 19.4 mm, right 13.4 mm. Pelvis max 13.8 mm, 2.7 mm at the end.
- Mug in hand t 18 to 20.2: slip 9.5 mm, tilt 15.3 deg.
- Motors at their limit in 67.3 % of frames, almost all left index_0 (67.0 %).

## Live fixes
1. Everything in cup\RESULT.md: stiff damped legs, integral term, finger torque offset, swing-foot servo.
2. Left hand servoed to its recorded world path from t 6.4 to 11, plus a 20 mm move toward +y from t 7.8 to 9.8. Seen in frames: the fingertips landed on the mug rim, so the grip missed. In MuJoCo soft contact let them pass behind the mug.
3. Stance-ankle balance on pelvis attitude from t 11.5, gains 1.0 / 0.1 (stage A only). Without it the fall comes sooner.

## Differs from the MuJoCo version
- The carry from t 14.3 to 18 is missing, and the two stages are separate takes.
- Stage A's fingers go 20 mm further behind the mug.
- Self-collision off.

## Frames read
- mug_A_film.mp4 at t 6, 10, 12.5, 13.4 next to 1_mug_film.mp4: frames\sheetA.jpg. Close match.
- mug_B_film.mp4 at clip t 18.5 and 22 next to the same clip: frames\sheetB.jpg. Close match.
- Live stills during the rack work (m1 to m5) and of the fall (mgF1 at t 14.3 to 15.2).
- Head camera, re-rendered: mug_A_eye.mp4 at t 9 and mug_B_eye.mp4 at t 3 next to 1_mug_eye.mp4: frames\sheet_eye2.jpg. The rack view matches at t 9. The set-down view is close, compared by eye at slightly different moments. The robot's head no longer shows.
- NOT SEEN: all other frames. The exact cause of the fall is not known. Frames at t 14.3 to 15.2 show the legs matching until t 14.9, then the right leg spreads. Pelvis height jumps 27 mm at t 14.5.

## Files
mug_A_rec.npz, mug_B_rec.npz (30 Hz), mug_A_film.mp4, mug_B_film.mp4 (same moving camera as the clip, real speed), mug_A_eye.mp4, mug_B_eye.mp4 (head camera), metrics_A.json, metrics_B.json, match_pair_A_*.png, match_pair_B_*.png.
The clip's film camera moves, so no fixed outside camera exists for this task yet.
