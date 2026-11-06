# Cup: take the cup and walk away. Isaac Sim PhysX, AI-operated

Take cg1. GPU PhysX, 600 Hz, one continuous run from the recording's first frame. Status: DONE for the motion. NOT done with real water (see below).

## Result (seen in frames)
- Left hand takes the mug by the body, lifts it, carries it with both hands in view, turns, walks along the counter to about (0.95, 1.78) and stands holding it. No fall.
- Mug end (0.752, 1.774, 0.945). Recording end (0.743, 1.774, 0.946). Difference 8.5 mm.
- Mug in hand from t 10 to the end: slip max 7.0 mm, 6.0 mm at the end. Tilt max 9.8 deg.

## Difference from the MuJoCo recording
- Body joints: max 0.109 rad (right_hip_yaw, from the foot fix below), rms 0.0058 rad.
- Fingers: max 0.179 rad (left thumb_1 at the grasp).
- Left hand: max 36.5 mm at t 20.6, mean 6.8 mm. Right hand: max 36.0 mm at t 20.7, mean 6.1 mm.
- Pelvis: max 31.2 mm, 8.0 mm at the end.
- Mug: max 39.7 mm, 8.5 mm at the end.

## Live fixes on top of the recorded joint targets
1. Stiff leg drives with high damping. At first the body swayed 1 deg at about 1 Hz.
2. Joint-space integral term, because a plain PD sags under load.
3. Finger closing offset = recorded motor torque / finger gain. The recording stores where contact stopped the fingers, so a plain replay squeezes nothing. The torque file holds magnitudes only, so the closing direction is set per joint.
4. Swing-foot servo in the walk. A pure joint replay drifted 13 cm and fell at t 22. While a foot is in the air in the recording, a small Jacobian offset lands it at the recorded spot, and the offset fades once the foot stands.

## Motor limits
Drive force is capped at the MJCF limits. The drive asked for more than the limit in 29.3 % of frames: left thumb_1 22.5 %, right ankle pitch 4.0 %, left ankle pitch 1.7 %. In those frames the motor sat at its limit. MuJoCo also held thumb_1 at its 1.4 N m limit during the grip. The peak_joint_force numbers in metrics.json go above 1.0, up to 1.36. They are PhysX projected joint forces, which include more than the motor drive, and I have not separated the parts.

## Differs from the MuJoCo version
- No tea in the mug. This take carries 0.45 kg as one rigid mass (0.30 kg mug + 150 ml), the same load as the recording. The MuJoCo clip's tea was added after the fact from a separate water pass. Real-water carry: NOT RUN.
- Robot self-collision is off here. It was on in MuJoCo.
- Contact model: PhysX rigid contacts, hands friction 1.0, soles 0.6, as the MJCF pair rule.
- No walking disclosure needed: this take walks.

## Frames read
- cup_film.mp4 at t 6.5, 13, 18, 21.9, next to 6_cup_film.mp4 at the same times: frames\sheet_film.jpg
- cup_eye.mp4 at t 14, next to 6_cup_eye.mp4: frames\sheet_eye.jpg
- Live still at the end of the first full walk (c3_d)
- NOT SEEN: every frame between those times. No contact-depth check was run.

## Files
cup_rec.npz (30 Hz, Stretch pz_rec keys where they apply, plus prop, objects, tr), cup_film.mp4 (fixed camera, same as the clip, real speed), cup_eye.mp4 (head camera), metrics.json, match_pair_mujoco_left_isaac_right.png (t 13).
