# Tea bag into the mug. Isaac Sim PhysX, AI-operated

Status: DONE on CPU PhysX. Take bcF1, CPU PhysX, 600 Hz, one continuous run from the recording's first frame, t 0.5 to 33.75 (33.2 s). After the start state nothing writes state: joint drive targets only, motor limits on, no welds, no attachments.

- DONE (seen in frames): the right hand reaches into the tray, pinches the standing bag by its top edge between thumb and index finger, lifts it out, carries it over the mug, lowers it, opens, and the bag falls into the mug. The hand withdraws.
- Bag at the end: (1.733, 2.242, 0.901). Mug (1.720, 2.240, 0.860). The bag's centre is 12.7 mm from the mug axis and 41 mm above the mug base, inside the cavity (inner wall 20.9 mm from the axis). In the recording it ends at (1.720, 2.239, 0.927).
- Mug moved 0.0 mm, tilt 0.0 deg. No fall, pelvis lowest 0.782 m.
- Tag at the end: (1.749, 2.254, 0.883), outside the mug, hanging on the string against the mug wall. In the recording it lies on the counter at (1.722, 2.277, 0.906).
- Bag leaves the tray at t 11.6, starts to fall from the fingers at t 26.55, at rest by t 28.2.
- Bag in the hand, t 13 to 26.5, measured in the wrist frame: 5.7 mm at most, 4.9 mm at the end. That is over a 5 mm slip limit by 0.7 mm.

## CPU and GPU
- This take is on CPU PhysX. The earlier take bgA1 (teabag_A files) ran on GPU PhysX from t 13 with the bag already pinched, and there the bag dropped at t 21 and ended on the counter. With the same live fixes, CPU PhysX holds the bag to the release. The cause of the GPU drop is still not established.
- The pick from the tray was first tried in this session, live on CPU, and worked at the first try. It has NOT been tried on GPU.
- The saved take repeats the live run to the last digit shown (same bag and tag end positions).

## Live fixes on top of the recorded joint targets
1. Everything in cup\RESULT.md: stiff damped legs, integral term, finger torque offset.
2. Finger torque x1.5 for the whole take (tools\fix_teabag.py).
3. From t 13 to 26.9 the right-hand fingers keep at least the torque of t 14 x1.5, so the pinch does not relax when the recorded torque drops.

## Difference from the MuJoCo recording
- Body joints: max 0.008 rad. Fingers: max 0.046 rad.
- Left hand: max 7.3 mm, mean 1.7 mm. Right hand: max 10.5 mm, mean 2.6 mm. Pelvis: max 7.6 mm, 0.1 mm at the end.
- Bag end: 13 mm to the side of and 26 mm lower than in the recording. It leans on the mug wall inside the mug.
- String: a limp 12 cm distance joint between bag and tag, with no string geometry. The string in the video is drawn only, a curve from staple to tag. In MuJoCo it was 20 capsules.
- Robot self-collision is off here. It was on in MuJoCo.
- Tea bag set-up as in the MuJoCo version: the bag stands in a slot at the front of the pile and is folded lengthwise, 6.5 x 2.5 cm.

## Motors
- Peak joint force over limit: 0.58 at most (waist pitch). No joint above its limit.
- The drive asked for more than the limit in 9.4 % of frames, all left ankle pitch. In those frames the motor sat at its limit.
- No contact-depth check was run.

## Frames read
- Live, rendered inside the sim process, right-hand close-ups at t 10.0, 10.6, 11.0, 11.4, 11.8, 12.2, 12.6, 13.0 (pinch and lift), t 15.5, 19.0, 21.6 (carry), t 25.0, 27.2, 28.0, 29.5, 33.7 (lowering, release, end). Top-down frame at t 33.7: the bag lies inside the mug cavity (bag_in_mug_topdown_t33.7.png). These live frames come from the sim process itself (tools\live2.py, tools\ops2.py), not from the film renderer.
- teabag_F_film.mp4 at recording t 10.0, 12.2, 15.0, 21.1: frames\sheet_F1.jpg. Hand in the tray, bag in the pinch above the tray with string and tag hanging, bag held above the mug.
- teabag_F_film.mp4 at recording t 26.7, 28.0, 33.0: frames\sheet_F2.jpg. Bag lowered into the mug, hand open above the mug, hand withdrawn with the red tag hanging outside the mug wall. From this camera the bag inside the mug is hidden by the mug wall.
- NOT SEEN: every frame between those times.

## Files
teabag_F_rec.npz (30 Hz, 22 props: bag, 20 drawn string points, tag), teabag_F_film.mp4 (fixed camera, same as the MuJoCo clip, real speed), teabag_F_eye.mp4 (head camera), metrics_F.json, match_pair_F_mujoco_left_isaac_right.png.

## Earlier take on GPU PhysX: FAILED (teabag_A files)
Take bgA1, GPU PhysX, one run from t 13.0 to 22.3 with the bag already pinched. The hand lifts and carries the bag and holds it until t 21.0. Between t 21.0 and 21.4, as the hand starts to lower, the bag drops out of the pinch and ends on the counter at (1.786, 2.164, 0.862), beside the mug. Holding the torque of t 14 at 0.6, 0.8 and 1.0 of its value still dropped the bag at t 21 to 22; at 1.5 it dropped at about t 18; grip x2.5 squeezed the bag out at the start. The recorded 20-segment string went NaN on GPU PhysX at 5 mg per segment and stood stiff at 20 mg, which is why the string is a distance joint. Files: teabag_A_rec.npz, metrics_A.json, match_pair_A_mujoco_left_isaac_right.png (t 21.6). NOT SEEN: the instant of that drop.
