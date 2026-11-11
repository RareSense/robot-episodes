# Tea bag into the mug. Isaac Sim PhysX, AI-operated

Status: NOT DONE. The bag does not reach the mug. Take bgA1, GPU PhysX, 600 Hz, one continuous run from t 13.0 to 22.3 (9.3 s).

- What works (seen in frames): starting from the recorded state with the bag already pinched, the hand lifts it and carries it over the mug, and it holds until t 21.0.
- FAIL: between t 21.0 and 21.4, as the hand starts to lower, the bag drops out of the pinch and hits the counter. It ends at (1.786, 2.164, 0.862) with the tag at (1.779, 2.266, 0.860), next to the mug and not in it. In the recording the bag ends in the mug at (1.720, 2.239, 0.927).
- NOT DONE: picking the bag from the tray (t 0.5 to 13) was not tried, because the spoon's pinch failed the same way. Lowering and release into the mug (t 21 to 28) were never reached.

## What was tried live
- Grip x1.0 was not tried on the bag. x1.5 holds to t 21. x2.5 squeezed the bag out at the start, landing at (2.359, 2.061, 0.862).
- Holding the torque of t 14 at 0.6, 0.8 and 1.0 of its value all still dropped the bag at t 21 to 22. At 1.5 it dropped sooner, at about t 18. So the drop at t 21 does not depend on grip strength in that range. Cause NOT established. Frames show the bag pivoting about the pinch in the first 0.3 s, then hanging at a different angle from the recording, and the numbers show it creeping 12 mm in the pinch from t 19 to 21 before it goes.
- Starting at t 21.5 from the recorded in-hand state: the bag is thrown out at once.
- String: the recorded 20-segment string went NaN on GPU PhysX at 5 mg per segment, and at 20 mg it stood stiff and levered the bag. This take uses a limp 12 cm distance joint between bag and tag, with no string geometry. The string in the video is drawn only, a curve from staple to tag. That is a difference from MuJoCo.

## Difference from the MuJoCo recording, t 13 to 22.3
- Body joints: max 0.013 rad. Fingers: max 0.039 rad.
- Left hand: max 7.9 mm, mean 2.3 mm. Right hand: max 14.1 mm, mean 3.8 mm. Pelvis: max 7.3 mm.
- Bag: within about 24 mm of the recording until the drop.
- Motors at their limit in 16.7 % of frames.

## Frames read
- teabag_A_film.mp4 at video t 1, 6, 8.6, 9.2 (recording t 14, 19, 21.6, 22.2) next to 2_teabag_film.mp4 at the same recording times: frames\sheet.jpg. Robot matches. At t 14 and 19 the bag is in the hand. At t 21.6 and 22.2 the clip still holds it, while in Isaac bag and tag lie on the counter.
- Live close-ups at t 13.3 to 17 and t 20 to 21.8, Isaac next to the recording: frames\close_1.jpg and close_2.jpg.
- frames\b6_t21.4_bag_and_tag_on_counter.png: the bag and tag on the counter.
- NOT SEEN: the instant of the drop. The closest frames are t 21.0 (held) and t 21.4 (already down).
- Do not use b5_end.png. It was rendered with the wrong scene and drew the bag's pose as a spoon.

## Files
teabag_A_rec.npz (30 Hz), teabag_A_film.mp4 (fixed camera, same as the clip, real speed), teabag_A_eye.mp4 (head camera; read at recording t 19 next to 2_teabag_eye.mp4 in frames\sheet_eye.jpg, close match, the hand hides the bag), metrics_A.json, match_pair_A_mujoco_left_isaac_right.png (t 21.6).
