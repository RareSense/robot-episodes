# G1 tea-making sim demos (LeRobot v3.0)

5 whole-body demonstrations of a Unitree G1 (with 3-finger hands) making tea in a digital twin of a real kitchen,
physically simulated in MuJoCo (600 Hz, torque-limited motors), logged at 60 Hz. One episode per sub-task:

| episode | task |
|---|---|
| 0 | Take the mug from the rack. |
| 1 | Put the tea bag into the mug. |
| 2 | Pour hot water from the kettle into the mug. |
| 3 | Stir the tea in the mug with the spoon. |
| 4 | Pick up the cup of tea and walk away with it. |

## Features (per frame)

- `observation.state` - 43 joint positions (rad): legs, waist, arms, fingers
- `observation.velocity` - 43 joint velocities (rad/s), central differences of the 60 Hz positions
- `observation.base_pose` - floating base (pelvis) position (m) and quaternion (w x y z), world frame
- `observation.environment_state` - pose (x y z qw qx qy qz) of kettle, milk_powder_jar, mug, teabag_bag, spoon; zeros when absent
- `observation.object_present` - 1 if that object is in the episode's scene
- `observation.eye_camera_pose` - head camera position + 3x3 rotation (rows = camera axes in world; MuJoCo camera: -z forward, +y up)
- `observation.effort_abs` - |torque| of each of the 43 motors (N m). **Magnitude only** (the recorder kept absolute values)
- `action` - joint position targets: the 43 joint positions of the next frame (last frame repeats)

## Not included

- camera images (no RGB in this version)
- signed motor torques, IMU, contact forces
- variations: one demo per sub-task

Robot model, scene and the code that produced the runs: https://github.com/abdullahanxie/tea-robot-physics
