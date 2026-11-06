# Disclosures

Everything in this episode that was added, modelled, assumed, changed along the way, or that fails. If a line here makes the data point worthless for your purpose, it should do so before you spend time on it.

## The short version

- **The strict verdict cannot be PASS with this code.** The kettle step's strict status still requires a tilt the pour never reaches (section 4). This is a rule left over from an older pour, and it was deliberately not edited.
- Strict slip and strict torque failed in every step test listed in section 4. Only one isolated rack grasp (`gpu_grip01`) has passed all six gates.
- Task-level checks are separate from strict gates, and they are a working rule that has not been independently confirmed.
- The powder is 300 rigid spheres. The tea does not steep, heat, or mix.
- The controller reads ground truth from the simulator. There is no perception.
- This is one episode in one scene. It was tuned on that scene, run by run.

## 1. Added to the scene

None of these exists in the source scene. All are built in the session layer at startup; the source USD is not changed.

| Item | What it is | Why | Code |
|---|---|---|---|
| Moulded scoop | One-piece free rigid body, 45 g. Bowl 20 mm inner radius, 28 mm deep. Long tapered handle: 130 mm from bowl axis to mid-grip, thicker grip section, 40 mm tail. The tail deepens into a slim moulded heel so that the scoop stands level with the grip off the counter (`P_SPAWN_MODE=foot`). Collision is convex pieces, with the same floor and 16-wall bowl as the earlier cup tool | The earlier tool had no handle, so the hand had to go to the jar mouth (66 mm opening) and the fingers hit it. All GPU powder failures traced to this. It replaced that cup tool, which itself replaced a thin spoon the hand could not carry through a wrist turn | `code/steps/powder/pw_scoop.py` |
| Jar lid | New free rigid body, 30 g, top disc and 20-box skirt, resting on the neck with 3 mm radial clearance. **No thread and no joint.** The real jar has a screw lid; unscrewing is not modelled | The source jar is one rigid body with no lid | `code/steps/powder/pwprops.py` |
| Lid knob | Added 30 x 20 x 48 mm grab block on the lid | The hand cannot span the 77 mm rim without shoving the lid | same |
| Jar interior | Solid hull replaced by floor, 16 wall boxes, shoulder and neck rings, and a rigid packed fill with a funnel crater | So grains can sit in a jar | same |
| Tea bag and tag | Two rigid boxes: folded bag 65 x 25 x 4 mm, 2 g; card tag 25 x 20 x 1 mm, 0.5 g | Stand-in, see section 3 | `code/props/props.py` |
| Stir spoon | Rigid steel spoon, solid ellipsoid bowl | Source scene has none as a free body | same |
| Powder spoon | A second rigid spoon in the tub, not used to scoop. It stays in the scene | Left over from the earlier design | same |

## 2. Assumptions

| Assumption | Value | Status |
|---|---|---|
| Hand pad friction | 1.0, static and dynamic, on 26 hand links (`PAD=1`) | **Assumption.** Not measured from a real hand. The real hand has rubber tips. Grips depend on this value |
| Finite fingertip contact | Not used in the final run path: `pad_radius` defaults to 0 | Earlier tests used a 3 mm torsional patch; do not read those as this configuration |
| Object friction | Steel 0.4, bag paper 0.6, tag card 0.9, jar glass 0.5, jar fill 0.7, grains 0.7 | Chosen, not measured |
| Object mass | Scoop 45 g, lid 30 g, bag 2 g, tag 0.5 g, jar plus fill 0.55 kg less the mass moved into grains | Chosen, not measured |
| Water parameters | See README | Tuned for a stable PBD fluid. Viscosity and cohesion are not calibrated against water |
| Task pass rules | 25 mm mug placement, 100 ml serving, 20 grains, and so on | Working rule, not confirmed. 20 grains is a working target |

## 3. Stand-ins and models

- **Powder: 300 rigid grains.** Rigid spheres, radius 2.5 mm, 600 kg/m3, 39 mg each, total 11.8 g, friction 0.7. This is a powder model only in the loosest sense: real milk powder is about two orders of magnitude finer and cohesive. The first model had 120 loose grains over a rigid fill and transferred nothing, so 180 more grains' worth of mass was moved from the rigid fill to loose grains (total preserved). That is an explicit model change made to get grains into the scoop.
- **Grain angular damping 60, sleeping off.** PhysX has no rolling resistance, so angular damping stands in for it. The value was raised from 8 because grains rolled out of the lifted scoop at 8. This is a tuning choice in the direction of success.
- **Tea bag string.** Not a string: a `UsdPhysics.DistanceJoint` with a 12 cm maximum between bag and tag, slack below that, no geometry, nothing to snag. This is the only joint between free objects, and it does not touch the robot.
- **Water** is PBD particles at room temperature. Nothing boils, heats, steeps or dissolves. Grains stay rigid in water. "Milk tea" here means water, a rigid bag and rigid spheres in a mug.
- **Kettle.** The solid hull is disabled and replaced by thin box walls under the same rigid body, so water can be inside it. The handle hulls stay.
- **Tubs.** Solid blocks in the source scene, replaced by a floor and four walls.
- **Kettle switch and power.** Not modelled. The kettle is set down beside the mug, not returned to its base.
- **Convex hulls** of 34 scene colliders are limited to 32 vertices so that particles collide with them on the GPU, which changes their shape slightly. Above the limit, particles ignored those hulls.
- **Robot self-collision is off** (`allow_self_collision=False` at import). Arms can pass through the body. This must be addressed before any hardware or training claim.

## 4. Strict gates that fail

Gates are in `code/props/gates.py`, checked at 600 Hz, with these limits: fall, pelvis at or above 0.6 m and tilt at most 55 deg; torque ratio at most 1.0; penetration at most 2 mm; slip at most 5 mm; moved under 5 mm; spill at most 3 ml. **No limit was raised to make anything pass.**

Final-run values are filled in by `tools/collect_run.py`:

<!-- FINAL-GATES:BEGIN -->
**NOT FILLED IN. The final run has not finished.**
<!-- FINAL-GATES:END -->

Values from earlier **test setups only, not the final run**:

| Gate | Limit | `cup_gpu03` | `s_gpu05` | `p2_gr3` | `kl03` |
|---|---|---|---|---|---|
| Slip, mm | 5 | **13.2** | not measured | **57.2** | **69.5** |
| Torque ratio, projected | 1.0 | **2.83** left ankle pitch, 169.9 Nm | **2.00** right ankle pitch, 120.2 Nm | **3.07** left index proximal | **50.4** right pinky proximal |
| Steps over limit | 0 | 74 | 18 | 2621 | 4748 |
| Longest run over, steps | 0 | 2 | 1 | 4 | 155 |
| Motor estimate ratio | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| Spill, ml | 3 | **3.2** | **6.5** | not measured | **93.2** |

Bold fails its limit. `kl03` predates the rocking fix in section 5.

**Slip.** The mug moves up to about 13 to 15 mm in the hand during carries, and held tools move further. The 5 mm limit has been met only in the isolated rack grasp, not in any longer test. Slip here means a contact grip that shifts in the hand, so it is a real property of these grasps, not a measuring artefact.

**One-step torque spikes.** The gate compares `get_dof_projected_joint_forces` with URDF limits, and by default any single sample over the limit fails (`SPIKE=0`). Foot impacts give spikes of 1 to 4 steps far above what the clamped drive can deliver, while the motor estimate (clipped PD plus actuation force) stays at or under 1.0. So for the leg joints this is most likely contact impulse rather than motor torque. **That explanation does not cover everything:** hand joints stayed over the limit for 155 steps in `kl03` and 441 steps in an earlier run. Read the two numbers apart, and do not substitute one for the other. A looser gate, `SPIKE=6`, exists and was reported beside the strict one in some tests; it is off here.

**Stale kettle rule: `pour_tilt_reached >= 85`.** The kettle stage's strict `status` still requires a pour tilt of at least 85 deg (`code/steps/kettle/stage_live.py`). That came from an older pour with less water. The current pour is capped at 72 deg (`tilt_cap`), because the hand loses the kettle past about 80 deg, and it reached 69.3 deg in test `kl07`. So **the kettle's strict status is FAIL by construction, and `full_episode_passed` cannot be true.** The `task_status` does not use this rule. It was left in place on purpose, because changing a gate during the push was forbidden. It is a stale rule, not a physics finding.

The other strict step statuses are also stricter than their task checks: the kettle's includes the 5 mm slip limit, and the mug's needs 5 mm placement.

## 5. Changes made along the way

**Measurement**

- **Spill measurement fix in `code/fluids/fluids.py`.** A particle sample can be up to 0.2 s old when it is read, and `report()` compared it with the current kettle and mug pose. A moving mug therefore read as spill: `cup_gpu02` reported 113 ml spilled with 109.9 ml still in the mug. The fix stores the kettle and mug pose taken at the same instant as the sample. The limit did not change. Because this fix lowers reported spill, you can check it: recompute spill from the particle file and the recorded poses.
- Batched reads of watched bodies and a per-tick link cache were added for speed. Checked equal to direct reads over 120 ticks; gates remain at 600 Hz.

**Control and setup**

- **Pour servo gain 0.12 to 0.04 from `over_mug` onward (stage `k15`).** The loaded arm rocked about 1.4 deg with a 0.6 s period. Lowering the correction gain took tip swing from 34.7 mm to 1.2 mm in 3 s in a frozen test.
- **Flow-held tilt (stage `k16`).** After the ramp, tilt rises slowly when the measured flow into the mug is under 6 ml/s and backs off above 10 ml/s, capped at 72 deg. The serve target went from 85 to 97 ml. **This uses the live particle count as feedback**, which a real robot would not have.
- Kettle fill went from 400 ml to 700 ml, because water reached the spout only past 80 deg at 400 ml and the hand loses the kettle there.
- Hand gains change in stages: thumb yaw stiffness 10 during grips, finger gains in `arm_servo.configure_fingers`. Arm damping is 2.5 times the base value. Arm gravity compensation is fed forward as joint effort, with the drive limit reduced by the same amount so that the URDF cap holds.
- Stance, approach paths, grip offsets and crouch depths were tuned on this exact scene over many runs. Stance varies 4 to 7 cm and the stages tolerate that, but they are not general skills.

**Process**

- **Stages bind late.** `steps/<step>/stage_live.py` and `cup_walk_away.py` are read when the step starts, not at launch. In the final run, a stage for a later step could be replaced while the run was already on an earlier step, and the powder stage was. So the run was not one frozen code version. `runs/<tag>/CODE_PROVENANCE.json` compares every released file with the launch snapshot and records what changed.
- The final run was started with `P_SCOOP=real` in the environment. Any other environment value is assumed to be the default; that cannot be read back from a running process.
- Debug tools that froze a session and ran live commands were used in tests. They are not in this repo, and they were not used in the final run's launch. Tests under them, such as `kl07`, are labelled as such.
- Physics crashed with a GPU launch failure twice, about 30 s after another simulator process started, and a power loss ended `final05`. Other test processes were running on the same machine during the final run.

## 6. Render differences

A video is a replay of recorded poses, rendered afterwards. It is not a live physics view, and what it shows differs from what was simulated:

- **Drawn but not simulated.** Physics runs on the plain twin. Look layers add set dressing at render time only: sachets, packaging, textures and room details. These have no collision, or differ from the collider. Do not judge contact from a video.
- **Simulated but not drawn.** Thin collider walls in the kettle, jar and tubs are invisible (`guide` purpose or hidden). A hand can look as if it stops on nothing, or an object can look unsupported.
- **Stale or wrong copies of objects.** Without `P_SCOOP=real` at render time the renderer builds the old cup tool and poses it from the new scoop's recording. Scene objects that are not in the recording stay at their spawn pose even if physics moved them; only `object_0295` to `object_0321` and the props are recorded.
- **Water** is drawn as a point cloud from 10 Hz samples, held between samples, so it steps at 30 fps. Particles inside the kettle are hidden.
- **Grains** are drawn as points of fixed width, not as spheres of their real size.
- Hand close-up cameras follow smoothed hand positions.

## 7. Known limitations

- Single episode, single scene. No statistics. A repeat will differ.
- No perception, and the controller uses privileged state.
- Recordings store poses and joint positions only. **Joint velocities, commanded targets, torques, contact forces and camera images are not recorded**, so this is not yet a complete imitation-learning sample.
- Not validated on hardware. No sim-to-real claim of any kind.
- An earlier fall gate missed a robot leaning on the counter. A tilt limit was added, but check frames as well as the gate.
- Penetration only sees pairs with a body in the sensor list; sleeping bodies report no contacts.
- The twin differs from the real room in places, including the tray layout and the jar shape.
- Sequence and timing are slower than a person.
- Isaac Sim 6.2.0-rc.14 is a release candidate, and results may not carry to a final release.

## Final chain, 6 Nov 2026

The full episode is joined from five saved runs: `final08`, `f11`, `f13`, `f17`, `f20`. Each restarts from the saved state of the run before it. Only the last join, at t 465.9, falls inside a step, with the spoon held; `f20` commands the recorded squeeze before physics resumes. It is not one single full run, and `full_episode_passed` stays false in the result files.

Results: mug, tea bag, kettle (102.3 ml, 1.1 ml spilled), powder (35 grains counted in the mug; grains in the water are not visible in frames), stir (2.4 full turns at the bowl, mug moved 0.0 mm), spoon on the counter, cup carried to the exit (102.3 ml, 0.0 ml lost in the carry). Task checks pass for these steps, reading the stir task across `f17` and `f20`: `f17` holds the stirring and `f20` holds the put-down, and `f20`'s stir step PASS covers only the put-down. In the cup step, strict fall, penetration, moved and spill pass; strict torque (2.96) and slip (18.26 mm) fail. Gates were not reported for every run.

Live corrections, AI-operated, during `f17` and `f20`:

- The hand was rolled to the handle as found, and the fingers were tipped 7 then 10 degrees about the pinch line, after the wrist pressed the tray with 62 N.
- That correction dragged the tray 16 mm. The spoon moved with it and resettled; its pose was re-read and the approach repeated.
- At the put-down the little finger pressed the tray with 36 N. The hand was lifted back, the spoon levelled, and a new spot chosen. In `f17` that spot was past the counter edge and the spoon fell. `f20` restarts before that release and uses y 2.09.
- First lift slowed from 1.0 s to 2.5 s.

Collision audit of the render meshes at recorded poses, sampled at 5 Hz:

- The right elbow, hand and wrist pass into the robot's own torso and pelvis by up to 38.5 mm, faintly visible in a frame. Self-collision was not acting.
- The right knee and hip pass through a cabinet divider by up to 51 mm, below the counter and not seen in frames.
- The palm and little finger overlap the render kettle by up to 9.8 mm; its physics shape is simpler than its render mesh (colliders 1.1 to 1.7 mm).
- Soles sit 2.36 mm into the floor at all times, which is a constant offset.
