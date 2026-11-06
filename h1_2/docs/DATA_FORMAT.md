# Data format

Checked against real recordings of this project with `examples/load_rec.py`. World frame: metres, Z up, the robot faces +Y at the counter.
Poses are 7 numbers: `x, y, z, qx, qy, qz, qw`, scalar last.

## `<tag>_rec.npz`: poses at 30 Hz

Written by `code/props/record.py`. Load with `numpy.load(path, allow_pickle=False)`.

| Key | Type, shape | Meaning |
|---|---|---|
| `t` | float64 `(F,)` | Sim time of each frame, seconds |
| `fps` | float64 scalar | 30.0: one frame every 20 physics steps |
| `link_paths` | unicode `(L,)` | USD prim path of each robot link. L = 55 |
| `link_T` | float32 `(F, L, 7)` | Link poses, in the order of `link_paths` |
| `body_paths` | unicode `(B,)` | Free bodies, then the grains `/World/fluid/grains/gNNN` |
| `body_T` | float32 `(F, B, 7)` | Body poses, in the order of `body_paths` |
| `p_t` | float64 `(S,)` | Sim time of each water sample, seconds. Only with water |
| `p_n` | int64 `(S,)` | Points in each water sample. Only with water |

Bodies of interest: `/World/Objects/object_0316` the mug, `object_0295` the kettle, `object_0309` the jar, `/World/Props/jar_lid`, `/World/Props/powder_scoop`, `/World/Props/teabag_bag`, `/World/Props/teabag_tag`, `/World/Props/stir_spoon`.
Only free bodies near the tea corner are recorded (`object_0295` to `object_0321`, and the props). Sim time starts after a short settle, so `t[0]` is not zero.

The file is rewritten about every 15 sim seconds during a run and at each step's end, so a copy taken from a running or killed run is a valid but shorter recording.

## `<tag>_rec.npz.particles.f32`: water positions at 10 Hz

Raw little-endian float32, no header. Each sample is `p_n[i]` points of `x, y, z`, in world metres, and the samples follow each other in order:

```python
off = np.r_[0, np.cumsum(p_n)][:-1]
P = np.memmap(path, dtype="<f4", mode="r").reshape(-1, 3)
sample_i = P[off[i]:off[i] + p_n[i]]
```

- The file size must equal `p_n.sum() * 12` bytes for a recording saved at the end of the run. `particles.ref.json` records this check as `size_matches_rec`. If the recording was saved earlier than the last sample, the file holds extra samples at its end, which the offsets above ignore.
- A sample that failed a finite-value check was skipped, so `p_t` can have a gap. `p_n` still indexes the file correctly.
- All particles are in every sample, wherever they are: 80179 in the full episode, each 0.00873 ml. Particles that left the scene can be far away (a `z` of a few metres below the floor has been seen).
- At 80179 particles this is 9.6 MB per sim second.
- Grains are not in this file. They are in `body_T`.

### Recomputing water amounts

`code/fluids/fluids.py` `Fluids.report` holds the rule. For a sample and the mug and kettle pose at the same instant, in each object's frame: in the mug means within 33.5 mm of the axis, above the base and below the rim plus 4 mm. Inside the kettle is a cylinder plus a spout box. Spill is whatever is in neither and below the mug rim. Multiply counts by 0.00873 ml. Poses from `body_T` are at 30 Hz, so take the frame nearest `p_t[i]`; that is close to but not the same as the live reading.

## `<tag>.npz`: joint positions at 30 Hz

Written by `code/full_episode.py` when the run writes its final report, so a run that is killed has none.

| Key | Type, shape | Meaning |
|---|---|---|
| `t` | float64 `(F,)` | Sim time |
| `links` | float32 `(F, L, 7)` | Link poses |
| `q` | float32 `(F, 51)` | Joint positions, radians, in the order of `dof_names`. 51 includes the 12 mimic joints |
| `mug` | float32 `(F, 7)` | Mug pose |
| `root` | empty | Never filled. Use `links[:, 0]`, the pelvis |
| `link_names`, `dof_names` | unicode | Names |

## `<tag>_gates.json`: strict gates

`overall` (PASS, FAIL, INCOMPLETE), `steps_checked`, `sim_time`, and `gates` with `fall`, `torque`, `penetration`, `slip`, `moved`, `spill`. Each has `status` (PASS, FAIL, NOT MEASURED), `value`, `unit`, `limit`, `samples`, `note`, plus extra detail such as `torque.joint`, `torque.longest_run_steps`, `torque.motor_estimate_ratio`, `slip.per_object`, `moved.per_object`. A gate with no samples is NOT MEASURED, never PASS.

## `<tag>.json`: step report

`result`, `sim_s`, `wall_s`, `overall`, `full_episode_passed` (the strict verdict), `task_done` (the working-rule verdict, with its checks), `task_steps`, `required_steps`, `simulation` (mode, friction, continuous, water), `gates` (including `steps`, one entry per step with `status`, `task_status`, numbers and water), `stages`, `physics_gates` (same as the gates file).

`<tag>_partial.json` is the same `gates` and `stages` written after each step, so a run that dies still leaves step numbers.

## Collected run files

| File | Meaning |
|---|---|
| `RUN_STATUS.json` | Kind of run from its own report: `complete_full_run`, `stopped_full_run`, `test_setup`, `incomplete`. `is_final_run_result` is true only for the first |
| `RESULTS.md` | Tables generated from the JSON |
| `particles.ref.json` | Particle file size, sha256, sample count, URL once hosted |
| `CODE_PROVENANCE.json` | Each released code file against its launch hash |
| `MANIFEST.json` | sha256 of everything in the run folder |

Logs and JSON in `runs/` are sanitised copies: folders are replaced by `<path>/`, and file names are kept. The `.npz` files are unchanged copies.

## Loader

```
python examples/load_rec.py runs/<tag>/<tag>_rec.npz --particles <folder>
```

```python
from examples.load_rec import Recording, tilt_deg
r = Recording("runs/<tag>/<tag>_rec.npz", "<folder>/<tag>_rec.npz.particles.f32")
mug = r.body("object_0316")          # (F, 7)
hand = r.link("L_hand_base_link")    # (F, 7)
grains = r.grains()                  # (F, G, 3)
water = r.water_at(70.0)             # (N, 3)
```
