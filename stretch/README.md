<div align="center">

# Stretch pizza

A Stretch 3 puts a pizza in an oven and takes it out, in one continuous take

<!-- STRETCH_FILL: hero image, videos/previews/stretch_hero.jpg, linked to its Drive video -->

</div>

A Stretch 3 takes a plate with a pizza, carries it past the oven and sets it down, opens the oven door, places the plate on the rack and closes the door. It turns three knobs, waits, opens the door, takes the plate out and carries it back.

| | |
|---|---|
| Robot | Hello Robot Stretch 3 |
| Simulator | Isaac Sim, CPU PhysX |
| Operation | AI-operated |
| Episode | One continuous take, about 20 minutes of sim time |
| Recording | 30 samples per sim second |
| Camera views | Wrist and fixed |

<!-- STRETCH_FILL: Stretch GIF previews from the pz_final03 views, each linked to its Drive video -->

[All Stretch videos](videos/README.md)

---

## Data

### `data/`

NumPy `.npz`, one sample every 1/30 of a sim second.

| Key | Shape | Meaning |
|---|---|---|
| `t` | N | Sim time, seconds |
| `names` | 20 | Joint names |
| `links` | 37 | Link names |
| `q`, `dq` | N x 20 | Joint positions and velocities |
| `qd`, `vd` | N x 20 | Commanded position and velocity targets |
| `root` | N x 7 | Base pose: x y z, quaternion x y z w |
| `link` | N x 37 x 7 | Pose of every link |
| `oven` | N x 5 | Oven joints, door first, radians |
| `plate`, `pizza` | N x 7 | Object poses |

<!-- STRETCH_FILL: add the extra pz_final03 keys (oven body pose, heat state) and the view-frame index -->

`lerobot/` holds the same take as a LeRobot v3.0 dataset with camera video.

---

## Load the data

Run these from this folder.

```bash
python examples/load_stretch.py data/pz_final03_rec.npz
python examples/load_lerobot.py lerobot
```

```python
import numpy as np

d = dict(np.load("data/pz_final03_rec.npz"))
print(d["t"].shape, d["q"].shape, d["plate"][-1])
```

---

## Layout

```
videos/    video index and previews
data/      full recording
lerobot/   LeRobot v3.0 dataset
code/      task, oven, kitchen, recorder, replay and export code
examples/  data loaders
```
