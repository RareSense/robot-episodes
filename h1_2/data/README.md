# H1-2 saved runs

Results here come from these saved runs. Final results will be added.

The runs form a chain. Each one restarts from the saved state of the run before it, so this is not one single take. Recording format: [docs/DATA_FORMAT.md](../docs/DATA_FORMAT.md). Loader: [examples/load_rec.py](../examples/load_rec.py).

| Run | Covers | Used in the chain, sim s | Recording spans, sim s | Result in the run's log |
|---|---|---|---|---|
| `final08` | Walk in, mug, tea bag | 0.5 to 135.9 | 0.5 to 437.8 | Recording is on Drive, over the repo's file size limit |
| `f11` | Kettle pour | 135.9 to 272.3 | 135.9 to 350.0 | 102.3 ml in the mug, 1.1 ml spilled |
| `f13` | Milk powder | 272.3 to 383.8 | 272.3 to 392.5 | 35 grains in the mug, none spilled |
| `f17` | Stir | From 383.8 | 383.3 to 481.9 | Spoon lifted and stirred, mug moved 0.0 mm. Spoon not yet back on the counter, so the stir step's task check reads FAIL |
| `f16` | Cup step, a separate continuation from 383.8 | From 383.8 | 383.3 to 484.6 | Cup step task PASS. Its stir step reads FAIL, spoon not lifted |
| `f20` | H12_FILL_f20 | H12_FILL_f20 | H12_FILL_f20 | H12_FILL_f20 |

`f16` and `f17` both start from `f13` at the stir, so they are two branches, not one line.

## Files

Each `data/<run>/` folder holds the stage log and the recording `<run>_rec.npz`. `f16` also has its step report and gate report. The large files are on Drive.

| File | What it is | Link |
|---|---|---|
| `final08_rec.npz` | Pose recording of `final08`, 114 MB | [Download](https://drive.google.com/file/d/1iBlajLGsc7xGyQzvELRo2MeHc3-Cjgln/view) |
| `final08_rec.npz.particles.f32` | Water particles of `final08`, 4.4 GB | [Download](https://drive.google.com/file/d/1bq-s-fAfFuDwHO3GFLhzPUbBUwY7duLm/view) |
| `f11_rec.npz.particles.f32` | Water particles of `f11`, 1.0 GB | [Download](https://drive.google.com/file/d/1YuCYrOJrsW1MrnE-CobdRDCA64ssE82D/view) |
| `f13_rec.npz.particles.f32` | Water particles of `f13`, 591 MB | [Download](https://drive.google.com/file/d/1Wu5eDCXfcRm8-3c3QVItT6HdSlLH6GDu/view) |
| `f16_rec.npz.particles.f32` | Water particles of `f16`, 487 MB | [Download](https://drive.google.com/file/d/17nA5TzbI5YG_qYzPAP_BNTY1BQ68j45-/view) |
| `f17_rec.npz.particles.f32` | Water particles of `f17`, 535 MB | [Download](https://drive.google.com/file/d/1blHlUQpLRynChChNFwEoujWnZNh0aP4Z/view) |

Put a particle file next to its `<run>_rec.npz` to load the water.

[Open the H1-2 data folder on Google Drive](https://drive.google.com/drive/folders/1dm9UartuNUNqptnRVfjWUEutjQPI-oJ6)
