# Nova robotics package

Extract the entire ZIP and keep its directory together: native scene files reference meshes by relative path. Open only the entrypoint for your simulator.

| Simulator | Entrypoint | Status in this build |
|---|---|---|
| MuJoCo | environment.xml | built and motion-tested here |
| PyBullet | run_pybullet.py | loaded and motion-checked here as a second opinion |
| Isaac Sim | environment.usda | files exported; not run here |
| Gazebo | world.sdf | files exported; not run here |

Isaac Sim: open the .usda. MuJoCo: open the .xml with its viewer. Gazebo: load the .sdf — with moving obstacles run `python3 run_gazebo.py` and open the viewer with `gz sim -g`. MuJoCo with moving obstacles: run `python run_mujoco.py` (a bare viewer shows the world static). PyBullet: install pybullet, then run `python run_pybullet.py` (`--headless` for a bounded passive test).

Movable bodies: 4 hinge(s), 3 slider(s), 28 free prop(s); 360 static objects. Hinges and sliders are passive joints to the world with limits, damping and friction; free props fall, slide and can be picked up. No latches, closers or driven joints.

Environment mechanisms were settled and pushed in MuJoCo 3.13.0 (status passed). This is not a robot task, a training run or proof of behaviour in another engine.

No robot is included. Robot clearance, controllers and training are not validated.

## Limitations

- Joint dry friction: approximated in PyBullet (zero-velocity motor capped at the friction force).
- Joint dry friction: approximated in Isaac Sim (physxJoint:jointFriction, PhysX-specific).
- Exact triangle contact for fixed parts: approximated in MuJoCo (fixed parts collide as convex pieces or hulls).
- Isaac Sim: files exported, nothing was run here; open them in your own installation.
- Gazebo: files exported, nothing was run here; open them in your own installation.

## File map

In the Nova project, asset.py is the editable geometry source, out/model.glb is the browser preview, and out/simulation.zip is this entire directory as one download. Meshes, terrain grids and robot assets live only inside the ZIP. The preview uses Y-up; native scenes use metres and Z-up.

| Path in this package | Purpose |
|---|---|
| environment.usda | Open this in isaac_sim. |
| environment.xml | Open this in mujoco. |
| run_pybullet.py | Open this in pybullet. |
| world.sdf | Open this in gazebo. |
| meshes/, terrain/ | Visual meshes, collision parts and elevation grids the native scenes reference. |
| scene_spec.json | The world description: named objects, physical properties with their sources, joints, terrain, water, support matrix. |
| validation.json | Measured results, findings, limitations and checks that did not run. |
| build_configuration.json | Exact settings and overrides used for this build. |
| robot_inputs.json, robot/ | Uploaded robot inventory and files, when included. |
| nova_water.py, water_volumes.json | Buoyancy and drag hook for water regions, when declared. |
| manifest.json | Selected entrypoints, engine statuses and hashes of the packaged files. |

Native files can be edited in your simulator; rebuilding from asset.py produces a new export and does not merge those external edits.
