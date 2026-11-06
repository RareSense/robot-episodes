"""The support matrix: what each engine does with each phenomenon, as data.

Every claim a package makes about itself — the README, the limitations list, the
per-engine status a creator sees before downloading — is generated from this table
and from what the build actually ran. Nothing here is hand-written per build, so a
phenomenon the scene uses on an engine that cannot realise it is always reported as
"declared, not simulated" instead of being forgotten in prose.

Statuses: `native` — the engine simulates it with its own primitive; `approximated` —
a documented stand-in; `unsupported` — exported as data only, nothing moves.
"""

ENGINES = {
    "mujoco": "MuJoCo",
    "pybullet": "PyBullet",
    "isaac_sim": "Isaac Sim",
    "gazebo": "Gazebo",
}

# phenomenon -> (label, {engine: (status, detail)})
PHENOMENA = {
    "rigid_bodies": ("Rigid bodies with mass and inertia", {
        "mujoco": ("native", "body with inertial"),
        "pybullet": ("native", "URDF body per movable object"),
        "isaac_sim": ("native", "RigidBodyAPI + MassAPI"),
        "gazebo": ("native", "non-static model per movable object"),
    }),
    "passive_joints": ("Hinges and sliders to the world", {
        "mujoco": ("native", "hinge/slide joint, soft limits"),
        "pybullet": ("native", "revolute/prismatic joint, hard limits"),
        "isaac_sim": ("native", "Revolute/PrismaticJoint, hard limits"),
        "gazebo": ("native", "revolute/prismatic joint, hard limits"),
    }),
    "joint_friction": ("Joint dry friction", {
        "mujoco": ("native", "frictionloss"),
        "pybullet": ("approximated", "zero-velocity motor capped at the friction force"),
        "isaac_sim": ("approximated", "physxJoint:jointFriction, PhysX-specific"),
        "gazebo": ("native", "axis dynamics friction"),
    }),
    "convex_contact": ("Convex contact for moving parts", {
        "mujoco": ("native", "convex pieces or hull"),
        "pybullet": ("native", "convex pieces or hull"),
        "isaac_sim": ("native", "convexHull approximation"),
        "gazebo": ("native", "convex_hull optimization"),
    }),
    "static_mesh_contact": ("Exact triangle contact for fixed parts", {
        "mujoco": ("approximated", "fixed parts collide as convex pieces or hulls"),
        "pybullet": ("native", "concave triangle mesh, or the authored pieces"),
        "isaac_sim": ("native", "MeshCollisionAPI approximation none"),
        "gazebo": ("native", "triangle mesh collision"),
    }),
    "restitution": ("Bounce (restitution)", {
        "mujoco": ("unsupported", "no restitution parameter; solver defaults apply"),
        "pybullet": ("native", "restitution"),
        "isaac_sim": ("native", "physics material restitution"),
        "gazebo": ("native", "bounce restitution_coefficient"),
    }),
    "rolling_friction": ("Torsional and rolling friction", {
        "mujoco": ("native", "friction triple, condim 4/6"),
        "pybullet": ("native", "spinning and rolling friction"),
        "isaac_sim": ("unsupported", "recorded in scene_spec.json only"),
        "gazebo": ("unsupported", "recorded in scene_spec.json only"),
    }),
    "prescribed_motion": ("Prescribed kinematic motion (motion_program.json)", {
        "mujoco": ("approximated", "position/velocity actuators + mocap bodies; run_mujoco.py "
                                   "or the Studio preview evaluates the program at physics time"),
        "pybullet": ("approximated", "run_pybullet.py plays joint drives and kinematic poses "
                                     "from the program at physics time"),
        "isaac_sim": ("native", "time-sampled drive targets and kinematic-body transforms"),
        "gazebo": ("approximated", "kinematic links and unlimited joints; run_gazebo.py "
                                   "plays the program at simulation time"),
    }),
    "terrain": ("Elevation-grid terrain", {
        "mujoco": ("native", "hfield geom"),
        "pybullet": ("native", "GEOM_HEIGHTFIELD"),
        "isaac_sim": ("native", "the grid triangulated, exact static mesh"),
        "gazebo": ("native", "the grid triangulated, exact static mesh"),
    }),
    "water": ("Water buoyancy and drag", {
        "mujoco": ("approximated", "nova_water.py hook: displaced-weight lift + blunt drag"),
        "pybullet": ("approximated", "nova_water.py hook in run_pybullet.py"),
        "isaac_sim": ("unsupported", "region recorded in USD custom data only"),
        "gazebo": ("unsupported", "region recorded in scene_spec.json only"),
    }),
    "textures": ("Image textures", {
        "mujoco": ("unsupported", "solid colours"),
        "pybullet": ("unsupported", "solid colours"),
        "isaac_sim": ("native", "UsdUVTexture"),
        "gazebo": ("unsupported", "solid colours"),
    }),
    "controllers": ("Bundled controllers", {
        "mujoco": ("native", "Go2 policy or scripted household mission"),
        "pybullet": ("unsupported", "passive only"),
        "isaac_sim": ("unsupported", "passive only"),
        "gazebo": ("unsupported", "passive only"),
    }),
    "fluids": ("Fluid dynamics", {e: ("unsupported", "no solver") for e in ENGINES}),
    "deformables": (
        "Cloth and soft bodies", {e: ("unsupported", "rigid stand-ins") for e in ENGINES}
    ),
    "thermal": ("Heat and temperature", {e: ("unsupported", "task values only") for e in ENGINES}),
    "sensors": ("Sensor simulation", {e: ("unsupported", "not exported") for e in ENGINES}),
}


def used_phenomena(scene):
    """Which rows apply to this scene, judged from the description, not from prose."""
    used = {"static_mesh_contact"}
    if any(obj.body is not None for obj in scene.objects):
        used.update({"rigid_bodies", "convex_contact"})
    if any(
        obj.body is not None and obj.body.motion in {"hinge", "slide"}
        for obj in scene.objects
    ):
        used.add("passive_joints")
    if any(obj.body is not None and obj.body.drive is not None for obj in scene.objects):
        used.add("prescribed_motion")
    if any(obj.body is not None and obj.body.frictionloss for obj in scene.objects):
        used.add("joint_friction")
    if any(obj.restitution for obj in scene.objects):
        used.add("restitution")
    if any(obj.torsional_friction or obj.rolling_friction for obj in scene.objects):
        used.add("rolling_friction")
    if any(obj.heightfield is not None for obj in scene.objects):
        used.add("terrain")
    if scene.water_volumes:
        used.add("water")
    if any(
        getattr(getattr(obj.mesh.visual, "material", None), "baseColorTexture", None) is not None
        for obj in scene.objects
    ):
        used.add("textures")
    return used


def matrix(scene, exports, tested, controller=None):
    """Per-phenomenon, per-engine support for this build plus what was measured.

    `exports` are the engines that received files; `tested` maps an engine to the
    measured status of its motion checks (or None when nothing ran there).
    """
    used = used_phenomena(scene)
    if controller:
        used.add("controllers")
    rows = []
    for key, (label, engines) in PHENOMENA.items():
        rows.append({
            "phenomenon": key,
            "label": label,
            "used": key in used,
            "engines": {
                engine: {
                    "status": status if engine in exports else "not_exported",
                    "detail": detail,
                }
                for engine, (status, detail) in engines.items()
            },
        })
    return {
        "schema_version": "nova.support-matrix/v1",
        "engines": {
            engine: {
                "label": ENGINES[engine],
                "exported": engine in exports,
                "measured": tested.get(engine),
            }
            for engine in ENGINES
        },
        "rows": rows,
    }


def limitations(support):
    """Plain sentences for the README and the creator: what this package does not do,
    only for phenomena the scene actually uses."""
    sentences = []
    for row in support["rows"]:
        if not row["used"]:
            continue
        for engine, entry in row["engines"].items():
            if entry["status"] == "unsupported":
                sentences.append(
                    f"{row['label']}: declared, not simulated in {ENGINES[engine]} "
                    f"({entry['detail']})."
                )
            elif entry["status"] == "approximated":
                sentences.append(
                    f"{row['label']}: approximated in {ENGINES[engine]} ({entry['detail']})."
                )
    for entry in support["engines"].values():
        if entry["exported"] and entry["measured"] is None:
            sentences.append(
                f"{entry['label']}: files exported, nothing was run here; open them in your "
                "own installation."
            )
    return sentences


def engine_statuses(support, robot_engine=None, robot_included=False):
    """The download-side status per engine a creator reads before opening a file."""
    result = []
    for engine, entry in support["engines"].items():
        if not entry["exported"]:
            status, detail = "not_exported", "No files for this engine in this build."
        elif entry["measured"] == "acceptance":
            status, detail = "tested_here", "Built and motion-tested here; the acceptance engine."
        elif entry["measured"] == "compatibility":
            status = "checked_here"
            detail = "Loaded and motion-checked here as a compatibility opinion."
        else:
            status, detail = "exported_only", "Files exported; this engine was not run here."
        if robot_included and robot_engine and engine != robot_engine:
            detail += " The uploaded robot is native to another engine and is not in these files."
        result.append({
            "engine": engine, "label": entry["label"], "status": status, "detail": detail,
            "robot_included": robot_included and engine == robot_engine,
        })
    return result
