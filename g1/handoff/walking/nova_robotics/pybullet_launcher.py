"""Nova passive robot + environment launcher. Install pybullet; python run_pybullet.py.

Static objects load as fixed concave meshes, or through their authored convex pieces
when the scene supplied them. Terrain loads as a native PyBullet heightfield built
from the same elevation grid MuJoCo uses. Movable objects (doors, drawers, lids, loose
props) load from their own URDF: hinges/sliders are passive joints on a fixed anchor,
free props are dynamic bodies. Push them in the GUI (ctrl+drag) or apply forces from
your own script. Water regions, if any, apply buoyancy and drag through nova_water.py.
No walking controller is supplied. --headless runs a bounded two-second smoke test.
"""

import argparse
import importlib.util
import json
import struct
import time
from array import array
from pathlib import Path


def read_heightfield(directory, spec):
    """Absolute heights in metres, row-major with rows along +Y, from the package's
    MuJoCo-format grid: an int32 row/column header then normalized float32 values."""
    payload = (directory / spec["file"]).read_bytes()
    rows, cols = struct.unpack_from("<ii", payload, 0)
    if (rows, cols) != (spec["rows"], spec["cols"]):
        raise ValueError(f"{spec['file']}: grid header does not match the scene description")
    values = array("f")
    values.frombytes(payload[8 : 8 + 4 * rows * cols])
    lowest, elevation = spec["lowest_m"], spec["elevation_m"]
    return rows, cols, [lowest + value * elevation for value in values]


def contact_properties(p, body, link, item):
    """Lateral, rolling and spinning friction, matching the MJCF's friction triple."""
    p.changeDynamics(
        body,
        link,
        lateralFriction=item["friction"],
        rollingFriction=item.get("rolling_friction", 0.0),
        spinningFriction=item.get("torsional_friction", 0.0),
        restitution=item["restitution"],
    )


def load_terrain(p, directory, item):
    """Exact ground: the collision heightfield plus the same grid as a visible mesh.

    PyBullet centres a heightfield on the midpoint of its own data range, so the body
    is offset by that midpoint to put the surface back where the scene authored it.
    """
    spec = item["heightfield"]
    rows, cols, heights = read_heightfield(directory, spec)
    size_x, size_y = spec["size_m"]
    center_x, center_y = spec["center_m"]
    shape = p.createCollisionShape(
        p.GEOM_HEIGHTFIELD,
        meshScale=[size_x / (cols - 1), size_y / (rows - 1), 1.0],
        heightfieldData=heights,
        numHeightfieldRows=cols,
        numHeightfieldColumns=rows,
    )
    middle = (min(heights) + max(heights)) / 2.0
    body = p.createMultiBody(
        baseMass=0, baseCollisionShapeIndex=shape, basePosition=[center_x, center_y, middle]
    )
    contact_properties(p, body, -1, item)
    # The visual surface is already in world coordinates and must not be offset.
    p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=p.createVisualShape(
            p.GEOM_MESH, fileName=str(directory / item["mesh"]), rgbaColor=item["color"]
        ),
    )
    return body


def load_static(p, directory, item):
    """A fixed object. It collides through the author's explicit convex pieces when it
    has them — the same pieces MuJoCo, Isaac Sim and Gazebo use — and otherwise as an
    exact concave mesh. A door frame's pieces are set back from the leaf; its visual
    mesh carries hinge knuckles and stop beads millimetres from the door, and colliding
    that would start every door inside its own frame."""
    mesh = str(directory / item["mesh"])
    pieces = item.get("collision") or []
    if not item["collidable"]:
        collision = -1
    elif pieces:
        collision = p.createCollisionShapeArray(
            shapeTypes=[p.GEOM_MESH] * len(pieces),
            fileNames=[str(directory / piece) for piece in pieces],
        )
    else:
        collision = p.createCollisionShape(
            p.GEOM_MESH, fileName=mesh, flags=p.GEOM_FORCE_CONCAVE_TRIMESH
        )
    visual = p.createVisualShape(p.GEOM_MESH, fileName=mesh, rgbaColor=item["color"])
    body = p.createMultiBody(
        baseMass=0, baseCollisionShapeIndex=collision, baseVisualShapeIndex=visual
    )
    contact_properties(p, body, -1, item)
    return body


def load_water(directory, config):
    """Buoyancy and drag for the scene's water regions, or None when it has none."""
    volumes = config.get("water_volumes") or []
    if not volumes:
        return None
    spec = importlib.util.spec_from_file_location("nova_water", directory / "nova_water.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PyBulletWater(volumes, config["gravity"])


def load_movable(p, directory, item):
    spec = item["body"]
    body = p.loadURDF(
        str(directory / spec["urdf"]),
        basePosition=spec["origin"],
        useFixedBase=spec["motion"] != "free",
        flags=p.URDF_USE_INERTIA_FROM_FILE,
    )
    for link in range(-1, p.getNumJoints(body)):
        contact_properties(p, body, link, item)
    for joint in range(p.getNumJoints(body)):
        if p.getJointInfo(body, joint)[2] in {p.JOINT_REVOLUTE, p.JOINT_PRISMATIC}:
            p.changeDynamics(body, joint, jointDamping=spec["damping"])
            if spec.get("drive"):
                # Disable the default holding motor (target velocity 0, large
                # force) that loadURDF leaves on every joint: it silently brakes
                # the prescribed playback — the rotor lost a third of its rate.
                p.setJointMotorControl2(body, joint, p.VELOCITY_CONTROL, force=0)
                continue  # the motion program owns this joint
            # A zero-velocity motor limited to the friction force is PyBullet's
            # standard passive dry-friction model; it does not drive the joint.
            p.setJointMotorControl2(
                body, joint, p.VELOCITY_CONTROL, targetVelocity=0, force=spec["frictionloss"]
            )
    return body


def load_motion(directory):
    """The shared motion evaluators, shipped beside this launcher."""
    path = directory / "nova_motion.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("nova_motion", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def apply_motion(p, motion, driven, t):
    """Prescribed drives at time t: strong joint motors for oscillate/spin,
    a direct kinematic pose for patrol bodies."""
    for uid, spec in driven:
        drive = spec["drive"]
        if spec["motion"] == "kinematic":
            position, yaw = motion.patrol_pose(drive, t)
            p.resetBasePositionAndOrientation(
                uid, list(position), p.getQuaternionFromEuler([0, 0, yaw])
            )
            continue
        for joint in range(p.getNumJoints(uid)):
            if p.getJointInfo(uid, joint)[2] not in {p.JOINT_REVOLUTE, p.JOINT_PRISMATIC}:
                continue
            # The prescribed trajectory is the authority, exactly as for patrol
            # bodies: play it kinematically. A motor tracking the same sinusoid
            # lagged a third of the amplitude at PyBullet's default gains and
            # overshot its limits at stiffer ones.
            p.resetJointState(
                uid,
                joint,
                targetValue=motion.joint_position(drive, t),
                targetVelocity=motion.joint_velocity(drive, t),
            )


def main():
    import pybullet as p

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    config = json.loads((directory / "pybullet_scene.json").read_text())
    client = p.connect(p.DIRECT if args.headless else p.GUI)
    try:
        p.setGravity(0, 0, -config["gravity"])
        p.setTimeStep(0.002)
        p.setRealTimeSimulation(0)
        movable = 0
        driven = []
        for item in config["objects"]:
            if item.get("heightfield"):
                load_terrain(p, directory, item)
            elif item.get("body"):
                uid = load_movable(p, directory, item)
                movable += 1
                if item["body"].get("drive"):
                    driven.append((uid, item["body"]))
            else:
                load_static(p, directory, item)
        water = load_water(directory, config)
        motion = load_motion(directory) if driven else None
        if config.get("robot"):
            robot = p.loadURDF(
                str(directory / config["robot"]),
                basePosition=config["spawn"],
                useFixedBase=config["fixed_base"],
                flags=p.URDF_USE_INERTIA_FROM_FILE,
            )
            for joint in range(p.getNumJoints(robot)):
                if p.getJointInfo(robot, joint)[2] in {p.JOINT_REVOLUTE, p.JOINT_PRISMATIC}:
                    p.setJointMotorControl2(robot, joint, p.VELOCITY_CONTROL, force=0)
        steps = 0
        while p.isConnected() and (not args.headless or steps < 1000):
            if water is not None:
                water.apply(p)
            if motion is not None:
                apply_motion(p, motion, driven, steps * 0.002)
            p.stepSimulation()
            steps += 1
            if not args.headless:
                time.sleep(0.002)
        print(
            f"Passive simulation finished after {steps} steps with {movable} movable "
            f"environment bodies ({len(driven)} on prescribed drives). "
            "No controller or training policy was run."
        )
    finally:
        if p.isConnected(client):
            p.disconnect(client)


if __name__ == "__main__":
    main()
