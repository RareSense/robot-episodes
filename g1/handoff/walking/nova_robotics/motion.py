"""Prescribed-motion evaluation and the cross-target motion program.

One clock, one math, every engine: exporters bake these trajectories natively
where the format allows (USD time samples), and the shipped runners evaluate
them live where it does not (MuJoCo, PyBullet). The motion program JSON is the
package's single source of truth; its parameters are exact, never baked samples,
so a runner at any timestep reproduces the same pose.

q(t) definitions (t in seconds from simulation start):
- oscillate: q = amplitude * sin(2*pi * (t + phase_s) / period_s)
- spin:      q = rate_rad_s * t   (continuous hinge, unlimited)
- patrol:    the body origin moves along the waypoint polyline at speed_m_s;
             loop="cycle" retraces the path (ping-pong), loop="wrap" closes it.
             face_travel yaws the body (about world Z) along the travel heading,
             relative to its authored orientation.
"""

import math

import numpy as np

PROGRAM_FILE = "motion_program.json"
PROGRAM_SCHEMA = 1


def joint_position(drive, t):
    """Joint value in radians/metres for oscillate and spin drives."""
    if drive["kind"] == "oscillate":
        return drive["amplitude"] * math.sin(
            2 * math.pi * (t + drive["phase_s"]) / drive["period_s"]
        )
    if drive["kind"] == "spin":
        return drive["rate_rad_s"] * t
    raise ValueError(f"not a joint drive: {drive['kind']!r}")


def joint_velocity(drive, t):
    if drive["kind"] == "oscillate":
        w = 2 * math.pi / drive["period_s"]
        return drive["amplitude"] * w * math.cos(w * (t + drive["phase_s"]))
    if drive["kind"] == "spin":
        return drive["rate_rad_s"]
    raise ValueError(f"not a joint drive: {drive['kind']!r}")


def _polyline(drive):
    points = np.asarray(drive["waypoints"], dtype=float)
    if drive["loop"] == "wrap" and not np.allclose(points[0], points[-1]):
        points = np.vstack([points, points[0]])
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return points, lengths, float(lengths.sum())


def patrol_pose(drive, t):
    """(position xyz, yaw) of the body origin at time t."""
    points, lengths, total = _polyline(drive)
    distance = drive["speed_m_s"] * t
    if drive["loop"] == "cycle":
        lap = distance % (2 * total)
        s = lap if lap <= total else 2 * total - lap
        forward = lap <= total
    else:
        s = distance % total
        forward = True
    walked = 0.0
    for index, length in enumerate(lengths):
        if s <= walked + length or index == len(lengths) - 1:
            u = 0.0 if length == 0 else (s - walked) / length
            u = min(max(u, 0.0), 1.0)
            position = points[index] * (1 - u) + points[index + 1] * u
            heading = points[index + 1] - points[index]
            if not forward:
                heading = -heading
            yaw = math.atan2(heading[1], heading[0]) if drive["face_travel"] else 0.0
            return position, yaw
        walked += length
    return points[-1], 0.0


def loop_duration(drive):
    """Seconds after which the trajectory repeats exactly."""
    if drive["kind"] == "oscillate":
        return drive["period_s"]
    if drive["kind"] == "spin":
        return 2 * math.pi / abs(drive["rate_rad_s"])
    points, lengths, total = _polyline(drive)
    per_lap = total / drive["speed_m_s"]
    return 2 * per_lap if drive["loop"] == "cycle" else per_lap


def yaw_quaternion_wxyz(yaw):
    half = yaw / 2
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def drive_dict(drive):
    """The drive as plain JSON data (works for both dataclass and dict input)."""
    if isinstance(drive, dict):
        return dict(drive)
    return {
        "kind": drive.kind,
        "amplitude": drive.amplitude,
        "period_s": drive.period_s,
        "phase_s": drive.phase_s,
        "rate_rad_s": drive.rate_rad_s,
        "waypoints": [list(point) for point in drive.waypoints],
        "speed_m_s": drive.speed_m_s,
        "loop": drive.loop,
        "face_travel": drive.face_travel,
    }


def motion_program(entries):
    """The package-level program: every driven body with its per-target handles.

    `entries` are the exporter's environment entries ({"object", "id", ...}).
    """
    bodies = []
    for entry in entries:
        obj = entry["object"]
        body = getattr(obj, "body", None)
        if body is None or body.drive is None:
            continue
        drive = drive_dict(body.drive)
        bodies.append(
            {
                "id": entry["id"],
                "object": obj.name,
                "motion": body.motion,
                "origin": list(body.origin),
                "axis": list(body.axis),
                "drive": drive,
                "loop_duration_s": loop_duration(drive),
                "mujoco": {
                    "body": entry["id"] + "_body",
                    "joint": entry["id"] + "_joint"
                    if body.motion in {"hinge", "slide"}
                    else None,
                    "actuator": entry["id"] + "_drive"
                    if body.motion in {"hinge", "slide"}
                    else None,
                    "mocap": body.motion == "kinematic",
                },
                "pybullet": {"urdf": f"meshes/{entry['id']}.urdf"},
                "usd": {"prim": f"/World/Objects/{entry['id']}"},
                "sdf": {"model": f"nova_environment_{entry['id']}"},
            }
        )
    if not bodies:
        return None
    return {
        "schema": PROGRAM_SCHEMA,
        "clock": "seconds from simulation start; trajectories are exact functions",
        "bodies": bodies,
    }
