"""Calibrated robot-camera observations reconstructed at exact pre-action states."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import av
import h5py
import mujoco
import numpy as np
from PIL import Image

from .spec import write_json

CAMERAS = ("sim_head_rgb", "sim_right_wrist_rgb")


def cull_outside_camera(model, data, camera_id, original_groups, width, height):
    """Conservative sphere/frustum culling avoids transforming the entire café mesh.

    Only the isolated render model's visibility groups change. Objects intersecting
    any part of the viewing frustum remain; planes always remain. Shadows/reflections
    are disabled, so out-of-view geometry cannot contribute via those paths.
    """
    rotation = data.cam_xmat[camera_id].reshape(3, 3)
    points = (data.geom_xpos - data.cam_xpos[camera_id]) @ rotation
    radius = model.geom_rbound
    depth = -points[:, 2]
    ty = np.tan(np.deg2rad(model.cam_fovy[camera_id]) / 2)
    tx = ty * width / height
    visible = (
        (depth + radius >= 0)
        & (np.abs(points[:, 0]) <= depth * tx + radius * np.sqrt(1 + tx * tx))
        & (np.abs(points[:, 1]) <= depth * ty + radius * np.sqrt(1 + ty * ty))
    )
    visible |= model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE
    model.geom_group[:] = np.where(visible, original_groups, 5)


def decoded_frames(path):
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            yield frame.to_ndarray(format="rgb24")


def render_episode(out, request):
    out = Path(out)
    m = mujoco.MjModel.from_binary_path(str(out / "scene.mjb"))
    d = mujoco.MjData(m)
    width, height = request.width, request.height
    m.vis.global_.offwidth = max(width, int(m.vis.global_.offwidth))
    m.vis.global_.offheight = max(height, int(m.vis.global_.offheight))
    camera_meta = {}
    for name in CAMERAS:
        cam = m.camera(name)
        fy = height / (2 * np.tan(np.deg2rad(float(cam.fovy[0])) / 2))
        camera_meta[name] = dict(
            width=width,
            height=height,
            fps=request.policy_hz,
            parent_body=m.body(int(cam.bodyid[0])).name,
            local_position=cam.pos.tolist(),
            local_quaternion_wxyz=cam.quat.tolist(),
            intrinsics=[[fy, 0, width / 2], [0, fy, height / 2], [0, 0, 1]],
            axes="MuJoCo camera: +X right, +Y up, -Z forward",
            sensor="ideal simulated RGB",
            scene_cutaway=False,
        )
    write_json(out / "cameras.json", camera_meta)
    processes = {}
    render = mujoco.Renderer(m, height, width)
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0  # collision proxies; preserve all scene visual geometry
    option.geomgroup[5] = 0
    original_groups = m.geom_group.copy()
    try:
        for name in CAMERAS:
            processes[name] = subprocess.Popen(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-nostdin",
                    "-n",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    f"{width}x{height}",
                    "-framerate",
                    str(request.policy_hz),
                    "-i",
                    "pipe:0",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-threads",
                    "2",
                    "-preset",
                    "fast",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(out / f"{name}.mp4"),
                ],
                stdin=subprocess.PIPE,
            )
        with h5py.File(out / "trace.h5", "r") as trace:
            period = int(trace.attrs["commands_per_observation"])
            frames = int(trace.attrs["complete_intervals"])
            poses = np.empty((frames, len(CAMERAS), 12))
            timestamps = np.empty(frames)
            for frame in range(frames):
                index = frame * period
                d.time = float(trace["time"][index])
                d.qpos[:] = trace["qpos"][index]
                d.qvel[:] = trace["qvel"][index]
                mujoco.mj_kinematics(m, d)
                mujoco.mj_comPos(m, d)
                mujoco.mj_camlight(m, d)
                timestamps[frame] = d.time
                for camera_index, name in enumerate(CAMERAS):
                    camera_id = m.camera(name).id
                    poses[frame, camera_index] = np.r_[d.cam_xpos[camera_id], d.cam_xmat[camera_id]]
                    cull_outside_camera(m, d, camera_id, original_groups, width, height)
                    render.update_scene(d, camera=name, scene_option=option)
                    render.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                    render.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
                    pixels = render.render()
                    processes[name].stdin.write(pixels.tobytes())
                    if frame in {0, frames // 2, frames - 1}:
                        Image.fromarray(pixels).save(out / f"{name}_{frame:05d}.jpg")
                if frame % 500 == 0:
                    print(json.dumps(dict(stage="render", frame=frame, total=frames)), flush=True)
            np.savez_compressed(out / "camera_poses.npz", time=timestamps, world_poses=poses)
    finally:
        render.close()
        for process in processes.values():
            process.stdin.close()
        codes = [process.wait() for process in processes.values()]
    if any(codes):
        raise RuntimeError(f"Camera video encoding failed: {codes}")
    counts = {name: sum(1 for _ in decoded_frames(out / f"{name}.mp4")) for name in CAMERAS}
    if any(count != frames for count in counts.values()):
        raise RuntimeError(f"Camera frame mismatch: {counts}; expected {frames}")
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-n",
            "-i",
            str(out / f"{CAMERAS[0]}.mp4"),
            "-i",
            str(out / f"{CAMERAS[1]}.mp4"),
            "-filter_complex",
            "hstack=inputs=2",
            "-c:v",
            "libx264",
            "-threads",
            "2",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out / "video.mp4"),
        ],
        check=True,
    )
    result = dict(
        passed=True,
        frames=frames,
        decoded_frames=counts,
        timestamps="camera_poses.npz",
        view="head left, right wrist right",
        state_alignment="pre-action, no time interpolation",
    )
    write_json(out / "camera_validation.json", result)
    return result
