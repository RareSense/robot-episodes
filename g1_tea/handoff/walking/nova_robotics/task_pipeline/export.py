"""LeRobot 0.4.4 writer/reader integration; no handcrafted lookalike dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from .rendering import CAMERAS, decoded_frames
from .spec import write_json


def scenario_lineage(episode):
    provenance = json.loads((Path(episode) / "provenance.json").read_text())
    # Appearance changes do not create independent scenarios for held-out evaluation.
    identity = dict(
        source=provenance["source_sha256"],
        adapter=provenance["scene_adapter"],
        task=provenance["task_family"],
        objects=provenance["task_fixture"]["added_clutter"],
    )
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def split_for(lineage):
    bucket = int(lineage[:8], 16) % 10
    return "test" if bucket == 9 else "validation" if bucket == 8 else "train"


def export_dataset(episodes, root, request):
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if not episodes:
        raise ValueError("Cannot export an empty successful-demonstration dataset")
    for episode in episodes:
        report = json.loads((Path(episode) / "validation.json").read_text())
        if not all(
            report.get(key)
            for key in (
                "usable",
                "task_success",
                "simulation_valid",
                "camera_valid",
                "replay_valid",
            )
        ):
            raise ValueError(f"Episode has not passed all collection gates: {episode}")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    assignments = [
        {"episode": str(Path(ep).resolve()), "lineage": scenario_lineage(ep)} for ep in episodes
    ]
    for item in assignments:
        item["split"] = split_for(item["lineage"])
    write_json(root / "splits.json", assignments)
    outputs = {}
    for split in ("train", "validation", "test"):
        members = [Path(item["episode"]) for item in assignments if item["split"] == split]
        if not members:
            continue
        with h5py.File(members[0] / "trace.h5", "r") as trace:
            motors = json.loads(trace.attrs["motor_names"])
            period = int(trace.attrs["commands_per_observation"])
        action_names = [f"step_{step:02d}/{motor}/Nm" for step in range(period) for motor in motors]
        state_names = [f"{motor}/{unit}" for unit in ("rad", "rad_s") for motor in motors]
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (len(state_names),),
                "names": state_names,
            },
            "action": {"dtype": "float32", "shape": (len(action_names),), "names": action_names},
            **{
                f"observation.images.{name}": {
                    "dtype": "video",
                    "shape": (request.height, request.width, 3),
                    "names": ["height", "width", "channels"],
                }
                for name in CAMERAS
            },
        }
        repo_id = f"nova-local/cafe-{split}"
        dataset = LeRobotDataset.create(
            repo_id,
            fps=request.policy_hz,
            root=root / split,
            features=features,
            robot_type="unitree_g1_sim",
            video_backend="pyav",
            vcodec="h264",
            streaming_encoding=True,
            encoder_threads=2,
        )
        total = 0
        for ep in members:
            videos = {name: iter(decoded_frames(ep / f"{name}.mp4")) for name in CAMERAS}
            try:
                with h5py.File(ep / "trace.h5", "r") as trace:
                    qa, va = trace["motor_qpos_indices"][:], trace["motor_qvel_indices"][:]
                    if np.any(qa < 0) or np.any(va < 0):
                        raise ValueError("Joint encoder export requires joint motors")
                    count = int(trace.attrs["complete_intervals"])
                    for frame in range(count):
                        index = frame * period
                        state = np.r_[trace["qpos"][index][qa], trace["qvel"][index][va]].astype(
                            "float32"
                        )
                        action = trace["ctrl"][index : index + period].reshape(-1).astype("float32")
                        dataset.add_frame(
                            {
                                "observation.state": state,
                                "action": action,
                                "task": request.prompt,
                                **{
                                    f"observation.images.{name}": next(videos[name])
                                    for name in CAMERAS
                                },
                            }
                        )
                    total += count
                    if any(next(iterator, None) is not None for iterator in videos.values()):
                        raise ValueError("Camera has extra frames beyond action intervals")
                dataset.save_episode()
            finally:
                for iterator in videos.values():
                    iterator.close()
        dataset.finalize()
        reader = LeRobotDataset(repo_id, root=root / split, video_backend="pyav")
        if len(reader) != total:
            raise ValueError("LeRobot reader frame count differs from the source")
        # Check numeric alignment and camera decoding at each episode's first/last frame.
        offset = 0
        for ep in members:
            with h5py.File(ep / "trace.h5", "r") as trace:
                count = int(trace.attrs["complete_intervals"])
                for local_index in {0, count // 2, count - 1}:
                    sample = reader[offset + local_index]
                    expected = trace["ctrl"][
                        local_index * period : (local_index + 1) * period
                    ].reshape(-1)
                    np.testing.assert_allclose(
                        sample["action"].numpy(), expected, rtol=1e-6, atol=1e-5
                    )
                    for name in CAMERAS:
                        if tuple(sample[f"observation.images.{name}"].shape) != (
                            3,
                            request.height,
                            request.width,
                        ):
                            raise ValueError("Decoded image shape mismatch")
                offset += count
        batch = next(
            iter(torch.utils.data.DataLoader(reader, batch_size=min(2, total), num_workers=0))
        )
        if (
            not torch.isfinite(batch["action"]).all()
            or not torch.isfinite(batch["observation.state"]).all()
        ):
            raise ValueError("Non-finite training batch")
        outputs[split] = dict(
            episodes=len(members),
            frames=total,
            reader_verified=True,
            batch_shape=list(batch["action"].shape),
        )
    result = dict(
        export_compatible=True,
        learning_evaluated=False,
        format="LeRobot v3",
        writer_version="0.4.4",
        splits=outputs,
        action_adapter="g1_motor_chunk_500hz_v1",
        action_units="joint torque Nm",
        normalization=(
            "Each split is a separate dataset; use ONLY train/meta/stats.json "
            "for learning/evaluation"
        ),
        missing_splits=[s for s in ("train", "validation", "test") if s not in outputs],
        limitations=[
            "No held-out policy success claim; reader/batch verification only",
            "Torque-chunk policy interface, not the proposed Cartesian adapter",
        ],
    )
    write_json(root / "export_validation.json", result)
    (root / "README.md").write_text(
        "# Café G1 simulated demonstrations\n\n"
        "Observations precede the following 20 motor commands (500 Hz) in each 25 Hz frame. "
        "Action is a flattened [20, 43] array in actuator order, in Nm. Execute using "
        "`nova_robotics.task_pipeline.recording.execute_motor_chunk`. The expert used "
        "privileged scene state; student inputs contain only joint encoders "
        "and two virtual RGB views. "
        "Calibration and full float64 audit traces remain in the episode bundles.\n\n"
        "Train/validation/test are independent LeRobot datasets "
        "grouped by initial physical scenario. "
        "Use training split statistics for all policy evaluation. Empty splits remain empty; "
        "this small sample does not establish learning utility or real-robot transfer.\n"
    )
    return result
