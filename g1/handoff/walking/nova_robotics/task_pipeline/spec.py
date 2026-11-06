"""A deliberately bounded prompt resolver and immutable executable request contract."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

CAFE_HASH = "a832b68df2e22c871950524fb5c3927936a9b562d1a58533531e1594d0c37678"
PROJECT_ID = "prj_01a12fae51c60f193db4911cd4fa"
SCHEMA = "nova.robot-task/v1"


def write_json(path: Path, value):
    """Atomic snapshots; callers hold the job lock when updating job state."""
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class TaskRequest:
    prompt: str
    scene: str
    robot: str
    task_family: str = "door_and_cleanup"
    episodes: int = 1
    max_attempts: int = 3
    seed: int = 0
    variation_m: float = 0.005
    lighting_variation: float = 0.1
    max_wall_seconds: int = 7200
    attempt_seconds: int = 1800
    max_sim_seconds: int = 300
    width: int = 320
    height: int = 240
    policy_hz: int = 25
    action_adapter: str = "g1_motor_chunk_500hz_v1"
    schema_version: str = SCHEMA
    source_sha256: str = CAFE_HASH
    project_id: str = PROJECT_ID

    def validate(self, check_files=True):
        if self.schema_version != SCHEMA:
            raise ValueError("Unsupported request version; use `plan` to resolve a new request")
        if self.task_family not in {"door_and_cleanup", "tabletop_cleanup"}:
            raise ValueError("Supported tasks: G1 café door-and-cleanup or tabletop cleanup")
        if self.action_adapter != "g1_motor_chunk_500hz_v1" or self.policy_hz != 25:
            raise ValueError("This worker records 20 motor commands per 25 Hz observation")
        for key in (
            "episodes",
            "max_attempts",
            "max_wall_seconds",
            "attempt_seconds",
            "max_sim_seconds",
            "width",
            "height",
            "policy_hz",
            "seed",
        ):
            value = getattr(self, key)
            if type(value) is not int or value < (0 if key == "seed" else 1):
                raise ValueError(f"{key} must be a positive integer (seed may be zero)")
        if self.max_attempts < self.episodes:
            raise ValueError("max_attempts must cover the requested successful episodes")
        if not 0 <= self.variation_m <= 0.015 or not 0 <= self.lighting_variation <= 0.2:
            raise ValueError(
                "Café adapter supports at most 15 mm placement and 20% light variation"
            )
        if (
            self.width % 2
            or self.height % 2
            or not 64 <= self.width <= 640
            or not 64 <= self.height <= 480
        ):
            raise ValueError("Camera dimensions must be even, within 64..640 by 64..480")
        if self.source_sha256 != CAFE_HASH or self.project_id != PROJECT_ID:
            raise ValueError(
                "This scene adapter is qualified only for the pinned Café Robotique export"
            )
        if check_files:
            if sha256(self.scene) != self.source_sha256:
                raise ValueError(
                    "Scene hash differs from the registered café; no guessed scene substitution"
                )
            if Path(self.robot).name != "g1_with_hands.xml" or not Path(self.robot).is_file():
                raise ValueError("This adapter requires the local G1 with articulated hands")
        return self

    def to_dict(self):
        return asdict(self)

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


def resolve(prompt: str, *, scene, robot, **options) -> TaskRequest:
    """Constrained local grammar: ambiguous/new capabilities fail before allocating a worker.

    A future model resolver can produce the same typed request, but cannot bypass validation.
    Unknown vocabulary is reported rather than silently changing the requested task.
    """
    normalized = prompt.lower().replace("café", "cafe")
    words = set(re.findall(r"[a-z]+[0-9]*", normalized))
    allowed = set(  # noqa: SIM905 -- readable vocabulary grouped by task meaning
        """use the existing generated database scene environment cafe robotique and a an
        g1 unitree robot it to open opens opening close closes closing door doors then start
        clean cleans cleaning cleanup clearing clear mess room put puts putting pick picks picking
        up place places placing move moves moving scattered colored red blue yellow blocks block
        cubes cube into in bin table tabletop sorting station all three with head wrist right
        camera cameras views video simulation dataset training data demonstrations demonstration
        episodes episode runs run generate collect export lerobot varied vary positions lighting
        please show me of from my this that make get both reuse staged props plus after through
        walk walks walking return working full task""".split()  # noqa: SIM905
    )
    unknown = sorted(words - allowed)
    if unknown:
        raise ValueError(
            "Unresolved task words: "
            + ", ".join(unknown)
            + ". Supported: café G1, blocks into bin, optionally open/close door."
        )
    if not words.intersection(
        {"block", "blocks", "cube", "cubes", "cleanup", "clean", "cleaning", "mess"}
    ):
        raise ValueError("Specify cleanup of the café's staged blocks into its bin")
    door = bool(words.intersection({"door", "doors"}))
    if door and not (
        words.intersection({"open", "opens", "opening"})
        and words.intersection({"close", "closes", "closing"})
    ):
        raise ValueError("The supported door task opens AND closes the passive kitchen panel")
    counts = re.findall(r"\b(\d+)\s+(?:demonstrations?|episodes?|runs?)\b", normalized)
    if len(set(counts)) > 1:
        raise ValueError("Conflicting episode counts in the prompt")
    if counts:
        count = int(counts[0])
        if options.get("episodes") is not None and options["episodes"] != count:
            raise ValueError("Prompt count conflicts with --episodes")
        options["episodes"] = count
    options = {key: value for key, value in options.items() if value is not None}
    options.setdefault("max_attempts", max(3, options.get("episodes", 1) * 2))
    return TaskRequest(
        prompt=prompt,
        scene=str(Path(scene).resolve()),
        robot=str(Path(robot).resolve()),
        task_family="door_and_cleanup" if door else "tabletop_cleanup",
        **options,
    ).validate()


def load_request(path):
    return TaskRequest(**json.loads(Path(path).read_text())).validate()
