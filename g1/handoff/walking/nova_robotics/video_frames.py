"""Walkthrough video → ordered still frames, before a robotics twin turn.

Runs inside the build cage as part of the pack's evidence step (twin_evidence.py):
the video is mounted read-only from the task's frozen attachments and never enters
the workspace; only the selected stills do.

A video is the one photo-only input that PROVES how rooms connect: frame K+1
was taken a moment after frame K, so the camera physically walked between
them. The evidence step turns each attached walkthrough into evidence the
analyst and builder can actually read:

1. Sample the video at 5 frames per second (ffmpeg, long edge 960 px).
2. Within every second keep only the SHARPEST frame (variance of the
   Laplacian), discarding motion-blurred ones — a walking phone blurs most
   frames.
3. Walk that 1 fps sharp sequence and keep a frame only when the view has
   visibly changed (perceptual distance), yielding the unique scene frames
   used for counting and geometry.
4. Write the keyframes to refs/video_frames/ (time-stamped names, walk
   order) and contact sheets of the whole sharp sequence for connectivity.

Everything is local ffmpeg + PIL; no model call happens here, and the video
itself never enters a model build — only the selected stills do.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

FRAMES_DIR = "refs/video_frames"
INDEX = "refs/video_frames/frames.json"

SAMPLE_FPS = 5
LONG_EDGE = 960
MAX_SECONDS = 720  # 12 minutes of walking is plenty for one site
MAX_VIDEOS = 3
MAX_KEYFRAMES = 36  # per video; unique views used for counting
MAX_SHEETS = 8  # per video; the whole walk at a glance
SHEET_COLUMNS, SHEET_ROWS = 3, 4
NOVELTY_THRESHOLD = 11.0  # mean |Δ| on a 32×32 grayscale, 0..255


def _run(arguments: list[str], timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(
        arguments, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout
    )


def _duration_seconds(ffprobe: str, video: Path) -> float:
    probe = _run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
        timeout=60,
    )
    try:
        return float(probe.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        raise ValueError(f"ffprobe could not read {video.name}: {probe.stderr[-200:]}") from None


def sharpness(image: Image.Image) -> float:
    """Variance of the Laplacian on a small grayscale copy: motion blur flattens
    edges, so blurred frames score far below their sharp neighbours."""

    small = image.convert("L")
    small.thumbnail((384, 384))
    pixels = np.asarray(small, dtype=np.float32)
    interior = pixels[1:-1, 1:-1]
    laplacian = (
        pixels[:-2, 1:-1] + pixels[2:, 1:-1] + pixels[1:-1, :-2] + pixels[1:-1, 2:]
        - 4.0 * interior
    )
    return float(laplacian.var())


def _signature(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L").resize((32, 32)), dtype=np.float32)


def novelty(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).mean())


def select_frames(
    scored: list[tuple[float, float, Path]]
) -> tuple[list[tuple[float, Path]], list[tuple[float, Path]]]:
    """(sharp 1 fps sequence, unique keyframes) from (t, sharpness, path) samples."""

    by_second: dict[int, tuple[float, float, Path]] = {}
    for t, score, path in scored:
        bucket = int(t)
        if bucket not in by_second or score > by_second[bucket][1]:
            by_second[bucket] = (t, score, path)
    sharp = [(t, path) for t, _score, path in
             (by_second[k] for k in sorted(by_second))]

    keyframes: list[tuple[float, Path]] = []
    last_signature: np.ndarray | None = None
    for t, path in sharp:
        with Image.open(path) as image:
            signature = _signature(image)
        if last_signature is None or novelty(signature, last_signature) >= NOVELTY_THRESHOLD:
            keyframes.append((t, path))
            last_signature = signature
    if len(keyframes) > MAX_KEYFRAMES:
        chosen = np.linspace(0, len(keyframes) - 1, MAX_KEYFRAMES).round().astype(int)
        keyframes = [keyframes[i] for i in sorted(set(chosen.tolist()))]
    return sharp, keyframes


def _sheets(sharp: list[tuple[float, Path]], folder: Path, ordinal: int) -> list[str]:
    """Contact sheets of the whole sharp sequence, in walk order, each tile
    labelled with its timestamp — the connectivity evidence."""

    per_sheet = SHEET_COLUMNS * SHEET_ROWS
    if len(sharp) > MAX_SHEETS * per_sheet:
        chosen = np.linspace(0, len(sharp) - 1, MAX_SHEETS * per_sheet).round().astype(int)
        sharp = [sharp[i] for i in sorted(set(chosen.tolist()))]
    tile_w = 320
    written: list[str] = []
    for start in range(0, len(sharp), per_sheet):
        batch = sharp[start:start + per_sheet]
        tiles = []
        for t, path in batch:
            with Image.open(path) as image:
                tile = image.convert("RGB")
                tile.thumbnail((tile_w, tile_w))
            draw = ImageDraw.Draw(tile)
            label = f"t={t:.0f}s"
            draw.rectangle((0, 0, 8 + 7 * len(label), 16), fill=(0, 0, 0))
            draw.text((4, 2), label, fill=(255, 255, 60))
            tiles.append(tile)
        height = max(tile.height for tile in tiles)
        sheet = Image.new("RGB", (SHEET_COLUMNS * tile_w, SHEET_ROWS * height), (24, 24, 24))
        for i, tile in enumerate(tiles):
            sheet.paste(tile, ((i % SHEET_COLUMNS) * tile_w, (i // SHEET_COLUMNS) * height))
        name = f"sheet_{ordinal}_{start // per_sheet + 1}.jpg"
        sheet.save(folder / name, quality=85)
        written.append(name)
    return written


def frame_name(ordinal: int, t: float) -> str:
    """A kept frame's file, named by its time to the tenth of a second. Two views kept
    within one second (a fast turn) must never share a name: one would overwrite the
    other and the inventory would read the survivor twice."""

    return f"vf{ordinal}_{t:06.1f}s.jpg"


def extract(video: Path, folder: Path, ordinal: int) -> dict[str, Any]:
    """One video → keyframes + sheets inside `folder`. Pure local computation."""

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise ValueError("ffmpeg is not installed on this host")
    duration = min(_duration_seconds(ffprobe, video), MAX_SECONDS)
    with tempfile.TemporaryDirectory(prefix="nova-video-") as temporary:
        work = Path(temporary)
        scale = f"scale='if(gt(iw,ih),{LONG_EDGE},-2)':'if(gt(iw,ih),-2,{LONG_EDGE})'"
        result = _run(
            [ffmpeg, "-nostdin", "-v", "error", "-t", str(duration), "-i", str(video),
             "-map", "0:v:0", "-an", "-vf", f"fps={SAMPLE_FPS},{scale}",
             "-q:v", "4", str(work / "f_%05d.jpg")],
            timeout=max(120.0, duration * 4),
        )
        samples = sorted(work.glob("f_*.jpg"))
        if result.returncode != 0 or not samples:
            raise ValueError(f"ffmpeg could not decode {video.name}: {result.stderr[-300:]}")
        scored = []
        for i, path in enumerate(samples):
            with Image.open(path) as image:
                scored.append((i / SAMPLE_FPS, sharpness(image), path))
        sharp, keyframes = select_frames(scored)
        kept = []
        for t, path in keyframes:
            name = frame_name(ordinal, t)
            shutil.copyfile(path, folder / name)
            kept.append({"file": name, "t": round(t, 1)})
        sheets = _sheets(sharp, folder, ordinal)
    return {
        "duration_s": round(duration, 1),
        "frames_sampled": len(samples),
        "sharp_per_second": len(sharp),
        "keyframes": kept,
        "sheets": sheets,
    }


def materials(workspace_root: Path) -> tuple[list[Path], list[Path], str]:
    """(keyframes, sheets, prompt note) for the analysis stage; empty when no video."""

    index_path = workspace_root / INDEX
    if not index_path.is_file() or index_path.is_symlink():
        return [], [], ""
    try:
        report = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return [], [], ""
    folder = workspace_root / FRAMES_DIR
    keyframes: list[Path] = []
    sheets: list[Path] = []
    lines: list[str] = []
    for video in report.get("videos", []):
        if video.get("error"):
            continue
        lines.append(
            f"- {video.get('name', 'walkthrough')}: {video['duration_s']} s walk, "
            f"{len(video['keyframes'])} unique views kept"
        )
        for frame in video["keyframes"]:
            path = folder / frame["file"]
            if path.is_file() and not path.is_symlink():
                keyframes.append(path)
        for name in video.get("sheets", []):
            path = folder / name
            if path.is_file() and not path.is_symlink():
                sheets.append(path)
    if not keyframes:
        return [], [], ""
    note = (
        "\n\nWALKTHROUGH VIDEO (frames in TIME ORDER — this is one continuous walk):\n"
        + "\n".join(lines)
        + "\nThe vf*_<t>s.jpg images are the unique scene frames, named by the second "
        "they were taken; consecutive frames are moments apart, so the camera moved "
        "continuously between them. Use that order to connect spaces exactly: a doorway "
        "seen ahead at t=12s is the room the camera is inside at t=15s, so those rooms "
        "adjoin through that doorway. The sheet_*.jpg images show the whole walk at one "
        "frame per second for the route. An object seen in several consecutive frames is "
        "ONE object, not several — count it once, from its clearest frame; a genuinely "
        "repeated object appears in clearly different places along the walk."
    )
    return keyframes, sheets, note
