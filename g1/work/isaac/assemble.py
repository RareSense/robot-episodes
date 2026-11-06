"""The final film: per step, the person's head-cam segment | the G1's eye camera | the film camera, each at
real speed (the shorter side holds its last frame), a title per step. 1920x1080, 60 fps.
usage: assemble.py <plan.json> <out.mp4>
plan.json: {"eye": "...eye.mp4", "film": "...film.mp4", "steps": [
   {"label": "Mug from the rack", "robot": [t0, t1], "person": [t0, t1], "eye": "...", "film": "..."},
   {"label": "Tea bag", "person": [t0, t1], "card": "not simulated yet"}, ...]}
A step's own "eye"/"film" override the plan's; a "card" step shows that text on the robot side.
Optional: plan "speed" plays the robot clips that much faster (the step title says so); plan "sync": true instead
speeds each robot clip up to the length of the person's segment, so both end together; a step's "note" is a caption
along the bottom (what the step does not pass, what is not simulated)."""
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "tea_paths.py").exists())))
from tea_paths import HANDOFF, ISAAC_PY, WORK  # noqa: E402,F401
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HUMAN = str(HANDOFF / "inputs/vid.mp4")
FONT = "C\\:/Windows/Fonts/arial.ttf"
plan = json.load(open(sys.argv[1]))
out = sys.argv[2]
tmp = Path(tempfile.mkdtemp(prefix="assemble_"))
parts = []


SPEED = float(plan.get("speed", 1.0))


def esc(text):
    return text.replace("\\", "/").replace("'", "’").replace(":", "\\:").replace("%", "\\%")


def header(i, label, note=""):
    h = (f"drawbox=x=0:y=0:w=iw:h=58:color=black@0.55:t=fill,"
         f"drawtext=fontfile='{FONT}':text='Person':x=16:y=14:fontsize=28:fontcolor=white,"
         f"drawtext=fontfile='{FONT}':text='G1 eye (physics)':x=556:y=14:fontsize=28:fontcolor=white,"
         f"drawtext=fontfile='{FONT}':text='{i + 1}. {label}':x=1096:y=14:fontsize=28:fontcolor=white")
    if note:
        h += (f",drawbox=x=0:y=ih-50:w=iw:h=50:color=black@0.6:t=fill,"
              f"drawtext=fontfile='{FONT}':text='{esc(note)}':x=16:y=h-37:fontsize=22:fontcolor=0xe0e0e0")
    return h


for i, st in enumerate(plan["steps"]):
    h0, h1 = st["person"]
    dh = h1 - h0
    # "sync": every robot clip is played faster so it ends with the person's action (playback only, no re-run)
    speed = (st["robot"][1] - st["robot"][0]) / dh if plan.get("sync") and "robot" in st else SPEED
    label = esc(st["label"]) + (f" (robot {speed:.1f}x speed)" if abs(speed - 1) > 0.01 and "card" not in st else "")
    note = st.get("note", "")
    part = tmp / f"part{i:02d}.mp4"
    if "card" in st:
        text = st["card"].replace("'", "").replace(":", "\\:")
        fc = (f"[0:v]scale=-2:1080,fps=60,crop=540:1080[h];"
              f"color=c=0x15171a:s=1380x1080:r=60:d={dh:.3f},"
              f"drawtext=fontfile='{FONT}':text='{text}':x=(w-tw)/2:y=(h-th)/2:fontsize=40:fontcolor=0xd8dadc[c];"
              f"[h][c]hstack=inputs=2,{header(i, label, note)}[v]")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{h0}", "-t", f"{dh}", "-i", HUMAN,
                        "-filter_complex", fc, "-map", "[v]", "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p",
                        "-r", "60", "-t", f"{dh}", str(part)], check=True)
        print("step", i + 1, label, f"card, person {dh:.1f} s", flush=True)
    else:
        r0, r1 = st["robot"]
        dr = (r1 - r0) / speed                          # on screen
        total = max(dr, dh)
        pad_r, pad_h = max(0.0, total - dr), max(0.0, total - dh)
        eye, film = st.get("eye", plan.get("eye")), st.get("film", plan.get("film"))
        sp = f"setpts=PTS/{speed:.6f}," if abs(speed - 1) > 0.01 else ""
        fc = (f"[0:v]scale=-2:1080,fps=60,tpad=stop_mode=clone:stop_duration={pad_h:.3f},crop=540:1080[h];"
              f"[1:v]{sp}scale=-2:1080,fps=60,tpad=stop_mode=clone:stop_duration={pad_r:.3f},crop=540:1080[e];"
              f"[2:v]{sp}scale=-2:1080,fps=60,tpad=stop_mode=clone:stop_duration={pad_r:.3f},crop=840:1080[f];"
              f"[h][e][f]hstack=inputs=3,{header(i, label, note)}[v]")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                        "-ss", f"{h0}", "-t", f"{dh}", "-i", HUMAN,
                        "-ss", f"{r0}", "-t", f"{r1 - r0}", "-i", eye,
                        "-ss", f"{r0}", "-t", f"{r1 - r0}", "-i", film,
                        "-filter_complex", fc, "-map", "[v]", "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p",
                        "-r", "60", str(part)], check=True)
        print("step", i + 1, label, f"robot {dr:.1f} s on screen, person {dh:.1f} s", flush=True)
    parts.append(part)
lst = tmp / "list.txt"
lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts))
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", out],
               check=True)
print("wrote", out)
