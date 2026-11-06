"""Copy the code the final episode needs out of the working tree, with machine paths replaced. The source tree is only read.

    python tools/export_code.py --source <working tree>

Each file is changed in exactly three ways and nothing else: absolute paths become calls into code/release_paths.py, one bootstrap
block is added so that module can be imported, and the REWORD comments below are reworded. A file that still holds a drive path after
that, or that no longer compiles, stops the export. code/SOURCE_MANIFEST.json records the hash of every source and output file.
"""
import argparse
import difflib
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sanitize as SZ

# (path under the working tree, path under code/). ".." entries sit beside the working tree.
FILES = [
    ("run_live_episode.py", "run_live_episode.py"),
    ("full_episode.py", "full_episode.py"),
    ("h12_physx.py", "h12_physx.py"),
    ("mug_place.py", "mug_place.py"),
    ("arm_servo.py", "arm_servo.py"),
    ("cup_walk_away.py", "cup_walk_away.py"),
    ("import_h12.py", "import_h12.py"),
    ("props/gates.py", "props/gates.py"),
    ("props/props.py", "props/props.py"),
    ("props/record.py", "props/record.py"),
    ("fluids/boot.py", "fluids/boot.py"),
    ("fluids/fluids.py", "fluids/fluids.py"),
    ("steps/kettle/stage.py", "steps/kettle/stage.py"),
    ("steps/kettle/stage_live.py", "steps/kettle/stage_live.py"),
    ("steps/powder/stage.py", "steps/powder/stage.py"),
    ("steps/powder/stage_live.py", "steps/powder/stage_live.py"),
    ("steps/powder/pwprops.py", "steps/powder/pwprops.py"),
    ("steps/powder/pw_scoop.py", "steps/powder/pw_scoop.py"),
    ("steps/teabag_stir/stage.py", "steps/teabag_stir/stage.py"),
    ("steps/teabag_stir/stage_live.py", "steps/teabag_stir/stage_live.py"),
    ("../isaac_native/balance_support/crouch_support.py", "balance_support/crouch_support.py"),
    ("../work/isaac/water/water_layer.py", "water/water_layer.py"),
]
REWORD = [("# read the other agent's module, leave nothing in its folder", "# leave no bytecode beside the shared modules"),
          ("Measured-height crouch and COM balance for Claude's native PhysX H12 Sim.", "Measured-height crouch and COM balance for the native PhysX H1-2 Sim.")]

LITERAL = re.compile(r"""\b(rf|fr|r|f)?(["'])([A-Za-z]:\\[^"'\n]*?)\2""")
STEP_CONCAT = re.compile(r"""ROOT \+ r"\\(\w+)\\stage\.py\"""")
BARE = [(re.compile(r"[A-Za-z]:\\+isaacsim\w*\\+"), "<isaac-sim>/"),
        (re.compile(r"[A-Za-z]:\\+tea_robot\\+isaac_delivery"), "<repo>/code"),
        (re.compile(r"[A-Za-z]:\\+tea_robot\\+\w+"), "<repo>")]


def args_of(parts):
    return ", ".join(repr(p) for p in parts)


def map_literal(m):
    fstr = "f" in (m.group(1) or "")
    p = m.group(3).replace("\\\\", "\\").replace("/", "\\")
    low = p.lower()

    def tail(anchor):
        i = low.index(anchor) + len(anchor)
        return [x for x in p[i:].split("\\") if x]

    if "scene_v3\\" in low:
        return f"RP.scene({tail('scene_v3')[0]!r})"
    if "h1_2_description\\h1_2.urdf" in low:
        return 'RP.asset("robot", "h1_2.urdf")'
    if low.endswith("usd\\h1_2\\h1_2.usda"):
        return 'RP.asset("robot", "h1_2", "h1_2.usda")'
    if "policies\\h1_2_walk_v1\\" in low:
        return f'RP.asset("policy", "h1_2_walk_v1", {tail("h1_2_walk_v1")[0]!r})'
    if low.endswith("isaac_h12\\usd"):
        return 'RP.out_dir("usd")'
    if low.endswith("isaac_h12\\pylib"):
        return "RP.pylib()"
    if low.endswith("isaac_native\\balance_support"):
        return 'RP.code("balance_support")'
    if low.endswith("work\\isaac\\water"):
        return 'RP.code("water")'
    if low.endswith("tea-robot-physics\\work\\isaac"):
        return "RP.looks()"
    if "isaac_delivery" in low:
        t = tail("isaac_delivery")
        if t[:1] == ["logs"]:
            if fstr and len(t) == 2 and t[1] == "{TAG}":
                return 'RP.out("logs", TAG)'
            return f"RP.out({args_of(t)})" if len(t) > 1 else 'RP.out_dir("logs")'
        if t[:1] == ["previews"]:
            return 'RP.out_dir("previews")'
        if t[:2] == ["props", "logs"]:
            return f'RP.out("props_logs", {args_of(t[2:])})' if len(t) > 2 else 'RP.out_dir("props_logs")'
        return f"RP.code({args_of(t)})" if t else "RP.code()"
    raise SystemExit(f"no mapping for a path literal in the source: extend map_literal ({len(p)} characters)")


def bootstrap(depth):
    up = ", ".join(['".."'] * depth)
    root = f'os.path.join(os.path.dirname(os.path.abspath(__file__)), {up})' if depth else "os.path.dirname(os.path.abspath(__file__))"
    return (f"import os\nimport sys\nsys.path.insert(0, os.path.abspath({root}))    # release_paths.py lives at the code root\n"
            "import release_paths as RP\n")


def insert_bootstrap(text, depth):
    """After the module docstring and any __future__ import."""
    import ast
    tree = ast.parse(text)
    line = 0
    for node in tree.body:
        doc = isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant) and isinstance(node.value.value, str) and node is tree.body[0]
        if doc or (isinstance(node, ast.ImportFrom) and node.module == "__future__"):
            line = node.end_lineno
        else:
            break
    nl = "\r\n" if "\r\n" in text else "\n"
    rows = text.split(nl)
    return nl.join(rows[:line] + bootstrap(depth).rstrip("\n").split("\n") + rows[line:])


def transform(text, dst):
    for a, b in REWORD:
        text = text.replace(a, b)
    text = STEP_CONCAT.sub(lambda m: f'os.path.join(ROOT, "{m.group(1)}", "stage.py")', text)
    text = LITERAL.sub(map_literal, text)
    for rx, rep in BARE:
        text = rx.sub(rep, text)
    if "RP." in text:
        text = insert_bootstrap(text, len(Path(dst).parts) - 1)
    return text


def sha(b):
    return hashlib.sha256(b).hexdigest()


def export(source, repo):
    source, repo = Path(source).resolve(), Path(repo).resolve()
    rows, bad = [], []
    for src, dst in FILES:
        raw = (source / src).read_bytes()
        bom = raw.startswith(b"\xef\xbb\xbf")
        text = (raw[3:] if bom else raw).decode("utf-8")
        new = transform(text, dst)
        compile(new, dst, "exec")
        issues = [i for i in SZ.find_issues(new) if i[1] in ("windows path", "home path", "attribution word") or "key" in i[1] or "token" in i[1]]
        if issues:
            bad.append((dst, issues[:5]))
        out = repo / "code" / dst
        out.parent.mkdir(parents=True, exist_ok=True)
        data = new.encode("utf-8")
        out.write_bytes(data)
        changed = sum(1 for x in difflib.ndiff(text.splitlines(), new.splitlines()) if x[:1] in "+-")
        rows.append(dict(file=dst, source_sha256=sha(raw), release_sha256=sha(data), lines_added_or_removed=changed))
    if bad:
        for d, i in bad:
            print("STILL PRIVATE:", d, i)
        raise SystemExit(1)
    (repo / "code" / "SOURCE_MANIFEST.json").write_text(json.dumps(dict(
        note="source_sha256 is the file as it ran; release_sha256 is the copy here, with paths and bootstrap changed only (see tools/export_code.py)",
        files=rows), indent=1), encoding="utf-8")
    print("exported", len(rows), "files")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="the working tree, read only")
    ap.add_argument("--repo", default=str(HERE.parent))
    A = ap.parse_args()
    export(A.source, A.repo)
