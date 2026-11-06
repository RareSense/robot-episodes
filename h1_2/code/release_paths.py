"""Where the release finds its files. Nothing here is machine-specific: everything comes from the repo layout or an environment variable.

    H12_SCENE_DIR   folder holding environment.usda and corrections.json. NOT in this repo, see THIRD_PARTY.md. Required to run.
    H12_PYLIB       folder holding onnxruntime, if Isaac Sim's Python does not already have it. Optional.
    H12_OUT         where logs and recordings are written. Default: <repo>/out
    H12_LOOKS_DIR   render look layers. NOT in this repo. Leave unset and render with --no-looks.
"""
import os

CODE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(CODE)


def code(*parts):
    return os.path.join(CODE, *parts)


def asset(*parts):
    return os.path.join(REPO, "assets", *parts)


def scene(name):
    d = os.environ.get("H12_SCENE_DIR")
    if not d:
        raise RuntimeError("H12_SCENE_DIR is not set. The tea-corner scene is not distributed with this repo: see THIRD_PARTY.md.")
    return os.path.join(d, name)


def pylib():
    return os.environ.get("H12_PYLIB", CODE)


def looks():
    return os.environ.get("H12_LOOKS_DIR", "")


def out(*parts):
    """A path under the output folder. Its parent folder is created."""
    p = os.path.join(os.environ.get("H12_OUT", os.path.join(REPO, "out")), *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def out_dir(*parts):
    p = os.path.join(os.environ.get("H12_OUT", os.path.join(REPO, "out")), *parts)
    os.makedirs(p, exist_ok=True)
    return p
