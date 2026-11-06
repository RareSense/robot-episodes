"""Where the repo's pieces live. Every script resolves its paths from here, so the repo runs from any folder.
Override with environment variables when the pieces live elsewhere:
  TEA_WORK     this folder (code, agents, isaac assets)          default: <repo>/work
  TEA_HANDOFF  controller package, robot model, twin scene, video default: <repo>/handoff
  ISAAC_PY     Isaac Sim's python launcher (python.bat / python.sh), only needed for rendering"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WORK = Path(os.environ.get("TEA_WORK", REPO / "work"))
HANDOFF = Path(os.environ.get("TEA_HANDOFF", REPO / "handoff"))
_PC_ISAAC = r"D:\isaacsim62\python.bat"
ISAAC_PY = os.environ.get("ISAAC_PY", _PC_ISAAC if os.path.exists(_PC_ISAAC) else "python.sh")
