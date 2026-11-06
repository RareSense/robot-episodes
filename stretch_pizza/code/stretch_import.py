"""Re-import the Stretch MJCF into usd2. The first import in usd is unreadable since the 07:08 power cut (access denied, left untouched). Same settings as stretch_stand.py.
usage (through stretch_launch.py): stretch_import.py <tag>"""
import os
import sys

TAG = sys.argv[1]
from isaacsim import SimulationApp

app = SimulationApp({"headless": True})
from isaacsim.core.utils.extensions import enable_extension

for e in ("isaacsim.asset.importer.mjcf", "isaacsim.robot.schema"):
    enable_extension(e)
app.update()
from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig

ROOT = "D:/tea_robot/stretch"
cfg = MJCFImporterConfig(mjcf_path="E:/robot_controller_big/mujoco_menagerie/hello_robot_stretch_3/stretch.xml", usd_path=ROOT + "/usd2", import_scene=False, allow_self_collision=False)
print("IMPORTED", MJCFImporter(cfg).import_mjcf(), flush=True)
p = ROOT + "/usd2/stretch/stretch.usda"
print("exists", os.path.exists(p), "readable", os.access(p, os.R_OK), flush=True)
print(open(p).read()[:200], flush=True)
print("ALL DONE", flush=True)
app.close()
