"""G1 + Dex3 MJCF -> USD for PhysX, once. AI-operated."""
import os, sys
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from isaacsim.core.utils.extensions import enable_extension
for e in ("isaacsim.asset.importer.mjcf", "isaacsim.robot.schema"):
    enable_extension(e)
app.update()
from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig
cfg = MJCFImporterConfig(mjcf_path="E:/robot_episodes_repo/g1/handoff/robots/unitree_g1/g1_with_hands.xml", usd_path="E:/g1_isaac/usd", import_scene=False, allow_self_collision=False)
print("IMPORTED", MJCFImporter(cfg).import_mjcf(), flush=True)
print("ALL DONE", flush=True)
app.close()
