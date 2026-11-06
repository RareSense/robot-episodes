"""H1-2 + Inspire hands URDF -> USD for PhysX. Run: <isaac-sim>/python.bat import_h12.py"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.abspath(__file__))))    # release_paths.py lives at the code root
import release_paths as RP
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from isaacsim.core.utils.extensions import enable_extension
for e in ("isaacsim.asset.importer.urdf", "isaacsim.robot.schema"):
    enable_extension(e)
app.update()
from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
cfg = URDFImporterConfig(
    urdf_path=RP.asset("robot", "h1_2.urdf"),
    usd_path=RP.out_dir("usd"),
    merge_fixed_joints=False, fix_base=False, allow_self_collision=False,
    joint_drive_type="force", joint_target_type="position",
    override_joint_stiffness=100.0, override_joint_damping=5.0,
    robot_type="Humanoid")
print("OUT", URDFImporter(cfg).import_urdf(), flush=True)
app.close()
