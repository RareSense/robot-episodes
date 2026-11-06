"""Fill light for the live renders, same as stretch_still.py. Run in the open sim with exec."""
from pxr import Gf, UsdGeom, UsdLux
d = UsdLux.DistantLight.Define(s.stage, "/World/RecFill")
d.CreateIntensityAttr(2500.0)
d.CreateAngleAttr(30.0)
d.CreateColorAttr(Gf.Vec3f(1.0, .97, .92))
x = UsdGeom.Xformable(d)
x.ClearXformOpOrder()
x.AddRotateXYZOp().Set(Gf.Vec3f(75.0, 0.0, -20.0))
