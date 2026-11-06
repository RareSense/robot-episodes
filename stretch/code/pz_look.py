"""Look changes made live, in the open sim, to match pizza_oven.py after edits."""
from pxr import Gf, UsdGeom, UsdLux, UsdShade
import pizza_oven as O
st_ = s.stage
for nm, f in (("ov_steel/nrm", "brushed_n2.png"), ("ov_steel/rgh", "brushed_r2.png"), ("ov_panel/rgh", "brushed_r2.png")):
    UsdShade.Shader(st_.GetPrimAtPath("/World/Looks/" + nm)).GetInput("file").Set(O.TEX + f)
sh = UsdShade.Shader(st_.GetPrimAtPath("/World/Looks/ov_liner/s"))
sh.GetInput("diffuseColor").Set(Gf.Vec3f(.78, .78, .75))
sh.GetInput("metallic").Set(.55)
sh.GetInput("roughness").Set(.34)
L = UsdLux.SphereLight(st_.GetPrimAtPath(O.BODY + "/lamp"))
L.GetRadiusAttr().Set(.02)
L.GetIntensityAttr().Set(260000.0)
L.GetColorAttr().Set(Gf.Vec3f(1., .90, .74))
UsdGeom.Xformable(L).GetOrderedXformOps()[0].Set(Gf.Vec3d(O.CX1 - .05, O.CY1 - .10, O.CZ1 - .03))
L2 = UsdLux.SphereLight.Define(st_, O.BODY + "/lamp2")
L2.CreateRadiusAttr(.02)
L2.CreateIntensityAttr(260000.0)
L2.CreateColorAttr(Gf.Vec3f(1., .90, .74))
O.place(L2, (O.CX0 + .05, O.CY0 + .12, O.CZ1 - .03))
