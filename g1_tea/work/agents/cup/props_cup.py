"""Cup step scene detail (extras module for tea_sim.session): the mug handle's contact model.

The twin's handle collision (object_0316_collision_09) is one convex piece; MuJoCo gives a convex-convex touch ONE
contact point, and with the default 3-D contact (condim 3) a point contact resists no twist about its normal. Real
silicone finger pads touch over a patch and do resist twist (the soft-finger contact model). The Dex3 pad geoms already
carry that torsional coefficient in the maker's model (friction "1 0.005 0.0001": 5 mm); it is only used when the
contact is 4-D. This module makes the handle's contacts 4-D (condim 4) so that coefficient applies. No friction
coefficient is raised (sliding friction stays as in the model: max(pad 1.0, mug 0.55)); nothing is added to the bodies.
"""
ACTIVE = []
MUG = "object_0316"


def add(root, bodies):
    b = bodies[MUG + "_body"]
    for g in b.findall("geom"):
        if g.get("name") == MUG + "_collision_09":
            g.set("condim", "4")
