"""Assembles pz_rooms.py: the launch and robot/oven setup of pz_bench.py (proven at 1,024 copies), a scene whose counter, walls and furniture can be placed per copy, and the skill in rooms_body.txt.
Run once after editing either."""
import ast
import os

D = os.path.dirname(os.path.abspath(__file__))
src = open(f"{D}/pz_bench.py", encoding="utf-8").read()
head, rest = src.split("def main():", 1)
pre = rest.split('        rb, ov, pl, pi = scene["robot"], scene["oven"], scene["plate"], scene["pizza"]', 1)[0]
i1 = head.index('"""', 3) + 3
head = ('"""Stretch plate-to-rack skill in N different rooms at once on GPU PhysX, Isaac Lab. AI-operated.\n'
        "Every copy gets its own room, generated in code: room size, counter anywhere at any angle, oven and plate anywhere on it, furniture on the floor, robot starting anywhere at any heading.\n"
        "The route to the counter is planned on a floor grid of that room; the hand work is computed in the counter's own frame. Door authored open (test start).\n"
        "State is written once at the spawn (the room), never after. No welds. Walls and furniture are kinematic boxes: they block, they cannot be pushed.\n"
        'usage: pz_rooms.py --tag T --num_envs 30 --spacing 9 [--seed 1] [--cam] [--shots 30]"""') + head[i1:]
a = 'ap.add_argument("--print_rows", type=int, default=150)'
assert a in head
head = head.replace(a, a + '\nap.add_argument("--seed", type=int, default=1)\nap.add_argument("--shots", type=int, default=32)\nap.add_argument("--rec_every", type=int, default=5)\nap.add_argument("--vid_dt", type=float, default=0.)\nap.add_argument("--vid_r", type=int, default=3)')
b = 'RUN = f"C:/Users/raresense/robot_dataset/bench/{args.tag}"'
assert b in head
head = head.replace(b, 'RUN = f"E:/isaaclab_mimic_test/rooms/{args.tag}"')
assert "import argparse\nimport json" in head
head = head.replace("import argparse\nimport json", "import argparse\nimport json\nimport math")
top = '    top = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/CounterTop", spawn=sim_utils.CuboidCfg(size=(2.75, .55, .04), **SOLID), init_state=AssetBaseCfg.InitialStateCfg(pos=(1.525, 2.325, .84)))\n'
body_ = '    body = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/CounterBody", spawn=sim_utils.CuboidCfg(size=(2.41, .48, .82), **SOLID), init_state=AssetBaseCfg.InitialStateCfg(pos=(1.355, 2.36, .41)))\n'
assert top in head and body_ in head
new = '''    ctop: RigidObjectCfg = kbox("CounterTop", (2.75, .55, .04), (.55, .32, .12), (1.525, 2.325, .84))
    cbody: RigidObjectCfg = kbox("CounterBody", (2.41, .48, .82), (.42, .38, .34), (1.355, 2.36, .41))
    w0: RigidObjectCfg = kbox("Wall0", (8.0, .1, 1.1), (.86, .86, .82))
    w1: RigidObjectCfg = kbox("Wall1", (8.0, .1, 1.1), (.86, .86, .82))
    w2: RigidObjectCfg = kbox("Wall2", (8.0, .1, 1.1), (.80, .82, .86))
    w3: RigidObjectCfg = kbox("Wall3", (8.0, .1, 1.1), (.80, .82, .86))
    ob0: RigidObjectCfg = kbox("Table", (1.2, .8, .75), (.45, .30, .20))
    ob1: RigidObjectCfg = kbox("Island", (1.0, .6, .9), (.78, .78, .74))
    ob2: RigidObjectCfg = kbox("ChairA", (.45, .45, .9), (.20, .35, .60))
    ob3: RigidObjectCfg = kbox("ChairB", (.45, .45, .9), (.60, .22, .20))
    ob4: RigidObjectCfg = kbox("Cabinet", (.5, 1.0, 1.8), (.33, .33, .36))
    ob5: RigidObjectCfg = kbox("Crate", (.5, .5, .5), (.20, .50, .30))
'''
head = head.replace(top, "").replace(body_, new)
kin = '''KIN = dict(rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), collision_props=sim_utils.CollisionPropertiesCfg(), physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.5, dynamic_friction=.5))


def kbox(name, size, rgb, pos=(0., 0., -6.)):
    """A box that can be placed per copy before the first step and then stays put (kinematic)."""
    return RigidObjectCfg(prim_path="{ENV_REGEX_NS}/" + name, spawn=sim_utils.CuboidCfg(size=size, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb), **KIN), init_state=RigidObjectCfg.InitialStateCfg(pos=pos))


@configclass
class SceneCfg'''
assert "@configclass\nclass SceneCfg" in head
head = head.replace("@configclass\nclass SceneCfg", kin)
body = open(f"{D}/rooms_body.txt", encoding="utf-8").read()
out = head + "def main():" + pre + body
ast.parse(out)
open(f"{D}/pz_rooms.py", "w", encoding="utf-8").write(out)
print("pz_rooms.py written", len(out))
