"""Live session: the sim stays open; commands go in through live/<tag>/cmd.py, output comes back in out_<n>.txt. Debug tool, AI-operated.
usage: g1_live.py <tag> <task> cpu|gpu"""
import contextlib, os, sys, time, traceback
TAG, TASK, MODE = sys.argv[1:4]
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
sys.path.insert(0, "E:/g1_isaac/tools")
import g1_core as C
import numpy as np
D = f"E:/g1_isaac/live/{TAG}"
os.makedirs(D, exist_ok=True)
s = C.Sim(app, TASK, gpu=MODE == "gpu")
ns = dict(s=s, C=C, np=np, app=app, D=D, TAG=TAG, os=os)
exec(open("E:/g1_isaac/tools/g1_ops.py", encoding="utf-8").read(), ns)
print("READY", flush=True)
open(D + "/ready.txt", "w").write(time.strftime("%H:%M:%S"))
n = 0
while True:
    cmd = D + "/cmd.py"
    if not os.path.exists(cmd):
        time.sleep(.1)
        continue
    n += 1
    run = f"{D}/cmd_{n}.py"
    os.replace(cmd, run)
    tmp = f"{D}/out_{n}.tmp"
    with open(tmp, "w", encoding="utf-8") as f, contextlib.redirect_stdout(f):
        try:
            exec(compile(open(run, encoding="utf-8").read(), run, "exec"), ns)
        except SystemExit:
            raise
        except BaseException:
            traceback.print_exc(file=f)
    os.replace(tmp, f"{D}/out_{n}.txt")
