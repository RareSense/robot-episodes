"""Live loop: the sim stays open and frozen at chosen stages; commands go in, the result comes out, with no restart.
Debug only: never use in a final run. Stages to stop at: env LIVE_PAUSE=a,b,c ("*" = every stage). Always stops at the end of a step.
live\<tag>\paused.txt   where it stopped
live\<tag>\cmd.py       write Python here (atomically): runs in the stage's scope, so s, kmove, ktarget, par, ... exist
live\<tag>\out_<n>.txt  its output. Helpers: go() continue to the next stop, snap() save the recording for stills, pause_at("a,b") change stops."""
import builtins, contextlib, os, sys, time, traceback
from pathlib import Path
ROOT = Path(r"D:\tea_robot\isaac_delivery")
TAG = sys.argv[1]
D = ROOT / "live" / TAG
D.mkdir(parents=True, exist_ok=True)
STOPS = set(x for x in os.environ.get("LIVE_PAUSE", "").split(",") if x)
_print = builtins.print
state = dict(n=0, busy=False, go=False)


def _scope():
    fr, frames = sys._getframe(3), []
    while fr is not None:
        frames.append(fr)
        fr = fr.f_back
    ns = {}
    for f in reversed(frames):
        ns.update(f.f_globals)
        ns.update(f.f_locals)
    return ns


def pause(where, row):
    ns = _scope()

    def go():
        state["go"] = True

    def snap():
        p = str(ROOT / "logs" / TAG) + "_rec.npz"
        ns["rec"].save(p)
        return p

    def pause_at(names):
        STOPS.clear()
        STOPS.update(x for x in names.split(",") if x)

    ns.update(go=go, snap=snap, pause_at=pause_at)
    (D / "paused.txt").write_text(f"{time.strftime('%H:%M:%S')} {where} {row}\n", encoding="utf-8")
    _print("[live] paused at", where, flush=True)
    state["go"] = False
    cmd = D / "cmd.py"
    while not state["go"]:
        if not cmd.exists():
            time.sleep(0.5)
            continue
        state["n"] += 1
        n = state["n"]
        run = D / f"cmd_{n}.py"
        os.replace(cmd, run)
        tmp = D / f"out_{n}.tmp"
        state["busy"] = True
        with open(tmp, "w", encoding="utf-8") as f, contextlib.redirect_stdout(f):
            try:
                exec(compile(run.read_text(encoding="utf-8"), str(run), "exec"), ns)
            except BaseException:
                traceback.print_exc(file=f)
        state["busy"] = False
        os.replace(tmp, D / f"out_{n}.txt")
    (D / "paused.txt").write_text(f"{time.strftime('%H:%M:%S')} running after {where}\n", encoding="utf-8")


def tap(*a, **k):
    _print(*a, **k)
    if state["busy"] or not a or not isinstance(a[0], str):
        return
    name = str(a[1]).strip() if len(a) > 1 else ""
    if "END status" in a[0]:
        pause("END", a[0][:200])
    elif name and len(a) > 2 and (name in STOPS or ("*" in STOPS and isinstance(a[2], dict))):
        pause(name, a[2])


builtins.print = tap
