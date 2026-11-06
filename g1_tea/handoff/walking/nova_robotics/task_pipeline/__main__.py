"""python -m nova_robotics.task_pipeline --help"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from .spec import load_request, resolve, write_json


def main():
    parser = argparse.ArgumentParser(
        description="Prompt → physics task → cameras → validated dataset"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "generate"):
        p = sub.add_parser(command)
        p.add_argument("prompt")
        p.add_argument(
            "--scene", required=True, help="Path to the existing café export environment.xml"
        )
        p.add_argument("--robot", required=True, help="Path to G1 g1_with_hands.xml")
        p.add_argument("--out", required=True)
        p.add_argument("--episodes", type=int)
        p.add_argument("--max-attempts", type=int)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--variation-m", type=float, default=0.005)
        p.add_argument("--max-wall-seconds", type=int, default=7200)
    p = sub.add_parser("run", help="Run or resume an immutable request")
    p.add_argument("request")
    p.add_argument("--out", required=True)
    for command in ("status", "pause", "resume", "cancel"):
        p = sub.add_parser(command)
        p.add_argument("job")
    p = sub.add_parser("replay")
    p.add_argument("episode")
    p = sub.add_parser("_worker", help=argparse.SUPPRESS)
    p.add_argument("kind", choices=["episode", "export"])
    p.add_argument("request")
    p.add_argument("out")
    p.add_argument("argument")
    args = parser.parse_args()
    if args.command in {"plan", "generate"}:
        request = resolve(
            args.prompt,
            scene=args.scene,
            robot=args.robot,
            episodes=args.episodes,
            max_attempts=args.max_attempts,
            seed=args.seed,
            variation_m=args.variation_m,
            max_wall_seconds=args.max_wall_seconds,
        )
        if args.command == "plan":
            path = Path(args.out)
            if path.exists():
                raise ValueError("Plan output already exists; choose a new filename")
            path.parent.mkdir(parents=True, exist_ok=True)
            write_json(path, request.to_dict())
            print(json.dumps(request.to_dict(), indent=2))
            return
        from .jobs import run_job

        run_job(request, args.out)
    elif args.command == "run":
        from .jobs import run_job

        run_job(load_request(args.request), args.out)
    elif args.command in {"status", "pause", "resume", "cancel"}:
        root = Path(args.job)
        if not (root / "job.json").is_file():
            raise ValueError("Unknown job directory")
        if args.command == "status":
            print((root / "job.json").read_text())
        elif args.command in {"pause", "cancel"}:
            (root / f"{args.command}.request").touch()
            print(f"{args.command} requested at the next attempt boundary")
        else:
            from .jobs import run_job

            (root / "pause.request").unlink(missing_ok=True)
            (root / "cancel.request").unlink(missing_ok=True)
            run_job(load_request(root / "request.json"), root)
    elif args.command == "replay":
        from .validation import replay

        print(json.dumps(replay(args.episode), indent=2))
    elif args.command == "_worker":
        request = load_request(args.request)
        if args.kind == "episode":
            from .jobs import episode_worker

            episode_worker(request, args.out, int(args.argument))
        else:
            from .export import export_dataset

            export_dataset(json.loads(Path(args.argument).read_text()), args.out, request)


if __name__ == "__main__":
    main()
