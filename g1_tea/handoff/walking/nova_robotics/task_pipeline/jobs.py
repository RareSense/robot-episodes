"""Resumable single-machine worker orchestration, isolated from project history.

These directories are portable job artifacts, not a replacement for Nova's PostgreSQL
ledger. A service integration must authorize with ProjectAccessPolicy and register a
Candidate through the Project Kernel; this worker never promotes Accepted or publishes.
"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from .spec import sha256, write_json


@contextmanager
def job_lock(root):
    with (root / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another worker already owns this job") from error
        yield


def code_fingerprint():
    runtime = Path(__file__).resolve().parents[1]
    files = [
        *runtime.glob("task_pipeline/*.py"),
        *runtime.glob("byor/*.py"),
        *runtime.glob("policies/g1_walk_v1/*"),
    ]
    return {
        str(path.relative_to(runtime)): sha256(path) for path in sorted(files) if path.is_file()
    }


def run_worker(arguments, log_path, timeout):
    env = dict(os.environ, MUJOCO_GL="osmesa", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    with log_path.open("a") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "nova_robotics.task_pipeline", "_worker", *arguments],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        try:
            return process.wait(timeout=max(0.1, timeout))
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            import signal

            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise


def inventory(root):
    return [
        {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name
        not in {
            "artifacts.json",
            "job.json",
            "events.jsonl",
            ".lock",
            "pause.request",
            "cancel.request",
        }
    ]


def run_job(request, root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with job_lock(root):
        state_path = root / "job.json"
        start = time.monotonic()
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state["request_fingerprint"] != request.fingerprint:
                raise ValueError("Resume must use the original immutable request")
            if json.loads((root / "code_fingerprint.json").read_text()) != code_fingerprint():
                raise ValueError("Worker code changed; start a new job to preserve reproducibility")
            if state["state"] == "complete":
                return state
            # The lock proves the former worker is gone. Never count an uncommitted attempt.
            for attempt in state["attempts"]:
                if attempt["state"] == "running":
                    attempt.update(
                        state="interrupted", usable=False, reason="Previous worker interrupted"
                    )
        else:
            write_json(root / "request.json", request.to_dict())
            write_json(root / "code_fingerprint.json", code_fingerprint())
            state = dict(
                schema="nova.offline-task-job/v1",
                request_fingerprint=request.fingerprint,
                state="running",
                attempts=[],
                successful=0,
                worker_seconds=0.0,
                target=request.episodes,
                dataset=None,
                learning_evaluated=False,
            )
        spent = state["worker_seconds"]

        def save(event):
            state["worker_seconds"] = spent + time.monotonic() - start
            write_json(state_path, state)
            with (root / "events.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        dict(
                            event=event,
                            time_unix=time.time(),
                            successful=state["successful"],
                            state=state["state"],
                        )
                    )
                    + "\n"
                )
            print(json.dumps(dict(event=event, **state)), flush=True)

        def remaining():
            return request.max_wall_seconds - spent - (time.monotonic() - start)

        save("started_or_resumed")
        try:
            while (
                state["successful"] < request.episodes
                and len(state["attempts"]) < request.max_attempts
            ):
                if (root / "pause.request").exists():
                    state["state"] = "paused"
                    save("paused_at_attempt_boundary")
                    return state
                if (root / "cancel.request").exists():
                    state["state"] = "cancelled"
                    save("cancelled_at_attempt_boundary")
                    return state
                if remaining() <= 0:
                    break
                number = len(state["attempts"])
                directory = root / f"attempt_{number:04d}"
                attempt = dict(
                    index=number,
                    seed=request.seed + number,
                    path=directory.name,
                    state="running",
                    usable=False,
                )
                state["attempts"].append(attempt)
                save("attempt_started")
                try:
                    code = run_worker(
                        [
                            "episode",
                            str(root / "request.json"),
                            str(directory),
                            str(attempt["seed"]),
                        ],
                        root / f"attempt_{number:04d}.log",
                        min(request.attempt_seconds, remaining()),
                    )
                    report_path = directory / "validation.json"
                    if code or not report_path.exists():
                        attempt.update(
                            state="failed", reason=f"Worker exit {code}; inspect attempt log"
                        )
                    else:
                        report = json.loads(report_path.read_text())
                        attempt.update(
                            state="succeeded" if report["usable"] else "failed",
                            usable=report["usable"],
                            validation=report,
                        )
                        if report["usable"]:
                            state["successful"] += 1
                except subprocess.TimeoutExpired:
                    attempt.update(
                        state="timed_out", reason="Attempt or total compute budget exhausted"
                    )
                save("attempt_finished")
            state["state"] = "exporting"
            save("export_started")
            episodes = [str(root / a["path"]) for a in state["attempts"] if a["usable"]]
            if episodes and remaining() > 0:
                export_index = len(list(root.glob("dataset_*")))
                export_path = root / f"dataset_{export_index:03d}"
                input_path = root / f"export_{export_index:03d}.json"
                write_json(input_path, episodes)
                code = run_worker(
                    ["export", str(root / "request.json"), str(export_path), str(input_path)],
                    root / f"export_{export_index:03d}.log",
                    remaining(),
                )
                if code:
                    state["state"] = "export_failed"
                    save("export_failed")
                    return state
                state["dataset"] = str(export_path.relative_to(root))
                state["export_validation"] = json.loads(
                    (export_path / "export_validation.json").read_text()
                )
            state["state"] = (
                "complete"
                if state["successful"] >= request.episodes and state["dataset"]
                else "partial"
            )
            state["stop_reason"] = (
                "target_reached"
                if state["state"] == "complete"
                else "budget_or_attempt_limit_exhausted"
            )
            write_json(root / "artifacts.json", inventory(root))
            save("finished")
            return state
        except (KeyboardInterrupt, subprocess.TimeoutExpired) as error:
            state["state"] = "paused" if isinstance(error, KeyboardInterrupt) else "partial"
            save("interrupted" if isinstance(error, KeyboardInterrupt) else "budget_exhausted")
            return state


def episode_worker(request, out, seed):
    from .cafe import run_episode
    from .rendering import render_episode
    from .validation import replay

    report = run_episode(request, seed, out)
    # Retain camera evidence for failed attempts too, provided there is an interval.
    if report["observation_intervals"]:
        cameras = render_episode(out, request)
        audit = replay(out)
        report.update(camera_valid=cameras["passed"], replay_valid=audit["passed"])
        report["usable"] &= cameras["passed"] and audit["passed"]
    else:
        report.update(camera_valid=False, replay_valid=False, usable=False)
    write_json(Path(out) / "validation.json", report)
    return report
