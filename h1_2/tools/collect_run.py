"""Bring one finished run into the release. The source tree is only read. No simulator is started and nothing is uploaded.

    python tools/collect_run.py <tag> --source <working tree>
    python tools/collect_run.py <tag> --source <working tree> --update-docs        also fill the marked sections of README.md and DISCLOSURES.md
    python tools/collect_run.py <tag> --source <working tree> --refresh-code       also re-export code/ from the tree as it stands now
    python tools/collect_run.py <tag> --source <working tree> --stage-data <folder>  also write the particle file there in parts, for the data release

Writes runs/<tag>/:
    RUN_STATUS.json     what kind of run this is, judged from its own report
    RESULTS.md          gate and step tables, generated from the JSON, nothing typed by hand
    <tag>.log           sanitised, UTF-8
    <tag>.json, <tag>_gates.json, <tag>_partial.json, <tag>_process.json, <tag>_sources.json   sanitised
    <tag>_rec.npz, <tag>.npz   copied unchanged, after their string arrays are checked
    particles.ref.json  size, hash and sample count of the particle file; the file itself stays out of git
    CODE_PROVENANCE.json  each released code file against the hash taken at launch, and any late-bound stage the log names
    video_hold/         videos, metadata stripped, git-ignored until cleared (see below)
    MANIFEST.json       sha256 of everything above

Kinds of run, from the run's own report:
    complete_full_run   full_episode report, from the entrance, GPU, live water, result "finished"
    stopped_full_run    same report, ended early (a fall, a lost object)
    test_setup          any other report: a step test, a run started part way
    incomplete          no report at all: still running, or killed. Refused unless --allow-incomplete.
--update-docs works only for complete_full_run or stopped_full_run, so a test setup can never land in the results section.

Videos render the look layers, whose tray textures show trademarked packaging (THIRD_PARTY.md). They go to video_hold/, which git ignores,
unless --video-cleared says the user has checked that file or rendered it with --no-looks.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import export_code
import sanitize as SZ

GATE_ORDER = ("fall", "torque", "penetration", "slip", "moved", "spill")
STEP_ORDER = ("mug", "teabag", "kettle", "powder", "stir", "cup_walk_away")
GITHUB_LIMIT = 100 * 1000 * 1000
PART = 1900 * 1000 * 1000            # GitHub release assets stop at 2 GiB


def sha(path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def read_json(path):
    return json.loads(SZ.decode(Path(path).read_bytes()))


def classify(tag, logs):
    """(kind, reason). Only the run's own report decides; an exit code or a stage line never does."""
    f = logs / f"{tag}.json"
    if not f.exists():
        alive = logs / f"{tag}_process.json"
        return "incomplete", "no final report" + ("; launcher finished, so the run stopped before writing one" if alive.exists() else "; launcher has not finished, it may still be running")
    d = read_json(f)
    sim = d.get("simulation")
    if d.get("test_setup") or not isinstance(sim, dict):
        return "test_setup", "report is not a full-episode report"
    full = sim.get("continuous_from_entrance") and sim.get("live_water") and sim.get("mode") == "gpu"
    if not full:
        return "test_setup", f"not a continuous GPU run with live water: {sim}"
    if d.get("result") == "finished":
        return "complete_full_run", "full-episode report, result finished"
    return "stopped_full_run", f"full-episode report, ended early: {d.get('result')}"


def sanitised_copy(src, dst, terms, as_json):
    text, counts = SZ.clean_text(SZ.decode(Path(src).read_bytes()), terms)
    if as_json:
        json.loads(text)
    Path(dst).write_text(text, encoding="utf-8", newline="\n")
    return counts


def sources_relative(src, dst, root):
    """<tag>_sources.json with paths made relative to the working tree."""
    rows = read_json(src)
    rows = rows if isinstance(rows, list) else [rows]
    root_s = str(root).replace("/", "\\").lower()
    out = []
    for r in rows:
        p = str(r.get("Path", "")).replace("/", "\\")
        rel = p[len(root_s) + 1:] if p.lower().startswith(root_s) else Path(p).name
        out.append(dict(file=rel.replace("\\", "/"), sha256=str(r.get("SHA256", "")).lower()))
    Path(dst).write_text(json.dumps(out, indent=0), encoding="utf-8")
    return {x["file"]: x["sha256"] for x in out}


def check_rec(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as d:
        keys = set(d.files)
        need = {"t", "fps", "link_paths", "link_T", "body_paths", "body_T"}
        info = dict(keys=sorted(keys), missing=sorted(need - keys))
        if not info["missing"]:
            t, L, B = d["t"], d["link_T"], d["body_T"]
            info.update(frames=int(len(t)), sim_seconds=float(t[-1]) if len(t) else 0.0, fps=float(d["fps"]), links=int(L.shape[1]), bodies=int(B.shape[1]),
                        shapes_agree=bool(L.shape[0] == B.shape[0] == len(t)))
        if "p_t" in keys:
            pn = d["p_n"]
            info.update(particle_samples=int(len(pn)), particle_points=int(pn.sum()), particle_bytes_expected=int(pn.sum()) * 12, particle_last_count=int(pn[-1]) if len(pn) else 0)
    return info


def strip_video(src, dst):
    if shutil.which("ffmpeg"):
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-map_metadata", "-1", "-map", "0", "-c", "copy", str(dst)], capture_output=True, text=True)
        if r.returncode == 0:
            return "metadata stripped"
    shutil.copy2(src, dst)
    return "copied as is: run tools/scan_release.py before clearing"


def provenance(tag, logs, source, launched, process, log_text):
    finish = None
    if process:
        try:
            finish = datetime.fromisoformat(str(process.get("FinishedUTC"))[:26].rstrip("Z")).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            pass
    rows = []
    for src, dst in export_code.FILES:
        p = (Path(source) / src).resolve()
        now = sha(p)
        rel = src if not src.startswith("..") else None
        at_launch = launched.get(rel) if rel else None
        row = dict(file=dst, sha256_now=now, sha256_at_launch=at_launch, changed_since_launch=None if at_launch is None else now != at_launch,
                   modified_utc=datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(timespec="seconds"))
        if finish is not None:
            row["modified_after_run_finished"] = p.stat().st_mtime > finish
        rows.append(row)
    late = sorted(set(re.findall(r"\[late-bind\] using \S*?([\w-]+)[\\/]stage_live\.py for (\w+)", log_text)))
    moved = [r["file"] for r in rows if r.get("changed_since_launch")]
    after = [r["file"] for r in rows if r.get("modified_after_run_finished")]
    return dict(rule="Stages bind late: steps/<step>/stage_live.py and cup_walk_away.py are read when the step starts, so a file changed after launch may be what ran. "
                     "A file changed after the run finished cannot be what ran.",
                late_bound_in_log=[dict(step_folder=a, function=b) for a, b in late], changed_since_launch=moved, modified_after_run_finished=after, files=rows)


def fmt(v):
    if v is None:
        return "not measured"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def results_md(tag, kind, reason, report, gates, rec, part):
    """Tables from the JSON files. Nothing here is typed in by hand."""
    o = [f"Run tag: `{tag}`", f"Kind of run: **{kind}** ({reason})", ""]
    if report:
        o += [f"- Strict verdict (`full_episode_passed`): **{report.get('overall', 'not reported')}**", f"- Run result line: {report.get('result')}",
              f"- Simulated time: {report.get('sim_s')} s, wall time: {report.get('wall_s')} s"]
        td = report.get("task_done")
        if td:
            o += [f"- Task-level verdict (working rule, not the strict verdict): **{'PASS' if td.get('passed') else 'FAIL'}**"]
        if report.get("report_error"):
            o += [f"- Report error: {report['report_error']}"]
        o += [""]
    if gates:
        G = gates.get("gates", {})
        o += ["| Gate | Status | Value | Unit | Limit | Samples | Note |", "|---|---|---|---|---|---|---|"]
        for k in GATE_ORDER:
            g = G.get(k, {})
            unit = str(g.get("unit", "")).replace("|", "\\|")        # "worst |tau|/limit" would split the table
            o.append(f"| {k} | {g.get('status', 'not reported')} | {fmt(g.get('value'))} | {unit} | {g.get('limit', '')} | {g.get('samples', '')} | {str(g.get('note', '')).replace('|', '/')} |")
        t = G.get("torque", {})
        if t:
            o += ["", f"Torque detail: worst joint {t.get('joint')}, {fmt(t.get('nm'))} Nm at t {fmt(t.get('at_t'))} s, {t.get('steps_over')} steps over, longest run {t.get('longest_run_steps')} steps, "
                      f"motor estimate {fmt(t.get('motor_estimate_ratio'))}."]
        o += [""]
    steps = (report or {}).get("gates", {}).get("steps", {}) if report else {}
    if steps:
        o += ["| Step | Strict status | Task status | Reason |", "|---|---|---|---|"]
        for k in STEP_ORDER:
            s = steps.get(k, {})
            o.append(f"| {k} | {s.get('status', 'not reported')} | {s.get('task_status', 'not reported')} | {str(s.get('reason') or s.get('why') or s.get('task_reason') or '').replace('|', '/')[:160]} |")
        w = {k: steps[k].get("water") for k in STEP_ORDER if isinstance(steps.get(k, {}).get("water"), dict)}
        if w:
            o += ["", "| After step | Mug ml | Spill ml | Kettle ml |", "|---|---|---|---|"]
            o += [f"| {k} | {v.get('mug_ml')} | {v.get('spill_ml')} | {v.get('kettle_ml')} |" for k, v in w.items()]
        o += [""]
    if rec:
        o += [f"Recording: {rec.get('frames')} frames at {rec.get('fps')} fps, {rec.get('sim_seconds')} s, {rec.get('links')} links, {rec.get('bodies')} bodies."]
    if part:
        o += [f"Particle file: {part['bytes']} bytes, {part.get('samples')} samples, size check {part.get('size_matches_rec')}."]
    return "\n".join(o) + "\n"


def replace_between(path, begin, end, body):
    s = Path(path).read_text(encoding="utf-8")
    if begin not in s or end not in s:
        raise SystemExit(f"markers missing in {path}")
    a, b = s.index(begin) + len(begin), s.index(end)
    Path(path).write_text(s[:a] + "\n" + body + s[b:], encoding="utf-8", newline="\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tag")
    ap.add_argument("--source", default=os.environ.get("H12_SOURCE_ROOT"), help="working tree, read only (or set H12_SOURCE_ROOT)")
    ap.add_argument("--repo", default=str(HERE.parent))
    ap.add_argument("--video", action="append", default=[], help="a video file; repeat for more. <tag>*.mp4 in previews/ is found anyway")
    ap.add_argument("--video-cleared", action="store_true", help="the user has checked these videos for branding: put them where git tracks them")
    ap.add_argument("--allow-incomplete", action="store_true", help="collect a run with no final report; it is marked as such")
    ap.add_argument("--update-docs", action="store_true")
    ap.add_argument("--refresh-code", action="store_true")
    ap.add_argument("--stage-data", default="", help="folder outside the repo for the data release")
    ap.add_argument("--no-hash", action="store_true", help="skip hashing the particle file, for a dry run")
    A = ap.parse_args()
    if not A.source:
        ap.error("--source is required")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", A.tag):
        ap.error("bad tag")
    source, repo, tag = Path(A.source).resolve(), Path(A.repo).resolve(), A.tag
    logs = source / "logs"
    if not (logs / f"{tag}.log").exists():
        raise SystemExit(f"no log for tag {tag}")
    kind, reason = classify(tag, logs)
    print("run kind:", kind, "-", reason)
    if kind == "incomplete" and not A.allow_incomplete:
        raise SystemExit("refused: this run has no final report. Wait for it to end, or pass --allow-incomplete to take a labelled partial copy.")
    if A.update_docs and kind not in ("complete_full_run", "stopped_full_run"):
        raise SystemExit("refused: only a full-episode report may fill the results section.")
    if not (repo / "README.md").exists():
        raise SystemExit("--repo is not the release")

    out = repo / "runs" / tag
    out.mkdir(parents=True, exist_ok=True)
    terms = SZ.private_terms(repo)
    redactions = {}

    def tally(c):
        for k, v in c.items():
            redactions[k] = redactions.get(k, 0) + v

    tally(sanitised_copy(logs / f"{tag}.log", out / f"{tag}.log", terms, False))
    for suffix in (".json", "_gates.json", "_partial.json", "_process.json"):
        f = logs / f"{tag}{suffix}"
        if f.exists():
            tally(sanitised_copy(f, out / f.name, terms, True))
    launched = {}
    if (logs / f"{tag}_sources.json").exists():
        launched = sources_relative(logs / f"{tag}_sources.json", out / f"{tag}_sources.json", source)

    rec = None
    f = logs / f"{tag}_rec.npz"
    if f.exists():
        rec = check_rec(f)
        shutil.copy2(f, out / f.name)
        if f.stat().st_size >= GITHUB_LIMIT:
            print("WARNING: recording is over GitHub's 100 MB file limit; .gitattributes sends *.npz to Git LFS, or move it to the data release")
    f = logs / f"{tag}.npz"
    if f.exists():
        shutil.copy2(f, out / f.name)

    part = None
    f = logs / f"{tag}_rec.npz.particles.f32"
    if f.exists():
        n = f.stat().st_size
        part = dict(file=f.name, bytes=n, dtype="float32 little-endian, xyz, metres, world frame", sha256=None if A.no_hash else sha(f),
                    samples=rec.get("particle_samples") if rec else None, url="NOT HOSTED YET",
                    size_matches_rec=(rec or {}).get("particle_bytes_expected") == n)
        if rec and not part["size_matches_rec"]:
            part["note"] = "file and p_n disagree: the recording was saved before the last samples were written, or the run did not save at its end"
        (out / "particles.ref.json").write_text(json.dumps(part, indent=1), encoding="utf-8")
        if A.stage_data:
            d = Path(A.stage_data).resolve()
            if repo in d.parents or d == repo:
                raise SystemExit("--stage-data must be outside the repo")
            d.mkdir(parents=True, exist_ok=True)
            parts = []
            with open(f, "rb") as fh:
                for i in range((n + PART - 1) // PART):
                    name = f"{f.name}.part{i:02d}"
                    h = hashlib.sha256()
                    with open(d / name, "wb") as w:
                        left = PART
                        while left:
                            b = fh.read(min(8 << 20, left))
                            if not b:
                                break
                            w.write(b)
                            h.update(b)
                            left -= len(b)
                    parts.append(dict(file=name, sha256=h.hexdigest()))
            part["parts"] = parts
            (out / "particles.ref.json").write_text(json.dumps(part, indent=1), encoding="utf-8")

    vids = sorted((source / "previews").glob(f"{tag}*.mp4")) + [Path(v) for v in A.video]
    held = []
    if vids:
        vd = out / ("video" if A.video_cleared else "video_hold")
        vd.mkdir(exist_ok=True)
        for v in vids:
            held.append(dict(file=v.name, how=strip_video(v, vd / v.name), folder=vd.name))
        if not A.video_cleared:
            print("videos are in video_hold/, ignored by git: the tray shows trademarked packaging unless rendered with --no-looks. See THIRD_PARTY.md.")

    log_text = (out / f"{tag}.log").read_text(encoding="utf-8")
    process = read_json(logs / f"{tag}_process.json") if (logs / f"{tag}_process.json").exists() else None
    prov = provenance(tag, logs, source, launched, process, SZ.decode((logs / f"{tag}.log").read_bytes()))
    (out / "CODE_PROVENANCE.json").write_text(json.dumps(prov, indent=1), encoding="utf-8")
    if prov["modified_after_run_finished"]:
        print("WARNING: changed after the run finished, so code/ may not match what ran:", prov["modified_after_run_finished"])
    if A.refresh_code:
        export_code.export(source, repo)

    report = json.loads((out / f"{tag}.json").read_text(encoding="utf-8")) if (out / f"{tag}.json").exists() else None
    gates = json.loads((out / f"{tag}_gates.json").read_text(encoding="utf-8")) if (out / f"{tag}_gates.json").exists() else None
    status = dict(tag=tag, kind=kind, reason=reason, is_final_run_result=kind == "complete_full_run", exit_code=(process or {}).get("ExitCode"),
                  collected_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), redactions=redactions, videos=held)
    (out / "RUN_STATUS.json").write_text(json.dumps(status, indent=1), encoding="utf-8")
    md = results_md(tag, kind, reason, report, gates, rec, part)
    (out / "RESULTS.md").write_text(md, encoding="utf-8", newline="\n")

    manifest = {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name != "MANIFEST.json"}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=0), encoding="utf-8")

    if A.update_docs:
        replace_between(repo / "README.md", "<!-- RESULTS:BEGIN -->", "<!-- RESULTS:END -->", md)
        tq = (gates or {}).get("gates", {})
        rows = ["| Gate | Status | Value | Limit |", "|---|---|---|---|"] + [f"| {k} | {tq.get(k, {}).get('status')} | {fmt(tq.get(k, {}).get('value'))} | {tq.get(k, {}).get('limit')} |" for k in ("torque", "slip", "spill")]
        rows += ["", f"Kettle strict status: {(report or {}).get('gates', {}).get('steps', {}).get('kettle', {}).get('status')}, pour_tilt_reached "
                     f"{(report or {}).get('gates', {}).get('steps', {}).get('kettle', {}).get('numbers', {}).get('pour_tilt_reached')} deg.", ""]
        replace_between(repo / "DISCLOSURES.md", "<!-- FINAL-GATES:BEGIN -->", "<!-- FINAL-GATES:END -->", f"From run `{tag}` ({kind}):\n\n" + "\n".join(rows))
    print("collected into", out)
    print("redactions:", redactions)
    import scan_release
    bad, n, _ = scan_release.scan(out, repo=repo)
    print("scan of this run:", "CLEAN" if not bad else "NOT CLEAN")
    for k, v in sorted(bad.items()):
        print("  ", k, sorted({i[1] for i in v}))
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
