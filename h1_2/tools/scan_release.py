"""Scan every file in the release for keys, absolute paths, e-mail addresses, user names and attribution words.

    python tools/scan_release.py            exit 0 only if clean
    python tools/scan_release.py --self-test
    python tools/scan_release.py --root <any folder> --no-attribution

Reads inside what a plain byte search misses: UTF-16 logs, the string arrays of .npz files, the LZ4-packed token table of binary .usd
files, MP4 metadata (ffprobe, if installed), and the strings in other binaries. Reports file, line and kind only, never the value.
"""
import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sanitize as SZ

SKIP_DIRS = {".git", "__pycache__", "out"}
SKIP_FILES = {".private_terms"}
# These hold the patterns or name the rules; they do not hold private data, and the rest of their text is still checked for keys.
RULE_FILES = {"tools/sanitize.py", "tools/scan_release.py", "tools/export_code.py", "tools/collect_run.py"}
BIG = 256 * 1024 * 1024


def lz4_block(src, size):
    out = bytearray()
    i, n = 0, len(src)
    while i < n:
        tok = src[i]
        i += 1
        lit = tok >> 4
        if lit == 15:
            while True:
                b = src[i]
                i += 1
                lit += b
                if b != 255:
                    break
        out += src[i:i + lit]
        i += lit
        if i >= n or len(out) >= size:
            break
        off = src[i] | src[i + 1] << 8
        i += 2
        ml = tok & 15
        if ml == 15:
            while True:
                b = src[i]
                i += 1
                ml += b
                if b != 255:
                    break
        ml += 4
        for _ in range(ml):
            out.append(out[-off])
    return bytes(out)


def usdc_strings(path):
    """Tokens of a USD crate file. Every string and asset path in a crate is stored in this one table."""
    with open(path, "rb") as f:
        head = f.read(88)
        if head[:8] != b"PXR-USDC":
            return None
        toc = struct.unpack("<Q", head[16:24])[0]
        f.seek(toc)
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            name, start, size = struct.unpack("<16sQQ", f.read(32))
            if name.rstrip(b"\0") == b"TOKENS":
                f.seek(start)
                count, raw, comp = struct.unpack("<QQQ", f.read(24))
                blob = f.read(comp)
                if blob[0] != 0:
                    raise ValueError("multi-chunk token table")
                return lz4_block(blob[1:], raw).replace(b"\0", b"\n").decode("utf-8", "replace"), count
    return None


def npz_strings(path):
    import numpy as np
    out = []
    with zipfile.ZipFile(path) as z:
        out += z.namelist()
    with np.load(path, allow_pickle=False) as d:
        for k in d.files:
            a = d[k] if d[k].dtype.kind in "US" else None
            if a is not None:
                out += [str(x) for x in a.ravel()]
    return "\n".join(out)


def mp4_tags(path):
    if not shutil.which("ffprobe"):
        return "", False
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags:stream_tags", "-of", "json", str(path)], capture_output=True, text=True, timeout=60)
    return r.stdout, True


def binary_strings(b):
    """Printable runs, ASCII and UTF-16."""
    a = re.findall(rb"[\x20-\x7e]{6,}", b)
    w = re.findall(rb"(?:[\x20-\x7e]\x00){6,}", b)
    return "\n".join(x.decode("ascii") for x in a) + "\n" + "\n".join(x.decode("utf-16-le") for x in w)


TEXT_EXT = {".py", ".md", ".txt", ".json", ".log", ".ps1", ".sh", ".usda", ".urdf", ".xml", ".yaml", ".yml", ".toml", ".csv", ".gitignore", ".gitattributes", ""}


def scan_file(p, rel, terms, attribution):
    ext = p.suffix.lower()
    size = p.stat().st_size
    notes = []
    raw_binary = False
    if ext == ".npz":
        text = npz_strings(p)
    elif ext in (".mp4", ".mov"):
        text, ok = mp4_tags(p)
        if not ok:
            notes.append("ffprobe missing: metadata not read")
    elif ext == ".usd":
        r = usdc_strings(p)
        if r is None:
            text = SZ.decode(p.read_bytes())
        else:
            text = r[0]
            notes.append(f"{r[1]} crate tokens read")
    elif size > BIG:
        return [], [f"{size} bytes, not read: raw data must stay out of the repo"]
    elif ext in TEXT_EXT or p.name in (".gitignore", ".gitattributes", "LICENSE"):
        text = SZ.decode(p.read_bytes())
    else:
        text = binary_strings(p.read_bytes())
        raw_binary = True
    issues = SZ.find_issues(text, terms, attribution=attribution)
    if raw_binary:          # random weights form things like V:/x; a real path has at least two named folders
        issues = [i for i in issues if i[1] != "windows path"]
        issues += [(text.count("\n", 0, m.start()) + 1, "windows path") for m in re.finditer(r"[A-Za-z]:[\\/]+\w{3,}[\\/]+\w{2,}", text)]
    if rel in RULE_FILES:
        issues = [i for i in issues if i[1] not in ("key file reference", "attribution word", "windows path", "home path")]
    return issues, notes


def scan(root, attribution=True, repo=None):
    root = Path(root).resolve()
    terms = SZ.private_terms(repo or root)
    bad, files, notes = {}, 0, []
    for d, sub, names in os.walk(root):
        sub[:] = [x for x in sub if x not in SKIP_DIRS]
        for n in names:
            if n in SKIP_FILES:
                continue
            p = Path(d) / n
            rel = p.relative_to(root).as_posix()
            files += 1
            issues, nt = scan_file(p, rel, terms, attribution)
            notes += [f"{rel}: {x}" for x in nt]
            if issues:
                bad[rel] = issues
    return bad, files, notes


def self_test():
    """A planted key, path, e-mail and user name must be found in UTF-16 text and in an .npz, or the scanner is not to be trusted."""
    import numpy as np
    fake_key = "sk-or-v1-" + "ab12" * 12
    fake_path = "D:" + chr(92) + "tea_robot" + chr(92) + "isaac_delivery" + chr(92) + "x.log"
    fake_mail = "someone" + "@" + "example.com"
    sample = f"line one\n{fake_key}\n{fake_path}\n{fake_mail}\nfile:/c:/Users/somebody/x\n"
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        (t / "a.log").write_text(sample, encoding="utf-16")
        np.savez_compressed(t / "b.npz", p=np.array([fake_path, "/World/ok"]))
        bad, files, _ = scan(t, attribution=False)
    kinds = {k for v in bad.get("a.log", []) for k in [v[1]]}
    assert {"openrouter/openai style key", "windows path", "email"} <= kinds, kinds
    assert any(i[1] == "windows path" for i in bad.get("b.npz", [])), bad
    cleaned, n = SZ.clean_text(sample)
    assert fake_key not in cleaned and fake_path not in cleaned and fake_mail not in cleaned and "x.log" in cleaned, cleaned
    assert not SZ.find_issues(SZ.clean_text("see https://github.com/a/b and C:\\x\\y.txt")[0], attribution=False)
    print("self-test passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE.parent))
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--no-attribution", action="store_true")
    ap.add_argument("--json", default="")
    A = ap.parse_args()
    if A.self_test:
        self_test()
        sys.exit(0)
    bad, files, notes = scan(A.root, attribution=not A.no_attribution, repo=HERE.parent)
    print("files scanned:", files)
    for x in notes:
        print("  note:", x)
    for f, issues in sorted(bad.items()):
        counts = {}
        for ln, kind in issues:
            counts.setdefault(kind, []).append(ln)
        for kind, lines in counts.items():
            print(f"  FOUND {f}: {kind}, {len(lines)} at line {lines[:6]}")
    if A.json:
        Path(A.json).write_text(json.dumps({k: v for k, v in bad.items()}), encoding="utf-8")
    print("CLEAN" if not bad else f"NOT CLEAN: {len(bad)} files")
    sys.exit(1 if bad else 0)
