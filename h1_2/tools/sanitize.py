"""Shared rules for finding and removing private strings. Used by export_code.py, collect_run.py and scan_release.py.

Nothing machine-specific is written in this file. The current user name, computer name and home folder come from the
environment at run time, and extra private terms come from an untracked file, .private_terms, one term per line.
Matched values are never printed: reports carry the kind of match and the line number only.
"""
import os
import re
from pathlib import Path

# Windows drive paths, with either slash, file: URLs, and lower-case drives as Kit writes them. The lookbehind stops https://.
WIN_PATH = re.compile(r"(?<![A-Za-z0-9])(?:file:/{1,3})?/?[A-Za-z]:[\\/]+[^\s'\"`,;()\[\]{}<>|]*")
UNIX_HOME = re.compile(r"(?<![A-Za-z0-9_])/(?:home|Users|mnt/scratch|root)/[^\s'\"`,;()\[\]{}<>|]+")
EMAIL = re.compile(r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@(?!np.)[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
PRIVATE_IP = re.compile(r"(?<![\d.])(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}(?![\d.])")
KEYS = [
    ("openrouter/openai style key", re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}")),
    ("github token", re.compile(r"(?<![A-Za-z0-9])(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})")),
    ("hugging face token", re.compile(r"(?<![A-Za-z0-9])hf_[A-Za-z0-9]{30,}")),
    ("aws key", re.compile(r"(?<![A-Za-z0-9])AKIA[0-9A-Z]{16}(?![A-Za-z0-9])")),
    ("google api key", re.compile(r"(?<![A-Za-z0-9])AIza[0-9A-Za-z_-]{35}")),
    ("slack token", re.compile(r"(?<![A-Za-z0-9])xox[baprs]-[A-Za-z0-9-]{10,}")),
    ("bearer header", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{20,}")),
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("key assignment", re.compile(r"(?i)\b(?:api[_-]?key|apikey|secret|token|passw(?:or)?d)\b\s*[=:]\s*['\"]?[A-Za-z0-9_\-/+=.]{16,}")),
    ("key file reference", re.compile(r"(?i)\.openrouter_key|openrouter[_-]?api")),
]
# Words that must not appear in any tracked file or commit. Provenance of EXCLUDED third-party textures is the one permitted
# use of "image model" wording, and it does not match these.
ATTRIBUTION = re.compile(r"(?i)\b(?:claude|anthropic|codex|chatgpt|openai|copilot|co-authored-by|generated with)\b")


def private_terms(repo=None):
    """User name, computer name, home-folder name, and any extra lines in .private_terms. Never printed."""
    out = set()
    for k in ("USERNAME", "USER", "COMPUTERNAME", "LOGNAME"):
        v = os.environ.get(k)
        if v and len(v) >= 3:
            out.add(v)
    home = os.environ.get("USERPROFILE") or os.environ.get("HOME")
    if home:
        n = Path(home).name
        if len(n) >= 3:
            out.add(n)
            out.add(n[:6] + "~1")                         # 8.3 short form of the home folder, as temp paths show it
    if repo:
        f = Path(repo) / ".private_terms"
        if f.exists():
            out |= {x.strip() for x in f.read_text(encoding="utf-8").splitlines() if x.strip() and not x.startswith("#")}
    return sorted(out, key=len, reverse=True)


def _short(m):
    """Keep the file name, which is what a reader of a log needs; drop every folder."""
    p = m.group(0).rstrip("\\/.:")
    base = re.split(r"[\\/]+", p)[-1]
    return "<path>/" + base if base and ":" not in base else "<path>"


def clean_text(text, terms=()):
    """Sanitised copy of a log or JSON text. Returns (text, {kind: count})."""
    n = {}

    def sub(rx, rep, s, kind):
        s, c = rx.subn(rep, s)
        if c:
            n[kind] = n.get(kind, 0) + c
        return s

    for kind, rx in KEYS:
        text = sub(rx, "<redacted-key>", text, kind)
    text = sub(WIN_PATH, _short, text, "windows path")
    text = sub(UNIX_HOME, _short, text, "home path")
    text = sub(EMAIL, "<email>", text, "email")
    text = sub(PRIVATE_IP, "<private-ip>", text, "private ip")
    for t in terms:
        text = sub(re.compile(re.escape(t), re.I), "<user>", text, "user or machine name")
    return text, n


def find_issues(text, terms=(), attribution=True):
    """[(line number, kind)] for one text. Used by the scanner; values are not returned."""
    found = []
    rules = [(k, r) for k, r in KEYS] + [("windows path", WIN_PATH), ("home path", UNIX_HOME), ("email", EMAIL), ("private ip", PRIVATE_IP)]
    if attribution:
        rules.append(("attribution word", ATTRIBUTION))
    rules += [("user or machine name", re.compile(re.escape(t), re.I)) for t in terms]
    for kind, rx in rules:
        for m in rx.finditer(text):
            found.append((text.count("\n", 0, m.start()) + 1, kind))
    return found


def decode(b):
    """Text from bytes. PowerShell 5.1 redirection writes UTF-16 with a BOM, so a plain byte search of those logs finds nothing."""
    if b[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return b.decode("utf-16", "replace")
    if b[:3] == b"\xef\xbb\xbf":
        return b[3:].decode("utf-8", "replace")
    head = b[:4096]
    if head.count(b"\x00") > len(head) // 4 and len(head) > 16:      # UTF-16 without a BOM
        return b.decode("utf-16-le" if head[1:2] == b"\x00" else "utf-16-be", "replace")
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return b.decode("latin-1")
