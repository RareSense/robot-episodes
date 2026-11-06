# Sizes and where the data goes

GitHub rejects any file over 100 MB in a normal git push, and warns above 50 MB. Sizes below are measured from this project's runs.

| File | Size | Plan |
|---|---|---|
| `code/`, docs, JSON, sanitised logs | Under 1 MB per run | Normal git |
| `assets/` (robot USD 14 MB, policy 127 KB) | 14 MB, largest file 13.9 MB | Normal git |
| `<tag>_rec.npz` | 19 to 45 MB in earlier part-runs. About 0.25 MB per sim second so far, so a long run could near 100 MB | Git LFS (`.gitattributes` already covers `*.npz`), or the data release |
| `<tag>.npz` | Written at the end of a run. Size unknown until one finishes | Same |
| Video `.mp4` | Tens of MB per view, depending on render settings | Git LFS, after branding is cleared (THIRD_PARTY.md). Until then kept in the git-ignored `runs/<tag>/video_hold/` |
| `<tag>_rec.npz.particles.f32` | **9.6 MB per sim second.** An earlier stopped run (`final03`) reached 2.28 GB. A full episode will likely pass 2 GB and may reach 3 to 4 GB | **Never in git. Separate data release** |

## Why the particle file cannot go in Git LFS or a plain GitHub release

- GitHub limits a single LFS object to 2 GB on Free and Pro plans, and LFS storage and bandwidth are metered. Check current numbers before relying on them.
- GitHub release assets must each be under 2 GiB.
- Cloning with LFS would pull gigabytes that most readers do not need.

## Options for the data release (user to choose)

| Host | Per-file limit | Notes |
|---|---|---|
| Hugging Face dataset | Large files accepted | Natural for robotics data, versioned, streams by file |
| Zenodo | 50 GB per record by default | Permanent DOI, citable, immutable |
| GitHub release assets | Under 2 GiB each | `collect_run.py --stage-data <folder>` splits the file into 1.9 GB parts with hashes. Join with `copy /b` or `cat` |

Recommendation: Hugging Face dataset, or Zenodo if a citable DOI matters more.

## How the repo refers to the data

`runs/<tag>/particles.ref.json` holds file name, size in bytes, sha256, sample count and `url`. The URL reads `NOT HOSTED YET` until the user uploads the file and edits it. `examples/load_rec.py` works without the particle file, and everything except the water reads from the recording.

Compression is untested. Float positions of a slowly moving fluid should compress somewhat; a compressed release would need a matching change in the loader.

## Before the first push

- Run `git lfs install --local` on the machine that pushes. `.gitattributes` routes `*.npz`, `*.mp4`, `*.mov` and `*.f32` to LFS.
- Run `python tools/scan_release.py`.
- `git count-objects -vH` and a look at the largest blobs: no blob over 100 MB may be in history, because deleting a file later does not remove it from history.
