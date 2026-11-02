#!/usr/bin/env python3
"""Package Walker's Git-ignored research artifacts for a Zenodo data record.

Collects every file Git ignores that belongs to the research record (experiment
outputs, datasets, checkpoints, traces and backups), leaving out caches, virtual
environments, compiled binaries, machine configuration and credentials. Files are
grouped into .tar.gz archives that keep repository-relative paths, so extracting
them in the repository root restores the original layout.

Usage (from anywhere; Python 3.8+, standard library only):
  python3 tools/zenodo/package_data.py plan --repo <repo> [--include-logs]
  python3 tools/zenodo/package_data.py run  --repo <repo> --out <folder> [--include-logs]

`plan` lists the archives, file counts and sizes without writing anything.
`run` writes the archives, MANIFEST.tsv (every file with its size and archive),
SHA256SUMS (checksums of the archives) and README.md (description for the record).
"""
import argparse, datetime, hashlib, os, shutil, subprocess, sys, tarfile
from collections import OrderedDict
from pathlib import PurePosixPath

# Directories whose whole contents form the research record (kept except caches).
RECORD_DIRS = ('experiments', 'artifacts', 'evaluation/raw')
BACKUP_PREFIXES = ('evaluation_backup_', 'gail_backup_', 'dagger_backup_')
# Research file types kept wherever they appear.
DATA_SUFFIXES = ('.npz', '.npy', '.pth', '.pt', '.ckpt', '.onnx', '.h5', '.hdf5')
DATA_NAMES = ('expert_split_metadata.json',)
# Never included.
SKIP_PARTS = ('__pycache__', '.venv', 'venv', 'env', '.conda', '.pytest_cache',
              '.mypy_cache', '.ruff_cache', 'htmlcov', '.git')
SKIP_SUFFIXES = ('.pyc', '.pyo', '.pyd', '.so', '.o', '.a', '.c_info', '.pid', '.lock')
SKIP_NAMES = ('config.json', '.env', '.coverage')
ZENODO_MAX_BYTES, ZENODO_MAX_FILES = 50_000_000_000, 100
SPLIT_BYTES = 5_000_000_000   # archives larger than this are split along subfolders


def ignored_files(repo):
    out = subprocess.run(['git', '-C', repo, 'ls-files', '--others', '--ignored',
                          '--exclude-standard', '-z'], check=True, capture_output=True).stdout
    return [p for p in out.decode('utf-8', 'surrogateescape').split('\0') if p]


def classify(path, include_logs):
    """Return the archive name for a path, or None to leave it out."""
    p = PurePosixPath(path)
    parts = p.parts
    if any(x in SKIP_PARTS for x in parts) or p.name in SKIP_NAMES or p.name.startswith('.env') \
            or p.suffix in SKIP_SUFFIXES or p.name.endswith('.egg-info'):
        return None
    top = parts[0]
    if top == 'experiments':           # one archive per experiment family
        return 'experiments_' + parts[1] if len(parts) > 2 else 'experiments_root'
    if top == 'artifacts':
        return 'artifacts'
    if path.startswith('evaluation/raw/'):
        return 'evaluation_raw'
    if top.startswith(BACKUP_PREFIXES):
        return 'backup_' + top
    if top == 'logs':
        return 'logs' if include_logs else None
    if p.suffix in DATA_SUFFIXES or p.name in DATA_NAMES:
        return 'models_and_datasets'
    return None


def human(n):
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n < 1000 or unit == 'TB':
            return f'{n:.1f} {unit}' if unit != 'B' else f'{n} B'
        n /= 1000


def split(name, files, depth):
    """Split a large experiments_* group along the path component at `depth`."""
    if sum(s for _, s in files) <= SPLIT_BYTES or not name.startswith('experiments_'):
        return {name: files}
    sub = OrderedDict()
    for path, size in files:
        parts = PurePosixPath(path).parts
        key = parts[depth] if len(parts) > depth + 1 else '_files'
        sub.setdefault(key, []).append((path, size))
    if len(sub) == 1 and '_files' not in sub:      # single subfolder: look one level deeper
        return split(name, files, depth + 1) if depth < 8 else {name: files}
    out = {}
    for key, part in sub.items():
        out.update(split(f'{name}_{key}', part, depth + 1) if key != '_files' else {f'{name}_files': part})
    return out


def collect(repo, include_logs):
    groups = OrderedDict()
    for path in sorted(ignored_files(repo)):
        name = classify(path, include_logs)
        full = os.path.join(repo, path)
        if name is None or not os.path.isfile(full) or os.path.islink(full):
            continue
        groups.setdefault(name, []).append((path, os.path.getsize(full)))
    final = {}
    for name, files in groups.items():
        final.update(split(name, files, 2))
    return OrderedDict(sorted(final.items()))


def plan(groups):
    total = sum(s for files in groups.values() for _, s in files)
    print(f'{"archive":45s} {"files":>8s} {"size":>10s}')
    for name, files in groups.items():
        print(f'{name + ".tar.gz":45s} {len(files):8d} {human(sum(s for _, s in files)):>10s}')
    print(f'{"TOTAL (before compression)":45s} {sum(len(f) for f in groups.values()):8d} {human(total):>10s}')
    n_upload = len(groups) + 3                     # + MANIFEST.tsv, SHA256SUMS, README.md
    if total > ZENODO_MAX_BYTES or n_upload > ZENODO_MAX_FILES:
        print(f'WARNING: exceeds Zenodo limits ({human(ZENODO_MAX_BYTES)}, {ZENODO_MAX_FILES} files per record).')
    return total


def sha256(path, block=8 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(block), b''):
            h.update(chunk)
    return h.hexdigest()


def git_commit(repo):
    try:
        return subprocess.run(['git', '-C', repo, 'rev-parse', 'HEAD'], check=True,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return 'unknown'


def run(repo, out, groups, total):
    os.makedirs(out, exist_ok=True)
    if os.listdir(out):
        sys.exit(f'Output folder {out} is not empty; choose a new folder.')
    free = shutil.disk_usage(out).free
    if free < total * 1.05:
        sys.exit(f'Not enough free space in {out}: need about {human(total)}, have {human(free)}.')
    with open(os.path.join(out, 'MANIFEST.tsv'), 'w', encoding='utf-8') as man:
        man.write('archive\tpath\tbytes\n')
        for i, (name, files) in enumerate(groups.items(), 1):
            arc = name + '.tar.gz'
            print(f'[{i}/{len(groups)}] {arc}: {len(files)} files, {human(sum(s for _, s in files))}', flush=True)
            with tarfile.open(os.path.join(out, arc), 'w:gz', compresslevel=1) as tar:
                for path, size in files:
                    tar.add(os.path.join(repo, path), arcname=path, recursive=False)
                    man.write(f'{arc}\t{path}\t{size}\n')
    archives = sorted(f for f in os.listdir(out) if f.endswith('.tar.gz'))
    with open(os.path.join(out, 'SHA256SUMS'), 'w') as sums:
        for arc in archives:
            print(f'checksum {arc}', flush=True)
            sums.write(f'{sha256(os.path.join(out, arc))}  {arc}\n')
    rows = '\n'.join(f'| `{n}.tar.gz` | {len(f)} | {human(sum(s for _, s in f))} |' for n, f in groups.items())
    with open(os.path.join(out, 'README.md'), 'w', encoding='utf-8') as r:
        r.write(f"""# Walker: research data and checkpoints

Data record for the Walker project, which compares behavioural cloning (BC), DAgger and
GAIL for simulated NAO walking in the RoboCup 3D league, imitating the FC Portugal walking
controller. The code is at https://github.com/TafaraC/Walker.

This record holds the research files that are not in Git: experiment outputs, evaluation
records and traces, datasets, model checkpoints and development backups. It was packaged
on {datetime.date.today().isoformat()} from repository commit `{git_commit(repo)}`.

## Restoring the files

Extract every archive in the root of a clone of the repository; paths inside the archives
are relative to the repository root:

```bash
sha256sum -c SHA256SUMS
for f in *.tar.gz; do tar -xzf "$f" -C /path/to/Walker; done
```

`MANIFEST.tsv` lists every file with its size and the archive that contains it. Absolute
paths inside JSON records are provenance from the machine that produced them.

## Archives

| Archive | Files | Size before compression |
|---|---:|---:|
{rows}

## Licence

Data: CC BY 4.0. Code (separate repository): GNU GPL v3, as for the upstream FC Portugal codebase.
""")
    print(f'Done: {len(archives)} archives in {out}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=['plan', 'run'])
    ap.add_argument('--repo', required=True, help='repository root containing the research files')
    ap.add_argument('--out', help='new, empty output folder (run only)')
    ap.add_argument('--include-logs', action='store_true', help='also package the top-level logs/ folder')
    a = ap.parse_args()
    repo = os.path.abspath(a.repo)
    groups = collect(repo, a.include_logs)
    total = plan(groups)
    if a.stage == 'run':
        if not a.out:
            ap.error('run needs --out')
        run(repo, os.path.abspath(a.out), groups, total)


if __name__ == '__main__':
    main()
