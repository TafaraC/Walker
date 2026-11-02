# Zenodo data archive

Two scripts for publishing the research files that are not in Git (experiment outputs, datasets, checkpoints, traces and backups) as a Zenodo data record. Both use only the Python standard library.

## 1. Package

Run in the copy of the repository that holds the complete research files. Check first, then package into a new folder outside the repository with about 25 GB free:

```bash
python3 tools/zenodo/package_data.py plan --repo /path/to/Walker
python3 tools/zenodo/package_data.py run  --repo /path/to/Walker --out ~/walker_zenodo
```

`plan` lists the archives and sizes without writing anything. `run` writes one `.tar.gz` per experiment family (plus `artifacts`, `evaluation_raw`, backups and `models_and_datasets`), with `MANIFEST.tsv`, `SHA256SUMS` and a `README.md` for the record. Caches, virtual environments, compiled binaries, `config.json`, `.env` files and the top-level `logs/` folder are left out (add `--include-logs` to keep `logs/`).

## 2. Upload

Review `metadata.json` first (authors, description, licence, access). Create a Zenodo personal access token with only the `deposit:write` scope; it can upload but not publish.

```bash
export ZENODO_TOKEN=...      # keep private; never commit it
python3 tools/zenodo/upload_zenodo.py --dir ~/walker_zenodo
```

The script creates a draft, reserves its DOI, uploads every file and checks each against Zenodo's checksum. It never publishes. If the upload is interrupted, rerun with `--deposition <draft id>` to skip files already uploaded. Add `--sandbox` to practise on sandbox.zenodo.org (separate account and token).

Packaging and uploading tens of gigabytes take a while; run them in `tmux` or with `nohup` so a closed terminal does not stop them.

## 3. Publish

Open the draft link the script prints, review it, and press **Publish** on the website. Then add the DOI to `REPOSITORY.md`, the README and the paper.
