#!/usr/bin/env python3
"""Upload the packaged Walker data to a Zenodo draft. It never publishes.

Creates a draft record with the metadata in metadata.json, reserves its DOI, and
uploads every file in the package folder (archives, MANIFEST.tsv, SHA256SUMS,
README.md). Each upload is checked against the MD5 checksum Zenodo reports.
A dropped connection is retried automatically into the same draft. Files already
in the draft with the same size are skipped, so a stopped run can be resumed with
--deposition <id>; running without it creates a new draft. Publish the draft
yourself on the Zenodo website after reviewing it.

The token comes from the ZENODO_TOKEN environment variable; create it on Zenodo
under Applications > Personal access tokens with only the deposit:write scope.

Usage (Python 3.8+, standard library only):
  export ZENODO_TOKEN=...            # never commit or share the token
  python3 tools/zenodo/upload_zenodo.py --dir <package folder> [--sandbox] [--deposition <id>] [--tries N]
"""
import argparse, hashlib, http.client, json, os, sys, time, urllib.error, urllib.parse, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


class UploadError(Exception):
    def __init__(self, message, retry=True):
        super().__init__(message)
        self.retry = retry


def api(base, method, path, token, payload=None):
    req = urllib.request.Request(base + path, method=method,
                                 data=None if payload is None else json.dumps(payload).encode(),
                                 headers={'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req) as r:
            body = r.read()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        sys.exit(f'Zenodo API error {e.code} on {method} {path}: {e.read().decode(errors="replace")[:800]}')


def human(n):
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n < 1000 or unit == 'TB':
            return f'{n:.1f} {unit}'
        n /= 1000


def server_reply(conn):
    """After a dropped upload, return whatever Zenodo answered before closing, if anything."""
    try:
        if conn.sock is not None:
            conn.sock.settimeout(10)
        resp = conn.getresponse()
        return f'; Zenodo answered {resp.status} {resp.reason}: {resp.read(800).decode(errors="replace")}'
    except Exception:
        return '; no answer from Zenodo'


def put_once(bucket_url, token, path):
    """Stream one file to the bucket; return Zenodo's file record."""
    size = path.stat().st_size
    url = urllib.parse.urlsplit(bucket_url + '/' + urllib.parse.quote(path.name))
    conn = (http.client.HTTPSConnection if url.scheme == 'https' else http.client.HTTPConnection)(url.netloc, timeout=600)
    md5, sent, t0, last = hashlib.md5(), 0, time.time(), 0.0
    try:
        try:
            conn.putrequest('PUT', url.path)
            conn.putheader('Authorization', f'Bearer {token}')
            conn.putheader('Content-Type', 'application/octet-stream')
            conn.putheader('Content-Length', str(size))
            conn.endheaders()
        except OSError as e:
            raise UploadError(f'could not connect ({e})')
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                try:
                    conn.send(chunk)
                except OSError as e:
                    raise UploadError(f'connection dropped after {human(sent)} and {time.time() - t0:.0f} s '
                                      f'({e}){server_reply(conn)}')
                md5.update(chunk)
                sent += len(chunk)
                now = time.time()
                if now - last > 2 or sent == size:
                    rate = sent / max(now - t0, 1e-6)
                    print(f'\r  {path.name}: {100 * sent / max(size, 1):5.1f}% of {human(size)} '
                          f'({human(rate)}/s)', end='', flush=True)
                    last = now
        print()
        try:
            resp = conn.getresponse()
            body = resp.read().decode(errors='replace')
        except (OSError, http.client.HTTPException) as e:
            raise UploadError(f'all {human(size)} sent, but no answer from Zenodo ({e})')
        if resp.status not in (200, 201):
            raise UploadError(f'Zenodo answered {resp.status}: {body[:800]}', retry=resp.status >= 500 or resp.status == 429)
        rec = json.loads(body)
        remote = rec.get('checksum', '')
        if remote and remote != 'md5:' + md5.hexdigest():
            raise UploadError(f'checksum mismatch: local md5:{md5.hexdigest()}, Zenodo {remote}')
        return rec
    finally:
        conn.close()


def put_file(bucket_url, token, path, tries):
    for attempt in range(1, tries + 1):
        try:
            return put_once(bucket_url, token, path)
        except UploadError as e:
            print(f'\n  {path.name}: attempt {attempt} of {tries} failed: {e}')
            if not e.retry or attempt == tries:
                raise
            wait = min(300, 15 * 2 ** (attempt - 1))
            print(f'  retrying in {wait} s')
            time.sleep(wait)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', required=True, help='folder written by package_data.py run')
    ap.add_argument('--metadata', default=str(HERE / 'metadata.json'))
    ap.add_argument('--sandbox', action='store_true', help='use sandbox.zenodo.org (practice; separate account and token)')
    ap.add_argument('--deposition', help='existing draft id, to resume an interrupted upload')
    ap.add_argument('--tries', type=int, default=8, help='attempts per file before stopping (default 8)')
    a = ap.parse_args()
    token = os.environ.get('ZENODO_TOKEN') or sys.exit('Set ZENODO_TOKEN first (see the usage notes).')
    base = os.environ.get('ZENODO_BASE') or ('https://sandbox.zenodo.org' if a.sandbox else 'https://zenodo.org')
    folder = Path(a.dir)
    files = sorted(p for p in folder.iterdir() if p.is_file() and not p.name.startswith('.'))
    if not any(p.name == 'SHA256SUMS' for p in files):
        sys.exit(f'{folder} does not look like a package folder (no SHA256SUMS).')
    meta = json.loads(Path(a.metadata).read_text(encoding='utf-8'))
    meta['metadata']['prereserve_doi'] = True

    if a.deposition:
        dep = api(base, 'GET', f'/api/deposit/depositions/{a.deposition}', token)
    else:
        dep = api(base, 'POST', '/api/deposit/depositions', token, {})
        dep = api(base, 'PUT', f'/api/deposit/depositions/{dep["id"]}', token, meta)
    doi = (dep.get('metadata', {}).get('prereserve_doi') or {}).get('doi', '(not reserved)')
    resume = f'python {Path(sys.argv[0])} --dir "{folder}" --deposition {dep["id"]}'
    print(f'Draft {dep["id"]}  reserved DOI {doi}')
    print(f'If this run stops, resume into the same draft with:\n  {resume}')

    existing = {f['filename']: f.get('filesize') for f in api(base, 'GET', f'/api/deposit/depositions/{dep["id"]}/files', token) or []}
    bucket = dep['links']['bucket']
    todo = [p for p in files if existing.get(p.name) != p.stat().st_size]
    for p in files:
        if p not in todo:
            print(f'  {p.name}: already uploaded, skipped')
    print(f'Uploading {len(todo)} of {len(files)} files, {human(sum(p.stat().st_size for p in todo))} to send')
    try:
        for p in todo:
            put_file(bucket, token, p, a.tries)
    except (UploadError, KeyboardInterrupt) as e:
        sys.exit(f'\nUpload stopped{": " + str(e) if str(e) else ""}\n'
                 f'Files that finished stay in draft {dep["id"]}. Resume with:\n  {resume}')

    print(f'\nAll files uploaded. Draft (not published): {base}/uploads/{dep["id"]}')
    print(f'Reserved DOI: {doi}')
    print('Review the draft on the Zenodo website, then press Publish there.')


if __name__ == '__main__':
    main()
