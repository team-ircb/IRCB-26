"""Minimal DANDI REST client: list assets and download selected NWB files.

No Code Ocean paths or credentials. Beware: NWB assets can be very large.
"""
from pathlib import Path
import sys
import time
import requests

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

API = "https://api.dandiarchive.org/api"


def _iter_with_progress(response, desc, total_bytes=None):
    if tqdm is not None and sys.stderr.isatty():
        total = total_bytes if total_bytes is not None else None
        yield from tqdm(response.iter_content(chunk_size=8 * 1024 * 1024), total=total,
                       unit="B", unit_scale=True, unit_divisor=1024,
                       desc=desc, leave=True, file=sys.stderr, dynamic_ncols=True)
        return

    downloaded = 0
    last_report = 0.0
    last_percent = -1.0
    total = total_bytes if total_bytes is not None else 0
    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
        if not chunk:
            continue
        downloaded += len(chunk)
        now = time.monotonic()
        if total > 0:
            percent = downloaded / total * 100.0
            if percent >= last_percent + 1.0 or downloaded >= total:
                print(f"{desc}: {percent:.1f}% ({downloaded / 1e9:.2f} / {total / 1e9:.2f} GB)", file=sys.stderr, flush=True)
                last_percent = percent
        elif now - last_report >= 1.0:
            print(f"{desc}: {downloaded / 1e9:.2f} GB downloaded", file=sys.stderr, flush=True)
            last_report = now
        yield chunk

    if total > 0:
        print(f"{desc}: 100.0% ({downloaded / 1e9:.2f} / {total / 1e9:.2f} GB)", file=sys.stderr, flush=True)
    else:
        print(f"{desc}: {downloaded / 1e9:.2f} GB downloaded", file=sys.stderr, flush=True)

# Dandiset version the benchmark's real-data datasets are built from. Its unit curation is the one
# our models were fit on; later versions (from 0.230822.0128) relabel units, which changes the
# neurons in each area. Its session files are small (about 220 MB: spikes, trials and behavior,
# without raw voltage or video), so they are quick to download whole.
DANDI_VERSION = "0.230808.2053"


def list_assets(dandiset="000363", version=DANDI_VERSION):
    url = f"{API}/dandisets/{dandiset}/versions/{version}/assets/"
    session = requests.Session()
    while url:
        response = session.get(url, params={"page_size": 100} if "?" not in url else None, timeout=60)
        response.raise_for_status()
        page = response.json()
        for asset in page["results"]:
            yield asset
        url = page.get("next")

def selected_assets(assets, subjects=None, contains=None):
    subjects = [str(s).removeprefix("sub-") for s in (subjects or [])]
    for a in assets:
        path = a.get("path", "")
        if not path.endswith(".nwb"): continue
        if subjects and not any(f"sub-{s}" in path for s in subjects): continue
        if contains and contains not in path: continue
        yield a

def download_asset(asset, output_dir, overwrite=False):
    path = Path(output_dir) / asset["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; delete partial files before retrying")
    temp = path.with_name(path.name + ".part")
    if temp.exists():
        raise FileExistsError(f"Partial download exists: {temp}; delete it before retrying")
    url = f"{API}/assets/{asset['asset_id']}/download/"
    try:
        with requests.get(url, stream=True, timeout=(30, 120), allow_redirects=True) as r:
            r.raise_for_status()
            total = r.headers.get("Content-Length")
            total_bytes = int(total) if total is not None and total.isdigit() else int(asset.get("size", 0) or 0)
            if total_bytes <= 0:
                total_bytes = None
            print(f"Downloading {path.name}", flush=True)
            with temp.open("wb") as f:
                for chunk in _iter_with_progress(r, str(path.name), total_bytes):
                    if chunk:
                        f.write(chunk)
        expected = asset.get("size")
        if expected is not None and temp.stat().st_size != int(expected):
            raise IOError(f"Size mismatch for {path}: got {temp.stat().st_size}, expected {expected}")
        temp.replace(path)
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    return path
