"""Read DANDI 000363 NWB files over HTTP instead of downloading them.

Each Chen et al. (2024) session file is 250-400 GB, mostly raw ephys and video. The conversion
only needs the units table (spike times, region labels, quality) and the trials table, which
are a few hundred MB, so streaming them is far cheaper than a download. This follows DANDI's
recommended streaming route (``remfile`` + ``h5py`` + ``pynwb``).

Requires the optional dependency ``remfile`` (``pip install remfile``).
"""
from contextlib import contextmanager

import h5py
import requests
from pynwb import NWBHDF5IO

from .dandi_download import API


def asset_url(asset):
    """Resolve a DANDI asset (dict from ``list_assets`` or an asset ID) to its S3 URL."""
    asset_id = asset["asset_id"] if isinstance(asset, dict) else str(asset)
    # The API answers with a redirect to a presigned S3 URL. Read it from the Location header:
    # following it with HEAD fails (403), because the URL is signed for GET only.
    r = requests.head(f"{API}/assets/{asset_id}/download/", allow_redirects=False, timeout=60)
    if r.status_code in (301, 302, 303, 307, 308) and "Location" in r.headers:
        return r.headers["Location"]
    r.raise_for_status()
    return r.url


@contextmanager
def open_remote_nwb(url):
    """Yield the NWBFile at ``url``, streamed lazily (only the arrays that are read are fetched)."""
    try:
        import remfile
    except ImportError as e:  # pragma: no cover
        raise ImportError("Streaming needs remfile: pip install remfile") from e
    rf = remfile.File(url)
    h5 = h5py.File(rf, "r")
    io = NWBHDF5IO(file=h5, mode="r", load_namespaces=True)
    try:
        yield io.read()
    finally:
        io.close()
        h5.close()
        rf.close()
