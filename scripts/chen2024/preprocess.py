#!/usr/bin/env python3
"""Convert Chen et al. (2024) NWB files (DANDI 000363) to subject/session/area HDF5.

Local files:
    python scripts/chen2024/preprocess.py --config configs/real_data/chen2024/example_440959.yaml --input raw --output processed_440959

Streamed from DANDI, without downloading the 250-400 GB session files (needs remfile):
    python scripts/chen2024/preprocess.py --config configs/real_data/chen2024/example_440959.yaml --dandi-subject 440959 --dandi-session 20190225T142613 --output processed_440959
"""
import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.chen2024.config import load_config
from src.chen2024.nwb_reader import nwb_paths
from src.chen2024.dandi_download import DANDI_VERSION
from src.chen2024.preprocess import convert_file, convert_dandi_asset


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/real_data/chen2024/default.yaml")
    p.add_argument("--input", help="NWB file or directory; overrides config input_dir")
    p.add_argument("--output", help="Output directory; overrides config output_dir")
    p.add_argument("--max-files", type=int, default=0)
    p.add_argument("--dandi-subject", action="append",
                   help="Stream this subject's sessions from DANDI instead of reading local files (repeatable)")
    p.add_argument("--dandi-session", action="append",
                   help="With --dandi-subject: only sessions whose path contains this text, e.g. 20190225T142613 (repeatable)")
    p.add_argument("--dandi-version", default=DANDI_VERSION, help="Dandiset version (default: the one the benchmark uses)")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(a.config)
    output = a.output or cfg.get("output_dir", "processed")

    if a.dandi_subject:
        from src.chen2024.dandi_download import list_assets, selected_assets
        assets = list(selected_assets(list_assets(version=a.dandi_version), a.dandi_subject))
        if a.dandi_session:
            assets = [x for x in assets if any(s in x["path"] for s in a.dandi_session)]
        assets.sort(key=lambda x: x["path"])
        if a.max_files: assets = assets[:a.max_files]
        if not assets: raise SystemExit("No DANDI assets matched")
        for asset in assets:
            logging.info("Streaming %s", asset["path"])
            convert_dandi_asset(asset, output, cfg, version=a.dandi_version)
        return

    paths = nwb_paths(a.input or cfg.get("input_dir", "raw"))
    if a.max_files: paths = paths[:a.max_files]
    if not paths: raise SystemExit("No NWB files found")
    for path in paths:
        convert_file(path, output, cfg)


if __name__ == "__main__": main()
