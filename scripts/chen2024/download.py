#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.chen2024.dandi_download import DANDI_VERSION, list_assets, selected_assets, download_asset

def main():
    p = argparse.ArgumentParser(description="List/download selected DANDI 000363 NWB assets")
    p.add_argument("--version", default=DANDI_VERSION, help="Dandiset version (default: the one the benchmark uses)")
    p.add_argument("--subject", action="append", help="Repeat to select subjects, e.g. --subject 449141")
    p.add_argument("--contains", help="Filter paths by substring")
    p.add_argument("--output", default="raw")
    p.add_argument("--download", action="store_true", help="Actually download; default is list-only")
    p.add_argument("--list-subjects", action="store_true",
                   help="List unique subject IDs present in matching NWB asset paths")
    p.add_argument("--max-assets", type=int, default=0, help="Limit selected assets; 0 means no limit")
    a = p.parse_args()
    selected = list(selected_assets(list_assets(version=a.version), a.subject, a.contains))
    if a.max_assets: selected = selected[:a.max_assets]
    if not selected:
        raise SystemExit("No NWB assets matched; check subject IDs/path filters/version")
    if a.list_subjects:
        subject_ids = sorted(
            {
                match.group(1)
                for x in selected
                for match in [re.search(r"sub-([A-Za-z0-9]+)", x.get("path", ""))]
                if match is not None
            }
        )
        print(f"Found {len(subject_ids)} subjects")
        for subject_id in subject_ids:
            print(subject_id)
        return
    total = sum(x.get("size", 0) for x in selected)
    print(f"Matched {len(selected)} assets; total {total/1e9:.2f} GB")
    for x in selected:
        print(f"{x['path']} ({x.get('size', 0)/1e9:.2f} GB)")
    if not a.download:
        print("Dry run only. Add --download to transfer files.")
        return
    for x in selected:
        print("Downloading", x["path"], flush=True)
        print("Saved", download_asset(x, a.output), flush=True)

if __name__ == "__main__": main()
