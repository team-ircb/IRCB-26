#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.chen2024.nwb_reader import nwb_paths, describe

def main():
    p = argparse.ArgumentParser(description="Inspect local NWB trial/unit schema")
    p.add_argument("--input", required=True, help="NWB file or directory")
    p.add_argument("--max-files", type=int, default=3)
    a = p.parse_args()
    paths = nwb_paths(a.input)
    if not paths: raise SystemExit("No .nwb files found")
    for path in paths[:a.max_files]:
        print(json.dumps(describe(path), indent=2, default=str))

if __name__ == "__main__": main()
