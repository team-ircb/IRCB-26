#!/usr/bin/env python3
import argparse
from pathlib import Path
import h5py
import numpy as np

def main():
    p = argparse.ArgumentParser(description="Validate processed HDF5 schema and print summary")
    p.add_argument("--input", required=True, help="HDF5 file or processed root")
    a = p.parse_args()
    root = Path(a.input)
    paths = [root] if root.is_file() else sorted(root.rglob("*.h5"))
    if not paths: raise SystemExit("No HDF5 files found")
    errors = 0
    for path in paths:
        with h5py.File(path, "r") as f:
            for session, g in f.items():
                if not isinstance(g, h5py.Group): continue
                areas = [k for k in g if k.startswith("area-")]
                n_trials = len(g["start_time"])
                indices = g["source_trial_row"][:]
                if len(indices) != n_trials or len(np.unique(indices)) != n_trials: errors += 1
                for key in areas:
                    ds = g[key]
                    if ds.ndim != 3 or ds.shape[0] != n_trials or ds.shape[2] != g.attrs["n_bins"]:
                        errors += 1
                    meta = g["unit_metadata"][key[5:]]
                    if len(meta["unit_id"]) != ds.shape[1]: errors += 1
                print(f"{path.name}/{session}: {n_trials} trials; {len(areas)} regions; " +
                      ", ".join(f"{k[5:]}={g[k].shape[1]}" for k in areas))
    if errors: raise SystemExit(f"Validation failed: {errors} errors")
    print("Validation passed")

if __name__ == "__main__": main()
