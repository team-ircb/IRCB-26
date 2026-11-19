"""Select one task's trials from a multitask dataset file.

Every file written by scripts/generate_multitask_datasets.py has a root attribute
``task_names`` and a group ``task_index/<task>`` holding that task's trial indices
(positions along the first axis of every per-trial dataset in that file).

    from utils.multitask_io import task_names, task_trials, load_task
    idx = task_trials("out/poisson/test.h5", "dm1")
    data = load_task("out/poisson/test.h5", "dm1")        # dict: name -> array for dm1 trials only
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def task_names(path: str | Path) -> list[str]:
    with h5py.File(path, "r") as f:
        return [n.decode() if isinstance(n, bytes) else str(n) for n in f.attrs["task_names"]]


def task_trials(path: str | Path, task: str) -> np.ndarray:
    with h5py.File(path, "r") as f:
        if task not in f["task_index"]:
            raise KeyError(f"{task!r} not in {path}; tasks: {list(f['task_index'])}")
        return f["task_index"][task][:]


def load_task(path: str | Path, task: str, keys: list[str] | None = None) -> dict[str, np.ndarray]:
    """Root-level datasets restricted to one task's trials (datasets without a trial axis are
    returned whole). Groups such as observation/ are not included."""
    idx = task_trials(path, task)
    out = {}
    with h5py.File(path, "r") as f:
        n_trials = f["area-A0"].shape[0]
        for k, d in f.items():
            if not isinstance(d, h5py.Dataset) or (keys is not None and k not in keys):
                continue
            out[k] = d[:][idx] if d.shape and d.shape[0] == n_trials else d[:]
    return out
