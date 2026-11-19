"""Write one single-task dataset per task from a multitask dataset, in the single-task layout.

    python scripts/split_multitask_by_task.py --data <out> [--tasks dm1 dmsgo ...]

For each task this writes <out>/per_task/<task>/{gaussian,poisson,ground_truth}/{train,val,test}.h5:
the same files as the multitask ones, restricted to that task's trials, with dataset and
group attributes (comm_inp_*, the message layout, rate attributes) and the train
observation masks copied unchanged. Anything written for the single-task datasets (model
fitting, scripts/evaluate.py) can then be pointed at one task's folder as if it were a
single-task network.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from utils.multitask_io import task_names  # noqa: E402


def copy_attrs(src, dst):
    for k, v in src.attrs.items():
        dst.attrs[k] = v


def subset_file(src_path: Path, dst_path: Path, task: str, label: str):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(src_path, "r") as f, h5py.File(dst_path, "w", track_order=True) as g:
        idx = f["task_index"][task][:]
        # every trial belongs to exactly one task, so the task indices count the trials
        n_trials = sum(len(f["task_index"][t]) for t in f["task_index"])
        copy_attrs(f, g)
        g.attrs["task"] = task
        g.attrs["subset_of"] = label

        def visit(name, obj):
            if name.startswith("task_index"):
                return
            if isinstance(obj, h5py.Group):
                copy_attrs(obj, g.require_group(name))
                return
            per_trial = "/" not in name and obj.shape and obj.shape[0] == n_trials
            data = obj[:][idx] if per_trial else obj[:]
            d = g.create_dataset(name, data=data, track_times=False)
            copy_attrs(obj, d)

        f.visititems(visit)
        tg = g.create_group("task_index")
        tg.create_dataset(task, data=np.arange(len(idx), dtype=np.int64), track_times=False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True, help="output folder of generate_multitask_datasets.py")
    p.add_argument("--tasks", nargs="*", default=None, help="tasks to split out (default: all)")
    args = p.parse_args(argv)
    tasks = args.tasks or task_names(args.data / "gaussian" / "test.h5")
    for task in tasks:
        for kind in ("gaussian", "poisson", "ground_truth"):
            for split in ("train", "val", "test"):
                src = args.data / kind / f"{split}.h5"
                if src.exists():
                    subset_file(src, args.data / "per_task" / task / kind / f"{split}.h5", task, f"{kind}/{split}.h5")
        print("wrote", args.data / "per_task" / task)


if __name__ == "__main__":
    main()
