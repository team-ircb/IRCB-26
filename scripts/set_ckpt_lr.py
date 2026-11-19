#!/usr/bin/env python
"""Copy a Lightning checkpoint with the optimizer's learning rate set to a new value.

A resume with ``--ckpt_path`` restores the optimizer state, including its learning rate, so
``model.lr_init`` is ignored on a resume. The multitask networks are trained in two stages:
from scratch at LR 1e-3 until every task is >= 0.8 in the 10-epoch means, then resumed at
LR 3e-4 until acceptance stopping fires. This script makes the checkpoint for the second stage.

    python scripts/set_ckpt_lr.py --ckpt runs/<run>/lightning_checkpoints/<epoch>-<step>.ckpt \
        --lr 3e-4 --out runs/warm_sources/<run>_lr3e-4.ckpt
    python scripts/generate_data.py multitask_dgn --run_name <new run> <the run's overrides> \
        --ckpt_path runs/warm_sources/<run>_lr3e-4.ckpt

Only the learning rate in every optimizer parameter group changes. The weights, the optimizer
moments, the epoch and step, and the callback state (including acceptance stopping's streak)
are kept as they are.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, required=True, help="source checkpoint")
    p.add_argument("--lr", type=float, required=True, help="new learning rate")
    p.add_argument("--out", type=Path, required=True, help="where to write the copy (must not exist)")
    args = p.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; not overwriting")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    old = sorted({g["lr"] for o in ck["optimizer_states"] for g in o["param_groups"]})
    for o in ck["optimizer_states"]:
        for g in o["param_groups"]:
            g["lr"] = args.lr
    if ck.get("lr_schedulers"):
        print("note: the checkpoint has lr_scheduler state, which is kept unchanged")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, args.out)
    print(f"{args.ckpt} (epoch {ck['epoch']}, LR {old}) -> {args.out} (LR {args.lr})")


if __name__ == "__main__":
    main()
