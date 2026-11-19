"""Generate datasets from a trained multitask Cognitive Task Suite network.

One command (from the repository root):

    python scripts/generate_multitask_datasets.py --ckpt <best.ckpt> --out <dir>

Everything is rebuilt from the checkpoint (model and data-module hyperparameters are stored
in it), so no config file is needed. The output follows the single-task dataset layout
(docs/dataset_reference.md) with a few additions for the multitask setting:

    <out>/gaussian/{train,val,test}.h5     hidden-state activity (single-task layout)
    <out>/poisson/{train,val,test}.h5      spike counts per 10 ms bin; test adds rate-* in Hz
    <out>/ground_truth/{train,val,test}.h5 messages, task variables, rule input and network
                                           outputs for EVERY split (not for fitting). True latents
                                           are the gaussian area-* arrays; rates follow from them
                                           with the zscore_* attributes on the poisson area-* arrays
    <out>/accuracy.csv                     per-task accuracy on each split and over all trials
                                           (exact direction bin, Yang's last-step measure, and
                                           the strict Yang-style measure)
    <out>/acceptance.json                  the network's acceptance verdict: strict Yang-style
                                           performance >= 0.80 on every task, over all of the
                                           dataset's trials (fresh trials, unseen in training and
                                           checkpoint selection)
    <out>/message_layout.json              channel layout of message-mesgs
    <out>/cts_message_layouts_entry.json   the same layout keyed as src/evaluation/data/cts_message_layouts.json
    <out>/manifest.json                    arguments, seeds, versions and SHA-256 of every file

Additions relative to the single-task files: every file has a ``task_index/<task>`` group
with that task's trial indices and root attributes ``task_names`` (plus the settings the
network was trained with); test (and ground_truth) files add ``truth-rule`` (the rule input
as fed), ``truth-task``, ``truth-go``/``truth-end``/``truth-cond`` and the noise-free task
variables; ground_truth files also hold the network's outputs.

Seeds are fixed, so rerunning with the same arguments on the same machine and device gives
identical arrays (and byte-identical files).
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import platform
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from models import CognitiveTaskSuite  # noqa: E402
from datamodules import CognitiveTaskSuite as TaskSuiteData  # noqa: E402
from utils.cognitive_tasks import task_map  # noqa: E402
from utils import yang_tasks  # noqa: E402
from utils.common_utils import categorize, normalize_torch  # noqa: E402

FORMAT_VERSION = "multitask-dgn-3"
SPLITS = ("train", "val", "test")
TRUTH_NOTES = {
    "fix": "fixation input as fed (noise included)",
    "stim1": "modality-1 input as fed, slots 1-2 (noise included). encoding attr 'raw': angle in rad; "
             "'cartesian': s*cos(theta)",
    "amp1": "modality-1 input as fed, slots 1-2 (noise included). 'raw': strength; 'cartesian': s*sin(theta)",
    "stim2": "as stim1, modality 2", "amp2": "as amp1, modality 2",
    "task": "task index (task_names order)",
    "resp": "target response direction (rad); 0 = no response",
    "sacc": "saccade/fixation target (1 = keep fixating)",
    "rule": "one-hot rule input as fed to the rule area(s), rule_noise included",
    "go": "step at which the response is cued (fixation off, stimulus onset for RT tasks, "
          "stimulus-2 onset for match tasks)",
    "cond": "trial condition: context tasks 0 agree / 1 conflict; match tasks 0 match / 1 non-match; else 0",
    "end": "first padding step after the trial (the trial is scored at end - 1)",
    "theta1": "noise-free direction of each modality-1 slot (rad; 0 where the slot is silent)",
    "strength1": "noise-free strength of each modality-1 slot (0 = silent)",
    "theta2": "as theta1, modality 2", "strength2": "as strength1, modality 2",
}
OBS_PCTS = (25, 50, 75, 100)
RATE_MAX = 40.0  # Hz, as in the single-task datasets' rate-* attributes
DT = 0.01  # s per bin, as in the single-task datasets


# ----------------------------------------------------------------------------- model
def _plain(v):
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(v):
            return OmegaConf.to_container(v, resolve=True)
    except ImportError:
        pass
    return v


def _filter_kwargs(cls, hp: dict) -> dict:
    names = inspect.signature(cls.__init__).parameters
    return {k: _plain(v) for k, v in hp.items() if k in names and k != "self"}


def load_checkpoint(path: Path, device: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    hp = dict(ck["hyper_parameters"])
    model = CognitiveTaskSuite(**_filter_kwargs(CognitiveTaskSuite, hp))
    model.load_state_dict(ck["state_dict"], strict=True)
    trained = [list(map(str, e)) for e in _plain(hp["diagram"])]
    rebuilt = [list(map(str, e)) for e in _plain(model.hparams.diagram)]
    if trained != rebuilt:
        raise RuntimeError("rebuilt graph differs from the checkpoint's graph")
    model.eval().to(device)
    model.log_dict = lambda *a, **k: None
    dm = TaskSuiteData(**_filter_kwargs(TaskSuiteData, dict(ck["datamodule_hyper_parameters"])))
    return ck, model, dm


def trained_noise(hp, epoch: int) -> float:
    """Hidden-state noise the network was trained with at this epoch (fixed-noise ramp)."""
    if hp.noise_increase > 0:
        ramp = min(1.0, max(0.0, (epoch + 1 - hp.noise_start) / (hp.noise_increase + 1)))
        return float(hp.noise * (hp.noise_init + (1 - hp.noise_init) * ramp))
    return float(hp.noise)


def message_layout(model) -> list[dict]:
    """Channel layout of message-mesgs, read from the model's own insert/slice bookkeeping.

    Each source area writes one block per downstream area, in ascending target index
    (src/models/cognitive_task_suite.py: input_indices uses sorted successors), then its readout block if it
    feeds the output stage. Output blocks are marked ``output: true``.
    """
    hps = model.hparams
    out_area = f"A{hps.num_areas}"
    yang = getattr(hps, "readout_type", "binned") == "yang"
    layout, start = [], 0
    for ia, sizes in enumerate(hps.total_mesgs):
        src = f"A{ia}"
        targets = sorted(int(n[1:]) for n in model.G.successors(src))
        if yang:  # the 'yang' readout is read at the same step, not through the message buffer
            targets = [t for t in targets if t < hps.num_areas]
        if len(targets) != len(sizes):
            raise RuntimeError(f"{src}: {len(targets)} successors but {len(sizes)} message blocks")
        for tgt, size in zip(targets, sizes):
            tname = f"A{tgt}"
            layout.append({"source": src, "target": tname, "start": start, "end": start + size,
                           "output": tname == out_area})
            start += size
    return layout


def comm_weights(model) -> dict:
    """area -> (sources, weights (n_sources, hidden, channels)) for areas receiving messages.

    The weights are the columns of the target RNN's input matrix that multiply each
    incoming message, in the order the model concatenates them (input_indices).
    """
    hps = model.hparams
    out = {}
    for area in model.area_names:
        idx = model.input_indices[area]
        if not idx:
            continue
        w_ih = model.areas[area].rnn.model.weight_ih.detach().cpu().double().numpy()
        widths = [hps.total_mesgs[src][k] for src, k in idx]
        n_ext = w_ih.shape[1] - sum(widths)
        blocks, col = [], n_ext
        for wdt in widths:
            blocks.append(w_ih[:, col:col + wdt])
            col += wdt
        if len({b.shape for b in blocks}) != 1:
            raise RuntimeError(f"{area}: incoming message widths differ; cannot stack")
        out[area] = ([f"A{src}" for src, _ in idx], np.stack(blocks))
    return out


# ----------------------------------------------------------------------------- trials
def make_trials(dm, task_objs, n_per_task: int, seed: int):
    """n_per_task trials of every task, generated task by task with fixed seeds."""
    parts, cleans = [], []
    state = np.random.get_state()
    try:
        for it in range(len(task_objs)):
            np.random.seed(seed * 1000 + it)  # task generators and gen_data use np.random
            ds = dm.gen_data(task_objs, np.full(n_per_task, it, dtype=float))
            parts.append([np.asarray(a) for a in ds.iter_data])
            cleans.append(getattr(dm, "_last_clean", None))
    finally:
        np.random.set_state(state)
    arrays = [np.concatenate([p[k] for p in parts], axis=0) for k in range(len(parts[0]))]
    clean = None
    if all(c is not None for c in cleans):
        clean = {k: np.concatenate([c[k] for c in cleans], axis=0).astype(np.float32) for k in cleans[0]}
    return arrays, clean


def stratified_split(task_of_trial: np.ndarray, n_tasks: int, seed: int):
    """80/10/10 per task (val = test = floor(10%), as in the release's 820/102/102), then
    each split's trials are shuffled so tasks are interleaved."""
    rng = np.random.default_rng(seed)
    idx = {s: [] for s in SPLITS}
    for it in range(n_tasks):
        trials = np.flatnonzero(task_of_trial == it)
        trials = rng.permutation(trials)
        n_ho = len(trials) // 10
        if n_ho == 0:
            raise ValueError("need at least 10 trials per task for an 80/10/10 split")
        idx["val"].append(trials[:n_ho])
        idx["test"].append(trials[n_ho:2 * n_ho])
        idx["train"].append(trials[2 * n_ho:])
    return {s: rng.permutation(np.concatenate(v)) for s, v in idx.items()}


# ----------------------------------------------------------------------------- simulation
@torch.no_grad()
def simulate(model, arrays, noise: float, seed: int, device: str, batch: int):
    fix = arrays[0]
    hps = model.hparams
    n_hidden = hps.num_areas * hps.hidden_size
    K = fix.shape[0]
    hidden = {a: [] for a in model.area_names}
    mesgs, logits, sacc_logit, rule_fed = [], [], [], []
    rule_area = (list(hps.rule_input_areas) or [None])[0]
    for b0 in range(0, K, batch):
        sl = slice(b0, b0 + batch)
        t = [torch.as_tensor(a[sl], dtype=torch.float32, device=device) for a in arrays]
        model.h_noise_weight = torch.full((n_hidden,), float(noise), device=device)
        model.c_noise_weight = torch.zeros_like(model.c_noise_weight, device=device)
        torch.manual_seed(seed * 100003 + b0)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(seed * 100003 + b0)
        out = model.forward([t[0], torch.cat([t[1], t[2]], 2), torch.cat([t[3], t[4]], 2), t[5]], "valid")
        for a in model.area_names:
            hidden[a].append(model.hidden_states[a].cpu().numpy())
        if rule_area is not None:  # the rule as fed (rule_noise included): last num_tasks inputs
            rule_fed.append(model.inputs[rule_area][..., -hps.num_tasks:].cpu().numpy())
        mesgs.append(model.save_var.mesgs.cpu().numpy())
        logits.append(out.cpu().numpy())
        sacc_logit.append(model.save_var.latents.cpu().numpy())
    cat = lambda xs: np.concatenate(xs, axis=0).astype(np.float32)  # noqa: E731
    return {"hidden": {a: cat(v) for a, v in hidden.items()}, "mesgs": cat(mesgs),
            "logits": cat(logits), "sacc_logit": cat(sacc_logit),
            "rule": cat(rule_fed) if rule_fed else None}


def readout(model, sim):
    """Fixation probability (K, T), decoded direction (K, T, rad) and its direction bin (K, T),
    as the model scores them (src/models/cognitive_task_suite.py: readout_signals). 'yang' readout: sigmoid fixation unit
    and the population vector over the ring; 'binned': mean saccade probability over the trained
    areas and the argmax bin of the 36 response logits."""
    hps = model.hparams
    width = 2 * math.pi / hps.num_angles
    if getattr(hps, "readout_type", "binned") == "yang":
        y = 1 / (1 + np.exp(-sim["logits"].astype(np.float64)))
        pref = np.arange(hps.n_ring) * (2 * math.pi / hps.n_ring)
        ang = np.arctan2((y[..., 1:] * np.sin(pref)).sum(-1), (y[..., 1:] * np.cos(pref)).sum(-1))
        pred = np.clip(np.floor(np.mod(ang + math.pi, 2 * math.pi) / width), 0, hps.num_angles - 1).astype(int)
        return y[..., 0], ang, pred
    cols = model._sacc_columns()
    p_fix = (1 / (1 + np.exp(-sim["sacc_logit"][..., cols].astype(np.float64)))).mean(-1)
    pred = sim["logits"].argmax(-1)
    return p_fix, -math.pi + (pred + 0.5) * width, pred


def score(model, sim, resp, sacc):
    """Per-trial correctness at the last step: decoded direction in the target's bin, and
    Yang's last-step criterion."""
    hps = model.hparams
    # Same binning as the training loss (src/models/cognitive_task_suite.py: one_hot_encode(normalize_torch(resp))).
    # one_hot_encode itself mis-indexes a single time step, so categorize directly.
    target = categorize(normalize_torch(torch.as_tensor(resp[:, -1, 0], dtype=torch.float32)),
                        [-math.pi, math.pi], hps.num_angles).numpy()
    p_fix, ang, pred = readout(model, sim)
    exact = pred[:, -1] == target
    fix_out = p_fix[:, -1] > 0.5
    should_fix = sacc[:, -1, 0] > 0.5
    dist = np.abs(np.angle(np.exp(1j * (ang[:, -1] - resp[:, -1, 0]))))
    yang = np.where(should_fix, fix_out, (~fix_out) & (dist < 0.2 * math.pi))
    return exact, yang


def score_strict(model, sim, resp, sacc, go, end):
    """Yang-style performance as the model's ``perfstrict_`` metric (src/models/cognitive_task_suite.py): scored at the
    trial's last step (end - 1), with the fixation output > 0.5 at every step from
    max(5, pre_on_steps) up to the go cue (go trials) or the end (no-go trials)."""
    hps = model.hparams
    K, T = resp.shape[:2]
    rows = np.arange(K)
    e = np.clip(end.astype(int) - 1, 0, T - 1)
    p_fix, ang, _ = readout(model, sim)  # (K, T) each
    fix_end = p_fix[rows, e] > 0.5
    should_fix = sacc[rows, e, 0] > 0.5
    dist = np.abs(np.angle(np.exp(1j * (ang[rows, e] - resp[rows, e, 0]))))
    t0 = max(5, int(getattr(hps, "pre_on_steps", 0) or 0))
    tt = np.arange(T)[None]
    until = np.where(should_fix, end, go)[:, None]
    window = (tt >= t0) & (tt < until)
    held = ((p_fix > 0.5) | ~window).all(1)
    return np.where(should_fix, held, held & ~fix_end & (dist < 0.2 * math.pi))


# ----------------------------------------------------------------------------- observation
def _round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def observation_masks(areas, n_units: int, seed: int) -> dict:
    """Masks as described in docs/dataset_reference.md: at p < 100%, observed_idx is p% of the
    units and held_out_idx is 10% of the rest; at 100%, held_out_idx is 10% of all units and is
    removed from observed_idx. Leave-one-region-out masks drop A0 or A1 at 100%."""
    rng = np.random.default_rng(seed)
    masks = {}
    for pct in OBS_PCTS:
        masks[f"pct_{pct}"] = {}
        for a in areas:
            perm = rng.permutation(n_units)
            if pct < 100:
                n_obs = _round_half_up(n_units * pct / 100)
                n_ho = _round_half_up(0.1 * (n_units - n_obs))
                obs, ho = perm[:n_obs], perm[n_obs:n_obs + n_ho]
            else:
                n_ho = _round_half_up(0.1 * n_units)
                ho, obs = perm[:n_ho], perm[n_ho:]
            masks[f"pct_{pct}"][a] = (np.sort(obs).astype(np.int64), np.sort(ho).astype(np.int64))
    for drop in ("A0", "A1"):
        m = {a: masks["pct_100"][a] for a in areas}
        m[drop] = (np.zeros(0, dtype=np.int64), masks["pct_100"][drop][1])
        masks[f"regional/drop_{drop}"] = m
    return masks


def zscore_stats(hidden: dict, chunk: int = 1024) -> dict:
    """Per area: mean and SD over all trials, steps and units of the whole dataset (all splits
    and tasks), computed once, so every split and task shares one activity-to-rate map."""
    stats = {}
    for a, h in hidden.items():
        n, s1, s2 = 0, 0.0, 0.0
        for i in range(0, h.shape[0], chunk):
            x = h[i:i + chunk].astype(np.float64)
            n, s1, s2 = n + x.size, s1 + x.sum(), s2 + np.square(x).sum()
        mu = s1 / n
        stats[a] = (float(mu), float(np.sqrt(s2 / n - mu * mu)))
    return stats


def rates_from_hidden(h: np.ndarray, mu: float, sd: float) -> np.ndarray:
    """Rate = 40 Hz (z + 1) / 2 with z the z-scored activity, clipped at 0."""
    return np.maximum(0.0, RATE_MAX * ((h.astype(np.float64) - mu) / sd + 1.0) / 2.0).astype(np.float32)


def poisson_counts(rate: np.ndarray, seed: int, split: str, area_index: int) -> np.ndarray:
    """Counts per 10 ms bin ~ Poisson(rate * dt), one fixed stream per (split, area)."""
    rng = np.random.default_rng([seed, SPLITS.index(split), area_index])
    return rng.poisson(rate.astype(np.float64) * DT).astype(np.float32)


# ----------------------------------------------------------------------------- writing
def _ds(f, name, data, **attrs):
    d = f.create_dataset(name, data=data, track_times=False)
    for k, v in attrs.items():
        d.attrs[k] = v
    return d


def _root_attrs(f, meta, split, task_names, order, task_of_trial):
    for k, v in meta.items():
        f.attrs[k] = v
    f.attrs["split"] = split
    f.attrs["task_names"] = np.array(task_names, dtype=h5py.string_dtype())
    g = f.create_group("task_index", track_order=True)
    tasks_here = task_of_trial[order]
    for it, name in enumerate(task_names):
        _ds(g, name, np.flatnonzero(tasks_here == it).astype(np.int64))


def write_split(path, split, obs, sim, stats, truth, order, task_of_trial,
                task_names, comm, layout, masks, meta, lag, seed):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w", track_order=True) as f:
        _root_attrs(f, meta, split, task_names, order, task_of_trial)
        rates = {}
        for ia, a in enumerate(sim["hidden"]):
            h = sim["hidden"][a][order]
            if obs == "gaussian":
                d = _ds(f, f"area-{a}", h, type="hidden_state")
            else:
                rates[a] = rates_from_hidden(h, *stats[a])
                d = _ds(f, f"area-{a}", poisson_counts(rates[a], seed, split, ia), type="hidden_state",
                        units="spike counts per bin", dt=DT, rate_max=RATE_MAX,
                        zscore_mean=stats[a][0], zscore_sd=stats[a][1], zscore_scope="all splits, per area")
            if split == "test" and a in comm:
                d.attrs["comm_inp_source"] = np.array(comm[a][0], dtype="S2")
                d.attrs["comm_inp_weights"] = comm[a][1]
        if split == "train":
            for key, per_area in masks.items():
                for a, (o, h) in per_area.items():
                    _ds(f, f"observation/{key}/observed_idx/{a}", o)
                    _ds(f, f"observation/{key}/held_out_idx/{a}", h)
        if split == "test":
            _write_truth(f, sim, truth, order, layout, lag, meta.get("message_source", "pre_noise"))
            if obs == "poisson":
                _write_rates(f, rates, stats)


def _write_truth(f, sim, truth, order, layout, lag, message_source="pre_noise"):
    state = "noisy (post-noise)" if message_source == "post_noise" else "noiseless (pre-noise)"
    _ds(f, "message-mesgs", sim["mesgs"][order], type="message", layout=json.dumps(layout),
        lag_steps=lag, message_source=message_source,
        note=f"Produced at step t from the source's {state} state; enters the "
        "target's update at step t + lag_steps.")
    for name, arr in truth.items():
        _ds(f, f"truth-{name}", arr[order], type="ground_truth", note=TRUTH_NOTES.get(name, ""))


def _write_rates(f, rates, stats):
    for a, r in rates.items():
        _ds(f, f"rate-{a}", r, type="firing_rate", units="Hz", dt=DT, rate_max=RATE_MAX,
            zscored=True, source=f"gaussian/area-{a}", zscore_mean=stats[a][0], zscore_sd=stats[a][1],
            zscore_scope="all splits, per area")


def write_ground_truth(path, split, sim, truth, order, task_of_trial, task_names,
                       comm, layout, meta, lag, sacc_cols):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w", track_order=True) as f:
        _root_attrs(f, meta, split, task_names, order, task_of_trial)
        for a in comm:
            g = f.require_group("comm")
            _ds(g, f"{a}/sources", np.array(comm[a][0], dtype="S2"))
            _ds(g, f"{a}/weights", comm[a][1])
        _write_truth(f, sim, truth, order, layout, lag, meta.get("message_source", "pre_noise"))
        if meta.get("readout_type", "binned") == "yang":
            y = 1 / (1 + np.exp(-sim["logits"][order].astype(np.float64)))
            n_ring = y.shape[-1] - 1
            _ds(f, "output-yang", y.astype(np.float32), type="network_output", n_ring=n_ring,
                note="Yang et al. (2019) readout, after the sigmoid: unit 0 = fixation output (> 0.5 "
                f"means still fixating), units 1..{n_ring} = response ring with preferred directions "
                f"2 pi k / {n_ring}, k = 0..{n_ring - 1}; decoded by population vector.")
            _ds(f, "output-fixation_prob", y[..., :1].astype(np.float32), type="network_output",
                note="Fixation output (sigmoid), > 0.5 means still fixating (unit 0 of output-yang).")
            return
        _ds(f, "output-resp_logits", sim["logits"][order], type="network_output",
            note="Response-output logits over 36 direction bins (bin centres -pi + (k + 0.5) 2pi/36).")
        p = 1 / (1 + np.exp(-sim["sacc_logit"][order][..., sacc_cols].astype(np.float64)))
        _ds(f, "output-fixation_prob", p.mean(-1, keepdims=True).astype(np.float32), type="network_output",
            note="Fixation output (sigmoid), > 0.5 means still fixating. Trained in A3 only.")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------- main
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, required=True, help="trained multitask checkpoint (.ckpt)")
    p.add_argument("--out", type=Path, required=True, help="output folder (created; must be empty or absent)")
    p.add_argument("--trials-per-task", type=int, default=256,
                   help="trials per task before the 80/10/10 split (default 256, as decided for the "
                        "multitask DGN; the single-task datasets have 1024 = 820/102/102)")
    p.add_argument("--accept", type=float, default=0.80,
                   help="acceptance threshold on each task's strict performance over all trials (default 0.80)")
    p.add_argument("--seed", type=int, default=0, help="trial, split, noise and Poisson seed (default 0)")
    p.add_argument("--mask-seed", type=int, default=42, help="seed for the observation masks (default 42)")
    p.add_argument("--noise", type=float, default=None, help="hidden-state noise (default: as trained)")
    p.add_argument("--device", default="cpu", help="torch device for the forward pass (default cpu)")
    p.add_argument("--batch", type=int, default=512, help="trials per forward pass (default 512)")
    p.add_argument("--threads", type=int, default=4, help="torch CPU threads (default 4)")
    p.add_argument("--no-ground-truth-files", action="store_true", help="skip <out>/ground_truth/")
    args = p.parse_args(argv)

    torch.set_num_threads(args.threads)
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} exists and is not empty")
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    ck, model, dm = load_checkpoint(args.ckpt, args.device)
    hps = model.hparams
    task_names = list(_plain(hps.task_names))
    tmap = yang_tasks.task_map if getattr(dm.hparams, "task_variant", "suite") == "yang" else task_map
    task_objs = [tmap[t][0](tmap[t][1], dm.hparams.time_total) for t in task_names]
    noise = trained_noise(hps, int(ck["epoch"])) if args.noise is None else args.noise
    lag = int(hps.delay) + 1

    arrays, clean = make_trials(dm, task_objs, args.trials_per_task, args.seed)
    fix, stim1, amp1, stim2, amp2, task, resp, sacc = arrays[:8]
    T = resp.shape[1]
    if len(arrays) >= 11:
        go_a, cond_a, end_a = arrays[8], arrays[9], arrays[10]
    else:  # older data module: go = first step with saccade target 0, no conditions, no padding
        off = sacc[..., 0] < 0.5
        go_a = np.where(off.any(1), off.argmax(1), T)[:, None, None] * np.ones_like(resp)
        cond_a, end_a = np.zeros_like(resp), np.full_like(resp, T)
    task_of_trial = task[:, 0, 0].astype(int)
    splits = stratified_split(task_of_trial, len(task_names), args.seed)
    sim = simulate(model, arrays, noise, args.seed, args.device, args.batch)
    exact, yang = score(model, sim, resp, sacc)
    strict = score_strict(model, sim, resp, sacc, go_a[:, 0, 0], end_a[:, 0, 0])

    if sim["rule"] is not None:  # as fed to the rule area(s), noise included
        rule = sim["rule"]
    else:
        rule = np.zeros(task.shape[:2] + (len(task_names),), dtype=np.float32)
        rule[np.arange(len(task_of_trial)), :, task_of_trial] = 1.0
    truth = {"fix": fix, "stim1": stim1, "amp1": amp1, "stim2": stim2, "amp2": amp2,
             "task": task, "resp": resp, "sacc": sacc, "rule": rule,
             "go": go_a, "cond": cond_a, "end": end_a}
    if clean is not None:  # noise-free direction (rad; 0 = silent slot) and strength of each slot
        truth.update(clean)

    layout = message_layout(model)
    if layout[-1]["end"] != sim["mesgs"].shape[-1]:
        raise RuntimeError("message layout does not cover message-mesgs")
    comm = comm_weights(model)
    signature = "|".join(f"{a}=" + ",".join(comm[a][0]) if a in comm else f"{a}=" for a in model.area_names)
    stats = zscore_stats(sim["hidden"])
    masks = observation_masks(model.area_names, hps.hidden_size, args.mask_seed)

    meta = {"format_version": FORMAT_VERSION, "source_checkpoint": args.ckpt.name,
            "checkpoint_sha256": sha256(args.ckpt), "checkpoint_epoch": int(ck["epoch"]),
            "trials_per_task": args.trials_per_task, "seed": args.seed, "mask_seed": args.mask_seed,
            "hidden_noise": noise, "message_lag_steps": lag, "dt": DT,
            "task_variant": str(getattr(dm.hparams, "task_variant", "suite")),
            "stimulus_encoding": str(getattr(dm.hparams, "encoding", "raw")),
            "input_noise": float(getattr(dm.hparams, "input_noise", 0.0) or 0.0),
            "fix_noise": float(dm.hparams.input_noise if getattr(dm.hparams, "fix_noise", None) is None
                               else dm.hparams.fix_noise),
            "rule_noise": float(getattr(hps, "rule_noise", 0.0) or 0.0),
            "readout_type": str(getattr(hps, "readout_type", "binned")),
            "readout_areas": ",".join(map(str, hps.readout_areas)),
            "message_source": str(getattr(hps, "message_source", "pre_noise")),
            "yang_task_names": ",".join(task_names) if getattr(dm.hparams, "task_variant", "suite") == "yang" else "",
            "suite_task_names": ",".join(yang_tasks.YANG_TO_SUITE.get(t, t) for t in task_names)
                               if getattr(dm.hparams, "task_variant", "suite") == "yang" else ",".join(task_names),
            "rule_input_areas": ",".join(map(str, hps.rule_input_areas)),
            "task_step_ms": 20.0 if getattr(dm.hparams, "task_variant", "suite") == "yang" else 10.0}
    # GRU checkpoints only (vanilla RNN output is unchanged): name the cell and the row order of
    # comm_inp_weights, which for a GRU stacks the three gate blocks. The last attribute is kept as in
    # the released multitask dataset, so regenerated files match it byte for byte.
    if any(isinstance(model.areas[a].rnn.model, torch.nn.GRUCell) for a in model.area_names):
        meta["rnn_type"] = "grucell"
        meta["comm_inp_weights_rows"] = ("3 blocks of hidden_size rows in torch.nn.GRUCell order: r (reset), "
                                         "z (update), n (candidate)")
        meta["comm_strength_ground_truth"] = "silencing (silencing/<run>.json next to this dataset)"

    for obs in ("gaussian", "poisson"):
        for s in SPLITS:
            write_split(args.out / obs / f"{s}.h5", s, obs, sim, stats, truth, splits[s],
                        task_of_trial, task_names, comm, layout, masks, meta, lag, args.seed)
    if not args.no_ground_truth_files:
        for s in SPLITS:
            write_ground_truth(args.out / "ground_truth" / f"{s}.h5", s, sim, truth, splits[s],
                               task_of_trial, task_names, comm, layout, meta, lag, model._sacc_columns())

    rows = ["task,split,n_trials,acc_exact_bin,perf_yang,perf_strict,go_trials"]
    for s in SPLITS:
        o = splits[s]
        for it, name in enumerate(task_names):
            sel = o[task_of_trial[o] == it]
            go = int((sacc[sel, -1, 0] < 0.5).sum())
            rows.append(f"{name},{s},{len(sel)},{exact[sel].mean():.4f},{yang[sel].mean():.4f},"
                        f"{strict[sel].mean():.4f},{go}")
    for it, name in enumerate(task_names):
        sel = np.flatnonzero(task_of_trial == it)
        go = int((sacc[sel, -1, 0] < 0.5).sum())
        rows.append(f"{name},all,{len(sel)},{exact[sel].mean():.4f},{yang[sel].mean():.4f},"
                    f"{strict[sel].mean():.4f},{go}")
    (args.out / "accuracy.csv").write_text("\n".join(rows) + "\n")

    # Acceptance: strict Yang-style performance >= threshold
    # on every task, over all trials_per_task trials. These trials are generated here, after
    # training, with this script's seed, so neither training nor checkpoint selection saw them.
    per_task = {name: float(strict[task_of_trial == it].mean()) for it, name in enumerate(task_names)}
    failing = {k: v for k, v in per_task.items() if v < args.accept}
    accepted = not failing
    (args.out / "acceptance.json").write_text(json.dumps({
        "accepted": accepted, "threshold": args.accept, "metric": "perf_strict (Yang-style, perfstrict_ in src/models/cognitive_task_suite.py)",
        "trials_per_task": args.trials_per_task, "seed": args.seed, "per_task": per_task,
        "failing": failing}, indent=1) + "\n")
    (args.out / "message_layout.json").write_text(json.dumps(layout, indent=1) + "\n")
    (args.out / "cts_message_layouts_entry.json").write_text(json.dumps({signature: layout}, indent=1) + "\n")

    files = sorted(q for q in args.out.rglob("*") if q.is_file() and q.name != "manifest.json")
    manifest = {"format_version": FORMAT_VERSION, "command": " ".join(sys.argv), "args": {
        k: str(v) for k, v in vars(args).items()}, "meta": meta, "graph_signature": signature,
        "task_names": task_names, "trials": {s: int(len(v)) for s, v in splits.items()},
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                     "h5py": h5py.__version__, "platform": platform.platform()},
        "seconds": round(time.time() - t0, 1),
        "sha256": {str(q.relative_to(args.out)): sha256(q) for q in files}}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")

    print(f"wrote {args.out} in {time.time() - t0:.0f} s: " + ", ".join(f"{s} {len(v)}" for s, v in splits.items()))
    print(f"acceptance (strict perf >= {args.accept} on every task, {args.trials_per_task} trials/task): "
          + ("ACCEPTED" if accepted else "NOT accepted; failing: "
             + ", ".join(f"{k} {v:.2f}" for k, v in sorted(failing.items(), key=lambda kv: kv[1]))))
    print(f"{'task':14s} {'train exact':>11s} {'test exact':>10s} {'train Yang':>10s} {'test Yang':>9s} {'train strict':>12s} {'test strict':>11s}")
    for it, name in enumerate(task_names):
        tr = splits["train"][task_of_trial[splits["train"]] == it]
        te = splits["test"][task_of_trial[splits["test"]] == it]
        print(f"{name:14s} {exact[tr].mean():11.2f} {exact[te].mean():10.2f} {yang[tr].mean():10.2f} {yang[te].mean():9.2f} "
              f"{strict[tr].mean():12.2f} {strict[te].mean():11.2f}")


if __name__ == "__main__":
    main()
