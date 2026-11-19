"""Check a generated multitask dataset against the network that produced it.

    python scripts/check_multitask_dataset.py --ckpt <best.ckpt> --data <out> [--split test]

Using only what the files store (gaussian activity, the truth-* inputs, message-mesgs and its
``layout`` attribute, and comm_inp_source / comm_inp_weights) plus the checkpoint's weights,
it recomputes every area's update

    h_t = tanh(W_ext x_t + sum_j W_{j->i} m^{j->i}_{t-lag} + W_hh h_{t-1} + b) + noise_t

(for ``rnn_type: grucell``, torch's GRUCell update from the same input drive, whose weights then
have 3 x hidden rows: the r, z and n gates), and every message m^{i->k}_t = W_out (h_t - noise_t) + b_out (message_source 'pre_noise') or
W_out h_t + b_out ('post_noise'). If the layout, the comm weights
and the lag are right, the residual noise is white with the trained SD (0.1) and the messages
match to float precision. A wrong pathway order or lag shows up as a much larger residual.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from generate_multitask_datasets import load_checkpoint  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True, help="output folder of generate_multitask_datasets.py")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--max-trials", type=int, default=200)
    args = p.parse_args(argv)

    _, model, _ = load_checkpoint(args.ckpt, "cpu")
    hps = model.hparams
    gpath = args.data / "gaussian" / f"{args.split}.h5"
    tpath = args.data / ("gaussian" if args.split == "test" else "ground_truth") / f"{args.split}.h5"
    with h5py.File(gpath, "r") as g, h5py.File(tpath, "r") as t, h5py.File(args.data / "gaussian" / "test.h5", "r") as tst:
        n = min(args.max_trials, g["area-A0"].shape[0])
        sigma = float(g.attrs.get("hidden_noise", 0.1))  # noise the data were generated with
        H = {a: g[f"area-{a}"][:n].astype(np.float64) for a in model.area_names}
        X = {k: t[f"truth-{k}"][:n].astype(np.float64) for k in ("fix", "stim1", "amp1", "stim2", "amp2", "rule", "task")}
        M = t["message-mesgs"][:n].astype(np.float64)
        layout = json.loads(t["message-mesgs"].attrs["layout"])
        lag = int(t["message-mesgs"].attrs["lag_steps"])
        source = str(t["message-mesgs"].attrs.get("message_source", "pre_noise"))
        comm = {a: ([s.decode() for s in tst[f"area-{a}"].attrs["comm_inp_source"]], tst[f"area-{a}"].attrs["comm_inp_weights"])
                for a in model.area_names if "comm_inp_source" in tst[f"area-{a}"].attrs}
    block = {(d["source"], d["target"]): slice(d["start"], d["end"]) for d in layout}
    T = M.shape[1]
    Mlag = np.zeros_like(M)
    Mlag[:, lag:] = M[:, :T - lag]

    ext_inputs = {a: [] for a in model.area_names}
    for i, k in enumerate(("fix", "polar1", "polar2", "task")):  # "task": scalar task slot, if any
        for a in model._stream_areas(i):  # an entry may be one area or a list (version-2 routing)
            ext_inputs[a].append(k)
    ok = True
    print(f"{args.split}: {n} trials, lag {lag} steps, messages from the {source} state, "
          f"expected residual SD {sigma:.4f}")
    k0 = 1 if getattr(hps, "readout_type", "binned") == "binned" else 0  # leading saccade-logit unit
    for a in model.area_names:
        rnn = model.areas[a].rnn.model
        gru = isinstance(rnn, torch.nn.GRUCell)
        w_ih = rnn.weight_ih.detach().double().numpy()
        w_hh = rnn.weight_hh.detach().double().numpy()
        b_ih = rnn.bias_ih.detach().double().numpy()
        b_hh = rnn.bias_hh.detach().double().numpy()
        parts = []
        for k in ext_inputs[a]:
            parts += [X[k]] if k in ("fix", "task") else [X[f"stim{k[-1]}"], X[f"amp{k[-1]}"]]
        if a in hps.rule_input_areas:
            parts.append(X["rule"])
        x = np.concatenate(parts, -1) if parts else np.zeros((n, T, 0))
        n_ext = x.shape[-1]
        # Input drive (for a GRU: the r, z and n gates' input drives, 3 x hidden rows)
        gi = x @ w_ih[:, :n_ext].T + b_ih
        srcs, W = comm.get(a, ([], np.zeros((0, w_ih.shape[0], 0))))
        for src, w in zip(srcs, W):
            gi += Mlag[..., block[(src, a)]] @ w.T
        h_prev = np.concatenate([np.zeros_like(H[a][:, :1]), H[a][:, :-1]], 1)
        gh = h_prev @ w_hh.T + b_hh
        if gru:  # torch.nn.GRUCell: gates in the order r, z, n
            Hs = hps.hidden_size
            sig = lambda v: 1.0 / (1.0 + np.exp(-v))  # noqa: E731
            r = sig(gi[..., :Hs] + gh[..., :Hs])
            z = sig(gi[..., Hs:2 * Hs] + gh[..., Hs:2 * Hs])
            cand = np.tanh(gi[..., 2 * Hs:] + r * gh[..., 2 * Hs:])
            clean = (1 - z) * cand + z * h_prev
        else:
            clean = np.tanh(gi + gh)
        noise = H[a] - clean
        # messages from the noiseless (pre-noise) or the noisy (post-noise) state
        out = model.areas[a].output.model.linear0
        state = H[a] if source == "post_noise" else clean
        m_hat = state @ out.weight.detach().double().numpy().T + out.bias.detach().double().numpy()
        cols = [d for d in layout if d["source"] == a]
        sizes = [d["end"] - d["start"] for d in cols]
        offs = np.cumsum([k0] + sizes)  # 'binned': the first output unit is the saccade readout
        m_err = max(np.abs(m_hat[..., offs[i]:offs[i + 1]] - M[..., d["start"]:d["end"]]).max() for i, d in enumerate(cols))
        lag1 = np.mean([np.corrcoef(noise[:, 1:, u].ravel(), noise[:, :-1, u].ravel())[0, 1] for u in range(0, hps.hidden_size, 8)])
        good = abs(noise.std() - sigma) < 0.05 * sigma and abs(lag1) < 0.05 and m_err < 1e-4
        ok &= good
        print(f"  {a}: from {srcs or 'none'}  residual SD {noise.std():.4f}, lag-1 autocorr {lag1:+.3f}, "
              f"max message error {m_err:.1e}  {'OK' if good else 'MISMATCH'}")
    print("all consistent" if ok else "INCONSISTENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
