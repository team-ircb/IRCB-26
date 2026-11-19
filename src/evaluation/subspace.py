"""Subspace alignment: does a model read each message from the right directions of source activity?

This applies to models whose inferred message on a pathway j -> i is a linear readout of the
source area's neurons, m = C x. For each true pathway we compare two subspaces of the source
area's observed (held-in) neurons:
  - the ground-truth communication subspace, spanned by the rows of the network's own readout
    expressed in firing-rate coordinates (shipped in data/true_readouts/ for the released
    datasets, or given with --true-readouts);
  - the inferred subspace, spanned by the rows of the model's readout C, given as the attribute
    `source_readout` (channels x source neurons, every neuron of the source area in dataset order;
    columns of held-out neurons are ignored) on the prediction's pathway-<SRC>-to-<TGT> dataset.
    The readout should act on firing rates or spike counts (any uniform scale).

The inferred subspace is cut to its top d directions, where d is the true message width, so a
readout of higher rank isn't rewarded for covering more directions. The score is the share of the
true subspace that the inferred one captures, the mean squared cosine of the principal angles
between them (1 = identical, 0 = orthogonal). It is reported
  "subspace alignment"                 in the space of all observed neurons;
  "subspace alignment (95% variance)"  within the top principal components of the source area's
                                       true rates that hold 95% of their variance, where the
                                       message can actually be read from;
each with the value expected for a random subspace of the same size ("... chance").
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .io import PATHWAY_PREFIX, RATE_PREFIX, area_names
from .layouts import DATA_DIR, incoming_weights, message_keys, message_pathways
from .reconstruction import DT, nan_mean
from .scores import Score

READOUT_ATTRIBUTE = "source_readout"
VARIANCE_KEPT = 0.95
BUNDLED_READOUTS = DATA_DIR / "true_readouts"


def communication_fingerprint(truth: dict, attrs: dict) -> str:
    """A hash of a dataset's communication weights, used to match ground-truth readout files to the
    dataset they were built for."""
    digest = hashlib.sha256()
    for area in area_names(truth):
        for source, weights in sorted(incoming_weights(area, attrs).items()):
            digest.update(f"{area}<-{source}".encode())
            digest.update(np.ascontiguousarray(np.round(weights, 6), dtype=np.float64).tobytes())
    return digest.hexdigest()


def load_true_readouts(path: Path | str) -> tuple[dict, dict[str, np.ndarray]]:
    """(meta, {"<SRC>-><TGT>": (channels, source neurons) readout}) from a true-readouts .npz file."""
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        readouts = {key.split("/", 1)[1]: z[key] for key in z.files if key.startswith("readout/")}
    return meta, readouts


def find_true_readouts(truth: dict, attrs: dict) -> Path | None:
    """The bundled true-readouts file for this dataset, if there is one."""
    if not BUNDLED_READOUTS.is_dir():
        return None
    fingerprint = communication_fingerprint(truth, attrs)
    for path in sorted(BUNDLED_READOUTS.glob("*.npz")):
        meta, _ = load_true_readouts(path)
        if meta.get("fingerprint") == fingerprint:
            return path
    return None


def _orthonormal_basis(a: np.ndarray, rank: int | None = None) -> np.ndarray:
    # Orthonormal basis of the column space of a, optionally its top `rank` directions.
    u, s, _ = np.linalg.svd(a, full_matrices=False)
    if s.size:
        u = u[:, s > s[0] * 1e-10]
    return u[:, :rank] if rank is not None else u


def _captured(true_basis: np.ndarray, inferred_basis: np.ndarray) -> float:
    # Mean squared cosine of the principal angles, as a share of the true subspace's dimension.
    if inferred_basis.shape[1] == 0:
        return 0.0
    cosines = np.clip(np.linalg.svd(true_basis.T @ inferred_basis, compute_uv=False), 0, 1)
    return float((cosines**2).sum() / true_basis.shape[1])


def _chance(dim: int, d_inferred: int) -> float:
    # Expected share captured by a random d_inferred-dimensional subspace of a dim-dimensional
    # space: the expected squared cosines sum to d_true * d_inferred / dim.
    return min(d_inferred, dim) / dim


def alignment(
    true_readout: np.ndarray,
    inferred_readout: np.ndarray,
    rates: np.ndarray,
    variance_kept: float = VARIANCE_KEPT,
) -> dict[str, float]:
    """Subspace alignment of one pathway.

    true_readout and inferred_readout are (channels, neurons) over the same observed source
    neurons; rates is (trials, time, neurons) true activity of those neurons. Returns the shares
    captured in the neuron space and in the high-variance PC space, with their chance values.
    """
    w_true = np.asarray(true_readout, dtype=np.float64).T          # (neurons, d_true)
    w_inferred = np.asarray(inferred_readout, dtype=np.float64).T  # (neurons, r)
    d_true = w_true.shape[1]

    activity = np.asarray(rates, dtype=np.float64).reshape(-1, w_true.shape[0])
    eigvals, eigvecs = np.linalg.eigh(np.cov(activity - activity.mean(0), rowvar=False))
    eigvals, eigvecs = np.clip(eigvals[::-1], 0, None), eigvecs[:, ::-1]
    n_pcs = int(np.searchsorted(np.cumsum(eigvals) / eigvals.sum(), variance_kept) + 1)

    result = {}
    for label, projection in (("neurons", np.eye(w_true.shape[0])), ("pcs", eigvecs[:, :n_pcs])):
        true_basis = _orthonormal_basis(projection.T @ w_true)
        inferred_basis = _orthonormal_basis(projection.T @ w_inferred, d_true)
        result[label] = _captured(true_basis, inferred_basis)
        result[f"{label} chance"] = _chance(projection.shape[1], inferred_basis.shape[1])
    result["n_pcs"] = n_pcs
    return result


def subspace_alignment_scores(
    truth: dict,
    preds: dict,
    attrs: dict,
    pred_attrs: dict,
    masks: dict | None = None,
    true_readouts: Path | str | None = None,
    dt: float = DT,
) -> list[Score]:
    """Subspace alignment for every true pathway whose predicted pathway-* has a source readout.

    Uses the observed neurons of `masks` (all neurons without masks) and the true readouts from
    `true_readouts` or, if not given, the bundled file that matches this dataset.
    """
    has_readouts = any(
        READOUT_ATTRIBUTE in pred_attrs.get(key, {}) for key in preds if key.startswith(PATHWAY_PREFIX)
    )
    if not has_readouts or not message_keys(truth):
        return []

    path = Path(true_readouts) if true_readouts is not None else find_true_readouts(truth, attrs)
    if path is None:
        raise ValueError(
            "the prediction has source_readout attributes, but no ground-truth readouts match this "
            "dataset; pass them with --true-readouts (see docs/evaluation.md)"
        )
    _, truth_readouts = load_true_readouts(path)

    scores = []
    collected: dict[str, list[float]] = {}
    for key in message_keys(truth):
        for pathway in message_pathways(key, truth, attrs):
            pred_key = f"{PATHWAY_PREFIX}{pathway.source}-to-{pathway.target}"
            inferred = pred_attrs.get(pred_key, {}).get(READOUT_ATTRIBUTE)
            if inferred is None:
                continue
            if pathway.name not in truth_readouts:
                raise ValueError(f"{path} has no ground-truth readout for {pathway.name}")

            rate_key = f"{RATE_PREFIX}{pathway.source}"
            if rate_key not in truth:
                raise ValueError(f"subspace alignment needs {rate_key} in the ground-truth file (Poisson test files)")

            n_neurons = truth[rate_key].shape[-1]
            inferred = np.atleast_2d(np.asarray(inferred, dtype=np.float64))
            true_readout = np.asarray(truth_readouts[pathway.name], dtype=np.float64)
            for label, readout in (("predicted", inferred), ("ground-truth", true_readout)):
                if readout.shape[-1] != n_neurons:
                    raise ValueError(
                        f"{pathway.name}: the {label} readout has {readout.shape[-1]} columns, but area "
                        f"{pathway.source} has {n_neurons} neurons"
                    )

            observed = np.arange(n_neurons)
            if masks and pathway.source in masks and masks[pathway.source].get("held-in") is not None:
                observed = masks[pathway.source]["held-in"]

            result = alignment(
                true_readout[:, observed],
                inferred[:, observed],
                np.asarray(truth[rate_key], dtype=np.float64)[..., observed] * dt,
            )
            values = {
                "subspace alignment": result["neurons"],
                "subspace alignment chance": result["neurons chance"],
                f"subspace alignment ({VARIANCE_KEPT:.0%} variance)": result["pcs"],
                f"subspace alignment ({VARIANCE_KEPT:.0%} variance) chance": result["pcs chance"],
            }
            for metric, value in values.items():
                scores.append(Score(pathway.name, metric, value))
                collected.setdefault(metric, []).append(value)

    for metric, values in collected.items():
        scores.append(Score("communication", metric, nan_mean(values)))
    return scores
