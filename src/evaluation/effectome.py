"""Effectome similarity: does a model find the right pathways, with the right relative strengths?

The effectome M is an (areas x areas) matrix whose entry (i, j) is the strength of communication
from area j to area i: the L2 norm of the message over every trial, time bin and channel. The
ground-truth effectome is 0 wherever the network has no pathway, so communication a model infers
on such a pair lowers its score. The diagonal is always 0.

Four scores compare the inferred and true effectomes:
  norm effectome              cosine similarity of the two matrices
  norm effectome pearson      Pearson correlation of their off-diagonal entries
  dynamic effectome           as above, with each message first mapped into its target area's
  dynamic effectome pearson   dynamics (W m). Needs the model's own map W on every pathway.

Effectome entries are nonnegative, so the cosine has a floor: a model that puts the same strength
on every pair already scores well. The Pearson correlation removes that floor (a uniform effectome
scores 0 or nan). With two areas there are only two entries, so Pearson can only be +1, -1 or nan.
"""

from __future__ import annotations

import numpy as np

from .io import PATHWAY_PREFIX, area_names
from .layouts import message_keys, message_pathways
from .metrics import cosine_similarity, off_diagonal, pearson_correlation, signal_strength, weighted_signal_strength
from .scores import Score
from .timing import shared_pathway_delay

# Attribute on a predicted pathway-* dataset holding the model's (hidden, channels) map from that
# message into the target area's dynamics.
WEIGHTS_ATTRIBUTE = "comm_inp_weights"


def parse_pathway_key(key: str, areas: list[str]) -> tuple[str, str]:
    """("A0", "A1") from "pathway-A0-to-A1". Raises ValueError for an unknown area or a self-loop."""
    source, sep, target = key[len(PATHWAY_PREFIX):].partition("-to-")
    if not sep or source not in areas or target not in areas:
        raise ValueError(f"{key}: expected pathway-<SRC>-to-<TGT> with areas from {areas}")
    if source == target:
        raise ValueError(f"{key}: a pathway connects two different areas")
    return source, target


def predicted_pathways(preds: dict, pred_attrs: dict, areas: list[str]) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray | None]]:
    """{(source, target): (message, weights or None)} for every pathway-* dataset in a prediction."""
    out = {}
    for key in sorted(k for k in preds if k.startswith(PATHWAY_PREFIX)):
        source, target = parse_pathway_key(key, areas)
        message = np.asarray(preds[key], dtype=np.float64)
        weights = pred_attrs.get(key, {}).get(WEIGHTS_ATTRIBUTE)
        if weights is not None:
            weights = np.asarray(weights, dtype=np.float64)
            if weights.ndim != 2 or weights.shape[1] != message.shape[-1]:
                raise ValueError(
                    f"{key}: {WEIGHTS_ATTRIBUTE} should be (hidden, {message.shape[-1]}), got {weights.shape}"
                )
        out[(source, target)] = (message, weights)
    return out


def true_effectomes(truth: dict, attrs: dict, drop_last: int = 0) -> tuple[np.ndarray, np.ndarray] | None:
    """Ground-truth (norm, dynamic) effectomes over every ordered pair of areas, or None if the file
    has no messages. `drop_last` leaves out the last bins (for received timing)."""
    keys = message_keys(truth)
    if not keys:
        return None

    areas = area_names(truth)
    n = len(areas)
    norm, dynamic = np.zeros((n, n)), np.zeros((n, n))
    for key in keys:
        message = np.asarray(truth[key], dtype=np.float64)
        if drop_last:
            message = message[:, :-drop_last]
        for p in message_pathways(key, truth, attrs):
            block = message[..., p.start:p.end]
            norm[p.target_idx, p.source_idx] = signal_strength(block)
            dynamic[p.target_idx, p.source_idx] = weighted_signal_strength(p.weights, block)
    return norm, dynamic


def effectome_scores(
    truth: dict,
    preds: dict,
    attrs: dict,
    pred_attrs: dict,
    message_delay: int | None = None,
) -> list[Score]:
    """Effectome similarity from the prediction's pathway-* datasets.

    Without pathway-* datasets no effectome is reported (a message-* prediction alone doesn't say
    which pathways the model thinks exist). A model that reports no communication on any pathway
    scores nan. The dynamic scores are reported only when every predicted pathway has the
    comm_inp_weights attribute, since there are no true weights for pathways the network lacks.
    """
    areas = area_names(truth)
    pathway_keys = [k for k in preds if k.startswith(PATHWAY_PREFIX)]
    if not pathway_keys or not message_keys(truth):
        return []

    pathways = predicted_pathways(preds, pred_attrs, areas)
    delay = shared_pathway_delay(truth, attrs, pred_attrs, pathway_keys, message_delay)

    true_norm, true_dynamic = true_effectomes(truth, attrs, drop_last=delay)
    n = len(areas)
    pred_norm, pred_dynamic = np.zeros((n, n)), np.zeros((n, n))
    for (source, target), (message, weights) in pathways.items():
        if delay:
            message = message[:, delay:]
        i, j = areas.index(target), areas.index(source)
        pred_norm[i, j] = signal_strength(message)
        if weights is not None:
            pred_dynamic[i, j] = weighted_signal_strength(weights, message)

    scores = [
        Score("communication", "norm effectome", cosine_similarity(pred_norm, true_norm)),
        Score("communication", "norm effectome pearson", pearson_correlation(off_diagonal(pred_norm), off_diagonal(true_norm))),
    ]
    if all(weights is not None for _, weights in pathways.values()):
        scores += [
            Score("communication", "dynamic effectome", cosine_similarity(pred_dynamic, true_dynamic)),
            Score("communication", "dynamic effectome pearson", pearson_correlation(off_diagonal(pred_dynamic), off_diagonal(true_dynamic))),
        ]
    return scores
