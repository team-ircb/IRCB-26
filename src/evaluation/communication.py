"""Single-trial communication recovery (fidelity) and specificity, per pathway.

For each ground-truth pathway j -> i:
  communication fidelity     decode the true message from the model's message on that pathway
  communication specificity  decode the model's message from the true message

Low fidelity means the model misses what was communicated; low specificity means its message
carries things the true message doesn't. Each is R² on held-out trials, per channel, averaged over
channels (see decoders.py), then averaged over the true pathways. Each pathway is also reported on
its own row, named "<SRC>-><TGT>".

The model's message on pathway j -> i is, in order of preference:
  1. pathway-<j>-to-<i> in the prediction;
  2. that pathway's channels of a predicted message-* with the same width as the truth;
  3. the whole predicted message-*, when the truth has a single pathway.
A prediction that has pathway-* datasets but none for a true pathway sends nothing on it: its
fidelity is about 0 and its specificity undefined (nan, left out of the average).

External inputs (truth-inp, truth-fix, truth-stim1, truth-stim2) are scored the same way, each on
its own, when the prediction has a dataset of the same name.
"""

from __future__ import annotations

import numpy as np

from .decoders import get_decoder, normalize_decoder_name
from .io import PATHWAY_PREFIX
from .layouts import Pathway, has_pathway_info, message_keys, message_pathways
from .reconstruction import nan_mean
from .scores import Score
from .timing import pair_received, prediction_timing, resolve_delay

# External inputs a model may infer. truth-inp is the Delayed-Memory and Relay-Decision input; the
# Cognitive Task Suite inputs are fixation and the two stimuli.
INPUT_DATASETS = ("truth-inp", "truth-fix", "truth-stim1", "truth-stim2")


def metric_name(direction: str, decoder: str) -> str:
    """"communication fidelity" for the linear decoder, "communication fidelity (mlp)" otherwise."""
    return f"communication {direction}" + ("" if decoder == "linear" else f" ({decoder})")


def predicted_message(
    pathway: Pathway, key: str, truth: dict, preds: dict
) -> tuple[np.ndarray | None, str | None]:
    """The model's message on a true pathway of ground-truth message `key`, and the prediction
    dataset it came from. (None, None) if the prediction says nothing about this pathway."""
    pathway_key = f"{PATHWAY_PREFIX}{pathway.source}-to-{pathway.target}"
    if pathway_key in preds:
        return np.asarray(preds[pathway_key], dtype=np.float64), pathway_key

    if key in preds:
        full = np.asarray(preds[key], dtype=np.float64)
        if full.shape[-1] == truth[key].shape[-1]:
            return full[..., pathway.start:pathway.end], key
    return None, None


def aligned_messages(
    pathway: Pathway,
    key: str,
    truth: dict,
    preds: dict,
    attrs: dict,
    pred_attrs: dict,
    n_pathways: int,
    message_delay: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    """(true message, predicted message, delay) on one pathway, paired in time.

    The predicted message is None if the prediction gives none for this pathway (silent). `delay`
    is the number of bins the pair was shifted by (0 unless the prediction has received timing):
    the true message keeps bins [0, T - delay) and the prediction bins [delay, T).
    """
    true_message = np.asarray(truth[key], dtype=np.float64)[..., pathway.start:pathway.end]
    pred, source_key = predicted_message(pathway, key, truth, preds)

    # A message-* narrower or wider than the truth can only be matched when there is one pathway.
    if pred is None and key in preds and n_pathways == 1:
        pred, source_key = np.asarray(preds[key], dtype=np.float64), key

    if pred is None:
        return true_message, None, 0

    delay = 0
    if prediction_timing(pred_attrs, source_key) == "received":
        delay = resolve_delay(key, truth, attrs, message_delay)
        true_message, pred = pair_received(true_message, pred, delay)
    return true_message, pred, delay


def communication_scores(
    truth: dict,
    preds: dict,
    attrs: dict | None = None,
    pred_attrs: dict | None = None,
    message_delay: int | None = None,
    decoders: tuple[str, ...] = ("linear",),
) -> list[Score]:
    """Communication fidelity and specificity for every true pathway, with each decoder in
    `decoders` ("linear", "kernel ridge", "mlp")."""
    attrs, pred_attrs = attrs or {}, pred_attrs or {}
    decoders = tuple(normalize_decoder_name(name) for name in decoders)
    has_pathways = any(k.startswith(PATHWAY_PREFIX) for k in preds)

    scores = []
    collected = {(d, direction): [] for d in decoders for direction in ("fidelity", "specificity")}

    for key in message_keys(truth):
        if not has_pathways and key not in preds:
            continue

        # A truth file that doesn't say which channels belong to which pathway is scored as one
        # block, the whole message (none of the released datasets is like this).
        if not has_pathway_info(truth, attrs):
            if key in preds:
                scores += _score_whole_message(key, truth, preds, attrs, pred_attrs, message_delay, decoders)
            continue

        pathways = message_pathways(key, truth, attrs)
        for pathway in pathways:
            true_message, pred, _ = aligned_messages(
                pathway, key, truth, preds, attrs, pred_attrs, len(pathways), message_delay
            )
            if pred is None:
                if not has_pathways:
                    raise ValueError(
                        f"{key}: the predicted message has {preds[key].shape[-1]} channels, not the "
                        f"ground truth's {truth[key].shape[-1]}, so its pathways can't be told apart. "
                        "Give one pathway-<SRC>-to-<TGT> dataset per pathway instead."
                    )
                # Silent pathway: a constant prediction.
                pred = np.zeros(true_message.shape[:2] + (1,))

            for decoder in decoders:
                fidelity = get_decoder(decoder, "fidelity")(pred, true_message)[2]
                specificity = get_decoder(decoder, "specificity")(true_message, pred)[2]
                scores.append(Score(pathway.name, metric_name("fidelity", decoder), fidelity))
                scores.append(Score(pathway.name, metric_name("specificity", decoder), specificity))
                collected[(decoder, "fidelity")].append(fidelity)
                collected[(decoder, "specificity")].append(specificity)

    for decoder in decoders:
        if collected[(decoder, "fidelity")]:
            for direction in ("fidelity", "specificity"):
                value = nan_mean(collected[(decoder, direction)])
                scores.append(Score("communication", metric_name(direction, decoder), value))
    return scores


def _score_whole_message(key, truth, preds, attrs, pred_attrs, message_delay, decoders) -> list[Score]:
    true_message = np.asarray(truth[key], dtype=np.float64)
    pred = np.asarray(preds[key], dtype=np.float64)
    if prediction_timing(pred_attrs, key) == "received":
        true_message, pred = pair_received(true_message, pred, resolve_delay(key, truth, attrs, message_delay))

    scores = []
    for decoder in decoders:
        scores.append(Score("communication", metric_name("fidelity", decoder), get_decoder(decoder, "fidelity")(pred, true_message)[2]))
        scores.append(Score("communication", metric_name("specificity", decoder), get_decoder(decoder, "specificity")(true_message, pred)[2]))
    return scores


def input_scores(truth: dict, preds: dict) -> list[Score]:
    """Input fidelity and specificity for each external input the prediction infers (linear decoder)."""
    keys = [key for key in INPUT_DATASETS if key in truth and key in preds]
    scores = []
    for key in keys:
        true_input = np.asarray(truth[key], dtype=np.float64)
        pred_input = np.asarray(preds[key], dtype=np.float64)
        label = "input" if key == "truth-inp" and len(keys) == 1 else key
        scores.append(Score(label, "input fidelity", get_decoder("linear")(pred_input, true_input)[2]))
        scores.append(Score(label, "input specificity", get_decoder("linear")(true_input, pred_input)[2]))
    return scores
