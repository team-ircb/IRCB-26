"""Evaluate a preds.h5 file against a ground-truth .h5 file."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .eval_utils import (
    area_names,
    check_alignment,
    cosine_similarity,
    effectome_slots,
    l2_volume,
    linear_decode,
    load_h5,
    mcfadden_r2_poisson,
    standard_r2,
    weighted_l2_volume,
    zero_diagonal,
)

# truth-inp is the Delayed-Memory and Relay-Decision external input.
# Cognitive Task Suite external inputs are fixation and the two stimuli.
INPUT_DATASETS = (
    "truth-inp",
    "truth-fix",
    "truth-stim1",
    "truth-stim2",
)


def paired_keys(truth: dict[str, np.ndarray], preds: dict[str, np.ndarray], prefix: str) -> list[str]:
    return [key for key in truth if key.startswith(prefix) and key in preds]


# Neural activity reconstruction between ground-truth and predicted activity.
def neural_activity_recon(truth: dict[str, np.ndarray], preds: dict[str, np.ndarray]):
    # Area arrays are spike counts. rate-* are firing rates in Hz. Both are scored directly.
    rows = []
    mcfadden_vals, rate_vals = [], []

    for area in area_names(truth):
        pred = preds[f"area-{area}"]
        mcf = mcfadden_r2_poisson(truth[f"area-{area}"], pred)
        r2 = standard_r2(truth[f"rate-{area}"], pred)
        rows.append((f"area-{area}", "mcfadden r2", mcf))
        rows.append((f"area-{area}", "rate r2", r2))
        mcfadden_vals.append(mcf)
        rate_vals.append(r2)

    rows.append(("all", "mcfadden r2", float(np.nanmean(mcfadden_vals))))
    rows.append(("all", "rate r2", float(np.nanmean(rate_vals))))
    return rows


# Effectome similarity between ground-truth and predicted effectomes.
def effectome_similarity(truth: dict[str, np.ndarray], preds: dict[str, np.ndarray], attrs: dict[str, dict]):
    # Norm effectome is ||m||. Dynamic effectome is ||W m|| with comm_inp_weights.
    # The readout block in a Cognitive Task Suite message is not an inter-area edge.
    rows = []
    keys = paired_keys(truth, preds, "message-")
    areas = area_names(truth)
    for key in keys:
        true_m = np.asarray(truth[key], dtype=np.float64)
        pred_m = np.asarray(preds[key], dtype=np.float64)
        slots = effectome_slots(areas, truth, attrs, true_m.shape[-1])

        if pred_m.shape[-1] == true_m.shape[-1]:
            true_for_fx, pred_for_fx = true_m, pred_m
        else:
            true_for_fx, pred_for_fx, _ = linear_decode(pred_m, true_m)

        n = len(areas)
        true_norm, true_dyn = np.zeros((n, n)), np.zeros((n, n))
        pred_norm, pred_dyn = np.zeros((n, n)), np.zeros((n, n))
        for slot in slots:
            true_block = true_for_fx[..., slot["start"] : slot["end"]]
            pred_block = pred_for_fx[..., slot["start"] : slot["end"]]
            ti, sj = slot["target_idx"], slot["source_idx"]
            true_norm[ti, sj] = l2_volume(true_block)
            pred_norm[ti, sj] = l2_volume(pred_block)
            true_dyn[ti, sj] = weighted_l2_volume(slot["weight"], true_block)
            pred_dyn[ti, sj] = weighted_l2_volume(slot["weight"], pred_block)

        label = "communication" if len(keys) == 1 else key
        rows.append((label, "norm effectome", cosine_similarity(zero_diagonal(pred_norm), zero_diagonal(true_norm))))
        rows.append((label, "dynamic effectome", cosine_similarity(zero_diagonal(pred_dyn), zero_diagonal(true_dyn))))
    return rows


def communication_fidelity_specificity(truth: dict[str, np.ndarray], preds: dict[str, np.ndarray]):
    # Fidelity decodes the ground-truth messages from the inferred ones. Specificity is the reverse.
    rows = []
    keys = paired_keys(truth, preds, "message-")
    for key in keys:
        true_m = np.asarray(truth[key], dtype=np.float64)
        pred_m = np.asarray(preds[key], dtype=np.float64)
        _, _, fidelity = linear_decode(pred_m, true_m)
        _, _, specificity = linear_decode(true_m, pred_m)
        label = "communication" if len(keys) == 1 else key
        rows.append((label, "communication fidelity", fidelity))
        rows.append((label, "communication specificity", specificity))
    return rows


def input_fidelity_specificity(truth: dict[str, np.ndarray], preds: dict[str, np.ndarray]):
    # External inputs are optional. A preds dataset with the same name is the inferred input.
    rows = []
    keys = [key for key in INPUT_DATASETS if key in truth and key in preds]
    for key in keys:
        true_s = np.asarray(truth[key], dtype=np.float64)
        pred_s = np.asarray(preds[key], dtype=np.float64)
        _, _, fidelity = linear_decode(pred_s, true_s)
        _, _, specificity = linear_decode(true_s, pred_s)
        label = "input" if key == "truth-inp" and len(keys) == 1 else key
        rows.append((label, "input fidelity", fidelity))
        rows.append((label, "input specificity", specificity))
    return rows


# Evaluates the predicted .h5 file against the ground-truth .h5 file.
def evaluate(truth_h5: Path | str, preds_h5: Path | str) -> list[tuple[str, str, float]]:
    # Ground truth supplies activity, and on test files the messages and comm_inp_weights.
    truth, attrs = load_h5(truth_h5)
    preds, _pred_attrs = load_h5(preds_h5)
    check_alignment(truth, preds)
    rows = neural_activity_recon(truth, preds)
    rows.extend(effectome_similarity(truth, preds, attrs))
    rows.extend(communication_fidelity_specificity(truth, preds))
    rows.extend(input_fidelity_specificity(truth, preds))
    return rows
