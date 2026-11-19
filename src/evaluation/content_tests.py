"""Content tests for Delayed-Memory and Relay-Decision: what do the inferred messages carry?

Both tasks define exactly what each message should contain, so we can check for specific
failure modes beyond fidelity and specificity.

Delayed-Memory lag test
    On every pathway the true message emitted at bin t is the source's stimulus `stimulus_lag`
    bins back (2 in the released network): m_t = s_{t-2}. Stimulus history is spread across the
    network, so a model may pick up other lags. For each lag l we decode the true message shifted
    to that lag, m_{t-(l-2)} = s_{t-l}, from the inferred message at bin t:
        "lagged content fidelity" at lag l    R²(inferred m_t -> s_{t-l}), averaged over pathways
    l = 2 is the communicated content; other lags should be lower, and future lags (l < 0) are a
    null. Every lag is scored on the same bins, so they are comparable.
        "lagged content specificity"          R²(s_{t-l} for every lag at once -> inferred m_t)
    is how much of the inferred message the stimulus history explains, to compare with the
    specificity at l = 2 alone.

Relay-Decision decision test
    The relay area R passes the stimulus s to the decision area D, which integrates it into the
    decision variable d_t = s_0 + ... + s_t. A model that puts the integration in R infers a
    message that tracks d instead of s. We report, on each pathway,
        "stimulus fidelity"            R²(inferred m -> s)
        "decision variable fidelity"   R²(inferred m -> d)
    and the reverse directions ("... specificity"): how much of the inferred message s or d explains.

All scores use the linear decoder (decoders.linear_decode).
"""

from __future__ import annotations

import numpy as np

from .communication import aligned_messages
from .decoders import get_decoder
from .layouts import has_pathway_info, message_keys, message_pathways
from .reconstruction import nan_mean
from .scores import Score

STIMULUS_LAG = 2        # the Delayed-Memory message is the stimulus this many bins back
LAGS = tuple(range(-3, 7))  # stimulus lags scored by the lag test


def _r2(x: np.ndarray, y: np.ndarray) -> float:
    # R² of decoding y from x; nan if x is constant (a silent pathway has nothing to decode from).
    return float(get_decoder("linear")(x, y)[2])


def _pathway_messages(truth, preds, attrs, pred_attrs, message_delay):
    # Yield (pathway, true message, predicted message or None, delay) for every true pathway.
    if not has_pathway_info(truth, attrs):
        return
    for key in message_keys(truth):
        if key not in preds and not any(k.startswith("pathway-") for k in preds):
            continue
        pathways = message_pathways(key, truth, attrs)
        for pathway in pathways:
            yield (pathway, *aligned_messages(pathway, key, truth, preds, attrs, pred_attrs, len(pathways), message_delay))


def lag_test_scores(
    truth: dict,
    preds: dict,
    attrs: dict,
    pred_attrs: dict,
    message_delay: int | None = None,
    stimulus_lag: int = STIMULUS_LAG,
    lags: tuple[int, ...] = LAGS,
) -> list[Score]:
    """The Delayed-Memory lag test (see the module docstring)."""
    lags = tuple(sorted(set(int(lag) for lag in lags)))
    if stimulus_lag not in lags:
        lags = tuple(sorted(lags + (stimulus_lag,)))

    fidelity = {lag: [] for lag in lags}
    joint_specificity = []

    for _, true_m, pred_m, _ in _pathway_messages(truth, preds, attrs, pred_attrs, message_delay):
        n_time = true_m.shape[1]
        shifts = [lag - stimulus_lag for lag in lags]
        first, stop = max(0, max(shifts)), n_time + min(0, min(shifts))
        if stop - first < 2:
            raise ValueError(f"lags {lags} leave too few bins in {n_time}-bin trials")

        silent = pred_m is None or not np.any(pred_m)
        pred_window = np.zeros((true_m.shape[0], stop - first, 1)) if pred_m is None else pred_m[:, first:stop]

        shifted = {}
        for lag, shift in zip(lags, shifts):
            shifted[lag] = true_m[:, first - shift : stop - shift]
            fidelity[lag].append(_r2(pred_window, shifted[lag]))

        # A silent pathway has no content to explain, so its specificity is undefined.
        if not silent:
            history = np.concatenate([shifted[lag] for lag in lags], axis=-1)
            joint_specificity.append(_r2(history, pred_window))

    if not any(fidelity.values()):
        return []

    scores = [Score(f"lag {lag}", "lagged content fidelity", nan_mean(fidelity[lag])) for lag in lags]
    scores.append(Score("all lags", "lagged content specificity", nan_mean(joint_specificity)))
    return scores


def decision_test_scores(
    truth: dict,
    preds: dict,
    attrs: dict,
    pred_attrs: dict,
    message_delay: int | None = None,
) -> list[Score]:
    """The Relay-Decision decision test (see the module docstring). Needs truth-inp."""
    if "truth-inp" not in truth:
        return []

    stimulus_full = np.asarray(truth["truth-inp"], dtype=np.float64)
    decision_full = np.cumsum(stimulus_full, axis=1)

    scores = []
    for pathway, _, pred_m, delay in _pathway_messages(truth, preds, attrs, pred_attrs, message_delay):
        if pred_m is None or not np.any(pred_m):
            for metric in ("stimulus", "decision variable"):
                scores.append(Score(pathway.name, f"{metric} fidelity", float("nan")))
                scores.append(Score(pathway.name, f"{metric} specificity", float("nan")))
            continue

        # s and d are indexed by emission bin, like the true message.
        n_time = stimulus_full.shape[1]
        stimulus = stimulus_full[:, : n_time - delay]
        decision = decision_full[:, : n_time - delay]

        scores += [
            Score(pathway.name, "stimulus fidelity", _r2(pred_m, stimulus)),
            Score(pathway.name, "decision variable fidelity", _r2(pred_m, decision)),
            Score(pathway.name, "stimulus specificity", _r2(stimulus, pred_m)),
            Score(pathway.name, "decision variable specificity", _r2(decision, pred_m)),
        ]
    return scores
