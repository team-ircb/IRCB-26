"""Neural activity reconstruction: how well the predicted activity matches the data."""

from __future__ import annotations

import numpy as np

from .io import AREA_PREFIX, RATE_PREFIX, area_names, is_count_data
from .metrics import mcfadden_r2, per_neuron_r2
from .scores import Score

# Bin width in seconds of the released datasets. rate-* is in Hz; counts per bin are rate * DT.
DT = 0.01


def neuron_groups(masks: dict | None) -> list[tuple[str, str | None]]:
    """(metric suffix, mask label) pairs: all neurons, then held-in and held-out when masks are given."""
    groups = [("", None)]
    if masks:
        groups += [(" held-in", "held-in"), (" held-out", "held-out")]
    return groups


def reconstruction_scores(
    truth: dict[str, np.ndarray],
    preds: dict[str, np.ndarray],
    masks: dict[str, dict[str, np.ndarray]] | None = None,
    dt: float = DT,
) -> list[Score]:
    """Rate R² and McFadden pseudo-R² of the predicted activity, per area and averaged over areas.

    For spike-count (Poisson) files the predicted area-* arrays are expected spike counts per bin
    (the Poisson mean):
      - McFadden R² scores them against the observed counts;
      - rate R² scores them against the true rates converted to counts per bin (rate-* x dt). It
        is computed for each neuron over trials and time, then averaged over neurons. Files
        without rate-* (real recordings) get McFadden R² only.
    For continuous (Gaussian) files, the prediction is scored by "activity r2", the same per-neuron
    R² against the area-* arrays themselves.

    With observation `masks` ({area: {"held-in": idx, "held-out": idx}}), every score is also
    reported over the held-in and held-out neurons separately.
    """
    scores = []
    groups = neuron_groups(masks)
    per_group: dict[str, dict[str, list[float]]] = {suffix: {} for suffix, _ in groups}

    for area in area_names(truth):
        observed = truth[f"{AREA_PREFIX}{area}"]
        pred = preds[f"{AREA_PREFIX}{area}"]
        counts = is_count_data(observed)
        rates = truth.get(f"{RATE_PREFIX}{area}")
        rates_per_bin = np.asarray(rates, dtype=np.float64) * dt if rates is not None else None

        for suffix, label in groups:
            if label is None:
                neurons = slice(None)
            else:
                idx = (masks or {}).get(area, {}).get(label)
                if idx is None or idx.size == 0:
                    continue
                neurons = idx

            values = {}
            if counts:
                values["mcfadden r2"] = mcfadden_r2(observed[..., neurons], pred[..., neurons])
                if rates_per_bin is not None:
                    values["rate r2"] = per_neuron_r2(rates_per_bin[..., neurons], pred[..., neurons])
            else:
                values["activity r2"] = per_neuron_r2(observed[..., neurons], pred[..., neurons])

            for metric, value in values.items():
                scores.append(Score(f"area-{area}", f"{metric}{suffix}", value))
                per_group[suffix].setdefault(metric, []).append(value)

    # Average each score over areas.
    for suffix, _ in groups:
        for metric, values in per_group[suffix].items():
            scores.append(Score("all", f"{metric}{suffix}", nan_mean(values)))
    return scores


def nan_mean(values) -> float:
    """Mean that ignores nan; nan if there is nothing to average."""
    arr = np.asarray(values, dtype=np.float64)
    if not np.isfinite(arr).any():
        return float("nan")
    return float(np.nanmean(arr))
