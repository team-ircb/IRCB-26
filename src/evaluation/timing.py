"""Pairing predicted messages with true messages in time.

A ground-truth message-* is indexed by the bin in which the source area emits each message; the
target area receives it `delay` bins later (see layouts.message_delay). A predicted message-* or
pathway-* dataset is read the same way by default (timing = "emitted").

Some models infer the message a target receives in bin t, for example a model that predicts each
area from the others' recent activity. Such a prediction sets the dataset attribute
timing = "received", and bin t is then paired with the true message emitted `delay` bins earlier.
The first `delay` predicted bins and the last `delay` true bins have no partner and are left out.

The delay is a property of the dataset, never of the model, so a model can't choose it.
"""

from __future__ import annotations

import numpy as np

from .io import attribute_text
from .layouts import message_delay, message_keys

TIMINGS = ("emitted", "received")


def prediction_timing(pred_attrs: dict[str, dict], key: str) -> str:
    """The `timing` attribute of a predicted dataset ("emitted" if it has none)."""
    timing = attribute_text(pred_attrs.get(key, {}).get("timing", "emitted"))
    if timing not in TIMINGS:
        raise ValueError(f"{key}: the timing attribute must be one of {TIMINGS}, got {timing!r}")
    return timing


def resolve_delay(key: str, truth: dict, attrs: dict, override: int | None = None) -> int:
    """The delay of ground-truth message `key`: `override` if given, otherwise the dataset's own."""
    delay = override if override is not None else message_delay(key, truth, attrs)
    if delay is None:
        raise ValueError(
            f"{key}: a prediction uses timing='received', but this file's message delay is unknown. "
            "Pass it with --message-delay, or add a delivery_delay attribute to the ground-truth message."
        )
    delay = int(delay)
    if delay < 0:
        raise ValueError(f"{key}: the message delay must be >= 0, got {delay}")
    if delay >= truth[key].shape[1]:
        raise ValueError(f"{key}: a delay of {delay} bins leaves no bins to compare")
    return delay


def pair_received(true_message: np.ndarray, pred_message: np.ndarray, delay: int) -> tuple[np.ndarray, np.ndarray]:
    """Pair the prediction at bin t with the true message emitted at bin t - delay."""
    if delay == 0:
        return true_message, pred_message
    return true_message[:, :-delay], pred_message[:, delay:]


def shared_pathway_delay(truth: dict, attrs: dict, pred_attrs: dict, pathway_keys: list[str], override: int | None) -> int:
    """The bins to drop when comparing pathway-* predictions with the truth: 0 for emitted timing,
    the dataset's delay for received timing. Every pathway-* of a prediction must use one timing."""
    timings = {prediction_timing(pred_attrs, key) for key in pathway_keys}
    if len(timings) > 1:
        raise ValueError("the pathway-* datasets mix timing='emitted' and timing='received'; use one for all")
    if timings != {"received"}:
        return 0

    delays = {resolve_delay(key, truth, attrs, override) for key in message_keys(truth)}
    if len(delays) > 1:
        raise ValueError(f"the ground-truth messages have different delays {sorted(delays)}")
    return delays.pop() if delays else 0
