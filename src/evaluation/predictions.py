"""Writing prediction files in the format the evaluator reads (see docs/submission_format.md)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from .effectome import WEIGHTS_ATTRIBUTE
from .subspace import READOUT_ATTRIBUTE
from .timing import TIMINGS


def _as_array(name: str, value, ndim: int = 3) -> np.ndarray:
    arr = np.asarray(value)
    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"{name} must be numeric, got {arr.dtype}")
    if arr.ndim != ndim:
        raise ValueError(f"{name} should have {ndim} dimensions, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains nan or inf")
    return arr.astype(np.float32)


def save_predictions(
    path: Path | str,
    activity: dict[str, np.ndarray],
    pathways: dict[tuple[str, str], np.ndarray] | None = None,
    messages: dict[str, np.ndarray] | None = None,
    inputs: dict[str, np.ndarray] | None = None,
    timing: str = "emitted",
    pathway_weights: dict[tuple[str, str], np.ndarray] | None = None,
    source_readouts: dict[tuple[str, str], np.ndarray] | None = None,
) -> Path:
    """Write a preds.h5 file.

    activity         {area: (trials, time, neurons)} predicted activity of every neuron, held-out
                     ones included. For spike-count data, expected counts per bin (>= 0).
    pathways         {(source, target): (trials, time, channels)} inferred message on each
                     directed pathway the model reports (used for the effectome and per-pathway scores).
    messages         {name: (trials, time, channels)} for message-<name>, laid out like the
                     ground-truth message (an alternative to `pathways` for communication scores).
    inputs           {name: (trials, time, channels)} inferred external inputs, e.g. {"inp": ...}
                     for truth-inp, or {"fix": ..., "stim1": ...} for the Cognitive Task Suite.
    timing           "emitted" (bin t is the message sent at t) or "received" (bin t is the
                     message the target receives at t). Applies to every message and pathway.
    pathway_weights  {(source, target): (hidden, channels)} the model's map from each pathway's
                     message into the target's dynamics, for the dynamic effectome.
    source_readouts  {(source, target): (channels, source neurons)} the model's linear readout of
                     the source area's neurons, for subspace alignment.
    """
    if timing not in TIMINGS:
        raise ValueError(f"timing must be one of {TIMINGS}, got {timing!r}")
    if not activity:
        raise ValueError("activity is required: predicted area-* arrays for every area")

    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        for area, values in activity.items():
            arr = _as_array(f"activity[{area!r}]", values)
            f.create_dataset(f"area-{area}", data=arr)

        for (source, target), values in (pathways or {}).items():
            name = f"pathway-{source}-to-{target}"
            dataset = f.create_dataset(name, data=_as_array(name, values))
            dataset.attrs["timing"] = timing

            weights = (pathway_weights or {}).get((source, target))
            if weights is not None:
                dataset.attrs[WEIGHTS_ATTRIBUTE] = _as_array(f"{name} weights", weights, ndim=2)
            readout = (source_readouts or {}).get((source, target))
            if readout is not None:
                dataset.attrs[READOUT_ATTRIBUTE] = _as_array(f"{name} readout", readout, ndim=2)

        for name, values in (messages or {}).items():
            key = name if name.startswith("message-") else f"message-{name}"
            dataset = f.create_dataset(key, data=_as_array(key, values))
            dataset.attrs["timing"] = timing

        for name, values in (inputs or {}).items():
            key = name if name.startswith("truth-") else f"truth-{name}"
            f.create_dataset(key, data=_as_array(key, values))

    return path
