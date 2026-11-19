"""Reading ground-truth and prediction files, and checking that they line up.

Both kinds of file are HDF5 with one dataset per array at the root (or, for older files, under a
group named "0"). Arrays are (trials, time, channels), with names such as:

    area-<AREA>                    neural activity of one area
    rate-<AREA>                    true firing rates in Hz (Poisson test files only)
    message-<NAME>                 true or predicted communication, every pathway in one array
    pathway-<SRC>-to-<TGT>         predicted communication on one directed pathway
    truth-<NAME>                   task variables and external inputs

See docs/submission_format.md for what a prediction file may contain.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

AREA_PREFIX = "area-"
RATE_PREFIX = "rate-"
MESSAGE_PREFIX = "message-"
PATHWAY_PREFIX = "pathway-"
TRUTH_PREFIX = "truth-"


def load_h5(path: Path | str) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """Load every dataset of an HDF5 file into memory.

    Returns (arrays, attrs): arrays maps dataset name to array, and attrs maps dataset name to a
    dict of that dataset's attributes. Groups (such as observation masks) are skipped.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such file: {path}")

    arrays: dict[str, np.ndarray] = {}
    attrs: dict[str, dict] = {}
    with h5py.File(path, "r") as f:
        # Some early files keep their arrays in a group called "0".
        node = f["0"] if isinstance(f.get("0"), h5py.Group) else f
        for key, obj in node.items():
            if not isinstance(obj, h5py.Dataset):
                continue
            arrays[key] = np.asarray(obj)
            attrs[key] = {k: obj.attrs[k] for k in obj.attrs.keys()}
    return arrays, attrs


def load_observation_masks(train_h5: Path | str, mask: str = "pct_100") -> dict[str, dict[str, np.ndarray]]:
    """Held-in and held-out neuron indices of one observation mask, read from a train.h5 file.

    `mask` names a group under "observation", such as "pct_100", "pct_50" or "regional/drop_A0".
    Returns {area: {"held-in": indices, "held-out": indices}}.
    """
    path = Path(train_h5)
    if not path.is_file():
        raise FileNotFoundError(f"no such file: {path}")

    masks: dict[str, dict[str, np.ndarray]] = {}
    with h5py.File(path, "r") as f:
        group_name = f"observation/{mask}"
        if group_name not in f:
            available = list_observation_masks(f)
            raise KeyError(f"{path} has no observation mask {mask!r}; available: {available}")

        group = f[group_name]
        for kind, label in (("observed_idx", "held-in"), ("held_out_idx", "held-out")):
            if kind not in group:
                continue
            for area in group[kind].keys():
                masks.setdefault(area, {})[label] = np.asarray(group[kind][area], dtype=np.int64)
    return masks


def list_observation_masks(f: h5py.File) -> list[str]:
    """Names of the observation masks in an open train.h5 file, e.g. ["pct_100", "regional/drop_A0"]."""
    names = []
    if "observation" not in f:
        return names

    def visit(name, obj):
        if isinstance(obj, h5py.Group) and "observed_idx" in obj:
            names.append(name)

    f["observation"].visititems(visit)
    return sorted(names)


def area_names(arrays: dict[str, np.ndarray]) -> list[str]:
    """Sorted area names of a file, e.g. ["A0", "A1", "A2"] for area-A0, area-A1 and area-A2."""
    return sorted(k[len(AREA_PREFIX):] for k in arrays if k.startswith(AREA_PREFIX))


def as_strings(value) -> list[str]:
    """An HDF5 attribute holding one or more names (bytes or str) as a list of str."""
    items = np.atleast_1d(np.asarray(value)).reshape(-1)
    return [item.decode() if isinstance(item, bytes) else str(item) for item in items]


def attribute_text(value) -> str | None:
    """An HDF5 string attribute as str (None stays None)."""
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray) and value.shape == ():
        return attribute_text(value.item())
    return str(value)


def check_predictions(truth: dict[str, np.ndarray], preds: dict[str, np.ndarray]) -> list[str]:
    """Check that a prediction file lines up with its ground truth.

    Raises ValueError listing every problem found. Returns a list of warnings about datasets that
    will be ignored, so a user can spot a misspelled name.
    """
    problems = []
    warnings = []
    truth_areas = area_names(truth)
    pred_areas = area_names(preds)

    if not truth_areas:
        problems.append("the ground-truth file has no area-* datasets")
    if not pred_areas:
        problems.append("the prediction file has no area-* datasets (predicted activity is required)")
    elif pred_areas != truth_areas:
        missing = sorted(set(truth_areas) - set(pred_areas))
        extra = sorted(set(pred_areas) - set(truth_areas))
        if missing:
            problems.append(f"missing predicted areas: {', '.join('area-' + a for a in missing)}")
        if extra:
            problems.append(f"areas not in the ground truth: {', '.join('area-' + a for a in extra)}")

    n_trials, n_time = None, None
    for area in truth_areas:
        true = truth[f"{AREA_PREFIX}{area}"]
        if n_trials is None and true.ndim == 3:
            n_trials, n_time = true.shape[:2]
        if area not in pred_areas:
            continue

        pred = preds[f"{AREA_PREFIX}{area}"]
        if pred.ndim != 3:
            problems.append(f"area-{area} should be (trials, time, neurons), got shape {pred.shape}")
            continue
        if pred.shape != true.shape:
            problems.append(
                f"area-{area} has shape {pred.shape}; the ground truth has {true.shape} "
                "(predict every neuron, held-out ones included)"
            )
            continue
        if not np.issubdtype(pred.dtype, np.number):
            problems.append(f"area-{area} is not numeric ({pred.dtype})")
            continue
        if not np.isfinite(pred).all():
            problems.append(f"area-{area} contains nan or inf")
        elif is_count_data(true) and (pred < 0).any():
            problems.append(
                f"area-{area} has negative values; predictions for spike counts must be expected "
                "counts per bin (>= 0)"
            )

    # Communication, inputs and task variables are optional, but must match trials and time.
    for key, pred in preds.items():
        if key.startswith(AREA_PREFIX):
            continue
        if not key.startswith((MESSAGE_PREFIX, PATHWAY_PREFIX, TRUTH_PREFIX)):
            warnings.append(f"{key}: not a recognised dataset name, ignored")
            continue
        if key.startswith((MESSAGE_PREFIX, TRUTH_PREFIX)) and key not in truth:
            warnings.append(f"{key}: no dataset of this name in the ground truth, ignored")
            continue
        if not np.issubdtype(pred.dtype, np.number):
            problems.append(f"{key} is not numeric ({pred.dtype})")
            continue
        if pred.ndim != 3:
            problems.append(f"{key} should be (trials, time, channels), got shape {pred.shape}")
            continue
        if n_trials is not None and pred.shape[:2] != (n_trials, n_time):
            problems.append(f"{key} has trials/time {pred.shape[:2]}; the ground truth has {(n_trials, n_time)}")
            continue
        if pred.shape[-1] == 0:
            problems.append(f"{key} has no channels")
            continue
        if not np.isfinite(pred).all():
            problems.append(f"{key} contains nan or inf")

    if problems:
        raise ValueError("the predictions do not line up with the ground truth:\n  - " + "\n  - ".join(problems))
    return warnings


def is_count_data(arr: np.ndarray) -> bool:
    """True if an array looks like spike counts: nonnegative integers (Poisson files)."""
    a = np.asarray(arr)
    if a.size == 0:
        return False
    if np.issubdtype(a.dtype, np.integer):
        return bool((a >= 0).all())
    sample = a.reshape(-1)[: 1_000_000]
    return bool(np.isfinite(sample).all() and (sample >= 0).all() and np.all(sample == np.round(sample)))
