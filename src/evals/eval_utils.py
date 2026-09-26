"""Helper functions and metrics used for evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


# Loads a .h5 file into a dictionary of arrays and attributes.
def load_h5(path: Path) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    arrays: dict[str, np.ndarray] = {}
    attrs: dict[str, dict] = {}
    with h5py.File(path, "r") as f:
        node = f["0"] if isinstance(f.get("0"), h5py.Group) else f
        for key in node.keys():
            obj = node[key]
            if not isinstance(obj, h5py.Dataset):
                continue
            arrays[key] = np.asarray(obj)
            attrs[key] = {k: obj.attrs[k] for k in obj.attrs.keys()}
    return arrays, attrs


# Gets the names of the areas in the .h5 file (e.g. "area-A0", "area-A1", ...)
def area_names(arrays: dict[str, np.ndarray]) -> list[str]:
    return sorted(k[len("area-") :] for k in arrays if k.startswith("area-"))


# Converts a list of bytes or strings to a list of strings.
def as_names(value) -> list[str]:
    arr = np.atleast_1d(np.asarray(value)).reshape(-1)
    names = []
    for item in arr:
        if isinstance(item, bytes):
            names.append(item.decode())
        else:
            names.append(str(item))
    return names


# Gets the incoming weights for a given area (from the ground-truth test.h5 file)
def incoming_weights(area: str, attrs: dict[str, dict]) -> list[tuple[str, np.ndarray]]:
    # comm_inp_weights on the ground-truth area is (hidden, channels) or (n_sources, hidden, channels).
    raw = attrs.get(f"area-{area}", {})
    if "comm_inp_source" not in raw or "comm_inp_weights" not in raw:
        return []
    sources = as_names(raw["comm_inp_source"])
    weights = np.asarray(raw["comm_inp_weights"], dtype=np.float64)
    if weights.ndim == 2:
        blocks = [weights]
    elif weights.ndim == 3:
        blocks = [weights[i] for i in range(weights.shape[0])]
    else:
        return []
    if len(blocks) != len(sources):
        return []
    return list(zip(sources, blocks))


# Gets the message slots for a given area (from the ground-truth test.h5 file)
def message_slots(areas: list[str], arrays: dict[str, np.ndarray], attrs: dict[str, dict]):
    # Row-major target, source. Diagonal slots are the private input, not a comm weight.
    n = len(areas)
    truth = arrays.get("truth-inp")
    rank = 0
    if truth is not None and truth.ndim >= 1 and n and truth.shape[-1] % n == 0:
        rank = int(truth.shape[-1] // n)
    incoming = {area: dict(incoming_weights(area, attrs)) for area in areas}
    slots = []
    offset = 0
    for ti, tgt in enumerate(areas):
        for sj, src in enumerate(areas):
            if src == tgt:
                width = rank
                weight = None
            else:
                weight = incoming[tgt].get(src)
                width = 0 if weight is None else int(weight.shape[-1])
            if width <= 0:
                continue
            if src != tgt and (weight is None or weight.shape[-1] != width):
                raise ValueError(f"area-{tgt} is missing comm_inp_weights for {src}")
            slots.append(
                {
                    "target": tgt,
                    "source": src,
                    "target_idx": ti,
                    "source_idx": sj,
                    "self": src == tgt,
                    "output": False,
                    "width": width,
                    "start": offset,
                    "end": offset + width,
                    "weight": weight,
                }
            )
            offset += width
    return slots


# Released Cognitive Task Suite graphs. message-mesgs channel order is not stored in the h5.
def cts_layouts() -> dict:
    cached = getattr(cts_layouts, "cache", None)
    if cached is None:
        path = Path(__file__).with_name("cts_message_layouts.json")
        with path.open(encoding="utf-8") as handle:
            cached = json.load(handle)
        cts_layouts.cache = cached
    return cached


# Predecessor lists in area order. This identifies a released Cognitive Task Suite graph.
def graph_signature(areas: list[str], attrs: dict[str, dict]) -> str:
    parts = []
    for area in areas:
        sources = [src for src, _weight in incoming_weights(area, attrs)]
        parts.append(area + "=" + ",".join(sources))
    return "|".join(parts)


# True when the slots tile [0, width) with no gaps.
def layout_covers(layout: list[dict], width: int) -> bool:
    cursor = 0
    for slot in sorted(layout, key=lambda item: (item["start"], item["end"])):
        if slot["start"] != cursor or slot["end"] <= slot["start"]:
            return False
        cursor = slot["end"]
    return cursor == width


# Inter-area message blocks for effectome scoring, with channel slices into the message array.
def effectome_slots(areas: list[str], arrays: dict[str, np.ndarray], attrs: dict[str, dict], message_width: int):
    # Delayed-Memory packs private input and communication row-major, and that width matches the message.
    packed = message_slots(areas, arrays, attrs)
    packed_width = sum(slot["width"] for slot in packed)
    if packed and packed_width == message_width:
        layout = packed
    else:
        edges = []
        for ti, tgt in enumerate(areas):
            for src, weight in incoming_weights(tgt, attrs):
                if src not in areas:
                    continue
                edges.append((tgt, ti, src, areas.index(src), weight))
        # Relay-Decision is one message, R to D, whose width is the whole message.
        if len(edges) == 1 and int(edges[0][4].shape[-1]) == message_width:
            tgt, ti, src, sj, weight = edges[0]
            width = int(weight.shape[-1])
            layout = [
                {
                    "target": tgt,
                    "source": src,
                    "target_idx": ti,
                    "source_idx": sj,
                    "self": False,
                    "output": False,
                    "width": width,
                    "start": 0,
                    "end": width,
                    "weight": weight,
                }
            ]
        else:
            spec = cts_layouts().get(graph_signature(areas, attrs))
            if spec is None:
                raise ValueError(
                    f"message has {message_width} channels and does not match a Delayed-Memory, "
                    "Relay-Decision, or released Cognitive Task Suite layout"
                )
            incoming = {area: dict(incoming_weights(area, attrs)) for area in areas}
            layout = []
            for item in spec:
                start, end = int(item["start"]), int(item["end"])
                source, target = item["source"], item["target"]
                output = bool(item["output"])
                self_edge = source == target
                weight = None
                if not output and not self_edge and target in incoming:
                    weight = incoming[target].get(source)
                layout.append(
                    {
                        "target": target,
                        "source": source,
                        "target_idx": areas.index(target) if target in areas else None,
                        "source_idx": areas.index(source) if source in areas else None,
                        "self": self_edge,
                        "output": output,
                        "width": end - start,
                        "start": start,
                        "end": end,
                        "weight": weight,
                    }
                )

    if not layout_covers(layout, message_width):
        raise ValueError(f"message layout does not cover {message_width} channels")

    slots = []
    for slot in layout:
        if slot.get("self") or slot.get("output"):
            continue
        weight = slot["weight"]
        width = slot["end"] - slot["start"]
        if weight is None or weight.shape[-1] != width:
            raise ValueError(
                f"area-{slot['target']} comm_inp_weights do not match the {slot['source']} message"
            )
        if slot["target_idx"] is None or slot["source_idx"] is None:
            raise ValueError(
                f"communication edge {slot['source']} -> {slot['target']} is not between recorded areas"
            )
        slots.append(slot)
    if not slots:
        raise ValueError("message layout has no inter-area communication")
    return slots


# Checks that the predicted .h5 file is aligned with the ground-truth .h5 file so evaluation can be performed.
def check_alignment(truth: dict[str, np.ndarray], preds: dict[str, np.ndarray]) -> None:
    problems = []
    truth_areas = area_names(truth)
    pred_areas = area_names(preds)
    if not truth_areas:
        problems.append("ground truth has no area-* datasets")
    if pred_areas != truth_areas:
        problems.append(f"areas differ: truth {truth_areas}, preds {pred_areas}")

    # Check that the predicted area activity is aligned with the ground-truth area activity.
    for area in truth_areas:
        if area not in pred_areas:
            continue
        true = truth[f"area-{area}"]
        pred = preds[f"area-{area}"]
        if true.ndim != 3 or pred.ndim != 3:
            problems.append(f"area-{area} should be (trials, time, neurons)")
            continue
        if pred.shape != true.shape:
            problems.append(f"area-{area} shape {tuple(pred.shape)} != truth {tuple(true.shape)}")
        rate_key = f"rate-{area}"
        if rate_key not in truth:
            problems.append(f"ground truth is missing {rate_key}")
        elif truth[rate_key].shape != true.shape:
            problems.append(f"{rate_key} shape {tuple(truth[rate_key].shape)} != area-{area}")

    # Messages and task variables are optional. When both files have one, trials and time must match.
    for key, pred in preds.items():
        if key not in truth:
            continue
        if not (key.startswith("message-") or key.startswith("truth-")):
            continue
        true = truth[key]
        if pred.shape[0] != true.shape[0]:
            problems.append(f"{key} trials {pred.shape[0]} != truth {true.shape[0]}")
            continue
        if pred.ndim != true.ndim:
            problems.append(f"{key} ndim {pred.ndim} != truth {true.ndim}")
            continue
        if pred.ndim >= 3 and pred.shape[1] != true.shape[1]:
            problems.append(f"{key} time {pred.shape[1]} != truth {true.shape[1]}")

    if problems:
        raise ValueError("preds do not line up with the ground truth:\n" + "\n".join(problems))


# Computes the standard R² score for a given true and predicted array.
def standard_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_t = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_p = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_t.size != y_p.size:
        raise ValueError(f"shape mismatch after flatten: {y_t.size} vs {y_p.size}")
    if not (np.isfinite(y_t).all() and np.isfinite(y_p).all()):
        return float("nan")
    if np.var(y_t) < 1e-12:
        return float("nan")
    return float(r2_score(y_t, y_p))


# Computes the McFadden R² score for a given true and predicted array.
def mcfadden_r2_poisson(data: np.ndarray, output_params: np.ndarray) -> float:
    # 1 - nll(model) / nll(mean rate). Null is the mean count over trials.
    data_t = torch.from_numpy(np.asarray(data)).float()
    params = torch.from_numpy(np.asarray(output_params)).float()
    nll_model = F.poisson_nll_loss(input=params, target=data_t, log_input=False, full=True, reduction="none")
    output_null = torch.mean(data_t, dim=0, keepdim=True).expand_as(params)
    nll_null = F.poisson_nll_loss(input=output_null, target=data_t, log_input=False, full=True, reduction="none")
    mask = nll_null > 0
    denom = (nll_null * mask).mean()
    if torch.isclose(denom, torch.zeros_like(denom)):
        return float("nan")
    r2 = 1 - (nll_model * mask).mean() / denom
    return float(r2.item())


# Computes the cosine similarity between two arrays.
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


# Computes the L² volume of a given array.
def l2_volume(arr: np.ndarray) -> float:
    x = np.asarray(arr, dtype=np.float64)
    return float(np.sqrt(np.sum(np.square(x))))


# Computes the weighted L² volume of a given weight and message array.
def weighted_l2_volume(weight: np.ndarray, message: np.ndarray) -> float:
    # ||W m|| over trials and time. W is (hidden, channels) from the ground-truth area.
    w = np.asarray(weight, dtype=np.float64)
    m = np.asarray(message, dtype=np.float64)
    if m.ndim != 3:
        raise ValueError(f"expected message (trials, time, channels), got {m.shape}")
    if w.ndim != 2 or w.shape[1] != m.shape[-1]:
        raise ValueError(f"weight {w.shape} does not match message {m.shape}")
    contrib = np.einsum("btc,hc->bth", m, w)
    return l2_volume(contrib)


# Sets the diagonal of a matrix to 0.
def zero_diagonal(matrix: np.ndarray) -> np.ndarray:
    out = np.array(matrix, dtype=np.float64, copy=True)
    np.fill_diagonal(out, 0.0)
    return out


# Fits a linear map from x to y on 80% of trials and scores R² on the rest.
def linear_decode(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    # Trial-level variables such as stimulus strength have no time axis.
    if x.ndim == 2:
        x = x[:, None, :]
    if y.ndim == 2:
        y = y[:, None, :]
    if x.ndim != 3 or y.ndim != 3:
        raise ValueError(f"expected (trials, time, channels), got {x.shape} and {y.shape}")
    if x.shape[0] != y.shape[0] or x.shape[1] != y.shape[1]:
        raise ValueError(f"trials/time {tuple(x.shape[:2])} != {tuple(y.shape[:2])}")
    idx = np.arange(y.shape[0])
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=0, shuffle=True)
    x_train = x[train_idx].reshape(-1, x.shape[-1])
    y_train = y[train_idx].reshape(-1, y.shape[-1])
    decoder = LinearRegression()
    decoder.fit(x_train, y_train)
    mapped = decoder.predict(x[test_idx].reshape(-1, x.shape[-1]))
    mapped = mapped.reshape(len(test_idx), y.shape[1], y.shape[-1])
    r2 = standard_r2(y[test_idx], mapped)
    return y[test_idx], mapped, r2
