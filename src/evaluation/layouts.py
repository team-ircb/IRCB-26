"""Where each pathway lives in a ground-truth message, and when its messages arrive.

A ground-truth `message-*` dataset holds every pathway of the network side by side in its last
axis. To score a pathway we need its channel slice, its source and target areas, and the weights
that carry it into the target area (`comm_inp_weights` on the target's `area-*` dataset).

The channel layout is found in this order:
  1. a JSON `layout` attribute on the message dataset (list of {source, target, start, end, output}),
     written by scripts/generate_multitask_datasets.py;
  2. Delayed-Memory's packing: one block per (target, source) pair in row-major order, where the
     diagonal blocks are each area's private input;
  3. a single pathway that fills the whole message (Relay-Decision);
  4. the bundled table of released Cognitive Task Suite graphs (data/cts_message_layouts.json).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .io import AREA_PREFIX, MESSAGE_PREFIX, area_names, as_strings, attribute_text

DATA_DIR = Path(__file__).with_name("data")

# Bins between a message's index in message-* (the step its source area emits it) and the step its
# target area receives it, for the released datasets (see src/models/):
#   Relay-Decision: the R -> D message is used in the step it is computed (0).
#   Delayed-Memory: a message emitted at step t is the target's input at step t + 1 (1).
#   Cognitive Task Suite: the network's delay is 5 steps, plus one step to enter the update (6).
# Newer files store the delay on the message itself (`delivery_delay` or `lag_steps`).
RELEASED_MESSAGE_DELAYS = {"relay_decision": 0, "delayed_memory": 1, "cognitive_task_suite": 6}


@dataclass(frozen=True)
class Pathway:
    """One directed pathway of a ground-truth message: channels [start, end) carry source -> target."""

    source: str
    target: str
    source_idx: int
    target_idx: int
    start: int
    end: int
    weights: np.ndarray  # (hidden, channels) map from this message into the target area

    @property
    def name(self) -> str:
        return f"{self.source}->{self.target}"

    @property
    def width(self) -> int:
        return self.end - self.start


def message_keys(arrays: dict) -> list[str]:
    """Names of the message-* datasets in a file."""
    return [key for key in arrays if key.startswith(MESSAGE_PREFIX)]


def incoming_weights(area: str, attrs: dict[str, dict]) -> dict[str, np.ndarray]:
    """{source area: (hidden, channels) weights} for the messages `area` receives.

    Read from the `comm_inp_source` and `comm_inp_weights` attributes of area-<area> in a
    ground-truth test file. Areas that receive nothing give {}.
    """
    raw = attrs.get(f"{AREA_PREFIX}{area}", {})
    if "comm_inp_source" not in raw or "comm_inp_weights" not in raw:
        return {}

    sources = as_strings(raw["comm_inp_source"])
    weights = np.asarray(raw["comm_inp_weights"], dtype=np.float64)
    if weights.ndim == 2:
        blocks = [weights]
    elif weights.ndim == 3:
        blocks = list(weights)
    else:
        raise ValueError(f"area-{area}: comm_inp_weights should be 2-D or 3-D, got shape {weights.shape}")

    if len(blocks) != len(sources):
        raise ValueError(f"area-{area}: {len(sources)} comm_inp_source names but {len(blocks)} weight blocks")
    return dict(zip(sources, blocks))


def graph_signature(areas: list[str], attrs: dict[str, dict]) -> str:
    """The communication graph as text, e.g. "A0=A3,A2|A1=A3|...". Identifies a released
    Cognitive Task Suite network in data/cts_message_layouts.json."""
    parts = []
    for area in areas:
        sources = list(incoming_weights(area, attrs))
        parts.append(area + "=" + ",".join(sources))
    return "|".join(parts)


@lru_cache(maxsize=1)
def released_cts_layouts() -> dict:
    """Channel layouts of the released Cognitive Task Suite networks, keyed by graph signature."""
    with (DATA_DIR / "cts_message_layouts.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def _layout_from_attribute(layout_json) -> list[dict]:
    text = attribute_text(layout_json)
    try:
        spec = json.loads(text)
    except json.JSONDecodeError as err:
        raise ValueError(f"the message's layout attribute is not valid JSON: {err}") from None
    return [
        {
            "source": str(item["source"]),
            "target": str(item["target"]),
            "start": int(item["start"]),
            "end": int(item["end"]),
            "output": bool(item.get("output", False)),
        }
        for item in spec
    ]


def _packed_layout(areas: list[str], arrays: dict, attrs: dict[str, dict]) -> list[dict]:
    # Delayed-Memory: row-major over (target, source). Diagonal blocks are each area's private
    # input, as wide as truth-inp's share per area; off-diagonal blocks are as wide as their weights.
    n = len(areas)
    private = 0
    inputs = arrays.get("truth-inp")
    if inputs is not None and n and inputs.shape[-1] % n == 0:
        private = int(inputs.shape[-1] // n)

    layout, offset = [], 0
    for target in areas:
        incoming = incoming_weights(target, attrs)
        for source in areas:
            width = private if source == target else (incoming[source].shape[-1] if source in incoming else 0)
            if width <= 0:
                continue
            layout.append({"source": source, "target": target, "start": offset, "end": offset + width, "output": False})
            offset += width
    return layout


def _covers(layout: list[dict], width: int) -> bool:
    # True when the blocks tile [0, width) with no gaps or overlaps.
    cursor = 0
    for block in sorted(layout, key=lambda b: (b["start"], b["end"])):
        if block["start"] != cursor or block["end"] <= block["start"]:
            return False
        cursor = block["end"]
    return cursor == width


def message_layout(key: str, arrays: dict, attrs: dict[str, dict]) -> list[dict]:
    """The channel layout of a ground-truth message: every block, including private-input
    (source == target) and readout (output = True) blocks."""
    areas = area_names(arrays)
    width = int(arrays[key].shape[-1])

    own = attrs.get(key, {}).get("layout")
    if own is not None:
        layout = _layout_from_attribute(own)
    else:
        packed = _packed_layout(areas, arrays, attrs)
        edges = [(s, t) for t in areas for s in incoming_weights(t, attrs) if s in areas]

        if packed and sum(b["end"] - b["start"] for b in packed) == width:
            layout = packed
        elif len(edges) == 1 and incoming_weights(edges[0][1], attrs)[edges[0][0]].shape[-1] == width:
            source, target = edges[0]
            layout = [{"source": source, "target": target, "start": 0, "end": width, "output": False}]
        else:
            layout = released_cts_layouts().get(graph_signature(areas, attrs))
            if layout is None:
                raise ValueError(
                    f"{key}: can't tell which channels belong to which pathway ({width} channels). "
                    "Add a JSON 'layout' attribute to the message dataset."
                )

    if not _covers(layout, width):
        raise ValueError(f"{key}: the layout does not tile its {width} channels exactly")
    return layout


def message_pathways(key: str, arrays: dict, attrs: dict[str, dict]) -> list[Pathway]:
    """The inter-area pathways of a ground-truth message, with their channel slices and weights.

    Private-input and readout blocks are not pathways and are left out.
    """
    areas = area_names(arrays)
    pathways = []
    for block in message_layout(key, arrays, attrs):
        source, target = block["source"], block["target"]
        if block["output"] or source == target:
            continue
        if source not in areas or target not in areas:
            raise ValueError(f"{key}: pathway {source} -> {target} is not between recorded areas")

        weights = incoming_weights(target, attrs).get(source)
        width = block["end"] - block["start"]
        if weights is None or weights.shape[-1] != width:
            raise ValueError(f"area-{target}: comm_inp_weights do not match the {source} -> {target} message")

        pathways.append(
            Pathway(source, target, areas.index(source), areas.index(target), block["start"], block["end"], weights)
        )

    if not pathways:
        raise ValueError(f"{key}: the message has no inter-area pathways")
    return pathways


def has_pathway_info(arrays: dict, attrs: dict[str, dict]) -> bool:
    """True if the ground truth says which channels belong to which pathway."""
    if any("layout" in attrs.get(key, {}) for key in message_keys(arrays)):
        return True
    return any(incoming_weights(area, attrs) for area in area_names(arrays))


def dataset_family(arrays: dict, attrs: dict[str, dict]) -> str | None:
    """Which kind of network a ground-truth file comes from: "relay_decision", "delayed_memory",
    "cognitive_task_suite" (single-task), "multitask", or None if it can't be told."""
    keys = message_keys(arrays)
    if "message-r-to-d" in keys:
        return "relay_decision"
    if "message-mesgs" not in keys:
        return None
    if "layout" in attrs.get("message-mesgs", {}):
        return "multitask" if "truth-rule" in arrays else None
    if "truth-inp" in arrays:
        return "delayed_memory"
    if "truth-fix" in arrays and graph_signature(area_names(arrays), attrs) in released_cts_layouts():
        return "cognitive_task_suite"
    return None


def message_delay(key: str, arrays: dict, attrs: dict[str, dict]) -> int | None:
    """Bins between when a ground-truth message is emitted (its index in message-*) and when its
    target receives it. None when the file doesn't say and isn't a released dataset."""
    own = attrs.get(key, {})
    for name in ("delivery_delay", "lag_steps"):
        if name in own:
            return int(np.asarray(own[name]))

    family = dataset_family(arrays, attrs)
    return RELEASED_MESSAGE_DELAYS.get(family)
