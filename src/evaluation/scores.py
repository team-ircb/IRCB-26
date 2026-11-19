"""The Score record every evaluation function returns, and helpers to print and save scores."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import NamedTuple


class Score(NamedTuple):
    """One number: `metric` (e.g. "rate r2 held-out") for `name` (an area such as "area-A0", a
    pathway such as "A0->A1", or "all" / "communication" for an average)."""

    name: str
    metric: str
    value: float


def format_scores(scores: list[Score]) -> str:
    """Scores as an aligned text table."""
    if not scores:
        return "(no scores)"
    width_name = max(len("name"), *(len(s.name) for s in scores))
    width_metric = max(len("metric"), *(len(s.metric) for s in scores))
    lines = [f"{'name':{width_name}}  {'metric':{width_metric}}  value"]
    for s in scores:
        value = "nan" if math.isnan(s.value) else f"{s.value:.4f}"
        lines.append(f"{s.name:{width_name}}  {s.metric:{width_metric}}  {value}")
    return "\n".join(lines)


def save_scores(path: Path | str, rows: list[dict]) -> Path:
    """Save score rows (dicts with the same keys) as .csv or .json, chosen by the file extension."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("no scores to save")

    if path.suffix.lower() == ".json":
        clean = [{k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()} for row in rows]
        path.write_text(json.dumps(clean, indent=1) + "\n")
        return path

    if path.suffix.lower() != ".csv":
        raise ValueError(f"save scores as .csv or .json, not {path.suffix or 'no extension'}")

    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path
