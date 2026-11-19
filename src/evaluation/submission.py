"""Evaluate a whole submission: a folder of prediction files laid out like the dataset release.

    <data root>/                              <submission root>/
      delayed_memory/poisson/test.h5            delayed_memory/poisson/preds.h5
      relay_decision/poisson/test.h5            relay_decision/poisson/preds.h5
      cognitive_task_suite/dm_1/network_1/      cognitive_task_suite/dm_1/network_1/
          poisson/test.h5                           poisson/preds.h5
      ...                                       ...

Every preds.h5 in the submission is scored against the file of the chosen split (test.h5 by
default) in the matching folder of the data root. A submission may cover any subset of the
datasets.
"""

from __future__ import annotations

import fnmatch
import math
import warnings
from pathlib import Path

import numpy as np

from .evaluate import EvalOptions, evaluate_file
from .io import check_predictions, load_h5

PREDICTION_NAME = "preds.h5"
SPLITS = ("train", "val", "test")

# Scores shown in the summary table, in this order (when present).
HEADLINE_METRICS = (
    ("all", "rate r2 held-out"),
    ("all", "mcfadden r2 held-out"),
    ("all", "activity r2 held-out"),
    ("communication", "norm effectome"),
    ("communication", "norm effectome pearson"),
    ("communication", "dynamic effectome"),
    ("communication", "communication fidelity"),
    ("communication", "communication specificity"),
)


def find_predictions(submission_root: Path | str, include: list[str] | None = None) -> list[Path]:
    """Folders (relative to the submission root) that hold a preds.h5, sorted.

    `include` keeps only folders matching one of the shell patterns, e.g. ["cognitive_task_suite/*"].
    """
    root = Path(submission_root).expanduser()
    if not root.is_dir():
        raise NotADirectoryError(f"no such submission folder: {root}")

    folders = sorted(p.parent.relative_to(root) for p in root.rglob(PREDICTION_NAME))
    if include:
        folders = [f for f in folders if any(fnmatch.fnmatch(f.as_posix(), pattern) for pattern in include)]
    return folders


def find_datasets(data_root: Path | str, split: str = "test") -> list[Path]:
    """Folders (relative to the data root) that hold a file of `split`, sorted."""
    root = Path(data_root).expanduser()
    return sorted(p.parent.relative_to(root) for p in root.rglob(f"{split}.h5"))


def _paths(data_root: Path, submission_root: Path, folder: Path, split: str) -> tuple[Path, Path]:
    return data_root / folder / f"{split}.h5", submission_root / folder / PREDICTION_NAME


def check_submission(
    data_root: Path | str,
    submission_root: Path | str,
    split: str = "test",
    include: list[str] | None = None,
) -> dict[str, list[str]]:
    """Check every prediction file's format without scoring it.

    Returns {dataset folder: problems}; an empty list means the file is ready to score. Warnings
    about ignored datasets are listed too, prefixed with "warning:".
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    data_root, submission_root = Path(data_root).expanduser(), Path(submission_root).expanduser()

    report = {}
    folders = find_predictions(submission_root, include)
    if not folders:
        raise FileNotFoundError(f"no {PREDICTION_NAME} files under {submission_root}")

    for folder in folders:
        truth_path, preds_path = _paths(data_root, submission_root, folder, split)
        if not truth_path.is_file():
            report[folder.as_posix()] = [f"no ground truth at {truth_path}"]
            continue
        try:
            truth, _ = load_h5(truth_path)
            preds, _ = load_h5(preds_path)
            report[folder.as_posix()] = [f"warning: {w}" for w in check_predictions(truth, preds)]
        except (OSError, ValueError, KeyError) as err:
            report[folder.as_posix()] = str(err).splitlines()
    return report


def evaluate_submission(
    data_root: Path | str,
    submission_root: Path | str,
    options: EvalOptions | None = None,
    split: str = "test",
    include: list[str] | None = None,
    keep_going: bool = False,
    progress: bool = True,
) -> list[dict]:
    """Score every preds.h5 in a submission. Returns one row per score:
    {"dataset", "name", "metric", "value"}.

    With keep_going, a file that fails is reported (as an "error" row) and the rest are still
    scored; otherwise the first failure stops the evaluation.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    options = options or EvalOptions()
    data_root, submission_root = Path(data_root).expanduser(), Path(submission_root).expanduser()
    if not data_root.is_dir():
        raise NotADirectoryError(f"no such data folder: {data_root}")

    folders = find_predictions(submission_root, include)
    if not folders:
        raise FileNotFoundError(f"no {PREDICTION_NAME} files under {submission_root}")

    rows = []
    for i, folder in enumerate(folders, start=1):
        dataset = folder.as_posix()
        truth_path, preds_path = _paths(data_root, submission_root, folder, split)
        if progress:
            print(f"[{i}/{len(folders)}] {dataset}", flush=True)

        try:
            if not truth_path.is_file():
                raise FileNotFoundError(f"no ground truth at {truth_path}")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                scores = evaluate_file(truth_path, preds_path, options)
            if progress:
                for message in dict.fromkeys(str(w.message) for w in caught):
                    print(f"    warning: {message}")
        except Exception as err:  # noqa: BLE001 - reported per file when keep_going
            if not keep_going:
                raise RuntimeError(f"{dataset}: {err}") from err
            rows.append({"dataset": dataset, "name": "error", "metric": type(err).__name__, "value": float("nan"),
                         "error": str(err).replace("\n", " ")})
            continue

        for score in scores:
            rows.append({"dataset": dataset, "name": score.name, "metric": score.metric, "value": score.value,
                         **options.tags})
    return rows


def family_of(dataset: str) -> str:
    """The task family of a dataset folder: its first path component (e.g. "cognitive_task_suite")."""
    return dataset.split("/", 1)[0]


def summarize(rows: list[dict]) -> list[dict]:
    """The headline scores of each dataset, then their mean over each task family.

    Family means average over datasets (for example the Cognitive Task Suite's tasks and networks),
    ignoring datasets where a score is nan.
    """
    by_dataset: dict[str, dict[str, float]] = {}
    for row in rows:
        key = (row["name"], row["metric"])
        if key in HEADLINE_METRICS:
            by_dataset.setdefault(row["dataset"], {})[row["metric"]] = row["value"]

    metrics = [m for _, m in HEADLINE_METRICS if any(m in v for v in by_dataset.values())]
    summary = [{"dataset": d, **{m: v.get(m, float("nan")) for m in metrics}} for d, v in by_dataset.items()]

    families: dict[str, list[dict]] = {}
    for row in summary:
        families.setdefault(family_of(row["dataset"]), []).append(row)
    for family, members in families.items():
        if len(members) < 2:
            continue
        mean_row = {"dataset": f"{family} (mean of {len(members)})"}
        for m in metrics:
            values = np.asarray([r[m] for r in members], dtype=np.float64)
            mean_row[m] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
        summary.append(mean_row)
    return summary


def format_summary(summary: list[dict]) -> str:
    """The summary as an aligned text table."""
    if not summary:
        return "(nothing to summarize)"
    columns = list(dict.fromkeys(k for row in summary for k in row))
    widths = {c: max(len(c), *(len(_cell(row.get(c))) for row in summary)) for c in columns}
    lines = ["  ".join(f"{c:{widths[c]}}" for c in columns)]
    for row in summary:
        lines.append("  ".join(f"{_cell(row.get(c)):{widths[c]}}" for c in columns))
    return "\n".join(lines)


def _cell(value) -> str:
    if isinstance(value, float):
        return "nan" if math.isnan(value) else f"{value:.3f}"
    return "" if value is None else str(value)
