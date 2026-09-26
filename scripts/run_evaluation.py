"""Evaluate a predicted .h5 file (e.g. preds.h5) against a ground-truth .h5 file.

Example:
    python scripts/run_evaluation.py --data data/delayed_memory/test.h5 --preds data/delayed_memory/preds.h5
"""

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evals.evaluate import evaluate

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Ground-truth .h5 (train, val, or test)")
    parser.add_argument("--preds", type=Path, required=True, help="Predicted area activity (area-A0, area-R, ...), inferred inter-regional communication, and/or inferred external inputs")
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional csv path")
    args = parser.parse_args()

    data_h5 = args.data.expanduser().resolve()
    preds_h5 = args.preds.expanduser().resolve()
    if not data_h5.is_file():
        raise SystemExit(f"ground truth file not found: {data_h5}")
    if not preds_h5.is_file():
        raise SystemExit(f"preds file not found: {preds_h5}")

    rows = evaluate(data_h5, preds_h5)
    for name, metric, value in rows:
        print(f"{name:16} {metric:28} {value:.4f}")

    if args.output_csv is not None:
        out = args.output_csv.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "metric", "value"])
            writer.writerows(rows)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
