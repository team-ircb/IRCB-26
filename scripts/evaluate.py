"""Score one prediction file against one ground-truth file.

Example:
    python scripts/evaluate.py --data IRCB-26/delayed_memory/poisson/test.h5 --preds preds.h5

The observation masks for held-in / held-out scores are read from the train.h5 next to --data
(or --train). See docs/evaluation.md for every score.
"""

import argparse
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import EvalOptions, evaluate_file, format_scores, save_scores  # noqa: E402
from src.evaluation.cli import add_options, options_from_args  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="ground-truth file, usually a test.h5")
    parser.add_argument("--preds", type=Path, required=True, help="prediction file (see docs/submission_format.md)")
    parser.add_argument("--train", type=Path, default=None,
                        help="train.h5 holding the observation masks (default: the train.h5 next to --data)")
    parser.add_argument("--output", type=Path, default=None, help="also save the scores to this .csv or .json file")
    add_options(parser)
    args = parser.parse_args(argv)

    options: EvalOptions = options_from_args(args, train_path=args.train)
    if args.train is None and not args.data.expanduser().with_name("train.h5").is_file():
        print("No train.h5 next to --data and no --train given: scoring all neurons together.")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            scores = evaluate_file(args.data, args.preds, options)
        except (FileNotFoundError, KeyError, ValueError) as err:
            print(f"error: {err}", file=sys.stderr)
            return 1
    for message in dict.fromkeys(str(w.message) for w in caught):
        print(f"warning: {message}")

    print(format_scores(scores))
    if args.output is not None:
        rows = [score._asdict() for score in scores]
        print(f"wrote {save_scores(args.output, rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
