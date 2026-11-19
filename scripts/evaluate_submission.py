"""Score a submission: a folder of preds.h5 files laid out like the dataset release.

Example:
    python scripts/evaluate_submission.py --data IRCB-26 --submission my_model --output results.csv

Each <submission>/<dataset folder>/preds.h5 is scored against <data>/<dataset folder>/test.h5.
The full scores go to --output; a summary of the headline scores, with means over each task
family, is printed and saved next to it as <output>_summary.csv. Use --check to validate the
files' format without scoring them.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import check_submission, evaluate_submission, save_scores, summarize  # noqa: E402
from src.evaluation.cli import add_options, options_from_args  # noqa: E402
from src.evaluation.submission import format_summary  # noqa: E402


def run_check(args) -> int:
    report = check_submission(args.data, args.submission, split=args.split, include=args.include)
    n_bad = 0
    for dataset, problems in report.items():
        errors = [p for p in problems if not p.startswith("warning:")]
        n_bad += bool(errors)
        print(f"{'FAIL' if errors else 'ok  '}  {dataset}")
        for problem in problems:
            print(f"        {problem}")
    print(f"\n{len(report) - n_bad} of {len(report)} prediction files are ready to score.")
    return 1 if n_bad else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="dataset root (the IRCB-26 folder)")
    parser.add_argument("--submission", type=Path, required=True, help="submission root holding the preds.h5 files")
    parser.add_argument("--output", type=Path, default=Path("results.csv"),
                        help="file for every score, .csv or .json (default: results.csv)")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="ground-truth split to score against (default: test)")
    parser.add_argument("--include", nargs="+", default=None, metavar="PATTERN",
                        help="only score dataset folders matching these patterns, e.g. 'cognitive_task_suite/*'")
    parser.add_argument("--keep-going", action="store_true", help="report files that fail and score the rest")
    parser.add_argument("--check", action="store_true", help="only check the files' format; don't score")
    parser.add_argument("--model-name", default=None, help="label added to every result row")
    add_options(parser)
    args = parser.parse_args(argv)

    try:
        if args.check:
            return run_check(args)

        tags = {"model": args.model_name} if args.model_name else {}
        options = options_from_args(args, tags=tags)
        rows = evaluate_submission(args.data, args.submission, options, split=args.split,
                                   include=args.include, keep_going=args.keep_going)
    except (FileNotFoundError, NotADirectoryError, RuntimeError, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    output = save_scores(args.output, rows)
    summary = summarize(rows)
    summary_path = output.with_name(f"{output.stem}_summary{output.suffix}")
    if summary:
        save_scores(summary_path, summary)

    print()
    print(format_summary(summary))
    print(f"\nwrote {output}" + (f" and {summary_path}" if summary else ""))

    failed = sorted({row["dataset"] for row in rows if row["name"] == "error"})
    if failed:
        print(f"\n{len(failed)} file(s) failed: " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
