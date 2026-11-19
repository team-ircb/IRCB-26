"""Command-line options shared by scripts/evaluate.py and scripts/evaluate_submission.py."""

from __future__ import annotations

import argparse

from .content_tests import LAGS, STIMULUS_LAG
from .evaluate import EvalOptions
from .reconstruction import DT


def add_options(parser: argparse.ArgumentParser) -> None:
    """Add the evaluation settings (EvalOptions) to a parser."""
    group = parser.add_argument_group("evaluation settings")
    group.add_argument("--mask", default="pct_100",
                       help="observation mask that splits held-in and held-out neurons, e.g. pct_100, pct_50 "
                            "or regional/drop_A0 (default: pct_100)")
    group.add_argument("--dt", type=float, default=DT,
                       help=f"bin width in seconds, to turn rate-* (Hz) into counts per bin (default: {DT})")
    group.add_argument("--message-delay", type=int, default=None,
                       help="bins between a message being sent and received, for predictions with "
                            "timing='received'. Known for the released datasets; needed only for files that don't store it")
    group.add_argument("--decoders", nargs="+", default=["linear"], metavar="NAME",
                       help="decoders for communication fidelity / specificity: linear, kernel-ridge, mlp "
                            "(default: linear; the nonlinear ones are much slower)")
    group.add_argument("--nonlinear", action="store_true",
                       help="shorthand for --decoders linear kernel-ridge mlp")
    group.add_argument("--content-tests", action="store_true",
                       help="also run the Delayed-Memory lag test and the Relay-Decision decision test")
    group.add_argument("--stimulus-lag", type=int, default=STIMULUS_LAG,
                       help=f"Delayed-Memory: the stimulus lag the true message carries (default: {STIMULUS_LAG})")
    group.add_argument("--lags", type=int, nargs="+", default=list(LAGS),
                       help=f"Delayed-Memory: stimulus lags scored by the lag test (default: {LAGS[0]} to {LAGS[-1]})")
    group.add_argument("--true-readouts", default=None,
                       help="ground-truth readouts (.npz) for subspace alignment (default: the bundled file "
                            "matching the dataset)")


def options_from_args(args: argparse.Namespace, **extra) -> EvalOptions:
    """EvalOptions from parsed command-line arguments."""
    decoders = ["linear", "kernel ridge", "mlp"] if args.nonlinear else args.decoders
    return EvalOptions(
        mask=args.mask,
        dt=args.dt,
        message_delay=args.message_delay,
        decoders=tuple(decoders),
        content_tests=args.content_tests,
        stimulus_lag=args.stimulus_lag,
        lags=tuple(args.lags),
        true_readouts=args.true_readouts,
        **extra,
    )
