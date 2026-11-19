"""IRCB '26 evaluation: score a model's predictions against the ground truth of a dataset.

    from src.evaluation import evaluate_file, EvalOptions
    scores = evaluate_file("data/delayed_memory/poisson/test.h5", "preds.h5")

See docs/evaluation.md for every score and docs/submission_format.md for the prediction files.
"""

from .evaluate import EvalOptions, evaluate_file
from .predictions import save_predictions
from .scores import Score, format_scores, save_scores
from .submission import check_submission, evaluate_submission, summarize

__all__ = [
    "EvalOptions",
    "Score",
    "check_submission",
    "evaluate_file",
    "evaluate_submission",
    "format_scores",
    "save_predictions",
    "save_scores",
    "summarize",
]
