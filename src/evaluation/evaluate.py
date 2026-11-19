"""Score one prediction file against one ground-truth file."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

from .communication import communication_scores, input_scores
from .content_tests import LAGS, STIMULUS_LAG, decision_test_scores, lag_test_scores
from .decoders import normalize_decoder_name
from .effectome import effectome_scores
from .io import check_predictions, load_h5, load_observation_masks
from .layouts import dataset_family
from .reconstruction import DT, reconstruction_scores
from .scores import Score
from .subspace import subspace_alignment_scores


@dataclass
class EvalOptions:
    """Settings for an evaluation. The defaults give the benchmark scores."""

    # train.h5 holding the observation masks. None: use the train.h5 next to the ground truth, if any.
    train_path: Path | str | None = None
    # Which observation mask splits neurons into held-in and held-out (a group under observation/).
    mask: str = "pct_100"
    # Bin width in seconds, to turn rate-* (Hz) into counts per bin.
    dt: float = DT
    # Override the dataset's message delay (only used by predictions with timing="received").
    message_delay: int | None = None
    # Decoders for communication fidelity / specificity: "linear", "kernel ridge", "mlp".
    decoders: tuple[str, ...] = ("linear",)
    # Run the Delayed-Memory lag test and the Relay-Decision decision test where they apply.
    content_tests: bool = False
    stimulus_lag: int = STIMULUS_LAG
    lags: tuple[int, ...] = LAGS
    # Ground-truth readouts for subspace alignment. None: the bundled file that matches the dataset.
    true_readouts: Path | str | None = None
    # Extra labels copied into every result row of a submission (not used for scoring).
    tags: dict = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.decoders, str):
            self.decoders = (self.decoders,)
        self.decoders = tuple(dict.fromkeys(normalize_decoder_name(d) for d in self.decoders))
        if "linear" not in self.decoders:
            self.decoders = ("linear",) + self.decoders
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")


def find_train_file(truth_path: Path, train_path: Path | str | None) -> Path | None:
    """The train.h5 with the observation masks: `train_path`, or the one next to the ground truth."""
    if train_path is not None:
        path = Path(train_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"no such train file: {path}")
        return path
    sibling = truth_path.with_name("train.h5")
    return sibling if sibling.is_file() and sibling != truth_path else None


def evaluate_file(truth_path: Path | str, preds_path: Path | str, options: EvalOptions | None = None) -> list[Score]:
    """Score a prediction file against a ground-truth file (usually a test.h5).

    Always scored: neural activity reconstruction. Scored when the prediction has what they need:
    effectome similarity (pathway-* datasets), communication fidelity and specificity (message-*
    or pathway-*), input fidelity and specificity (truth-* inputs), and subspace alignment
    (pathway-* with a source_readout attribute). Optional: the content tests.
    """
    options = options or EvalOptions()
    truth_path = Path(truth_path).expanduser()
    preds_path = Path(preds_path).expanduser()

    truth, attrs = load_h5(truth_path)
    preds, pred_attrs = load_h5(preds_path)
    for message in check_predictions(truth, preds):
        warnings.warn(f"{preds_path.name}: {message}", stacklevel=2)

    train_file = find_train_file(truth_path, options.train_path)
    masks = load_observation_masks(train_file, options.mask) if train_file is not None else None

    scores = reconstruction_scores(truth, preds, masks=masks, dt=options.dt)
    scores += effectome_scores(truth, preds, attrs, pred_attrs, message_delay=options.message_delay)
    scores += communication_scores(
        truth, preds, attrs, pred_attrs, message_delay=options.message_delay, decoders=options.decoders
    )
    scores += input_scores(truth, preds)
    scores += subspace_alignment_scores(
        truth, preds, attrs, pred_attrs, masks=masks, true_readouts=options.true_readouts, dt=options.dt
    )

    if options.content_tests:
        family = dataset_family(truth, attrs)
        if family == "delayed_memory":
            scores += lag_test_scores(
                truth, preds, attrs, pred_attrs, options.message_delay, options.stimulus_lag, options.lags
            )
        elif family == "relay_decision":
            scores += decision_test_scores(truth, preds, attrs, pred_attrs, options.message_delay)

    return scores
