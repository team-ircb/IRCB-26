# Evaluation

How a model's predictions are scored. The prediction file format is in [submission_format.md](submission_format.md).

```bash
# one dataset
python scripts/evaluate.py --data IRCB-26/delayed_memory/poisson/test.h5 --preds preds.h5

# a whole submission
python scripts/evaluate_submission.py --data IRCB-26 --submission my_model --output results.csv
```

The same from Python:

```python
from src.evaluation import EvalOptions, evaluate_file, format_scores

scores = evaluate_file("IRCB-26/relay_decision/poisson/test.h5", "preds.h5",
                       EvalOptions(decoders=("linear", "mlp"), content_tests=True))
print(format_scores(scores))
```

Every score is a row `(name, metric, value)`. `name` is an area (`area-A0`), a pathway (`A0->A1`), an input, or an average (`all` for reconstruction, `communication` for communication scores).

## Neural activity reconstruction

Always scored, from the `area-*` predictions.

- **Rate R²** (`rate r2`). R² between the predicted activity and the true firing rates, converted to counts per bin (`rate-* × dt`, with `dt = 0.01` s). It is computed for each neuron over trials and time bins, against that neuron's own mean, then averaged over the neurons of an area and over areas. Neurons whose true rate doesn't vary are left out.
- **McFadden pseudo-R²** (`mcfadden r2`). 1 − log L(model) / log L(null) for the observed spike counts under a Poisson distribution with the predicted means. The null model predicts each neuron's mean count over trials in each time bin. Computed per area, then averaged over areas. It depends on the Poisson assumption, so treat it as a supplementary score.
- Both are reported over all neurons and separately over the **held-in** and **held-out** neurons of an observation mask (`… held-in`, `… held-out`). The masks are read from the `train.h5` next to the ground truth (or `--train`); `--mask` picks one: `pct_100` (default), `pct_25`, `pct_50`, `pct_75`, or, for the Cognitive Task Suite, `regional/drop_A0` and `regional/drop_A1`. Without a `train.h5`, only the all-neuron scores are reported.
- Files without `rate-*` (real recordings) get McFadden R² only. Continuous (Gaussian) files get `activity r2`, the per-neuron R² against the `area-*` arrays themselves.

## Effectome similarity

Scored from `pathway-*` predictions. The effectome M is an areas × areas matrix whose entry (i, j) is the strength of communication from area j to area i: the L2 norm of the message over every trial, time bin and channel. The ground-truth effectome is 0 on pairs the network doesn't connect, so communication a model infers there lowers its score.

- **Norm effectome.** Cosine similarity between the inferred and ground-truth effectomes.
- **Norm effectome pearson.** Pearson correlation of their off-diagonal entries. Effectome entries are nonnegative, so the cosine has a floor: a model that puts the same strength on every pair already scores well. The Pearson correlation removes it (a uniform effectome scores 0 or `nan`). With two areas (Relay-Decision) there are only two entries, so it can only be +1, −1 or `nan`, and it says only whether the stronger direction is the true one.
- **Dynamic effectome** (and its Pearson version). The same, with each message first mapped into its target area's dynamics, ‖W m‖: the network's own input weights for the truth, and the model's `comm_inp_weights` attribute for the prediction. Reported only when every predicted pathway has weights.
- A model that reports no communication on any pathway scores `nan`.

## Communication fidelity and specificity

Scored from `pathway-*` or `message-*` predictions, for each ground-truth pathway j → i:

- **Communication fidelity.** R² of decoding the true message on the pathway from the model's message on the same pathway: does the model recover what was communicated?
- **Communication specificity.** R² of decoding the model's message from the true message: does the model's message carry anything the true one doesn't?

Each decoder is fit on 80% of the trials and scored on the other 20% (the same split for every decoder). R² is computed for each output channel against that channel's own mean, averaged over channels, and the dataset's score is the average over its true pathways. Each pathway is also reported on its own row. A pathway that a prediction leaves silent scores fidelity about 0 and specificity `nan`.

**Decoders.** The benchmark score uses a linear decoder (least squares). A linear decoder can miss content that an inferred message carries in a nonlinearly transformed form, so two nonlinear decoders are available too, reported as `communication fidelity (kernel ridge)`, `communication fidelity (mlp)` and so on:

- `--decoders linear kernel-ridge mlp`, or `--nonlinear` for all three.
- **Kernel ridge:** a linear kernel plus a scaled RBF kernel (Nyström approximation with 1,000 landmarks), with the RBF scale and width and the ridge penalty chosen by 3-fold cross-validation over the training trials.
- **MLP:** a linear skip connection plus two GELU hidden layers (256 units for fidelity, 64 for specificity), trained with AdamW and early stopping on 20% of the training trials. The skip connection starts at the least-squares solution, so training starts from the linear decoder.
- Both contain the linear decoder as a special case. They are much slower (minutes per pathway on the larger datasets). Every setting is listed at the top of `src/evaluation/decoders.py`.

**Input fidelity and specificity.** The same two scores for each external input a model infers (`truth-inp` for Delayed-Memory and Relay-Decision; `truth-fix`, `truth-stim1` and `truth-stim2` for the Cognitive Task Suite), each on its own, with the linear decoder.

## Content tests (optional)

`--content-tests` adds two checks for the networks whose message content is defined by the task.

**Delayed-Memory lag test.** Each true message is its source's stimulus two bins back, m_t = s_{t−2}. Because stimulus history is spread across the network, a model may pick up other lags. For each stimulus lag l (−3 to 6 by default, `--lags`), we decode s_{t−l} from the inferred message at bin t (`lagged content fidelity`, rows `lag l`, averaged over pathways). Lag 2 is the communicated content; other lags should be lower, and future lags (l < 0) act as a null. All lags use the same bins, so they are comparable. `lagged content specificity` (row `all lags`) is how much of the inferred message the whole stimulus history explains together, to compare with the specificity at lag 2 alone. `--stimulus-lag` changes the true lag for networks built with another `lag`.

**Relay-Decision decision test.** The relay area passes the stimulus s to the decision area, which integrates it into the decision variable d_t = s_0 + … + s_t. A model that puts the integration in the relay area infers a message that tracks d instead of s. Reported on the pathway: `stimulus fidelity` (decode s from the inferred message), `decision variable fidelity` (decode d), and the reverse directions (`… specificity`).

## Subspace alignment (optional)

For models whose message on a pathway is a linear readout of the source area's neurons, m = C x. Give the readout as the attribute `source_readout` on the `pathway-*` dataset, shaped (channels, source neurons), over every neuron of the source area in dataset order (columns of held-out neurons are ignored). The readout should act on firing rates or spike counts.

For each true pathway we compare, in the space of the source area's observed neurons, the ground-truth communication subspace (spanned by the network's own message readout in firing-rate coordinates) with the inferred one (spanned by C, cut to its top d directions, where d is the true message width). The score is the share of the true subspace the inferred one captures, the mean squared cosine of the principal angles (1 = identical):

- `subspace alignment`: over all observed neurons;
- `subspace alignment (95% variance)`: within the top principal components of the source area's true rates that hold 95% of their variance;
- each with `… chance`, the share a random subspace of the same size captures.

The ground-truth readouts of the released Delayed-Memory, Relay-Decision and multitask datasets are bundled in `src/evaluation/data/true_readouts/` and found automatically (each file stores a fingerprint of its dataset's communication weights). For other datasets, pass a file with `--true-readouts`: an `.npz` with one array `readout/<SRC>-><TGT>` (channels × source neurons) per pathway and a JSON `meta` entry.

## Message timing

A ground-truth message at bin t is the one its source area emits at t. A prediction whose message at bin t is what the target receives at t sets `timing="received"` (see [submission_format.md](submission_format.md)); it is then compared with the true message emitted `delay` bins earlier, and the first `delay` predicted bins and last `delay` true bins are left out. The delay is read from the dataset: a `delivery_delay` or `lag_steps` attribute on the true message, or the released value for the task family (0 Relay-Decision, 1 Delayed-Memory, 6 single-task Cognitive Task Suite). `--message-delay` sets it for files that don't store it.

## Checks on the input

Before scoring, the evaluator checks that the prediction lines up with the ground truth and stops with a list of every problem: missing or extra areas, wrong shapes, mismatched trials or time bins, non-numeric arrays, `nan` or `inf`, and negative predictions for spike counts. Datasets it doesn't recognise are ignored with a warning, so a misspelled name is easy to spot.
