# Prediction files and submissions

A model is evaluated from one prediction file per dataset, `preds.h5`, scored against that dataset's `test.h5`. This page lists what a prediction file may contain. What each score means is in [evaluation.md](evaluation.md).

## A prediction file

An HDF5 file with one dataset per array at the root. Every array is `(trials, time, channels)`, with the trials in the same order and the same number of time bins as the ground truth. Only the `area-*` arrays are required; everything else is scored when it is present.

| Dataset | Shape | Scored as |
| --- | --- | --- |
| `area-<AREA>` (required) | as the ground truth's `area-<AREA>` | neural activity reconstruction |
| `pathway-<SRC>-to-<TGT>` | `(trials, time, any)` | effectome similarity, communication fidelity and specificity, content tests |
| `message-<NAME>` | as the ground truth's `message-<NAME>` | communication fidelity and specificity |
| `truth-inp`, `truth-fix`, `truth-stim1`, `truth-stim2` | `(trials, time, any)` | input fidelity and specificity |

Datasets with other names are ignored, with a warning.

### Predicted activity

- Give one `area-<AREA>` array for every area of the ground truth, using its names (`area-A0`, `area-R`, ...).
- Predict **every neuron**, held-out ones included, so the array has the ground truth's shape. Held-in and held-out neurons are told apart by the observation masks in `train.h5`.
- For spike-count (Poisson) data, predict the **expected spike count per bin** (the Poisson mean), not a rate in Hz. Values must be finite and nonnegative.
- For continuous (Gaussian) data, predict the activity itself.

### Inferred communication

- **Per pathway (recommended).** Give each directed pathway your model infers its own dataset, `pathway-<SRC>-to-<TGT>` (for example `pathway-R-to-D` or `pathway-A0-to-A2`), with any number of channels. Include every pathway the model reports, not only the ones you expect to be true: the effectome is computed over all ordered pairs of areas, and communication on pairs the network doesn't have lowers the score.
- **As one message.** Alternatively give `message-<NAME>` with the same name and width as the ground truth's message, each pathway in its channels. This is scored for fidelity and specificity but not for the effectome, since it doesn't say which pathways the model thinks exist. A message of a different width can only be scored when the network has a single pathway (Relay-Decision).
- A pathway that a prediction with `pathway-*` datasets leaves out is taken to be silent.

### Optional attributes

Set these as HDF5 attributes on the datasets above.

| Attribute | On | Value | Used for |
| --- | --- | --- | --- |
| `timing` | `pathway-*`, `message-*` | `"emitted"` (default) or `"received"` | pairing messages in time (below) |
| `comm_inp_weights` | `pathway-*` | `(hidden, channels)`: the model's map from this message into the target area's dynamics | dynamic effectome |
| `source_readout` | `pathway-*` | `(channels, source neurons)`: the model's linear readout of the source area's neurons, every neuron in dataset order | subspace alignment |

**Message timing.** A ground-truth message at bin t is the one the source area *emits* at t; the target receives it a few bins later (0 bins on Relay-Decision, 1 on Delayed-Memory, 6 on the single-task Cognitive Task Suite, 3 on the multitask network). By default a predicted message is read the same way. If your model's message at bin t is what the target *receives* at bin t (for example, a model that predicts each area from the others' recent activity), set `timing="received"`; it is then compared with the true message emitted `delay` bins earlier. The delay comes from the dataset, never from the model. Every `pathway-*` dataset of a file must use the same timing.

### Writing a file

`save_predictions` writes a file in this format and checks the arrays first:

```python
from src.evaluation import save_predictions

save_predictions(
    "my_model/delayed_memory/poisson/preds.h5",
    activity={"A0": rates_a0, "A1": rates_a1, "A2": rates_a2},       # expected counts per bin
    pathways={("A0", "A1"): msg_01, ("A1", "A2"): msg_12, ("A2", "A0"): msg_20},
    timing="emitted",
)
```

## A submission

A submission is a folder of prediction files laid out like the dataset release:

```text
my_model/
  delayed_memory/poisson/preds.h5
  relay_decision/poisson/preds.h5
  cognitive_task_suite/dm_1/network_1/poisson/preds.h5
  cognitive_task_suite/dm_1/network_2/poisson/preds.h5
  ...
```

Each `my_model/<folder>/preds.h5` is scored against `<data root>/<folder>/test.h5`. A submission may cover any subset of the datasets. Check the files first, then score them:

```bash
python scripts/evaluate_submission.py --data IRCB-26 --submission my_model --check
python scripts/evaluate_submission.py --data IRCB-26 --submission my_model --output my_model_results.csv
```

The second command writes every score to `my_model_results.csv` and the headline scores, with means over each task family, to `my_model_results_summary.csv`.
