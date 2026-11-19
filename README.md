# IRCB '26

This repository hosts the data generation and evaluation framework for the **Inter-Regional Communication Benchmark 2026 (IRCB '26)**: a standardized benchmark suite for evaluating inter-regional communication in multi-regional models.

**Public datasets**:
https://huggingface.co/datasets/ircb/IRCB-26

Each dataset is a recording of a data-generating network (DGN): a multi-region recurrent network trained on a task, whose inter-regional messages are known. A communication model is fit to the activity in `train.h5`, predicts the activity and communication of the trials in `test.h5`, and is scored on how well it recovers them.

What each dataset stores is listed in the [dataset reference](docs/dataset_reference.md).


## Setup

```bash
pip install -r requirements.txt
```

Run every command from the repository root.


## Repository layout

```text
configs/            Hydra configs, one folder per DGN (plus real_data/ for the recordings)
docs/               dataset, evaluation, submission and parameter references
scripts/            command-line entry points
src/models/         the DGNs: Delayed-Memory, Relay-Decision, Cognitive Task Suite
src/datamodules/    the inputs and targets each DGN is trained on
src/callbacks/      training callbacks: stopping rules, saving data, diagnostic plots
src/evaluation/     scoring predictions against the ground truth
src/chen2024/       export of the Chen et al. (2024) recordings
src/utils/          task definitions and shared helpers
```


## Evaluation

Write your model's predictions for a dataset to a `preds.h5` file, then score it against the dataset's `test.h5`:

```bash
python scripts/evaluate.py --data IRCB-26/delayed_memory/poisson/test.h5 --preds preds.h5
```

To score a model on many datasets at once, lay its prediction files out like the dataset release and evaluate the folder:

```bash
python scripts/evaluate_submission.py --data IRCB-26 --submission my_model --check    # format only
python scripts/evaluate_submission.py --data IRCB-26 --submission my_model --output results.csv
```

What a prediction file contains is in [docs/submission_format.md](docs/submission_format.md). In short: the predicted activity of every neuron (`area-<AREA>`, expected spike counts per bin), and, optionally, the communication the model infers on each directed pathway (`pathway-<SRC>-to-<TGT>`) and the external inputs it infers.

### Reported scores

Each score is described in [docs/evaluation.md](docs/evaluation.md).

| Group | Scores | Needs |
| --- | --- | --- |
| Neural activity reconstruction | rate R², McFadden pseudo-R², over all, held-in and held-out neurons | `area-*` |
| Effectome similarity | cosine and Pearson similarity of the norm effectome, and of the dynamic effectome | `pathway-*` (dynamic: their `comm_inp_weights`) |
| Communication recovery | communication fidelity and specificity per pathway, input fidelity and specificity | `pathway-*` or `message-*`, `truth-*` inputs |

Optional settings:

- `--decoders linear kernel-ridge mlp` (or `--nonlinear`): also score communication fidelity and specificity with nonlinear decoders.
- `--content-tests`: the Delayed-Memory lag test and the Relay-Decision decision test.
- Subspace alignment of linear readouts, when the prediction gives each pathway's `source_readout`.
- `--mask` picks the observation mask that defines held-in and held-out neurons (default `pct_100`).


## Data generation

Configs live under `configs/`. Constructor arguments and the released settings are listed in the [parameter reference sheet](docs/model_parameter_reference_sheet.md).

To generate data, pass the name of a folder under `configs/`:

```bash
python scripts/generate_data.py [config_folder]
```

`delayed_memory`, `relay_decision`, and `cognitive_task_suite` are the Delayed-Memory, Relay-Decision, and Cognitive Task Suite configuration folders used to generate the released datasets. Passing one of those reproduces that release. To build your own dataset, add a configuration folder and YAML files under `configs/` and pass that folder name instead. Any setting can be changed from the command line with `--override key=value`. Runs are written to `runs/<run_name>/`.

### Multitask and single-task networks on Yang et al. (2019)'s tasks

`multitask_dgn` trains one network on all 20 of Yang et al. (2019)'s tasks. Every design choice is a commented default in `configs/multitask_dgn/`, and the defaults reproduce the released multitask network. Training has two stages: from scratch at learning rate 1e-3, then resumed at 3e-4 until every task passes.

```bash
# 1. From scratch. Stop once every task is >= 0.8 in its 10-epoch validation means.
python scripts/generate_data.py multitask_dgn --run_name stage1

# 2. Resume at a lower learning rate (a resume restores the optimizer's rate, so set it in a copy).
#    Training stops by itself once the worst task is >= 0.8 for 20 validation epochs in a row.
python scripts/set_ckpt_lr.py --ckpt runs/stage1/lightning_checkpoints/<epoch>-<step>.ckpt --lr 3e-4 --out runs/stage1_lr3e-4.ckpt
python scripts/generate_data.py multitask_dgn --run_name stage2 --override +stage=finetune --ckpt_path runs/stage1_lr3e-4.ckpt
```

The same config trains a single-task network with the same design, without the task rule:

```bash
python scripts/generate_data.py multitask_dgn --override +mode=single_task --override "datamodule.task_names=[contextdm1]"
```

From a trained checkpoint (either kind, or a `cognitive_task_suite` network), write a dataset in the layout of the release and check it against the network:

```bash
python scripts/generate_multitask_datasets.py --ckpt runs/stage2/lightning_checkpoints/best_worst_task.ckpt --out <dir>
python scripts/check_multitask_dataset.py --ckpt <same ckpt> --data <dir>
python scripts/split_multitask_by_task.py --data <dir>     # optional: one single-task dataset per task
```

`generate_multitask_datasets.py` also writes `acceptance.json`: whether the network reaches Yang-style performance >= 0.80 on every task over the dataset's trials, which are drawn after training and never seen by it.


## Real data

`src/chen2024/` exports the public Neuropixels recordings of Chen et al. (2024) (DANDI 000363, version 0.230808.2053) as spike counts per area, in the same format as the synthetic datasets, so the benchmarked models can be fit to real neural data:

```bash
python scripts/chen2024/download.py --subject 440959 --download --output raw
python scripts/chen2024/preprocess.py --config configs/real_data/chen2024/example_440959.yaml --input raw --output processed_440959
```

Session files are large, so `preprocess.py` can also stream only the tables it needs from DANDI (`--dandi-subject 440959 --dandi-session 20190225T142613`). See [docs/chen2024.md](docs/chen2024.md) for the steps, the output format and the [configuration reference](docs/chen2024_config_reference.md). Real recordings have no ground-truth communication, so they are scored on reconstruction only.
