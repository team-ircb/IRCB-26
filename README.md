# IRCB '26

This repository hosts the data generation and evaluation framework for the **Inter-Regional Communication Benchmark 2026 (IRCB '26)**: a standardized benchmark suite for evaluating inter-regional communication in multi-regional models.

**Public datasets**:
https://huggingface.co/datasets/ircb/IRCB-26

What each task family stores in `train.h5`, `val.h5`, and `test.h5` is listed in the [dataset reference](docs/dataset_reference.md). Train files have area activity and observation masks. Validation files have area activity only. Test files add the communication signals, task variables, and communication weights. Poisson test files also include firing rates.


## Setup

```bash
pip install -r requirements.txt
```

## Data generation

Configs live under `configs/`. Constructor arguments and the released settings are listed in the [parameter reference sheet](docs/model_parameter_reference_sheet.md).

To generate data, pass the name of a folder under `configs/`:

```bash
python scripts/generate_data.py [config_folder]
```

`delayed_memory`, `relay_decision`, and `cognitive_task_suite` are the Delayed-Memory, Relay-Decision, and Cognitive Task Suite configuration folders used to generate the released datasets. Passing one of those reproduces that release. To build your own dataset, add a configuration folder and YAML files under `configs/` and pass that folder name instead. 


## Evaluation

Compare a prediction file to a ground-truth Poisson file from the dataset release. Area arrays are spike counts and `rate-*` are firing rates. Test files also score communication when the prediction file includes the matching message datasets.

```bash
python scripts/run_evaluation.py --data path/to/test.h5 --preds path/to/preds.h5
```

`--output-csv` writes the same rows to a csv. Predicted area activity uses the ground-truth names, such as `area-A0` or `area-R`. Those arrays are compared directly to `rate-*` and to the spike counts.

Communication datasets are `message-mesgs` (Delayed-Memory and the Cognitive Task Suite) or `message-r-to-d` (Relay-Decision). External inputs are `truth-inp` for Delayed-Memory and Relay-Decision. Cognitive Task Suite external inputs are `truth-fix`, `truth-stim1`, and `truth-stim2`. Each is scored when the prediction file contains the same dataset.


### Reported metrics

**Neural activity reconstruction**
- **Rate R².** Coefficient of determination between the predicted neural activity and the ground-truth firing rates.
- **McFadden R².** McFadden pseudo-R² of the observed spike counts under a Poisson distribution specified by the predicted activity.

**Effectome similarity**
- **Norm effectome.** Cosine similarity between the inferred and ground-truth effectomes, whose entries are the L2 norm of communication along each directed pathway.
- **Dynamic effectome.** Cosine similarity between dynamically weighted effectomes, where each communication pathway is weighted by its mapping into the target region's dynamics.

**Single-trial communication recovery**
- **Communication fidelity.** R² from a linear decode of the ground-truth communication from the inferred communication.
- **Input fidelity.** R² from a linear decode of the ground-truth external inputs from the inferred external inputs.

**Single-trial communication specificity**
- **Communication specificity.** R² from a linear decode of the inferred communication from the ground-truth communication.
- **Input specificity.** R² from a linear decode of the inferred external inputs from the ground-truth external inputs.

Communication and input scores are fit by linear regression on 80% of trials and evaluated on the remaining 20%.
