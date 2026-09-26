# IRCB '26 datasets

The public files are at https://huggingface.co/datasets/ircb/IRCB-26. Each task is released under a gaussian observation model and a poisson observation model. Arrays are stored at the root of the HDF5 file. The leading dimension is trials, then time, then neurons or channels. Every trial is 200 timesteps.

| Split | Trials |
|---|---|
| train | 820 |
| val | 102 |
| test | 102 |

Gaussian `area-*` arrays are continuous hidden-state activity. Poisson `area-*` arrays are spike counts in the same shape. Poisson `test.h5` files also store `rate-*` firing rates in Hz, computed from the matching gaussian activity after z-scoring (`dt = 0.01`, `rate_max = 40`).

## What each split contains

**Train.** Area activity only, plus an `observation` group of neuron masks. Messages, task variables, and communication weights are not included.

**Validation.** Area activity only. No observation masks and no communication targets.

**Test.** Area activity, the ground-truth communication signals, the external inputs or task variables, and `comm_inp_weights` / `comm_inp_source` on each area that receives communication. Poisson test files also include `rate-*`.

## Observation masks

Masks are only on `train.h5`. For each area they are integer indices at `observation/pct_{25,50,75,100}/observed_idx/{area}` and `held_out_idx/{area}`.

At 25%, 50%, and 75%, `observed_idx` is that fraction of the neurons. `held_out_idx` is 10% of the neurons that were not observed, so the two sets do not overlap. At 100%, `held_out_idx` is 10% of all neurons and those neurons are removed from `observed_idx`.

Delayed-Memory and Relay-Decision use seed 42. Each Cognitive Task Suite task has its own seed. Gaussian and poisson copies of a task share that draw, and so do `network_1` and `network_2`.

| Family | Neurons | 25% obs / held-out | 50% | 75% | 100% |
|---|---|---|---|---|---|
| Delayed-Memory | 512 | 128 / 38 | 256 / 26 | 384 / 13 | 461 / 51 |
| Relay-Decision | 256 | 64 / 19 | 128 / 13 | 192 / 6 | 230 / 26 |
| Cognitive Task Suite | 64 | 16 / 5 | 32 / 3 | 48 / 2 | 58 / 6 |

Cognitive Task Suite train files also have leave-one-region-out masks at `observation/regional/drop_A0` and `observation/regional/drop_A1`. The dropped area has an empty `observed_idx`. Its `held_out_idx` is the six neurons from the 100% mask. The other areas keep their 100% masks (58 observed, 6 held out).

## Delayed-Memory

Areas `A0`, `A1`, and `A2`, each with 512 neurons.

Test communication:
- `message-mesgs`, shape `(102, 200, 6)`
- `truth-inp`, shape `(102, 200, 3)`
- `area-A0` receives communication from `A2`, weights `(512, 1)`
- `area-A1` receives communication from `A0`, weights `(512, 1)`
- `area-A2` receives communication from `A1`, weights `(512, 1)`

```text
delayed_memory/{gaussian,poisson}/{train,val,test}.h5
```

## Relay-Decision

Areas `R` (relay) and `D` (decision), each with 256 neurons.

Test communication:
- `message-r-to-d`, shape `(102, 200, 2)`
- `truth-inp`, shape `(102, 200, 2)`
- `area-D` receives communication from `R`, weights `(256, 2)`
- `area-R` has no incoming communication weights

```text
relay_decision/{gaussian,poisson}/{train,val,test}.h5
```

## Cognitive Task Suite

Twenty tasks, two networks each. Areas `A0`, `A1`, `A2`, and `A3`, each with 64 neurons. `network_1` and `network_2` use different communication graphs, so `comm_inp_source` and the leading dimension of `comm_inp_weights` differ. Each incoming block is `(n_sources, 64, 8)`.

Test communication and task variables:
- `message-mesgs`, shape `(102, 200, 100)`
- `truth-fix`, `truth-stim1`, `truth-stim2`, `truth-resp`, `truth-sacc`, `truth-task`, each `(102, 200, 1)`
- `truth-amp1` and `truth-amp2`, each `(102, 1)`

`truth-fix`, `truth-stim1`, and `truth-stim2` are stimulus angles. `truth-amp1` and `truth-amp2` are stimulus strengths. `truth-resp` is the response angle (`0` when no response is required). `truth-sacc` marks fixation versus a required saccade. `truth-task` is the task indicator; single-task files are zero.

```text
cognitive_task_suite/<task>/{network_1,network_2}/{gaussian,poisson}/{train,val,test}.h5
```

Tasks: `ctxt_dm_1`, `ctxt_dm_2`, `ctxt_dm_max`, `dm_1`, `dm_2`, `dly_dm_1`, `dly_dm_2`, `dly_dm_max`, `dly_dm_mod_1`, `dly_dm_mod_2`, `dly_go`, `dly_go_anti`, `dmc`, `dmc_nogo`, `dms`, `dms_nogo`, `fd_go`, `fd_go_anti`, `rt_go`, `rt_go_anti`.
