# Parameter Reference

Constructor parameters for each model and data module. Released dataset values are the settings in `configs/` used for the public IRCB '26 files. The three single-task families use `rnn_type: rnncell`, `max_epochs: 1000`, and `min_epochs: 300`. The multitask network's settings are at the end.

---

### Delayed-Memory model (`models.DelayedMemory`)

Required:
- `ranks` (list): private input dimensionality per area
- `connectome` (list[list]): binary connectivity, rows are targets and columns are sources

Optional:
- `lag` (int, default `1`): communication lag in timesteps
- `memory` (int, default `5`): number of past steps encoded in each area
- `noise` (float, default `0.0`): hidden-state noise
- `channel_noise` (float, default `0.0`): communication-channel noise
- `noise_type` (str, default `"fixed"`): `fixed` or `variable`
- `channel_noise_type` (str, default `"fixed"`): `fixed` or `variable`
- `hidden_size` (int, default `64`): recurrent units per area
- `lr` (float, default `4e-3`): learning rate
- `input_weight_init_var_scale` (float, default `1.0`): scale on the initial input-weight variance
- `rnn_type` (str, default `"grucell"`): `grucell` or `rnncell`
- `ext_input_dim` (int or list, default `0`): external perturbation dimension per area
- `ext_input_amp` (int, default `-1`): perturbation amplitude
- `ext_input_perc` (float, default `0.0`): fraction of trials that receive a perturbation

Released (`configs/delayed_memory`):
- `ranks: [1, 1, 1]`
- `connectome: [[1, 0, 1], [1, 1, 0], [0, 1, 1]]`
- `hidden_size: 512`
- `memory: 2`
- `lag: 2`
- `noise: 0.5`
- `channel_noise: 0.01`
- `rnn_type: rnncell`
- `input_weight_init_var_scale: 1`

### Delayed-Memory data module (`datamodules.NoisySources`)

Required:
- `batch_total` (int): number of trials
- `time_total` (int): timesteps per trial
- `input_dim` (list or int): signal dimension per area; a list is summed

Optional:
- `p_split` (list, default `[0.8, 0.2]`): train/validation split
- `batch_size` (int, default `64`): training batch size
- `mesg_type` (str, default `"white noise"`): `white noise`, `intg noise`, `filter noise`, or `sine wave`
- `mesg_kwargs` (dict, default `{}`): arguments for the chosen signal type
- `sig_smooth` (float or `None`, default `None`): Gaussian smoothing width
- `resultpath` (str, default `"."`): directory for saved outputs

Released (`configs/delayed_memory`):
- `batch_total: 2048`
- `time_total: 200`
- `input_dim` follows `model.ranks`
- `p_split: [0.5, 0.5]`

---

### Relay-Decision model (`models.RelayDecision`)

Optional:
- `lag` (int, default `0`): communication lag in timesteps
- `noise_p` (float, default `0.0`): noise in the relay area
- `noise_d` (float, default `0.0`): noise in the decision area
- `hidden_size` (int, default `64`): recurrent units per area
- `lr` (float, default `4e-3`): learning rate
- `input_weight_init_var_scale` (float, default `1.0`): scale on the initial input-weight variance
- `p_to_d_coef` (float, default `1.0`): weight on the relay-to-decision reconstruction loss
- `rep_coef` (float, default `0.0`): weight on the representation loss
- `binary_output` (bool, default `True`): binary cross-entropy versus mean squared error
- `rnn_nonlinearity` (str, default `"tanh"`): nonlinearity used by `rnncell`
- `rnn_type` (str, default `"grucell"`): `grucell` or `rnncell`

Released (`configs/relay_decision`):
- `hidden_size: 256`
- `noise_p: 0.001`
- `noise_d: 0.001`
- `rnn_type: rnncell`
- `input_weight_init_var_scale: 1.0`

### Relay-Decision data module (`datamodules.LatentDecision`)

Required:
- `batch_total` (int): number of trials
- `time_total` (int): timesteps per trial (`>= 200`)

Optional:
- `decay_factor` (float, default `1.0`): decay in the latent filter
- `latent_factor` (float, default `1.0`): latent amplitude
- `p_split` (list, default `[0.8, 0.2]`): train/validation split
- `batch_size` (int, default `64`): training batch size
- `lag` (int, default `0`): lag used when generating trajectories
- `mesg_dist` (str, default `"normal"`): `normal`, `exponential`, `uniform`, or `sine wave`
- `binary_decision` (bool, default `False`): whether decision trajectories are binarized
- `sig_smooth` (float or `None`, default `None`): Gaussian smoothing width
- `resultpath` (str, default `"."`): directory for saved outputs

Released (`configs/relay_decision`):
- `batch_total: 2048`
- `time_total: 200`
- `p_split: [0.5, 0.5]`
- `mesg_dist: exponential`
- `sig_smooth: 1`

---

### Cognitive Task Suite model (`models.CognitiveTaskSuite`)

Required:
- `num_areas` (int): number of recurrent areas
- `task_names` (list): tasks trained together

Optional:
- `diagram` (list or `None`, default `None`): explicit edges; if unset, the graph is sampled from `graph_kwargs`
- `stim_input_areas` (list or `None`, default `None`): areas for fixation, stimulus 1, stimulus 2, and task
- `sacc_output_areas` (list or `None`, default `None`): areas that contribute to the saccade output
- `sacc_scale` (float, default `1.0`): weight on the fixation and saccade loss
- `graph_kwargs` (dict, default `{}`): random-graph settings (`perc_conns`, `shortest`, optional `seed`)
- `delay` (int, default `0`): communication delay in timesteps
- `hidden_size` (int, default `32`): recurrent units per area
- `lr_init` (float, default `4e-3`): learning rate
- `rnn_type` (str, default `"grucell"`): `grucell` or `rnncell`
- `num_angles` (int, default `36`): direction-readout bins
- `num_channels` (int, default `4`): channels on each communication edge
- `noise` (float, default `0.0`): hidden-state noise, applied after the recurrent update
- `noise_type` (str, default `"fixed"`): `fixed` or `variable`
- `noise_start` (int, default `0`): epoch when the noise ramp begins
- `noise_increase` (int, default `0`): length of the noise ramp
- `noise_init` (float, default `0.0`): starting scale of the noise ramp
- `channel_noise` (float, default `0.0`): communication noise
- `channel_noise_type` (str, default `"fixed"`): `fixed` or `variable`
- `angle_scale` (float, default `0.0`): final scale of the direction loss
- `angle_start_epoch` (int, default `50`): epoch when the direction-loss ramp begins
- `angle_increase_epoch` (int, default `100`): length of that ramp
- `l2_scale` (float, default `0.0`): L2 on non-communication parameters
- `l2_start` (int, default `0`): epoch when the L2 ramp begins
- `l2_increase` (int, default `0`): length of the L2 ramp
- `l2_init` (float, default `0.0`): starting scale of the L2 ramp
- `l2_comm_scale` (float, default `0.0`): L2 on communication weights
- `l2_comm_start` (int, default `0`): epoch when the communication L2 ramp begins
- `l2_comm_increase` (int, default `0`): length of that ramp
- `l2_comm_init` (float, default `0.0`): starting scale of the communication L2 ramp
- `l1_scale` (float, default `0.0`): L1 on communication weights
- `l1_start_epoch` (int, default `150`): epoch when the L1 ramp begins
- `l1_increase_epoch` (int, default `150`): length of the L1 ramp
- `smooth_scale` (float, default `0.0`): smoothness penalty
- `smooth_start_epoch` (int, default `100`): epoch when the smoothness ramp begins
- `smooth_increase_epoch` (int, default `200`): length of that ramp

Options added for the multitask network (each defaults to the single-task behaviour):
- `rule_input_areas` (list, `"all"` or `None`, default `None`): areas that receive a one-hot task rule (`num_tasks` channels)
- `rule_noise` (float, default `0.0`): SD of the per-step noise on the rule channels
- `readout_type` (str, default `"binned"`): `binned` (36 direction logits passed through the message buffer) or `yang` (a fixation unit plus an `n_ring`-unit response ring read from `readout_areas` at the same step, as in Yang et al. 2019)
- `n_ring` (int, default `32`): response-ring units for the `yang` readout
- `readout_areas` (list, default: the last area): areas the response is read from
- `role_sampling` (dict or `None`, default `None`): send each input stream to random areas and read the output from random areas (keys `seed`, `n_input_areas`, `n_readout_areas`)
- `message_source` (str, default `"pre_noise"`): compute messages from the hidden state before (`pre_noise`) or after (`post_noise`) the hidden noise
- `stim_channels` (int, default `4`): inputs per stimulus modality (32 for the data module's `ring` encoding)
- `pre_on_steps` (int, default `0`): first steps of each trial with no loss
- `grace_steps` (int or `"auto"`, default `0`): steps after the go cue with no loss; `auto` is the steps a signal needs to reach the readout plus 5
- `tail` (str, default `"hold"`): `hold` trains the response until the last step; `mask` stops the loss at the trial's end (trials are padded)
- `rule_aux_scale` (float, default `0.0`) and `rule_aux_skip` (int, default `10`): weight and first step of a rule-reconstruction loss (a training-only linear decoder per area)
- `comm_act_scale`, `comm_act_start`, `comm_act_increase`, `comm_act_init` (default `0`): group penalty on message activity and its ramp
- `l2_groups` (str, default `"split"`): `split` penalizes communication weights separately (`l2_comm_*`); `uniform` penalizes every weight matrix alike
- `lr_schedule` (str, default `"plateau"`): `plateau` or `constant`

If `diagram` is unset, `graph_kwargs` needs `perc_conns` and `shortest`.

Released (`configs/cognitive_task_suite`):
- `num_areas: 4`
- `hidden_size: 64`
- `num_channels: 8`
- `delay: 5`
- `rnn_type: rnncell`
- `lr_init: 5e-4`
- `noise: 0.1`, ramped from `noise_init: 0.3` over `noise_increase: 200` epochs
- `l2_scale: 1e-2` and `l2_comm_scale: 1e-4`, with the same 200-epoch ramp from `0.3`
- `angle_scale: 1`
- `sacc_scale: 0.1`
- `sacc_output_areas: ['3']`
- `stim_input_areas: [A0, A1, A1, A2]`
- `graph_kwargs: {perc_conns: 0.70, shortest: 2}`

### Cognitive Task Suite data module (`datamodules.CognitiveTaskSuite`)

Required:
- `task_names` (list): names from the task map
- `batch_total` (int): number of trials
- `time_total` (int): timesteps per trial

Optional:
- `p_split` (list, default `[0.8, 0.2]`): train/validation split
- `batch_size` (int, default `64`): training batch size
- `train_type` (str, default `"random"`): `random`, `batch_uniform`, or `curriculum_ratio`
- `train_type_kwargs` (dict, default `{}`): arguments for the chosen schedule
- `dm_seed` (int, default `0`): data-module seed
- `noise_sig` (float, default `0.0`): noise added to the generated task signals
- `resultpath` (str, default `"."`): directory for saved outputs

Released (`configs/cognitive_task_suite`):
- `batch_total: 2048`
- `time_total: 200`
- `batch_size: 32`
- `p_split: [0.5, 0.5]`
- `noise_sig: 0.01`
- `task_names: [rt_go]` in the checked-in config; replace that list to generate other tasks from the suite

Options added for the multitask network:
- `task_variant` (str, default `"suite"`): `suite` (`src/utils/cognitive_tasks.py`) or `yang` (Yang et al. 2019's 20 tasks at 20 ms per step, `src/utils/yang_tasks.py`)
- `encoding` (str, default `"raw"`): `raw` (angle and strength), `cartesian` (s cos θ, s sin θ) or `ring` (a 32-unit ring per modality)
- `input_noise` (float, default `0.0`) and `fix_noise` (float or `None`): per-step Gaussian noise on the stimulus and fixation inputs
- `resample_train` (bool, default `False`): draw fresh training trials every epoch (needs `trainer.reload_dataloaders_every_n_epochs: 1`)
- `train_type` also accepts `weighted` (fixed task frequencies, `train_type_kwargs.weights`, optional `cond_weights`) and `adaptive` (shares set from validation performance)
- `num_workers` (int, default `0`): DataLoader worker processes

---

### Multitask DGN (`configs/multitask_dgn`)

The defaults reproduce the released multitask network. Every setting is commented in the config files.

- 4 areas of 64 GRU units (`rnn_type: grucell`), 8 channels per pathway, `delay: 2` (a message sent at step t is used at step t + 3)
- graph: `perc_conns: 0.70`, `shortest: 2`, `seed: 2`
- inputs: fixation into A0, both stimulus modalities into A1, the one-hot rule into A2 (`rule_noise: 0.25`); `yang` readout from A3
- loss mask: `pre_on_steps: 5`, `grace_steps: auto`, `tail: mask`; `rule_aux_scale: 0.01`
- noise `0.1` and L2 `1e-3` (`l2_groups: uniform`), both ramped from 0.3 over 200 epochs; `message_source: post_noise`
- AdamW at a constant learning rate: `1e-3` from scratch, then `3e-4` (`+stage=finetune`); gradient clipping by norm at `1.0`
- data: Yang et al.'s 20 tasks (`task_variant: yang`), `cartesian` encoding, `input_noise: 0.02`, `fix_noise: 0.25`, 320 fresh training trials per task each epoch, context tasks sampled 5 times as often
- stopping: `AcceptanceStopping` once the worst task's strict validation performance is at least 0.8 for 20 epochs; the checkpoint kept is `best_worst_task.ckpt`
