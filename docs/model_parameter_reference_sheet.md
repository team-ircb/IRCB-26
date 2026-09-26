# Parameter Reference

Constructor parameters for each model and data module. Released dataset values are the settings in `configs/` used for the public IRCB '26 files. All three released runs use `rnn_type: rnncell`, `max_epochs: 1000`, and `min_epochs: 300`.

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
