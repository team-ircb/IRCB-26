"""Cognitive Task Suite: a multi-area RNN trained on one or several cognitive tasks."""

import math
import random
from itertools import permutations

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import networkx as nx

import utils.visualization_utils as vis
from utils.torch_utils import RNNChannel, EMAMetric
from utils.common_utils import (
    Messages,
    get_insert_func,
    flatten,
    one_hot_encode,
    normalize_torch,
    sum_nested,
)

from .base import SAVE_DIR, DGNBase, VariableNoise


class CognitiveTaskSuite(DGNBase):
    """
    Multi-area recurrent network that performs multiple tasks jointly (DGN).

    Hidden-state noise is applied after the RNN cell (outside tanh), i.e.
    ``h <- tanh(...) + e``, matching post-activation observation noise.
    """
    def __init__(
        self,
        num_areas: int,
        task_names: list,
        diagram: list = None,

        stim_input_areas: list = None,
        rule_input_areas=None,
        sacc_output_areas: list = None,
        sacc_scale: float = 1.0,
        graph_kwargs: dict = None,
        rule_noise: float = 0.0,
        readout_areas: list = None,
        role_sampling: dict = None,
        grace_steps=0,
        pre_on_steps: int = 0,
        tail: str = "hold",
        readout_type: str = "binned",
        n_ring: int = 32,
        message_source: str = "pre_noise",
        stim_channels: int = 4,
        rule_aux_scale: float = 0.0,
        rule_aux_skip: int = 10,
        comm_act_scale: float = 0.0,
        comm_act_start: int = 0,
        comm_act_increase: int = 0,
        comm_act_init: float = 0.0,

        delay: int = 0,
        hidden_size: int = 32,
        lr_init: float = 4.0e-3,
        lr_schedule: str = 'plateau',
        rnn_type: str = 'grucell',
        num_angles: int = 36,
        num_channels: int = 4,

        noise: float = 0.0,
        noise_type: str = 'fixed',
        noise_start: int = 0,
        noise_increase: int = 0,
        noise_init: float = 0.0,
        channel_noise: float = 0.0,
        channel_noise_type: str = 'fixed',

        angle_start_epoch: int = 50,
        angle_increase_epoch: int = 100,
        angle_scale: float = 0.0,
        l1_start_epoch: int = 150,
        l1_increase_epoch: int = 150,
        l1_scale: float = 0.0,
        l2_scale: float = 0.0,
        l2_start: int = 0,
        l2_increase: int = 0,
        l2_init: float = 0.0,
        l2_comm_scale: float = 0.0,
        l2_comm_start: int = 0,
        l2_comm_increase: int = 0,
        l2_comm_init: float = 0.0,
        l2_groups: str = "split",
        smooth_start_epoch: int = 100,
        smooth_increase_epoch: int = 200,
        smooth_scale: float = 0.0,
    ):
        """
        Args:
            num_areas: Number of recurrent areas before the readout stage.
            task_names: Task names matching the datamodule; list order is the task index.
            diagram: Weighted edges (source, target, task_weight) for the task routing graph.
                If empty/None, a random graph is built from ``graph_kwargs``.

            stim_input_areas: Which area receives fix, stim1, stim2 (and optionally task).
            rule_input_areas: Areas that receive a one-hot task rule (``num_tasks`` channels,
                constant over the trial, as in Yang et al. 2019). ``"all"`` sends it to every
                area. ``None`` keeps the original behaviour (no one-hot rule).
            sacc_output_areas: Areas included in the saccade mask; unset becomes empty.
            sacc_scale: Weight on fixation / saccade loss.
            readout_areas: Areas whose readout blocks feed the response output (and, unless
                ``sacc_output_areas`` says otherwise, the saccade output). Default: the last
                area only. Entries of ``stim_input_areas`` may also be lists of
                areas, so one input stream can enter several areas.
            role_sampling: Version-2 ("distributed") routing. A dict with ``seed`` and
                optionally ``n_input_areas`` (default 2) and ``n_readout_areas`` (default 2).
                Each input stream (fixation, modality 1, modality 2, rule) is sent to a random
                ``n_input_areas`` areas and the output is read from a random
                ``n_readout_areas`` areas, redrawn until no area receives every stream and no
                stream enters only readout areas. The draw overwrites ``stim_input_areas``,
                ``rule_input_areas``, ``readout_areas`` and ``sacc_output_areas`` (and is
                stored in hparams). The graph rule becomes: every area reaches some readout
                area within ``shortest`` hops. ``n_input_areas = n_readout_areas = num_areas``
                gives "everything everywhere".
            rule_noise: SD of Gaussian noise added to the one-hot rule at every step (the
                same draw for every area that receives it), as Yang et al. add input noise
                to their rule units. 0 = noiseless rule.
            grace_steps: Steps after the go cue with no loss (every term). An int, or
                'auto' = (delay + 1) * (longest shortest path, in hops, from a stimulus or
                fixation input area to the readout) + 5: the steps the network physically
                needs before its response can change, plus Yang et al.'s 100 ms (5 steps
                at 20 ms). 0 = no grace period.
            pre_on_steps: First steps of every trial with no loss (Yang: 100 ms = 5 steps).
            tail: 'hold' = the response target is held, and trained, until the last step;
                'mask' = no loss from the trial's ``end`` step on (Yang's variable-length
                trials padded to T). Only matters for the 'yang' tasks.
            readout_type: 'binned' (36 direction logits passed from the readout areas
                through the message buffer to a readout node, plus a per-area saccade logit;
                CE + base MSE + angle + saccade BCE losses) or 'yang' (Yang et al. 2019:
                1 fixation + ``n_ring`` response-ring sigmoid outputs, one linear layer
                reading the readout areas' post-noise hidden state at the same step, no lag;
                MSE to Yang's targets with his cost mask; population-vector decoding).
            n_ring: Response-ring units for readout_type='yang' (Yang: 32).
            message_source: 'pre_noise' (messages and per-area outputs are computed from
                the hidden state before the hidden noise is added) or 'post_noise' (from the
                noisy state, i.e. from what the area's activity actually is).
            stim_channels: Inputs per stimulus modality: 4 for the slot encodings ('raw',
                'cartesian': 2 slots x 2), 32 for the data module's encoding='ring'.
            rule_aux_scale: Weight of an auxiliary rule-reconstruction loss (0 = off, the
                default; the model is then exactly as without it). When > 0, each area gets a
                linear decoder from its hidden state to the task (``num_tasks`` logits), trained
                with cross-entropy at every trial step from ``rule_aux_skip`` to the trial's end,
                averaged over areas. It makes every area carry the rule, which then has to be
                communicated from the rule input area. The decoders are training-only (not part
                of the network's dynamics or outputs) and are excluded from the L2 penalties.
            rule_aux_skip: First step at which the rule-reconstruction loss applies (time for
                the rule to reach every area).
            comm_act_scale: Group penalty on inter-area message activity: scale times the sum
                over pathways of each pathway's RMS message (over batch, time and message
                dimensions). Pushes whole pathways to silence unless they pay for themselves.
                0 = off. The readout-node blocks of readout_type='binned' are not included.
            comm_act_start, comm_act_increase, comm_act_init: Ramp for ``comm_act_scale``,
                as for the L2 terms.
            graph_kwargs: Options for random graph construction when ``diagram``
                is empty (``perc_conns``, ``shortest``, optional ``seed``).

            delay: Delay between messages between areas: a message computed at step t is
                first used at step t + delay + 1 (so delay 2 = a 3-step lag). Inputs and the
                'yang' readout are not delayed.
            hidden_size: Hidden units per area.
            lr_init: Learning rate for AdamW.
            lr_schedule: 'plateau' (ReduceLROnPlateau on valid/loss_ema, factor 0.95,
                patience 6) or 'constant' (fixed lr_init, as in Yang et al. 2019).
            rnn_type: Recurrent cell type for each area. Either 'grucell'
                (gated) or 'rnncell' (vanilla tanh RNN).
            num_angles: Bins for direction readout loss.
            num_channels: Size of each inter-area message.

            noise: Hidden-state noise scale (applied after RNN / outside tanh).
            noise_type: fixed or variable hidden noise.
            noise_start: Epoch to begin ramping noise (variable schedule).
            noise_increase: Epochs over which to ramp noise scale.
            noise_init: Starting scale factor for the noise ramp.
            channel_noise: Inter-area message noise scale.
            channel_noise_type: fixed or variable channel noise.

            angle_start_epoch: Start epoch for ramping in angle MSE.
            angle_increase_epoch: Ramp length for angle loss.
            angle_scale: Angle loss scale after ramp.
            l1_start_epoch: Start epoch for L1 ramp on communication weights.
            l1_increase_epoch: Ramp length for L1.
            l1_scale: L1 strength.
            l2_scale: L2 on non-communication parameters (RNN, readout, etc.).
            l2_start: Epoch to begin ramping L2 (same schedule as noise).
            l2_increase: Epochs over which to ramp L2 from ``l2_init`` to 1.
            l2_init: Starting scale factor for the L2 ramp.
            l2_comm_scale: L2 on inter-area communication weights only.
            l2_comm_start: Epoch to begin ramping communication L2.
            l2_comm_increase: Epochs over which to ramp communication L2.
            l2_comm_init: Starting scale factor for the communication L2 ramp.
            l2_groups: 'split' = two groups, each penalised by its own scale times the mean
                square of its parameters (biases included): inter-area message weights
                (``l2_comm_scale``) and all other parameters (``l2_scale``). 'uniform' = one
                penalty, ``l2_scale`` times the mean square over every weight matrix (input,
                recurrent, communication and readout weights alike; biases not penalised),
                so every weight costs the same; ``l2_comm_scale`` is then unused.
            smooth_start_epoch: Start epoch for smoothness loss ramp.
            smooth_increase_epoch: Ramp length for smoothness loss.
            smooth_scale: Smoothness loss scale.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["task_names"])

        hps = self.hparams
        hps.task_names = task_names
        hps.num_tasks = len(hps.task_names)
        if graph_kwargs is None:
            hps.graph_kwargs = {}
        self.fit_metric = EMAMetric(momentum=0.3)

        # Default stimulus input areas (fix, stim1, stim2); task slot set when provided
        if not stim_input_areas:
            hps.stim_input_areas = ["A0"] * 3
        else:
            hps.stim_input_areas = stim_input_areas

        if rule_input_areas == "all":
            hps.rule_input_areas = [f"A{ia}" for ia in range(num_areas)]
        else:
            hps.rule_input_areas = list(rule_input_areas) if rule_input_areas else []
        if hps.rule_input_areas and len(hps.stim_input_areas) > 3:
            raise ValueError(
                "Use either the one-hot rule (rule_input_areas) or the scalar task slot "
                "(4th entry of stim_input_areas), not both."
            )
        if hps.num_tasks > 1 and not hps.rule_input_areas and len(hps.stim_input_areas) <= 3:
            raise ValueError(
                f"{hps.num_tasks} tasks but no task cue: set rule_input_areas (one-hot rule) or "
                "a 4th stim_input_areas entry (scalar task slot), or train a single task."
            )

        all_names = [f"A{i}" for i in range(num_areas)]
        hps.readout_areas = list(readout_areas) if readout_areas else [f"A{num_areas - 1}"]
        if role_sampling:
            self._sample_roles(dict(role_sampling), all_names)

        if hps.sacc_output_areas is None:
            hps.sacc_output_areas = []
        # A name that matches no area (e.g. '3' instead of 'A3') would silently leave the
        # saccade output untrained. Warn rather than raise, so such models can still be rebuilt.
        unknown = [a for a in hps.sacc_output_areas if a not in [f"A{i}" for i in range(num_areas)]]
        if unknown:
            import warnings
            warnings.warn(
                f"sacc_output_areas {unknown} match no area (names are A0..A{num_areas - 1}); "
                "the saccade output will not be trained."
            )

        if hps.readout_type not in ("binned", "yang"):
            raise ValueError(f"readout_type must be 'binned' or 'yang', got {hps.readout_type!r}")
        if hps.l2_groups not in ("split", "uniform"):
            raise ValueError(f"l2_groups must be 'split' or 'uniform', got {hps.l2_groups!r}")
        if hps.message_source not in ("pre_noise", "post_noise"):
            raise ValueError(f"message_source must be 'pre_noise' or 'post_noise', got {hps.message_source!r}")
        # First output block of each area: its saccade logit ('binned') or none ('yang').
        self._k0 = 1 if hps.readout_type == "binned" else 0

        # Build areas, insert function, slice function, loss functions
        self._build_areas()
        self.insert_func, self.slice_func = self.get_insert_func_nested(hps.total_mesgs)
        self.mseloss = nn.MSELoss(reduction="none")
        self.bceloss = nn.BCELoss(reduction="none")
        self.celoss = nn.CrossEntropyLoss(reduction="none")

        # Get indices corresponding to the input of an area
        nested_idxs = {area_name: [] for area_name in self.area_names + [f"A{hps.num_areas}"]}
        for ia, area_name in enumerate(self.area_names + [f"A{hps.num_areas}"]):
            src_idxs = [int(node[1:]) for node in self.G.predecessors(area_name)]
            for src_idx in src_idxs:
                decs = sorted([int(node[1:]) for node in self.G.successors(f"A{src_idx}")])
                nested_idxs[area_name].append((src_idx, decs.index(ia)))
        self.input_indices = nested_idxs

        # Build task-routing mask into the output area
        edges = {}
        for src, tar, data in self.G.edges(data=True):
            if tar == f"A{hps.num_areas}":
                if int(src[1:]) not in edges.keys():
                    edges[int(src[1:])] = []
                edges[int(src[1:])].append(int(data["weight"]) - 1)
        self.edges = edges

        # Variable / fixed noise generators
        assert self.hparams.noise_type in ['fixed', 'variable']
        self.noise_weight_generator = VariableNoise(
            self.hparams.noise,
            hps.num_areas * hps.hidden_size,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            off=(self.hparams.noise_type == 'fixed'),
            start=hps.noise_start,
            increase=hps.noise_increase,
            init=hps.noise_init,
        )
        assert self.hparams.channel_noise_type in ['fixed', 'variable']
        self.cnoise_weight_generator = VariableNoise(
            self.hparams.channel_noise,
            sum_nested(hps.total_mesgs),
            device='cuda' if torch.cuda.is_available() else 'cpu',
            off=(self.hparams.channel_noise_type == 'fixed'),
        )
        self.h_noise_weight = torch.ones(hps.num_areas * hps.hidden_size).to(self.device) * hps.noise
        self.c_noise_weight = torch.ones(sum_nested(hps.total_mesgs)).to(self.device) * hps.channel_noise

        # Auxiliary rule-reconstruction decoders: built last and only when used, so the default
        # model's parameters, random draws and state dict are unchanged
        if float(getattr(hps, "rule_aux_scale", 0.0) or 0.0) > 0:
            self.rule_decoders = nn.ModuleDict(
                {a: nn.Linear(hps.hidden_size, hps.num_tasks) for a in self.area_names})

    def on_load_checkpoint(self, checkpoint):
        """Warm start into a model with rule decoders from a checkpoint without them: add the
        decoders' fresh weights to the state dict, and their parameters (no Adam state yet) to
        the optimiser's single parameter group, where configure_optimizers puts them last."""
        if not hasattr(self, "rule_decoders"):
            return
        sd = checkpoint.get("state_dict", {})
        new = {f"rule_decoders.{k}": v for k, v in self.rule_decoders.state_dict().items()}
        missing = [k for k in new if k not in sd]
        if not missing:
            return
        for k in missing:
            sd[k] = new[k].clone()
        n_all = len(list(self.parameters()))
        n_new = len(list(self.rule_decoders.parameters()))
        for opt in checkpoint.get("optimizer_states", []):
            groups = opt.get("param_groups", [])
            if len(groups) == 1 and len(groups[0]["params"]) == n_all - n_new:
                groups[0]["params"] = list(groups[0]["params"]) + list(range(n_all - n_new, n_all))

    def forward(self, inp, step_type):
        hps = self.hparams
        batch, time, _ = inp[0].shape
        self._build_save_var(batch, time)

        h = torch.zeros(batch, hps.hidden_size * hps.num_areas).to(self.device)
        mesgs = torch.zeros(batch, sum_nested(hps.total_mesgs)).to(self.device)

        # Find task corresponding to each batch, shape = (batch,)
        task_idx = inp[3][:, 0, 0]

        # One-hot task rule, constant over the trial, shape = (batch, num_tasks)
        rule = None
        if hps.rule_input_areas:
            rule = torch.nn.functional.one_hot(task_idx.long(), hps.num_tasks).to(inp[0].dtype)

        # Per-step results are collected in lists and stacked once after the loop.
        # Writing each step into preallocated (batch, time, ...) buffers made every
        # backward step copy the whole buffer (cost ~ time^2); results are unchanged.
        hidden_list = {name: [] for name in self.area_names}
        input_list = {name: [] for name in self.area_names}
        mesgs_list, latents_list, projs_list = [], [], []

        self.outputs = []
        rule_noise = float(getattr(hps, "rule_noise", 0.0) or 0.0)
        yang = getattr(hps, "readout_type", "binned") == "yang"
        post_noise = getattr(hps, "message_source", "pre_noise") == "post_noise"
        readout_idx = [i for i, a in enumerate(self.area_names) if a in hps.readout_areas]
        for t in range(time):
            h_ias, fixs = [], []
            rule_t = rule
            if rule is not None and rule_noise > 0:
                rule_t = rule + torch.randn_like(rule) * rule_noise
            mesgs_new = torch.zeros(batch, sum_nested(hps.total_mesgs)).to(self.device)

            for ia, (area_name, area) in enumerate(self.areas.items()):
                # Gather external input
                inp_idxs = [i for i in range(len(hps.stim_input_areas)) if area_name in self._stream_areas(i)]
                inp_ia = [inp[inp_idx][:, t] for inp_idx in inp_idxs]
                if rule is not None and area_name in hps.rule_input_areas:
                    inp_ia.append(rule_t)

                if len(inp_ia) > 0:
                    input_list[area_name].append(torch.cat(inp_ia, dim=1).to(torch.float32))

                # Gather upstream input
                for idx in self.input_indices[area_name]:
                    inp_ia.append(self.slice_func(mesgs, *idx))
                inp_ia = torch.cat(inp_ia, dim=1).to(torch.float32)

                # Forward through area, then add noise outside tanh: h <- RNN(...) + e.
                # Messages (and per-area outputs) come from the pre-noise state
                # (message_source='pre_noise') or from the noisy state ('post_noise').
                h_prev = h[:, ia * hps.hidden_size: (ia + 1) * hps.hidden_size]
                e_ia = self.h_noise_weight[
                    ia * hps.hidden_size: (ia + 1) * hps.hidden_size
                ].reshape(1, -1).to(self.device)
                if post_noise:
                    h_ia = area.rnn(inp_ia, h_prev)
                    h_ia = h_ia + torch.randn_like(h_ia) * e_ia
                    mesg_ias = torch.split(area.output(h_ia), area.output_sizes, dim=-1)
                else:
                    h_ia, mesg_ias = area(inp_ia, h_prev)
                    h_ia = h_ia + torch.randn_like(h_ia) * e_ia
                h_ias.append(h_ia)

                # Store fixation (per-area saccade logit; 'binned' readout only)
                if not yang:
                    if len(hps.sacc_output_areas) > 0:
                        if area_name in hps.sacc_output_areas:
                            sacc_mask = torch.ones_like(mesg_ias[0]).to(self.device)
                        else:
                            sacc_mask = torch.zeros_like(mesg_ias[0]).to(self.device)
                    else:
                        sacc_mask = torch.ones_like(mesg_ias[0]).to(self.device)
                    fixs.append(mesg_ias[0] * sacc_mask)

                for im, mesg_ia in enumerate(mesg_ias[self._k0:]):
                    self.insert_func(mesgs_new, mesg_ia, ia, im)
                hidden_list[area_name].append(h_ia)

            if yang:
                # Same-step linear readout of the readout areas' post-noise state (Yang)
                output = self.yang_readout(torch.cat([h_ias[i] for i in readout_idx], dim=-1))
                self.outputs.append(output.unsqueeze(1))
                fix_logit = output[:, :1]
                latents_list.append(torch.cat(
                    [fix_logit if i in readout_idx else torch.zeros_like(fix_logit)
                     for i in range(hps.num_areas)], dim=-1))
            else:
                # Forward pass through output area (task-masked)
                inp_ia, mask = [], []
                for idx in self.input_indices[f"A{hps.num_areas}"]:
                    mask = (task_idx.unsqueeze(1) == torch.Tensor(self.edges[idx[0]]).to(self.device)).any(dim=1).to(int)
                    output = self.slice_func(mesgs, *idx)
                    inp_ia.append(output * mask.reshape(-1, 1))

                inp_ia = torch.cat(inp_ia, dim=1)
                output = self.output_area(inp_ia)
                self.outputs.append(output.unsqueeze(1))
                latents_list.append(torch.cat(fixs, dim=-1))
                projs_list.append(self.readout(output))

            mesgs_list.append(mesgs_new)
            h = torch.cat(h_ias, dim=-1)

            if t >= hps.delay:
                mesgs = mesgs_list[t - hps.delay]

            channel_noise = torch.randn_like(mesgs) * self.c_noise_weight.reshape(1, -1).to(self.device)
            mesgs = mesgs + channel_noise

        for area_name in self.area_names:
            self.hidden_states[area_name] = torch.stack(hidden_list[area_name], dim=1)
            if input_list[area_name]:
                self.inputs[area_name] = torch.stack(input_list[area_name], dim=1)
        self.save_var = self.save_var._replace(
            mesgs=torch.stack(mesgs_list, dim=1),
            latents=torch.stack(latents_list, dim=1),
        )
        self.projs = torch.stack(projs_list, dim=1) if projs_list else None

        self.h_noise_weight = self.noise_weight_generator(
            torch.cat([self.hidden_states[n] for n in self.area_names], dim=2),
            current_epoch=self.current_epoch,
        )
        self.c_noise_weight = self.cnoise_weight_generator(self.save_var.mesgs)

        self.outputs = torch.cat(self.outputs, dim=1)
        return self.outputs

    def _shared_step(self, batch, step_type):
        hps = self.hparams
        inp, info = batch
        self.current_batch = inp
        self.current_info = info

        fix, stim1, amp1, stim2, amp2, task, resp, sacc = inp[:8]
        resp = resp.float()
        B, T = resp.shape[0], resp.shape[1]
        steps_t = torch.arange(T, device=resp.device).reshape(1, T)
        if len(inp) >= 11:
            go = inp[8][:, 0, 0].long().reshape(B, 1)
            cond = inp[9][:, 0, 0].long()
            end = inp[10][:, 0, 0].long().reshape(B, 1)
        else:
            off = (sacc[..., 0] < 0.5)
            go = torch.where(off.any(1), off.float().argmax(1), torch.full((B,), T)).long().reshape(B, 1)
            cond = torch.zeros(B, dtype=torch.long)
            end = torch.full((B, 1), T, dtype=torch.long)
        # Loss mask (B, T, 1): 0 during the first pre_on steps, the grace period after the
        # go cue, and (tail='mask') the padding after the trial's end.
        grace = self._grace_steps()
        cm = (steps_t >= int(getattr(hps, "pre_on_steps", 0) or 0)).expand(B, T).clone()
        cm &= ~((steps_t >= go) & (steps_t < go + grace))
        if getattr(hps, "tail", "hold") == "mask":
            cm &= steps_t < end
        cm = cm.float().unsqueeze(-1).to(self.device)
        # Each modality is two alternatives: [θ0, θ1, s0, s1], strengths already over time.
        polar1 = torch.cat([stim1, amp1], dim=2)
        polar2 = torch.cat([stim2, amp2], dim=2)
        self.forward([fix, polar1, polar2, task], step_type)

        yang = getattr(hps, "readout_type", "binned") == "yang"
        resp_onehot = one_hot_encode(
            normalize_torch(resp), [-math.pi, math.pi], hps.num_angles
        )
        resp_mask = torch.where(
            resp != 0, torch.tensor(1).to(self.device), torch.tensor(0).to(self.device)
        )

        if yang:
            # Yang et al. (2019): MSE to his targets under his cost mask (grace and padding
            # handled by cm)
            loss_task, task_terms = self._yang_loss(resp, sacc, resp_mask, cm, go, steps_t)
        else:
            # Fixation loss, averaged over the areas whose saccade output is trained only.
            # (Averaging over all areas would dilute sacc_scale by the number of untrained
            # areas and add a constant.) Falls back to all areas when none match.
            sacc_cols = self._sacc_columns()
            sacc_prob = torch.sigmoid(self.save_var.latents)
            loss_sacc = self.bceloss(
                sacc_prob[..., sacc_cols],
                torch.tile(sacc, (1, 1, len(sacc_cols))).float(),
            )
            loss_sacc = torch.mean(loss_sacc * cm) * hps.sacc_scale

            # Response / angle losses
            angle_ramp = self._compute_ramp(hps.angle_start_epoch, hps.angle_increase_epoch)
            cosine_pred = torch.cos(self.projs)
            sine_pred = torch.sin(self.projs)
            cosine_true = torch.cos(resp).to(self.device)
            sine_true = torch.sin(resp).to(self.device)

            loss_angle = self.mseloss(cosine_true, cosine_pred) + self.mseloss(sine_true, sine_pred)
            loss_angle = torch.mean(loss_angle * resp_mask * cm) * hps.angle_scale

            loss_resp = self.celoss(flatten(self.outputs), flatten(resp_onehot)).reshape(*resp.shape)
            loss_resp = torch.mean(loss_resp * resp_mask * cm)

            loss_base = self.mseloss(self.outputs, torch.zeros_like(self.outputs).to(self.device))
            base_mask = torch.where(
                resp == 0, torch.tensor(1).to(self.device), torch.tensor(0).to(self.device)
            )
            loss_base = torch.mean(loss_base * base_mask * cm)
            loss_task = loss_sacc + loss_resp + loss_base + loss_angle * angle_ramp
            task_terms = {"loss_sacc": loss_sacc, "loss_resp": loss_resp,
                          "loss_base": loss_base, "loss_angle": loss_angle}

        # L1 / L2 on communication vs. all other weights (disjoint slices)
        comm_kernels, other_kernels = self._comm_and_other_kernels()

        l1_ramp = self._compute_ramp(hps.l1_start_epoch, hps.l1_increase_epoch)
        loss_l1, kernel_size = 0.0, 0
        if hps.l1_scale > 0:
            for kernel in comm_kernels:
                if kernel.ndim != 2:
                    continue
                loss_l1 += hps.l1_scale * torch.norm(kernel, 1)
                kernel_size += kernel.numel()
            loss_l1 /= kernel_size + 1e-8

        l2_ramp = self._compute_scale_ramp(hps.l2_start, hps.l2_increase, hps.l2_init)
        l2_comm_ramp = self._compute_scale_ramp(
            hps.l2_comm_start, hps.l2_comm_increase, hps.l2_comm_init
        )
        if getattr(hps, "l2_groups", "split") == "uniform":
            # One penalty over every weight matrix (biases excluded), so no weight type is
            # cheaper than another
            weights = [p for n, p in self.named_parameters() if p.ndim >= 2 and not n.startswith("rule_decoders.")]
            loss_l2 = self._mean_sq(weights) * hps.l2_scale if hps.l2_scale > 0 else 0.0
            loss_l2_comm = 0.0
        else:
            loss_l2 = self._mean_sq(other_kernels) * hps.l2_scale if hps.l2_scale > 0 else 0.0
            loss_l2_comm = (
                self._mean_sq(comm_kernels) * hps.l2_comm_scale if hps.l2_comm_scale > 0 else 0.0
            )

        # Group penalty on message activity: sum over pathways of each pathway's RMS message
        comm_act_scale = float(getattr(hps, "comm_act_scale", 0.0) or 0.0)
        loss_comm_act = None
        if comm_act_scale > 0 and self.pathways:
            m = self.save_var.mesgs
            loss_comm_act = comm_act_scale * sum(
                torch.sqrt(m[..., a:b].pow(2).mean() + 1e-8) for _, _, a, b in self.pathways
            )
            comm_act_ramp = self._compute_scale_ramp(
                hps.comm_act_start, hps.comm_act_increase, hps.comm_act_init
            )

        # Smoothness loss
        smooth_ramp = self._compute_ramp(hps.smooth_start_epoch, hps.smooth_increase_epoch)
        loss_smooth = 0.0
        for area_name in self.areas:
            diff = self.hidden_states[area_name][:, 1:] - self.hidden_states[area_name][:, :-1]
            loss_smooth += self.mseloss(diff, torch.zeros_like(diff))
        loss_smooth = loss_smooth.mean()
        loss_smooth *= hps.smooth_scale

        # Auxiliary rule reconstruction from every area (off unless rule_aux_scale > 0)
        loss_rule, rule_acc = None, {}
        if hasattr(self, "rule_decoders"):
            T_ = task.shape[1]
            tid = task[:, 0, 0].long()
            steps_ = torch.arange(T_, device=task.device).reshape(1, T_)
            rmask = ((steps_ >= int(getattr(hps, "rule_aux_skip", 10))) & (steps_ < end)).float()
            denom = rmask.sum().clamp(min=1.0)
            loss_rule = 0.0
            for a in self.area_names:
                logits_r = self.rule_decoders[a](self.hidden_states[a])  # (B, T, num_tasks)
                ce = self.celoss(logits_r.reshape(-1, hps.num_tasks), tid.repeat_interleave(T_)).reshape(-1, T_)
                loss_rule = loss_rule + (ce * rmask).sum() / denom
                with torch.no_grad():
                    rule_acc[a] = ((logits_r.argmax(-1) == tid[:, None]).float() * rmask).sum() / denom
            loss_rule = loss_rule / len(self.area_names)

        # Fixation probability and decoded direction, each (B, T)
        fix_all, ang_all = self.readout_signals()
        bin_width = 2 * math.pi / hps.num_angles

        # Per-task accuracy at the last time point: decoded direction in the target's bin
        if yang:
            pred = ((torch.remainder(ang_all[:, -1] + math.pi, 2 * math.pi)) / bin_width).floor().long()
            pred = pred.clamp(0, hps.num_angles - 1)
        else:
            pred = torch.argmax(self.outputs[:, -1].detach().cpu(), dim=-1)
        target = torch.argmax(resp_onehot[:, -1].detach().cpu(), dim=-1)
        acc = (pred == target).float()
        per_task_acc = {}
        task_last = task[:, -1, 0].detach().cpu().long()
        for itask, task_name in enumerate(hps.task_names):
            mask = (task_last == itask)
            if mask.any():
                per_task_acc[task_name] = acc[mask].mean().item()

        # Yang et al. (2019) performance at the last time step. A trial is correct if
        #   no-go / should fixate (saccade target 1): the fixation output still says fixate;
        #   go (saccade target 0): it has broken fixation AND the decoded direction is
        #   within 0.2*pi of the target.
        # Decoded direction: population vector over the ring ('yang' readout) or the centre
        # of the argmax bin ('binned').
        fix_out = fix_all[:, -1] > 0.5
        should_fix = sacc[:, -1, 0].detach().cpu() > 0.5
        dist = torch.remainder(ang_all[:, -1] - resp[:, -1, 0].detach().cpu() + math.pi, 2 * math.pi) - math.pi
        go_ok = (~fix_out) & (dist.abs() < 0.2 * math.pi)
        perf = torch.where(should_fix, fix_out, go_ok).float()
        per_task_perf = {}
        for itask, task_name in enumerate(hps.task_names):
            mask = (task_last == itask)
            if mask.any():
                per_task_perf[task_name] = perf[mask].mean().item()

        # Stricter Yang-style performance ("perfstrict"), as Yang et al.'s text describes it:
        # scored at the trial's last step (end - 1; T - 1 without padding) and, in addition,
        # the fixation output must say "fixate" (> 0.5) at every step from pre_on (at least
        # step 5, Yang's unscored 100 ms) up to the go cue, or, on no-go trials, up to the end.
        endc, goc = end.cpu().reshape(-1), go.cpu().reshape(-1)
        idx = (endc - 1).clamp(0, T - 1)
        rows = torch.arange(B)
        fix_end = fix_all[rows, idx] > 0.5
        should_fix_end = sacc.detach().cpu()[rows, idx, 0] > 0.5
        tgt = resp.detach().cpu()[rows, idx, 0]
        dist_e = torch.remainder(ang_all[rows, idx] - tgt + math.pi, 2 * math.pi) - math.pi
        t0 = max(5, int(getattr(hps, "pre_on_steps", 0) or 0))
        tt = torch.arange(T).reshape(1, T)
        hold_until = torch.where(should_fix_end, endc, goc).reshape(B, 1)
        window = (tt >= t0) & (tt < hold_until)
        held = ((fix_all > 0.5) | ~window).all(dim=1)
        strict = torch.where(should_fix_end, held, held & ~fix_end & (dist_e.abs() < 0.2 * math.pi)).float()
        condc = cond.cpu()
        from utils.yang_tasks import CONDITION_NAMES
        for itask, task_name in enumerate(hps.task_names):
            mask = (task_last == itask)
            if mask.any():
                per_task_perf[f"strict_{task_name}"] = strict[mask].mean().item()
                for ic, cname in enumerate(CONDITION_NAMES.get(task_name, ())):
                    mc = mask & (condc == ic)
                    if mc.any():
                        per_task_perf[f"strict_{task_name}_{cname}"] = strict[mc].mean().item()

        loss = (
            loss_task
            + loss_l1 * l1_ramp
            + loss_l2 * l2_ramp
            + loss_l2_comm * l2_comm_ramp
            + loss_smooth * smooth_ramp
        )
        if loss_comm_act is not None:
            loss = loss + loss_comm_act * comm_act_ramp
        if loss_rule is not None:
            loss = loss + loss_rule * float(hps.rule_aux_scale)
        metrics = {f"{step_type}/loss": loss}
        if loss_rule is not None:
            metrics[f"{step_type}/loss_rule_aux"] = loss_rule
            for a, v in rule_acc.items():
                metrics[f"{step_type}/rule_acc_{a}"] = v
        metrics.update({f"{step_type}/{k}": v for k, v in task_terms.items()})
        metrics.update({
            f"{step_type}/loss_l1": loss_l1,
            f"{step_type}/loss_l2": loss_l2,
            f"{step_type}/loss_l2_comm": loss_l2_comm,
            f"{step_type}/loss_smooth": loss_smooth,
        })
        if loss_comm_act is not None:
            metrics[f"{step_type}/loss_comm_act"] = loss_comm_act
        if step_type == "valid" and self.pathways:
            # Traffic per inter-area pathway: RMS message over trials, steps and dimensions
            with torch.no_grad():
                m = self.save_var.mesgs
                for src, tgt, a, b in self.pathways:
                    metrics[f"valid/msg_rms_{src}_to_{tgt}"] = m[..., a:b].pow(2).mean().sqrt()
        for task_name, acc_val in per_task_acc.items():
            metrics[f"{step_type}/acc_{task_name}"] = acc_val
        for task_name, perf_val in per_task_perf.items():
            if task_name.startswith("strict_"):
                metrics[f"{step_type}/perfstrict_{task_name[7:]}"] = perf_val
            else:
                metrics[f"{step_type}/perf_{task_name}"] = perf_val

        if step_type == "valid":
            batch_size = len(fix)
            self.fit_metric.update(loss, batch_size)
            metrics["valid/loss_ema"] = self.fit_metric.compute()
            # Acceptance summaries over tasks (strict metric). The "_after_ramp" version is -1
            # until the noise and L2 ramps have finished, so checkpoint selection and
            # acceptance-based stopping only see networks at their final noise level.
            task_strict = [per_task_perf[f"strict_{t}"] for t in hps.task_names if f"strict_{t}" in per_task_perf]
            if task_strict:
                ramp_end = max(hps.noise_start + hps.noise_increase, hps.l2_start + hps.l2_increase,
                               hps.l2_comm_start + hps.l2_comm_increase)
                if float(getattr(hps, "comm_act_scale", 0.0) or 0.0) > 0:
                    ramp_end = max(ramp_end, hps.comm_act_start + hps.comm_act_increase)
                mn = float(min(task_strict))
                metrics["valid/perfstrict_min"] = mn
                metrics["valid/n_tasks_pass_0.8"] = float(sum(v >= 0.8 for v in task_strict))
                metrics["valid/perfstrict_min_after_ramp"] = mn if self.current_epoch + 1 >= ramp_end else -1.0

        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            batch_size=inp[0].shape[0],
        )
        return loss

    def _sample_roles(self, rs, names):
        hps = self.hparams
        if "seed" not in rs:
            raise ValueError("role_sampling needs a seed, so a checkpoint rebuilds the same roles")
        rng = random.Random(int(rs["seed"]))
        n_in = int(rs.get("n_input_areas", 2))
        n_out = int(rs.get("n_readout_areas", 2))
        everywhere = n_in >= len(names)
        for _ in range(10000):
            streams = [sorted(rng.sample(names, n_in)) for _ in range(4)]  # fix, mod1, mod2, rule
            readout = sorted(rng.sample(names, n_out))
            if everywhere:
                break
            gets_all = any(all(a in st for st in streams) for a in names)
            only_readout = any(set(st) <= set(readout) for st in streams)
            if not gets_all and not only_readout:
                break
        else:
            raise RuntimeError("role sampling found no valid assignment")
        hps.stim_input_areas = streams[:3]
        hps.rule_input_areas = streams[3]
        hps.readout_areas = readout
        hps.sacc_output_areas = list(readout)

    def _stream_areas(self, i):
        """Areas receiving external input stream i (0 fix, 1 modality 1, 2 modality 2, 3 task)."""
        hps = self.hparams
        if i >= len(hps.stim_input_areas):
            return []
        v = hps.stim_input_areas[i]
        return [v] if isinstance(v, str) else list(v)

    def _ext_input_size(self, area_name) -> int:
        hps = self.hparams
        n = 0
        sw = int(getattr(hps, "stim_channels", 4) or 4)
        for i, width in enumerate([1, sw, sw, 1]):  # fix; per modality 2 slots x 2 (or a ring); scalar task
            if area_name in self._stream_areas(i):
                n += width
        if area_name in hps.rule_input_areas:
            n += hps.num_tasks
        return n

    def _grace_steps(self) -> int:
        g = getattr(self.hparams, "grace_steps", 0) or 0
        if g != "auto":
            return int(g)
        if getattr(self, "_grace_auto", None) is None:
            import networkx as nx
            hps = self.hparams
            if getattr(hps, "readout_type", "binned") == "yang":
                # Hops from each input stream to the nearest readout area (0 if it enters one);
                # the 'yang' readout itself adds no lag.
                def hops_to_readout(a):
                    return min(nx.shortest_path_length(self.G, a, r)
                               for r in hps.readout_areas if a == r or nx.has_path(self.G, a, r))
            else:
                out = f"A{hps.num_areas}"

                def hops_to_readout(a):
                    return nx.shortest_path_length(self.G, a, out)
            hops = max(min(hops_to_readout(a) for a in self._stream_areas(i))
                       for i in range(3) if self._stream_areas(i))
            self._grace_auto = (hps.delay + 1) * hops + 5
        return self._grace_auto

    def readout_signals(self):
        """Fixation probability and decoded response direction (radians), each (B, T), on the
        CPU, from the last forward pass. 'yang' readout: the sigmoid fixation unit, and the
        population vector over the ring (Yang et al. 2019). 'binned': the mean saccade
        probability over the trained areas, and the centre of the argmax direction bin."""
        hps = self.hparams
        if getattr(hps, "readout_type", "binned") == "yang":
            y = torch.sigmoid(self.outputs.detach()).cpu()
            ring = y[..., 1:]
            pref = torch.arange(hps.n_ring, dtype=y.dtype) * (2 * math.pi / hps.n_ring)
            ang = torch.atan2((ring * torch.sin(pref)).sum(-1), (ring * torch.cos(pref)).sum(-1))
            return y[..., 0], ang
        cols = self._sacc_columns()
        fix = torch.sigmoid(self.save_var.latents)[..., cols].mean(dim=-1).detach().cpu()
        bw = 2 * math.pi / hps.num_angles
        ang = -math.pi + (torch.argmax(self.outputs.detach().cpu(), dim=-1).float() + 0.5) * bw
        return fix, ang

    def readout_targets(self, resp, sacc):
        """Yang et al. (2019) targets for the 'yang' readout, (B, T, 1 + n_ring): 0.05
        baseline; fixation unit 0.8 while the network should fixate; ring bump
        0.8 * exp(-d^2 / (2 (pi/8)^2)) around the response direction while responding."""
        hps = self.hparams
        resp = resp.float()
        pref = torch.arange(hps.n_ring, device=resp.device, dtype=resp.dtype) * (2 * math.pi / hps.n_ring)
        d = torch.remainder(resp - pref.reshape(1, 1, -1) + math.pi, 2 * math.pi) - math.pi
        responding = (resp != 0).to(resp.dtype)
        y_ring = 0.05 + 0.8 * torch.exp(-0.5 * (d / (math.pi / 8)) ** 2) * responding
        y_fix = 0.05 + 0.75 * (sacc > 0.5).to(resp.dtype)
        return torch.cat([y_fix, y_ring], dim=-1)

    def _yang_loss(self, resp, sacc, resp_mask, cm, go, steps_t):
        """Yang et al. (2019) least-squares loss: mean over trial x step x output of
        ((target - output) * c)^2, with c = 0 in the masked steps (cm: first pre_on steps,
        grace period, padding), 1 before the go cue, 5 from it on, and the fixation column
        doubled ("Fixation is important", Yang task.py:151-152)."""
        hps = self.hparams
        B, T = resp.shape[0], resp.shape[1]
        y_hat = torch.sigmoid(self.outputs)
        y = self.readout_targets(resp.to(self.device), sacc.to(self.device))
        after_go = (steps_t.to(self.device) >= go.to(self.device)).unsqueeze(-1).float()
        c = cm * (1.0 + 4.0 * after_go)
        n = B * T * (1 + hps.n_ring)
        loss_fix = (((y[..., :1] - y_hat[..., :1]) * c * 2.0) ** 2).sum() / n
        loss_ring = (((y[..., 1:] - y_hat[..., 1:]) * c) ** 2).sum() / n
        return loss_fix + loss_ring, {"loss_fix": loss_fix, "loss_ring": loss_ring}

    def on_before_optimizer_step(self, optimizer):
        # Gradient statistics before clipping (Lightning clips after this hook), so the
        # sweep can show how often a clip threshold binds.
        grads = [p.grad.detach().flatten() for p in self.parameters() if p.grad is not None]
        if not grads:
            return
        g = torch.cat(grads)
        self.log_dict(
            {
                "grad/norm": g.norm(),
                "grad/max_abs": g.abs().max(),
                "grad/frac_abs_gt_0.5": (g.abs() > 0.5).float().mean(),
            },
            on_step=False,
            on_epoch=True,
        )

    def _sacc_columns(self) -> list:
        """Indices (into the per-area saccade outputs) of the areas trained on the saccade."""
        names = self.hparams.sacc_output_areas
        if getattr(self.hparams, "readout_type", "binned") == "yang":
            names = self.hparams.readout_areas  # the fixation unit is read from these areas
        cols = [ia for ia, a in enumerate(self.area_names) if a in names]
        return cols if cols else list(range(self.hparams.num_areas))

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            try:
                _, ax = plt.subplots(1, 1)
                self.draw(ax)
                vis.savefig(f"{SAVE_DIR}network.png")
            except Exception:
                pass
        return self._shared_step(batch, "valid")

    def configure_optimizers(self):
        hps = self.hparams
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=hps.lr_init,
            betas=(0.9, 0.99),
            eps=1.0e-8,
            weight_decay=0.0,
        )
        schedule = getattr(hps, "lr_schedule", "plateau")
        if schedule == "constant":
            return optimizer
        if schedule != "plateau":
            raise ValueError(f"lr_schedule must be 'plateau' or 'constant', got {schedule!r}")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=0.95,
            patience=6,
            threshold=0.0,
            min_lr=1.0e-5,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "valid/loss_ema",
            },
        }

    def _build_graph(self):
        """Sample a random inter-area connectome when ``diagram`` is empty.

        ``stim_input_areas`` is left unchanged. Requires ``graph_kwargs``:
            perc_conns: fraction of directed area-to-area edges to keep
            shortest: max path length from every area to A{num_areas-1}
            seed: optional; if unset, uses the process RNG (``--seed``)
        A draw is rejected unless every area has in-degree and out-degree
        at least 1, and every area reaches the output in at most ``shortest``
        hops. The realized edge list is written back to ``hparams.diagram``.
        """
        hps = self.hparams
        if "perc_conns" not in hps.graph_kwargs or "shortest" not in hps.graph_kwargs:
            raise ValueError(
                "Random diagrams need graph_kwargs.perc_conns and graph_kwargs.shortest "
                "(and model.diagram=null)."
            )

        self.area_names = [f"A{ia}" for ia in range(hps.num_areas)]
        all_conns = list(permutations(self.area_names, 2))
        n_keep = int(hps.graph_kwargs["perc_conns"] * len(all_conns))
        max_len = int(hps.graph_kwargs["shortest"])
        readout = list(getattr(hps, "readout_areas", None) or [f"A{hps.num_areas - 1}"])
        rng = (
            random.Random(int(hps.graph_kwargs["seed"]))
            if hps.graph_kwargs.get("seed") is not None
            else random
        )

        stop_search, iters, edges = False, 0, []
        while (not stop_search) and (iters <= 500):
            rng.shuffle(all_conns)
            edges = all_conns[:n_keep]

            stop_search = True
            graph = nx.DiGraph(edges)
            graph.add_nodes_from(self.area_names)
            for area_name in self.area_names:
                if graph.in_degree(area_name) == 0 or graph.out_degree(area_name) == 0:
                    stop_search = False
                    break
            if stop_search:
                try:
                    for area_name in self.area_names:
                        if area_name in readout:
                            continue
                        dists = [nx.shortest_path_length(graph, area_name, r)
                                 for r in readout if nx.has_path(graph, area_name, r)]
                        if not dists or min(dists) > max_len:
                            stop_search = False
                            break
                except nx.NetworkXNoPath:
                    stop_search = False
            iters += 1

        if not stop_search:
            raise RuntimeError("Graph search exceeded max iters.")

        wedges = [[source, target, 0] for source, target in edges]
        for r in readout:
            for itask in range(hps.num_tasks):
                wedges.append([r, f"A{hps.num_areas}", itask + 1])

        graph = nx.MultiDiGraph()
        graph.add_weighted_edges_from(wedges)
        hps.diagram = wedges
        self.edges = wedges
        self.G = graph
        return graph

    def _build_specified_graph(self):
        hps = self.hparams
        self.area_names = [f"A{ia}" for ia in range(hps.num_areas)]
        wedges = hps.diagram

        graph = nx.MultiDiGraph()
        graph.add_weighted_edges_from(wedges)
        self.edges = wedges
        self.G = graph
        return graph

    def _build_areas(self):
        hps = self.hparams
        self.areas = nn.ModuleDict()

        if hps.diagram:
            G = self._build_specified_graph()
        else:
            G = self._build_graph()

        sG = nx.ego_graph(G, f"A{hps.num_areas}", undirected=True)
        rG = G.copy()
        rG.remove_node(f"A{hps.num_areas}")

        hps.total_mesgs = []
        hps.successor_list = {}
        for ia in range(hps.num_areas):
            area_name = f"A{ia}"

            num_input = self._ext_input_size(area_name)  # fix 1, each modality 4, task 1, rule

            num_up = len(list(rG.predecessors(f"A{ia}"))) * hps.num_channels
            num_down = [hps.num_channels] * len(list(rG.successors(f"A{ia}")))

            if hps.readout_type == "binned" and (f"A{ia}", f"A{hps.num_areas}") in sG.edges:
                num_out = [hps.num_angles]
                flag_out = -hps.num_angles
            else:
                num_out = []
                flag_out = -1

            self.areas[f"A{ia}"] = RNNChannel(
                num_input + num_up,
                hps.hidden_size,
                [1] * self._k0 + num_down + num_out,
                None,
                override_single=True,
                rnn_type=hps.rnn_type,
            )
            hps.total_mesgs.append(num_down + num_out)
            hps.successor_list[area_name] = (self._k0, flag_out)

        # Inter-area pathways in the flat message vector: (source, target, start, stop).
        # Block im of area ia goes to the im-th of its sorted successors (as input_indices).
        self.pathways, off = [], 0
        for ia in range(hps.num_areas):
            decs = sorted(int(n[1:]) for n in G.successors(f"A{ia}"))
            for im, width in enumerate(hps.total_mesgs[ia]):
                if decs[im] < hps.num_areas:
                    self.pathways.append((f"A{ia}", f"A{decs[im]}", off, off + width))
                off += width

        if hps.readout_type == "yang":
            # Yang et al. (2019): one linear layer, fixation unit + response ring, read from
            # the readout areas' (post-noise) hidden state at the same step.
            self.yang_readout = nn.Linear(hps.hidden_size * len(hps.readout_areas), 1 + hps.n_ring)
        else:
            num_up = len(list(sG.predecessors(f"A{hps.num_areas}")))
            self.output_area = nn.Linear(num_up * hps.num_angles, hps.num_angles)
            self.readout = nn.Sequential(
                nn.Linear(hps.num_angles, 1),
            )

    def _build_save_var(self, batch_size, time):
        hps = self.hparams
        self.save_var = Messages(
            mesgs=torch.zeros(batch_size, time, sum_nested(hps.total_mesgs)).to(self.device),
            latents=torch.zeros(batch_size, time, hps.num_areas).to(self.device),
        )

        self.hidden_states = {}
        self.inputs = {}
        for ia, area_name in enumerate(self.area_names):
            self.hidden_states[area_name] = torch.zeros(
                batch_size, time, hps.hidden_size
            ).to(self.device)

            num_input = self._ext_input_size(area_name)
            self.inputs[area_name] = torch.zeros(batch_size, time, num_input).to(self.device)

        self.projs = torch.zeros(batch_size, time, 1).to(self.device)

    def draw(self, ax=None):
        hps = self.hparams
        graph = self.G.copy()
        for i, label in enumerate(["fix", "stim1", "stim2", "task"]):
            for a in self._stream_areas(i):
                graph.add_edge(label, a)
        for area_name in hps.rule_input_areas:
            graph.add_edge("rule", area_name)

        def color_rule(node_name):
            if "A" in node_name:
                if f"A{self.hparams.num_areas}" != node_name:
                    return "skyblue"
                return "limegreen"
            return "salmon"

        color_map = [color_rule(node) for node in graph.nodes]
        if not ax:
            _, ax = plt.subplots(1, 1, figsize=(4, 3))
        pos = nx.circular_layout(graph)
        nx.draw(
            graph,
            pos,
            with_labels=True,
            node_size=800,
            node_color=color_map,
            font_weight='bold',
            connectionstyle='arc3, rad = 0.1',
            ax=ax,
        )

    @staticmethod
    def get_insert_func_nested(arr):
        """Get insert/slice functions for nested indices."""
        arr_flat = []
        for item in arr:
            arr_flat += item

        insert_func, _, slice_func = get_insert_func(arr_flat, return_slice=True)

        def get_idx(idx1, idx2):
            sum_idx = 0
            for i in range(idx1):
                sum_idx += len(arr[i])
            sum_idx += len(arr[idx1][:idx2])
            return sum_idx

        def insert_wrap(tensor, data, idx1, idx2):
            converted_idx = get_idx(idx1, idx2)
            insert_func(tensor, data, converted_idx)

        def slice_wrap(tensor, idx1, idx2):
            converted_idx = get_idx(idx1, idx2)
            return slice_func(tensor, converted_idx)

        return insert_wrap, slice_wrap

    def _compute_ramp(self, start, increase):
        ramp = (self.current_epoch + 1 - start) / (increase + 1)
        return torch.clamp(torch.tensor(ramp), 0, 1)

    def _compute_scale_ramp(self, start, increase, init=0.0):
        """Epoch curriculum matching VariableNoise: ``init`` -> 1 over ``increase``.

        If ``increase <= 0``, returns 1 so the term is applied at full scale.
        """
        if increase <= 0:
            return torch.tensor(1.0)
        ramp = self._compute_ramp(start, increase)
        return init + (1.0 - init) * ramp

    def _comm_and_other_kernels(self):
        """Disjoint communication vs. non-communication parameter slices.

        Communication is the inter-area message rows of each area's output
        linear (same slice as L1). All other parameters, including fixation
        and task-output rows of those linears, are the weight-L2 group.
        ``end == -1`` means no task-output block, so comm is ``weight[start:]``.
        """
        hps = self.hparams
        comm_kernels = []
        other_kernels = []
        mixed_ids = set()

        for area_name, area in self.areas.items():
            start, end = hps.successor_list[area_name]
            w_end = None if end == -1 else end
            for layer in area.output.model:
                if not isinstance(layer, nn.Linear):
                    continue
                mixed_ids.add(id(layer.weight))
                comm_kernels.append(layer.weight[start:w_end])
                if start > 0:
                    other_kernels.append(layer.weight[:start])
                if w_end is not None:
                    other_kernels.append(layer.weight[w_end:])
                if layer.bias is not None:
                    mixed_ids.add(id(layer.bias))
                    comm_kernels.append(layer.bias[start:w_end])
                    if start > 0:
                        other_kernels.append(layer.bias[:start])
                    if w_end is not None:
                        other_kernels.append(layer.bias[w_end:])

        rule_ids = {id(p) for p in self.rule_decoders.parameters()} if hasattr(self, "rule_decoders") else set()
        for param in self.parameters():
            if id(param) not in mixed_ids and id(param) not in rule_ids:
                other_kernels.append(param)
        return comm_kernels, other_kernels

    @staticmethod
    def _mean_sq(kernels):
        total = 0.0
        n = 0
        for kernel in kernels:
            total = total + kernel.pow(2).sum()
            n += kernel.numel()
        return total / (n + 1e-8)
