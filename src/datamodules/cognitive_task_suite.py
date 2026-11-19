"""Trials for the Cognitive Task Suite (the single-task suite and Yang et al. 2019's 20 tasks)."""

import json
import os

import numpy as np

from utils import yang_tasks
from utils.cognitive_tasks import task_map
from utils.common_utils import normalize

from .base import BasicDataset, DGNDataModuleBase

# encoding='ring': units per stimulus ring and their preferred directions (Yang et al. 2019: n_eachring = 32)
N_RING_IN = 32
RING_PREF = np.arange(N_RING_IN) * (2 * np.pi / N_RING_IN)


class CognitiveTaskSuite(DGNDataModuleBase):
    """
    DataModule for generating Cognitive Task Suite data, used with the `CognitiveTaskSuite` model.
    """
    def __init__(
        self,
        task_names: list,
        batch_total: int,
        time_total: int,
        p_split: list = [0.8, 0.2],
        batch_size: int = 64,
        train_type: str = "random",
        train_type_kwargs: dict = {},
        dm_seed: int = 0,
        noise_sig: float = 0.0,
        resultpath: str = ".",
        num_workers: int = 0,
        resample_train: bool = False,
        task_variant: str = "suite",
        encoding: str = "raw",
        input_noise: float = 0.0,
        fix_noise: float = None,
    ):
        """
        Args:
            task_names: List of task names.
            batch_total: Total number of batches (trials).
            time_total: Total number of time steps.
            p_split: Train/validation split percentage, must sum up to 1.
            batch_size: Size of each training batch. Validation batch size is always
                the total number of validation batches.
            train_type: Type of training. Options are:
                'random', 'batch_uniform', 'curriculum_ratio'.
            train_type_kwargs: Keyword arguments for the training type.
            dm_seed: Seed for the data module.
            noise_sig: Standard deviation for noise.
            resultpath: Path to save the results.
            num_workers: DataLoader worker processes. The trials are generated in memory
                up front, so workers only add start-up cost (large on macOS, where each
                worker is a fresh process every epoch). 0 loads in the main process.
            resample_train: If True, draw a fresh training set (new trials, reshuffled
                task order) every epoch, as Yang et al. (2019) did, instead of reusing
                one fixed set. Validation trials stay fixed. Needs
                ``trainer.reload_dataloaders_every_n_epochs=1``.
            task_variant: 'suite' (``utils/cognitive_tasks.py``, the single-task suite's tasks) or
                'yang' (``utils/yang_tasks.py``, Yang et al. 2019's definitions at 20 ms
                per step).
            encoding: 'raw' (angle in radians and strength per slot) or
                'cartesian' (s*cos(theta) in the ``stim`` arrays and s*sin(theta) in the
                ``amp`` arrays; a silent slot is (0, 0)). The model concatenates the two,
                so either way each modality is 4 inputs. Or 'ring': Yang et al. (2019)'s
                encoding, a 32-unit ring per modality (preferred directions 2*pi*k/32), each
                stimulus adding s * 0.8 * exp(-d^2 / 2) with d = circular distance / (pi/8)
                (Yang's add_x_loc); the ring is in the ``stim`` arrays and the ``amp`` arrays
                are empty, so each modality is 32 inputs (set the model's stim_channels=32).
            input_noise: SD of Gaussian noise added independently at every step to the
                fixation input and every stimulus channel (after encoding), as Yang et al.
                do. Used instead of ``noise_sig`` when > 0. The rule input's noise is the
                model's ``rule_noise``.
            fix_noise: SD of the input noise on the fixation channel, if it should differ
                from ``input_noise`` (None = same as ``input_noise``). Only used when
                ``input_noise`` > 0.
        """
        super().__init__()
        self.save_hyperparameters()
        self.current_epoch = 0
        self.sampler_log = []
        
    def setup(self, stage=None):
        """
        Setup the dataset and split into train and validation sets.
        """
        hps = self.hparams
        
        # Construct all "Task" classes
        tasks = []
        tmap = yang_tasks.task_map if hps.task_variant == "yang" else task_map
        if hps.task_variant not in ("suite", "yang"):
            raise ValueError(f"task_variant must be 'suite' or 'yang', got {hps.task_variant!r}")
        if hps.encoding not in ("raw", "cartesian", "ring"):
            raise ValueError(f"encoding must be 'raw', 'cartesian' or 'ring', got {hps.encoding!r}")
        for task_name in hps.task_names:
            task_class, task_config = tmap[task_name]
            task = task_class(task_config, hps.time_total)
            tasks.append(task)
            if hps.task_variant == "suite":
                try:
                    task.draw(save=True, figname=task_name, n_batches=4)
                except:
                    print('Task draw() failed. Continuing...')
        self.tasks = tasks
            
        # Generate train and validation sizes
        batch_train = int(hps.batch_total * hps.p_split[0])
        batch_val = hps.batch_total - batch_train
        
        # Generate validation set (random order)
        self.val_batch_order = self.random_order(batch_val)
        self.val_ds = self.gen_data(tasks, self.val_batch_order)
        
        # Generate train set
        self.gen_train_ds(batch_train)
        
    def gen_train_ds(self, batch_train):
        """
        Generate the train dataset according to the train type.
        """
        hps = self.hparams
        if hps.train_type == "random": # random order
            train_batch_order = self.random_order(batch_train)
        elif hps.train_type == "batch_uniform": # single task per batch
            train_batch_order = self.single_task_per_batch_order(batch_train, **hps.train_type_kwargs,)
        elif hps.train_type == "curriculum_ratio": # task ratio changes over time
            train_batch_order = self.curriculum_ratio_order(batch_train, **hps.train_type_kwargs,)
        elif hps.train_type == "weighted": # fixed relative task frequencies, interleaved
            train_batch_order = self.weighted_order(batch_train, **hps.train_type_kwargs,)
            if isinstance(train_batch_order, tuple):  # cond_weights given: per-trial conditions too
                train_batch_order, cond_order = train_batch_order
                self.train_batch_order = train_batch_order
                self.train_ds = self.gen_data(self.tasks, train_batch_order, cond_order)
                return
        elif hps.train_type == "adaptive": # task x condition shares from validation perf
            train_batch_order, cond_order = self.adaptive_order(batch_train, **hps.train_type_kwargs,)
            self.train_batch_order = train_batch_order
            self.train_ds = self.gen_data(self.tasks, train_batch_order, cond_order)
            return
        else:
            raise ValueError(f"Unknown train_type: {hps.train_type}")
            
        # Generate train dataset
        self.train_batch_order = train_batch_order
        self.train_ds = self.gen_data(self.tasks, self.train_batch_order)
        
    def gen_data(self, tasks, batch_order, cond_order=None):
        """
        Generate data for a given batch order. ``cond_order`` (Yang tasks only) requests a
        trial condition per trial (-1 = any; see ``utils/yang_tasks.py``).

        Returns a dataset of 11 arrays, each (batch, T, ·): fix, stim1, amp1, stim2, amp2,
        task, resp, sacc, go, cond, end. ``go`` is the step at which the response is cued,
        ``cond`` the trial condition and ``end`` the first padding step (T if none), each
        tiled over time. For the 'suite' variant, go = first step with saccade target 0 (T if
        none), cond = 0 and end = T.
        """
        hps = self.hparams

        def gen_noise(arr, mag=1):
            return np.random.normal(0, 1, arr.shape) * mag * hps.noise_sig

        batch_size = len(batch_order)

        # Two directional alternatives per modality: angle (T, 2) and strength (T, 2).
        n_alt = 2
        ring = hps.encoding == "ring"
        n_stim, n_amp = (N_RING_IN, 0) if ring else (n_alt, n_alt)
        fixs = np.zeros((batch_size, hps.time_total, 1))
        stim1s = np.zeros((batch_size, hps.time_total, n_stim))
        stim2s = np.zeros((batch_size, hps.time_total, n_stim))
        resps = np.zeros((batch_size, hps.time_total, 1))
        saccs = np.zeros((batch_size, hps.time_total, 1))
        amp1s = np.zeros((batch_size, hps.time_total, n_amp))
        amp2s = np.zeros((batch_size, hps.time_total, n_amp))

        def noisy_angles(stim, amp):
            """Jitter active directions. Silent slots (strength 0) stay at 0."""
            jittered = normalize(stim + gen_noise(stim, mag=0.1))
            return np.where(amp > 0, jittered, 0.0)

        T = hps.time_total
        gos = np.full(batch_size, T, dtype=float)
        conds = np.zeros(batch_size, dtype=float)
        ends = np.full(batch_size, T, dtype=float)
        sig_in = float(hps.get("input_noise", 0.0) or 0.0)
        sig_fix = hps.get("fix_noise", None)
        sig_fix = sig_in if sig_fix is None else float(sig_fix)

        def in_noise(arr, sig=None):
            return np.random.normal(0, 1, arr.shape) * (sig_in if sig is None else sig)

        # Noise-free task variables per slot (angle in rad, strength), for dataset writers
        # (scripts/generate_multitask_datasets.py); not part of the training tuple.
        clean = {k: np.zeros((batch_size, T, n_alt)) for k in ("theta1", "strength1", "theta2", "strength2")}
        self._last_clean = clean

        def encode(stim, amp):
            if hps.encoding == "cartesian":
                return amp * np.cos(stim), amp * np.sin(stim)
            if ring:  # Yang's add_x_loc per slot, summed; silent slots (strength 0) add nothing
                d = np.abs(np.angle(np.exp(1j * (stim[..., None] - RING_PREF))))  # (T, slots, 32)
                bumps = 0.8 * np.exp(-((d / (np.pi / 8)) ** 2) / 2)
                return (amp[..., None] * bumps).sum(1), np.zeros((stim.shape[0], 0))
            return stim, amp

        for b, tpe in enumerate(batch_order):
            if hps.task_variant == "yang":
                want = None if cond_order is None or cond_order[b] < 0 else int(cond_order[b])
                fix, (stim1, amp1), (stim2, amp2), resp, sacc, tinfo = tasks[int(tpe)].gen_single_trial(cond=want)
                gos[b], conds[b], ends[b] = tinfo["go"], tinfo["cond"], tinfo["end"]
            else:
                fix, (stim1, amp1), (stim2, amp2), resp, sacc = tasks[int(tpe)].gen_single_trial()
                off = np.where(sacc[:, 0] < 0.5)[0]
                gos[b] = off[0] if len(off) else T
            if sig_in > 0:
                x1, y1 = encode(stim1, amp1)
                x2, y2 = encode(stim2, amp2)
                fixs[b, :] = fix + in_noise(fix, sig_fix)
                stim1s[b, :], amp1s[b, :] = x1 + in_noise(x1), y1 + in_noise(y1)
                stim2s[b, :], amp2s[b, :] = x2 + in_noise(x2), y2 + in_noise(y2)
            else:
                fixs[b, :] = fix + gen_noise(fix)
                if hps.encoding in ("cartesian", "ring"):
                    stim1s[b, :], amp1s[b, :] = encode(stim1, amp1)
                    stim2s[b, :], amp2s[b, :] = encode(stim2, amp2)
                else:
                    stim1s[b, :] = noisy_angles(stim1, amp1)
                    stim2s[b, :] = noisy_angles(stim2, amp2)
                    amp1s[b, :] = amp1
                    amp2s[b, :] = amp2
            resps[b, :] = resp
            saccs[b, :] = sacc
            clean["theta1"][b], clean["strength1"][b] = stim1, amp1
            clean["theta2"][b], clean["strength2"][b] = stim2, amp2

        # Create task indices and return dataset
        def tile(v):
            return np.tile(np.asarray(v, dtype=float).reshape(-1, 1, 1), (1, T, 1))

        task_idxs = tile(batch_order)
        return BasicDataset(fixs, stim1s, amp1s, stim2s, amp2s, task_idxs, resps, saccs,
                            tile(gos), tile(conds), tile(ends))
            
    def random_order(self, batch_size, **kwargs):
        """
        Repeated shuffled permutations of all task ids.
        """
        hps = self.hparams
        num_task = len(hps.task_names)
        task_idxs = np.arange(num_task).astype(int)
        
        seed = hps.dm_seed + (self.current_epoch if hps.get("resample_train", False) else 0)
        rng = np.random.default_rng(seed)
        num_mini_batch = int(np.ceil(batch_size/num_task))
        batch = np.zeros(num_mini_batch * num_task)
        
        # Shuffle task indices for each mini-batch
        for b in range(num_mini_batch):
            rng.shuffle(task_idxs)
            batch[b * num_task: (b+1) * num_task] = task_idxs
        
        return batch[:batch_size]
    
    def weighted_order(self, batch_size, weights=None, cond_weights=None, **kwargs):
        """
        Interleaved task ids with fixed relative frequencies, e.g. ``weights={contextdm1: 5,
        contextdm2: 5}`` (unlisted tasks weigh 1). Yang et al. (2019) sample Ctx DM 1 and 2
        five times as often as the other tasks, because otherwise networks settle on
        integrating both modalities (~75% correct). Driscoll et al. (2024) say they do the
        same, but their code samples tasks uniformly. Counts are
        exact (largest-remainder rounding) and the order is shuffled, reshuffled each epoch
        when ``resample_train`` is set.

        ``cond_weights`` (Yang tasks only; optional) sets relative condition frequencies within
        a task, e.g. ``{contextdm1: {agree: 1, conflict: 3}}`` (condition names from
        ``yang_tasks.SAMPLING_CONDITIONS``; unlisted conditions weigh 1, except the sampling-only
        ``conflict_uncued_stronger``, which weighs 0 unless listed). It changes which trials a
        task's share is spent on, not the share. Without it, conditions follow each task's own
        statistics (agree / conflict 50:50) and the result is unchanged from before; with it,
        returns (task order, condition order), -1 = any condition.
        """
        hps = self.hparams
        weights = dict(weights or {})
        unknown = set(weights) - set(hps.task_names)
        if unknown:
            raise ValueError(f"weights for unknown tasks: {sorted(unknown)}")
        w = np.array([float(weights.get(t, 1.0)) for t in hps.task_names])
        exact = batch_size * w / w.sum()
        counts = np.floor(exact).astype(int)
        for i in np.argsort(-(exact - counts))[: batch_size - counts.sum()]:
            counts[i] += 1
        order = np.repeat(np.arange(len(w)), counts).astype(float)
        seed = hps.dm_seed + (self.current_epoch if hps.get("resample_train", False) else 0)
        np.random.default_rng(seed).shuffle(order)
        if not cond_weights:
            return order
        if hps.task_variant != "yang":
            raise ValueError("cond_weights needs task_variant='yang' (trial conditions)")
        cond_weights = {t: dict(v) for t, v in dict(cond_weights).items()}
        bad = set(cond_weights) - set(yang_tasks.SAMPLING_CONDITIONS)
        if bad:
            raise ValueError(f"cond_weights for tasks without conditions: {sorted(bad)}")
        cond_order = np.full(len(order), -1.0)
        rng = np.random.default_rng(seed + 7919)  # separate stream: the task order is unchanged
        for t, cw in cond_weights.items():
            if t not in hps.task_names:
                continue
            names = yang_tasks.SAMPLING_CONDITIONS[t]
            if set(cw) - set(names):
                raise ValueError(f"cond_weights[{t}]: unknown conditions {sorted(set(cw) - set(names))}")
            pos = np.flatnonzero(order == list(hps.task_names).index(t))
            cwv = np.array([float(cw.get(c, 0.0 if c == "conflict_uncued_stronger" else 1.0)) for c in names])
            ex = len(pos) * cwv / cwv.sum()
            cnt = np.floor(ex).astype(int)
            for i in np.argsort(-(ex - cnt))[: len(pos) - cnt.sum()]:
                cnt[i] += 1
            cond_order[pos] = rng.permutation(np.repeat(np.arange(len(names)), cnt))
        return order, cond_order

    def adaptive_order(self, batch_size, target=0.9, floor=0.5, ema=0.5,
                       metric="valid/perfstrict", **kwargs):
        """
        Task x condition training mix set each epoch from validation performance
        (group weighting as in Sagawa et al. 2020).

        Groups: every task; the context tasks (agree / conflict) and match tasks (match /
        non-match) split by condition (``yang_tasks.CONDITION_NAMES``). A group's raw weight
        is max(0, target - perf_g), with perf_g the model's logged ``<metric>_<task>[_<cond>]``
        from the last validation epoch. Shares = floor/G + (1 - floor) * w_g / sum(w)
        (uniform when every group is at target), smoothed across epochs:
        share <- ema * share_prev + (1 - ema) * share_new. Before any validation result,
        shares are uniform over groups. Counts use largest-remainder rounding; order shuffled.
        Each epoch's shares are appended to ``sampler_weights.jsonl`` in the run directory.
        """
        hps = self.hparams
        if hps.task_variant != "yang":
            raise ValueError("adaptive sampling needs task_variant='yang' (trial conditions)")
        groups = []
        for it, t in enumerate(hps.task_names):
            cn = yang_tasks.CONDITION_NAMES.get(t)
            if cn:
                groups += [(it, ic, f"{t}_{cn[ic]}") for ic in range(len(cn))]
            else:
                groups.append((it, -1, t))
        G = len(groups)
        metrics = {}
        trainer = getattr(self, "trainer", None)
        if trainer is not None:
            metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
        perf = np.array([metrics.get(f"{metric}_{name}", np.nan) for _, _, name in groups])
        if np.all(np.isnan(perf)):
            new = np.full(G, 1.0 / G)
        else:
            w = np.maximum(0.0, target - np.nan_to_num(perf, nan=0.0))
            new = np.full(G, floor / G)
            new += (1 - floor) * (w / w.sum() if w.sum() > 0 else np.full(G, 1.0 / G))
        prev = getattr(self, "_adaptive_shares", None)
        shares = new if prev is None else ema * prev + (1 - ema) * new
        shares = shares / shares.sum()
        self._adaptive_shares = shares

        exact = batch_size * shares
        counts = np.floor(exact).astype(int)
        for i in np.argsort(-(exact - counts))[: batch_size - counts.sum()]:
            counts[i] += 1
        task_order = np.repeat([g[0] for g in groups], counts).astype(float)
        cond_order = np.repeat([g[1] for g in groups], counts).astype(float)
        seed = hps.dm_seed + (self.current_epoch if hps.get("resample_train", False) else 0)
        perm = np.random.default_rng(seed).permutation(batch_size)
        rec = {"epoch": int(self.current_epoch),
               "shares": {g[2]: float(sh) for g, sh in zip(groups, shares)},
               "perf": {g[2]: (None if np.isnan(p) else float(p)) for g, p in zip(groups, perf)}}
        try:
            with open(os.path.join(os.getcwd(), "sampler_weights.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass
        return task_order[perm], cond_order[perm]

    def single_task_per_batch_order(self, batch_total, **kwargs):
        """
        Long runs of one task (``batch_size * persist`` samples), then the next task, repeating.
        """
        defaults = {'persist': 1}
        defaults.update(kwargs)
        persist = defaults['persist']
        
        hps = self.hparams
        num_task = len(hps.task_names)
        task_idxs = np.arange(num_task).astype(int)
        
        # Tile task indices to create a base pattern
        base = np.tile(task_idxs.reshape(1, -1), (hps.batch_size * persist, 1)) # shape = (bs & persist, num_tasks)
        base = base.flatten(order='F') # [0, 0, 0... 1, 1, 1....,]
        num_repeats = int(np.ceil(batch_total / len(base)))
        return np.tile(base, num_repeats)[:batch_total]
    
    def curriculum_ratio_order(self, batch_total: int, **kwargs):
        """
        Per-block task draws with weights moving from uniform toward ``final_ratio`` as ``current_epoch`` increases.
        """
        defaults = {
            "persist": 1,
            "final_ratio": None,   # if None -> stays uniform
            "decay": 50.0,
        }
        defaults.update(kwargs)
        persist = int(defaults["persist"])
        final_ratio = defaults["final_ratio"]
        decay = float(defaults["decay"])

        hps = self.hparams
        num_task = len(hps.task_names)
        epoch = getattr(self, "current_epoch", 0)

        # Define initial (uniform) and target ratios
        init_ratio = np.ones(num_task, dtype=float) / num_task

        # Define target ratio and ensure it matches the number of tasks
        if final_ratio is None:
            target_ratio = init_ratio.copy()
        else:
            target_ratio = np.asarray(final_ratio, dtype=float)
            assert target_ratio.shape[0] == num_task, \
                f"final_ratio length {target_ratio.shape[0]} must match num_tasks={num_task}"
            target_ratio = target_ratio / target_ratio.sum()

        # alpha ~ 0 => uniform, alpha ~ 1 => target_ratio
        if decay <= 0:
            alpha = 1.0
        else:
            alpha = 1.0 - np.exp(-epoch / decay)
        alpha = float(np.clip(alpha, 0.0, 1.0))

        # Interpolate between initial and target ratios
        curr_ratio = (1.0 - alpha) * init_ratio + alpha * target_ratio
        curr_ratio = curr_ratio / curr_ratio.sum()

        # Sample block-wise tasks according to curr_ratio
        block_size = hps.batch_size * persist
        num_blocks = int(np.ceil(batch_total / block_size))

        # Make RNG depend on epoch so pattern changes across epochs
        rng = np.random.default_rng(hps.dm_seed + int(epoch))

        # Choose a task id for each block
        block_tasks = rng.choice(num_task, size=num_blocks, p=curr_ratio)
        batch = np.repeat(block_tasks, block_size)
        return batch[:batch_total]

    def train_dataloader(self, shuffle: bool = False):
        # Batch order is produced by train_type (curriculum, etc.)
        trainer = getattr(self, "trainer", None)
        epoch = int(trainer.current_epoch) if trainer is not None else 0
        if epoch != self.current_epoch:
            self.current_epoch = epoch  # lets curriculum_ratio advance
            if self.hparams.resample_train:
                self.gen_train_ds(len(self.train_ds))
        return super().train_dataloader(shuffle=shuffle)
