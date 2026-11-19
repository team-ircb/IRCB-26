"""
Yang et al. (2019) task definitions in the Cognitive Task Suite trial format.

Selected with ``datamodule.task_variant: yang``. Definitions follow Yang's code
(gyyang/multitask @ 817e23d, ``task.py``), ``mode='random'``, with Yang's Δt = 20 ms per
step: every duration below is Yang's value in ms divided by 20, truncated like Yang's
``int(ms/dt)``. Line references are to that ``task.py``.

Trial format, unchanged from ``utils/cognitive_tasks.py``:

    fix                  (T, 1)  fixation input (1 = fixate)
    (stim1, amp1)        (T, 2), (T, 2)  modality 1: direction and strength of 2 slots
    (stim2, amp2)        same for modality 2
    resp                 (T, 1)  target direction during the response, 0 elsewhere
                                 (0 is the "no response" code used by the loss)
    sacc                 (T, 1)  saccade/fixation target (1 = keep fixating)

plus an ``info`` dict, returned as a sixth element:

    go    step at which the response is cued: fixation offset, or stimulus onset (RT
          tasks), or stimulus-2 onset (match tasks; also on no-go trials)
    end   first padding step: the trial is scored at end - 1 and no loss is given from end on
    cond  trial condition for condition-level sampling and scoring:
          context tasks: 0 = modalities agree, 1 = conflict;
          match tasks:   0 = match, 1 = non-match; all other tasks: 0

Departures from Yang's implementation:
  * Fixed T = 200 steps (4 s at 20 ms). As in Yang, a trial starts at step 0 and ends
    ``resp_dur`` steps after the go cue (500 ms, or Yang's DMC values); ``info['end']`` is
    that step. Steps from ``end`` to T are padding: the inputs and the response target keep
    their end-of-trial values there; with ``tail: mask`` the model gives them no loss, and it
    scores each trial at step ``end - 1``, which is Yang's last step. (Right-aligning trials instead would put every go cue at the same step, so a
    network could respond on a clock.) The only Yang trials longer than T are Dly DM-family
    trials (up to 210 steps); for those the initial fixation is shortened, never below
    Yang's 5 unscored steps.
  * Timings are drawn per trial (Yang: once per single-task batch), because batches mix tasks.
  * A stimulus occupies one of two slots per modality; a single stimulus goes in a random
    slot, two alternatives in random order. The data module encodes each slot as
    (s cos θ, s sin θ). Yang instead adds Gaussian bumps on a 32-unit ring per modality.
  * Input noise is added by the data module, not here.
"""

import numpy as np

from utils.common_utils import normalize

pi = np.pi
N_SLOTS = 2
DT = 20  # ms per step (Yang's dt)

# Condition codes
AGREE, CONFLICT = 0, 1
# Sampling-only condition for the context tasks: a conflict trial on which the uncued modality has
# the larger coherence, so adding the modalities always picks the wrong answer. Such trials are
# drawn from the task's own distribution (rejection sampling) and scored as CONFLICT.
CONFLICT_UNCUED_STRONGER = 2
MATCH, NONMATCH = 0, 1


def steps(ms):
    """Yang's int(ms/dt) conversion."""
    return int(ms / DT)


def u(lo, hi):
    """Yang draws continuous uniform times in ms, then truncates to steps."""
    return steps(np.random.uniform(lo, hi))


def choice(vals):
    return steps(float(np.random.choice(vals)))


def rand_dir():
    """Direction U[0, 2π), normalised to (-π, π]; never exactly 0 (0 codes 'no response')."""
    th = float(normalize(np.random.uniform(0, 2 * pi)))
    return th if th != 0.0 else 1e-6


class YangTask:
    """Parent class. Subclasses build a trial on a local time axis of length ``L`` (Yang's
    trial), which ``_place`` right-aligns into the ``T``-step trial."""

    MIN_LEAD = 5  # Yang's first 100 ms (pre_on), which the cost mask never scores

    def __init__(self, hparams: dict, T: int = 200):
        self.hps = dict(hparams)
        self.T = T

    # --- trial assembly ---------------------------------------------------------------
    def _new(self):
        T = self.T
        self.fix = np.ones((T, 1))
        self.stim = [np.zeros((T, N_SLOTS)), np.zeros((T, N_SLOTS))]
        self.amp = [np.zeros((T, N_SLOTS)), np.zeros((T, N_SLOTS))]
        self.resp = np.zeros((T, 1))
        self.sacc = np.ones((T, 1))

    def _offset(self, length, first_on):
        """Shift (<= 0) applied to every event time so a trial of ``length`` steps fits in
        T: only the initial fixation (before ``first_on``) is shortened."""
        off = min(0, self.T - length)
        if first_on + off < self.MIN_LEAD:
            raise ValueError(f"Yang trial of {length} steps does not fit in T={self.T}")
        self.end = length + off
        return off

    def _pulse(self, mod, slot, theta, strength, start, end):
        self.stim[mod][start:end, slot] = normalize(theta)
        self.amp[mod][start:end, slot] = float(strength)

    def _go(self, go, theta, fixation_off=True, respond=True):
        """Cue the response at ``go``: fixation input off (unless an RT/match task),
        saccade target 0 and response direction from ``go`` on, or keep fixating."""
        if fixation_off:
            self.fix[go:] = 0.0
        if respond:
            self.sacc[go:] = 0.0
            th = float(normalize(theta))
            self.resp[go:, 0] = th if th != 0.0 else 1e-6

    def _out(self, go, cond=0):
        assert 0 < go < self.end <= self.T
        return (
            self.fix,
            (self.stim[0], self.amp[0]),
            (self.stim[1], self.amp[1]),
            self.resp,
            self.sacc,
            {"go": int(go), "cond": int(cond), "end": int(self.end)},
        )

    def gen_single_trial(self, cond=None):
        raise NotImplementedError


class Go(YangTask):
    """Go, RT Go, Dly Go and their Anti versions (task.py:213-302, :473-653)."""

    def gen_single_trial(self, cond=None):
        self._new()
        timing, anti = self.hps["timing"], self.hps["anti"]
        theta = rand_dir()
        mod = np.random.randint(2)
        slot = np.random.randint(N_SLOTS)
        resp_dur = steps(500)
        if timing == "go":  # fdgo :593-596
            stim_on = u(300, 700)
            go = stim_on + u(500, 1500)
            L = go + resp_dur
            o = self._offset(L, stim_on)
            self._pulse(mod, slot, theta, 1.0, o + stim_on, self.T)  # stays on (:636)
            self._go(o + go, theta + pi * anti)
        elif timing == "rt":  # reactgo :498-499; fixation input on all trial (:540)
            stim_on = u(500, 2500)
            L = stim_on + resp_dur
            o = self._offset(L, stim_on)
            self._pulse(mod, slot, theta, 1.0, o + stim_on, self.T)
            go = stim_on
            self._go(o + go, theta + pi * anti, fixation_off=False)
        elif timing == "dly":  # delaygo :241-246
            stim_on = choice([300, 500, 700])
            stim_dur = choice([200, 400, 600])
            delay = choice([200, 400, 800, 1600])
            go = stim_on + stim_dur + delay
            L = go + resp_dur
            o = self._offset(L, stim_on)
            self._pulse(mod, slot, theta, 1.0, o + stim_on, o + stim_on + stim_dur)
            self._go(o + go, theta + pi * anti)
        else:
            raise ValueError(timing)
        return self._out(o + go)


def genstim(cohs):
    """Yang's strength pair: mean U(0.8, 1.2) ± coherence·sign (task.py:305-313)."""
    mean = np.random.uniform(0.8, 1.2)
    coh = float(np.random.choice(cohs))
    sign = float(np.random.choice([-1.0, 1.0]))
    return mean + coh * sign, mean - coh * sign, sign


class Decision(YangTask):
    """DM 1/2, Ctx DM 1/2, MultSen DM and their delayed versions.

    hps: ``delayed`` (bool), ``rule`` in {"mod1", "mod2", "ctx1", "ctx2", "multi"}.
      mod1/mod2: two stimuli in that modality only (dm, delaydm; :660-899)
      ctx1/ctx2: same two directions in both modalities, each modality's strengths drawn
                 independently; the cued modality decides (:316-466, :902-1059)
      multi:     one strength pair, each stimulus's strength split across modalities;
                 the summed strength decides (:362-375, :950-964)
    """

    COHS = [0.01, 0.02, 0.04, 0.08]
    COHS_DELAY = [0.08, 0.16, 0.32]

    def gen_single_trial(self, cond=None):
        self._new()
        rule, delayed = self.hps["rule"], self.hps["delayed"]
        cohs = self.COHS_DELAY if delayed else self.COHS
        th1 = rand_dir()
        th2 = float(normalize(th1 + np.random.uniform(0.5 * pi, 1.5 * pi) * np.random.choice([-1, 1])))
        if th2 == 0.0:
            th2 = 1e-6

        # strengths[mod] = (s for stimulus 1, s for stimulus 2); None = modality silent
        strengths = [None, None]
        trial_cond = 0
        if rule in ("mod1", "mod2"):
            s1, s2, _ = genstim(cohs)
            strengths[0 if rule == "mod1" else 1] = (s1, s2)
            target = th1 if s1 > s2 else th2
        elif rule in ("ctx1", "ctx2"):
            for _ in range(1000):
                a1, a2, sg_a = genstim(cohs)
                b1, b2, sg_b = genstim(cohs)
                trial_cond = AGREE if sg_a == sg_b else CONFLICT
                if cond == CONFLICT_UNCUED_STRONGER:
                    coh_cued, coh_unc = (abs(a1 - a2), abs(b1 - b2)) if rule == "ctx1" else (abs(b1 - b2), abs(a1 - a2))
                    if trial_cond == CONFLICT and coh_unc > coh_cued + 1e-9:
                        break
                    continue
                if cond is None or cond == trial_cond:
                    break
            strengths = [(a1, a2), (b1, b2)]
            att = strengths[0] if rule == "ctx1" else strengths[1]
            target = th1 if att[0] > att[1] else th2
        elif rule == "multi":
            s1, s2, _ = genstim(cohs)
            d1 = s1 * np.random.uniform(0.2, 0.8) * np.random.choice([1, -1])
            d2 = s2 * np.random.uniform(0.2, 0.8) * np.random.choice([1, -1])
            strengths = [(s1 + d1 / 2, s2 + d2 / 2), (s1 - d1 / 2, s2 - d2 / 2)]
            target = th1 if s1 > s2 else th2
        else:
            raise ValueError(rule)

        resp_dur = steps(500)
        # Which slot carries stimulus 1 / 2. ONE permutation per trial, shared by both modalities:
        # in Ctx DM, MultSen DM and their delayed versions this shared order (not the direction)
        # is what tells the network which modality-1 stimulus goes with which modality-2
        # stimulus. Load-bearing: drawing it per modality would pose a cross-modal binding
        # problem that Yang's ring never poses.
        slots = np.random.permutation(N_SLOTS)
        if not delayed:  # dm :707-713, stimuli end at the go cue, no delay
            stim_on = u(100, 400)
            stim_dur = choice([400, 800, 1600])
            go = stim_on + stim_dur
            L = go + resp_dur
            o = self._offset(L, stim_on)
            spans = [(o + stim_on, o + go), (o + stim_on, o + go)]
        else:  # delaydm :822-833
            s1_on = choice([200, 400, 600])
            s1_dur = choice([200, 400, 600])
            delay = choice([200, 400, 800, 1600])
            s2_dur = choice([200, 400, 600])
            s2_on = s1_on + s1_dur + delay
            go = s2_on + s2_dur + u(100, 300)
            L = go + resp_dur
            o = self._offset(L, s1_on)
            spans = [(o + s1_on, o + s1_on + s1_dur), (o + s2_on, o + s2_on + s2_dur)]

        for mod in range(2):
            if strengths[mod] is None:
                continue
            for k, (theta, (start, end)) in enumerate(zip((th1, th2), spans)):
                self._pulse(mod, slots[k], theta, strengths[mod][k], start, end)
        self._go(o + go, target)
        return self._out(o + go, trial_cond)


class Match(YangTask):
    """DMS, DNMS, DMC, DNMC (task.py:1066-1320).

    Fixation input stays on all trial; stimulus-2 onset cues the response. Stimulus 2
    stays on to the end. The two stimuli's modalities are drawn independently.
    hps: ``cond`` in {"point", "category"}, ``nogo`` (bool: respond on non-match).
    """

    CATS = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9]) * pi

    @staticmethod
    def category(theta):
        return float(np.remainder(theta, 2 * pi)) < pi  # boundary at 0 / 180 deg (:1280-1282)

    def gen_single_trial(self, cond=None):
        self._new()
        if self.hps["cond"] == "point":  # :1104-1108
            th1 = rand_dir()
            match = (np.random.randint(2) == 0) if cond is None else (cond == MATCH)
            if match:
                th2 = th1
            else:
                th2 = float(normalize(th1 + np.random.uniform(pi / 9, 17 * pi / 9)))
            resp_dur = steps(500)
        else:  # :1231-1238, 10 discrete directions, independent draws
            th1 = float(np.random.choice(self.CATS))
            if cond is None:
                th2 = float(np.random.choice(self.CATS))
            else:
                same = [a for a in self.CATS if self.category(a) == self.category(th1)]
                other = [a for a in self.CATS if self.category(a) != self.category(th1)]
                th2 = float(np.random.choice(same if cond == MATCH else other))
            match = self.category(th1) == self.category(th2)
            resp_dur = choice([200, 400, 600])
        th1 = float(normalize(th1)) or 1e-6
        th2 = float(normalize(th2)) or 1e-6

        s1_on = choice([200, 400, 600])
        s1_dur = choice([200, 400, 600])
        delay = choice([200, 400, 800, 1600])
        s2_on = s1_on + s1_dur + delay
        L = s2_on + resp_dur
        o = self._offset(L, s1_on)
        m1, m2 = np.random.randint(2), np.random.randint(2)
        self._pulse(m1, np.random.randint(N_SLOTS), th1, 1.0, o + s1_on, o + s1_on + s1_dur)
        self._pulse(m2, np.random.randint(N_SLOTS), th2, 1.0, o + s2_on, self.T)

        go_trial = (not match) if self.hps["nogo"] else match
        self._go(o + s2_on, th2, fixation_off=False, respond=go_trial)
        return self._out(o + s2_on, MATCH if match else NONMATCH)


# Yang's task key -> generator (the single-task suite's name for the task in the comment).
# Order = Yang's rule order.
task_map = {
    "fdgo": [Go, {"timing": "go", "anti": False}],  # suite: fd_go
    "reactgo": [Go, {"timing": "rt", "anti": False}],  # suite: rt_go
    "delaygo": [Go, {"timing": "dly", "anti": False}],  # suite: dly_go
    "fdanti": [Go, {"timing": "go", "anti": True}],  # suite: fd_go_anti
    "reactanti": [Go, {"timing": "rt", "anti": True}],  # suite: rt_go_anti
    "delayanti": [Go, {"timing": "dly", "anti": True}],  # suite: dly_go_anti
    "dm1": [Decision, {"delayed": False, "rule": "mod1"}],  # suite: dm_1
    "dm2": [Decision, {"delayed": False, "rule": "mod2"}],  # suite: dm_2
    "contextdm1": [Decision, {"delayed": False, "rule": "ctx1"}],  # suite: ctxt_dm_1
    "contextdm2": [Decision, {"delayed": False, "rule": "ctx2"}],  # suite: ctxt_dm_2
    "multidm": [Decision, {"delayed": False, "rule": "multi"}],  # suite: ctxt_dm_max
    "delaydm1": [Decision, {"delayed": True, "rule": "mod1"}],  # suite: dly_dm_mod_1
    "delaydm2": [Decision, {"delayed": True, "rule": "mod2"}],  # suite: dly_dm_mod_2
    "contextdelaydm1": [Decision, {"delayed": True, "rule": "ctx1"}],  # suite: dly_dm_1
    "contextdelaydm2": [Decision, {"delayed": True, "rule": "ctx2"}],  # suite: dly_dm_2
    "multidelaydm": [Decision, {"delayed": True, "rule": "multi"}],  # suite: dly_dm_max
    "dmsgo": [Match, {"cond": "point", "nogo": False}],  # suite: dms
    "dmsnogo": [Match, {"cond": "point", "nogo": True}],  # suite: dms_nogo
    "dmcgo": [Match, {"cond": "category", "nogo": False}],  # suite: dmc
    "dmcnogo": [Match, {"cond": "category", "nogo": True}],  # suite: dmc_nogo
}

# Tasks whose trials split into two conditions (for condition-level sampling and scoring)
CONDITION_NAMES = {
    **{t: ("agree", "conflict") for t in ["contextdm1", "contextdm2", "contextdelaydm1", "contextdelaydm2"]},
    **{t: ("match", "nonmatch") for t in ["dmsgo", "dmsnogo", "dmcgo", "dmcnogo"]},
}
# Conditions that cond_weights may request (sampling only; scoring and the adaptive sampler use
# CONDITION_NAMES): the context tasks add "conflict_uncued_stronger" (code 2, scored as conflict).
SAMPLING_CONDITIONS = {**CONDITION_NAMES,
                       **{t: CONDITION_NAMES[t] + ("conflict_uncued_stronger",)
                          for t in ["contextdm1", "contextdm2", "contextdelaydm1", "contextdelaydm2"]}}

# Yang et al. (2019) task keys (task.py:1512-1534) and display names (task.py:1536-1557), in
# Yang's rule order, with the single-task suite's names for them (utils/cognitive_tasks.py).
# Note that the suite's dly_dm_1/2 are Yang's *context* delayed DM tasks, and *_nogo are the non-match
# (DNMS/DNMC) versions, not no-go tasks.
YANG_TO_SUITE = {
    "fdgo": "fd_go",
    "reactgo": "rt_go",
    "delaygo": "dly_go",
    "fdanti": "fd_go_anti",
    "reactanti": "rt_go_anti",
    "delayanti": "dly_go_anti",
    "dm1": "dm_1",
    "dm2": "dm_2",
    "contextdm1": "ctxt_dm_1",
    "contextdm2": "ctxt_dm_2",
    "multidm": "ctxt_dm_max",
    "delaydm1": "dly_dm_mod_1",
    "delaydm2": "dly_dm_mod_2",
    "contextdelaydm1": "dly_dm_1",
    "contextdelaydm2": "dly_dm_2",
    "multidelaydm": "dly_dm_max",
    "dmsgo": "dms",
    "dmsnogo": "dms_nogo",
    "dmcgo": "dmc",
    "dmcnogo": "dmc_nogo",
}
SUITE_TO_YANG = {v: k for k, v in YANG_TO_SUITE.items()}
DISPLAY_NAMES = {
    "fdgo": "Go",
    "reactgo": "RT Go",
    "delaygo": "Dly Go",
    "fdanti": "Anti",
    "reactanti": "RT Anti",
    "delayanti": "Dly Anti",
    "dm1": "DM 1",
    "dm2": "DM 2",
    "contextdm1": "Ctx DM 1",
    "contextdm2": "Ctx DM 2",
    "multidm": "MultSen DM",
    "delaydm1": "Dly DM 1",
    "delaydm2": "Dly DM 2",
    "contextdelaydm1": "Ctx Dly DM 1",
    "contextdelaydm2": "Ctx Dly DM 2",
    "multidelaydm": "MultSen Dly DM",
    "dmsgo": "DMS",
    "dmsnogo": "DNMS",
    "dmcgo": "DMC",
    "dmcnogo": "DNMC",
}
