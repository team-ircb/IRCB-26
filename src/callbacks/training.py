"""Callbacks that control training: early stopping, acceptance stopping, and running the
per-epoch callbacks listed in a config."""

import os
from copy import deepcopy

import pytorch_lightning as pl

from .plots import SAVE_DIR


class DelayedEarlyStopping(pl.callbacks.EarlyStopping):
    """Early stopping that ignores the monitor until ``delay_epochs``.

    Patience does not tick during the delay, so a noise / loss curriculum can
    finish before stopping decisions begin.
    """

    def __init__(self, delay_epochs: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.delay_epochs = delay_epochs

    def _run_early_stopping_check(self, trainer):
        if trainer.current_epoch < self.delay_epochs:
            return
        return super()._run_early_stopping_check(trainer)


class AcceptanceStopping(pl.Callback):
    """Stop once every task passes: ``monitor`` (default the worst task's strict validation
    performance, -1 until the ramps end) >= ``threshold`` for ``patience`` consecutive
    validation epochs. Replaces loss-based early stopping for the multitask DGN, which can
    fire while one task still fails."""

    def __init__(self, monitor: str = "valid/perfstrict_min_after_ramp", threshold: float = 0.8,
                 patience: int = 20):
        self.monitor, self.threshold, self.patience = monitor, threshold, patience
        self.streak = 0

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        v = trainer.callback_metrics.get(self.monitor)
        if v is None:
            return
        self.streak = self.streak + 1 if float(v) >= self.threshold else 0
        if self.streak >= self.patience:
            trainer.should_stop = True
            print(f"AcceptanceStopping: {self.monitor} >= {self.threshold} for {self.patience} epochs; stopping.")

    def state_dict(self):
        return {"streak": self.streak}

    def load_state_dict(self, state_dict):
        self.streak = state_dict.get("streak", 0)


class OnEpochStartCalls(pl.Callback):
    """
    Custom callbacks at the start of train/validation epochs.
    """
    def __init__(
        self,
        callbacks: list,
        priority: int = 1
    ):
        self.priority = priority
        self.callbacks = callbacks
        os.makedirs(SAVE_DIR, exist_ok=True)
        
    def run(self, trainer, pl_module, step_type):
        kwargs = {"step_type": step_type}
        for i, callback in enumerate(self.callbacks):
            # Use log if present as kwargs
            if callback.name == "log": kwargs["metrics"] = callback.metrics
                
            if step_type in callback.run_steps:
                callback.run(trainer, pl_module, **kwargs)

    def on_train_epoch_start(self, trainer, pl_module):
        self.run(trainer, pl_module, "train")
            
    def on_validation_epoch_start(self, trainer, pl_module):
        self.run(trainer, pl_module, "valid")


class OnEpochEndCalls(pl.Callback):
    """
    Custom callbacks at the end of train/validation epochs.
    """
    def __init__(
        self,
        callbacks: list,
        priority: int = 1
    ):
        self.priority = priority
        self.callbacks = callbacks
        os.makedirs(SAVE_DIR, exist_ok=True)
        
    def run(self, trainer, pl_module, step_type):
        kwargs = {"step_type": step_type}
        for i, callback in enumerate(self.callbacks):
            # Use log if present as kwargs
            if callback.name == "log": kwargs["metrics"] = callback.metrics

            if step_type in callback.run_steps:
                callback.run(trainer, pl_module, **kwargs)

    def on_train_epoch_end(self, trainer, pl_module):
        self.run(trainer, pl_module, "train")
            
    def on_validation_epoch_end(self, trainer, pl_module):
        self.run(trainer, pl_module, "valid")
        

class Log:
    """
    Keeps a running history of train/validation metrics so other callbacks can plot or
    summarize how the run is progressing.
    """
    def __init__(
        self,
        run_steps: list = ["train", "valid"]
    ):
        self.name = "log"
        self.run_steps_count = 0
        self.run_steps = run_steps
        self.run_steps_copy = deepcopy(run_steps)
        self.metrics = {}
    
    def run(self, trainer, pl_module, **kwargs):
        if len(self.run_steps_copy) > 0:
            try:
                self.run_steps_copy.remove(kwargs["step_type"])
                for key in trainer.logged_metrics.keys():
                    self.metrics[key] = []
            except:
                pass
        
        for key, value in trainer.logged_metrics.items():
            self.metrics[key].append(value.item())
