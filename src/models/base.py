"""Shared pieces of the data-generating networks: the noise schedule and the Lightning base class."""

import math

import torch
import torch.nn as nn
import pytorch_lightning as pl


SAVE_DIR = "./graphs/"


class VariableNoise:
    """
    Variable noise update class for hidden state and channel noise for DGNs.
    """
    def __init__(
        self,
        init_noise,
        feature_dim,
        ema_decay=0.99,
        device='cpu',
        off=False,
        start: int = 0,
        increase: int = 0,
        init: float = 0.0,
    ):
        """
        Args:
            init_noise: Initial noise magnitude.
            feature_dim: Number of features.
            ema_decay: Exponential moving average decay rate.
            device: Device to use.
            off: If True, use fixed ``init_noise`` (no data-dependent scaling).
            start: Epoch to begin ramping noise in (when ``increase`` > 0).
            increase: Epochs over which to ramp the noise scale from ``init`` to 1.
            init: Starting scale factor for the epoch ramp (0 = no noise until ramp).
        """
        self.init_noise = float(init_noise)
        self.feature_dim = feature_dim
        self.ema_decay = ema_decay
        self.device = device
        self.off = off
        self.start = start
        self.increase = increase
        self.init = float(init)

        self.noise_dist = torch.empty(feature_dim, device=device).uniform_(0.2, 1.0)
        self.ema_norm = torch.zeros(feature_dim, device=device)

    def __call__(self, data, current_epoch=None):
        B, T, F = data.shape

        # If noise is turned off:
        if self.off:
            weights = torch.full((F,), self.init_noise, device=data.device, dtype=data.dtype)
        else:
            # Compute per-feature norms
            with torch.no_grad():
                # Norm across time per sample, then average across batch
                mean_sq = data.pow(2).mean(dim=(0, 1))  # shape: (F,)
                norm = torch.sqrt(mean_sq + 1e-8)       # per-feature magnitude

                # Update EMA
                if self.ema_norm.sum() == 0: # first update → direct assign
                    self.ema_norm = norm
                else:
                    self.ema_norm = (self.ema_decay * self.ema_norm + (1 - self.ema_decay) * norm)

            weights = self.ema_norm * self.noise_dist

        # Optional epoch ramp (curriculum) on top of fixed or variable weights
        if self.increase > 0 and current_epoch is not None:
            ramp = (current_epoch + 1 - self.start) / (self.increase + 1)
            ramp = float(max(0.0, min(1.0, ramp)))
            factor = self.init + (1.0 - self.init) * ramp
            weights = weights * factor

        return weights


class DGNBase(pl.LightningModule):
    """
    Base class for all Data Generating Networks (DGN) Lightning Modules.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        
    def forward(self, *args, **kwargs):
        raise NotImplementedError('forward function must be implemented in subclass!')
     
    def _shared_step(self, *args, **kwargs):
        raise NotImplementedError('_shared_step function must be implemented in subclass!')
        
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")
    
    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "valid")
    
    def predict_step(self, batch, batch_idx):
        return self._shared_step(batch, "valid")
    
    def configure_optimizers(self):
        hps = self.hparams
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr = hps.lr,
        )
        return optimizer

    def _scale_gru_input_weights_by_var(self) -> None:
        """
        Scale GRUCell/RNNCell input weights (weight_ih) to increase variance.
        """
        var_scale = float(getattr(self.hparams, "input_weight_init_var_scale", 1.0))
        if var_scale == 1.0:
            return
        std_scale = math.sqrt(var_scale)

        # Scale input weights
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, (nn.GRUCell, nn.RNNCell)):
                    m.weight_ih.mul_(std_scale)
