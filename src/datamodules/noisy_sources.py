"""Inputs for Delayed-Memory: independent noisy stimuli, one per area."""

import numpy as np
from torch.utils.data import random_split
from scipy.ndimage import gaussian_filter1d

from utils.common_utils import generate_noisy_sine_waves

from .base import BasicDataset, DGNDataModuleBase


class NoisySources(DGNDataModuleBase):
    """
    DataModule for generating noisy sources, used with the `DelayedMemory` model.
    """
    def __init__(
        self,
        batch_total: int,
        time_total: int,
        input_dim: list | int,
        p_split: list = [0.8, 0.2],
        batch_size: int = 64,
        mesg_type: str = "white noise",
        mesg_kwargs: dict = {},
        sig_smooth: float = None,
        resultpath: str = ".",
    ):
        """
        Args:
            batch_total: Total number of batches (trials).
            time_total: Total number of time steps.
            input_dim: Dimensionality of input for each area. If an integer 
                is passed, all areas will share the same dimensionality.
            p_split: Train/validation split percentage, must sum up to 1.
            batch_size: Size of each training batch. Validation batch size is always
                the total number of validation batches.
            mesg_type: Distribution type for signals. Options are:
                'white noise', 'intg noise', 'filter noise' or 'sine wave'.
                See `transform()` function for each option.
            mesg_kwargs: Corresponding kwargs for the chosen `mesg_type`.
            sig_smooth: Standard deviation for Gaussian smoothing.
            resultpath: Path to save the results.
        """
        super().__init__()
        self.save_hyperparameters()
        hps = self.hparams
        
        if isinstance(hps.input_dim, list):
            hps.input_dim = sum(hps.input_dim)
        
    def setup(self, stage=None):
        """
        Setup the dataset and split into train and validation sets.
        """
        hps = self.hparams

        # Generate input for sine wave message type
        if hps.mesg_type == 'sine wave':
            freq = np.random.uniform(0.1, 1.0, size=hps.input_dim)
            inp = generate_noisy_sine_waves(hps.batch_total, hps.time_total, hps.input_dim, freq, noise_level=0.0)
        else:
            # Generate white noise for all other message types
            inp = np.random.normal(size=(hps.batch_total, hps.time_total, hps.input_dim))
            
            # Preprocess message type
            if isinstance(hps.mesg_type, str):
                hps.mesg_type = [hps.mesg_type] * hps.input_dim
            
            # Change noise index by index according to mesg_type
            for idim, mtype in enumerate(hps.mesg_type): 
                inp = self.transform(inp, idim, mtype)
        
        # Apply smoothing if specified
        if hps.sig_smooth:
            inp = gaussian_filter1d(inp, sigma=hps.sig_smooth, axis=1)

        # Construct the dataset and split into train and validation sets
        ds = BasicDataset(
            inp.astype(np.float32),
        )
        self.train_ds, self.val_ds = random_split(ds, hps.p_split)
        
    def transform(self, x, idx, mesg_type):
        """
        Transform the input according to the message type.

        Args:
            x (numpy.ndarray): Input array.
            idx (int): Index of the input dimension to transform.
            mesg_type (str): Message type.

        Returns:
            x (numpy.ndarray): Transformed input array.
        """
        hps = self.hparams
        if mesg_type == "white noise": # no changes
            pass 
        elif mesg_type == "intg_noise": # performs cumsum
            x[..., idx] = np.cumsum(x[..., idx], axis=1)
        elif mesg_type == "filter noise": # gaussian filter
            assert 'std' in hps.mesg_kwargs.keys()
            x[..., idx] = gaussian_filter1d(x[..., idx], hps.mesg_kwargs['std'][idx], axis=1)
        else:
            raise ValueError()
        return x
