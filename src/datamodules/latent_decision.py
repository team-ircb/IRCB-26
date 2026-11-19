"""Inputs for Relay-Decision: noisy evidence streams and their latent decision variable."""

import numpy as np
from torch.utils.data import random_split
from scipy.ndimage import gaussian_filter1d

from utils.common_utils import generate_noisy_sine_waves

from .base import BasicDataset, DGNDataModuleBase


class LatentDecision(DGNDataModuleBase):
    """
    DataModule for generating latent decisions, used with the `RelayDecision` model.
    """
    def __init__(
        self,
        batch_total: int,
        time_total: int,
        decay_factor: float = 1.0,
        latent_factor: float = 1.0,
        p_split: list = [0.8, 0.2],
        batch_size: int = 64,
        lag: int = 0,
        mesg_dist: str = "normal",
        binary_decision: bool = False,
        sig_smooth: float = None,
        resultpath: str = ".",
    ):
        """
        Args:
            batch_total: Total number of batches (trials).
            time_total: Total number of time steps.
            decay_factor: Decay factor for latent signal filtering.
            latent_factor: Factor for latent signal amplitude scaling.
            p_split: Train/validation split percentage, must sum up to 1.
            batch_size: Size of each training batch. Validation batch size is always
                the total number of validation batches.
            lag: Time-step lag for trajectory generation alignment.
            mesg_dist: Distribution type for input signal. Options are:
                'normal', 'exponential', 'uniform', 'sine wave'.
            binary_decision: Whether to generate binary-style decision trajectories.
            sig_smooth: Standard deviation for Gaussian smoothing.
            resultpath: Path to save the results.
        """
        super().__init__()
        self.save_hyperparameters()
        assert time_total >= 200 # latest cue time 150, smallest action time 50
        
        # Fixed hparams
        hps = self.hparams
        hps.input_dim = 2
        
    def setup(self, stage=None):
        """
        Setup the dataset and split into train and validation sets.
        """
        hps = self.hparams

        # Generate input signals
        if hps.mesg_dist == "normal":
            inp = np.random.normal(size=(hps.batch_total, hps.time_total, hps.input_dim)) * 3
        elif hps.mesg_dist == "exponential":
            inp = np.random.exponential(scale=3.0, size=(hps.batch_total, hps.time_total, hps.input_dim)) - 3.0
        elif hps.mesg_dist == "uniform":
            inp = np.random.uniform(-5., 5., size=(hps.batch_total, hps.time_total, hps.input_dim))
        elif hps.mesg_dist == "sine wave":
            freq = np.random.uniform(0.1, 1.0, size=hps.input_dim)
            inp = generate_noisy_sine_waves(hps.batch_total, hps.time_total, hps.input_dim, freq, noise_level=0.5) * 4.
        else:
            raise ValueError(f"Invalid message distribution: {hps.mesg_dist}")
        
        # Apply smoothing if specified
        if hps.sig_smooth:
            inp = gaussian_filter1d(inp, sigma=hps.sig_smooth, axis=1)
            
        # Generate latent signal 
        latent = np.random.normal(size=(hps.batch_total, hps.time_total, 1))
        latent = self.exponential_filter(latent, hps.decay_factor, 1) * hps.latent_factor

        # Generate decision signal
        dec = np.cumsum(inp, axis=1)
        
        # Random sample go time, go cue duration = 10
        go_times = np.random.randint(low=100, high=150, size=hps.batch_total)
        go = np.zeros((hps.batch_total, hps.time_total, 1))
        for b, go_time in enumerate(go_times):
            go[b, go_time: go_time + 10, 0] = np.ones(10)
            
        # Generate context signal (at the start of the trial, 10-20 timestep)
        ctxt = np.zeros((hps.batch_total, hps.time_total, 1))
        for b, go_time in enumerate(go_times):
            ctxt[b, 10:20, 0] = np.ones(10) * np.random.choice([1, -1])
        
        # Generate trajectories
        action = np.zeros((hps.batch_total, hps.time_total, 3))
        ang_velocity = np.pi / (hps.time_total - 100) # doesn't turn more than half a circle (pi)

        lag = hps.lag
        for b in range(hps.batch_total):
            
            # Generate trajectory for continuous decision
            if not hps.binary_decision:
                traj = self.gen_trajectory_3d(
                    *dec[b, go_times[b] - lag],
                    latent[b, go_times[b] - lag].item(),
                    hps.time_total - go_times[b],
                    ang_velocity,
                    np.sign(ctxt[b].mean()),
                )
            else: # generate trajectory for binary decision
                traj = self.gen_trajectory_3d(
                    * np.sign(dec[b, go_times[b] - lag]) * 3, # amplitude=3
                    0,                                        # does not use latent
                    hps.time_total - go_times[b],
                    ang_velocity,
                    np.sign(ctxt[b].mean()),
                )
                
            # Pad trajectory with zeros to match total time
            traj_embed = np.pad(traj, ((go_times[b], 0), (0,0)), mode="constant", constant_values=0)
            action[b] = traj_embed
        
        # Construct the dataset and split into train and validation sets
        ds = BasicDataset(
            inp.astype(np.float32), 
            latent.astype(np.float32), 
            go.astype(np.float32), 
            ctxt.astype(np.float32), 
            action.astype(np.float32), 
        )
        self.train_ds, self.val_ds = random_split(ds, hps.p_split)
        
    @staticmethod
    def gen_trajectory_3d(x0, y0, z0, time, v, ctxt):
        """
        Generate the 3D trajectory points for a decision signal.

        Args:
            x0 (float): x dimension of the decision signal.
            y0 (float): y dimension of the decision signal.
            z0 (float): z dimension of the decision signal.
            time (int): time steps.
            v (float): angular velocity.
            ctxt (int): context signal.

        Returns:
            traj (numpy.ndarray): 3D trajectory points (time, 3)
        """
        sigmoid = lambda x: 1/(1+np.exp(-0.02 * x)) # scaled sigmoid function
        
        # Calculate radius and signs
        if ctxt == 1: # use x dimension
            r = sigmoid(np.sqrt(x0**2 + z0**2))
            sign = np.sign(x0)
        else: # use y dimension
            r = sigmoid(np.sqrt(y0**2 + z0**2))
            sign = np.sign(y0)

        # Calculate angles
        angles = np.arange(0, time) * v
        phi0 = (1-sign) * np.pi / 2

        # Calculate trajectory points
        if ctxt == 1:
            x_traj = r * np.cos(phi0 + angles)
            y_traj = np.zeros_like(x_traj)
        else:
            y_traj = r * np.cos(phi0 + angles)
            x_traj = np.zeros_like(y_traj)
        
        # Calculate z dimension
        z_traj = r * np.sin(phi0 + angles)
        traj = np.vstack([x_traj, y_traj, z_traj])
        
        return traj.T
    
    @staticmethod
    def exponential_filter(array, alpha, axis):
        """
        Apply a decaying exponential filter to an array along a specified dimension.

        Args:
            array (numpy.ndarray): Input array.
            alpha (float): Decay factor, where 0 < alpha < 1.
            axis (int): Dimension along which to apply the filter.

        Returns:
            numpy.ndarray: Filtered array.
        """
        def filter_func(data):
            decay_weights = alpha ** np.arange(len(data))[::-1]
            return np.convolve(data, decay_weights, mode='full')[:len(data)]

        return np.apply_along_axis(filter_func, axis, array)
