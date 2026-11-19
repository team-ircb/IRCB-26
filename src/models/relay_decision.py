"""Relay-Decision: a relay area passes the stimulus to a decision area that integrates it."""

import torch
import torch.nn as nn

from utils.torch_utils import RNNChannel, MLPBase
from utils.common_utils import RelayDecisionMotionMessages

from .base import DGNBase


class RelayDecision(DGNBase):
    """
    Two-area recurrent Relay-Decision network with area-specific input signals and integration of task signals (DGN).
    """
    def __init__(
        self,
        lag: int = 0,

        noise_p: float = 0.0,
        noise_d: float = 0.0,

        hidden_size: int = 64,
        lr: float = 4.0e-3,
        input_weight_init_var_scale: float = 1.0,

        p_to_d_coef: float = 1.0,
        rep_coef: float = 0.0,
        binary_output: bool = True,
        rnn_nonlinearity: str = "tanh",
        rnn_type: str = "grucell",
    ):
        """
        Args:
            lag: Time-step lag for Relay-Decision communication.

            noise_p: Dynamic noise applied to Pass-area hidden states.
            noise_d: Dynamic noise applied to Decision-area hidden states.

            hidden_size: Number of hidden units per area.
            lr: Learning rate.
            input_weight_init_var_scale: Scale factor for the initial variance
                of the RNN input weights.

            p_to_d_coef: Loss weight for matching the Pass-Decision channel to
                the raw input trajectory.
            rep_coef: Loss weight for matching Decision hidden readouts to the
                input (representation regularization).
            binary_output: If True, decision decoder uses sigmoid + BCE against
                a binarized cumulative input; if False, MSE against the
                cumulative input.
            rnn_nonlinearity: Nonlinearity for ``RNNChannel`` cells.
            rnn_type: Recurrent cell type for each area. Either 'grucell'
                (gated) or 'rnncell' (vanilla tanh RNN).
        """
        super().__init__()
        self.save_hyperparameters()
        hps = self.hparams

        # Fixed hps
        hps.input_dim = 2
        hps.channel_size = 2
        hps.output_size = hps.hidden_size

        # Pass area: stimulus + latent (input_dim)
        self.P_area = RNNChannel(hps.input_dim, hps.hidden_size, [hps.input_dim], 
                                 None, rnn_nonlinearity=hps.rnn_nonlinearity,
                                 rnn_type=hps.rnn_type)
        # Decision area: input + latent (input_dim)
        self.D_area = RNNChannel(hps.input_dim, hps.hidden_size, [hps.channel_size], 
                                 None, rnn_nonlinearity=hps.rnn_nonlinearity,
                                 rnn_type=hps.rnn_type)

        # Loss functions and decoders
        nonlinearity = "sigmoid" if hps.binary_output else None
        self.decoder = MLPBase([[hps.channel_size, hps.input_dim, nonlinearity]])
        self.mseloss = nn.MSELoss()
        self.bceloss = nn.BCELoss() # expects probability (sigmoid)
        self.celoss = nn.CrossEntropyLoss() # does not expect probability (no sigmoid)
        
        # Decoder on Decision hidden state (representation matching to input)
        self.D_decoder = MLPBase([[hps.hidden_size, hps.input_dim, None]])

        # Increase variance of the RNN input weights
        self._scale_gru_input_weights_by_var()
        
    def forward(self, inp, latent, go, ctxt, step_type):
        hps = self.hparams
        batch, time, _ = inp.shape # (batch, time, input_dim)
        self._build_save_var(batch, time)

        # Initialize hidden states
        h_p = torch.zeros(batch, hps.hidden_size).to(self.device)
        h_d = torch.zeros(batch, hps.hidden_size).to(self.device)
        
        # Main loop
        for t in range(time):

            # Forward pass through individual areas
            h_p, p_to_d = self.P_area(inp[:,t,:], h_p)
            h_p = h_p + torch.randn_like(h_p) * hps.noise_p
            h_d, d_to_m = self.D_area(p_to_d, h_d)
            h_d = h_d + torch.randn_like(h_d) * hps.noise_d
            
            # Decode decision
            d = self.decoder(d_to_m)
            d_rep = self.D_decoder(h_d)

            # Save variables
            self.save_var.p_to_d[:,t] = p_to_d
            self.save_var.d_to_m[:,t] = d_to_m
            self.save_var.d[:, t] = d
            self.save_var.d_rep[:, t] = d_rep
            
            # Save hidden states
            self.hidden_states["P"][:,t] = h_p
            self.hidden_states["D"][:,t] = h_d

        return list(self.save_var._asdict().values()) + list(self.hidden_states.values())
    
    def _shared_step(self, batch, step_type):
        hps = self.hparams
        self.current_batch, self.current_info = batch

        # Setup lag
        if not hps.lag:
            plag = nlag = None
        else:
            plag = hps.lag
            nlag = -hps.lag
        
        # Forward pass
        inp, latent, go, ctxt, action = self.current_batch
        p_to_d, d_to_m, d_pred, a, d_rep, m_rep, h_p, h_d = self.forward(inp, latent, go, ctxt, step_type)
        
        # Get decision labels
        batch_size, time, _ = inp.shape
        dec = torch.cumsum(inp, dim=1) # shape = (batch, time, input_dim) still
        d_true_sign = torch.sign(dec) # becomes -1, 0 (highly unlikely) or 1
        d_true = (d_true_sign + 1) / 2 # becomes 0, 1
        
        # Loss for Pass on input
        p_to_d_loss = self.mseloss(p_to_d[:, plag:, :hps.input_dim], inp[:, :nlag])
            
        # Get decision decoder loss
        if hps.binary_output:
            d_loss = self.bceloss(d_pred[:, plag:].reshape(-1), d_true[:, :nlag].reshape(-1))
        else: # decode continuous, linear output instead
            d_loss = self.mseloss(d_pred[:, plag:].reshape(-1), dec[:, :nlag].reshape(-1))
        
        # Get area D representation loss
        d_rep_loss = self.mseloss(d_rep[:, plag:], inp[:, :nlag])
          
        # Get total loss
        loss = d_loss \
                + p_to_d_loss * hps.p_to_d_coef \
                + d_rep_loss * hps.rep_coef
        
        # Get accuracy
        d_label = (d_pred >= 0.5).float()
        accuracy = (d_true[:, :nlag] == d_label[:, plag:]).float().mean()

        # Log metrics
        metrics = {
            f"{step_type}/loss": loss,
            f"{step_type}/d_loss": d_loss,
            f"{step_type}/d_rep_loss": d_rep_loss * hps.rep_coef,
            f"{step_type}/p_to_d_loss": p_to_d_loss * hps.p_to_d_coef,
            f"{step_type}/accuracy": accuracy,
        }
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        return loss
    
    def _build_save_var(self, batch_size, time):
        hps = self.hparams

        # Storage for Relay-Decision communication
        self.save_var = RelayDecisionMotionMessages(
            p_to_d = torch.zeros(batch_size, time, hps.input_dim).to(self.device),
            d_to_m = torch.zeros(batch_size, time, hps.channel_size).to(self.device),
            d = torch.zeros(batch_size, time, hps.input_dim).to(self.device),
            a = torch.zeros(0).to(self.device),
            d_rep = torch.zeros(batch_size, time, hps.input_dim).to(self.device),
            m_rep = torch.zeros(0).to(self.device),
        )

        # Storage for hidden states
        self.hidden_states = {
            "P": torch.zeros(batch_size, time, hps.output_size).to(self.device),
            "D": torch.zeros(batch_size, time, hps.output_size).to(self.device),
        }
