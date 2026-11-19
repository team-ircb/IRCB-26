"""Saving a DGN's activity, messages and ground truth to HDF5 during training."""

import h5py

from utils.common_utils import area_activity_to_poisson_counts


class SaveAsH5:
    """
    Periodically saves model tensors (hidden states, messages, ground truth, info, inputs)
    to data.h5 for further analysis and visualization.
    """
    def __init__(
        self,
        log_every_n_epochs: int = 1,
        ground_truth: list = [],
        run_steps: list = ["valid"],
        poisson_activity: bool = False,
        poisson_dt: float = 0.1,
        poisson_rate_max: float = 50.0,
        poisson_seed: int | None = 42,
    ):
        """
        Args:
            log_every_n_epochs: The frequency of saving the data.
            ground_truth: The ground truth channels to save.
            run_steps: The steps to run the callback on.
            poisson_activity: Whether to transform area hidden state activity to Poisson counts in ``data.h5``.
            poisson_dt: The time bin in seconds.
            poisson_rate_max: The maximum rate (Hz).
            poisson_seed: RNG seed for Poisson draws (``None`` = nondeterministic each save).
        """
        self.name = "saveash5"
        self.run_steps = run_steps
        self.log_every_n_epochs = log_every_n_epochs
        self.ground_truth = ground_truth
        self.best = 1e10
        self.poisson_activity = poisson_activity
        self.poisson_dt = poisson_dt
        self.poisson_rate_max = poisson_rate_max
        self.poisson_seed = poisson_seed

    def run(self, trainer, pl_module, **kwargs):
        if (trainer.current_epoch % self.log_every_n_epochs) != 0: return
    
        print("Saving data from epoch ", trainer.current_epoch, "...")

        # Get task_reward info
        ground_truth_arr = pl_module.current_batch
        info = pl_module.current_info
        # The configured names label the batch's leading arrays in order; a config may name fewer arrays than the
        # batch holds (Relay-Decision's batch has inp, latent, go, ctxt, action and the release saves only inp).
        assert len(self.ground_truth) <= len(ground_truth_arr), (
            f"{len(self.ground_truth)} ground-truth names for a batch of {len(ground_truth_arr)} arrays"
        )
        
        override = False
        new = kwargs["metrics"]["valid/loss"][-1]
        if new < self.best:
            self.best = new
            override = True
            print("OVERRIDE data.h5")
        
        if override:
            with h5py.File("data.h5", "w") as file:
                # Save hidden states
                for area_name, arr in pl_module.hidden_states.items():
                    data = arr.cpu().detach().numpy()
                    if self.poisson_activity:
                        data = area_activity_to_poisson_counts(data, self.poisson_dt, self.poisson_rate_max, seed=self.poisson_seed)
                    h5ds = file.create_dataset(f"area-{area_name}", data=data)
                    h5ds.attrs["type"] = "hidden_state"
                    if self.poisson_activity:
                        h5ds.attrs["representation"] = "poisson_counts"

                # Save messages
                for mes_name, arr in pl_module.save_var._asdict().items():
                    h5ds = file.create_dataset(f"message-{mes_name}", data=arr.cpu().detach().numpy())
                    h5ds.attrs["type"] = "message"

                # Save ground truth
                for ig in range(len(self.ground_truth)):
                    h5ds = file.create_dataset(f"truth-{self.ground_truth[ig]}", data=ground_truth_arr[ig].cpu().detach().numpy())
                    h5ds.attrs["type"] = "ground_truth"

                # Save info
                for info_name, info_val in info.items():
                    h5ds = file.create_dataset(f"info-{info_name}", data=info_val.cpu().detach().numpy())
                    h5ds.attrs["type"] = "info"
                    
                # Save inputs
                if hasattr(pl_module, "inputs"):
                    for area_name, arr in pl_module.inputs.items():
                        h5ds = file.create_dataset(f'inputs-{area_name}', data=arr.cpu().detach().numpy())
                        h5ds.attrs["type"] = "inputs"
