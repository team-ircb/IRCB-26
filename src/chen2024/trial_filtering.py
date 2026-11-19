import numpy as np

TARGET_TRIAL_DURATION_S = 5.0

def photostim_mask(values):
    """True for trials with stimulation; missing column is handled by caller.

    Same rule as the original extraction's loader: a trial counts as
    unstimulated only if ``photostim_onset`` is exactly "N/A"; any other value counts as
    stimulated.
    """
    def stimulated(x):
        if isinstance(x, bytes):
            x = x.decode("utf-8", errors="replace")
        return str(x) != "N/A"
    return np.array([stimulated(x) for x in values], dtype=bool)

def select_trials(trials, config):
    starts = np.asarray(trials["start_time"], dtype=float)
    stops = np.asarray(trials["stop_time"], dtype=float)
    dur = stops - starts
    tol = float(config.get("duration_tolerance_s", 0.5))
    valid = np.isfinite(dur) & (dur > 0) & (np.abs(dur - TARGET_TRIAL_DURATION_S) <= tol)
    stim = np.zeros(len(starts), dtype=bool)
    if "photostim_onset" in trials:
        stim = photostim_mask(trials["photostim_onset"].to_numpy())
    elif config.get("photostim", "exclude") == "only":
        raise ValueError("Cannot select photostim-only trials: photostim_onset absent")
    mode = config.get("photostim", "exclude")
    if mode == "exclude": valid &= ~stim
    elif mode == "only": valid &= stim
    return np.flatnonzero(valid), stim
