import numpy as np

def bin_unit(spike_times, starts, stops, n_bins):
    """Return [trials, bins] spike counts using per-trial equal-width edges.

    Bin convention is [left, right), including the trial start, excluding stop.
    Times are NWB absolute seconds. No time warping of spike timestamps.
    """
    spikes = np.asarray(spike_times, dtype=np.float64)
    if spikes.size and np.any(np.diff(spikes) < 0):
        spikes = np.sort(spikes)
    starts, stops = np.asarray(starts, float), np.asarray(stops, float)
    result = np.zeros((len(starts), n_bins), dtype=np.int32)
    for i, (start, stop) in enumerate(zip(starts, stops)):
        if not (np.isfinite(start) and np.isfinite(stop) and stop > start):
            raise ValueError("Invalid trial bounds")
        lo, hi = np.searchsorted(spikes, [start, stop], side="left")
        positions = (spikes[lo:hi] - start) * (n_bins / (stop - start))
        indices = np.minimum(np.floor(positions).astype(np.int64), n_bins - 1)
        result[i] = np.bincount(indices, minlength=n_bins).astype(np.int32)
    return result
