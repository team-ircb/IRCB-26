"""Basic scores shared by the evaluation modules: R², McFadden pseudo-R², cosine and Pearson
similarity, and the L2 strength of a communication signal."""

from __future__ import annotations

import numpy as np

# Variance below which a neuron or channel counts as constant and is left out of R².
MIN_VARIANCE = 1e-12


def per_channel_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² of each channel (last axis) over every trial and time bin, averaged over channels.

    Each channel is compared with its own mean, so a prediction gets no credit for differences in
    mean between channels. Channels whose true values don't vary have no defined R² and are left
    out; if none vary, the result is nan.
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    if y_t.shape != y_p.shape:
        raise ValueError(f"shape mismatch: {y_t.shape} vs {y_p.shape}")
    if not (np.isfinite(y_t).all() and np.isfinite(y_p).all()):
        return float("nan")

    y_t = y_t.reshape(-1, y_t.shape[-1])
    y_p = y_p.reshape(-1, y_p.shape[-1])
    sst = np.sum((y_t - y_t.mean(axis=0)) ** 2, axis=0)
    sse = np.sum((y_t - y_p) ** 2, axis=0)

    varies = sst / y_t.shape[0] >= MIN_VARIANCE
    if not varies.any():
        return float("nan")
    return float(np.mean(1.0 - sse[varies] / sst[varies]))


# Rate R² is the same computation with neurons as the channels.
per_neuron_r2 = per_channel_r2


def poisson_log_likelihood(counts: np.ndarray, rates: np.ndarray) -> np.ndarray:
    """Elementwise Poisson log-likelihood of `counts` under means `rates` (counts per bin).

    log k! uses Stirling's approximation for k > 1 (and 0 for k <= 1), as PyTorch's
    poisson_nll_loss(full=True) does, so scores match those computed with it.
    """
    k = np.asarray(counts, dtype=np.float64)
    lam = np.asarray(rates, dtype=np.float64)

    # A small floor keeps log(0) finite: predicting exactly 0 where a spike occurred is penalized
    # heavily instead of giving -inf.
    log_k_factorial = np.zeros_like(k)
    big = k > 1
    log_k_factorial[big] = k[big] * np.log(k[big]) - k[big] + 0.5 * np.log(2 * np.pi * k[big])

    return k * np.log(lam + 1e-8) - lam - log_k_factorial


def mcfadden_r2(counts: np.ndarray, rates: np.ndarray) -> float:
    """McFadden pseudo-R² of spike counts under Poisson means `rates` (trials, time, neurons).

    1 - log L(model) / log L(null), where the null predicts each neuron's mean count over trials in
    each time bin. Bins where the null fits perfectly (its log-likelihood is 0) are left out, as
    they carry no information. nan if nothing is left.
    """
    k = np.asarray(counts, dtype=np.float64)
    lam = np.asarray(rates, dtype=np.float64)
    if k.shape != lam.shape:
        raise ValueError(f"shape mismatch: {k.shape} vs {lam.shape}")

    null = np.broadcast_to(k.mean(axis=0, keepdims=True), k.shape)
    ll_model = poisson_log_likelihood(k, lam)
    ll_null = poisson_log_likelihood(k, null)

    keep = ll_null < 0
    denom = ll_null[keep].sum()
    if not keep.any() or np.isclose(denom, 0.0):
        return float("nan")
    return float(1.0 - ll_model[keep].sum() / denom)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two arrays, flattened. nan if either is all zeros."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (norm_a * norm_b))


def pearson_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two arrays, flattened: the cosine similarity after subtracting each
    array's mean. nan if either is constant."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return cosine_similarity(a - a.mean(), b - b.mean())


def off_diagonal(matrix: np.ndarray) -> np.ndarray:
    """The off-diagonal entries of a square matrix, as a vector (row-major)."""
    m = np.asarray(matrix, dtype=np.float64)
    return m[~np.eye(m.shape[0], dtype=bool)]


def signal_strength(signal: np.ndarray) -> float:
    """L2 norm of a signal over every trial, time bin and channel."""
    x = np.asarray(signal, dtype=np.float64)
    return float(np.sqrt(np.sum(np.square(x))))


def weighted_signal_strength(weights: np.ndarray, signal: np.ndarray) -> float:
    """L2 norm of W m over every trial and time bin: the strength of a message `signal`
    (trials, time, channels) after the map `weights` (hidden, channels) into its target area."""
    w = np.asarray(weights, dtype=np.float64)
    m = np.asarray(signal, dtype=np.float64)
    if m.ndim != 3:
        raise ValueError(f"expected a message shaped (trials, time, channels), got {m.shape}")
    if w.ndim != 2 or w.shape[1] != m.shape[-1]:
        raise ValueError(f"weights {w.shape} do not match a message with {m.shape[-1]} channels")
    return signal_strength(np.einsum("btc,hc->bth", m, w))
