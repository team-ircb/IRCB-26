"""Decoders for communication and input fidelity / specificity.

Each decoder maps x to y, both (trials, time, channels), fitting on 80% of the trials and scoring
on the other 20%. The score is R² per output channel (against that channel's own mean on the
held-out trials), averaged over channels. Every decoder uses the same trial split, so their scores
are comparable.

    linear         ordinary least squares. The benchmark score.
    kernel ridge   a linear kernel plus a scaled RBF kernel (optional, slower).
    mlp            a linear skip connection plus a small MLP (optional, slower).

Both nonlinear decoders contain the linear decoder as a special case, so on a linear relation they
should not score much below it.

Kernel ridge settings
  - Kernel k(x, x') = x.x' + s^2 exp(-gamma |x - x'|^2) on inputs z-scored over the training
    trials: s = 0 is linear ridge regression.
  - The RBF part uses the Nystroem approximation with 1,000 landmarks drawn from the training rows
    (seed 0). Eigenvalues of the landmark kernel below 1e-10 of the largest are dropped.
  - gamma = g / d for an input width d, with g in {0.1, 0.3, 1}. g = 1 is scikit-learn's
    gamma="scale" for unit-variance inputs.
  - Ridge penalty lam * n_train with lam in 1e-6 ... 1e2 (9 values), and s in
    {0, 0.01, 0.03, 0.1, 0.3, 1}. (g, s, lam) are chosen by 3-fold cross-validation over the
    training trials (seed 0), maximizing the mean per-channel R².

MLP settings
  - y = W x + g(x): a linear skip plus an MLP g with two GELU hidden layers of width `hidden`.
    W starts at the least-squares solution and g's last layer at zero, so training starts from the
    linear decoder, and that starting point is the first early-stopping candidate.
  - Inputs z-scored and outputs standardized per channel on the training trials, so the MSE loss
    weights channels as the per-channel R² does.
  - AdamW (learning rate 1e-3, weight decay 1e-4), batches of 1,024 trial-bins, at most 200
    epochs, early stopping after 20 epochs without improvement on 20% of the training trials
    (seed 0), best weights restored. Torch seed 0; other settings are PyTorch defaults.
  - Communication scores use width 256 for fidelity and 64 for specificity.
"""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split

from .metrics import per_channel_r2

# Trial split shared by every decoder.
TEST_FRACTION = 0.2
SPLIT_SEED = 0
MIN_TRIALS = 5  # fewer trials than this can't be split into fitting and test trials

# Kernel ridge grid.
RIDGE_PENALTIES = np.logspace(-6, 2, 9)
RBF_SCALES = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0)
GAMMA_MULTIPLIERS = (0.1, 0.3, 1.0)
N_LANDMARKS = 1000
INNER_FOLDS = 3

# MLP training.
MLP_HIDDEN = {"fidelity": 256, "specificity": 64}
MLP_LEARNING_RATE = 1e-3
MLP_WEIGHT_DECAY = 1e-4
MLP_BATCH = 1024
MLP_MAX_EPOCHS = 200
MLP_PATIENCE = 20

DECODER_NAMES = ("linear", "kernel ridge", "mlp")

# A decoder takes (x, y) and returns (y on the held-out trials, its prediction, R²).
DecodeResult = tuple[np.ndarray, np.ndarray, float]


def split_trials(n_trials: int) -> tuple[np.ndarray, np.ndarray]:
    """The 80/20 split of trial indices that every decoder uses."""
    if n_trials < MIN_TRIALS:
        raise ValueError(f"need at least {MIN_TRIALS} trials to fit and test a decoder, got {n_trials}")
    idx = np.arange(n_trials)
    return train_test_split(idx, test_size=TEST_FRACTION, random_state=SPLIT_SEED, shuffle=True)


def as_trials_time_channels(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Both arrays as float64 (trials, time, channels). Trial-level variables (trials, channels)
    get a time axis of length 1."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim == 2:
        x = x[:, None, :]
    if y.ndim == 2:
        y = y[:, None, :]
    if x.ndim != 3 or y.ndim != 3:
        raise ValueError(f"expected (trials, time, channels), got {x.shape} and {y.shape}")
    if x.shape[:2] != y.shape[:2]:
        raise ValueError(f"trials/time differ: {tuple(x.shape[:2])} vs {tuple(y.shape[:2])}")
    return x, y


def _rows(a: np.ndarray, trials: np.ndarray) -> np.ndarray:
    # Stack the time bins of the chosen trials into rows.
    return a[trials].reshape(-1, a.shape[-1])


def linear_decode(x: np.ndarray, y: np.ndarray) -> DecodeResult:
    """Least-squares decode of y from x (with an intercept)."""
    x, y = as_trials_time_channels(x, y)
    train_idx, test_idx = split_trials(y.shape[0])

    decoder = LinearRegression()
    decoder.fit(_rows(x, train_idx), _rows(y, train_idx))

    mapped = decoder.predict(_rows(x, test_idx)).reshape(len(test_idx), y.shape[1], y.shape[-1])
    return y[test_idx], mapped, per_channel_r2(y[test_idx], mapped)


# ------------------------------------------------------------------------------- kernel ridge


class _Nystroem:
    """Nystroem features of an RBF kernel with width gamma = g / input width."""

    def __init__(self, g: float, seed: int = 0):
        self.g = g
        self.seed = seed

    def fit(self, z: np.ndarray):
        self.gamma = self.g / z.shape[1]
        rng = np.random.default_rng(self.seed)
        self.landmarks = z[rng.choice(len(z), size=min(N_LANDMARKS, len(z)), replace=False)]

        eigvals, eigvecs = np.linalg.eigh(self._kernel(self.landmarks))
        keep = eigvals > 1e-10 * eigvals.max()
        self.projection = eigvecs[:, keep] / np.sqrt(eigvals[keep])
        return self

    def _kernel(self, a: np.ndarray) -> np.ndarray:
        sq = (a**2).sum(1)[:, None] + (self.landmarks**2).sum(1)[None, :] - 2 * a @ self.landmarks.T
        return np.exp(-self.gamma * np.maximum(sq, 0.0))

    def transform(self, z: np.ndarray, chunk: int = 10000) -> np.ndarray:
        # In chunks of rows, stored as float32, to keep memory low on the larger datasets.
        out = np.empty((len(z), self.projection.shape[1]), dtype=np.float32)
        for i in range(0, len(z), chunk):
            out[i : i + chunk] = self._kernel(z[i : i + chunk]) @ self.projection
        return out


def _standardize(train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu, sd = train.mean(0), train.std(0) + 1e-8
    return (train - mu) / sd, (other - mu) / sd


def _ridge_grid_scores(f_lin, f_rbf, y, fit_rows, val_rows) -> dict:
    # Validation R² for every (s, lam), from blockwise Gram matrices of the fitting rows.
    a, a_val = _standardize(f_lin[fit_rows], f_lin[val_rows])
    b, b_val = _standardize(f_rbf[fit_rows], f_rbf[val_rows])
    y_fit = y[fit_rows]
    y_mean = y_fit.mean(0)
    y_centred = y_fit - y_mean

    caa, cab, cbb = a.T @ a, a.T @ b, b.T @ b
    ra, rb = a.T @ y_centred, b.T @ y_centred
    n = len(fit_rows)

    scores = {}
    for s in RBF_SCALES:
        if s == 0:
            gram, rhs, features_val = caa, ra, a_val
        else:
            gram = np.block([[caa, s * cab], [s * cab.T, s * s * cbb]])
            rhs = np.vstack([ra, s * rb])
            features_val = np.hstack([a_val, s * b_val])

        # One eigendecomposition serves every penalty.
        eigvals, eigvecs = np.linalg.eigh(gram)
        rhs_rotated = eigvecs.T @ rhs
        features_rotated = features_val @ eigvecs
        for lam in RIDGE_PENALTIES:
            pred = features_rotated @ (rhs_rotated / (eigvals + lam * n)[:, None]) + y_mean
            scores[(s, lam)] = per_channel_r2(y[val_rows], pred)
    return scores


def _ridge_fit_predict(f_lin, f_rbf, y, fit_rows, pred_rows, s, lam) -> np.ndarray:
    a, a_pred = _standardize(f_lin[fit_rows], f_lin[pred_rows])
    blocks, pred_blocks = [a], [a_pred]
    if s > 0:
        b, b_pred = _standardize(f_rbf[fit_rows], f_rbf[pred_rows])
        blocks.append(s * b)
        pred_blocks.append(s * b_pred)

    features, features_pred = np.hstack(blocks), np.hstack(pred_blocks)
    y_mean = y[fit_rows].mean(0)
    gram = features.T @ features + lam * len(fit_rows) * np.eye(features.shape[1])
    coef = np.linalg.solve(gram, features.T @ (y[fit_rows] - y_mean))
    return features_pred @ coef + y_mean


def kernel_ridge_decode(x: np.ndarray, y: np.ndarray) -> DecodeResult:
    """Kernel ridge decode of y from x (linear plus RBF kernel; settings in the module docstring)."""
    x, y = as_trials_time_channels(x, y)
    train_idx, test_idx = split_trials(y.shape[0])
    n_time, n_out = y.shape[1], y.shape[-1]

    x_train = _rows(x, train_idx)
    mu, sd = x_train.mean(0), x_train.std(0) + 1e-8
    z = (x.reshape(-1, x.shape[-1]) - mu) / sd
    y_rows = y.reshape(-1, n_out)

    def rows_of(trials):
        return (trials[:, None] * n_time + np.arange(n_time)[None, :]).reshape(-1)

    train_rows = rows_of(train_idx)
    folds = list(KFold(INNER_FOLDS, shuffle=True, random_state=0).split(train_idx))

    # Choose (g, s, lam) by cross-validation over the training trials.
    best_score, best = -np.inf, None
    for g in GAMMA_MULTIPLIERS:
        f_rbf = _Nystroem(g).fit(z[train_rows]).transform(z)
        fold_scores: dict = {}
        for fit_f, val_f in folds:
            scores = _ridge_grid_scores(z, f_rbf, y_rows, rows_of(train_idx[fit_f]), rows_of(train_idx[val_f]))
            for key, value in scores.items():
                fold_scores.setdefault(key, []).append(value)

        for (s, lam), values in fold_scores.items():
            mean = np.nanmean(values) if np.isfinite(values).any() else -np.inf
            if mean > best_score:
                best_score, best = mean, (s, lam, f_rbf)

    if best is None:
        return y[test_idx], np.full_like(y[test_idx], np.nan), float("nan")

    s, lam, f_rbf = best
    pred = _ridge_fit_predict(z, f_rbf, y_rows, train_rows, rows_of(test_idx), s, lam)
    mapped = pred.reshape(len(test_idx), n_time, n_out)
    return y[test_idx], mapped, per_channel_r2(y[test_idx], mapped)


# ------------------------------------------------------------------------------- MLP


def mlp_decode(x: np.ndarray, y: np.ndarray, hidden: int = 256) -> DecodeResult:
    """MLP decode of y from x (linear skip plus two GELU layers; settings in the module docstring)."""
    import torch
    from torch import nn

    x, y = as_trials_time_channels(x, y)
    train_idx, test_idx = split_trials(y.shape[0])
    n_time, n_in, n_out = y.shape[1], x.shape[-1], y.shape[-1]

    # Early stopping uses 20% of the training trials.
    fit_trials, val_trials = train_test_split(train_idx, test_size=0.2, random_state=0, shuffle=True)

    x_train, y_train = _rows(x, train_idx), _rows(y, train_idx)
    x_mu, x_sd = x_train.mean(0), x_train.std(0) + 1e-8
    y_mu, y_sd = y_train.mean(0), y_train.std(0) + 1e-8

    def to_x(a):
        return torch.tensor((a - x_mu) / x_sd, dtype=torch.float32)

    def to_y(a):
        return torch.tensor((a - y_mu) / y_sd, dtype=torch.float32)

    x_fit, y_fit = to_x(_rows(x, fit_trials)), to_y(_rows(y, fit_trials))
    x_val, y_val = to_x(_rows(x, val_trials)), to_y(_rows(y, val_trials))

    torch.manual_seed(0)

    class SkipMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(n_in, n_out)
            self.mlp = nn.Sequential(
                nn.Linear(n_in, hidden), nn.GELU(),
                nn.Linear(hidden, hidden), nn.GELU(),
                nn.Linear(hidden, n_out),
            )
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

        def forward(self, a):
            return self.linear(a) + self.mlp(a)

    net = SkipMLP()

    # Start the linear skip at the least-squares solution on the fitting rows. It is solved in
    # float64: inferred message channels can be nearly collinear, with the decodable signal in a
    # low-variance direction that a float32 solve loses.
    design = np.c_[(_rows(x, fit_trials) - x_mu) / x_sd, np.ones(len(fit_trials) * n_time)]
    target = (_rows(y, fit_trials) - y_mu) / y_sd
    solution = torch.tensor(np.linalg.lstsq(design, target, rcond=None)[0], dtype=torch.float32)
    with torch.no_grad():
        net.linear.weight.copy_(solution[:-1].T)
        net.linear.bias.copy_(solution[-1])

    optimizer = torch.optim.AdamW(net.parameters(), lr=MLP_LEARNING_RATE, weight_decay=MLP_WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    generator = torch.Generator().manual_seed(0)

    # The linear starting point is the first early-stopping candidate.
    net.eval()
    with torch.no_grad():
        best_loss = float(loss_fn(net(x_val), y_val))
    best_state = {name: p.clone() for name, p in net.state_dict().items()}
    epochs_without_gain = 0

    for _ in range(MLP_MAX_EPOCHS):
        net.train()
        order = torch.randperm(len(x_fit), generator=generator)
        for i in range(0, len(x_fit), MLP_BATCH):
            batch = order[i : i + MLP_BATCH]
            optimizer.zero_grad()
            loss_fn(net(x_fit[batch]), y_fit[batch]).backward()
            optimizer.step()

        net.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(net(x_val), y_val))
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = {name: p.clone() for name, p in net.state_dict().items()}
            epochs_without_gain = 0
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= MLP_PATIENCE:
                break

    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pred = net(to_x(_rows(x, test_idx))).numpy() * y_sd + y_mu

    mapped = pred.reshape(len(test_idx), n_time, n_out)
    return y[test_idx], mapped, per_channel_r2(y[test_idx], mapped)


# ------------------------------------------------------------------------------- lookup


def normalize_decoder_name(name: str) -> str:
    """Accept "kernel ridge", "kernel_ridge" or "kernel-ridge" (any case)."""
    key = name.strip().lower().replace("_", " ").replace("-", " ")
    if key not in DECODER_NAMES:
        raise ValueError(f"unknown decoder {name!r}; choose from {', '.join(DECODER_NAMES)}")
    return key


def get_decoder(name: str, direction: str = "fidelity") -> Callable[[np.ndarray, np.ndarray], DecodeResult]:
    """The decoder called `name`. `direction` ("fidelity" or "specificity") sets the MLP width.

    The returned function scores nan (with a warning) instead of failing when there are too few
    trials to split, so the other scores of a small file are still reported.
    """
    name = normalize_decoder_name(name)
    if name == "linear":
        decode = linear_decode
    elif name == "kernel ridge":
        decode = kernel_ridge_decode
    else:
        hidden = MLP_HIDDEN.get(direction, MLP_HIDDEN["fidelity"])

        def decode(x, y):
            return mlp_decode(x, y, hidden=hidden)

    def decode_or_nan(x, y):
        n_trials = np.shape(y)[0]
        if n_trials < MIN_TRIALS:
            warnings.warn(f"only {n_trials} trials: decoding scores are nan (need {MIN_TRIALS})", stacklevel=2)
            return np.asarray(y), np.full(np.shape(y), np.nan), float("nan")
        return decode(x, y)

    return decode_or_nan
