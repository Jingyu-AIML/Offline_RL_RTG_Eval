"""Shared context, RTG intervention, and user-cluster bootstrap helpers."""

import numpy as np
import torch


def _assert_comparable_protocols(primary_cfg, control_cfg):
    """Reject control checkpoints trained on a different data protocol."""
    defaults = {
        'protocol': None,
        'dataset': 'movielens',
        'ratings_path': None,
        'movies_path': None,
        'anime_path': None,
        'genre_selection': 'rarest',
        'rtg_window': 20,
        'rtg_scale': 20.0,
        'min_interactions': 10,
        'rating_scale': 5.0,
        'drop_incomplete_rtg': False,
        'user_split': 'tail',
        'split_seed': 42,
        'val_fraction': 0.1,
        'test_fraction': 0.0,
        'alignment': 'history',
        'use_pad_token': False,
        'context_len': 20,
        'val_max_batches': 1000,
    }
    mismatches = []
    for key, default in defaults.items():
        primary_value = primary_cfg.get(key, default)
        control_value = control_cfg.get(key, default)
        if primary_value != control_value:
            mismatches.append(
                f"{key}: primary={primary_value!r}, control={control_value!r}"
            )
    if mismatches:
        raise ValueError(
            "primary and control checkpoints use different protocols: "
            + "; ".join(mismatches)
        )


def _assert_comparable_checkpoints(primary_ckpt, control_ckpt):
    """Require a paired control differing only in its intended RTG ablation."""
    _assert_comparable_protocols(primary_ckpt['config'], control_ckpt['config'])
    mismatches = []
    for key in (
        'state_dim', 'n_categories', 'categories', 'seed', 'data_max_rows',
    ):
        primary_value = primary_ckpt.get(key)
        control_value = control_ckpt.get(key)
        if primary_value != control_value:
            mismatches.append(
                f"{key}: primary={primary_value!r}, control={control_value!r}"
            )
    architecture_defaults = {
        'hidden_size': 128,
        'n_layer': 3,
        'n_head': 4,
        'dropout': 0.1,
        'max_steps': 50000,
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'warmup_steps': 1000,
        'batch_size': 512,
    }
    for key, default in architecture_defaults.items():
        primary_value = primary_ckpt['config'].get(key, default)
        control_value = control_ckpt['config'].get(key, default)
        if primary_value != control_value:
            mismatches.append(
                f"{key}: primary={primary_value!r}, control={control_value!r}"
            )
    if not primary_ckpt['config'].get('use_rtg', True):
        mismatches.append("primary checkpoint does not use RTG")
    if control_ckpt['config'].get('use_rtg', True):
        mismatches.append("control checkpoint still uses RTG")
    if mismatches:
        raise ValueError(
            "primary and control checkpoints are not a paired comparison: "
            + "; ".join(mismatches)
        )


def _build_context_batch(val_trajs, rtg_scale, context_len, state_dim,
                         pad_token=0, alignment='history'):
    """
    Returns (states, actions, rtgs, traj_ids) where traj_ids[i] gives the
    source trajectory index for the i-th context window. Used by the bootstrap
    to resample at the trajectory level (cluster bootstrap), since predictions
    within a trajectory share overlapping context windows and are not iid.

    pad_token: id used to fill action positions during left-padding. 0 = legacy;
    n_categories when paired with `use_pad_token=True` on the model.

    alignment:
      'history'       legacy protocol: inputs end before target action t.
      'current_state' include state[t] and rtg[t], with final action set to
                      pad_token so the target action is not supplied.
    """
    if alignment not in {'history', 'current_state'}:
        raise ValueError(
            f"alignment must be 'history' or 'current_state', got {alignment!r}"
        )
    all_states, all_actions, all_rtgs, traj_ids = [], [], [], []
    for j, traj in enumerate(val_trajs):
        T = len(traj['actions'])
        if T < 2:
            continue
        for t in range(1, T):
            end = t + 1 if alignment == 'current_state' else t
            start = max(0, end - context_len)
            L, pad = end - start, context_len - (end - start)
            s = torch.tensor(traj['states'][start:end],  dtype=torch.float32)
            a = torch.tensor(traj['actions'][start:end], dtype=torch.long)
            r = torch.tensor(traj['rtgs'][start:end] / rtg_scale, dtype=torch.float32).unsqueeze(-1)
            if alignment == 'current_state':
                a[-1] = pad_token
            if pad > 0:
                s = torch.cat([torch.zeros(pad, state_dim), s], dim=0)
                a = torch.cat([torch.full((pad,), pad_token, dtype=torch.long), a], dim=0)
                r = torch.cat([torch.zeros(pad, 1), r], dim=0)
            all_states.append(s)
            all_actions.append(a)
            all_rtgs.append(r)
            traj_ids.append(j)
    return (
        torch.stack(all_states),
        torch.stack(all_actions),
        torch.stack(all_rtgs),
        np.asarray(traj_ids, dtype=np.int32),
    )


def _bootstrap_genre_ci(buckets: np.ndarray, traj_ids: np.ndarray,
                        n_categories: int, n_boot: int = 1000,
                        alpha: float = 0.05, seed: int = 0):
    """
    Cluster bootstrap CI on per-genre share. Resamples trajectories with
    replacement (not individual predictions), since predictions within a
    trajectory share overlapping context windows.

    Returns (lo, hi): each shape (n_categories,), the alpha/2 and 1-alpha/2
    percentiles of the per-genre share across n_boot resamples.
    """
    rng = np.random.default_rng(seed)
    unique_trajs, inverse = np.unique(traj_ids, return_inverse=True)
    M = len(unique_trajs)
    counts_by_user = np.zeros((M, n_categories), dtype=np.int64)
    np.add.at(counts_by_user, (inverse, buckets), 1)
    # Multinomial weights are exactly the counts induced by drawing M users
    # with replacement, but matrix multiplication is much faster than
    # concatenating every resampled user's prediction indices.
    weights = rng.multinomial(
        M, np.full(M, 1.0 / M), size=n_boot
    )
    boot_counts = weights @ counts_by_user
    boot_props = boot_counts / boot_counts.sum(axis=1, keepdims=True)
    lo = np.percentile(boot_props, 100 * alpha / 2,       axis=0)
    hi = np.percentile(boot_props, 100 * (1 - alpha / 2), axis=0)
    return lo, hi


def _bootstrap_genre_delta_ci(
    low_buckets: np.ndarray,
    high_buckets: np.ndarray,
    traj_ids: np.ndarray,
    n_categories: int,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
):
    """Paired user-cluster CI for high-target minus low-target genre share."""
    if len(low_buckets) != len(high_buckets) or len(low_buckets) != len(traj_ids):
        raise ValueError("low/high predictions and traj_ids must have equal length")
    rng = np.random.default_rng(seed)
    unique_trajs, inverse = np.unique(traj_ids, return_inverse=True)
    n_users = len(unique_trajs)
    low_by_user = np.zeros((n_users, n_categories), dtype=np.int64)
    high_by_user = np.zeros((n_users, n_categories), dtype=np.int64)
    np.add.at(low_by_user, (inverse, low_buckets), 1)
    np.add.at(high_by_user, (inverse, high_buckets), 1)
    weights = rng.multinomial(
        n_users, np.full(n_users, 1.0 / n_users), size=n_boot
    )
    low_counts = weights @ low_by_user
    high_counts = weights @ high_by_user
    denominators = low_counts.sum(axis=1, keepdims=True)
    boot_delta = high_counts / denominators - low_counts / denominators
    lo = np.percentile(boot_delta, 100 * alpha / 2, axis=0)
    hi = np.percentile(boot_delta, 100 * (1 - alpha / 2), axis=0)
    return lo, hi


def _real_timestep_mask(states: torch.Tensor, rtgs: torch.Tensor) -> torch.Tensor:
    """Identify right-aligned real timesteps before applying an RTG intervention."""
    return states.abs().sum(dim=-1).ne(0) | rtgs.abs().sum(dim=-1).ne(0)


def _overwrite_trailing_real_rtgs(
    rtgs: torch.Tensor,
    real_mask: torch.Tensor,
    target_scaled: float,
    k: int | None,
) -> torch.Tensor:
    """Overwrite the last ``k`` real positions, never left-padding positions."""
    if rtgs.ndim != 3 or rtgs.shape[-1] != 1:
        raise ValueError(f"rtgs must have shape (B, T, 1), got {tuple(rtgs.shape)}")
    if real_mask.shape != rtgs.shape[:2]:
        raise ValueError(
            f"real_mask must have shape {tuple(rtgs.shape[:2])}, "
            f"got {tuple(real_mask.shape)}"
        )
    if k is not None and not 1 <= k <= rtgs.shape[1]:
        raise ValueError(f"k must be in [1, {rtgs.shape[1]}] or None, got {k}")

    selected = real_mask
    if k is not None:
        rank_from_right = torch.flip(
            torch.cumsum(torch.flip(real_mask.to(torch.int64), dims=[1]), dim=1),
            dims=[1],
        )
        selected = real_mask & rank_from_right.le(k)

    out = rtgs.clone()
    out[..., 0][selected] = target_scaled
    return out


def _rtg_target_grid(
    trajectories,
    n_levels: int,
    rtg_min: float | None,
    rtg_max: float | None,
    low_quantile: float = 0.05,
    high_quantile: float = 0.95,
) -> np.ndarray:
    """Build an in-support target grid unless explicit endpoints are supplied."""
    if n_levels < 2:
        raise ValueError(f"n_levels must be at least 2, got {n_levels}")
    if not 0 <= low_quantile < high_quantile <= 1:
        raise ValueError("RTG quantiles must satisfy 0 <= low < high <= 1")
    if not trajectories:
        raise ValueError("cannot construct an RTG grid from no trajectories")
    observed = np.concatenate([t['rtgs'] for t in trajectories])
    low = float(np.quantile(observed, low_quantile)) if rtg_min is None else rtg_min
    high = float(np.quantile(observed, high_quantile)) if rtg_max is None else rtg_max
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        raise ValueError(
            f"RTG grid endpoints must be finite with low < high, got {low}, {high}"
        )
    print(
        f"Observed RTG support: min={observed.min():.3f}, "
        f"q{low_quantile:.2f}={np.quantile(observed, low_quantile):.3f}, "
        f"q{high_quantile:.2f}={np.quantile(observed, high_quantile):.3f}, "
        f"max={observed.max():.3f}"
    )
    return np.linspace(low, high, n_levels)


def get_no_rtg_distribution(model, states_all, actions_all, rtgs_base,
                             device, categories, batch_size=2048):
    """Run no-RTG model once on the context windows (RTG input is ignored internally)."""
    model.eval()
    N = states_all.shape[0]
    context_len = states_all.shape[1]
    timesteps = torch.arange(context_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    real_mask = _real_timestep_mask(states_all, rtgs_base)
    all_preds = []
    with torch.no_grad():
        for i in range(0, N, batch_size):
            s  = states_all[i:i + batch_size].to(device)
            a  = actions_all[i:i + batch_size].to(device)
            r  = rtgs_base[i:i + batch_size].to(device)   # value irrelevant — zeroed in model
            ts = timesteps[:s.shape[0]].to(device)
            mask = real_mask[i:i + batch_size].to(device)
            logits = model(s, a, r, ts, attention_mask=mask)
            all_preds.append(logits[:, -1, :].argmax(dim=-1).cpu())
    buckets = torch.cat(all_preds).numpy().astype(np.int32)
    genre_counts = np.bincount(buckets, minlength=len(categories))
    return {g: float(genre_counts[i]) / len(buckets) for i, g in enumerate(categories)}
