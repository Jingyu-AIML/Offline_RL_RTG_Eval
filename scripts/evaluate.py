"""Evaluate the paper's next-genre hit rates with user-cluster intervals.

    python scripts/evaluate.py --checkpoint CHECKPOINT --split test --out METRICS.csv
"""

import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.dataset import (
    GENRES,
    MAL_GENRES,
    SASRecDataset,
    TrajectoryDataset,
    build_trajectories,
    load_mal,
    load_movielens,
    resolve_checkpoint_max_rows,
    select_evaluation_user_ids,
    split_user_ids_three_way,
)
from src.model import RecDecisionTransformer, SASRec


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def hit_rate_at_k(logits: np.ndarray, targets: np.ndarray, k: int) -> float:
    if logits.ndim != 2 or targets.ndim != 1 or len(logits) != len(targets):
        raise ValueError("logits must be (N, C) and targets must be (N,)")
    if not 1 <= k < logits.shape[1]:
        raise ValueError(
            f"k must be in [1, {logits.shape[1] - 1}]; k={k} is invalid "
            "or trivially includes every category"
        )
    top_k = np.argsort(-logits, axis=1)[:, :k]
    hits  = (top_k == targets[:, None]).any(axis=1)
    return float(hits.mean())


def trajectory_ids_for_samples(trajectories: list[dict]) -> np.ndarray:
    """Dataset order is trajectory-major with T-1 prediction steps per user."""
    return np.concatenate([
        np.full(len(traj['actions']) - 1, i, dtype=np.int32)
        for i, traj in enumerate(trajectories)
        if len(traj['actions']) >= 2
    ])


def cluster_bootstrap_mean(
    values: np.ndarray,
    traj_ids: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile CI after resampling users, preserving their full histories."""
    if n_boot < 1:
        return float('nan'), float('nan')
    if len(values) != len(traj_ids):
        raise ValueError("values and traj_ids must have the same length")
    unique, inverse = np.unique(traj_ids, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    sums = np.bincount(inverse, weights=values).astype(np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(unique), size=(n_boot, len(unique)))
    boot_means = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    bounds = np.asarray(np.percentile(boot_means, [2.5, 97.5]))
    return float(bounds[0]), float(bounds[1])




# ---------------------------------------------------------------------------
# Per-model scoring
# ---------------------------------------------------------------------------

def score_dt(model, val_trajs, context_len, rtg_scale, batch_size, device,
             pad_token=0, alignment='history'):
    val_ds = TrajectoryDataset(val_trajs, context_len=context_len,
                               rtg_scale=rtg_scale, pad_token=pad_token,
                               alignment=alignment)
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    all_logits, all_targets = [], []
    model.eval()
    with torch.no_grad():
        for states, actions, rtgs, timesteps, targets in loader:
            logits = model(states.to(device), actions.to(device),
                           rtgs.to(device), timesteps.to(device))
            all_logits.append(logits[:, -1, :].cpu().numpy())
            all_targets.append(targets.numpy())
    return np.concatenate(all_logits), np.concatenate(all_targets)


def score_sasrec(model, val_trajs, context_len, batch_size, device, pad_token=0):
    val_ds = SASRecDataset(val_trajs, context_len=context_len, pad_token=pad_token)
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    all_logits, all_targets = [], []
    model.eval()
    with torch.no_grad():
        for actions, targets in loader:
            logits = model(actions.to(device))
            all_logits.append(logits[:, -1, :].cpu().numpy())
            all_targets.append(targets.numpy())
    return np.concatenate(all_logits), np.concatenate(all_targets)




# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--ratings',    default=None, help='Override ratings.csv path')
    parser.add_argument('--movies',     default=None, help='Override movies.csv path')
    parser.add_argument('--k',          nargs='+', type=int, default=[1, 5, 10])
    parser.add_argument('--max-rows',   type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--n-boot', type=int, default=1000,
                        help='User-cluster bootstrap replicates for 95%% CIs.')
    parser.add_argument('--bootstrap-seed', type=int, default=0)
    parser.add_argument('--out', default=None,
                        help='Optional CSV path for machine-readable metrics.')
    parser.add_argument(
        '--split', choices=['auto', 'val', 'test'], default='val',
        help='Reporting cohort. Use val during development and explicitly pass '
             'test once the protocol is frozen.',
    )
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    ckpt         = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg          = ckpt['config']
    state_dim    = ckpt['state_dim']
    n_categories = ckpt['n_categories']
    model_type   = cfg.get('model', 'dt')
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Model type : {model_type} | state_dim={state_dim} | n_genres={n_categories}")

    if model_type not in {'dt', 'sasrec'}:
        raise ValueError('This package supports DT and SASRec checkpoints only.')
    if model_type == 'sasrec':
        model = SASRec(
            n_actions     = n_categories,
            hidden_size   = cfg.get('hidden_size', 128),
            max_length    = cfg.get('context_len', 20),
            n_layer       = cfg.get('n_layer', 2),
            n_head        = cfg.get('n_head', 4),
            dropout       = cfg.get('dropout', 0.2),
            use_pad_token = cfg.get('use_pad_token', False),
        )
    else:
        model = RecDecisionTransformer(
            state_dim=state_dim, n_actions=n_categories,
            hidden_size=cfg.get('hidden_size', 128),
            max_length=cfg.get('context_len', 20),
            n_layer=cfg.get('n_layer', 3),
            n_head=cfg.get('n_head', 4),
            dropout=cfg.get('dropout', 0.1),
            use_rtg=cfg.get('use_rtg', True),
            use_pad_token=cfg.get('use_pad_token', False),
        )
    model.load_state_dict(ckpt['model_state'])
    model = model.to(device)
    model.eval()

    seed = ckpt.get('seed', 42)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    rtg_window       = cfg.get('rtg_window', 20)
    min_interactions = cfg.get('min_interactions', 10)
    rating_scale     = cfg.get('rating_scale', 5.0)
    dataset          = cfg.get('dataset', 'movielens')
    effective_max_rows = resolve_checkpoint_max_rows(ckpt, args.max_rows)

    if dataset == 'mal':
        anime_path   = cfg.get('anime_path',   'data/mal/anime.csv')
        ratings_path = args.ratings or cfg.get('ratings_path', 'data/mal/animelist.csv')
        print(f"\nLoading MyAnimeList from {ratings_path}...")
        df = load_mal(anime_path, ratings_path)
    else:
        ratings_path = args.ratings or cfg.get('ratings_path', 'data/movielens/ml-25m/ratings.csv')
        movies_path  = args.movies  or cfg.get('movies_path',  'data/movielens/ml-25m/movies.csv')
        print(f"\nLoading MovieLens from {ratings_path}...")
        df = load_movielens(ratings_path, movies_path, nrows=effective_max_rows,
                            genre_selection=cfg.get('genre_selection', 'rarest'))

    split_strategy = cfg.get('user_split', 'tail')
    split_seed = cfg.get('split_seed', 42)
    val_fraction = cfg.get('val_fraction', 0.1)
    test_fraction = cfg.get('test_fraction', 0.0)
    drop_incomplete_rtg = cfg.get('drop_incomplete_rtg', False)
    train_ids, val_ids, test_ids = split_user_ids_three_way(
        df, val_fraction=val_fraction, test_fraction=test_fraction, seed=split_seed,
        strategy=split_strategy,
    )
    eval_ids, eval_split = select_evaluation_user_ids(
        val_ids, test_ids, args.split
    )
    train_df = df[df['userId'].isin(train_ids)]
    train_max_T = float(train_df.groupby('userId').size().max())
    eval_trajs = build_trajectories(
        df[df['userId'].isin(eval_ids)],
        rtg_window=rtg_window, n_categories=n_categories,
        min_interactions=min_interactions,
        max_t_override=train_max_T,
        rating_scale=rating_scale,
        drop_incomplete_rtg=drop_incomplete_rtg,
    )
    print(f"{eval_split.title()} trajectories: {len(eval_trajs):,}")

    context_len = cfg.get('context_len', 20)
    rtg_scale   = cfg.get('rtg_scale', 20.0)
    alignment   = cfg.get('alignment', 'history')

    pad_token = n_categories if cfg.get('use_pad_token', False) else 0
    invalid_k = [k for k in args.k if not 1 <= k < n_categories]
    if invalid_k:
        raise ValueError(
            f"uninformative/invalid K values {invalid_k}; choose K in "
            f"[1, {n_categories - 1}] for {n_categories} categories"
        )

    # Hit Rate @ K
    print("\nComputing Hit Rate @ K...")
    if model_type == 'sasrec':
        logits, targets = score_sasrec(model, eval_trajs, context_len, args.batch_size, device,
                                       pad_token=pad_token)
    elif model_type == 'dt':
        logits, targets = score_dt(model, eval_trajs, context_len, rtg_scale, args.batch_size, device,
                                   pad_token=pad_token, alignment=alignment)

    traj_ids = trajectory_ids_for_samples(eval_trajs)
    if len(traj_ids) != len(targets):
        raise RuntimeError(
            f"sample/trajectory mismatch: {len(targets)} predictions, "
            f"{len(traj_ids)} trajectory IDs"
        )

    records = []
    print(f"\n{'K':>6}  {'Hit Rate':>10}  {'95% cluster CI':>23}")
    print("-" * 45)
    for k in args.k:
        top_k = np.argsort(-logits, axis=1)[:, :k]
        hit_values = (top_k == targets[:, None]).any(axis=1).astype(np.float64)
        value = float(hit_values.mean())
        ci_lo, ci_hi = cluster_bootstrap_mean(
            hit_values, traj_ids, n_boot=args.n_boot,
            seed=args.bootstrap_seed + k,
        )
        print(f"{k:>6}  {value:>10.4f}  [{ci_lo:.4f}, {ci_hi:.4f}]")
        records.append({
            'checkpoint': os.path.basename(args.checkpoint),
            'seed': seed,
            'evaluation_split': eval_split,
            'metric': 'hit_rate',
            'k': k,
            'value': value,
            'ci_low': ci_lo,
            'ci_high': ci_hi,
            'n_predictions': len(targets),
            'n_users': len(eval_trajs),
        })

    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        pd.DataFrame(records).to_csv(args.out, index=False)
        print(f"Metrics → {args.out}")


if __name__ == '__main__':
    main()
