"""
Match-and-score evaluation: poor-man's offline test of RTG steering on logged
held-out trajectories. No simulator required.

Idea
----
For each target RTG t in the sweep:
  1. On every held-out context window, set the conditioning RTG slot(s) to t and
     get DT's top-K next-action predictions.
  2. Filter to windows where the logged next action is in the top-K ("match").
  3. Among matches, record the logged rating at that step.

Reports target_RTG vs (match rate, avg rating | matched, per-genre rating).

Two-confound layered analysis
-----------------------------
- "Overall avg rating | matched" is confounded by genre composition: if DT
  predicts Crime more at high RTG and Crime has higher mean rating than Comedy,
  the metric rises trivially with no per-user steering.
- The per-genre slice (for each genre G, mean rating when DT-top1 = G AND
  logged action = G) removes this particular composition difference. It remains
  a selected logged subset and is not a causal estimate of within-genre value.

Limitations
-----------
- Cannot evaluate predictions that don't match the logged action.
- Match rate at high RTG can drop sharply (smaller subset → noisier metric).
- Confounded by user-level selection bias: the held-out trajectory only contains
  items the user chose, not items they would have chosen if recommended
  differently.

Usage
-----
    python scripts/match_and_score.py \
        --checkpoint checkpoints/dt_movielens_current_state_seed42_best.pt \
        --mode last \
        --out results/match_and_score_movielens_seed42_last.csv
"""

import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.dataset import (
    GENRES, MAL_GENRES,
    build_trajectories, load_mal, load_movielens,
    resolve_checkpoint_max_rows,
    sample_trajectories, select_evaluation_user_ids,
    split_user_ids_three_way,
)
from src.model import RecDecisionTransformer
from src.diagnostics import (
    _overwrite_trailing_real_rtgs,
    _real_timestep_mask,
    _rtg_target_grid,
)


def build_eval_arrays(val_trajs, rtg_scale, context_len, state_dim,
                      pad_token=0, alignment='history'):
    """Build model arrays and source-user IDs for held-out prediction steps."""
    if alignment not in {'history', 'current_state'}:
        raise ValueError(
            f"alignment must be 'history' or 'current_state', got {alignment!r}"
        )
    all_states, all_actions, all_rtgs = [], [], []
    target_action, target_rating, traj_ids = [], [], []
    for traj_idx, traj in enumerate(val_trajs):
        T = len(traj['actions'])
        if T < 2:
            continue
        for t in range(1, T):
            end = t + 1 if alignment == 'current_state' else t
            start = max(0, end - context_len)
            L, pad = end - start, context_len - (end - start)
            s = torch.tensor(traj['states'][start:end], dtype=torch.float32)
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
            target_action.append(int(traj['actions'][t]))
            target_rating.append(float(traj['rewards'][t]))   # already in [0, 1]
            traj_ids.append(traj_idx)
    return (
        torch.stack(all_states),
        torch.stack(all_actions),
        torch.stack(all_rtgs),
        np.array(target_action, dtype=np.int64),
        np.array(target_rating, dtype=np.float32),
        np.array(traj_ids, dtype=np.int32),
    )


def _bootstrap_selected_metrics(
    match_top1: np.ndarray,
    match_topk: np.ndarray,
    target_rating: np.ndarray,
    traj_ids: np.ndarray,
    rating_scale: float,
    n_boot: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    """User-cluster percentile intervals for selection-based logged metrics."""
    if n_boot < 1:
        nan = (float('nan'), float('nan'))
        return {
            'match_rate_top1': nan,
            'match_rate_topk': nan,
            'avg_rating_top1_match': nan,
            'avg_rating_topk_match': nan,
        }
    _, inverse = np.unique(traj_ids, return_inverse=True)
    n_users = int(inverse.max()) + 1
    totals = np.bincount(inverse, minlength=n_users).astype(np.float64)
    count1 = np.bincount(
        inverse, weights=match_top1.astype(float), minlength=n_users
    )
    countk = np.bincount(
        inverse, weights=match_topk.astype(float), minlength=n_users
    )
    rating1 = np.bincount(
        inverse, weights=target_rating * match_top1, minlength=n_users
    )
    ratingk = np.bincount(
        inverse, weights=target_rating * match_topk, minlength=n_users
    )
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, n_users, size=(n_boot, n_users))
    total_boot = totals[sampled].sum(axis=1)
    count1_boot = count1[sampled].sum(axis=1)
    countk_boot = countk[sampled].sum(axis=1)
    metrics = {
        'match_rate_top1': count1_boot / total_boot,
        'match_rate_topk': countk_boot / total_boot,
        'avg_rating_top1_match': np.divide(
            rating1[sampled].sum(axis=1) * rating_scale,
            count1_boot,
            out=np.full(n_boot, np.nan),
            where=count1_boot > 0,
        ),
        'avg_rating_topk_match': np.divide(
            ratingk[sampled].sum(axis=1) * rating_scale,
            countk_boot,
            out=np.full(n_boot, np.nan),
            where=countk_boot > 0,
        ),
    }
    intervals = {}
    for name, values in metrics.items():
        bounds = np.asarray(np.nanpercentile(values, [2.5, 97.5]))
        intervals[name] = (float(bounds[0]), float(bounds[1]))
    return intervals


def sweep(model, states, actions, rtgs_base, target_action, target_rating,
          traj_ids, rtg_targets, rtg_scale, context_len, mode, top_k,
          categories, device, rating_scale, batch_size=2048,
          n_boot=0, bootstrap_seed=0):
    n_categories = len(categories)
    N = states.shape[0]
    timesteps = torch.arange(context_len, dtype=torch.long).unsqueeze(0)

    records = []
    real_mask = _real_timestep_mask(states, rtgs_base)
    for target_raw in rtg_targets:
        target_scaled = float(target_raw) / rtg_scale
        if mode == 'all':
            rtgs = _overwrite_trailing_real_rtgs(
                rtgs_base, real_mask, target_scaled, k=None
            )
        else:
            rtgs = _overwrite_trailing_real_rtgs(
                rtgs_base, real_mask, target_scaled, k=1
            )

        all_topk = []
        with torch.no_grad():
            for i in range(0, N, batch_size):
                s = states[i:i + batch_size].to(device)
                a = actions[i:i + batch_size].to(device)
                r = rtgs[i:i + batch_size].to(device)
                ts = timesteps.expand(s.shape[0], -1).to(device)
                mask = real_mask[i:i + batch_size].to(device)
                logits = model(
                    s, a, r, ts, attention_mask=mask
                )[:, -1, :]
                topk = logits.topk(top_k, dim=-1).indices.cpu().numpy()
                all_topk.append(topk)
        all_topk = np.concatenate(all_topk, axis=0)               # (N, top_k)
        top1 = all_topk[:, 0]

        match_topk = (all_topk == target_action[:, None]).any(axis=1)
        match_top1 = (top1 == target_action)

        record = {
            'target_rtg':      float(target_raw),
            'n_total':         int(N),
            'match_rate_top1': float(match_top1.mean()),
            'match_rate_topk': float(match_topk.mean()),
            'n_match_top1':    int(match_top1.sum()),
            'n_match_topk':    int(match_topk.sum()),
            'avg_rating_top1_match':
                float(target_rating[match_top1].mean() * rating_scale)
                if match_top1.sum() > 0 else float('nan'),
            'avg_rating_topk_match':
                float(target_rating[match_topk].mean() * rating_scale)
                if match_topk.sum() > 0 else float('nan'),
            'overall_avg_rating': float(target_rating.mean() * rating_scale),
        }
        intervals = _bootstrap_selected_metrics(
            match_top1, match_topk, target_rating, traj_ids,
            rating_scale=rating_scale, n_boot=n_boot, seed=bootstrap_seed,
        )
        for metric, (lo, hi) in intervals.items():
            record[f'{metric}_lo'] = lo
            record[f'{metric}_hi'] = hi

        # Per-genre slice: top1 prediction == logged action == G
        for g_idx, g_name in enumerate(categories):
            mask = (top1 == g_idx) & (target_action == g_idx)
            n = int(mask.sum())
            record[f'pergenre_n_{g_name}'] = n
            record[f'pergenre_avg_rating_{g_name}'] = (
                float(target_rating[mask].mean() * rating_scale) if n > 0 else float('nan')
            )

        records.append(record)
        print(
            f"  target_RTG={target_raw:5.2f}  "
            f"top1_match={match_top1.mean():.4f} (n={match_top1.sum():,})  "
            f"top{top_k}_match={match_topk.mean():.4f}  "
            f"avg_rating|top1={record['avg_rating_top1_match']:.4f}  "
            f"avg_rating|top{top_k}={record['avg_rating_topk_match']:.4f}",
            flush=True,
        )

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--ratings',    default=None)
    parser.add_argument('--movies',     default=None)
    parser.add_argument('--mode',       choices=['all', 'last'], default='last',
                        help="RTG override mode: 'last' is single-slot, "
                             "'all' is context-saturating")
    parser.add_argument('--top-k',      type=int,   default=5)
    parser.add_argument('--n-boot', type=int, default=1000,
                        help='User-cluster bootstrap replicates for 95%% CIs.')
    parser.add_argument('--bootstrap-seed', type=int, default=0)
    parser.add_argument('--n-levels',   type=int,   default=11)
    parser.add_argument('--rtg-min',    type=float, default=None)
    parser.add_argument('--rtg-max',    type=float, default=None)
    parser.add_argument('--rtg-low-quantile', type=float, default=0.05)
    parser.add_argument('--rtg-high-quantile', type=float, default=0.95)
    parser.add_argument('--traj-limit', type=int,   default=2000)
    parser.add_argument('--sample-seed', type=int, default=42,
                        help='Seed for random trajectory subsampling.')
    parser.add_argument(
        '--split', choices=['auto', 'val', 'test'], default='val',
        help='Reporting cohort. Use val while developing and test once after '
             'the protocol is frozen.',
    )
    parser.add_argument('--max-rows',   type=int,   default=None)
    parser.add_argument('--out',        default='results/match_and_score.csv')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg          = ckpt['config']
    state_dim    = ckpt['state_dim']
    n_categories = ckpt['n_categories']
    categories   = ckpt.get('categories', GENRES)
    if len(categories) != n_categories:
        raise ValueError(
            f"checkpoint category metadata has {len(categories)} labels but "
            f"n_categories={n_categories}"
        )
    if not 1 <= args.top_k < n_categories:
        raise ValueError(
            f"top-k must be in [1, {n_categories - 1}] to avoid a trivial match"
        )

    if cfg.get('model', 'dt') != 'dt' or not cfg.get('use_rtg', True):
        raise ValueError("This script requires a DT+RTG checkpoint.")

    print(f"Checkpoint   : {args.checkpoint}")
    print(f"Mode         : {args.mode}")
    context_len = cfg.get('context_len', 20)
    rtg_scale   = cfg.get('rtg_scale', 20.0)
    rtg_window  = cfg.get('rtg_window', 20)
    alignment   = cfg.get('alignment', 'history')
    rating_scale = cfg.get('rating_scale', 5.0)

    model = RecDecisionTransformer(
        state_dim=state_dim, n_actions=n_categories,
        hidden_size=cfg.get('hidden_size', 128),
        max_length=context_len,
        n_layer=cfg.get('n_layer', 3),
        n_head=cfg.get('n_head', 4),
        dropout=0.0, use_rtg=True,
        use_pad_token=cfg.get('use_pad_token', False),
    )
    model.load_state_dict(ckpt['model_state'])
    model = model.to(device)
    model.eval()

    seed = ckpt.get('seed', 42)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    min_interactions = cfg.get('min_interactions', 10)
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
    if eval_split == 'test':
        grid_trajs = build_trajectories(
            df[df['userId'].isin(val_ids)],
            rtg_window=rtg_window, n_categories=n_categories,
            min_interactions=min_interactions,
            max_t_override=train_max_T,
            rating_scale=rating_scale,
            drop_incomplete_rtg=drop_incomplete_rtg,
        )
        grid_split = 'val'
    else:
        grid_trajs = eval_trajs
        grid_split = eval_split
    eval_trajs = sample_trajectories(
        eval_trajs, args.traj_limit, seed=args.sample_seed
    )
    print(f"  {len(eval_trajs):,} {eval_split} trajectories used")

    print("  Building eval arrays...")
    pad_token = n_categories if cfg.get('use_pad_token', False) else 0
    states, actions, rtgs_base, target_action, target_rating, traj_ids = build_eval_arrays(
        eval_trajs, rtg_scale, context_len, state_dim, pad_token=pad_token,
        alignment=alignment
    )
    N = states.shape[0]
    print(f"  {N:,} eval steps")

    rtg_targets = _rtg_target_grid(
        grid_trajs, args.n_levels, args.rtg_min, args.rtg_max,
        low_quantile=args.rtg_low_quantile,
        high_quantile=args.rtg_high_quantile,
    )
    print(f"\nSweeping RTG {rtg_targets[0]:.2f} → {rtg_targets[-1]:.2f} "
          f"in {len(rtg_targets)} steps  (mode={args.mode}, top_k={args.top_k})")

    df_results = sweep(
        model, states, actions, rtgs_base, target_action, target_rating,
        traj_ids, rtg_targets, rtg_scale, context_len, args.mode, args.top_k,
        categories, device, rating_scale, n_boot=args.n_boot,
        bootstrap_seed=args.bootstrap_seed,
    )
    df_results.insert(0, 'evaluation_split', eval_split)
    df_results.insert(1, 'rtg_grid_split', grid_split)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    df_results.to_csv(args.out, index=False)
    print(f"\nResults → {args.out}")

    # Summary table: target_RTG, match rates, overall avg rating
    print("\n=== Summary ===")
    summary_cols = [
        'target_rtg', 'match_rate_top1', 'match_rate_topk',
        'avg_rating_top1_match', 'avg_rating_topk_match',
    ]
    print(df_results[summary_cols].to_string(index=False, float_format='%.4f'))

    # Per-genre breakdown for genres with at least 50 matches at any RTG
    pergenre_n_cols = [c for c in df_results.columns if c.startswith('pergenre_n_')]
    pergenre_n_max  = df_results[pergenre_n_cols].max()
    keep_genres = [c.replace('pergenre_n_', '') for c, n in pergenre_n_max.items() if n >= 50]
    if keep_genres:
        print("\n=== Per-genre avg rating (top-1 match, n_max ≥ 50) ===")
        cols = ['target_rtg'] + [f'pergenre_avg_rating_{g}' for g in keep_genres]
        print(df_results[cols].to_string(index=False, float_format='%.4f'))


if __name__ == '__main__':
    main()
