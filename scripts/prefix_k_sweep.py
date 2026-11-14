"""
RTG locality ladder sweep: interpolates between single-slot (K=1,
"last") and context-saturating (K=context_len, "all") RTG override modes.
K counts real timesteps; left padding is never overwritten.

Runs the RTG sweep at K = {1, 5, 10, 15, 20} on a single DT+RTG checkpoint,
loads data + model once, and produces:
  - Combined per-K CSV with full RTG curves
  - RTG-locality figure: x=K, y=Δ focus-genre share (validation q95 minus q05)
  - Per-K Crime-share curves on a shared figure

This is the central figure for the negative-result paper: it shows how
"controllability magnitude" decays from a context-saturating sweep to the
single-slot/current-decision intervention.

Usage
-----
    python scripts/prefix_k_sweep.py \\
        --checkpoint checkpoints/dt_movielens_current_state_seed42_best.pt \\
        --traj-limit 1000 --out results/
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
    _build_context_batch,
    _assert_comparable_checkpoints,
    _bootstrap_genre_ci,
    _bootstrap_genre_delta_ci,
    _overwrite_trailing_real_rtgs,
    _real_timestep_mask,
    _rtg_target_grid,
    get_no_rtg_distribution,
)


def run_one_k(model, states_all, actions_all, rtgs_base, rtg_targets, rtg_scale,
              context_len, device, categories, K, traj_ids, n_boot=0,
              batch_size=2048):
    """Run the RTG sweep with the last-K context positions overridden."""
    N = states_all.shape[0]
    real_mask = _real_timestep_mask(states_all, rtgs_base)
    timesteps = torch.arange(context_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    records = []
    endpoint_buckets = {}
    for target_raw in rtg_targets:
        target_scaled = float(target_raw) / rtg_scale
        rtgs = _overwrite_trailing_real_rtgs(
            rtgs_base, real_mask, target_scaled, k=K
        )

        all_preds = []
        with torch.no_grad():
            for i in range(0, N, batch_size):
                s  = states_all[i:i + batch_size].to(device)
                a  = actions_all[i:i + batch_size].to(device)
                r  = rtgs[i:i + batch_size].to(device)
                ts = timesteps[:s.shape[0]].to(device)
                mask = real_mask[i:i + batch_size].to(device)
                logits = model(s, a, r, ts, attention_mask=mask)
                all_preds.append(logits[:, -1, :].argmax(dim=-1).cpu())

        buckets = torch.cat(all_preds).numpy().astype(np.int32)
        if target_raw == rtg_targets[0] or target_raw == rtg_targets[-1]:
            endpoint_buckets[float(target_raw)] = buckets
        genre_counts = np.bincount(buckets, minlength=len(categories))
        record = {
            'K':            K,
            'rtg_target':   float(target_raw),
            'mode_genre':   categories[int(genre_counts.argmax())],
            'n_samples':    len(buckets),
        }
        for i, g in enumerate(categories):
            record[f'pct_{g}'] = float(genre_counts[i]) / len(buckets)

        if n_boot > 0:
            lo, hi = _bootstrap_genre_ci(buckets, traj_ids, len(categories),
                                          n_boot=n_boot,
                                          seed=(int(target_raw * 1000) + K) & 0xFFFF)
            for i, g in enumerate(categories):
                record[f'pct_{g}_lo'] = float(lo[i])
                record[f'pct_{g}_hi'] = float(hi[i])

        records.append(record)

    low_buckets = endpoint_buckets[float(rtg_targets[0])]
    high_buckets = endpoint_buckets[float(rtg_targets[-1])]
    low_counts = np.bincount(low_buckets, minlength=len(categories))
    high_counts = np.bincount(high_buckets, minlength=len(categories))
    delta = high_counts / len(high_buckets) - low_counts / len(low_buckets)
    if n_boot > 0:
        delta_lo, delta_hi = _bootstrap_genre_delta_ci(
            low_buckets, high_buckets, traj_ids, len(categories),
            n_boot=n_boot, seed=10_000 + K,
        )
    else:
        delta_lo = np.full(len(categories), np.nan)
        delta_hi = np.full(len(categories), np.nan)
    for record in records:
        for i, genre in enumerate(categories):
            record[f'delta_pct_{genre}'] = float(delta[i])
            record[f'delta_pct_{genre}_lo'] = float(delta_lo[i])
            record[f'delta_pct_{genre}_hi'] = float(delta_hi[i])
    return records


def plot_realism_ladder(df, out_path, focus_genre='Crime', no_rtg_share=None):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot")
        return

    pct_col = f'pct_{focus_genre}'
    if pct_col not in df.columns:
        print(f"Genre {focus_genre!r} not in results — skipping plot")
        return

    Ks = sorted(df['K'].unique())
    rtg_min = df['rtg_target'].min()
    rtg_max = df['rtg_target'].max()

    deltas = []
    for K in Ks:
        sub = df[df['K'] == K]
        lo = sub[sub['rtg_target'] == rtg_min][pct_col].values[0]
        hi = sub[sub['rtg_target'] == rtg_max][pct_col].values[0]
        deltas.append((K, hi - lo, lo, hi))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: RTG locality ladder — Δ Crime share vs K
    Ks_arr      = np.array([d[0] for d in deltas])
    delta_arr   = np.array([d[1] for d in deltas]) * 100
    delta_lo_col = f'delta_pct_{focus_genre}_lo'
    delta_hi_col = f'delta_pct_{focus_genre}_hi'
    if delta_lo_col in df and delta_hi_col in df:
        lows = np.array([
            df[df['K'] == k][delta_lo_col].iloc[0] for k in Ks
        ]) * 100
        highs = np.array([
            df[df['K'] == k][delta_hi_col].iloc[0] for k in Ks
        ]) * 100
    else:
        lows = highs = np.full_like(delta_arr, np.nan)
    if np.isfinite(lows).all() and np.isfinite(highs).all():
        yerr = np.vstack([delta_arr - lows, highs - delta_arr])
        ax1.errorbar(
            Ks_arr, delta_arr, yerr=yerr, fmt='o-', color='#e07b54',
            linewidth=2, markersize=10, capsize=4,
        )
    else:
        ax1.plot(
            Ks_arr, delta_arr, 'o-', color='#e07b54',
            linewidth=2, markersize=10,
        )
    ax1.set_xlabel(
        'K = number of context positions overridden\n'
        '(K=1: single-slot, K=20: context-saturating)',
                   fontsize=11)
    ax1.set_ylabel(f'Δ {focus_genre} share (%)\nRTG={rtg_max:.0f} vs RTG={rtg_min:.0f}',
                   fontsize=11)
    ax1.set_title(f'RTG locality ladder: {focus_genre} share shift',
                  fontsize=11)
    ax1.set_xticks(Ks_arr)
    ax1.axhline(0, color='black', linewidth=0.8, alpha=0.5)
    ax1.grid(axis='y', alpha=0.3)
    for K, d, lo, hi in deltas:
        ax1.annotate(f'{d*100:+.1f}pp', xy=(K, d * 100),
                     xytext=(0, 8), textcoords='offset points',
                     ha='center', fontsize=9)

    # Right: full Crime curves at each K
    palette = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']
    for i, K in enumerate(Ks):
        sub = df[df['K'] == K].sort_values('rtg_target')
        ax2.plot(sub['rtg_target'], sub[pct_col] * 100,
                 label=f'K={K}', color=palette[i % len(palette)], linewidth=2)
    if no_rtg_share is not None:
        ax2.axhline(no_rtg_share * 100, color='grey', linestyle='--', linewidth=1.2,
                    label='DT no-RTG (flat)')
    ax2.set_xlabel('RTG conditioning target', fontsize=11)
    ax2.set_ylabel(f'{focus_genre} share (%)', fontsize=11)
    ax2.set_title(f'{focus_genre} share vs target RTG, by K', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Plot saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--no-rtg-checkpoint', default=None,
                        help='DT no-RTG checkpoint. If provided, run one forward '
                             'pass on the val windows and overlay the focus-genre '
                             'share as a grey dashed flat line on the right panel.')
    parser.add_argument('--ratings',    default=None)
    parser.add_argument('--movies',     default=None)
    parser.add_argument('--n-levels',   type=int,   default=11,
                        help='RTG sweep resolution (default: 11 levels from '
                             'validation q05 to q95).')
    parser.add_argument('--rtg-min',    type=float, default=None)
    parser.add_argument('--rtg-max',    type=float, default=None)
    parser.add_argument('--rtg-low-quantile', type=float, default=0.05)
    parser.add_argument('--rtg-high-quantile', type=float, default=0.95)
    parser.add_argument('--traj-limit', type=int,   default=1000)
    parser.add_argument('--sample-seed', type=int, default=42,
                        help='Seed for random trajectory subsampling.')
    parser.add_argument(
        '--split', choices=['auto', 'val', 'test'], default='val',
        help='Reporting cohort. Use val while developing and test once after '
             'the protocol is frozen.',
    )
    parser.add_argument('--max-rows',   type=int,   default=None)
    parser.add_argument('--ks',         type=str, default='1,5,10,15,20',
                        help='Comma-separated K values. K=1 (single-slot) and '
                             'K=context_len (context-saturating) bracket the RTG locality ladder.')
    parser.add_argument('--n-boot',     type=int, default=0)
    parser.add_argument('--out',        default='results')
    parser.add_argument('--focus-genre', default='Crime',
                        help="Genre used for the RTG-locality figure (Δ share at RTG=max-min).")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else (
             'mps'  if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    # --- Load model ---
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg          = ckpt['config']
    state_dim    = ckpt['state_dim']
    n_categories = ckpt['n_categories']
    if cfg.get('model', 'dt') != 'dt' or not cfg.get('use_rtg', True):
        raise ValueError("Need a DT+RTG checkpoint.")

    context_len = cfg.get('context_len', 20)
    rtg_scale   = cfg.get('rtg_scale', 20.0)
    rtg_window  = cfg.get('rtg_window', 20)
    alignment   = cfg.get('alignment', 'history')
    categories  = ckpt.get('categories', GENRES)
    if len(categories) != n_categories:
        raise ValueError(
            f"checkpoint category metadata has {len(categories)} labels but "
            f"n_categories={n_categories}"
        )

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
    model = model.to(device).eval()

    # --- Optionally load DT no-RTG control model (flat-line baseline) ---
    no_rtg_model = None
    if args.no_rtg_checkpoint:
        print(f"Control baseline: {args.no_rtg_checkpoint}")
        ckpt_ctrl = torch.load(args.no_rtg_checkpoint, map_location='cpu', weights_only=False)
        cfg_ctrl  = ckpt_ctrl['config']
        _assert_comparable_checkpoints(ckpt, ckpt_ctrl)
        no_rtg_model = RecDecisionTransformer(
            state_dim=ckpt_ctrl['state_dim'], n_actions=ckpt_ctrl['n_categories'],
            hidden_size=cfg_ctrl.get('hidden_size', 128),
            max_length=cfg_ctrl.get('context_len', 20),
            n_layer=cfg_ctrl.get('n_layer', 3),
            n_head=cfg_ctrl.get('n_head', 4),
            dropout=0.0, use_rtg=False,
            use_pad_token=cfg_ctrl.get('use_pad_token', False),
        )
        no_rtg_model.load_state_dict(ckpt_ctrl['model_state'])
        no_rtg_model = no_rtg_model.to(device).eval()

    seed = ckpt.get('seed', 42)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    # --- Load data ---
    min_interactions = cfg.get('min_interactions', 10)
    rating_scale     = cfg.get('rating_scale', 5.0)
    dataset          = cfg.get('dataset', 'movielens')
    effective_max_rows = resolve_checkpoint_max_rows(ckpt, args.max_rows)

    if dataset == 'mal':
        anime_path   = cfg.get('anime_path',   'data/mal/anime.csv')
        ratings_path = args.ratings or cfg.get('ratings_path', 'data/mal/animelist.csv')
        print(f"\nLoading MAL from {ratings_path}...")
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
    print(f"  {len(eval_trajs):,} {eval_split} trajectories")
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

    pad_token = n_categories if cfg.get('use_pad_token', False) else 0

    print("Pre-building context windows...")
    states_all, actions_all, rtgs_base, traj_ids = _build_context_batch(
        eval_trajs, rtg_scale, context_len, state_dim, pad_token=pad_token,
        alignment=alignment
    )
    print(f"  {states_all.shape[0]:,} context windows ready")

    # No-RTG flat-line share for the right-panel overlay
    no_rtg_dist = None
    if no_rtg_model is not None:
        print("Running DT no-RTG control baseline (single forward pass)...")
        no_rtg_dist = get_no_rtg_distribution(
            no_rtg_model, states_all, actions_all, rtgs_base, device, categories
        )
        os.makedirs(args.out, exist_ok=True)
        no_rtg_csv = os.path.join(args.out, 'no_rtg_baseline.csv')
        pd.DataFrame([{'genre': g, 'pct': v} for g, v in no_rtg_dist.items()]).to_csv(
            no_rtg_csv, index=False
        )
        print(f"  No-RTG baseline → {no_rtg_csv}")
        print(f"  No-RTG {args.focus_genre} share: {no_rtg_dist.get(args.focus_genre, float('nan')):.3f}")

    rtg_targets = _rtg_target_grid(
        grid_trajs, args.n_levels, args.rtg_min, args.rtg_max,
        low_quantile=args.rtg_low_quantile,
        high_quantile=args.rtg_high_quantile,
    )
    Ks = [int(k) for k in args.ks.split(',')]
    invalid_ks = [k for k in Ks if not 1 <= k <= context_len]
    if invalid_ks or not Ks:
        raise ValueError(
            f"all K values must be in [1, {context_len}], got {invalid_ks or Ks}"
        )
    if args.focus_genre not in categories:
        raise ValueError(
            f"focus genre {args.focus_genre!r} is not in checkpoint categories"
        )
    print(f"\nSweeping K={Ks} × {len(rtg_targets)} RTG levels "
          f"({rtg_targets[0]:.1f} → {rtg_targets[-1]:.1f})")

    all_records = []
    for K in Ks:
        print(f"\n--- K = {K} ---")
        recs = run_one_k(
            model, states_all, actions_all, rtgs_base,
            rtg_targets, rtg_scale, context_len, device, categories,
            K=K, traj_ids=traj_ids, n_boot=args.n_boot,
        )
        all_records.extend(recs)
        # Compact summary line per K
        focus_col = f'pct_{args.focus_genre}'
        lo = recs[0].get(focus_col, np.nan)
        hi = recs[-1].get(focus_col, np.nan)
        delta_lo = recs[0][f'delta_pct_{args.focus_genre}_lo']
        delta_hi = recs[0][f'delta_pct_{args.focus_genre}_hi']
        ci_text = (
            f" [{delta_lo * 100:+.1f}, {delta_hi * 100:+.1f}]"
            if args.n_boot > 0 else ""
        )
        print(
            f"  {args.focus_genre}: RTG={rtg_targets[0]:.1f} → {lo:.3f}; "
            f"RTG={rtg_targets[-1]:.1f} → {hi:.3f}; "
            f"Δ = {(hi-lo)*100:+.1f}pp{ci_text}"
        )

    df_out = pd.DataFrame(all_records)
    df_out.insert(0, 'evaluation_split', eval_split)
    df_out.insert(1, 'rtg_grid_split', grid_split)

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, 'prefix_k_sweep.csv')
    df_out.to_csv(csv_path, index=False)
    print(f"\nResults → {csv_path}")

    plot_path = os.path.join(args.out, 'prefix_k_sweep.png')
    no_rtg_share = no_rtg_dist.get(args.focus_genre) if no_rtg_dist else None
    plot_realism_ladder(df_out, plot_path, focus_genre=args.focus_genre,
                        no_rtg_share=no_rtg_share)

    # Print compact RTG-locality summary
    print(f"\n=== RTG locality ladder ({args.focus_genre} share, "
          f"RTG={rtg_targets[-1]:.1f} − RTG={rtg_targets[0]:.1f}) ===")
    print(f"  K   Δ {args.focus_genre} (pp)       95% user-cluster CI")
    for K in Ks:
        sub = df_out[df_out['K'] == K]
        lo = sub[sub['rtg_target'] == rtg_targets[0]][f'pct_{args.focus_genre}'].values[0]
        hi = sub[sub['rtg_target'] == rtg_targets[-1]][f'pct_{args.focus_genre}'].values[0]
        delta_lo = sub[f'delta_pct_{args.focus_genre}_lo'].iloc[0]
        delta_hi = sub[f'delta_pct_{args.focus_genre}_hi'].iloc[0]
        ci_text = (
            f"[{delta_lo * 100:+.2f}, {delta_hi * 100:+.2f}]"
            if args.n_boot > 0 else "disabled"
        )
        print(f"  {K:3d}  {(hi-lo)*100:+.2f}                  {ci_text}")


if __name__ == '__main__':
    main()
