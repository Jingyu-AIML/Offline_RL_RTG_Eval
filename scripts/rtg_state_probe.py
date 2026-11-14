"""Predict scaled RTG from current state using Ridge and an MLP.

This is a descriptive overlap diagnostic, not proof that RTG is redundant.
Uses the same user split, complete RTG windows, and state features as training.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.dataset import (
    GENRES, MAL_GENRES,
    build_trajectories, load_mal, load_movielens,
    split_user_ids_three_way,
)


def gather_state_rtg(trajectories, rtg_scale):
    """Stack (state, rtg/rtg_scale) across all trajectories. Drop step 0
    (state is all-zeros by construction)."""
    states, rtgs = [], []
    for traj in trajectories:
        T = len(traj['actions'])
        if T < 2:
            continue
        states.append(traj['states'][1:])
        rtgs.append(traj['rtgs'][1:] / rtg_scale)
    X = np.concatenate(states, axis=0)
    y = np.concatenate(rtgs,   axis=0)
    return X.astype(np.float32), y.astype(np.float32)


def feature_names(dataset):
    if dataset == 'mal':
        return [f'genre_pref[{g}]' for g in MAL_GENRES] + ['mean_rating_norm', 'log_count_norm']
    return [f'genre_pref[{g}]' for g in GENRES] + ['mean_rating_norm', 'log_count_norm']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',  choices=['movielens', 'mal'], default='movielens')
    parser.add_argument('--ratings',  default=None)
    parser.add_argument('--movies',   default=None)
    parser.add_argument('--max-rows', type=int, default=None)
    parser.add_argument('--rtg-window', type=int, default=20)
    parser.add_argument('--rtg-scale',  type=float, default=None,
                        help='Default: 20 for both corrected datasets.')
    parser.add_argument('--min-interactions', type=int, default=None,
                        help='Default: 10 for MovieLens, 15 for MAL.')
    parser.add_argument('--split-seed', type=int, default=42)
    parser.add_argument('--val-fraction', type=float, default=0.1)
    parser.add_argument('--test-fraction', type=float, default=0.1)
    parser.add_argument(
        '--split', choices=['val', 'test'], default='val',
        help='Report validation during development; pass test only after all '
             'probe choices are frozen.',
    )
    parser.add_argument('--keep-incomplete-rtg', action='store_true',
                        help='Retain the legacy right-censored RTG suffix.')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Ridge regularization strength')
    parser.add_argument('--features', choices=['all', 'scalar_only', 'genre_only'],
                        default='all',
                        help="'scalar_only' = drop genre_pref (use mean_rating_norm + log_count_norm); "
                             "'genre_only' = drop scalar features. Useful for ablating which "
                             "features carry the state→RTG signal.")
    parser.add_argument('--mlp', action='store_true',
                        help='Also fit a 1-hidden-layer MLP for the nonlinear ceiling')
    parser.add_argument('--mlp-sample', type=int, default=1_000_000,
                        help='Random subsample of train rows used for MLP fit '
                             '(MLPRegressor scales poorly with N; default 1M).')
    parser.add_argument('--out', default='results/rtg_state_probe.csv')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if args.dataset == 'mal':
        rating_scale = 10.0
        rtg_scale    = args.rtg_scale or 20.0
        min_interactions = args.min_interactions or 15
        anime_path   = 'data/mal/anime.csv'
        ratings_path = args.ratings or 'data/mal/animelist.csv'
        print(f"Loading MAL from {ratings_path}...")
        df = load_mal(anime_path, ratings_path)
        n_categories = len(MAL_GENRES)
    else:
        rating_scale = 5.0
        rtg_scale    = args.rtg_scale or 20.0
        min_interactions = args.min_interactions or 10
        ratings_path = args.ratings or 'data/movielens/ml-25m/ratings.csv'
        movies_path  = args.movies  or 'data/movielens/ml-25m/movies.csv'
        print(f"Loading MovieLens from {ratings_path}...")
        df = load_movielens(ratings_path, movies_path, nrows=args.max_rows)
        n_categories = len(GENRES)

    train_ids, val_ids, test_ids = split_user_ids_three_way(
        df,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.split_seed,
        strategy='random',
    )
    print(
        f"  {len(train_ids):,} train / {len(val_ids):,} val / "
        f"{len(test_ids):,} test users"
    )
    eval_ids = val_ids if args.split == 'val' else test_ids

    print("Building trajectories...")
    train_df = df[df['userId'].isin(train_ids)]
    train_max_T = float(train_df.groupby('userId').size().max())
    train_trajs = build_trajectories(
        train_df,
        rtg_window=args.rtg_window, n_categories=n_categories,
        min_interactions=min_interactions,
        max_t_override=train_max_T,
        rating_scale=rating_scale,
        drop_incomplete_rtg=not args.keep_incomplete_rtg,
    )
    eval_trajs = build_trajectories(
        df[df['userId'].isin(eval_ids)],
        rtg_window=args.rtg_window, n_categories=n_categories,
        min_interactions=min_interactions,
        max_t_override=train_max_T,
        rating_scale=rating_scale,
        drop_incomplete_rtg=not args.keep_incomplete_rtg,
    )
    print(
        f"  {len(train_trajs):,} train / {len(eval_trajs):,} "
        f"{args.split} trajectories"
    )

    X_train, y_train = gather_state_rtg(train_trajs, rtg_scale)
    X_eval, y_eval = gather_state_rtg(eval_trajs, rtg_scale)

    feat_names = feature_names(args.dataset)
    if args.features == 'scalar_only':
        scalar_cols = [i for i, n in enumerate(feat_names) if not n.startswith('genre_pref')]
        X_train = X_train[:, scalar_cols]
        X_eval = X_eval[:, scalar_cols]
        feat_names = [feat_names[i] for i in scalar_cols]
        print(f"  Feature ablation: scalar_only ({len(scalar_cols)} features)")
    elif args.features == 'genre_only':
        genre_cols = [i for i, n in enumerate(feat_names) if n.startswith('genre_pref')]
        X_train = X_train[:, genre_cols]
        X_eval = X_eval[:, genre_cols]
        feat_names = [feat_names[i] for i in genre_cols]
        print(f"  Feature ablation: genre_only ({len(genre_cols)} features)")
    print(
        f"  X_train {X_train.shape}, X_{args.split} {X_eval.shape}"
    )
    print(f"  y_train mean={y_train.mean():.4f} std={y_train.std():.4f}  "
          f"y_{args.split} mean={y_eval.mean():.4f} std={y_eval.std():.4f}")

    # --- Linear probe ---
    print(f"\n--- Ridge regression (alpha={args.alpha}) ---")
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    ridge = Ridge(alpha=args.alpha)
    ridge.fit(X_train, y_train)
    yhat_train = ridge.predict(X_train)
    yhat_eval = ridge.predict(X_eval)
    r2_train = r2_score(y_train, yhat_train)
    r2_eval = r2_score(y_eval, yhat_eval)
    rmse_eval = float(np.sqrt(np.mean((y_eval - yhat_eval) ** 2)))
    print(f"  R² train: {r2_train:.4f}")
    print(f"  R² {args.split}: {r2_eval:.4f}")
    print(
        f"  RMSE {args.split}: {rmse_eval:.4f} "
        f"(y std = {y_eval.std():.4f})"
    )

    # Top features by |coef|
    names = feat_names
    coefs = ridge.coef_
    order = np.argsort(-np.abs(coefs))
    print(f"\n  Top-10 features by |coef|:")
    for i in order[:10]:
        print(f"    {names[i]:30s}  coef = {coefs[i]:+.4f}")
    print(f"  Intercept: {ridge.intercept_:+.4f}")

    if args.verbose:
        print("\n  All feature coefficients:")
        for n, c in zip(names, coefs):
            print(f"    {n:30s}  coef = {c:+.4f}")

    # --- Optional: MLP nonlinear ceiling ---
    r2_mlp_train = r2_mlp_eval = None
    if args.mlp:
        n_sub = min(args.mlp_sample, len(X_train))
        rng = np.random.default_rng(0)
        idx = rng.choice(len(X_train), size=n_sub, replace=False)
        X_sub, y_sub = X_train[idx], y_train[idx]
        print(f"\n--- MLP regression (1 hidden layer, 64 units, n_sub={n_sub:,}) ---")
        from sklearn.neural_network import MLPRegressor
        mlp = MLPRegressor(
            hidden_layer_sizes=(64,),
            max_iter=100,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=0,
            verbose=False,
        )
        mlp.fit(X_sub, y_sub)
        r2_mlp_train = r2_score(y_sub, mlp.predict(X_sub))
        r2_mlp_eval = r2_score(y_eval, mlp.predict(X_eval))
        print(f"  R² train: {r2_mlp_train:.4f}")
        print(f"  R² {args.split}: {r2_mlp_eval:.4f}")

    # --- Save summary ---
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    summary = pd.DataFrame([{
        'dataset':      args.dataset,
        'evaluation_split': args.split,
        'n_train':      len(X_train),
        'n_eval':       len(X_eval),
        'y_eval_std':   float(y_eval.std()),
        'ridge_alpha':  args.alpha,
        'ridge_r2_train': r2_train,
        'ridge_r2_eval':  r2_eval,
        'ridge_rmse_eval': rmse_eval,
        'mlp_r2_train':   r2_mlp_train,
        'mlp_r2_eval':    r2_mlp_eval,
    }])
    summary.to_csv(args.out, index=False)
    print(f"\nSummary → {args.out}")



if __name__ == '__main__':
    main()
