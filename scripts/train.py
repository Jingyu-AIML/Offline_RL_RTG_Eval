"""Train the paper's Decision Transformer variants or SASRec.

Run from this package root, for example:
    python scripts/train.py --config configs/dt_movielens_current_state.yaml --seed 42
"""

import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml  # type: ignore[import-untyped]
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.dataset import (
    GENRES,
    MAL_GENRES,
    SASRecDataset,
    TrajectoryDataset,
    build_trajectories,
    load_mal,
    load_movielens,
    state_dim_from_trajectories,
    split_user_ids_three_way,
)
from src.model import RecDecisionTransformer, SASRec
from src.trainer import RecDTTrainer, SASRecTrainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_warmup_scheduler(optimizer, warmup_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 1.0
    return LambdaLR(optimizer, lr_lambda)


def pick_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def validate_protocol_config(cfg: dict):
    """Fail fast when a config claims the corrected reporting protocol."""
    if cfg.get('protocol') != 'corrected_v3':
        raise ValueError("Use a corrected_v3 configuration from configs/.")

    required = {
        'drop_incomplete_rtg': True,
        'user_split': 'random',
    }
    for key, expected in required.items():
        if cfg.get(key) != expected:
            raise ValueError(
                f"corrected_v3 requires {key}={expected!r}, got {cfg.get(key)!r}"
            )
    if cfg.get('test_fraction', 0.0) <= 0:
        raise ValueError("corrected_v3 requires a non-empty held-out test split")

    model_type = cfg.get('model', 'dt')
    if model_type not in {'dt', 'sasrec'}:
        raise ValueError(f"unknown model type {model_type!r}")
    if model_type == 'dt' and cfg.get('alignment') != 'current_state':
        raise ValueError(
            f"corrected_v3 {model_type} requires alignment='current_state'"
        )
    if model_type in {'dt', 'sasrec'} and not cfg.get('use_pad_token', False):
        raise ValueError(
            f"corrected_v3 {model_type} requires use_pad_token=true"
        )
    if cfg.get('rtg_window', 20) < 1:
        raise ValueError("corrected_v3 requires rtg_window >= 1")
    if cfg.get('rtg_scale', 20.0) <= 0:
        raise ValueError("corrected_v3 requires rtg_scale > 0")


def fixed_validation_subset(dataset, max_samples: int | None, seed: int):
    """Return one reproducible random validation subset.

    Taking the first batches overweights the lowest-sorted validation users.
    Membership is fixed across checkpoints and seeds by ``split_seed``.
    """
    if max_samples is None or len(dataset) <= max_samples:
        return dataset
    if max_samples < 1:
        raise ValueError(f"max_samples must be positive or None, got {max_samples}")
    indices = np.random.default_rng(seed).choice(
        len(dataset), size=max_samples, replace=False
    )
    return Subset(dataset, np.sort(indices).tolist())


def validate_resume_protocol(current_cfg: dict, checkpoint_cfg: dict):
    """Prevent resuming weights trained with a different data cohort."""
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
        'rtg_shuffle': None,
        'model': 'dt',
        'use_rtg': True,
        'use_pad_token': False,
        'context_len': 20,
        'hidden_size': 128,
        'n_layer': 3,
        'n_layers': 3,
        'n_head': 4,
        'dropout': 0.1,
        'batch_size': 512,
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'warmup_steps': 1000,
        'val_max_batches': 1000,
    }
    mismatches = []
    for key, default in defaults.items():
        current_value = current_cfg.get(key, default)
        checkpoint_value = checkpoint_cfg.get(key, default)
        if current_value != checkpoint_value:
            mismatches.append(
                f"{key}: config={current_value!r}, checkpoint={checkpoint_value!r}"
            )
    if mismatches:
        raise ValueError(
            "cannot resume a checkpoint trained under a different data "
            "protocol: " + "; ".join(mismatches)
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-rows', type=int, default=None,
                        help='Subsample N rows from ratings.csv (dev mode)')
    parser.add_argument('--out', default='checkpoints')
    parser.add_argument('--resume', default=None,
                        help='Resume from a checkpoint (.pt); restores model weights and step count')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    validate_protocol_config(cfg)
    set_seed(args.seed)
    device = pick_device()
    print(f"Config : {args.config}")
    print(f"Device : {device}")
    print(f"Seed   : {args.seed}")

    model_type  = cfg.get('model', 'dt')
    context_len = cfg.get('context_len', 20)
    rtg_scale   = cfg.get('rtg_scale', 20.0)
    rtg_window  = cfg.get('rtg_window', 20)
    batch_size  = cfg.get('batch_size', 512)
    num_workers = cfg.get('num_workers', 4)
    pin_memory  = (device == 'cuda')
    alignment   = cfg.get('alignment', 'history')

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    dataset          = cfg.get('dataset', 'movielens')
    min_interactions = cfg.get('min_interactions', 10)
    rating_scale     = cfg.get('rating_scale', 5.0)
    t0 = time.time()

    if dataset == 'mal':
        anime_path   = cfg.get('anime_path',   'data/mal/anime.csv')
        ratings_path = cfg.get('ratings_path', 'data/mal/animelist.csv')
        print(f"\nLoading MyAnimeList from {ratings_path}...")
        df         = load_mal(anime_path, ratings_path)
        categories = MAL_GENRES
    else:
        ratings_path = cfg.get('ratings_path', 'data/movielens/ml-25m/ratings.csv')
        movies_path  = cfg.get('movies_path',  'data/movielens/ml-25m/movies.csv')
        print(f"\nLoading MovieLens from {ratings_path}...")
        df         = load_movielens(ratings_path, movies_path, nrows=args.max_rows,
                                     genre_selection=cfg.get('genre_selection', 'rarest'))
        categories = GENRES

    n_categories = len(categories)
    print(f"  {len(df):,} interactions  [{time.time()-t0:.1f}s]")

    rtg_shuffle = cfg.get('rtg_shuffle', None)
    if rtg_shuffle:
        print(f"  RTG shuffle mode: {rtg_shuffle} (train trajectories only; val keeps real RTG)")

    print(f"  Building trajectories (rtg_window={rtg_window}, min_interactions={min_interactions})...")
    t0 = time.time()
    # Split users before building so val log_count_norm uses train's max_T only.
    # Random splitting avoids coupling the validation cohort to user-ID order.
    split_strategy = cfg.get('user_split', 'tail')
    split_seed = cfg.get('split_seed', 42)
    val_fraction = cfg.get('val_fraction', 0.1)
    test_fraction = cfg.get('test_fraction', 0.0)
    drop_incomplete_rtg = cfg.get('drop_incomplete_rtg', False)
    train_ids, val_ids, test_ids = split_user_ids_three_way(
        df, val_fraction=val_fraction, test_fraction=test_fraction, seed=split_seed,
        strategy=split_strategy,
    )
    train_df = df[df['userId'].isin(train_ids)]
    train_max_T = float(train_df.groupby('userId').size().max())
    train_trajs = build_trajectories(train_df,
                                     rtg_window=rtg_window, n_categories=n_categories,
                                     min_interactions=min_interactions,
                                     max_t_override=train_max_T,
                                     rating_scale=rating_scale,
                                     rtg_shuffle=rtg_shuffle,
                                     shuffle_seed=args.seed,
                                     drop_incomplete_rtg=drop_incomplete_rtg)
    val_trajs   = build_trajectories(df[df['userId'].isin(val_ids)],
                                     rtg_window=rtg_window, n_categories=n_categories,
                                     min_interactions=min_interactions,
                                     max_t_override=train_max_T,
                                     rating_scale=rating_scale,
                                     drop_incomplete_rtg=drop_incomplete_rtg)
    print(
        f"  {len(train_trajs):,} train / {len(val_trajs):,} val trajectories"
        f" / {len(test_ids):,} held-out test users  [{time.time()-t0:.1f}s]"
    )

    state_dim = state_dim_from_trajectories(train_trajs)
    print(f"  state_dim={state_dim}, n_categories={n_categories}")

    use_pad_token = cfg.get('use_pad_token', False)
    pad_token     = n_categories if use_pad_token else 0

    if model_type == 'sasrec':
        train_ds = SASRecDataset(train_trajs, context_len=context_len, pad_token=pad_token)
        val_ds   = SASRecDataset(val_trajs,   context_len=context_len, pad_token=pad_token)
    else:
        train_ds = TrajectoryDataset(train_trajs, context_len=context_len,
                                     rtg_scale=rtg_scale, pad_token=pad_token,
                                     alignment=alignment)
        val_ds   = TrajectoryDataset(val_trajs,   context_len=context_len,
                                     rtg_scale=rtg_scale, pad_token=pad_token,
                                     alignment=alignment)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_max_batches = cfg.get('val_max_batches', 1000)
    val_eval_ds = fixed_validation_subset(
        val_ds,
        max_samples=val_max_batches * batch_size if val_max_batches is not None else None,
        seed=split_seed,
    )
    val_loader = DataLoader(
        val_eval_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
    )
    print(
        f"  train: {len(train_ds):,} samples | val: {len(val_ds):,} samples "
        f"({len(val_eval_ds):,} fixed random samples used for validation loss)"
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    if model_type == 'sasrec':
        model = SASRec(
            n_actions     = n_categories,
            hidden_size   = cfg.get('hidden_size', 128),
            max_length    = context_len,
            n_layer       = cfg.get('n_layer', 2),
            n_head        = cfg.get('n_head', 4),
            dropout       = cfg.get('dropout', 0.2),
            use_pad_token = use_pad_token,
        )
    else:
        model = RecDecisionTransformer(
            state_dim     = state_dim,
            n_actions     = n_categories,
            hidden_size   = cfg.get('hidden_size', 128),
            max_length    = context_len,
            n_layer       = cfg.get('n_layer', 3),
            n_head        = cfg.get('n_head', 4),
            dropout       = cfg.get('dropout', 0.1),
            use_rtg       = cfg.get('use_rtg', True),
            use_pad_token = use_pad_token,
        )

    # Resume from checkpoint if requested
    start_step = 0
    resume_ckpt = None
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        validate_resume_protocol(cfg, resume_ckpt['config'])
        if resume_ckpt.get('data_max_rows') != args.max_rows:
            raise ValueError(
                f"resume data scope mismatch: --max-rows={args.max_rows!r}, "
                f"checkpoint used {resume_ckpt.get('data_max_rows')!r}"
            )
        checkpoint_seed = resume_ckpt.get('seed', args.seed)
        if checkpoint_seed != args.seed:
            raise ValueError(
                f"resume seed mismatch: --seed={args.seed}, "
                f"checkpoint seed={checkpoint_seed}"
            )
        model.load_state_dict(resume_ckpt['model_state'])
        start_step = resume_ckpt.get('step', 0)
        print(f"\nResumed from {args.resume} at step {start_step:,}")

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel  : {model_type.upper()} | params: {n_params:,}")

    # ------------------------------------------------------------------
    # Optimizer + scheduler
    # ------------------------------------------------------------------
    optimizer = AdamW(
        model.parameters(),
        lr           = cfg.get('lr', 1e-4),
        weight_decay = cfg.get('weight_decay', 1e-4),
    )
    warmup_steps = cfg.get('warmup_steps', 1000)
    scheduler = get_warmup_scheduler(optimizer, warmup_steps) if warmup_steps > 0 else None
    if resume_ckpt is not None:
        if 'optimizer_state' not in resume_ckpt:
            raise ValueError(
                "checkpoint predates true-resume support and has no optimizer "
                "state; restart this corrected run from scratch"
            )
        optimizer.load_state_dict(resume_ckpt['optimizer_state'])
        saved_scheduler = resume_ckpt.get('scheduler_state')
        if scheduler is not None:
            if saved_scheduler is None:
                raise ValueError(
                    "checkpoint has no scheduler state; restart from scratch"
                )
            scheduler.load_state_dict(saved_scheduler)
        rng_state = resume_ckpt.get('rng_state')
        if rng_state is not None:
            random.setstate(rng_state['python'])
            np.random.set_state(rng_state['numpy'])
            torch.set_rng_state(rng_state['torch'])
            if torch.cuda.is_available() and rng_state.get('cuda') is not None:
                torch.cuda.set_rng_state_all(rng_state['cuda'])

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    max_steps = cfg.get('max_steps', 50000)
    print(f"\nTraining for {max_steps:,} steps (starting from {start_step:,})...")

    if model_type == 'sasrec':
        trainer = SASRecTrainer(model, optimizer, device=device, scheduler=scheduler)
    else:
        trainer = RecDTTrainer(model, optimizer, device=device, scheduler=scheduler)

    os.makedirs(args.out, exist_ok=True)
    config_stem = os.path.splitext(os.path.basename(args.config))[0]
    best_val_loss = (
        float(resume_ckpt.get('best_val_loss', float('inf')))
        if resume_ckpt is not None else float('inf')
    )

    def save_checkpoint(tag: str):
        path = os.path.join(args.out, f"{config_stem}_seed{args.seed}_{tag}.pt")
        torch.save({
            'model_state':  model.state_dict(),
            'config':       cfg,
            'state_dim':    state_dim,
            'n_categories': n_categories,
            'categories':   categories,
            'step':         step,
            'seed':         args.seed,
            'data_max_rows': args.max_rows,
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
            'best_val_loss': best_val_loss,
            'rng_state': {
                'python': random.getstate(),
                'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }, path)
        print(f"  checkpoint → {path}")

    step         = start_step
    log_every    = cfg.get('log_every', 500)
    val_every    = cfg.get('val_every', 5000)
    ckpt_every   = cfg.get('ckpt_every', 10000)
    running_loss = 0.0
    running_count = 0

    while step < max_steps:
        for batch in train_loader:
            if step >= max_steps:
                break

            loss = trainer.train_step(batch)
            running_loss += loss
            running_count += 1
            step += 1

            if running_count == log_every:
                avg = running_loss / running_count
                print(f"  step {step:6d}/{max_steps} | train_loss {avg:.4f}")
                running_loss = 0.0
                running_count = 0

            if step % val_every == 0:
                model.eval()
                val_loss, n = 0.0, 0
                with torch.no_grad():
                    for vbatch in val_loader:
                        if model_type == 'sasrec':
                            actions, targets = vbatch[0].to(device), vbatch[1].to(device)
                            logits = model(actions)
                            vl = F.cross_entropy(logits[:, -1, :], targets).item()
                        else:
                            states, actions, rtgs, timesteps, targets = [
                                v.to(device) for v in vbatch
                            ]
                            logits = model(states, actions, rtgs, timesteps)
                            vl = F.cross_entropy(logits[:, -1, :], targets).item()
                        val_loss += vl
                        n += 1
                mean_val_loss = val_loss / n
                print(f"  step {step:6d}/{max_steps} | val_loss  {mean_val_loss:.4f}")
                if mean_val_loss < best_val_loss:
                    best_val_loss = mean_val_loss
                    save_checkpoint("best")
                model.train()

            if step % ckpt_every == 0:
                save_checkpoint(f"step{step:07d}")

    save_checkpoint("final")


if __name__ == '__main__':
    main()
