"""
Data pipelines for RecDT experiments.

Supported datasets
------------------
  MovieLens 25M   — userId, movieId, rating (0.5–5.0), timestamp
  MyAnimeList     — user_id, anime_id, rating (1–10), watching_status (no timestamp)

MDP formulation
---------------
  s   = user context vector at step t (genre preference history + scalar features)
  a   = primary genre index
  r   = rating / rating_scale  (normalized to [0, 1])
  RTG = cumulative future normalized rating over next rtg_window steps
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# MovieLens genre vocabulary
# ---------------------------------------------------------------------------

GENRES = [
    'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
    'Horror', 'Musical', 'Mystery', 'Romance',
    'Sci-Fi', 'Thriller', 'War', 'Western',
]
N_GENRES = len(GENRES)  # 18  (IMAX removed — it is a format, not a content genre)
GENRE_TO_IDX = {g: i for i, g in enumerate(GENRES)}

CONTEXT_DIM   = N_GENRES + 2          # genre_pref(18) + mean_rating(1) + log_count(1) = 20
STATE_DIM     = CONTEXT_DIM           # 20 — DT uses this; genre_onehot excluded (it equals the action)


# ---------------------------------------------------------------------------
# MyAnimeList genre vocabulary  (15 genres, priority order — niche-first)
# ---------------------------------------------------------------------------

# Fixed priority order used in the paper when an anime has multiple tags.
MAL_GENRES = [
    'Psychological', 'Thriller', 'Horror', 'Mecha', 'Mystery',
    'Sports', 'Slice of Life', 'Supernatural', 'Drama', 'Sci-Fi',
    'Romance', 'Adventure', 'Fantasy', 'Action', 'Comedy',
]
MAL_GENRE_TO_IDX = {g: i for i, g in enumerate(MAL_GENRES)}
_MAL_ADULT_TAGS  = {'Hentai', 'Ecchi'}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_movielens(
    ratings_path: str,
    movies_path: str,
    nrows: int | None = None,
    genre_selection: str = 'rarest',
) -> pd.DataFrame:
    """
    Load and merge MovieLens ratings + movies.

    Returns a DataFrame with columns:
      userId, movieId, rating, timestamp, primary_genre, genre_idx

    Movies without any known genre or with '(no genres listed)' are dropped.

    genre_selection
    ---------------
      'rarest' (default): pick the rarest (most specific) genre among the
                          movie's tags — breaks the alphabetical-first bias
                          that inflates Action.
      'first':            pick the first genre in pipe-order — used for the
                          labeling-rule robustness ablation.
    """
    if genre_selection not in {'rarest', 'first'}:
        raise ValueError(f"genre_selection must be 'rarest' or 'first', got {genre_selection!r}")

    ratings = pd.read_csv(ratings_path, nrows=nrows,
                          dtype={'userId': 'int32', 'movieId': 'int32',
                                 'rating': 'float32', 'timestamp': 'int64'})

    movies = pd.read_csv(movies_path,
                         dtype={'movieId': 'int32', 'title': 'str', 'genres': 'str'})

    if genre_selection == 'rarest':
        from collections import Counter
        all_genre_lists = [
            [g for g in gs.split('|') if g in GENRE_TO_IDX]
            for gs in movies['genres']
            if isinstance(gs, str) and gs != '(no genres listed)'
        ]
        genre_freq = Counter(g for gl in all_genre_lists for g in gl)

        def primary_genre(genre_str: str) -> str | None:
            if not isinstance(genre_str, str) or genre_str == '(no genres listed)':
                return None
            known = [g for g in genre_str.split('|') if g in GENRE_TO_IDX]
            if not known:
                return None
            return min(known, key=lambda g: genre_freq[g])
    else:  # 'first'
        def primary_genre(genre_str: str) -> str | None:
            if not isinstance(genre_str, str) or genre_str == '(no genres listed)':
                return None
            for g in genre_str.split('|'):
                if g in GENRE_TO_IDX:
                    return g
            return None

    movies['primary_genre'] = movies['genres'].apply(primary_genre)
    movies = movies[movies['primary_genre'].notna()][['movieId', 'primary_genre']]
    movies['genre_idx'] = movies['primary_genre'].map(GENRE_TO_IDX).astype('int8')

    df = ratings.merge(movies, on='movieId', how='inner')
    df = df.sort_values(['userId', 'timestamp']).reset_index(drop=True)
    return df



def load_mal(anime_path: str, ratings_path: str) -> pd.DataFrame:
    """
    Load MyAnimeList Database 2020
    (Kaggle: hernan4444/anime-recommendation-database-2020).

    anime_path    : data/mal/anime.csv   — MAL_ID, Name, Score, Genres, ...
    ratings_path  : data/mal/animelist.csv — user_id, anime_id, rating, watching_status

    Filters applied:
      - watching_status == 2  (completed only)
      - rating > 0            (explicitly scored)
      - no adult content      (Hentai/Ecchi excluded via Genres column)
      - anime must map to one of the 15 MAL_GENRES (priority-order selection)

    No timestamps in the dataset — row order within each user is used as a
    temporal proxy (reflects rough insertion order in the Kaggle export).

    Returns a DataFrame with columns:
      userId, rating (float32, 1–10), timestamp (int64, 0-based per user),
      primary_genre, genre_idx
    """
    print('  Loading anime metadata...')
    anime = pd.read_csv(anime_path, dtype={'MAL_ID': 'int32'})

    def _map_mal_genre(genre_str: str) -> str | None:
        if not isinstance(genre_str, str):
            return None
        tags = {g.strip() for g in genre_str.split(',')}
        if tags & _MAL_ADULT_TAGS:
            return None
        for g in MAL_GENRES:        # priority order — first match wins
            if g in tags:
                return g
        return None

    anime['primary_genre'] = anime['Genres'].apply(_map_mal_genre)
    # Mark adult titles so we can filter them even if genre didn't match
    anime['_adult'] = anime['Genres'].apply(
        lambda s: bool(isinstance(s, str) and ({t.strip() for t in s.split(',')} & _MAL_ADULT_TAGS))
    )
    id_to_genre: dict[int, str] = {}
    id_to_adult: dict[int, bool] = {}
    for _, row in anime[['MAL_ID', 'primary_genre', '_adult']].iterrows():
        id_to_genre[int(row['MAL_ID'])] = row['primary_genre']   # may be None
        id_to_adult[int(row['MAL_ID'])] = bool(row['_adult'])

    print('  Loading animelist ratings (1.9 GB, streaming)...')
    chunks = []
    for chunk in pd.read_csv(
        ratings_path,
        chunksize=500_000,
        dtype={'user_id': 'int32', 'anime_id': 'int32',
               'rating': 'int8', 'watching_status': 'int8'},
    ):
        chunk = chunk[(chunk['watching_status'] == 2) & (chunk['rating'] > 0)]
        # Drop adult titles inline to keep memory low
        chunk = chunk[~chunk['anime_id'].map(id_to_adult).fillna(False)]
        chunks.append(chunk[['user_id', 'anime_id', 'rating']])
    df = pd.concat(chunks, ignore_index=True)

    df['primary_genre'] = df['anime_id'].map(id_to_genre)
    df = df[df['primary_genre'].notna()].copy()
    df['genre_idx'] = df['primary_genre'].map(MAL_GENRE_TO_IDX).astype('int8')
    df = df.rename(columns={'user_id': 'userId'})
    df['rating']    = df['rating'].astype('float32')

    # Assign per-user sequential timestamp from row order (insertion-order proxy)
    df = df.reset_index(drop=True)
    df['timestamp'] = df.groupby('userId').cumcount().astype('int64')

    df = df.sort_values(['userId', 'timestamp']).reset_index(drop=True)
    return df[['userId', 'rating', 'timestamp', 'primary_genre', 'genre_idx']]


# ---------------------------------------------------------------------------
# Trajectory builder
# ---------------------------------------------------------------------------

def build_trajectories(
    df: pd.DataFrame,
    rtg_window: int = 20,
    min_interactions: int = 10,
    max_t_override: float | None = None,
    n_categories: int | None = None,
    rating_scale: float = 5.0,
    rtg_shuffle: str | None = None,
    shuffle_seed: int = 0,
    drop_incomplete_rtg: bool = False,
) -> list[dict]:
    """
    Build per-user trajectories from the merged MovieLens DataFrame.

    Each trajectory is a dict:
      states:  (T, STATE_DIM)  float32
      actions: (T,)            int64    — genre index (0–18)
      rewards: (T,)            float32  — normalized rating (rating / 5.0)
      rtgs:    (T,)            float32  — cumulative future reward over rtg_window steps

    State layout at step t  [STATE_DIM = 20]:
      genre_pref[0..17]  — proportion of user's *past* ratings in each genre (18 dims)
      mean_rating_norm   — user's mean rating so far / 5.0 (1 dim)
      log_count_norm     — log(1+t) / log(1+MAX_T) for cold-start awareness (1 dim)

    genre_onehot intentionally excluded: it equals the action, so including it
    would let the model trivially copy the answer from the input.

    rtg_shuffle
    -----------
      None (default):    real per-position RTG.
      'within_traj':     permute rtgs along the time axis within each trajectory,
                         leaving states/actions/rewards intact. Tests whether RTG
                         content (not just presence) provides any learnable signal.
                         Shuffle is applied once, deterministically — seeded from
                         shuffle_seed XOR userId so each user gets a distinct
                         permutation but the whole dataset is reproducible.

    drop_incomplete_rtg
    -------------------
      False: retain the legacy, right-censored suffix whose RTG contains fewer
             than ``rtg_window`` rewards.
      True:  retain only positions with a complete future reward window. This
             is the appropriate setting when the dataset boundary is not a
             genuine terminal state; otherwise the final positions receive
             mechanically small RTGs solely because observation stopped.
    """
    if rtg_window < 1:
        raise ValueError(f"rtg_window must be positive, got {rtg_window}")
    if rtg_shuffle not in {None, 'within_traj'}:
        raise ValueError(f"rtg_shuffle must be None or 'within_traj', got {rtg_shuffle!r}")
    n_cats = n_categories if n_categories is not None else N_GENRES
    trajectories = []

    user_lengths = df.groupby('userId').size()
    # Use override when scoring val/test so max_T is fixed from train only.
    max_T = max_t_override if max_t_override is not None else float(user_lengths.max())

    for user_id, group in df.groupby('userId', sort=False):
        T = len(group)
        if T < min_interactions:
            continue

        actions = group['genre_idx'].values.astype(np.int64)   # (T,)
        rewards = (group['rating'].values / rating_scale).astype(np.float32)  # (T,)

        # RTG: vectorized cumsum trick. When the log boundary is censoring
        # rather than a real terminal state, discard the incomplete suffix
        # instead of interpreting missing future rewards as zero.
        cumsum = np.concatenate([[0.0], np.cumsum(rewards)])
        end_idx = np.minimum(np.arange(T) + rtg_window, T)
        rtgs = (cumsum[end_idx] - cumsum[np.arange(T)]).astype(np.float32)
        valid_T = T - rtg_window + 1 if drop_incomplete_rtg else T
        # Every downstream next-action dataset needs at least two positions.
        if valid_T < 2:
            continue
        if drop_incomplete_rtg:
            rtgs = rtgs[:valid_T]

        if rtg_shuffle == 'within_traj':
            # Per-user deterministic permutation. XOR with userId so each user
            # gets a distinct shuffle while the whole dataset stays reproducible
            # across runs with the same shuffle_seed.
            rng = np.random.default_rng(shuffle_seed ^ int(user_id))
            rng.shuffle(rtgs)

        # --- State features (fully vectorized) ---

        # genre_onehot at each step: (T, n_cats)
        genre_onehots = np.eye(n_cats, dtype=np.float32)[actions]

        # genre_pref[t] = proportion of genres in steps 0..t-1 (exclude current step)
        cumsum_genres = np.zeros((T + 1, n_cats), dtype=np.float32)
        np.cumsum(genre_onehots, axis=0, out=cumsum_genres[1:])
        t_idx = np.arange(T, dtype=np.float32)
        genre_pref = np.where(
            t_idx[:, None] > 0,
            cumsum_genres[:T] / np.maximum(t_idx[:, None], 1),
            0.0,
        )  # (T, N_GENRES)

        # mean_rating_norm[t] = mean(rewards[0..t-1]) / 5.0 — already normalized
        cumsum_rewards = np.zeros(T + 1, dtype=np.float32)
        np.cumsum(rewards, out=cumsum_rewards[1:])
        mean_rating = np.where(
            t_idx > 0,
            cumsum_rewards[:T] / np.maximum(t_idx, 1),
            0.0,
        )  # (T,) — already in [0,1] since rewards ∈ [0,1]

        # log_count_norm[t] = log(1+t) / log(1+max_T)
        log_count = np.log1p(t_idx) / np.log1p(max_T)  # (T,)

        states = np.concatenate([
            genre_pref,              # (T, 18)
            mean_rating[:, None],    # (T, 1)
            log_count[:, None],      # (T, 1)
        ], axis=1).astype(np.float32)  # (T, 20)

        if drop_incomplete_rtg:
            actions = actions[:valid_T]
            rewards = rewards[:valid_T]
            states = states[:valid_T]

        trajectories.append({
            'states':  states,
            'actions': actions,
            'rewards': rewards,
            'rtgs':    rtgs,
        })

    return trajectories




def split_user_ids_three_way(
    df: pd.DataFrame,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
    strategy: str = 'random',
) -> tuple[set, set, set]:
    """Return disjoint train, validation, and held-out test user sets.

    Validation is used for model selection. Test is reserved for the final
    frozen-protocol report. Setting ``test_fraction=0`` reproduces the legacy
    two-way split exactly and returns an empty test set.

    ``random`` sorts IDs before seeded shuffling, making membership independent
    of input row order. ``tail`` is retained only for old-checkpoint
    compatibility; corrected experiments should use ``random``.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in [0, 1), got {test_fraction}")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError(
            "val_fraction + test_fraction must be less than 1, got "
            f"{val_fraction + test_fraction}"
        )
    if strategy not in {'random', 'tail'}:
        raise ValueError(f"strategy must be 'random' or 'tail', got {strategy!r}")

    user_ids = np.sort(df['userId'].unique())
    if strategy == 'random':
        user_ids = user_ids.copy()
        np.random.default_rng(seed).shuffle(user_ids)

    n_users = len(user_ids)
    n_val = max(1, int(val_fraction * n_users))
    n_test = max(1, int(test_fraction * n_users)) if test_fraction > 0 else 0
    n_train = n_users - n_val - n_test
    if n_train < 1:
        raise ValueError(
            f"split leaves no training users: {n_users} users, "
            f"{n_val} validation, {n_test} test"
        )

    # Keep validation at the end so adding a test split preserves the legacy
    # validation cohort. Carve test users from the former training pool; this
    # avoids relabeling the already-inspected legacy validation cohort as the
    # corrected-rerun test set.
    train_end = n_train
    test_end = train_end + n_test
    train_ids = set(user_ids[:train_end])
    test_ids = set(user_ids[train_end:test_end])
    val_ids = set(user_ids[test_end:])
    return train_ids, val_ids, test_ids


def select_evaluation_user_ids(
    val_ids: set,
    test_ids: set,
    split: str = 'auto',
) -> tuple[set, str]:
    """Choose the cohort used by a reporting script.

    ``auto`` uses the held-out test set when configured and otherwise falls
    back to validation for old checkpoints. Asking explicitly for a missing
    test split is an error rather than a silent validation fallback.
    """
    if split not in {'auto', 'val', 'test'}:
        raise ValueError(f"split must be 'auto', 'val', or 'test', got {split!r}")
    if split == 'test':
        if not test_ids:
            raise ValueError(
                "this checkpoint has no held-out test split; rerun training "
                "with test_fraction > 0 or use --split val"
            )
        return test_ids, 'test'
    if split == 'val':
        return val_ids, 'val'
    if test_ids:
        return test_ids, 'test'
    return val_ids, 'val'


def resolve_checkpoint_max_rows(
    checkpoint: dict,
    requested_max_rows: int | None,
) -> int | None:
    """Keep evaluation on the exact row scope used for training.

    New checkpoints record ``data_max_rows``. An explicit evaluation override
    must match it; otherwise user membership and all split assignments change.
    """
    trained_max_rows = checkpoint.get('data_max_rows')
    if requested_max_rows is not None and requested_max_rows != trained_max_rows:
        raise ValueError(
            f"--max-rows={requested_max_rows} does not match checkpoint "
            f"training scope {trained_max_rows!r}"
        )
    return trained_max_rows


def sample_trajectories(
    trajectories: list[dict],
    limit: int | None,
    seed: int = 42,
) -> list[dict]:
    """Select a reproducible random subset without depending on user-ID order."""
    if limit is None or limit >= len(trajectories):
        return trajectories
    if limit < 1:
        raise ValueError(f"limit must be positive or None, got {limit}")
    indices = np.random.default_rng(seed).choice(
        len(trajectories), size=limit, replace=False
    )
    return [trajectories[int(i)] for i in indices]


def state_dim_from_trajectories(trajectories: list[dict]) -> int:
    return trajectories[0]['states'].shape[1]


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class TrajectoryDataset(Dataset):
    """
    Samples fixed-length context windows from trajectories.
    Each sample: (states, actions, rtgs, timesteps, target_action)

    Stores only (traj_idx, t) index pairs — windows are sliced on the fly in
    __getitem__ rather than pre-expanded. Reduces memory from ~20GB to ~5GB
    for the full MovieLens dataset.

    alignment:
      'history'       legacy protocol. Inputs are states/actions/rtgs[start:t],
                      target is actions[t].
      'current_state' corrected next-action protocol. Inputs include state[t]
                      and rtg[t], but the final action slot is replaced by
                      pad_token so the target action is not supplied.

    pad_token: id used to fill action positions during left-padding and as the
    unknown final action in 'current_state' alignment. Default 0 matches legacy
    behavior (uses real action 0 as padding). Pass n_genres for the dedicated
    PAD-token setup, matching `use_pad_token=True` on the model.
    """

    def __init__(self, trajectories: list[dict], context_len: int = 20,
                 rtg_scale: float = 20.0, pad_token: int = 0,
                 alignment: str = 'history'):
        if alignment not in {'history', 'current_state'}:
            raise ValueError(
                f"alignment must be 'history' or 'current_state', got {alignment!r}"
            )
        self.trajectories = trajectories
        self.context_len  = context_len
        self.rtg_scale    = rtg_scale
        self.pad_token    = pad_token
        self.alignment    = alignment
        # Build flat index: list of (traj_idx, t) pairs
        self.index = []
        for i, traj in enumerate(trajectories):
            T = len(traj['actions'])
            for t in range(1, T):
                self.index.append((i, t))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        traj_idx, t = self.index[idx]
        traj  = self.trajectories[traj_idx]
        if self.alignment == 'current_state':
            end = t + 1
        else:
            end = t
        start = max(0, end - self.context_len)
        L     = end - start
        pad   = self.context_len - L
        sd    = traj['states'].shape[1]

        states  = torch.tensor(traj['states'][start:end],  dtype=torch.float32)
        actions = torch.tensor(traj['actions'][start:end], dtype=torch.long)
        rtgs    = torch.tensor(traj['rtgs'][start:end] / self.rtg_scale, dtype=torch.float32).unsqueeze(-1)
        target  = torch.tensor(traj['actions'][t],       dtype=torch.long)

        if self.alignment == 'current_state':
            actions[-1] = self.pad_token

        if pad > 0:
            states  = torch.cat([torch.zeros(pad, sd), states], dim=0)
            actions = torch.cat([torch.full((pad,), self.pad_token, dtype=torch.long), actions], dim=0)
            rtgs    = torch.cat([torch.zeros(pad, 1), rtgs], dim=0)

        timesteps = torch.arange(self.context_len, dtype=torch.long)
        return states, actions, rtgs, timesteps, target


class SASRecDataset(Dataset):
    """
    Context windows for SASRec: action sequences only, no state or RTG.
    Same indexing structure as TrajectoryDataset.

    pad_token: id used for left-padding (default 0 = legacy; pass n_genres
    when paired with `use_pad_token=True` on the SASRec model).
    """

    def __init__(self, trajectories: list[dict], context_len: int = 20, pad_token: int = 0):
        self.trajectories = trajectories
        self.context_len  = context_len
        self.pad_token    = pad_token
        self.index = []
        for i, traj in enumerate(trajectories):
            T = len(traj['actions'])
            for t in range(1, T):
                self.index.append((i, t))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        traj_idx, t = self.index[idx]
        traj  = self.trajectories[traj_idx]
        start = max(0, t - self.context_len)
        L     = t - start
        pad   = self.context_len - L

        actions = torch.tensor(traj['actions'][start:t], dtype=torch.long)
        target  = torch.tensor(traj['actions'][t],       dtype=torch.long)

        if pad > 0:
            actions = torch.cat([torch.full((pad,), self.pad_token, dtype=torch.long), actions])

        return actions, target
