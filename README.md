# Auditing Return Conditioning as a Control Knob

Standalone reproduction code for **“Auditing Return Conditioning as a Control
Knob: An Offline Diagnostic for Decision Transformer Recommendation”** by
**Jingyu Wang**, accepted at CONSEQUENCES 2026, a workshop at ACM RecSys 2026.

[Accepted contributions](https://sites.google.com/view/consequences2026/contributions)
· [Paper on arXiv](https://arxiv.org/abs/2608.24815)

This folder contains the final MovieLens 25M and MyAnimeList 2020 experiments:
Decision Transformer with real RTG, no RTG, and within-user shuffled RTG;
SASRec; the RTG locality ladder; logged match and score; and the state-to-RTG
probe. It can be copied to another location and run independently.

## Disclaimer: AI-assisted organization and cleanup

This is **not the original experiment repository**. It was organized and
cleaned up with AI assistance to separate the paper's code from the original
research repository's exploratory experiment threads and development history.
This standalone package focuses on reproducing the reported methods and
results; it does not include the full experimental record.

The cleanup was checked with small end-to-end runs and comparisons against
archived results. The full paper experiments have **not** been rerun end to end
in this cleaned package. See [PROTOCOL.md](PROTOCOL.md) for the checks performed
and the limits of numerical reproduction.

## License and credits

This package's code and documentation are distributed under the
[MIT License](LICENSE), with copyright for Jingyu Wang's contributions dated
2026. Preserve the applicable notices in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), which includes the official
Decision Transformer implementation's MIT notice, adaptation provenance, and
research credits for Decision Transformer and SASRec.

External libraries and datasets retain their own licenses and terms. They are
not bundled or relicensed by this package.

## Install

Use Python 3.10 or newer. Run the following commands from this folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The complete reproduction trains **24 models** (8 configurations × 3 seeds).
Training uses CUDA when available, then Apple MPS, then CPU. Use a GPU machine
for the full experiment. The loaders materialize the filtered datasets and
trajectories in memory; the diagnostic scripts also materialize sampled
context windows. CPU/MPS checks do not establish identical CUDA training results.

A complete dependency lock for the published GPU runs is not available.
`requirements.txt` lists the runtime dependencies; see [PROTOCOL.md](PROTOCOL.md)
for the validation environment and the limits of numerical reproduction.

## Prepare the data

Obtain **MovieLens 25M** from [GroupLens](https://grouplens.org/datasets/movielens/25m/)
and **MyAnimeList Database 2020** from
[Kaggle](https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020).
For MAL, obtain the individual `anime.csv.zip` and `animelist.csv.zip` archives.
Use the original files and preserve their row order.

Place the three archives at these paths, then verify and extract them:

```text
data/movielens/ml-25m.zip
data/mal/anime.csv.zip
data/mal/animelist.csv.zip
```

```bash
shasum -a 256 -c DATA_SHA256SUMS
unzip data/movielens/ml-25m.zip -d data/movielens
unzip data/mal/anime.csv.zip -d data/mal
unzip data/mal/animelist.csv.zip -d data/mal
```

On Linux, `sha256sum -c DATA_SHA256SUMS` is an alternative. The expected
extracted layout is:

```text
data/
├── movielens/ml-25m/
│   ├── movies.csv
│   └── ratings.csv
└── mal/
    ├── anime.csv
    └── animelist.csv
```

Datasets and trained weights are not bundled. Preprocessing is performed by
each entry point; there is no separate cache or preprocessing command.

## Reproduce the paper

Preview the complete command list without loading data:

```bash
python scripts/reproduce.py --dry-run
```

Run all training, evaluation, diagnostics, probes, and report generation:

```bash
python scripts/reproduce.py
```

The runner uses seeds **42, 0, 1**, all eight frozen configs, the **test** split,
1,000 user bootstrap resamples, and validation-derived RTG targets. It runs
commands sequentially and stops on failure. Repeating a stage writes to the
same output paths; use `--checkpoints PATH --results PATH` for a separate run.

Stages can also be run separately, in this order:

```bash
python scripts/reproduce.py --stage train
python scripts/reproduce.py --stage evaluate
python scripts/reproduce.py --stage diagnostics
python scripts/reproduce.py --stage probe
python scripts/reproduce.py --stage report
```

If the published checkpoints are available separately, put the 24 `*_best.pt`
files listed in [PROTOCOL.md](PROTOCOL.md) under `checkpoints/` and begin with
`--stage evaluate`. The original binary files are unnecessary when training
from scratch. Always evaluate the validation-selected **best** checkpoint,
which can differ from the last checkpoint.

To reproduce the supporting validation results, use `--split val` with the
`evaluate`, `diagnostics`, `probe`, and `report` stages. Individual analysis
scripts default to validation; the paper runner explicitly selects test.

## Run one model or diagnostic

```bash
python scripts/train.py \
  --config configs/dt_movielens_current_state.yaml --seed 42 --out checkpoints

python scripts/evaluate.py \
  --checkpoint checkpoints/dt_movielens_current_state_seed42_best.pt \
  --split test --batch-size 1024 --k 1 5 10 \
  --out results/test/eval_dt_movielens_current_state_seed42.csv

python scripts/prefix_k_sweep.py \
  --checkpoint checkpoints/dt_movielens_current_state_seed42_best.pt \
  --no-rtg-checkpoint checkpoints/dt_no_rtg_movielens_current_state_seed42_best.pt \
  --split test --traj-limit 1000 --n-boot 1000 --focus-genre Crime \
  --out results/test/movielens_seed42_real

python scripts/match_and_score.py \
  --checkpoint checkpoints/dt_movielens_current_state_seed42_best.pt \
  --split test --mode last --top-k 5 --traj-limit 2000 --n-boot 1000 \
  --out results/test/match_movielens_seed42.csv

python scripts/rtg_state_probe.py \
  --dataset movielens --split test --mlp --out results/test/probe_movielens.csv
```

The locality example also requires training the matching no-RTG configuration
with the same seed. Its `--out` argument is a **directory**. The other reporting
entry points take a CSV filename. All eight configurations retain their original
filenames so their checkpoint names match the paper's provenance.

## Tables and figures

After all 44 result CSVs have been generated, rebuild the report with:

```bash
python scripts/summarize.py --results results/test --out results/test/summary
```

| Paper element | Generated file under `results/test/summary/` |
|---|---|
| Table 1, effective evaluation cohort | `scope_by_seed.csv` |
| Tables 2–3, locality by seed | `locality_focus_by_seed.csv`, `locality_focus_summary.csv` |
| All-genre locality audit | `locality_all_genres_by_seed.csv`, `locality_all_genres_summary.csv` |
| Table 4, match and score | `match_by_seed.csv`, `match_summary.csv` |
| Table 5, HR@1 and HR@5 (also includes HR@10) | `accuracy_by_seed.csv`, `accuracy_summary.csv` |
| Table 6, state-to-RTG probe | `probe_summary.csv` |
| Figure 1, three-seed MovieLens locality | `figures/ml_current_state_ladder_3seed.pdf` |
| Figure 2, match and score at seed 42 | `figures/match_and_score_current_state.pdf` |

Figures are also saved as PNG. Aggregation uses the **sample** standard
deviation across training seeds (`ddof=1`); this is distinct from the
within-seed user-cluster confidence intervals. Table 1's rounded full-dataset
counts come from the loader output; `scope_by_seed.csv` contains the effective
held-out users and prediction contexts. Table 7 is a conceptual interpretation
table, with no separate experiment.

Published reference results: MovieLens Crime shift is **1.77 ± 1.17 pp** at
K=1 and **23.61 ± 2.96 pp** at K=20; shuffled RTG gives **2.08 ± 1.20 pp** at
K=20. MAL Drama shift is **−0.03 ± 0.07 pp** at K=20. State-probe Ridge/MLP
R² is **0.341/0.505** for MovieLens and **0.736/0.755** for MAL.

## Code layout

```text
configs/                  eight final model/data configurations
src/dataset.py            loading, genre mapping, user splits, trajectories
src/model.py              Decision Transformer and SASRec
src/trainer.py            optimization steps
src/diagnostics.py        context windows, RTG intervention, bootstrap helpers
scripts/train.py           training and validation checkpoint selection
scripts/evaluate.py        next-genre hit rates
scripts/prefix_k_sweep.py  real/shuffled RTG locality ladder and no-RTG control
scripts/match_and_score.py logged action-match and rating diagnostic
scripts/rtg_state_probe.py Ridge and MLP state-to-RTG regression
scripts/reproduce.py       complete fixed experiment matrix
scripts/summarize.py       paper tables and all-genre summaries
scripts/render_figures.py  manuscript figure rendering, called by summarize
PROTOCOL.md               frozen settings, provenance, checkpoint identities
DATA_SHA256SUMS           original dataset archive identities
LICENSE                  MIT license for this package's contributions
THIRD_PARTY_NOTICES.md    upstream license notice and research credits
```

The package contains no test suite, exploratory datasets/configurations, training
logs, tracking/sync scripts, cached outputs, checkpoints, or manuscript builds.
The MLP **probe** remains because it is in the paper; the separate MLP
recommendation baseline is excluded.
