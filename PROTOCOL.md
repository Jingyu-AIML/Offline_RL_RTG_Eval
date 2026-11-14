# Frozen paper protocol

This document records the paper's frozen settings, checkpoint identities, and
validation of this standalone reproduction package. See the
[README disclaimer](README.md#disclaimer-ai-assisted-organization-and-cleanup)
for the role of AI assistance in organizing and cleaning up the code.

The eight configurations and model implementation preserve the experiment
settings. Dataset construction and the active DT/SASRec optimization paths
retain the numerical operations used for the paper.

Licensing and attribution are documented in [LICENSE](LICENSE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The model header was updated
to describe the rating-based task and record attribution; its executable
implementation is unchanged.

The extraction removes the MLP recommendation baseline, its dataset/trainer,
unreported recommendation-diversity evaluation, the unused two-way split
wrapper, and unrelated entry points/configurations. Shared context/intervention
and bootstrap functions are collected in `src/diagnostics.py` without changing
their bodies. The state-to-RTG probe keeps
both Ridge and MLP regression; its printed interpretation was shortened to
match the paper's descriptive claim. The runner and summary builder are new;
the figure renderer is adapted from the manuscript's rendering script.

## Model and data settings

- MovieLens: timestamp order, 18 genre actions, least-frequent catalog genre
  selected for multi-genre items; minimum 10 interactions; ratings divided by 5.
- MAL: row order after filtering, 15 genres in the fixed priority in
  `src/dataset.py`; only completed, explicitly scored entries; adult-tagged
  titles excluded; minimum 15 interactions; ratings divided by 10.
- User split: randomized 80/10/10 train/validation/test with split seed 42.
  Splitting precedes trajectory eligibility filtering. State count normalization
  uses the maximum history length in the training users only.
- State: prior genre histogram, prior normalized mean rating, and log count.
  Reward window W=20; positions with incomplete future RTGs are removed.
  Each eligible trajectory predicts at retained positions t=1 through T−1.
- DT: current state and RTG at position t, with the unknown current action
  replaced by a dedicated PAD ID; padding-aware attention. RTGs are divided
  by 20. Context length 20, hidden size 128, 3 layers, 4 heads, dropout 0.1.
- No-RTG DT zeros the RTG embedding. Shuffled RTG permutes the retained RTGs
  within each training user's trajectory using model seed XOR user ID. It
  preserves the RTG channel and user-level value distribution. Validation and
  test retain real RTGs.
- SASRec: action-history input, context length 20, hidden size 128, 2 layers,
  4 heads, dropout 0.2, dedicated PAD token.
- AdamW: learning rate 1e-4, weight decay 1e-4, batch size 512, gradient norm
  clipping at 1.0. MovieLens: 50,000 steps, 1,000 warmup steps. MAL: 20,000
  steps, 500 warmup steps. Four loader workers. Model seeds: 42, 0, 1.
- Checkpoint selection: lowest validation cross-entropy checked every 5,000
  steps, on a fixed split-seed-selected subset of at most 512,000 validation
  contexts. The original code averages batch losses for selection. Use
  `*_best.pt`, including the two controls selected at 45,000 steps below.

## Diagnostics and reporting

- Locality: overwrite the last K real context positions, K=1,5,10,15,20;
  padding is unchanged. Sample 1,000 trajectories with sample seed 42. Pair
  real/shuffled models with a no-RTG control trained with the same model seed.
- Target grid: 11 values from the full validation RTG q05 to q95. The code
  computes the full-precision values, approximately 9.70–17.70 for MovieLens
  and 11.80–18.20 for MAL. Do not substitute the rounded endpoints. Test
  contexts never determine the target grid.
- Match and score: last slot only (K=1); 2,000 trajectories, sample seed 42;
  top-1 and top-5 action matches, with ratings on the original scale. The
  paper's table and figure report top-1; Figure 2 uses model seed 42.
- Accuracy: all eligible contexts from held-out users; paper Table 5 reports
  HR@1 and HR@5. HR@10 is also generated as a supplementary metric.
- Bootstrap: 1,000 user-cluster resamples per seed. Aggregate metrics weight
  contexts equally, so longer histories contribute more. Locality uses paired
  endpoint differences. Across-model-seed summaries use mean and sample std,
  not confidence intervals.
- State probe: Ridge alpha=1 on all training contexts; MLP with one 64-unit
  hidden layer, max_iter=100, early stopping with validation_fraction=0.1,
  random_state=0, fitted to up to 1,000,000 training contexts sampled with RNG
  seed 0. Evaluate on the selected held-out split. Targets are RTG/20.

## Published checkpoint identities

These hashes identify the original binary artifacts; they do not imply that a
fresh training run on another software/hardware stack will yield identical
checkpoint bytes. Files are not included in this code package. If using the
published weights, put these filenames directly under `checkpoints/`.

| Checkpoint | Selected step | Validation loss | SHA-256 |
|---|---:|---:|---|
| `dt_mal_current_state_seed0_best.pt` | 20,000 | 2.011618 | `c4d57a7bd84d84599a78127f01787c18e09fd9f469eb6a2b064c0f4d1abdb6ea` |
| `dt_mal_current_state_seed1_best.pt` | 20,000 | 2.004412 | `2477c677a840e6f44a49f36481be3cf3d12f7e19ff29961536927cdf5872af58` |
| `dt_mal_current_state_seed42_best.pt` | 20,000 | 2.015358 | `95ca7f942c46953d39c7bb28f1469e0600e3f743f76c8bc7c463601f6db77061` |
| `dt_movielens_current_state_seed0_best.pt` | 50,000 | 2.424850 | `bc0958ebcf03f06fec3cd2f7c8a6c3dc972f02e9e791f857e8cf70b173af59cc` |
| `dt_movielens_current_state_seed1_best.pt` | 50,000 | 2.426957 | `138f4e4afafa6dc04df0629cf70d3d2cd1b74cd93f4bed7dd48fe82ba06d5243` |
| `dt_movielens_current_state_seed42_best.pt` | 50,000 | 2.424490 | `74543268fbf847b7b28a6f6505930a7c9454b9d7b642af3a8573c8322c4d5056` |
| `dt_no_rtg_mal_current_state_seed0_best.pt` | 20,000 | 2.013298 | `1704ed32c9334ece082755a0471e1757e6ff88a7c48b109b8217dce7ca84af64` |
| `dt_no_rtg_mal_current_state_seed1_best.pt` | 20,000 | 2.003512 | `adef3a81b5c6657349d733aee86f6940d2961d68eba4a4709043671392815dd6` |
| `dt_no_rtg_mal_current_state_seed42_best.pt` | 20,000 | 2.014496 | `45025b8d337016a78f535fa3733d0562e6f6387ec120cbe8b4f4e66626b1080f` |
| `dt_no_rtg_movielens_current_state_seed0_best.pt` | 50,000 | 2.428965 | `de82add24925492efdb0dcb7f0588573bfffe702d09f0f0939ba7be0c6ee36cd` |
| `dt_no_rtg_movielens_current_state_seed1_best.pt` | 45,000 | 2.430477 | `149165766822047af4fa8744fac60d9726f0dd827d907ab25b249f4a410d99cd` |
| `dt_no_rtg_movielens_current_state_seed42_best.pt` | 50,000 | 2.428699 | `dde76301407416dfdcbcfc7cd1128b537ed8cbfafa07dc54a604ecde40ae6000` |
| `dt_shuffled_rtg_mal_current_state_seed0_best.pt` | 20,000 | 2.011666 | `bc9b47a69045c1cb01618e672ee67f46f8d2818f69545bf9e666a297d4fbfe32` |
| `dt_shuffled_rtg_mal_current_state_seed1_best.pt` | 20,000 | 2.004331 | `640a3358b92760289dac7c343fbb9bbc6053cfb907fc487bb9fff5cd457077d1` |
| `dt_shuffled_rtg_mal_current_state_seed42_best.pt` | 20,000 | 2.015376 | `67f8a253f4e7e5c7a112526b782ac9a37eee504eb8f74dc79c4fca1784cc3299` |
| `dt_shuffled_rtg_movielens_current_state_seed0_best.pt` | 50,000 | 2.428304 | `ec22ce3e753154f6125347c5efe548ab83243176a6d0feeb4753ff8daea324cc` |
| `dt_shuffled_rtg_movielens_current_state_seed1_best.pt` | 45,000 | 2.430272 | `60f5f0b8e11b27e668fa9eb76120f3ff31217acb0b5422cee862310ac4feb645` |
| `dt_shuffled_rtg_movielens_current_state_seed42_best.pt` | 50,000 | 2.428177 | `f5f37c4d56272978a451d079ae9aff647c63d5e435d66155f74327fccb5d35ff` |
| `sasrec_mal_seed0_best.pt` | 20,000 | 2.068501 | `513596655b5aec23f753190ad79226ed95a79279bf2e4b9bb7810eaf1bec3d1e` |
| `sasrec_mal_seed1_best.pt` | 20,000 | 2.070074 | `dd874e7655cc21aa30afab38249e133d477510df881e4bf8342f8ab247a5945c` |
| `sasrec_mal_seed42_best.pt` | 20,000 | 2.070162 | `13eaed14c83fc00f3f423e7d5c7977a3f8fb1a91cc9cdad5cf8770ce0cfaa97d` |
| `sasrec_movielens_seed0_best.pt` | 50,000 | 2.452676 | `205e375309fb0dc9c1d2e4d77ac41866360d02d4d22e03fe9c8e26ed51b6b331` |
| `sasrec_movielens_seed1_best.pt` | 50,000 | 2.453038 | `dbb7f7d64bd3804061479188bed4e22c567ecc40aa3925681aede7313a9eca79` |
| `sasrec_movielens_seed42_best.pt` | 50,000 | 2.452504 | `b61c45ff2661c94b28e2755925cd0fe7a1e46d60d4b99b1b9e17791b880a0f82` |

Most selected checkpoints are the final scheduled checkpoint. Two MovieLens
controls selected step 45,000: no-RTG seed 1 and shuffled-RTG seed 1. Reporting
all runs as simply "trained for 50K" without this selection detail would be
imprecise.

## Reproduction limits and interpretation

The original requirements were unpinned and a complete GPU environment lock was
not archived. Local extraction checks used Python 3.12.2, PyTorch 2.7.0,
NumPy 1.26.4, pandas 2.0.3, scikit-learn 1.4.2, matplotlib 3.8.4, and PyYAML
6.0.1. These are validation-environment versions, not a claim about the original
training environment. Numerical training/probe results may vary with dependency
versions, hardware, and nondeterministic kernels.

Extraction validation covered all seven CLI entry points from outside the
package, eight two-step training runs on synthetic MovieLens/MAL data, eight
held-out evaluations, four locality sweeps, two match-and-score runs, and both
Ridge/MLP probes in a separate temporary copy. The report builder processed the
44 archived final CSVs and reproduced the published summaries and figures.
Retained model/data/trainer/helper functions were compared structurally with
the original source, and all eight YAML configs matched byte for byte. These
checks validate the extraction; the 24 full-data training runs were not repeated.
No verification data, logs, or test scripts are included in this folder.

Dataset selection was exploratory. Amazon Movies & TV, Tenrec QK-video,
MovieLens 10M, Yelp, and LFM-1b failed feasibility checks; Amazon CDs & Vinyl was
trained but excluded after a null exploratory sweep; MAL was retained as the
replication dataset. Crime selection and some analyses were exploratory.
Corrected test users are disjoint from corrected training/validation users,
but earlier exploratory work used these public datasets, so this is not a
pristine prospective holdout.

Logged RTG includes current/future outcomes. Real-RTG accuracy is diagnostic,
not deployable accuracy. Match and score evaluates a selected logged subset,
not counterfactual policy value. Full-context RTG overwriting creates synthetic
histories. The results concern genre prediction, not item-level reward control;
MAL's row ordering is not calendar time. State-probe R² describes predictability
and does not prove that the model's RTG channel is redundant.
