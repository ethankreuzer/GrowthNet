# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Behavior

When I ask you to do a task, interview me until you have 95% confidence about what I actually want, not what I think I should want.
 
## Environment

When in this project, always use the virtual environment created for it.

lrwxrwxrwx    1 ethan2 ethan2      30 Mar 31 11:37 .venv -> /home/ethan2/venvs/GrowthCurve

If there are ever any dependency issues, fix them by either doing 

```bash
uv add 
```
or edit the pyproject.toml file and rerun

```bash
uv sync
```

If you run any script in this project, you must do so by doing 

```bash
uv run
```

With the intended associated virtual environment


## Project Overview

GrowthNet trains multi-task neural networks to predict molecular behavior (optical density growth curves and compound activity) from chemical structure representations. The system uses PyTorch Lightning with hyperparameter optimization via Weights & Biases (W&B).

**Core problem:** Given a chemical compound (SMILES), its molecular representation(s), and experimental conditions (time, concentration), predict:
1. **Regression**: Optical Density (OD) measurements
2. **Classification**: Activity status (binary: active/inactive)


## Datasets and Cleaning

Two experimental datasets feed the pipeline:

### GrowthCurve (~2700 compounds)
Plate-based OD measurements with on-plate DMSO controls at each (Plate, Concentration, Timepoint) condition. Cleaning steps:
1. **Plate-level multiplicative correction**: Compute the median DMSO OD per plate, divide all wells on that plate by that median to anchor to 1.0.
2. **Well-level iterative correction**: After plate correction, use the median of inactive-compound wells per (Plate, Concentration, Timepoint) block for a second pass of multiplicative normalization.
3. **Activity threshold**: `median(DMSO) - 4 * MAD(DMSO)` per (Plate, Concentration, Timepoint). Compounds below this threshold at a given condition are labeled active (`is_Active = 1`).

### Enamine DR (t6/t12 panels)
No on-plate DMSO controls, so a bootstrapped correction is applied:
1. Transfer correction factors estimated from the GrowthCurve dataset to the Enamine plates.
2. Use the median of presumed-inactive compounds (those with OD near 1.0 after factor application) as the normalizing anchor.
3. Apply the same `median - 4*MAD` threshold for activity labeling.

Full methodology: `scripts/batch_correction_methodology.md`

Both cleaned datasets are saved as labeled DataFrames:
- `data/train/df_GrowthCurve_27000.pkl`
- `data/train/df_combined_Enamine.pkl`


## Building CompoundMeta Objects

**Entry point**: `scripts/build_compound_metas.py`

This script reads the two labeled DataFrames, then for each unique SMILES:
1. Computes the **Murko scaffold** via `datamol.to_scaffold_murcko`.
2. Looks up or computes molecular fingerprints, caching results in `data/smiles_representations.pkl`:
   - `maccs_fp`: 166-dim MACCS keys
   - `ecfp_fp`: 2048-dim ECFP4 (radius 2, folded)
   - `rdkit_fp`: 2048-dim RDKit path fingerprint
   - `minimol_fp`: 512-dim MiniMol embedding
   - `boltz2_rep`: 3072-dim Boltz2 structural embedding (optional, precomputed externally)
3. Builds a `CompoundMeta` dataclass with pivot tables (index: Timepoint, columns: Concentration) for both OD and `is_Active` labels, plus the fingerprint dict and scaffold.
4. Saves the full list to `data/splits/my_split_v1/all_compound_metas.pkl`.

**`CompoundMeta` fields** (defined in `sweeps/data_class.py`):
- `compound`, `smiles`, `scaffold`
- `pivot_od`, `pivot_cls`: DataFrames keyed by (Timepoint × Concentration)
- `t_vals`, `c_vals`: sorted unique timepoints and concentrations
- `single_conc`: True if only one concentration present (affects interpolation)
- `t_min`, `t_max`, `logc_min`, `logc_max`: grid bounds
- `is_active_at_12_50`: convenience flag for the t=12, c=50 condition
- `fps_by_family`: dict mapping family name → numpy fingerprint array


## Train/Val/Test Splits

**Entry point**: `scripts/make_splits.py`

Scaffold-cluster-based Monte Carlo split. Steps:

1. **Load** `data/splits/my_split_v1/all_compound_metas.pkl`.
2. **Scaffold clustering**: Compute Morgan FPs (r=2, 1024-dim) on each compound's Murko scaffold → kNN graph (k=15) → UMAP dimensionality reduction → Leiden community detection. Each cluster contains scaffolds that are structurally similar.
3. **Precompute Tanimoto matrix**: N×N pairwise ECFP4 Tanimoto similarity between all compounds (used during MC sampling to enforce distance constraints).
4. **Two-stage Monte Carlo sampling**:
   - **Test set first** (target ~250 actives at t=12, c=50): Randomly sample clusters; accept a cluster if adding it keeps median max-Tanimoto from test to train ≈ 0.40, and the active count stays within tolerance.
   - **Val set from remainder** (target median max-Tanimoto ≈ 0.42): Same procedure on the remaining non-test compounds.
   - Train = everything not in val or test.
5. **Outputs**:
   - `data/splits/smile_splits_v2/train.txt`, `val.txt`, `test.txt` — one SMILES per line
   - `data/splits/smile_splits_v2/clusters.csv` — cluster assignments

The split text files are what `PerCompoundDataset` and `build_val_dict_from_metas` use at training time; they index into `all_compound_metas.pkl` by SMILES.


## Model Architecture

**Entry point**: `sweeps/sweep_multihead_lightning.py`

### Input encoding
- Molecular fingerprint(s) concatenated from the selected `feature_set`. Current sweep uses `"minimol_classic"` = `["minimol_fp", "ecfp_fp", "maccs_fp", "rdkit_fp"]` → 4826-dim total.
- Condition encoding: `[c_raw, log(c_raw), fourier_time...]` where Fourier time uses sin/cos at frequencies 1–3, period T=15, offset `t' = t - 1`.

### Network
- **Trunk**: `trunk_layers` fully-connected blocks of (Linear → LayerNorm → ReLU → Dropout). Shared representation.
- **Regression head**: `reg_layers` blocks → scalar OD prediction.
- **Classification head**: `cls_layers` blocks → scalar logit (binary cross-entropy with logits).

### Training
- Loss: `MSE (regression) + loss_lambda * BCE (classification)`, weighted sum.
- Sampler: `WeightedRandomSampler` balances active/inactive compounds per batch, controlled by `active_fraction` hyperparameter.
- Scheduler: `CosineAnnealingLR` (OneCycleLR was tried; reverted).
- Optional Gaussian noise on fingerprint inputs during training (`regression_noise`).

### Dataset classes (`sweeps/data_class.py`)
- `PerCompoundDataset`: Training. For each compound, samples `k` (t,c) points per epoch and interpolates OD/activity targets using `RectBivariateSpline` in (time, log-concentration) space. Single-conc compounds use 1D interpolation.
- `build_val_dict_from_metas`: Builds a flat dict of explicit (t,c) points from pivot tables for val/test (no sampling).


## Metrics and Sweep Optimization

### Per-(t,c) subgroup metrics
For each unique (time, concentration) cell in the val/test data, compute:
- **AP** (Average Precision): classification, computed on all compounds in the cell.
- **MAE** (Mean Absolute Error): regression on active compounds only.
- **Pearson**: correlation between predicted and true OD on active compounds only.

Pearson and MAE are only computed when `active_mask.sum() >= N_MIN_ACTIVES_FOR_REGRESSION` (= 5) **and** both `pred.std() > 1e-6` and `true.std() > 1e-6`. Cells below threshold contribute to AP but are excluded from the MAE/Pearson averages.

### Aggregate metric
```
agg = AP_mean - 5 * MAE_mean + Pearson_mean
```
Where each mean is taken over only the subgroups that passed the threshold. If no subgroup qualifies, MAE and Pearson default to 0.0. If `agg` is still NaN after all guards, it is replaced with `-1e6` (finite sentinel so W&B Bayes still sees a signal).

### Logged metrics
- `val_main/AP+Pearson-5*MAE` — per-epoch aggregate (logged every epoch)
- `val_main/best_agg_metric` — running best across epochs (used by W&B sweep optimizer)
- `val_main/AP`, `val_main/MAE`, `val_main/Pearson` — individual components
- Per-concentration slices: same metrics broken out by concentration value
- `test_main/*` — test metrics logged when val improves

### Sweep config (`sweeps/multihead_sweep.yml`)
W&B Bayes sweep optimizing `val_main/best_agg_metric`. Key search ranges:
- `trunk_dim`: 450–700, `reg_layers`: [2,3,4], `reg_hidden`: 16–75
- `max_learning_rate`: 0.001–0.01 (log-uniform)
- `dropout_rate`: 0.10–0.30, `weight_decay`: 0.0001–0.1 (log-uniform)
- `active_fraction`: 0.30–0.50, `batch_size`: 4–15, `epochs`: 150–215
- `regression_noise`: [0.0, 0.00025, 0.0005]
- Fixed: `feature_set: minimol_classic`, `trunk_layers: 1`, `cls_layers: 0`


## Data Organization

```
data/
  smiles_representations.pkl            # SMILES → {family: fingerprint} cache
  train/
    df_GrowthCurve_27000.pkl            # Cleaned GrowthCurve labeled DataFrame
    df_combined_Enamine.pkl             # Cleaned Enamine DR labeled DataFrame
  splits/
    my_split_v1/
      all_compound_metas.pkl            # List[CompoundMeta] for all compounds
    smile_splits_v2/
      train.txt / val.txt / test.txt    # SMILES split files (one per line)
      clusters.csv                      # Scaffold cluster assignments
  boltz_yamls/                          # Config files for Boltz predictions
  boltz_output/                         # Raw Boltz embeddings

sweeps/
  data_class.py                         # CompoundMeta dataclass + Dataset classes
  sweep_multihead_lightning.py          # Model + training loop (PyTorch Lightning)
  multihead_sweep.yml                   # W&B sweep config
  evaluate_model.ipynb                  # Evaluation notebook

scripts/
  build_compound_metas.py               # Build all_compound_metas.pkl from raw DataFrames
  make_splits.py                        # Scaffold-cluster MC split builder
  build_representation_dict.py          # Build/update smiles_representations.pkl
  build_representation_dict.sh          # SLURM wrapper for representation building
  build_training_data.py                # Legacy: older pipeline (Celine_v1 split)
  batch_correction_methodology.md       # Detailed batch correction methodology
  figures/                              # Methodology figures
```


## Common Development Tasks

### Running a Hyperparameter Sweep

```bash
cd /home/ethan2/GrowthNet/sweeps
wandb sweep multihead_sweep.yml
# Outputs: <PROJECT>/<ENTITY>/<SWEEP_ID>
```

Sweeps must be run via SLURM. See `/home/ethan2/job.sh` for the sweep agent invocation pattern.

### Building Molecular Representations

```bash
# Stage 1: Generate Boltz YAML configs and standard fingerprints
uv run python scripts/build_representation_dict.py --stage prep

# Run Boltz externally: boltz predict ... (see script for details)

# Stage 2: Assemble Boltz embeddings into smiles_representations.pkl
uv run python scripts/build_representation_dict.py --stage assemble
```

### Rebuilding CompoundMeta Objects

```bash
uv run python scripts/build_compound_metas.py
# Reads df_GrowthCurve_27000.pkl + df_combined_Enamine.pkl
# Writes data/splits/my_split_v1/all_compound_metas.pkl
```

### Rebuilding the Train/Val/Test Split

```bash
uv run python scripts/make_splits.py
# Reads all_compound_metas.pkl
# Writes data/splits/smile_splits_v2/{train,val,test}.txt + clusters.csv
```


## Key Design Patterns

1. **Representation Flexibility**: `fps_by_family` dict in `CompoundMeta` stores all fingerprint families. At training time, `PerCompoundDataset` selects families listed in the sweep's `feature_set` parameter and concatenates them.

2. **Interpolation Strategy**: `RectBivariateSpline` in (time, log-concentration) space for compounds with >1 concentration. `single_conc=True` compounds fall back to 1D `UnivariateSpline` over time.

3. **Multi-Task Learning**: Single trunk (shared learned features) feeds separate regression and classification heads. Loss weighting controlled by `loss_lambda`.

4. **Metric Robustness**: Pearson and MAE are only computed on (t,c) subgroups with `N_MIN_ACTIVES_FOR_REGRESSION = 5` or more actives, plus a variance guard (`std > 1e-6`). This prevents NaN from propagating through the aggregate and killing the W&B Bayes optimizer.

5. **Data Caching**: `smiles_representations.pkl` is a persistent cache; `build_compound_metas.py` only recomputes fingerprints for SMILES not already in the cache.


## Debugging Notes

- **Missing CompoundMeta pickle**: `data_class.py` loads from `data/splits/my_split_v1/all_compound_metas.pkl`. If missing, run `build_compound_metas.py` first.
- **Missing split files**: `PerCompoundDataset` loads `data/splits/smile_splits_v2/{train,val,test}.txt`. If missing, run `make_splits.py` first.
- **Boltz embeddings**: `build_representation_dict.py --stage assemble` fails if `data/boltz_output/` is empty. Run stage 1 first and then the external boltz CLI.
- **Interpolation edge cases**: Single-concentration compounds use 1D splines; noisy data may need a larger `s` parameter in `rbs_reg`.
- **val_main/best_agg_metric stuck at -inf**: Means the aggregate metric is NaN every epoch. Check that `N_MIN_ACTIVES_FOR_REGRESSION` guard and variance guards are in place in all three eval paths of `sweep_multihead_lightning.py`.
- **W&B auth**: Ensure `wandb login` is completed before running sweeps.


## Dependencies (Not Exhaustive)

- PyTorch, PyTorch Lightning
- Weights & Biases (wandb)
- RDKit, datamol (chemistry)
- scipy (interpolation)
- pandas, numpy
- minimol (MiniMol embeddings)
- See `pyproject.toml` for full dependency list
