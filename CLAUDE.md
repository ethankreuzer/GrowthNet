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


## Project Overview

GrowthCurve is a machine learning project that trains multi-task neural networks to predict molecular behavior (optical density growth curves and compound activity) from chemical structure representations. The system uses PyTorch Lightning with hyperparameter optimization via Weights & Biases (W&B).

**Core problem:** Given a chemical compound (SMILES), its molecular representation(s), and experimental conditions (time, concentration), predict:
1. **Regression**: Optical Density (OD) measurements
2. **Classification**: Activity status (binary: active/inactive)

## Architecture Overview

### Data Pipeline

1. **Representations**: Molecular fingerprints/embeddings computed from SMILES strings
   - Built-in: MACCS (166-dim), ECFP (2048-dim), RDKit (2048-dim)
   - External: Boltz2 (3072-dim, computed via boltz CLI)
   - Custom: MiniMol (512-dim)
   - Config: `FEATURE_SETS` in `sweep_multihead_lightning.py`

2. **Compound Metadata** (`CompoundMeta`):
   - Stores per-compound time-concentration grid data
   - Tracks OD and classification targets across conditions
   - Precomputed from CSV data and pickled for reuse

3. **Dataset Classes** (`sweeps/data_class.py`):
   - `PerCompoundDataset`: Samples (t,c) points from each compound, interpolates targets
   - `ExplicitDataset`: For validation/test with exact (t,c) points
   - Uses `RectBivariateSpline` for 2D interpolation when multiple concentrations available
   - Custom collate function handles variable fingerprint dimensionalities

### Training Pipeline

**Entry point**: `sweeps/sweep_multihead_lightning.py`
- PyTorch Lightning module with dual-head architecture
- Trunk network (shared) → Regression head + Classification head
- Hyperparameter sweeps via W&B Bayes optimization
- Metrics: Pearson correlation (regression), AUROC/F1 (classification)

**Sweep config**: `sweeps/multihead_sweep.yml`
- 10+ hyperparameters: learning rate, dropout, architecture depth/width, noise injection
- Feature set selection: `"minimol_classic"` or other combinations

### Data Organization

```
data/
  unique_smiles.txt                    # List of unique SMILES
  smiles_index.pkl                     # SMILES → index mapping
  smiles_representations.pkl           # Index → fingerprint dict
  train/Celine_CompoundMetas_list.pkl  # Cached compound metadata
  boltz_yamls/                         # Config files for Boltz predictions
  boltz_output/                        # Raw Boltz embeddings

sweeps/
  data_class.py                        # Dataset definitions
  sweep_multihead_lightning.py         # Model + training loop
  multihead_sweep.yml                  # W&B sweep config
  evaluate_model.ipynb                 # Analysis notebook

scripts/
  build_representation_dict.py         # Build smiles_representations.pkl
  build_representation_dict.sh         # SLURM wrapper
  build_training_data.py               # Create compound metadata
  run_inference.py                     # Batch inference script
```

## Common Development Tasks

### Running a Hyperparameter Sweep

```bash
cd /home/ethan2/GrowthNet/sweeps
wandb sweep multihead_sweep.yml
# Get sweep ID from output, then in a SLURM script:
wandb agent <PROJECT>/<ENTITY>/<SWEEP_ID>
```

However, sweeps must be run using SLURM. You can see my script /home/ethan2/job.sh for an example of how to run a sweep given the <PROJECT>/<ENTITY>/<SWEEP_ID> output from 
doing 

### Building Molecular Representations

Two-stage process (Boltz requires external prediction):

```bash
# Stage 1: Prep
python scripts/build_representation_dict.py --stage prep

# Then run: boltz predict ... (external, see script for details)

# Stage 2: Assemble
python scripts/build_representation_dict.py --stage assemble
```

### Running Inference on New Compounds

```bash
python scripts/run_inference.py --input compounds.csv --model path/to/model.pt
```

### Preparing Training Data

```bash
python scripts/build_training_data.py --input raw_data.csv --output data/train/
```

## Key Design Patterns

1. **Representation Flexibility**: Code uses `features_by_family` dict to handle variable-length fingerprints. Models select which fingerprints via `feature_set` parameter.

2. **Interpolation Strategy**: Uses `RectBivariateSpline` for smooth 2D interpolation in time-concentration space. Single-concentration compounds fall back to 1D interpolation.

3. **Multi-Task Learning**: Single trunk (shared learned features) feeds regression and classification heads. Loss weighted by `loss_lambda` parameter.

4. **W&B Integration**: Metadata logged via `CompoundMeta`, sweeps configured in YAML, hyperparameter search runs asynchronously across compute cluster.

5. **Data Caching**: `CompoundMeta` objects are serialized to pickle to avoid recomputation; `smiles_representations.pkl` precomputes fingerprints.

## Debugging Notes

- **Missing CompoundMeta pickle**: `data_class.py` expects `data/train/Celine_CompoundMetas_list.pkl`. If missing, run `build_training_data.py` first.
- **Boltz embeddings**: `build_representation_dict.py` fails at stage 2 if boltz_output/ is empty. Ensure stage 1 YAML files are generated and boltz CLI ran.
- **Interpolation edge cases**: Single-concentration compounds use 1D splines; high-noise data may require larger `s` parameter in `rbs_reg`.
- **W&B auth**: Ensure `wandb login` completed before running sweeps.

## Dependencies (Not Exhaustive)

- PyTorch, PyTorch Lightning
- Weights & Biases (wandb)
- RDKit, datamol (chemistry)
- scipy (interpolation)
- pandas, numpy
- See sweep scripts for full imports

## Testing & Validation

- Validation metrics: Pearson (regression), AUROC/F1 (classification)
- Aggregated metric: `val_main/best_agg_metric` (used in sweep optimization)
- Test evaluation: `sweeps/Claude_evaluate_on_validation.ipynb`
- Model analysis: `sweeps/top30_sweep_analysis.ipynb`
