# GrowthNet

A model for predicting bacterial response to drug candidates (SMILES). Given a chemical compound (SMILES), its molecular representation(s), and experimental conditions, GrowthNet predicts optical density (OD) growth curves and compound activity.

## Overview

**Core Tasks:**
1. **Regression**: Predict Optical Density (OD) measurements across time and concentration
2. **Classification**: Predict compound activity status (binary: active/inactive)

**Key Features:**
- Multi-task learning with shared trunk and dual prediction heads
- Support for multiple molecular representations (MACCS, ECFP, RDKit, Boltz2, MiniMol)
- PyTorch Lightning for scalable training
- Hyperparameter optimization via Weights & Biases (W&B)
- 2D spline interpolation for smooth predictions across time-concentration space

## Quick Start

### Setup Environment

```bash
uv sync
```

### Running a Hyperparameter Sweep

```bash
cd sweeps
wandb sweep multihead_sweep.yml
# Get <SWEEP_ID> from output, then submit to SLURM:
wandb agent <PROJECT>/<ENTITY>/<SWEEP_ID>
```

See `job.sh` for a complete SLURM job script example.

### Building Molecular Representations

Two-stage process (Boltz requires external prediction):

```bash
# Stage 1: Prepare configuration for Boltz
python scripts/build_representation_dict.py --stage prep

# Stage 2: Assemble fingerprints from Boltz output
python scripts/build_representation_dict.py --stage assemble
```

### Running Inference

```bash
python scripts/run_inference.py --input compounds.csv --model path/to/model.pt
```

### Preparing Training Data

```bash
python scripts/build_training_data.py --input raw_data.csv --output data/train/
```

## Architecture

### Data Pipeline

**Molecular Representations** (`FEATURE_SETS` in `sweep_multihead_lightning.py`):
- Built-in fingerprints: MACCS (166-dim), ECFP (2048-dim), RDKit (2048-dim)
- External: Boltz2 (3072-dim, computed separately)
- Custom: MiniMol (512-dim)

**Compound Metadata** (`CompoundMeta`):
- Per-compound time-concentration grid data
- OD and classification targets across experimental conditions
- Precomputed and cached as pickle files for reuse

**Dataset Classes** (`sweeps/data_class.py`):
- `PerCompoundDataset`: Samples (time, concentration) points during training
- `ExplicitDataset`: For validation/test with exact (t, c) points
- `RectBivariateSpline`: 2D interpolation for multi-concentration data
- Custom collate function handles variable-length fingerprints

### Training Pipeline

**Model** (`sweeps/sweep_multihead_lightning.py`):
- PyTorch Lightning module
- Shared trunk network → dual heads (regression + classification)
- Configurable depth, width, dropout, noise injection

**Metrics**:
- Regression: Pearson correlation coefficient
- Classification: AUROC, F1 score
- Aggregated metric: `val_main/best_agg_metric` (sweep optimization target)

**Hyperparameters** (`sweeps/multihead_sweep.yml`):
- Learning rate, dropout, batch size
- Network architecture (depth, width)
- Feature set selection
- Loss weighting (`loss_lambda`)
- Noise injection for regularization

## Project Structure

```
data/
  unique_smiles.txt                    # SMILES lookup
  smiles_index.pkl                     # SMILES → index mapping
  smiles_representations.pkl           # Fingerprints by index
  train/
    Celine_CompoundMetas_list.pkl      # Cached compound metadata
  boltz_yamls/                         # Boltz config files
  boltz_output/                        # Boltz prediction results

sweeps/
  sweep_multihead_lightning.py         # Model definition + training
  multihead_sweep.yml                  # W&B sweep configuration
  data_class.py                        # Dataset classes
  evaluate_model.ipynb                 # Model evaluation
  top30_sweep_analysis.ipynb           # Sweep results analysis

scripts/
  build_representation_dict.py         # Build molecular fingerprints
  build_representation_dict.sh          # SLURM wrapper for above
  build_training_data.py               # Create CompoundMeta cache
  run_inference.py                     # Batch prediction on new compounds
```

## Key Design Patterns

1. **Flexible Representations**: Variable-length fingerprints handled via `features_by_family` dict. Models select subsets via `feature_set` parameter.

2. **Smooth Interpolation**: `RectBivariateSpline` for 2D interpolation; falls back to 1D for single-concentration compounds.

3. **Multi-Task Learning**: Shared trunk learns common features; independent heads specialize for regression and classification tasks.

4. **Data Caching**: `CompoundMeta` objects serialized to pickle for fast reloading; fingerprints precomputed in `smiles_representations.pkl`.

5. **W&B Integration**: Sweeps configured in YAML; hyperparameter search runs asynchronously across cluster.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Missing `CompoundMeta` pickle | Run `python scripts/build_training_data.py` |
| Boltz embeddings not found | Ensure `build_representation_dict.py --stage prep` ran and boltz CLI completed |
| Interpolation errors on sparse data | Adjust `s` parameter in `RectBivariateSpline` |
| W&B authentication fails | Run `wandb login` before starting sweeps |

## Dependencies

- **ML Framework**: PyTorch, PyTorch Lightning
- **Hyperparameter Optimization**: Weights & Biases (wandb)
- **Chemistry**: RDKit, datamol
- **Numerical**: scipy (interpolation), numpy
- **Data Handling**: pandas

See `pyproject.toml` for the complete dependency list.

## Analysis & Evaluation

- **Validation Metrics**: Pearson correlation (regression), AUROC/F1 (classification)
- **Sweep Analysis**: `sweeps/top30_sweep_analysis.ipynb` — visualize best runs
- **Model Evaluation**: `sweeps/evaluate_model.ipynb` — detailed predictions

## Configuration

**Environment Variable**:
```bash
export PYTHONPATH=$PYTHONPATH:/home/ethan2/GrowthNet
```

**Sweep Parameters** (`sweeps/multihead_sweep.yml`):
- Modify `parameters` section to change hyperparameter ranges
- Adjust `goal` and `metric` to change optimization target
- Update `feature_set` to select molecular representations

## License

[Add license info if applicable]

## Contact

[Add contact/author info if applicable]
