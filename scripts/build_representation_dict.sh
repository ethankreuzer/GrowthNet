#!/usr/bin/env bash

#SBATCH --job-name=build_rep_dict
#SBATCH --output=/home/ethan2/logs/build_rep_dict_%j.out
#SBATCH --error=/home/ethan2/logs/build_rep_dict_%j.err
#SBATCH --open-mode=append
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=64000
#SBATCH --gres=gpu:3

set -e

echo "Job started at $(date)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

eval "$(conda shell.bash hook 2>/dev/null)"

SCRIPT_DIR="/home/ethan2/GrowthNet/scripts"
SMILES_TXT="/home/ethan2/GrowthNet/data/unique_smiles.txt"
YAML_DIR="/home/ethan2/GrowthNet/data/boltz_yamls"
BOLTZ_OUT="/home/ethan2/GrowthNet/data/boltz_output"

# -------------------------------------------------------------------
# Stage 0: Extract unique SMILES (needs numpy 2.x from my_conda_env)
# -------------------------------------------------------------------
if [ ! -f "$SMILES_TXT" ]; then
    echo "=========================================="
    echo "Stage 0: Extracting unique SMILES"
    echo "=========================================="
    conda activate my_conda_env
    python -c "
import pandas as pd
df_train = pd.read_pickle('/home/ethan2/GrowthNet/data/train/df_well_train_Celine_clusters_mad_4.pkl')
df_val = pd.read_pickle('/home/ethan2/GrowthNet/data/validation/df_well_validation_Celine_clusters_mad_4.pkl')
col = 'Smiles_canonical'
all_smiles = sorted(set(df_train[col].dropna().unique()) | set(df_val[col].dropna().unique()))
with open('$SMILES_TXT', 'w') as f:
    for s in all_smiles:
        f.write(s + '\n')
print(f'Extracted {len(all_smiles)} unique SMILES to $SMILES_TXT')
"
    conda deactivate
else
    echo "SMILES file already exists at $SMILES_TXT, skipping extraction."
fi

# -------------------------------------------------------------------
# Stage 1: Compute fingerprints and write Boltz2 YAML files
# -------------------------------------------------------------------
echo "=========================================="
echo "Stage 1: Prep"
echo "=========================================="
conda activate boltz_env
python "$SCRIPT_DIR/build_representation_dict.py" --stage prep

# -------------------------------------------------------------------
# Stage 2: Run Boltz2 inference to generate embeddings
# -------------------------------------------------------------------
echo "=========================================="
echo "Stage 2: Boltz2 inference"
echo "=========================================="
boltz predict "$YAML_DIR" \
    --write_embeddings \
    --no_kernels \
    --devices 3 \
    --num_workers 8 \
    --out_dir "$BOLTZ_OUT" \
    --override

# -------------------------------------------------------------------
# Stage 3: Pool embeddings and assemble final dictionary
# -------------------------------------------------------------------
echo "=========================================="
echo "Stage 3: Assemble"
echo "=========================================="
python "$SCRIPT_DIR/build_representation_dict.py" --stage assemble

# -------------------------------------------------------------------
# Stage 4: Compute MiniMol molecular fingerprints
# -------------------------------------------------------------------
echo "=========================================="
echo "Stage 4: MiniMol embeddings"
echo "=========================================="
python "$SCRIPT_DIR/build_representation_dict.py" --stage add-minimol

echo "=========================================="
echo "Job completed at $(date)"
echo "=========================================="
