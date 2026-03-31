#!/usr/bin/env python3
"""
Efficiently builds an augmented validation dataframe with dense timepoints.
Vectorized to avoid Python loops — suitable for sbatch/HPC execution.
"""

import os
import numpy as np
import pandas as pd
import time

# === CONFIG ===
INPUT_PATH = "/home/ethan2/GrowthNet/data/validation/df_well_validation_Celine_clusters_mad_4.pkl"
OUTPUT_PATH = "/home/ethan2/GrowthNet/data/validation/df_well_validation_Celine_dense_timepoints_original_concs.pkl"

TIME_START = 0.0
TIME_END = 13.0
TIME_STEP = 0.1
CONCENTRATIONS = [0.2, 1.2, 3.13, 7.9, 12.5, 50]  # µM

t0 = time.time()

print(f"📂 Loading validation dataframe from: {INPUT_PATH}")
df_val = pd.read_pickle(INPUT_PATH)
print(f"✅ Loaded df_val with shape {df_val.shape}")

# === Extract relevant columns ===
fp_cols = [c for c in df_val.columns if c.endswith("_fp")]
meta_cols = ["Compound", "Smiles", "Smiles_canonical", "scaffold", "Control_Label"]
cols_to_keep = ["Concentration"] + meta_cols + fp_cols

# Keep one row per compound–concentration pair
df_base = (
    df_val[cols_to_keep]
    .drop_duplicates(subset=["Compound", "Concentration"])
    .reset_index(drop=True)
)

# Filter to target concentrations
df_base = df_base[df_base["Concentration"].isin(CONCENTRATIONS)].reset_index(drop=True)
n_pairs = len(df_base)
print(f"🧪 Retained {n_pairs} unique (Compound, Concentration) pairs at {len(CONCENTRATIONS)} concentrations.")

# === Dense time grid ===
time_grid = np.arange(TIME_START, TIME_END + TIME_STEP, TIME_STEP)
n_times = len(time_grid)
print(f"⏱️ Time grid: {TIME_START} → {TIME_END} every {TIME_STEP}h ({n_times} points)")

# === Vectorized expansion ===
# Repeat base rows for each timepoint
df_expanded = pd.concat([df_base] * n_times, ignore_index=True)
# Tile timepoints across all rows
df_expanded["Timepoint"] = np.tile(time_grid, n_pairs)
# Add dummy targets
df_expanded["OD"] = -1.0
df_expanded["is_Active"] = -1

print(f"✅ Expanded dataframe shape: {df_expanded.shape} "
      f"({n_pairs} pairs × {n_times} timepoints = {n_pairs * n_times:,} rows)")

# === Save ===
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df_expanded.to_pickle(OUTPUT_PATH)
print(f"💾 Saved dense augmented dataframe to:\n   {OUTPUT_PATH}")
print(f"⏱️ Done in {time.time() - t0:.2f} seconds.")
