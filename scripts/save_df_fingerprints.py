#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
import datamol as dm
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from multiprocessing import Pool, cpu_count
import os

df_no_correction=pd.read_pickle('/home/ethan2/GrowthNet/data/df_no_correction.pkl')
df_well=pd.read_pickle('/home/ethan2/GrowthNet/data/df_well_corrected.pkl')
df_plate_t12=pd.read_pickle('/home/ethan2/GrowthNet/data/df_well_plate_correction_t12.pkl')


# In[5]:


datasets=[df_no_correction,df_well,df_plate_t12]


def compute_scaffold(smiles):
    if not isinstance(smiles, str):  # Guard against NaN or non-string
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    return None

def maccs_to_fp(smile):
    try:

        return dm.to_fp(smile, fp_type="maccs")
    except Exception as e:

        print(f"Error processing SMILES '{smile}': {e}")
        return np.nan


def ecfp_to_fp(smile):
    try:

        return dm.to_fp(smile, fp_type="ecfp")
    except Exception as e:

        print(f"Error processing SMILES '{smile}': {e}")
        return np.nan


def rdkit_to_fp(smile):
    try:

        return dm.to_fp(smile, fp_type="rdkit")
    except Exception as e:

        print(f"Error processing SMILES '{smile}': {e}")
        return np.nan

# 2) Worker to compute all four features
def featurize(smile):
    return {
        'Smiles':   smile,
        'scaffold': compute_scaffold(smile),
        'maccs_fp':   maccs_to_fp(smile),
        'ecfp_fp':    ecfp_to_fp(smile),
        'rdkit_fp':   rdkit_to_fp(smile),
    }


if __name__ == "__main__":
    
    # 1) collect unique SMILES …
    all_smiles = pd.concat([
    df_no_correction['Smiles'],
    df_well['Smiles'],
    df_plate_t12['Smiles'],
    ]).dropna().unique()

    # 2) launch the pool with exactly `cpus` workers
    #    use chunksize to keep them well loaded
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", cpu_count()))
    chunk = max(1, len(all_smiles) // (cpus * 4))

    print(f"Running with {cpus=} workers, chunk size {chunk=}")
    
    with Pool(processes=cpus) as pool:
        rows = pool.map(featurize, all_smiles, chunksize=chunk)

    # 3) build fps_df, merge back, and finally…
    # 4) save each DataFrame as pickle

    fps_df = pd.DataFrame(rows)

    output_dir = "/home/ethan2/GrowthNet/data/"
    for name, df in [
        ("df_no_correction",  df_no_correction),
        ("df_well",           df_well),
        ("df_plate_t12",      df_plate_t12),
    ]:
        # a) Drop any stale fingerprint columns if re‐running
        df = df.drop(columns=['scaffold','maccs_fp','ecfp_fp','rdkit_fp'],
                     errors='ignore')
        # b) Merge on 'Smiles' to bring in the new features
        df = df.merge(fps_df, on='Smiles', how='left')
        # c) Save the enriched DataFrame
        path = f"{output_dir}{name}_fingerprints.pkl"
        df.to_pickle(path)
        print(f"Wrote {path}")

    print(f"Done with {cpus} workers, chunk size {chunk}")

