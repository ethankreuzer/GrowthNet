#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
import os


# In[14]:


Celine_train=pd.read_csv('/home/ethan2/GrowthNet/data/Brun_Arroyo_cluster_split_train.csv')
Celine_test=pd.read_csv('/home/ethan2/GrowthNet/data/Brun_Arroyo_cluster_split_test.csv')


df_train = pd.read_pickle("/home/ethan2/GrowthNet/data/train/df_well_train_mad_4.pkl")
df_val  = pd.read_pickle("/home/ethan2/GrowthNet/data/validation/df_well_test_mad_4.pkl") 


# Make my smiles match hers

# In[16]:


def canonicalize_smiles(smi):
    """Return RDKit-canonicalized SMILES, or None if parsing fails."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None

# Apply to your DataFrame
df_train["Smiles_canonical"] = df_train["Smiles"].apply(canonicalize_smiles)
df_val["Smiles_canonical"]   = df_val["Smiles"].apply(canonicalize_smiles)


# In[17]:


# SO whats weird now is I have overlaps. Some smiles map to same t,c OD values. Will need to resolve these conflicts

# In[18]:


df_all = pd.concat([df_train, df_val], ignore_index=True)


# In[19]:




# In[20]:



# Get SMILES sets from Celine’s splits
celine_train_smiles = set(Celine_train["smile_canonical"].unique())
celine_test_smiles = set(Celine_test["smile_canonical"].unique())

# Filter df_train and df_val based on Celine's splits
df_train_new = df_all[df_all["Smiles_canonical"].isin(celine_train_smiles)].copy()
df_val_new   = df_all[df_all["Smiles_canonical"].isin(celine_test_smiles)].copy()

df_train_new=df_train_new[~df_train_new['Concentration'].isin([0.781, 3.13, 12.5])]


# In[21]:





# None of these are active compounds anyway so ignore

# In[23]:


import seaborn as sns

def plot_activity_ratio_heatmap(df):
    # 1) Determine the exact list of concentrations and timepoints, in sorted order
    conc_values = sorted(df['Concentration'].unique())
    time_values = sorted(df['Timepoint'].unique())

    # 2) Build the “total” and “active” count tables, then reindex so they share the same shape/order
    total_counts = (
        df
        .groupby(['Concentration', 'Timepoint'])
        .size()
        .unstack(fill_value=0)
        .reindex(index=conc_values, columns=time_values, fill_value=0)
    )

    active_counts = (
        df[df['is_Active'] == 1]
        .groupby(['Concentration', 'Timepoint'])
        .size()
        .unstack(fill_value=0)
        .reindex(index=conc_values, columns=time_values, fill_value=0)
        .astype(int)
    )

    # 3) Compute fraction = active / total (avoiding division by zero)
    fraction = active_counts.divide(total_counts.replace(0, 1))
    fraction = fraction.fillna(0)

    # 4) Prepare annotation strings “active/total”
    annot = active_counts.astype(str) + "/" + total_counts.astype(str)

    # 5) Plot
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        fraction,
        annot=annot,
        fmt="",
        cmap="viridis",
        vmin=0,
        vmax=0.15,
        cbar_kws={'label': 'Fraction Active'}
    )

    # 6) Set the x‐ and y‐tick labels to the string versions of the numeric values
    ax.set_xticklabels([str(x) for x in time_values])
    ax.set_yticklabels([str(x) for x in conc_values], rotation=0)

    ax.set_xlabel('Timepoint')
    ax.set_ylabel('Concentration')
    ax.set_title('Active / Total Compounds (Test set)')

    plt.tight_layout()
    plt.show()




# In[35]:


#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datamol as dm
import os

# --- Step 1: Deduplicate by canonical SMILES ---
df_train_unique = df_train_new.drop_duplicates(subset=["Smiles_canonical"]).reset_index(drop=True)
df_val_unique   = df_val_new.drop_duplicates(subset=["Smiles_canonical"]).reset_index(drop=True)

# --- Step 2: Concatenate fingerprints ---
def concat_fps(row):
    fps = [row["maccs_fp"], row["ecfp_fp"], row["rdkit_fp"]]
    valid_fps = [fp for fp in fps if isinstance(fp, np.ndarray)]
    if not valid_fps:
        return None
    return np.concatenate(valid_fps)

df_train_unique["concat_fp"] = df_train_unique.apply(concat_fps, axis=1)
df_val_unique["concat_fp"]   = df_val_unique.apply(concat_fps, axis=1)

# Drop molecules without valid fingerprints
df_train_fp = df_train_unique.dropna(subset=["concat_fp"]).reset_index(drop=True)
df_val_fp   = df_val_unique.dropna(subset=["concat_fp"]).reset_index(drop=True)

train_fps = df_train_fp["concat_fp"].tolist()
val_fps   = df_val_fp["concat_fp"].tolist()

# --- Step 3: Compute Tanimoto similarity (vectorized NumPy method) ---
print("Computing similarity matrix with vectorized NumPy...")

# Stack fingerprints into binary matrices
X = np.vstack(train_fps).astype(bool)
Y = np.vstack(val_fps).astype(bool)

# Compute intersections and unions
inter = X @ Y.T
sum_X = X.sum(axis=1, keepdims=True)
sum_Y = Y.sum(axis=1, keepdims=True)
union = sum_X + sum_Y.T - inter

# Compute Tanimoto similarity
similarity_matrix = np.divide(inter, union, out=np.zeros_like(inter, dtype=float), where=union != 0)

print(f"✅ Similarity matrix shape: {similarity_matrix.shape}")

# --- Step 4: Plot heatmap ---
plt.figure(figsize=(10, 8))
sns.heatmap(similarity_matrix, cmap="viridis", vmin=0, vmax=1,
            xticklabels=df_val_fp["Compound"].values,
            yticklabels=df_train_fp["Compound"].values)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title("Tanimoto Similarity Between Unique Train and Validation Compounds", fontsize=14)
plt.xlabel("Validation Compounds")
plt.ylabel("Training Compounds")
plt.tight_layout()

# --- Step 5: Save to file ---
save_path = "/home/ethan2/GrowthNet/plots/tanimoto_sim.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"✅ Heatmap saved to: {save_path}")
