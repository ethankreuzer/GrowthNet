import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

# --------------------------
# Load datasets
# --------------------------
df_train = pd.read_pickle("/home/ethan2/GrowthNet/data/train/df_well_train_Celine_clusters_mad_4.pkl")
#df_val   = pd.read_pickle("/home/ethan2/GrowthNet/data/validation/df_well_validation_Celine_clusters_mad_4.pkl")
df_test=pd.read_pickle('/home/ethan2/GrowthNet/data/test/df_test_normalized_mean_OD_then_mad_4_t_12.pkl')


# --------------------------
# Deduplicate by compound
# --------------------------
def unique_compounds(df):
    """Keep one entry per unique canonical SMILES."""
    df_unique = df.drop_duplicates(subset=["Smiles_canonical"]).reset_index(drop=True)
    return df_unique


#Only look at active compounds


# Deduplicate full sets
df_train_u = unique_compounds(df_train)
df_test_u   = unique_compounds(df_test)

# Filter first, then deduplicate
df_test_active_u = unique_compounds(df_test[df_test["is_Active"] == 1])

print(f"Unique compounds: Train = {len(df_train_u)}, Test = {len(df_test_u)}")
print(f"Unique Active compounds: Test = {len(df_test_active_u)}")

# --------------------------
# Build concatenated fingerprint arrays
# --------------------------
def stack_fp(df):
    """Concatenate all binary fingerprint arrays into one matrix."""
    fps = []
    for fp_name in ["maccs_fp", "ecfp_fp", "rdkit_fp"]:
        fp_arr = np.stack(df[fp_name].to_numpy()).astype(np.uint8)
        fps.append(fp_arr)
    return np.concatenate(fps, axis=1)

train_fp = stack_fp(df_train_u)
test_fp   = stack_fp(df_test_u)

print(f"Train FP shape: {train_fp.shape}")
print(f"Test FP shape:   {test_fp.shape}")

# --------------------------
# Efficient Tanimoto similarity
# --------------------------
# ==========================================================
# Compute max similarity for each VALIDATION compound
# ==========================================================

def tanimoto_batch_reverse(B, A, batch_size=500):
    """
    Compute max Tanimoto similarity for each compound in B
    (e.g., validation) to all compounds in A (e.g., training).
    Uses batch processing for efficiency.
    """
    nB, nA = B.shape[0], A.shape[0]
    max_sims = np.zeros(nB, dtype=np.float32)

    A_sum = A.sum(axis=1).astype(np.float32)

    for i in tqdm(range(0, nB, batch_size), desc="Computing max similarity per validation compound"):
        B_batch = B[i:i+batch_size]
        inter = B_batch @ A.T
        B_sum = B_batch.sum(axis=1, keepdims=True).astype(np.float32)
        union = B_sum + A_sum - inter
        sims = inter / np.clip(union, 1e-9, None)
        max_sims[i:i+batch_size] = sims.max(axis=1)

    return max_sims


output_dir = "/home/ethan2/GrowthNet/plots/similarity"
os.makedirs(output_dir, exist_ok=True)

# Compute max similarity per validation compound
max_sims_val = tanimoto_batch_reverse(test_fp, train_fp, batch_size=500)
np.save(os.path.join(output_dir, "max_similarity_per_test.npy"), max_sims_val)

# ==========================================================
# Plot histogram
# ==========================================================
plt.figure(figsize=(8, 5))
sns.histplot(max_sims_val, bins=50, kde=True, color="darkorange")
plt.title("Distribution of Max Tanimoto Similarity of Compounds (Test→ Training)")
plt.xlabel("Max Similarity to Test Compounds")
plt.ylabel("Number of Test Compounds")
plt.xlim(0, 1)
plt.tight_layout()

save_path_val = os.path.join(output_dir, "max_similarity_test_histogram.png")
plt.savefig(save_path_val, dpi=300)
print(f"✅ Saved test similarity histogram to: {save_path_val}")
plt.show()
