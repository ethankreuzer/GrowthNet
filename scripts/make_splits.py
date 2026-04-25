#!/usr/bin/env python
"""
Build a scaffold-cluster-based train/val/test split from
`data/splits/my_split_v1/all_compound_metas.pkl`.

Pipeline (mirrors cluster_scaffolds_gneprop.py + test_train_split_clusters.py):
  1. Load CompoundMeta list, build per-compound DataFrame with smiles, scaffold,
     is_active_at_12_50, and active_strength = sum(pivot_cls).
  2. Cluster scaffolds with kNN (cosine, n_neighbors=15) → UMAP → Leiden
     (resolution=1.0) over Morgan FPs (radius=2, 2048 bits) of unique scaffolds.
  3. Two-stage Monte-Carlo cluster sampling:
       - Stage 1: pick test clusters (~250 actives, mean strength matches global).
       - Stage 2: pick val clusters from remaining (~250 actives, matched).
       - Train = the rest.
  4. Write train.txt / val.txt / test.txt and a clusters.csv.

Edit the path constants at the top of main() before running.
"""

import pickle
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=FutureWarning)

PROJ = Path("/home/ethan2/GrowthNet")
sys.path.insert(0, str(PROJ / "sweeps"))

from data_class import CompoundMeta  # noqa: E402  (sys.path setup above)


# ---------------------------------------------------------------------------
# Build per-compound DataFrame from CompoundMeta list
# ---------------------------------------------------------------------------
def metas_to_dataframe(metas):
    rows = []
    for m in metas:
        rows.append({
            "compound": m.compound,
            "smiles": m.smiles,
            "scaffold": m.scaffold,
            "is_active_at_12_50": bool(m.is_active_at_12_50),
            "active_strength": int(m.pivot_cls.values.sum()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Scaffold clustering via kNN + UMAP + Leiden
# ---------------------------------------------------------------------------
def morgan_fp_array(scaffolds, radius=2, n_bits=2048):
    arr = np.zeros((len(scaffolds), n_bits), dtype=np.int8)
    valid_mask = np.ones(len(scaffolds), dtype=bool)
    for i, s in enumerate(scaffolds):
        mol = Chem.MolFromSmiles(s) if s else None
        if mol is None:
            valid_mask[i] = False
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        arr[i] = np.frombuffer(fp.ToBitString().encode(), dtype=np.uint8) - ord("0")
    return arr, valid_mask


def cluster_scaffolds(df, radius=2, n_bits=2048, n_neighbors=15, leiden_resolution=2.5):
    import scanpy as sc
    from anndata import AnnData

    unique_scaffolds = df["scaffold"].dropna().unique().tolist()
    print(f"  Clustering {len(unique_scaffolds)} unique scaffolds...")

    fp_array, valid_mask = morgan_fp_array(unique_scaffolds, radius=radius, n_bits=n_bits)
    valid_scaffolds = [s for s, ok in zip(unique_scaffolds, valid_mask) if ok]
    fp_array = fp_array[valid_mask]
    n_invalid = (~valid_mask).sum()
    if n_invalid:
        print(f"  WARNING: {n_invalid} scaffolds failed FP generation and will be unclustered.")

    print("  Running kNN + UMAP + Leiden...")
    adata = AnnData(X=fp_array.astype(np.float32))
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X", metric="cosine")
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=leiden_resolution)

    cluster_ids = adata.obs["leiden"].values.astype(int)
    umap_coords = adata.obsm["X_umap"]
    scaffold_to_cluster = dict(zip(valid_scaffolds, cluster_ids))
    scaffold_to_umap = dict(zip(valid_scaffolds, umap_coords))

    df = df.copy()
    df["cluster"] = df["scaffold"].map(scaffold_to_cluster)
    df["umap_1"] = df["scaffold"].map(lambda s: scaffold_to_umap.get(s, [np.nan, np.nan])[0])
    df["umap_2"] = df["scaffold"].map(lambda s: scaffold_to_umap.get(s, [np.nan, np.nan])[1])

    n_clusters = df["cluster"].dropna().nunique()
    print(f"  → {n_clusters} clusters formed.")
    return df


# ---------------------------------------------------------------------------
# Monte Carlo cluster sampling (one stage)
# ---------------------------------------------------------------------------
def sample_clusters(
    cluster_summary: pd.DataFrame,   # one row per cluster, cols: cluster, n_actives, sum_strength
    target_actives: int,
    active_count_tol: float,
    strength_tolerance: float,
    mean_strength_global: float,
    mean_frac: float,
    std_frac: float,
    min_frac: float,
    max_frac: float,
    max_iter: int,
    rng: np.random.Generator,
    label: str = "test",
):
    all_clusters = cluster_summary["cluster"].to_numpy()
    n_clusters = len(all_clusters)
    cluster_actives  = cluster_summary["n_actives"].to_numpy()
    cluster_strength = cluster_summary["sum_strength"].to_numpy()

    a_min = int(target_actives * (1 - active_count_tol))
    a_max = int(target_actives * (1 + active_count_tol))

    best_score = -np.inf
    best_selection = None

    for it in range(max_iter):
        while True:
            frac = rng.normal(loc=mean_frac, scale=std_frac)
            if min_frac <= frac <= max_frac:
                break
        n_pick = max(1, int(round(frac * n_clusters)))
        idx = rng.choice(n_clusters, size=n_pick, replace=False)

        n_act = int(cluster_actives[idx].sum())
        if n_act == 0:
            continue
        mean_strength = cluster_strength[idx].sum() / n_act
        strength_dev  = abs(mean_strength - mean_strength_global) / mean_strength_global

        # Closeness score: penalise distance from active target and strength deviation.
        a_dist = abs(n_act - target_actives) / target_actives
        score = -(a_dist + strength_dev)

        valid = (a_min <= n_act <= a_max) and (strength_dev <= strength_tolerance)
        if score > best_score:
            best_score = score
            best_selection = all_clusters[idx].tolist()
            best_n_act = n_act
            best_strength = mean_strength
            if valid:
                print(f"    [{label}] iter {it+1}: VALID  "
                      f"n_clusters={n_pick}  n_actives={n_act}  mean_strength={mean_strength:.3f}")
                return best_selection
        if (it + 1) % 500 == 0:
            print(f"    [{label}] iter {it+1}/{max_iter}  best n_actives={best_n_act}  "
                  f"best mean_strength={best_strength:.3f}")

    print(f"    [{label}] WARNING: no fully valid selection in {max_iter} iters. "
          f"Returning best-effort: n_actives={best_n_act}, mean_strength={best_strength:.3f}")
    return best_selection


# ---------------------------------------------------------------------------
# Tanimoto leakage diagnostics (ECFP, full-molecule)
# ---------------------------------------------------------------------------
def collect_ecfp_array(metas, smiles_list):
    """Return (n, d) bit array of ECFP for the given canonical smiles, aligned to smiles_list."""
    smi_to_fp = {m.smiles: m.fps_by_family.get("ecfp_fp") for m in metas
                 if "ecfp_fp" in m.fps_by_family}
    sample = next(v for v in smi_to_fp.values() if v is not None)
    d = sample.shape[0]
    out = np.zeros((len(smiles_list), d), dtype=np.float32)
    missing = 0
    for i, s in enumerate(smiles_list):
        fp = smi_to_fp.get(s)
        if fp is None:
            missing += 1
            continue
        out[i] = fp
    if missing:
        print(f"  WARNING: {missing} compounds had no ECFP fingerprint.")
    return out


def per_row_mean_tanimoto(A, B, batch_size=500):
    """For each row in A, compute mean Tanimoto similarity to all rows in B (vectorised, batched)."""
    A = (A > 0).astype(np.float32)
    B = (B > 0).astype(np.float32)
    a_sum = A.sum(axis=1)
    b_sum = B.sum(axis=1)
    out = np.zeros(len(A), dtype=np.float32)
    for i in range(0, len(A), batch_size):
        a_batch = A[i:i + batch_size]
        intersection = a_batch @ B.T
        union = a_sum[i:i + batch_size, None] + b_sum[None, :] - intersection
        tani = intersection / np.maximum(union, 1.0)
        out[i:i + batch_size] = tani.mean(axis=1)
    return out


def plot_tanimoto_distributions(metas, df, output_dir):
    """
    Plot 1: distribution of per-train-compound mean Tanimoto similarity to
            (a) val compounds and (b) test compounds, overlaid.
    Plot 2: distribution of per-val-compound mean Tanimoto similarity to test.
    """
    train_smiles = df.loc[df["split"] == "train", "smiles"].drop_duplicates().tolist()
    val_smiles   = df.loc[df["split"] == "val",   "smiles"].drop_duplicates().tolist()
    test_smiles  = df.loc[df["split"] == "test",  "smiles"].drop_duplicates().tolist()

    print("\nComputing ECFP arrays for Tanimoto plots...")
    train_fp = collect_ecfp_array(metas, train_smiles)
    val_fp   = collect_ecfp_array(metas, val_smiles)
    test_fp  = collect_ecfp_array(metas, test_smiles)
    print(f"  train_fp={train_fp.shape}  val_fp={val_fp.shape}  test_fp={test_fp.shape}")

    # ---- Plot 1: train → val and train → test ----
    print("  Computing train→val mean Tanimoto...")
    train_to_val  = per_row_mean_tanimoto(train_fp, val_fp)
    print("  Computing train→test mean Tanimoto...")
    train_to_test = per_row_mean_tanimoto(train_fp, test_fp)

    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0,
                       max(train_to_val.max(), train_to_test.max()) * 1.05 + 1e-6,
                       60)
    ax.hist(train_to_val,  bins=bins, alpha=0.55, color="#1f77b4",
            label=f"train → val (n={len(train_to_val)}, μ={train_to_val.mean():.3f})",
            edgecolor="black", linewidth=0.4)
    ax.hist(train_to_test, bins=bins, alpha=0.55, color="#ff7f0e",
            label=f"train → test (n={len(train_to_test)}, μ={train_to_test.mean():.3f})",
            edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Mean Tanimoto similarity (per train compound)")
    ax.set_ylabel("Number of train compounds")
    ax.set_title("Per-train-compound mean ECFP Tanimoto similarity to held-out sets")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    p1 = output_dir / "tanimoto_train_to_val_test.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"  Wrote {p1}")

    # ---- Plot 2: val → test ----
    print("  Computing val→test mean Tanimoto...")
    val_to_test = per_row_mean_tanimoto(val_fp, test_fp)

    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, val_to_test.max() * 1.05 + 1e-6, 60)
    ax.hist(val_to_test, bins=bins, alpha=0.7, color="#2ca02c",
            label=f"val → test (n={len(val_to_test)}, μ={val_to_test.mean():.3f})",
            edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Mean Tanimoto similarity (per val compound)")
    ax.set_ylabel("Number of val compounds")
    ax.set_title("Per-val-compound mean ECFP Tanimoto similarity to test set")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    p2 = output_dir / "tanimoto_val_to_test.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"  Wrote {p2}")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def report_split(df: pd.DataFrame, total_actives: int, total_compounds: int):
    print("\n=== Split summary ===")
    grp = df.groupby("split")
    for name in ["train", "val", "test"]:
        sub = grp.get_group(name) if name in grp.groups else df.iloc[:0]
        n_comp = len(sub)
        n_clust = sub["cluster"].nunique()
        n_act = int(sub["is_active_at_12_50"].sum())
        actives_only = sub[sub["is_active_at_12_50"]]
        mean_str = actives_only["active_strength"].mean() if len(actives_only) else float("nan")
        med_str  = actives_only["active_strength"].median() if len(actives_only) else float("nan")
        print(f"  {name:5s}  compounds={n_comp:>6,d} ({n_comp/total_compounds:.1%})  "
              f"clusters={n_clust:>4d}  "
              f"actives={n_act:>4d} ({n_act/total_actives:.1%})  "
              f"mean_strength={mean_str:.2f}  median_strength={med_str:.1f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # =========================================================
    # Configure paths and parameters before running
    # =========================================================
    METAS_PATH = PROJ / "data/splits/my_split_v1/all_compound_metas.pkl"
    OUTPUT_DIR = PROJ / "data/splits/my_split_v1/smiles_splits"

    TARGET_ACTIVES        = 250
    ACTIVE_COUNT_TOL      = 0.20
    STRENGTH_TOLERANCE    = 0.15
    MEAN_CLUSTER_FRAC     = 0.10
    STD_CLUSTER_FRAC      = 0.03
    MIN_CLUSTER_FRAC      = 0.05
    MAX_CLUSTER_FRAC      = 0.25
    MAX_ITER              = 5000
    RANDOM_SEED           = 42
    # =========================================================

    print("Loading CompoundMeta list...")
    with open(METAS_PATH, "rb") as f:
        metas = pickle.load(f)
    print(f"  Loaded {len(metas):,} compounds.")

    df = metas_to_dataframe(metas)
    total_compounds = len(df)
    total_actives   = int(df["is_active_at_12_50"].sum())
    print(f"  Total actives at (t=12.48, c=50): {total_actives}")

    # Drop rows without a scaffold (cannot cluster)
    no_scaffold = df["scaffold"].isna() | (df["scaffold"] == "")
    if no_scaffold.any():
        print(f"  WARNING: {no_scaffold.sum()} compounds have no scaffold; will be excluded.")
        df = df[~no_scaffold].reset_index(drop=True)

    print("\nClustering scaffolds...")
    df = cluster_scaffolds(df)
    df = df.dropna(subset=["cluster"]).copy()
    df["cluster"] = df["cluster"].astype(int)

    # Per-cluster summary used by the MC sampler
    cluster_summary = df.groupby("cluster", as_index=False).agg(
        n_actives=("is_active_at_12_50", "sum"),
        sum_strength=("active_strength", lambda s: s[df.loc[s.index, "is_active_at_12_50"]].sum()),
    )
    cluster_summary["n_actives"] = cluster_summary["n_actives"].astype(int)
    cluster_summary["sum_strength"] = cluster_summary["sum_strength"].astype(int)

    # Global mean active strength (over actives only)
    actives_global = df[df["is_active_at_12_50"]]
    mean_strength_global = float(actives_global["active_strength"].mean())
    print(f"  Global mean active_strength = {mean_strength_global:.3f} "
          f"(n_actives={len(actives_global)})")

    rng = np.random.default_rng(RANDOM_SEED)

    # ---- Stage 1: select test clusters ----
    print("\nMonte Carlo: sampling TEST clusters...")
    test_clusters = sample_clusters(
        cluster_summary,
        target_actives=TARGET_ACTIVES,
        active_count_tol=ACTIVE_COUNT_TOL,
        strength_tolerance=STRENGTH_TOLERANCE,
        mean_strength_global=mean_strength_global,
        mean_frac=MEAN_CLUSTER_FRAC,
        std_frac=STD_CLUSTER_FRAC,
        min_frac=MIN_CLUSTER_FRAC,
        max_frac=MAX_CLUSTER_FRAC,
        max_iter=MAX_ITER,
        rng=rng,
        label="test",
    )

    # ---- Stage 2: select val clusters from remainder ----
    remaining = cluster_summary[~cluster_summary["cluster"].isin(test_clusters)].reset_index(drop=True)
    print("\nMonte Carlo: sampling VAL clusters from remaining...")
    val_clusters = sample_clusters(
        remaining,
        target_actives=TARGET_ACTIVES,
        active_count_tol=ACTIVE_COUNT_TOL,
        strength_tolerance=STRENGTH_TOLERANCE,
        mean_strength_global=mean_strength_global,
        mean_frac=MEAN_CLUSTER_FRAC,
        std_frac=STD_CLUSTER_FRAC,
        min_frac=MIN_CLUSTER_FRAC,
        max_frac=MAX_CLUSTER_FRAC,
        max_iter=MAX_ITER,
        rng=rng,
        label="val",
    )

    # ---- Annotate with split ----
    df["split"] = "train"
    df.loc[df["cluster"].isin(test_clusters), "split"] = "test"
    df.loc[df["cluster"].isin(val_clusters), "split"]  = "val"

    # ---- Write outputs ----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in ["train", "val", "test"]:
        smis = df.loc[df["split"] == name, "smiles"].drop_duplicates().tolist()
        (OUTPUT_DIR / f"{name}.txt").write_text("\n".join(smis) + "\n")
        print(f"  Wrote {len(smis):>6,d} SMILES → {OUTPUT_DIR / f'{name}.txt'}")

    cols = ["compound", "smiles", "scaffold", "cluster", "umap_1", "umap_2",
            "is_active_at_12_50", "active_strength", "split"]
    df[cols].to_csv(OUTPUT_DIR / "clusters.csv", index=False)
    print(f"  Wrote clusters.csv → {OUTPUT_DIR / 'clusters.csv'}")

    # ---- Report ----
    report_split(df, total_actives=total_actives, total_compounds=total_compounds)

    # ---- Tanimoto leakage diagnostics ----
    plot_tanimoto_distributions(metas, df, OUTPUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
