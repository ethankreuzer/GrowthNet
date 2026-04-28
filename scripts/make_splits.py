#!/usr/bin/env python
"""
Build a scaffold-cluster-based train/val/test split from
`data/splits/my_split_v1/all_compound_metas.pkl`.

Pipeline:
  1. Load CompoundMeta list, build per-compound DataFrame with smiles, scaffold,
     is_active_at_12_50, and active_strength = sum(pivot_cls).
  2. Cluster scaffolds with kNN (cosine, n_neighbors=15) → UMAP → Leiden
     over Morgan FPs (radius=2, 2048 bits) of unique scaffolds.
  3. Precompute the full pairwise ECFP4 (Morgan radius=2, 2048-bit) Tanimoto
     matrix over all clusterable compounds.
  4. Two-stage Monte-Carlo cluster sampling. Each candidate cluster selection
     is scored on three constraints:
       - n_actives ≈ TARGET_ACTIVES (within ACTIVE_COUNT_TOL),
       - mean active_strength matches the global mean (within STRENGTH_TOLERANCE),
       - median over selected compounds of max Tanimoto similarity to the
         (prospective) train set is within TANI_TOLERANCE of the stage target:
             test stage: target ≈ TARGET_MEDIAN_TANI_TEST (e.g. 0.45)
             val  stage: target ≈ TARGET_MEDIAN_TANI_VAL  (e.g. 0.50)
  5. Write train.txt / val.txt / test.txt, clusters.csv, and two PNGs:
       tanimoto_max_val_to_train.png, tanimoto_max_test_to_train.png.

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


def cluster_scaffolds(df, radius=2, n_bits=2048, n_neighbors=8, leiden_resolution=8):
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
    tani_target_median: float,
    tani_tolerance: float,
    tani_eval_fn,
    mean_frac: float,
    std_frac: float,
    min_frac: float,
    max_frac: float,
    max_iter: int,
    rng: np.random.Generator,
    label: str = "test",
    tani_floor: float = -np.inf,
):
    all_clusters = cluster_summary["cluster"].to_numpy()
    n_clusters = len(all_clusters)
    cluster_actives  = cluster_summary["n_actives"].to_numpy()
    cluster_strength = cluster_summary["sum_strength"].to_numpy()

    a_min = int(target_actives * (1 - active_count_tol))
    a_max = int(target_actives * (1 + active_count_tol))

    best_score = -np.inf
    best_selection = None
    best_n_act = 0
    best_strength = float("nan")
    best_tani = float("nan")

    best_tani_dev = np.inf
    best_tani_selection = None
    best_tani_n_act = 0
    best_tani_strength = float("nan")
    best_tani_value = float("nan")

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

        candidate_clusters = all_clusters[idx].tolist()
        median_max_tani = tani_eval_fn(candidate_clusters)
        tani_dev = abs(median_max_tani - tani_target_median)

        a_dist = abs(n_act - target_actives) / target_actives
        score = -(a_dist + strength_dev + 10 * tani_dev)

        valid = (
            (a_min <= n_act <= a_max)
            and (strength_dev <= strength_tolerance)
            and (tani_dev <= tani_tolerance)
            and (median_max_tani >= tani_floor)
        )
        if (a_min <= n_act <= a_max) and (median_max_tani >= tani_floor) and tani_dev < best_tani_dev:
            best_tani_dev = tani_dev
            best_tani_selection = candidate_clusters
            best_tani_n_act = n_act
            best_tani_strength = mean_strength
            best_tani_value = median_max_tani
        if score > best_score:
            best_score = score
            best_selection = candidate_clusters
            best_n_act = n_act
            best_strength = mean_strength
            best_tani = median_max_tani
            if valid:
                print(f"    [{label}] iter {it+1}: VALID  "
                      f"n_clusters={n_pick}  n_actives={n_act}  "
                      f"mean_strength={mean_strength:.3f}  "
                      f"median_max_tani={median_max_tani:.4f}")
                return best_selection
        if (it + 1) % 500 == 0:
            print(f"    [{label}] iter {it+1}/{max_iter}  "
                  f"score-best: n_act={best_n_act} strength={best_strength:.3f} tani={best_tani:.4f}  |  "
                  f"tani-best: n_act={best_tani_n_act} strength={best_tani_strength:.3f} tani={best_tani_value:.4f}")

    if best_tani_selection is not None:
        print(f"    [{label}] WARNING: no fully valid selection in {max_iter} iters. "
              f"Returning best-tani candidate (active-count valid): "
              f"n_actives={best_tani_n_act}, mean_strength={best_tani_strength:.3f}, "
              f"median_max_tani={best_tani_value:.4f}")
        return best_tani_selection
    print(f"    [{label}] WARNING: no fully valid selection in {max_iter} iters. "
          f"Returning best-effort: n_actives={best_n_act}, "
          f"mean_strength={best_strength:.3f}, median_max_tani={best_tani:.4f}")
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


def compute_tanimoto_matrix(F: np.ndarray) -> np.ndarray:
    """Full N×N Tanimoto similarity matrix from a binary fingerprint matrix F (N, d).
    Diagonal is set to -inf so a compound can't be its own nearest neighbour."""
    F = (F > 0).astype(np.float32)
    f_sum = F.sum(axis=1)
    inter = F @ F.T
    union = f_sum[:, None] + f_sum[None, :] - inter
    T = inter / np.maximum(union, 1.0)
    T = T.astype(np.float32)
    np.fill_diagonal(T, -np.inf)
    return T


def plot_max_tanimoto_distributions(df: pd.DataFrame, T: np.ndarray, output_dir: Path):
    """Two PNGs: per-val-compound and per-test-compound MAX Tanimoto similarity to train.
    df row positions must align with T row/column indices."""
    val_idx   = np.where(df["split"].values == "val")[0]
    test_idx  = np.where(df["split"].values == "test")[0]
    train_idx = np.where(df["split"].values == "train")[0]
    print(f"\nMax-Tanimoto plots: |train|={len(train_idx)}  |val|={len(val_idx)}  |test|={len(test_idx)}")

    val_max  = T[np.ix_(val_idx,  train_idx)].max(axis=1)
    test_max = T[np.ix_(test_idx, train_idx)].max(axis=1)

    for arr, name, color in [(val_max, "val", "#1f77b4"),
                             (test_max, "test", "#ff7f0e")]:
        mean_v = float(arr.mean())
        med_v  = float(np.median(arr))
        max_v  = float(arr.max())

        fig, ax = plt.subplots(figsize=(10, 5))
        bins = np.linspace(0, max(float(arr.max()), 1.0) * 1.02 + 1e-6, 60)
        ax.hist(arr, bins=bins, alpha=0.7, color=color,
                label=(f"{name} → train (n={len(arr)})\n"
                       f"mean={mean_v:.3f}  median={med_v:.3f}  max={max_v:.3f}"),
                edgecolor="black", linewidth=0.4)
        ax.axvline(med_v, color="red", linestyle="--", linewidth=1.0,
                   label=f"median = {med_v:.3f}")
        ax.set_xlabel("Max ECFP Tanimoto similarity (per held-out compound)")
        ax.set_ylabel(f"Number of {name} compounds")
        ax.set_title(
            f"Per-{name}-compound MAX ECFP Tanimoto similarity to train  "
            f"[mean={mean_v:.3f}  median={med_v:.3f}  max={max_v:.3f}]"
        )
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        out_path = output_dir / f"tanimoto_max_{name}_to_train.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Wrote {out_path}")


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
    OUTPUT_DIR = PROJ / "data/splits/my_split_v1/smile_splits_v2"

    TARGET_ACTIVES          = 250
    ACTIVE_COUNT_TOL        = 0.20
    STRENGTH_TOLERANCE      = 0.15
    TARGET_MEDIAN_TANI_TEST = 0.40
    TARGET_MEDIAN_TANI_VAL  = 0.42
    TANI_TOLERANCE          = 0.025
    MEAN_CLUSTER_FRAC       = 0.10
    STD_CLUSTER_FRAC        = 0.03
    MIN_CLUSTER_FRAC        = 0.05
    MAX_CLUSTER_FRAC        = 0.25
    MAX_ITER                = 20000
    RANDOM_SEED             = 42
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
    df = df.dropna(subset=["cluster"]).reset_index(drop=True)
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

    # ---- Precompute pairwise ECFP4 (Morgan radius=2, 2048-bit) Tanimoto matrix ----
    # df row positions are aligned with T row/column indices (df was reset_index'd above).
    print("\nComputing pairwise ECFP Tanimoto matrix over all clusterable compounds...")
    F = collect_ecfp_array(metas, df["smiles"].tolist())
    assert F.shape[1] == 2048, f"Expected 2048-bit ECFP fingerprints, got {F.shape[1]}"
    print(f"  Fingerprint matrix: {F.shape}")
    T = compute_tanimoto_matrix(F)
    print(f"  Tanimoto matrix: {T.shape}  ({T.nbytes / 1e6:.1f} MB)")

    n_total = len(df)
    cluster_to_idx = {
        int(c): np.where(df["cluster"].values == int(c))[0]
        for c in df["cluster"].unique()
    }

    print("\n--- Natural-ceiling diagnostic for median_max_tani ---")
    compound_max_tani = T.max(axis=1)
    print(f"  (1) Compound-level natural ceiling (every test compound sees its single closest neighbour in train):")
    print(f"        median={np.median(compound_max_tani):.4f}  mean={compound_max_tani.mean():.4f}  "
          f"p25={np.percentile(compound_max_tani, 25):.4f}  p75={np.percentile(compound_max_tani, 75):.4f}")

    per_cluster_tani = []
    for c, idx in cluster_to_idx.items():
        if len(idx) == 0 or len(idx) == n_total:
            continue
        rest_idx = np.setdiff1d(np.arange(n_total), idx, assume_unique=True)
        sub = T[np.ix_(idx, rest_idx)]
        per_cluster_tani.append(float(np.median(sub.max(axis=1))))
    per_cluster_tani = np.array(per_cluster_tani)
    print(f"  (2) Per-cluster median_max_tani (if cluster ALONE were the test set):")
    print(f"        min={per_cluster_tani.min():.4f}  p25={np.percentile(per_cluster_tani, 25):.4f}  "
          f"median={np.median(per_cluster_tani):.4f}  p75={np.percentile(per_cluster_tani, 75):.4f}  "
          f"max={per_cluster_tani.max():.4f}")

    not_own_cluster_max = np.full(n_total, np.nan)
    for c, idx in cluster_to_idx.items():
        if len(idx) == 0 or len(idx) == n_total:
            continue
        rest_idx = np.setdiff1d(np.arange(n_total), idx, assume_unique=True)
        sub = T[np.ix_(idx, rest_idx)]
        not_own_cluster_max[idx] = sub.max(axis=1)
    finite = not_own_cluster_max[np.isfinite(not_own_cluster_max)]
    print(f"  (3) Whole-cluster-holdout expectation (each compound vs not-own-cluster):")
    print(f"        median={np.median(finite):.4f}  mean={finite.mean():.4f}  "
          f"p25={np.percentile(finite, 25):.4f}  p75={np.percentile(finite, 75):.4f}")
    print(f"  → Set TARGET_MEDIAN_TANI_TEST/VAL near (3); the spread of (2) shows the achievable wiggle room.\n")

    def make_tani_eval(base_mask: np.ndarray):
        """Closure: given a list of selected cluster ids, return the median over
        selected compounds of max Tanimoto similarity to compounds in `base_mask`
        minus the selected compounds themselves."""
        def eval_fn(selected_clusters):
            sel_idx = np.concatenate([cluster_to_idx[int(c)] for c in selected_clusters])
            comp_mask = base_mask.copy()
            comp_mask[sel_idx] = False
            comp_idx = np.where(comp_mask)[0]
            if len(comp_idx) == 0 or len(sel_idx) == 0:
                return float("nan")
            sub = T[np.ix_(sel_idx, comp_idx)]
            return float(np.median(sub.max(axis=1)))
        return eval_fn

    rng = np.random.default_rng(RANDOM_SEED)

    # ---- Stage 1: select test clusters ----
    # complement (proxy for train) = all compounds except selected test
    base_mask_test = np.ones(n_total, dtype=bool)
    test_tani_eval = make_tani_eval(base_mask_test)

    print("\nMonte Carlo: sampling TEST clusters...")
    test_clusters = sample_clusters(
        cluster_summary,
        target_actives=TARGET_ACTIVES,
        active_count_tol=ACTIVE_COUNT_TOL,
        strength_tolerance=STRENGTH_TOLERANCE,
        mean_strength_global=mean_strength_global,
        tani_target_median=TARGET_MEDIAN_TANI_TEST,
        tani_tolerance=TANI_TOLERANCE,
        tani_eval_fn=test_tani_eval,
        mean_frac=MEAN_CLUSTER_FRAC,
        std_frac=STD_CLUSTER_FRAC,
        min_frac=MIN_CLUSTER_FRAC,
        max_frac=MAX_CLUSTER_FRAC,
        max_iter=MAX_ITER,
        rng=rng,
        label="test",
    )

    test_achieved_tani = test_tani_eval(test_clusters)
    print(f"  Test achieved median_max_tani = {test_achieved_tani:.4f}  (val will be floored at this value)")

    # ---- Stage 2: select val clusters from remainder ----
    # complement = train (i.e. all compounds except test and selected val)
    test_mask = np.zeros(n_total, dtype=bool)
    for c in test_clusters:
        test_mask[cluster_to_idx[int(c)]] = True
    base_mask_val = ~test_mask
    val_tani_eval = make_tani_eval(base_mask_val)

    remaining = cluster_summary[~cluster_summary["cluster"].isin(test_clusters)].reset_index(drop=True)
    print("\nMonte Carlo: sampling VAL clusters from remaining...")
    val_clusters = sample_clusters(
        remaining,
        target_actives=TARGET_ACTIVES,
        active_count_tol=ACTIVE_COUNT_TOL,
        strength_tolerance=STRENGTH_TOLERANCE,
        mean_strength_global=mean_strength_global,
        tani_target_median=TARGET_MEDIAN_TANI_VAL,
        tani_tolerance=TANI_TOLERANCE,
        tani_eval_fn=val_tani_eval,
        mean_frac=MEAN_CLUSTER_FRAC,
        std_frac=STD_CLUSTER_FRAC,
        min_frac=MIN_CLUSTER_FRAC,
        max_frac=MAX_CLUSTER_FRAC,
        max_iter=MAX_ITER,
        rng=rng,
        label="val",
        tani_floor=test_achieved_tani,
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

    # ---- Max-Tanimoto leakage diagnostics ----
    plot_max_tanimoto_distributions(df, T, OUTPUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
