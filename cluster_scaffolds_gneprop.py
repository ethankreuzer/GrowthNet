#script to cluster molecules based on scaffolds, using GNEprop approach, i.e. UMAP + Leiden algorithm.

import pandas as pd
import numpy as np
import sys
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from tap import tap, tapify
from anndata import AnnData
import scanpy as sc
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")  # Disable RDKit warnings

# Add root repo to sys.path so utils can be imported
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.logging import get_logger
from utils.data import get_scaffold, generate_fingerprints

# Initialize logger
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = Path(f"data/logs/c_{timestamp}.log")
logger = get_logger(log_file, name="clustering")
logger.info("Starting clustering script...")


def cluster_scaffolds(
    input_file: str,
    output_dir: str = "data/2_processed/clustered",
    top_n_plot: int = 20,
    logger=logger
) -> None:
    """
    Cluster scaffolds using kNN + UMAP + Leiden from an input CSV file,
    save DataFrame with cluster assignments, UMAP coordinates, and summary stats.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if logger:
        logger.info(f"Loading input file: {input_file}")

    # Read input file
    df = pd.read_csv(input_file)

    
    invalid_smiles = [] # to keep track of molecules where murcko scaffolds or fps could not be computed
    
    # Compute Murcko scaffolds
    unique_smiles = df['smile_canonical'].dropna().unique()
    if logger: logger.info("Computing Murcko scaffolds...")
            
    smiles_to_scaffold = {s: get_scaffold(s) for s in tqdm(unique_smiles, desc="Scaffolds")}
    df['murcko_scaffold'] = df['smile_canonical'].map(smiles_to_scaffold)
    # filter out invalid scaffolds
    no_scaffold = df[df['murcko_scaffold'].isna()] 
    logger.warning(f"{len(no_scaffold)} scaffolds could not be computed.")
    input_stem = Path(input_file).stem
    invalid_scaf_path = output_dir / f"invalid_scaffolds_{input_stem}.csv"
    no_scaffold.to_csv(invalid_scaf_path, index=False)
    
    unique_scaffolds = df['murcko_scaffold'].dropna().unique().tolist()

    # Generate Morgan fingerprints
    if logger: logger.info("Generating Morgan fingerprints...")
    fps = generate_fingerprints(unique_scaffolds, radius=2, fp_size=2048, desc="Fingerprints")

   # Remove None fps and convert to array
    valid_scaffolds = [s for s, fp in zip(unique_scaffolds, fps) if fp is not None]
    valid_fps = [fp for fp in fps if fp is not None]

    invalid_scaffolds = [s for s, fp in zip(unique_scaffolds, fps) if fp is None]

    
    # Log and save invalid scaffolds
    if logger:
        logger.warning(f"{len(invalid_scaffolds)} scaffolds had invalid fingerprints.")
        for s in invalid_scaffolds:
            logger.warning(f"Scaffold generating None fps: {s}")
    
    
    # save filename
    invalid_df = pd.DataFrame({"invalid_scaffolds": invalid_scaffolds})
    invalid_path = output_dir / f"invalid_scaffolds_fps_{input_stem}.csv"
    invalid_df.to_csv(invalid_path, index=False)
    
    #convert valid fps to array
    fp_array = np.array([np.array(list(fp.ToBitString()), dtype=int) for fp in valid_fps])

    
    # kNN + UMAP + Leiden clustering
    if logger: logger.info("Clustering scaffolds (kNN + UMAP + Leiden)...")
    adata = AnnData(X=fp_array)
    
    #compute kNN graph for all scaffolds based on their fingerprints
    sc.pp.neighbors(adata, n_neighbors=15, use_rep='X', metric='cosine')
    
    #UMAP dimensionality reduction
    sc.tl.umap(adata)
    
    #Leiden clustering
    sc.tl.leiden(adata, resolution=1.0)
    
    #create dictionaries to map back each scaffold to cluster and UMAP coordinates
    scaffold_to_cluster = dict(zip(unique_scaffolds, adata.obs['leiden'].values.astype(int)))
    scaffold_umap = dict(zip(unique_scaffolds, adata.obsm['X_umap']))

    df['cluster'] = df['murcko_scaffold'].map(scaffold_to_cluster)
    # save and remove molecules not clustered (invalid fps)
    unclustered = df[df['cluster'].isna()]
    if not unclustered.empty:
        unclustered_path = output_dir / f"unclustered_{Path(input_file).stem}.csv"
        unclustered.to_csv(unclustered_path, index=False)
        logger.warning(f"Saved {len(unclustered)} molecules with invalid fps/clusters to {unclustered_path}")
    
    df = df[df['cluster'].notna()].copy()
    
    df['umap_1'] = df['murcko_scaffold'].map(lambda x: scaffold_umap.get(x, [np.nan, np.nan])[0])
    df['umap_2'] = df['murcko_scaffold'].map(lambda x: scaffold_umap.get(x, [np.nan, np.nan])[1])

   
    # Summary statistics per cluster
    summary = df.groupby('cluster').agg(
        total_molecules=('murcko_scaffold', 'size'),
        active_molecules=('activity', 'sum'),
        unique_scaffolds=('murcko_scaffold', 'nunique')
    ).sort_values('total_molecules', ascending=False)

    if logger:
        logger.info(f"Total clusters: {summary.shape[0]}")
        logger.info("Top clusters summary:\n" + summary.head(top_n_plot).to_string())
        logger.info("Bottom clusters summary:\n" + summary.tail(top_n_plot).to_string())

    # Plot histogram
    plt.figure(figsize=(14,6))
    ax = summary[['total_molecules', 'active_molecules', 'unique_scaffolds']].plot(
        kind='bar',
        figsize=(14,6),
        color=['lightgrey', 'red', 'blue'],
        width=0.8
    )
    plt.xlabel("Cluster ID")
    plt.ylabel("Count")
    plt.title("Per-cluster summary: molecules, actives, unique scaffolds")
    plt.legend(["Total molecules", "Active molecules", "Unique scaffolds"])
    plt.xticks(rotation=90)
    plt.tight_layout()

    summary_plot_path = output_dir / "combined_cluster_summary.png"
    plt.savefig(summary_plot_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    if logger:
        logger.info(f"Saved cluster summary plot to {summary_plot_path}")

    # Save clustered DataFrame and summary
    df_path = output_dir / "combined_clustered.csv"
    summary_path = output_dir / "combined_cluster_summary.csv"
    df.to_csv(df_path, index=False)
    summary.to_csv(summary_path, index=True)
    if logger:
        logger.info(f"Saved clustered DataFrame to {df_path}")
        logger.info(f"Saved cluster summary CSV to {summary_path}")


if __name__ == "__main__":
    tapify(cluster_scaffolds)