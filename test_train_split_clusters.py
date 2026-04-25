"""
Split a dataset of molecules clustered by Murcko scaffolds into training and test sets.

The test set is selected such that:
- Molecules in the test set do not belong to the same scaffold clusters as the training set (to avoid data leakage).
- For each group defined (e.g. author + pathogen_id), constraints on the number of molecules, unique scaffolds, and active molecules are satisfied.
- The number of clusters selected for the test set is controlled via hard constraints (min/max fraction of clusters).

This script uses a Monte Carlo approach to find a valid train/test split: it repeatedly samples random subsets of scaffold clusters until a valid test set is found or the maximum number of iterations is reached.

Inputs:
- A clustered dataframe, containing at least the following columns: ['author', 'pathogen', 'strain', 'activity', 'murcko_scaffold', 'cluster']
- Hard constraints (e.g., fraction of clusters to select for test)
- Soft constraints per author/pathogen_id (min/max molecules, scaffolds, actives)

Outputs:
- List of clusters selected for test set
"""
import sys
import pandas as pd
import numpy as np
from tqdm import trange
from pathlib import Path
from tap import tap, tapify
from datetime import datetime

# Add root repo to sys.path to import utils modules 
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.logging import get_logger
from utils.data import get_scaffold, generate_fingerprints

# Initialize logger
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = Path(f"data/logs/cluster_test_split_{timestamp}.log")
logger = get_logger(log_file, name="sample_test_set")
logger.info("Starting cluster test split script...")


def create_pathogen_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create pathogen_id from pathogen + strain and merge back to df. 
    
    This function generates a 'pathogen_id' column of the form
    pathogen_n, where n is the strain number for that pathogen.
    This helps simplify outputs and summaries (full strain names
    may be very long /difficult to interpret).
    """
    mapping = (
        df[["strain", "pathogen"]]
        .drop_duplicates()
        .sort_values(["pathogen", "strain"], ascending=[True, True])
        .reset_index(drop=True)
    )
    mapping["strain_idx"] = mapping.groupby("pathogen").cumcount() + 1
    mapping["pathogen_id"] = mapping["pathogen"] + "_" + mapping["strain_idx"].astype(str)
    df = df.merge(mapping[["pathogen", "strain", "pathogen_id"]],
                  on=["pathogen", "strain"],
                  how="left")
    return df

def prepare_constraints(
    df: pd.DataFrame,
    group_cols: list[str] = ["author", "pathogen_id"],
    min_frac: float = 0.10,
    max_frac: float = 0.30,
    min_dataset_size: int = 2000,
    min_dataset_flex: float = 0.5 
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    Build soft and hard constraints for splitting clusters. 
    
    This function uses the same range of values - based on min and max % parameters - for all group columns. 
    For smaller datasets - defined by min_dataset_size - constraints are relaxed using an additional flexibility parameter.  

    Parameters
    ----------
    df : pd.DataFrame
        Must include group_cols, 'murcko_scaffold', 'activity'
    group_cols : list[str]
        Columns to group by (default ['author', 'pathogen_id'])
    min_frac : float
        Minimum fraction of clusters to select
    max_frac : float
        Maximum fraction of clusters to select
    min_dataset_size : int
        Datasets smaller than this size will get relaxed constraints
    min_dataset_flex : float
        additional flexibility applied to soft constraints for smaller datasets (e.g. 0.5 applies to min_frac (-0.5) and max_frac (+0.5))

    Returns
    -------
    soft_constraints : pd.DataFrame
    hard_constraints : dict
    summary_table : pd.DataFrame
    """
    summary = (
    df.groupby(group_cols)
        .agg(
            n_scaffolds=("murcko_scaffold", "nunique"),
            n_molecules=("murcko_scaffold", "size"),
            n_actives=("activity", lambda x: (x == 1).sum()),
        )
    )

    soft_constraints = pd.DataFrame(index=summary.index)
    soft_constraints["min_scaffolds"] = (summary["n_scaffolds"] * min_frac).astype(int)
    soft_constraints["max_scaffolds"] = (summary["n_scaffolds"] * max_frac).astype(int)
    soft_constraints["min_molecules"] = (summary["n_molecules"] * min_frac).astype(int)
    soft_constraints["max_molecules"] = (summary["n_molecules"] * max_frac).astype(int)
    soft_constraints["min_actives"] = (summary["n_actives"] * min_frac).astype(int)
    soft_constraints["max_actives"] = (summary["n_actives"] * max_frac).astype(int)

    # Relax constraints for small datasets
    small_mask = summary["n_molecules"] < min_dataset_size
    for col in ["scaffolds", "molecules", "actives"]:
        current_min = soft_constraints.loc[small_mask, f"min_{col}"]
        current_max = soft_constraints.loc[small_mask, f"max_{col}"]
        n_total = summary.loc[small_mask, f"n_{col}"]

        soft_constraints.loc[small_mask, f"min_{col}"] = (
            (current_min *(1-min_dataset_flex)).clip(lower=0).astype(int)
        )
        soft_constraints.loc[small_mask, f"max_{col}"] = (
            (current_max *(1+ min_dataset_flex)).clip(upper=n_total).astype(int)
        )
        

    hard_constraints = {"min_cluster_frac": min_frac, "max_cluster_frac": max_frac}

    return soft_constraints, hard_constraints



def sample_test_clusters(
    df,
    soft_constraints,
    hard_constraints,
    mean_frac,
    std_frac,
    max_iter=1000,
    random_seed=42,
    logger=None,
    log_every=100,
):
    """
    Monte Carlo sampling of test clusters.

    Parameters
    ----------
    df : pd.DataFrame
        Must include ['cluster', 'murcko_scaffold', 'activity'] + group columns used in soft_constraints
    soft_constraints : pd.DataFrame
        Indexed by group columns (e.g., author/pathogen_id) with min/max columns
        ['min_scaffolds','max_scaffolds','min_molecules','max_molecules','min_actives','max_actives']
    hard_constraints : dict
        Keys: 'min_cluster_frac', 'max_cluster_frac'
    mean_frac : mean fraction of clusters sampled from normal distribution
    std_frac : std dev of the fraction of clusters sampled
    max_iter : int
        Maximum number of Monte Carlo iterations
    random_seed : int
        For reproducibility
    logger : logging.Logger or None
        Logger for info messages. If None, no logging occurs.
    log_every : int
        How often to log in iterations

    Returns
    -------
    best_selection : list
        List of cluster IDs selected for test set
    """

    rng = np.random.default_rng(random_seed)
    all_clusters = df["cluster"].unique()
    n_clusters = len(all_clusters)
    logger.info(f"Found {n_clusters} distinct clusters, from {all_clusters.min()} to {all_clusters.max()}")

    min_pick = int(np.floor(hard_constraints["min_cluster_frac"] * n_clusters))
    max_pick = int(np.ceil(hard_constraints["max_cluster_frac"] * n_clusters))

    # Precompute group columns for indexing
    group_cols = soft_constraints.index.names
    total_constraints = len(soft_constraints)

    best_constraints_met = -1
    best_selection = None

    # Precompute summaries at group_columns + cluster level
    cluster_summary = (
        df.groupby(list(group_cols) + ["cluster"])
          .agg(
              n_scaffolds=("murcko_scaffold", "nunique"),
              n_molecules=("murcko_scaffold", "size"),
              n_actives=("activity", "sum"),
          )
          .reset_index()
    )

    for i in range(max_iter):
        # randomly select the number of clusters in test set : n_pick using hard constraints
        # sample from normal until it's in [min_frac, max_frac]
        min_frac = hard_constraints["min_cluster_frac"]
        max_frac = hard_constraints["max_cluster_frac"]
        while True:
            frac = rng.normal(loc=mean_frac, scale=std_frac)
            if min_frac <= frac <= max_frac:
                break

        # convert to integer number of clusters
        n_pick = int(np.round(frac * n_clusters))
        
        # randomly select n_pick cluster indices
        indices = rng.choice(len(all_clusters), size=n_pick, replace=False)
        chosen_clusters = all_clusters[indices]
      
        # Subset precomputed summary based on selected cluster indices
        chosen = cluster_summary[cluster_summary["cluster"].isin(chosen_clusters)]

        # sum summary columns over clusters
        test_summary = chosen.groupby(group_cols).agg(
            n_scaffolds=("n_scaffolds", "sum"),
            n_molecules=("n_molecules", "sum"),
            n_actives=("n_actives", "sum"),
        )

        # check constraints
        merged = soft_constraints.join(test_summary, how="left", rsuffix="_test").fillna(0)

        scaffolds_ok = (merged["n_scaffolds"] >= merged["min_scaffolds"]) & (merged["n_scaffolds"] <= merged["max_scaffolds"])
        molecules_ok = (merged["n_molecules"] >= merged["min_molecules"]) & (merged["n_molecules"] <= merged["max_molecules"])
        actives_ok = (merged["n_actives"] >= merged["min_actives"]) & (merged["n_actives"] <= merged["max_actives"])

        constraints_met = (scaffolds_ok & molecules_ok & actives_ok).sum()
        valid = constraints_met == total_constraints

        # Update best selection
        if constraints_met > best_constraints_met:
            best_constraints_met = constraints_met
            best_selection = chosen_clusters.tolist()
            if logger:
                logger.info(f"New best selection at iteration {i+1}: {constraints_met}/{total_constraints} constraints met")

        # Logging
        if logger and (i % log_every == 0):
            logger.info(f"Iteration {i+1}/{max_iter} | curr_met: {constraints_met}/{total_constraints} | best_met: {best_constraints_met}/{total_constraints}")

        if valid:
            if logger:
                logger.info(f"Valid selection found at iteration {i+1}")
                logger.info(f"Number of clusters selected : {n_pick}, out of {n_clusters} clusters.")
            return chosen_clusters.tolist()

    if logger:
        logger.info(f"No fully valid selection found. Best constraints met: {best_constraints_met}/{total_constraints}")
        logger.info(f"Number of clusters selected : {n_pick}, out of {n_clusters} clusters.")
    return best_selection


def main(
    clustered_file: str,
    output_dir: str = "data/processed",
    soft_constraints: pd.DataFrame = None,
    hard_constraints: dict = None,
    group_cols: list[str] = ["author", "pathogen_id"],
    min_frac: float = 0.10,
    max_frac: float = 0.30,
    min_dataset_size: int=2000,
    min_dataset_flex: float = 5.0,
    mean_frac: float = 0.2,
    std_frac: float = 0.05,
    max_iter: int = 5000,
    random_seed: int = 42,
    logger=logger
):
    """
    Split a clustered dataset into train/test sets based on scaffold clusters.

    If soft_constraints and hard_constraints are provided, they are used directly.
    Otherwise, they are computed from the dataset using prepare_constraints() and the min_frac, max_frac, group_cols parameters

    Parameters
    ----------
    clustered_file : str
        Path to the clustered CSV file (must include 'cluster', 'murcko_scaffold', 'activity').
    output_dir : str
        Directory to save train/test CSVs.
    soft_constraints : pd.DataFrame, optional
        Indexed by group columns with min/max constraints for scaffolds/molecules/actives.
    hard_constraints : dict, optional
        Keys: 'min_cluster_frac', 'max_cluster_frac'.
    group_cols : list of str
        Columns to group by when computing constraints (e.g., ['author', 'pathogen_id']).
    min_frac : float
        Minimum fraction of clusters to select for test set if computing constraints.
    max_frac : float
        Maximum fraction of clusters to select for test set if computing constraints.
    min_dataset_size: int
        minimum size of the dataset to apply soft constraints
    min_dataset_flex: float
        additional flexibility applied to solft constraints for smaller datasets
    mean_frac : mean fraction of clusters sampled from normal distribution
    std_frac : std dev of the fraction of clusters sampled
    max_iter : int
        Maximum number of Monte Carlo iterations to try.
    random_seed : int
        For reproducibility.
    logger : logging.Logger, optional
        Logger for info messages. If None, a default logger will be created.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if logger is None:
        log_file = output_dir / "cluster_test_split.log"
        logger = get_logger(log_file, name="sample_test_set")

    logger.info(f"Loading clustered file: {clustered_file}")
    df = pd.read_csv(clustered_file)

    # Create pathogen_id column if not present in the current df
    if "pathogen_id" not in df.columns:
        logger.info("Creating pathogen_id column...")
        df = create_pathogen_id(df)

    # Prepare constraints if they are not provided
    if soft_constraints is None or hard_constraints is None:
        logger.info(f"Preparing constraints from dataset with group_cols={group_cols}, min_frac={min_frac}, max_frac={max_frac}...")
        soft_constraints, hard_constraints = prepare_constraints(
            df,
            group_cols=group_cols,
            min_frac=min_frac,
            max_frac=max_frac,
            min_dataset_size = min_dataset_size,
            min_dataset_flex = min_dataset_flex
        )
    else:
        logger.info("Using provided soft and hard constraints.")
        
    # run the Monte Carlo sampling algo to select clusters included in the test set
    logger.info(f"Sampling clusters with max_iter={max_iter}...")
    test_clusters = sample_test_clusters(
        df,
        soft_constraints=soft_constraints,
        hard_constraints=hard_constraints,
        mean_frac = mean_frac,
        std_frac = std_frac,
        max_iter=max_iter,
        random_seed=random_seed,
        logger=logger
    )

     # Split datasets based on the selected cluster ids and save files
    test_df = df[df["cluster"].isin(test_clusters)].copy()
    train_df = df[~df["cluster"].isin(test_clusters)].copy()

    test_file = output_dir / f"{Path(clustered_file).stem}_test.csv"
    train_file = output_dir / f"{Path(clustered_file).stem}_train.csv"

    test_df.to_csv(test_file, index=False)
    train_df.to_csv(train_file, index=False)

    logger.info(f"Saved test set ({len(test_df)} molecules) to {test_file}")
    logger.info(f"Saved train set ({len(train_df)} molecules) to {train_file}")

    
    
    # Summary of molecules/scaff/n_actives in best selected test set per group
    
    group_summary = test_df.groupby(group_cols).agg(
        n_molecules=("smile_canonical", "size"),
        n_scaffolds=("murcko_scaffold", "nunique"),
        n_actives=("activity", lambda x: (x == 1).sum())
    )

    # Compute % of total per group
    total_per_group = df.groupby(group_cols).agg(
        total_molecules=("smile_canonical", "size"),
        total_scaffolds=("murcko_scaffold", "nunique"),
        total_actives=("activity", lambda x: (x == 1).sum())
    )
    # join summaries
    summary_with_pct = group_summary.join(total_per_group)
    summary_with_pct["% test mol"] = (summary_with_pct["n_molecules"] / summary_with_pct["total_molecules"] * 100).round(1)
    summary_with_pct["% test scaf"] = (summary_with_pct["n_scaffolds"] / summary_with_pct["total_scaffolds"] * 100).round(1)
    summary_with_pct["% test active"] = (summary_with_pct["n_actives"] / summary_with_pct["total_actives"] * 100).round(1)

    logger.info("Best Test set summary by group:")
    logger.info(summary_with_pct.to_string())



if __name__ == "__main__":
    tapify(main)