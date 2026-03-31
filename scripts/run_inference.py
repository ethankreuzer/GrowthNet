#!/usr/bin/env python
# coding: utf-8
"""
Script for preparing input data for GrowthCurve model inference (with fingerprint caching).
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import datamol as dm
from rdkit import Chem
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time
import json
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys
from torch.utils.data import Dataset, DataLoader
sys.path.append(str(Path("~/GrowthCurve/sweeps").expanduser()))
from sweep_multihead_lightning import MultiHeadLightning, batch_to_tensor
from typing import Any, Dict, Iterable, Optional, Tuple, List, Sequence
from torch.utils.data._utils.collate import default_collate

tqdm.pandas()

class ExplicitDataset(Dataset):
    """
    Test dataset that returns observed OD and label values
    without interpolation. One item = one (compound, t, c) row.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.max_time = 12.48
        self.num_fourier = 3

        # Collect fingerprint families
        self.fp_cols_by_family = sorted(
            [col for col in df.columns if col.endswith("_fp")]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        # Raw values
        t = float(row["Timepoint"])
        c = float(row["Concentration"])
        od = float(row["OD"])
        label = int(row["is_Active"])

        # Fourier time encoding
        t_enc = np.zeros(2 * self.num_fourier, dtype=np.float32)
        for j, k_freq in enumerate(range(1, self.num_fourier + 1)):
            angle = 2 * np.pi * k_freq * t / self.max_time
            t_enc[2 * j] = np.sin(angle)
            t_enc[2 * j + 1] = np.cos(angle)

        # Fingerprints
        features_by_family = {
            fam: torch.tensor(row[fam], dtype=torch.float32)
            for fam in self.fp_cols_by_family
        }

        return {
            "compound": row["Compound"],
            "smiles": row["SMILES"],
            "t_raw": torch.tensor(t, dtype=torch.float32),
            "t_fourier": torch.tensor(t_enc, dtype=torch.float32),
            "c_raw": torch.tensor(c, dtype=torch.float32),
            "c_log": torch.tensor(np.log(c), dtype=torch.float32),
            "y_reg": torch.tensor(od, dtype=torch.float32),
            "y_cls": torch.tensor(label, dtype=torch.float32),
            "features_by_family": features_by_family,
        }

def custom_collate(batch):
        """
        Custom collate function for PerCompoundDataset.

        Args:
            batch (list of dict): Each item is the output of __getitem__.

        Returns:
            dict: Batched output with stacked tensors and lists.
        """
        # Handle fingerprint features separately
        features_by_family = {}
        for fam in batch[0]['features_by_family'].keys():
            features_by_family[fam] = torch.stack(
                [item['features_by_family'][fam] for item in batch]
            )

        # Collate everything else using PyTorch’s default
        collated = {}
        for key in batch[0].keys():
            if key == 'features_by_family':
                continue
            collated[key] = default_collate([item[key] for item in batch])

        # Add fingerprints back
        collated['features_by_family'] = features_by_family
        return collated

# -------------------- Fingerprint functions --------------------
def maccs_to_fp(smile):
    try:
        return dm.to_fp(smile, fp_type="maccs")
    except Exception:
        return np.nan

def ecfp_to_fp(smile):
    try:
        return dm.to_fp(smile, fp_type="ecfp")
    except Exception:
        return np.nan

def rdkit_to_fp(smile):
    try:
        return dm.to_fp(smile, fp_type="rdkit")
    except Exception:
        return np.nan

def parallel_apply(series, func, n_jobs=None, desc="Processing"):
    """Apply a function in parallel with progress bar."""
    n_jobs = n_jobs or cpu_count()
    with Pool(n_jobs) as p:
        results = list(tqdm(p.imap(func, series), total=len(series), desc=desc))
    return results


# -------------------- Core logic --------------------
def load_csv(csv_path: str) -> pd.DataFrame:
    """Load CSV that may contain 'sep=,' artifact."""
    with open(csv_path, "r") as f:
        first_line = f.readline().strip()
    sep = "," if "sep=" in first_line else ","
    skiprows = 1 if "sep=" in first_line else 0
    df = pd.read_csv(csv_path, sep=sep, skiprows=skiprows)
    if "SMILES" not in df.columns:
        raise ValueError("❌ The CSV file must contain a column named 'SMILES'.")
    print(f"✅ Loaded {len(df)} compounds from {csv_path}")
    return df

def expand_features_by_conditions(df: pd.DataFrame, concentrations: List[float]) -> pd.DataFrame:
    """
    Expand the dataframe across all combinations of concentrations and hardcoded timepoints.
    Each compound (SMILES, CatalogId, fingerprints) is duplicated for each (Concentration, Timepoint).
    """
    df = df.copy()

    timepoints = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    expanded_rows = []

    for conc in concentrations:
        for t in timepoints:
            temp = df.copy()
            temp["Concentration"] = conc
            temp["Timepoint"] = t
            expanded_rows.append(temp)

    df_expanded = pd.concat(expanded_rows, ignore_index=True)
    print(f"✅ Expanded to {len(df_expanded)} rows ({len(concentrations)} concentrations × {len(timepoints)} timepoints × {len(df)} compounds)")
    return df_expanded

def add_dummy_features(df:pd.DataFrame) -> pd.DataFrame:
    "Add dummy features that are expected for the Explicit Dataset class to funciton"

    df=df.copy()

    df['OD'] = -1
    df['is_Active'] = -1

    return df


def main(args):
    start_time = time.time()

    csv_path = Path(args.csv_path)
    output_path = Path(args.output_path)
    cache_path = csv_path.with_name(csv_path.stem + "_fingerprints.pkl")  # cache file (pickle)

    # Step 1: if cached file exists, load and skip fingerprinting
    if cache_path.exists():
        print(f"⚡ Found cached fingerprint file → {cache_path}")
        start = time.time()
        df = pd.read_pickle(cache_path)
        elapsed = time.time() - start
        print(f"✅ Loaded cached DataFrame with {len(df)} samples")
        print(f"⏱️ Loaded Dataframe in {elapsed:.2f} seconds.")
    else:
        print(f"🧬 No cached file found. Computing fingerprints from scratch...")
        df = load_csv(csv_path)
        print(f"Shape of the inputted CSV: {df.shape}")
        

        n_jobs = min(128, cpu_count())
        print(f"🧠 Using {n_jobs} CPU cores for fingerprinting")

        df["maccs_fp"] = parallel_apply(df["SMILES"], maccs_to_fp, n_jobs, desc="MACCS")
        df["ecfp_fp"] = parallel_apply(df["SMILES"], ecfp_to_fp,  n_jobs, desc="ECFP")
        df["rdkit_fp"] = parallel_apply(df["SMILES"], rdkit_to_fp, n_jobs, desc="RDKit")

        print(f"Shape of the inputted CSV after adding fp features: {df.shape}")

        # Save fingerprint cache
        df.to_pickle(cache_path)
        print(f"💾 Saved fingerprint DataFrame cache → {cache_path}")
    
    print(df.shape)
    
    df = expand_features_by_conditions(df, args.concentration)
    df = add_dummy_features(df)
    print(df.shape)

    ###LOAD the model for inference

    run_id = args.run_id   # replace with your run ID
    save_dir = Path(f"/home/ethan2/GrowthNet/models/final_sweep/checkpoints/{run_id}")
    # Load hyperparameters
    with open(save_dir / "hparams.json") as f:
        hparams = json.load(f)

    ##Need to turn the df into dicitonary here

    print("\n🚀 Starting GPU-accelerated inference...")

    # 1️⃣ Prepare dataset & dataloader
    batch_size = 8192 if torch.cuda.is_available() else 2048

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device being used: {device}")

    ds_test = ExplicitDataset(df)
    ds_loader = DataLoader(
        ds_test,
        batch_size=batch_size,
        collate_fn=custom_collate,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    # 2️⃣ Load trained model
    run_id = args.run_id
    save_dir = Path(f"/home/ethan2/GrowthNet/models/final_sweep/checkpoints/{run_id}")

    with open(save_dir / "hparams.json") as f:
        hparams = json.load(f)

    model = MultiHeadLightning.load_from_checkpoint(
        save_dir / "best_params.ckpt",
        input_dim=4271,
        config=hparams
    ).to(device)
    model.eval()

    # Optionally enable mixed precision for speed
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.autocast if use_amp else torch.no_grad

    # 3️⃣ Inference loop
    all_preds_reg, all_preds_cls = [], []
    all_meta = []

    start = time.time()
    with torch.no_grad():
        for batch in tqdm(ds_loader, desc="Running inference", total=len(ds_loader)):
            # Non-blocking data transfer
            Xte, _, _ = batch_to_tensor(batch, device)

            # Use mixed precision if available
            with torch.cuda.amp.autocast(enabled=use_amp):
                y_reg_pred, y_cls_pred = model(Xte.to(device, non_blocking=True))

            # Move predictions to CPU asynchronously
            y_reg = y_reg_pred.detach().cpu().numpy().ravel()
            y_cls = torch.sigmoid(y_cls_pred.detach()).cpu().numpy().ravel()

            # Accumulate predictions and metadata
            all_preds_reg.append(y_reg)
            all_preds_cls.append(y_cls)
            all_meta.append(pd.DataFrame({
                "Compound": batch["compound"],
                "SMILES": batch["smiles"],
                "Concentration": batch["c_raw"].cpu().numpy().ravel(),
                "Timepoint": batch["t_raw"].cpu().numpy().ravel(),
            }))

    end = time.time()
    print(f"\n⏱️ Inference completed in {(end - start)/60:.2f} minutes.")

    # 4️⃣ Merge predictions
    df_out = pd.concat(all_meta, ignore_index=True)
    df_out["Pred_OD"] = np.concatenate(all_preds_reg)
    df_out["Pred_ProbActive"] = np.concatenate(all_preds_cls)

    # 5️⃣ Save predictions
    output_pred_path = output_path.with_name(output_path.stem + "_predictions.csv")
    df_out.to_csv(output_pred_path, index=False)
    print(f"💾 Saved predictions → {output_pred_path}")

    #May need to hard code the max timepoint in this class to the same model was trained on
    

    #Xte, _, _ = batch_to_tensor(ds_dict, torch.device("cpu"))

    print(df_out.head(10))
    #print(Xte.shape)
    
    #with torch.no_grad():
    #    y_reg_pred, y_cls_pred = model(Xte.to("cuda"))


    

    


# -------------------- CLI --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for GrowthCurve model inference (cached fingerprints)")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--run_id", type=str, required=True, help="Model run ID (used later for inference)")
    parser.add_argument(
        "--concentration",
        type=float,
        nargs="+",
        required=True,
        help="List of concentration values, e.g. --concentrations 1.2 7.9"
    )
    parser.add_argument("--output_path", type=str, required=True, help="Path to output CSV file")
    args = parser.parse_args()
    main(args)
