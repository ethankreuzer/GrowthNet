#!/usr/bin/env python
"""
Build the unified data artifact and SMILES split files for a given split.

Outputs (under data/splits/<split_name>/):
  - all_compound_metas.pkl   list[CompoundMeta] for ALL compounds (train + val)
  - train_smiles.txt         canonical SMILES for train compounds (one per line)
  - val_smiles.txt           canonical SMILES for val compounds (one per line)

Usage:
  python build_training_data.py [--split Celine_v1]
"""

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch.utils.data import DataLoader

PROJ = Path("/home/ethan2/GrowthNet")
sys.path.insert(0, str(PROJ / "sweeps"))

from data_class import CompoundMeta, ExplicitDataset, custom_collate

TRAIN_DF_PATH = PROJ / "data/train/df_well_train_Celine_clusters_mad_4.pkl"
VAL_DF_PATH   = PROJ / "data/validation/df_well_validation_Celine_clusters_mad_4.pkl"
REP_DICT_PATH = PROJ / "data/smiles_representations.pkl"

NUM_FOURIER = 3


def canonicalize(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"RDKit cannot parse SMILES: {smi}")
    return Chem.MolToSmiles(mol)


def build_recanon_lookup(rep_dict: dict) -> dict:
    """Map current-RDKit canonical SMILES → rep_dict key (handles version drift)."""
    lookup = {}
    for old_key in rep_dict:
        new_key = canonicalize(old_key)
        if new_key is not None:
            lookup[new_key] = old_key
    return lookup


def lookup_rep(canon_smi: str, rep_dict: dict, recanon: dict) -> Dict:
    """Look up fingerprints; fall back to re-canonicalized key on version drift."""
    if canon_smi in rep_dict:
        return rep_dict[canon_smi]
    old_key = recanon.get(canon_smi)
    if old_key is not None:
        return rep_dict[old_key]
    raise KeyError(f"SMILES not in rep_dict (tried direct + recanon): {canon_smi!r}")


def build_compound_metas(df: pd.DataFrame, rep_dict: dict, recanon: dict, label: str) -> list:
    """Build CompoundMeta objects from a DataFrame (train or val)."""
    metas = []
    grouped = df.groupby("Compound", sort=True)
    total = len(grouped)

    for i, (comp, sub) in enumerate(grouped):
        if (i + 1) % 500 == 0 or i == 0:
            print(f"  [{label}] CompoundMeta {i + 1}/{total}...")

        sub = sub.sort_values(["Timepoint", "Concentration"])

        piv_od = (
            sub.pivot(index="Timepoint", columns="Concentration", values="OD")
            .sort_index(axis=0)
            .sort_index(axis=1)
        )
        piv_cls = (
            sub.pivot(index="Timepoint", columns="Concentration", values="is_Active")
            .sort_index(axis=0)
            .sort_index(axis=1)
        )

        t_vals = piv_od.index.values.astype(float)
        c_vals = piv_od.columns.values.astype(float)

        raw_smiles = str(sub["Smiles"].iloc[0])
        canon_smi = canonicalize(raw_smiles)

        rep = lookup_rep(canon_smi, rep_dict, recanon)
        fps_by_family: Dict[str, np.ndarray] = {
            fam: vec.astype(np.float32) for fam, vec in sorted(rep.items())
        }

        single_conc = c_vals.size == 1

        is_active_at_12_50 = False
        try:
            if 12.48 in piv_cls.index and 50.0 in piv_cls.columns:
                val = piv_cls.loc[12.48, 50.0]
                is_active_at_12_50 = bool(val == 1)
        except Exception:
            is_active_at_12_50 = False

        meta = CompoundMeta(
            compound=comp,
            smiles=canon_smi,
            pivot_od=piv_od,
            pivot_cls=piv_cls,
            t_vals=t_vals,
            c_vals=c_vals,
            single_conc=single_conc,
            t_min=float(t_vals.min()),
            t_max=float(t_vals.max()),
            logc_min=float(np.log(c_vals.min())),
            logc_max=float(np.log(c_vals.max())),
            fps_by_family=fps_by_family,
            is_active_at_12_50=is_active_at_12_50,
        )
        metas.append(meta)

    return metas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="Celine_v1",
                        help="Name of the split directory under data/splits/")
    args = parser.parse_args()

    out_dir = PROJ / "data" / "splits" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading DataFrames...")
    df_train = pd.read_pickle(TRAIN_DF_PATH)
    df_val   = pd.read_pickle(VAL_DF_PATH)
    print(f"  Train rows: {len(df_train):,}  Val rows: {len(df_val):,}")

    print(f"Loading representations from {REP_DICT_PATH}...")
    with open(REP_DICT_PATH, "rb") as f:
        rep_dict = pickle.load(f)
    print(f"  {len(rep_dict):,} entries loaded.")

    print("  Building re-canonicalization lookup (handles RDKit version drift)...")
    recanon = build_recanon_lookup(rep_dict)

    # ── Train CompoundMetas ─────────────────────────────────────
    print("\nBuilding train CompoundMetas...")
    train_metas = build_compound_metas(df_train, rep_dict, recanon, label="train")
    train_smiles = [m.smiles for m in train_metas]
    print(f"  {len(train_metas)} train compounds.")

    # ── Val CompoundMetas ───────────────────────────────────────
    print("\nBuilding val CompoundMetas...")
    val_metas = build_compound_metas(df_val, rep_dict, recanon, label="val")
    val_smiles = [m.smiles for m in val_metas]
    print(f"  {len(val_metas)} val compounds.")

    # ── Sanity: no overlap ──────────────────────────────────────
    train_set = set(train_smiles)
    val_set   = set(val_smiles)
    overlap = train_set & val_set
    if overlap:
        raise ValueError(
            f"Train/val SMILES overlap detected ({len(overlap)} compounds)!\n"
            f"First 5: {list(overlap)[:5]}"
        )
    print(f"\nSanity check passed: train and val SMILES are disjoint.")

    # ── Write SMILES lists ──────────────────────────────────────
    train_smiles_path = out_dir / "train_smiles.txt"
    val_smiles_path   = out_dir / "val_smiles.txt"

    with open(train_smiles_path, "w") as f:
        f.write("\n".join(train_smiles))
    print(f"  Saved {len(train_smiles)} train SMILES → {train_smiles_path}")

    with open(val_smiles_path, "w") as f:
        f.write("\n".join(val_smiles))
    print(f"  Saved {len(val_smiles)} val SMILES → {val_smiles_path}")

    # ── Unified pickle ──────────────────────────────────────────
    all_metas = train_metas + val_metas
    out_pkl = out_dir / "all_compound_metas.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(all_metas, f)
    print(f"\n  Saved {len(all_metas)} CompoundMeta objects → {out_pkl}")

    # ── Summary ─────────────────────────────────────────────────
    sample = train_metas[0]
    fam_dims = {k: v.shape[0] for k, v in sample.fps_by_family.items()}
    print(f"\nFeature families: {fam_dims}")
    print(f"Total fp dim: {sum(fam_dims.values())}")
    print(f"Total input dim (fourier+conc+fps): {sum(fam_dims.values()) + 2 * NUM_FOURIER + 2}")
    print("\nDone.")


if __name__ == "__main__":
    main()
