#!/usr/bin/env python
"""
Rebuild CompoundMetas and validation dicts with canonical SMILES and all
feature families from smiles_representations.pkl.

Outputs:
  - data/train/Celine_CompoundMetas_list.pkl   (list of CompoundMeta)
  - data/test/dict_val_fourier_k_3_Celine.pkl  (full val dict)
  - data/test/dict_val_fourier_k_3_conc_*_Celine.pkl  (7 per-concentration val dicts)

Usage:
  python build_training_data.py
"""

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
VAL_DF_PATH = PROJ / "data/validation/df_well_validation_Celine_clusters_mad_4.pkl"
REP_DICT_PATH = PROJ / "data/smiles_representations.pkl"

METAS_OUT = PROJ / "data/train/Celine_CompoundMetas_list.pkl"
VAL_OUT_DIR = PROJ / "data/test"

NUM_FOURIER = 3


def canonicalize(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"RDKit cannot parse SMILES: {smi}")
    return Chem.MolToSmiles(mol)


def replace_features(val_dict: dict, rep_dict: dict) -> dict:
    """Replace features_by_family in a collated val dict with rep_dict lookups."""
    new_features: Dict[str, list] = {}
    for smi in val_dict["smiles"]:
        canon = canonicalize(smi)
        rep = rep_dict[canon]
        for fam, vec in sorted(rep.items()):
            new_features.setdefault(fam, []).append(
                torch.from_numpy(vec.astype(np.float32))
            )
    val_dict["features_by_family"] = {
        fam: torch.stack(vecs) for fam, vecs in sorted(new_features.items())
    }
    return val_dict


def build_compound_metas(df_train: pd.DataFrame, rep_dict: dict) -> list:
    """Build a list of CompoundMeta objects with canonical SMILES and full features."""
    metas = []
    grouped = df_train.groupby("Compound", sort=True)
    total = len(grouped)

    for i, (comp, sub) in enumerate(grouped):
        if (i + 1) % 5000 == 0 or i == 0:
            print(f"  CompoundMeta {i + 1}/{total}...")

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

        rep = rep_dict[canon_smi]
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


def build_val_dict(df: pd.DataFrame, rep_dict: dict) -> dict:
    """Build a single collated val dict and replace features from rep_dict."""
    ds = ExplicitDataset(df, num_fourier=NUM_FOURIER)
    loader = DataLoader(ds, batch_size=len(ds), collate_fn=custom_collate)
    val_dict = next(iter(loader))
    return replace_features(val_dict, rep_dict)


def main():
    print("Loading DataFrames...")
    df_train = pd.read_pickle(TRAIN_DF_PATH)
    df_val = pd.read_pickle(VAL_DF_PATH)

    print(f"Loading representations from {REP_DICT_PATH}...")
    with open(REP_DICT_PATH, "rb") as f:
        rep_dict = pickle.load(f)
    print(f"  {len(rep_dict)} entries loaded.")

    # ── Build CompoundMetas ──────────────────────────────────────────
    print("\nBuilding CompoundMetas...")
    metas = build_compound_metas(df_train, rep_dict)
    print(f"  {len(metas)} CompoundMeta objects created.")

    METAS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(METAS_OUT, "wb") as f:
        pickle.dump(metas, f)
    print(f"  Saved to {METAS_OUT}")

    # ── Build full validation dict ───────────────────────────────────
    print("\nBuilding full validation dict...")
    val_main = build_val_dict(df_val, rep_dict)
    out_path = VAL_OUT_DIR / "dict_val_fourier_k_3_Celine.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(val_main, f)
    print(f"  Saved to {out_path}  ({len(val_main['smiles'])} rows)")

    # ── Build per-concentration validation dicts ─────────────────────
    conc_specs = [
        (0.2, "0_2"),
        (0.781, "0_781"),
        (1.2, "1_2"),
        (3.13, "3_13"),
        (7.9, "7_9"),
        (12.5, "12_50"),
        (50, "50"),
    ]

    for conc, text in conc_specs:
        print(f"  Building val dict for conc={conc}...")
        df_subset = df_val[
            (df_val["Concentration"] == conc) & (df_val["Timepoint"] != 0)
        ].reset_index(drop=True)
        if len(df_subset) == 0:
            print(f"    WARNING: no rows for concentration {conc}, skipping.")
            continue
        val_conc = build_val_dict(df_subset, rep_dict)
        out_path = VAL_OUT_DIR / f"dict_val_fourier_k_3_conc_{text}_Celine.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(val_conc, f)
        print(f"    Saved ({len(val_conc['smiles'])} rows)")

    # ── Summary ──────────────────────────────────────────────────────
    sample = metas[0]
    fam_dims = {k: v.shape[0] for k, v in sample.fps_by_family.items()}
    total_fp = sum(fam_dims.values())
    print(f"\nFeature families per compound: {fam_dims}")
    print(f"Total fingerprint dim: {total_fp}")
    print(f"Total input dim (with fourier+conc): {total_fp + 2 * NUM_FOURIER + 2}")
    print("\nDone.")


if __name__ == "__main__":
    main()
