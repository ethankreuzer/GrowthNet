#!/usr/bin/env python
"""
Combine labeled DataFrames from normalization notebooks, compute molecular
fingerprints for any SMILES not already cached, build CompoundMeta objects,
and save the list as a pickle file.

Configure the three path constants at the top of main() before running:
  DF_PATHS      — list of .pkl paths for the labeled DataFrames
  REP_DICT_PATH — path to smiles_representations.pkl (used as cache)
  OUTPUT_PATH   — where to write the CompoundMeta list pickle

Usage:
  cd /home/ethan2/GrowthNet
  .venv/bin/python scripts/build_compound_metas.py
"""

import pickle
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

PROJ = Path("/home/ethan2/GrowthNet")
sys.path.insert(0, str(PROJ / "sweeps"))

from data_class import CompoundMeta


# ---------------------------------------------------------------------------
# Fingerprint / SMILES utilities (adapted from build_training_data.py and
# build_representation_dict.py — kept inline so this script is self-contained)
# ---------------------------------------------------------------------------

def canonicalize(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"RDKit cannot parse SMILES: {smi}")
    return Chem.MolToSmiles(mol)


def get_murko_scaffold(smi: str) -> str:
    """Compute Murko scaffold (core ring system) from SMILES."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return smi
        scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold_mol)
    except Exception:
        return smi


def build_recanon_lookup(rep_dict: dict) -> dict:
    """Map current-RDKit canonical SMILES → rep_dict key (handles version drift)."""
    lookup = {}
    for old_key in rep_dict:
        try:
            new_key = canonicalize(old_key)
            lookup[new_key] = old_key
        except ValueError:
            pass
    return lookup


def lookup_rep(canon_smi: str, rep_dict: dict, recanon: dict) -> dict:
    if canon_smi in rep_dict:
        return rep_dict[canon_smi]
    old_key = recanon.get(canon_smi)
    if old_key is not None:
        return rep_dict[old_key]
    raise KeyError(f"SMILES not in rep_dict (tried direct + recanon): {canon_smi!r}")


def compute_fingerprints(smi: str) -> dict:
    import datamol as dm
    result = {}
    for fp_type in ("maccs", "ecfp", "rdkit"):
        try:
            result[f"{fp_type}_fp"] = dm.to_fp(smi, fp_type=fp_type)
        except Exception:
            result[f"{fp_type}_fp"] = None
    return result


def compute_minimol_fps(smiles_list: List[str], batch_size: int = 512) -> Dict[str, Optional[np.ndarray]]:
    try:
        from minimol import Minimol
    except ImportError:
        warnings.warn("minimol not importable — minimol_fp will be None for all new SMILES.")
        return {smi: None for smi in smiles_list}

    model = Minimol()
    result = {}
    failed = []

    for start in range(0, len(smiles_list), batch_size):
        batch = smiles_list[start:start + batch_size]
        if start % (batch_size * 10) == 0:
            print(f"  MiniMol: {start}–{min(start+batch_size, len(smiles_list))} / {len(smiles_list)}...")
        try:
            embeddings = model(batch)
            for smi, emb in zip(batch, embeddings):
                result[smi] = emb.detach().cpu().numpy()
        except Exception:
            for smi in batch:
                try:
                    emb = model([smi])[0]
                    result[smi] = emb.detach().cpu().numpy()
                except Exception as e:
                    result[smi] = None
                    failed.append((smi, str(e)))

    if failed:
        print(f"  WARNING: {len(failed)} MiniMol failures.")
    return result


# ---------------------------------------------------------------------------
# Core pipeline steps
# ---------------------------------------------------------------------------

def ensure_representations(smiles_list: List[str], rep_dict: dict, recanon: dict) -> dict:
    """Compute fingerprints for any canonical SMILES not already in rep_dict."""
    misses = []
    for smi in smiles_list:
        if smi not in rep_dict and recanon.get(smi) not in rep_dict:
            misses.append(smi)

    print(f"  {len(smiles_list) - len(misses)} SMILES found in cache, {len(misses)} need computation.")

    if not misses:
        return rep_dict

    print("  Computing MACCS / ECFP / RDKit fingerprints for cache misses...")
    for i, smi in enumerate(misses):
        if (i + 1) % 500 == 0 or i == 0:
            print(f"    {i + 1}/{len(misses)}...")
        fps = compute_fingerprints(smi)
        rep_dict[smi] = {
            "maccs_fp": fps.get("maccs_fp"),
            "ecfp_fp": fps.get("ecfp_fp"),
            "rdkit_fp": fps.get("rdkit_fp"),
            "boltz2_rep": None,
            "minimol_fp": None,
        }

    print("  Computing MiniMol fingerprints for cache misses...")
    minimol_map = compute_minimol_fps(misses)
    for smi, emb in minimol_map.items():
        rep_dict[smi]["minimol_fp"] = emb

    return rep_dict


def build_metas(df: pd.DataFrame, rep_dict: dict, recanon: dict) -> List[CompoundMeta]:
    metas = []
    grouped = df.groupby("Compound", sort=True)
    total = len(grouped)
    skipped = []

    for i, (comp, sub) in enumerate(grouped):
        if (i + 1) % 500 == 0 or i == 0:
            print(f"  CompoundMeta {i + 1}/{total}...")

        # Aggregate any residual plate-level duplicates before pivoting
        sub_agg = (
            sub.groupby(["Timepoint", "Concentration"], as_index=False)
            .agg({"OD": "mean", "is_Active": "max", "Smiles": "first"})
        )
        sub_agg = sub_agg.sort_values(["Timepoint", "Concentration"])

        piv_od = (
            sub_agg.pivot(index="Timepoint", columns="Concentration", values="OD")
            .sort_index(axis=0).sort_index(axis=1)
        )
        piv_cls = (
            sub_agg.pivot(index="Timepoint", columns="Concentration", values="is_Active")
            .sort_index(axis=0).sort_index(axis=1)
        )

        t_vals = piv_od.index.values.astype(float)
        c_vals = piv_od.columns.values.astype(float)

        raw_smiles = str(sub_agg["Smiles"].iloc[0])
        try:
            canon_smi = canonicalize(raw_smiles)
        except ValueError as e:
            skipped.append((comp, str(e)))
            continue

        try:
            rep = lookup_rep(canon_smi, rep_dict, recanon)
        except KeyError as e:
            skipped.append((comp, str(e)))
            continue

        fps_by_family: Dict[str, np.ndarray] = {
            fam: vec.astype(np.float32)
            for fam, vec in sorted(rep.items())
            if vec is not None
        }

        is_active_at_12_50 = False
        if 12.48 in piv_cls.index and 50.0 in piv_cls.columns:
            try:
                is_active_at_12_50 = bool(piv_cls.loc[12.48, 50.0] == 1)
            except Exception:
                pass

        scaffold = get_murko_scaffold(canon_smi)

        meta = CompoundMeta(
            compound=comp,
            smiles=canon_smi,
            scaffold=scaffold,
            pivot_od=piv_od,
            pivot_cls=piv_cls,
            t_vals=t_vals,
            c_vals=c_vals,
            single_conc=(c_vals.size == 1),
            t_min=float(t_vals.min()),
            t_max=float(t_vals.max()),
            logc_min=float(np.log(c_vals.min())),
            logc_max=float(np.log(c_vals.max())),
            fps_by_family=fps_by_family,
            is_active_at_12_50=is_active_at_12_50,
        )
        metas.append(meta)

    if skipped:
        print(f"  WARNING: skipped {len(skipped)} compounds (SMILES parse or rep lookup failed).")
        for comp, err in skipped[:10]:
            print(f"    {comp}: {err}")
        if len(skipped) > 10:
            print(f"    ... and {len(skipped) - 10} more")

    return metas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # =========================================================
    # Configure these paths before running
    # =========================================================
    DF_PATHS = [
        PROJ / "data/train/df_GrowthCurve_27000.pkl",   # ← path to GrowthCurve labeled df pkl
        PROJ / "data/train/df_combined_Enamine.pkl",   # ← path to DR combined df pkl
    ]
    REP_DICT_PATH = PROJ / "data/smiles_representations.pkl"
    OUTPUT_PATH   = PROJ / "data/splits/my_split_v1/all_compound_metas.pkl"
    # =========================================================

    # 1. Load and combine DataFrames
    print("Loading DataFrames...")
    dfs = []
    for p in DF_PATHS:
        df = pd.read_pickle(p)
        print(f"  {p.name}: {len(df):,} rows")
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)
    print(f"  Combined: {len(df_all):,} rows")

    # 2. Keep only test compounds
    df_test = df_all[df_all["Control_Label"] == 0].copy()
    print(f"  After dropping controls: {len(df_test):,} rows, "
          f"{df_test['Compound'].nunique():,} unique compounds")

    # Warn on cross-source compound overlap
    if len(dfs) > 1:
        compound_sets = [set(df[df["Control_Label"] == 0]["Compound"].unique()) for df in dfs]
        overlap = compound_sets[0].intersection(*compound_sets[1:])
        if overlap:
            print(f"  WARNING: {len(overlap)} compound name(s) appear in multiple source DataFrames "
                  f"and will be merged by groupby. First 5: {list(overlap)[:5]}")

    # 3. Canonicalize unique SMILES
    print("\nCanonicalizating SMILES...")
    raw_smiles = df_test.groupby("Compound")["Smiles"].first()
    canon_map = {}
    bad = []
    for comp, smi in raw_smiles.items():
        try:
            canon_map[comp] = canonicalize(str(smi))
        except ValueError as e:
            bad.append((comp, str(e)))
    if bad:
        print(f"  WARNING: {len(bad)} compounds have unparseable SMILES and will be skipped.")
        df_test = df_test[~df_test["Compound"].isin({c for c, _ in bad})]
    unique_canon = list(set(canon_map.values()))
    print(f"  {len(unique_canon)} unique canonical SMILES")

    # 4. Load rep_dict cache
    print(f"\nLoading representations from {REP_DICT_PATH}...")
    if REP_DICT_PATH.exists():
        with open(REP_DICT_PATH, "rb") as f:
            rep_dict = pickle.load(f)
        print(f"  {len(rep_dict):,} entries in cache.")
    else:
        print("  Cache not found — starting empty.")
        rep_dict = {}

    # 5. Build recanon lookup
    recanon = build_recanon_lookup(rep_dict)

    # 6. Compute fingerprints for any missing SMILES
    print("\nEnsuring representations for all SMILES...")
    rep_dict = ensure_representations(unique_canon, rep_dict, recanon)

    # 7. Save updated rep_dict back to disk
    print(f"\nSaving updated rep_dict ({len(rep_dict):,} entries) to {REP_DICT_PATH}...")
    with open(REP_DICT_PATH, "wb") as f:
        pickle.dump(rep_dict, f)

    # Rebuild recanon after adding new entries
    recanon = build_recanon_lookup(rep_dict)

    # 8. Build CompoundMeta list
    print("\nBuilding CompoundMeta objects...")
    metas = build_metas(df_test, rep_dict, recanon)
    print(f"  Built {len(metas)} CompoundMeta objects.")

    # 9. Save output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(metas, f)
    print(f"\nSaved {len(metas)} CompoundMeta objects → {OUTPUT_PATH}")

    # 10. Summary
    if metas:
        sample = metas[0]
        fam_dims = {k: v.shape[0] for k, v in sample.fps_by_family.items()}
        print(f"\nFeature families: {fam_dims}")
        print(f"Total fp dim: {sum(fam_dims.values())}")
    print("\nDone.")


if __name__ == "__main__":
    main()
