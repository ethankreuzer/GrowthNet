#!/usr/bin/env python
"""
Build a dictionary mapping canonical SMILES -> precomputed representations.

Representations include:
  - maccs_fp    (166-dim, via datamol)
  - ecfp_fp     (2048-dim, via datamol)
  - rdkit_fp    (2048-dim, via datamol)
  - boltz2_rep  (3072-dim pooled Boltz2 pair representation)
  - minimol_fp  (512-dim, MiniMol molecular fingerprint)

Usage (called by the SLURM wrapper):
  python build_representation_dict.py --stage prep
  boltz predict ...  (run externally)
  python build_representation_dict.py --stage assemble
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SMILES_TXT = Path("/home/ethan2/GrowthNet/data/unique_smiles.txt")
YAML_DIR = Path("/home/ethan2/GrowthNet/data/boltz_yamls")
BOLTZ_OUT = Path("/home/ethan2/GrowthNet/data/boltz_output")
INDEX_PKL = Path("/home/ethan2/GrowthNet/data/smiles_index.pkl")
OUTPUT_PKL = Path("/home/ethan2/GrowthNet/data/smiles_representations.pkl")

# ---------------------------------------------------------------------------
# Fingerprint helpers (matches existing codebase: dm.to_fp defaults)
# ---------------------------------------------------------------------------
def compute_fingerprints(smi: str) -> dict:
    import datamol as dm
    result = {}
    for fp_type in ("maccs", "ecfp", "rdkit"):
        try:
            result[f"{fp_type}_fp"] = dm.to_fp(smi, fp_type=fp_type)
        except Exception:
            result[f"{fp_type}_fp"] = None
    return result


# ---------------------------------------------------------------------------
# Boltz2 pooling utilities (from boltz-as-FM/ADMET/construct_dataset/utils.py)
# ---------------------------------------------------------------------------
def smiles_to_edge_index(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = AllChem.AddHs(mol)
    canonical_order = AllChem.CanonicalRankAtoms(mol)
    mol = rdmolops.RenumberAtoms(mol, canonical_order)
    rdkit2node = {}
    atom_idx = 0
    for e, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum() == 1:
            continue
        rdkit2node[e] = atom_idx
        atom_idx += 1
    src, dst = [], []
    for bond in mol.GetBonds():
        i_raw, j_raw = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i_raw in rdkit2node and j_raw in rdkit2node:
            i, j = rdkit2node[i_raw], rdkit2node[j_raw]
            src += [i, j]
            dst += [j, i]
    edge_index = np.asarray([src, dst], dtype=np.int32)
    return edge_index, atom_idx


def k_hop_masks(adj, k_max=1):
    n = adj.shape[0]
    dist = np.full((n, n), np.inf)
    np.fill_diagonal(dist, 0)
    remaining = np.isinf(dist)
    reach = adj.copy()
    k = 1
    while k <= k_max and remaining.any():
        newly = reach & remaining
        dist[newly] = k
        remaining[newly] = False
        reach = (reach @ adj) > 0
        k += 1
    masks = [dist == r for r in range(0, k_max + 1)]
    return dist, masks


def pool_embedding(embedding, edge_index):
    z_full = np.concatenate(
        [embedding["z"][-1], embedding["z"][-2],
         embedding["z"][-3], embedding["z"][-4]], axis=-1
    )
    z = (z_full + np.transpose(z_full, (1, 0, 2))) / 2
    adjacency = np.zeros(z.shape[:2], bool)
    if edge_index.size > 0:
        adjacency[edge_index[0], edge_index[1]] = True
    np.fill_diagonal(adjacency, False)

    _, masks = k_hop_masks(adjacency, k_max=1)
    rings_mean, rings_std = [], []
    for mask in masks:
        vals = z[mask]
        if vals.size:
            rings_mean.append(vals.mean(0))
            rings_std.append(vals.std(0))
        else:
            rings_mean.append(np.zeros(z.shape[-1]))
            rings_std.append(np.zeros(z.shape[-1]))

    rings_mean = np.stack(rings_mean).reshape(-1)
    rings_std = np.stack(rings_std).reshape(-1)
    all_pooled = z_full.mean(axis=(0, 1))
    all_pooled_std = z_full.reshape(-1, z_full.shape[-1]).std(axis=0)
    return np.concatenate([rings_mean, rings_std, all_pooled, all_pooled_std], axis=0)


# ===================================================================
# STAGE 1: PREP — extract unique SMILES, compute fps, write YAMLs
# ===================================================================
def stage_prep():
    print("=== Stage 1: PREP ===")

    print(f"Loading SMILES from {SMILES_TXT}...")
    with open(SMILES_TXT) as f:
        all_smiles = [line.strip() for line in f if line.strip()]
    print(f"Total unique canonical SMILES: {len(all_smiles)}")

    YAML_DIR.mkdir(parents=True, exist_ok=True)

    index = {}
    print("Computing fingerprints and writing Boltz2 YAML files...")
    for i, smi in enumerate(all_smiles):
        if (i + 1) % 5000 == 0 or i == 0:
            print(f"  Processing {i + 1}/{len(all_smiles)}...")

        fps = compute_fingerprints(smi)

        yaml_name = f"mol_{i + 1}"
        yaml_content = {
            "version": 1,
            "sequences": [
                {"protein": {"id": "A", "sequence": "X", "msa": "empty"}},
                {"ligand": {"id": "B", "smiles": smi}},
            ],
        }
        yaml_path = YAML_DIR / f"{yaml_name}.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f, sort_keys=False)

        index[smi] = {
            "yaml_name": yaml_name,
            "maccs_fp": fps["maccs_fp"],
            "ecfp_fp": fps["ecfp_fp"],
            "rdkit_fp": fps["rdkit_fp"],
        }

    with open(INDEX_PKL, "wb") as f:
        pickle.dump(index, f)

    print(f"Wrote {len(all_smiles)} YAML files to {YAML_DIR}")
    print(f"Saved index to {INDEX_PKL}")
    print("=== Stage 1 complete ===")


# ===================================================================
# STAGE 3: ASSEMBLE — load Boltz2 embeddings, pool, build final dict
# ===================================================================
def stage_assemble():
    print("=== Stage 3: ASSEMBLE ===")

    print("Loading index...")
    with open(INDEX_PKL, "rb") as f:
        index = pickle.load(f)

    result = {}
    failed = []
    total = len(index)

    print(f"Processing {total} molecules...")
    for i, (smi, info) in enumerate(index.items()):
        if (i + 1) % 5000 == 0 or i == 0:
            print(f"  Assembling {i + 1}/{total}...")

        yaml_name = info["yaml_name"]
        emb_path = (
            BOLTZ_OUT
            / f"boltz_results_boltz_yamls/predictions/{yaml_name}"
            / f"embeddings_{yaml_name}.npz"
        )

        boltz2_rep = None
        if emb_path.exists():
            try:
                embedding = np.load(emb_path)
                edge_index, _ = smiles_to_edge_index(smi)
                boltz2_rep = pool_embedding(embedding, edge_index)
            except Exception as e:
                failed.append((smi, str(e)))
        else:
            failed.append((smi, f"Embedding file not found: {emb_path}"))

        result[smi] = {
            "maccs_fp": info["maccs_fp"],
            "ecfp_fp": info["ecfp_fp"],
            "rdkit_fp": info["rdkit_fp"],
            "boltz2_rep": boltz2_rep,
        }

    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(result, f)

    print(f"Saved {len(result)} entries to {OUTPUT_PKL}")
    if failed:
        print(f"WARNING: {len(failed)} molecules had Boltz2 embedding failures.")
        for smi, err in failed[:10]:
            print(f"  {smi[:60]}... -> {err}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")
    print("=== Stage 3 complete ===")


# ===================================================================
# STAGE 4: ADD-MINIMOL — compute MiniMol fingerprints for existing dict
# ===================================================================
def stage_add_minimol():
    import torch
    from minimol import Minimol

    print("=== Stage 4: ADD-MINIMOL ===")

    print(f"Loading existing representations from {OUTPUT_PKL}...")
    with open(OUTPUT_PKL, "rb") as f:
        result = pickle.load(f)

    all_smiles = list(result.keys())
    print(f"Computing MiniMol embeddings for {len(all_smiles)} molecules...")

    model = Minimol()
    BATCH_SIZE = 512
    minimol_map = {}
    failed = []

    for batch_start in range(0, len(all_smiles), BATCH_SIZE):
        batch = all_smiles[batch_start:batch_start + BATCH_SIZE]
        batch_end = min(batch_start + BATCH_SIZE, len(all_smiles))
        if (batch_start // BATCH_SIZE) % 10 == 0:
            print(f"  Batch {batch_start}–{batch_end} / {len(all_smiles)}...")
        try:
            embeddings = model(batch)
            for smi, emb in zip(batch, embeddings):
                minimol_map[smi] = emb.detach().cpu().numpy()
        except Exception:
            for smi in batch:
                try:
                    emb = model([smi])[0]
                    minimol_map[smi] = emb.detach().cpu().numpy()
                except Exception as e:
                    minimol_map[smi] = None
                    failed.append((smi, str(e)))

    for smi in result:
        result[smi]["minimol_fp"] = minimol_map.get(smi)

    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(result, f)

    print(f"Updated {len(result)} entries in {OUTPUT_PKL}")
    if failed:
        print(f"WARNING: {len(failed)} molecules failed MiniMol embedding.")
        for smi, err in failed[:10]:
            print(f"  {smi[:60]}... -> {err}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")
    print("=== Stage 4 complete ===")


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build SMILES representation dictionary")
    parser.add_argument(
        "--stage", required=True, choices=["prep", "assemble", "add-minimol"],
        help="Which stage to run: 'prep', 'assemble', or 'add-minimol'",
    )
    args = parser.parse_args()

    if args.stage == "prep":
        stage_prep()
    elif args.stage == "assemble":
        stage_assemble()
    elif args.stage == "add-minimol":
        stage_add_minimol()
