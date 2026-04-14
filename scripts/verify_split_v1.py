#!/usr/bin/env python
"""
Equivalence test: proves data/splits/Celine_v1/ reproduces the contents of
the old train and validation pickle files exactly.

Checks:
  1. Train CompoundMeta equivalence (field by field, compound by compound)
  2. val_main equivalence (after consistent sort)
  3. All 7 concentration-slice val dicts reproduced by masking new val_main
  4. SMILES list sanity (disjoint, all present in unified pickle)

Exits 0 on full pass, 1 on any failure.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJ = Path("/home/ethan2/GrowthNet")
sys.path.insert(0, str(PROJ / "sweeps"))

from data_class import CompoundMeta

# ── Old artifact paths ───────────────────────────────────────────────────────
OLD_TRAIN_METAS = PROJ / "data/train/Celine_CompoundMetas_list.pkl"
OLD_VAL_MAIN    = PROJ / "data/test/dict_val_fourier_k_3_Celine.pkl"
OLD_CONC_SLICES = {
    0.2:   PROJ / "data/test/dict_val_fourier_k_3_conc_0_2_Celine.pkl",
    0.781: PROJ / "data/test/dict_val_fourier_k_3_conc_0_781_Celine.pkl",
    1.2:   PROJ / "data/test/dict_val_fourier_k_3_conc_1_2_Celine.pkl",
    3.13:  PROJ / "data/test/dict_val_fourier_k_3_conc_3_13_Celine.pkl",
    7.9:   PROJ / "data/test/dict_val_fourier_k_3_conc_7_9_Celine.pkl",
    12.5:  PROJ / "data/test/dict_val_fourier_k_3_conc_12_50_Celine.pkl",
    50.0:  PROJ / "data/test/dict_val_fourier_k_3_conc_50_Celine.pkl",
}

# ── New artifact paths ───────────────────────────────────────────────────────
SPLIT_DIR      = PROJ / "data/splits/Celine_v1"
NEW_ALL_METAS  = SPLIT_DIR / "all_compound_metas.pkl"
NEW_TRAIN_SMILES = SPLIT_DIR / "train_smiles.txt"
NEW_VAL_SMILES   = SPLIT_DIR / "val_smiles.txt"

NUM_FOURIER = 3

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

failures = 0


def fail(msg: str):
    global failures
    failures += 1
    print(f"  {FAIL}  {msg}")


def ok(msg: str):
    print(f"  {PASS}  {msg}")


# ── Helpers ──────────────────────────────────────────────────────────────────

def build_val_dict_from_metas(metas: list, num_fourier: int = NUM_FOURIER) -> dict:
    """
    Flatten CompoundMeta pivot tables directly into a collated val dict.
    Rows sorted by (Compound, Timepoint, Concentration) — same order as
    build_compound_metas sorts before pivoting.
    """
    compounds, smiles_list = [], []
    t_raw_list, c_raw_list, y_reg_list, y_cls_list = [], [], [], []
    fps_by_family: dict = {fam: [] for fam in sorted(metas[0].fps_by_family.keys())}

    # Sort metas by compound name to ensure deterministic order
    for meta in sorted(metas, key=lambda m: m.compound):
        for t_idx in sorted(meta.pivot_od.index):
            for c_idx in sorted(meta.pivot_od.columns):
                od  = meta.pivot_od.loc[t_idx, c_idx]
                cls = meta.pivot_cls.loc[t_idx, c_idx]
                if pd.isna(od) or pd.isna(cls):
                    continue
                compounds.append(meta.compound)
                smiles_list.append(meta.smiles)
                t_raw_list.append(float(t_idx))
                c_raw_list.append(float(c_idx))
                y_reg_list.append(float(od))
                y_cls_list.append(float(cls))
                for fam in fps_by_family:
                    fps_by_family[fam].append(meta.fps_by_family[fam])

    t_raw = np.array(t_raw_list, dtype=np.float32)
    c_raw = np.array(c_raw_list, dtype=np.float32)

    # Fourier encoding: t' = t - 1, T = 15, sin/cos pairs for k=1..num_fourier
    T = 15.0
    t_enc = np.zeros((len(t_raw), 2 * num_fourier), dtype=np.float32)
    for j, k_freq in enumerate(range(1, num_fourier + 1)):
        angle = 2 * np.pi * k_freq * (t_raw - 1.0) / T
        t_enc[:, 2*j]   = np.sin(angle)
        t_enc[:, 2*j+1] = np.cos(angle)

    return {
        "compound": compounds,
        "smiles":   smiles_list,
        "t_raw":    torch.from_numpy(t_raw),
        "t_fourier": torch.from_numpy(t_enc),
        "c_raw":    torch.from_numpy(c_raw),
        "c_log":    torch.from_numpy(np.log(c_raw)),
        "y_reg":    torch.from_numpy(np.array(y_reg_list, dtype=np.float32)),
        "y_cls":    torch.from_numpy(np.array(y_cls_list, dtype=np.float32)),
        "features_by_family": {
            fam: torch.from_numpy(np.stack(vecs)) for fam, vecs in fps_by_family.items()
        },
    }


def sort_val_dict(d: dict) -> tuple:
    """Return sort indices by (compound, t_raw, c_raw)."""
    compounds = d["compound"] if isinstance(d["compound"], list) else list(d["compound"])
    t = d["t_raw"]
    c = d["c_raw"]
    if isinstance(t, torch.Tensor):
        t = t.numpy()
        c = c.numpy()
    # Compound is a list of strings; sort lexicographically then by t then c
    keys = list(zip(compounds, t.tolist(), c.tolist()))
    return sorted(range(len(keys)), key=lambda i: keys[i])


def reorder_val_dict(d: dict, idx: list) -> dict:
    """Reorder all arrays/lists in a val dict by index list."""
    out = {}
    for k, v in d.items():
        if k == "features_by_family":
            out[k] = {fam: arr[idx] for fam, arr in v.items()}
        elif isinstance(v, torch.Tensor):
            out[k] = v[idx]
        elif isinstance(v, np.ndarray):
            out[k] = v[idx]
        elif isinstance(v, list):
            out[k] = [v[i] for i in idx]
        else:
            out[k] = v
    return out


def assert_arrays_close(a, b, name: str, atol: float = 1e-5):
    if isinstance(a, torch.Tensor): a = a.numpy()
    if isinstance(b, torch.Tensor): b = b.numpy()
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        fail(f"{name}: shape mismatch {a.shape} vs {b.shape}")
        return
    if not np.allclose(a, b, atol=atol, equal_nan=True):
        diff = np.abs(a - b)
        fail(f"{name}: max diff {diff.max():.6f} at index {diff.argmax()}")
    else:
        ok(f"{name}: arrays match (shape {a.shape})")


# ─────────────────────────────────────────────────────────────────────────────
# Load artifacts
# ─────────────────────────────────────────────────────────────────────────────

print("Loading old artifacts...")
with open(OLD_TRAIN_METAS, "rb") as f:
    old_train_metas = pickle.load(f)
print(f"  Old train metas: {len(old_train_metas)}")

with open(OLD_VAL_MAIN, "rb") as f:
    old_val_main = pickle.load(f)
print(f"  Old val_main rows: {len(old_val_main['compound'])}")

print("\nLoading new artifacts...")
with open(NEW_ALL_METAS, "rb") as f:
    all_metas = pickle.load(f)
print(f"  All compound metas: {len(all_metas)}")

train_smiles = set(open(NEW_TRAIN_SMILES).read().splitlines())
val_smiles   = set(open(NEW_VAL_SMILES).read().splitlines())
print(f"  Train SMILES: {len(train_smiles)}  Val SMILES: {len(val_smiles)}")

new_train_metas = [m for m in all_metas if m.smiles in train_smiles]
new_val_metas   = [m for m in all_metas if m.smiles in val_smiles]
print(f"  Filtered train metas: {len(new_train_metas)}  Val metas: {len(new_val_metas)}")

# ─────────────────────────────────────────────────────────────────────────────
# Check 1: SMILES list sanity
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Check 1: SMILES list sanity ─────────────────────────────────────────")

overlap = train_smiles & val_smiles
if overlap:
    fail(f"Train/val overlap: {len(overlap)} SMILES, e.g. {list(overlap)[0]!r}")
else:
    ok(f"No overlap between train and val SMILES")

all_smiles_in_pkl = {m.smiles for m in all_metas}
missing_train = train_smiles - all_smiles_in_pkl
missing_val   = val_smiles   - all_smiles_in_pkl
if missing_train:
    fail(f"{len(missing_train)} train SMILES not in all_compound_metas.pkl")
else:
    ok("All train SMILES present in unified pickle")
if missing_val:
    fail(f"{len(missing_val)} val SMILES not in all_compound_metas.pkl")
else:
    ok("All val SMILES present in unified pickle")

# ─────────────────────────────────────────────────────────────────────────────
# Check 2: Train CompoundMeta equivalence
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Check 2: Train CompoundMeta equivalence ─────────────────────────────")

if len(new_train_metas) != len(old_train_metas):
    fail(f"Count mismatch: new={len(new_train_metas)} old={len(old_train_metas)}")
else:
    ok(f"Count matches: {len(new_train_metas)} compounds")

old_by_compound = {m.compound: m for m in old_train_metas}
new_by_compound = {m.compound: m for m in new_train_metas}

compound_mismatches = set(old_by_compound) ^ set(new_by_compound)
if compound_mismatches:
    fail(f"Compound name mismatch: {len(compound_mismatches)} differences, e.g. {list(compound_mismatches)[:3]}")
else:
    ok("Compound names match exactly")

scalar_mismatches = 0
pivot_mismatches  = 0
fps_mismatches    = 0

for comp in sorted(old_by_compound):
    if comp not in new_by_compound:
        continue
    old_m = old_by_compound[comp]
    new_m = new_by_compound[comp]

    # Scalar fields
    for field in ("single_conc", "is_active_at_12_50", "t_min", "t_max", "logc_min", "logc_max"):
        ov = getattr(old_m, field)
        nv = getattr(new_m, field)
        if isinstance(ov, float):
            if not np.isclose(ov, nv, atol=1e-6):
                scalar_mismatches += 1
                if scalar_mismatches == 1:
                    fail(f"First scalar mismatch: {comp!r}.{field}: old={ov} new={nv}")
                break
        elif ov != nv:
            scalar_mismatches += 1
            if scalar_mismatches == 1:
                fail(f"First scalar mismatch: {comp!r}.{field}: old={ov} new={nv}")
            break

    # Pivot tables
    try:
        pd.testing.assert_frame_equal(old_m.pivot_od, new_m.pivot_od, check_names=False, rtol=1e-5)
        pd.testing.assert_frame_equal(old_m.pivot_cls, new_m.pivot_cls, check_names=False)
    except AssertionError as e:
        pivot_mismatches += 1
        if pivot_mismatches == 1:
            fail(f"First pivot mismatch: {comp!r}: {e}")

    # Fingerprints
    for fam in old_m.fps_by_family:
        if fam not in new_m.fps_by_family:
            fps_mismatches += 1
            if fps_mismatches == 1:
                fail(f"First fp mismatch: {comp!r} missing family {fam!r} in new")
            continue
        if not np.array_equal(old_m.fps_by_family[fam], new_m.fps_by_family[fam]):
            fps_mismatches += 1
            if fps_mismatches == 1:
                fail(f"First fp mismatch: {comp!r} family {fam!r}")

if scalar_mismatches == 0:
    ok(f"All scalar fields match across {len(old_by_compound)} compounds")
else:
    fail(f"{scalar_mismatches} compounds have scalar field mismatches")

if pivot_mismatches == 0:
    ok(f"All pivot_od and pivot_cls tables match")
else:
    fail(f"{pivot_mismatches} compounds have pivot mismatches")

if fps_mismatches == 0:
    ok(f"All fps_by_family arrays match bit-identically")
else:
    fail(f"{fps_mismatches} compounds have fingerprint mismatches")

# ─────────────────────────────────────────────────────────────────────────────
# Check 3: val_main equivalence
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Check 3: val_main equivalence ───────────────────────────────────────")

print("  Building new val_main from val CompoundMetas...")
new_val_main = build_val_dict_from_metas(new_val_metas)
print(f"  New val_main rows: {len(new_val_main['compound'])}")
print(f"  Old val_main rows: {len(old_val_main['compound'])}")

if len(new_val_main['compound']) != len(old_val_main['compound']):
    fail(f"Row count mismatch: new={len(new_val_main['compound'])} old={len(old_val_main['compound'])}")
else:
    ok(f"Row counts match: {len(new_val_main['compound'])}")

    # Sort both by (compound, t_raw, c_raw)
    new_idx = sort_val_dict(new_val_main)
    old_idx = sort_val_dict(old_val_main)
    new_sorted = reorder_val_dict(new_val_main, new_idx)
    old_sorted = reorder_val_dict(old_val_main, old_idx)

    # Check compound/smiles lists
    if new_sorted["compound"] != old_sorted["compound"]:
        fail("compound lists differ after sort")
    else:
        ok("compound lists match after sort")

    for key in ("t_raw", "c_raw", "c_log", "y_reg", "y_cls"):
        assert_arrays_close(new_sorted[key], old_sorted[key], f"val_main/{key}")

    for dim_key in ("t_fourier",):
        assert_arrays_close(new_sorted[dim_key], old_sorted[dim_key], f"val_main/{dim_key}")

    for fam in old_sorted["features_by_family"]:
        assert_arrays_close(
            new_sorted["features_by_family"][fam],
            old_sorted["features_by_family"][fam],
            f"val_main/features_by_family/{fam}",
        )

# ─────────────────────────────────────────────────────────────────────────────
# Check 4: Concentration slice equivalence
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Check 4: Concentration slice equivalence ────────────────────────────")

# Get new val_main as numpy for masking
new_t = new_val_main["t_raw"].numpy() if isinstance(new_val_main["t_raw"], torch.Tensor) else np.asarray(new_val_main["t_raw"])
new_c = new_val_main["c_raw"].numpy() if isinstance(new_val_main["c_raw"], torch.Tensor) else np.asarray(new_val_main["c_raw"])

for conc, old_path in OLD_CONC_SLICES.items():
    with open(old_path, "rb") as f:
        old_slice = pickle.load(f)

    # Derive slice from new val_main: t != 0 AND c == conc
    mask = (np.abs(new_c - conc) < 0.001) & (new_t != 0)
    indices = np.where(mask)[0].tolist()

    if len(indices) != len(old_slice["compound"]):
        fail(f"conc={conc}: row count mismatch new={len(indices)} old={len(old_slice['compound'])}")
        continue

    new_slice = reorder_val_dict(new_val_main, indices)

    # Sort both for comparison
    new_s_idx = sort_val_dict(new_slice)
    old_s_idx = sort_val_dict(old_slice)
    new_slice_s = reorder_val_dict(new_slice, new_s_idx)
    old_slice_s = reorder_val_dict(old_slice, old_s_idx)

    ok(f"conc={conc}: row count {len(indices)} matches")
    for key in ("t_raw", "c_raw", "y_reg", "y_cls"):
        assert_arrays_close(new_slice_s[key], old_slice_s[key], f"  conc={conc}/{key}")

# ─────────────────────────────────────────────────────────────────────────────
# Result
# ─────────────────────────────────────────────────────────────────────────────
print()
if failures == 0:
    print(f"\033[32m{'═'*60}\033[0m")
    print(f"\033[32m  ALL CHECKS PASSED\033[0m")
    print(f"\033[32m{'═'*60}\033[0m")
    sys.exit(0)
else:
    print(f"\033[31m{'═'*60}\033[0m")
    print(f"\033[31m  {failures} CHECK(S) FAILED — do not delete old pickles\033[0m")
    print(f"\033[31m{'═'*60}\033[0m")
    sys.exit(1)
