from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.interpolate import RectBivariateSpline
from torch.utils.data._utils.collate import default_collate
import pickle

@dataclass(frozen=True)
class CompoundMeta:
    compound: str
    smiles: str
    pivot_od: pd.DataFrame            # index: Timepoint, columns: Concentration → OD
    pivot_cls: pd.DataFrame           # same axes → is_Active
    t_vals: np.ndarray                # sorted unique timepoints (float)
    c_vals: np.ndarray                # sorted unique concentrations (float)
    single_conc: bool                 # True if only one conc present
    t_min: float
    t_max: float
    logc_min: float
    logc_max: float
    is_active_at_12_50: bool  # NEW: if compound is active at time 12 and conc 50
    fps_by_family: Dict[str, np.ndarray]  # NEW: per-library fingerprint vectors
    


class PerCompoundDataset(Dataset):
    """
    Returns one item per compound containing k sampled (t,c) with:
      - y_reg: interpolated OD (regression target)
      - y_cls: interpolated classification target
      - features_by_family: dict {family: fingerprint tensor}

    If a compound has >1 concentration → local RectBivariateSpline in (time, log(conc)).
    If only one concentration → calls _interpolate_single_conc().

    Output dict keys:
      {
        'compound', 'smiles', 'single_conc',
        't': FloatTensor[k], 'c': FloatTensor[k],
        'y_reg': FloatTensor[k], 'y_cls': FloatTensor[k],
        'features_by_family': dict[str, FloatTensor]
      }
    """

    def __init__(
        self,
        metas_path: str,
        smiles_list_path: str,
        *,
        k: int,
        seed: Optional[int] = None,
        num_fourier: int,
        noise: float,
    ):
        self.num_fourier = int(num_fourier)
        self.k = int(k)
        self.rbs_reg = {'kx': 1, 'ky': 1, 's': 0.0}
        self.kx = int(self.rbs_reg.get('kx'))
        self.ky = int(self.rbs_reg.get('ky'))
        self.noise = noise

        self.rng = np.random.default_rng(seed)

        split_smiles = set(open(smiles_list_path).read().splitlines())

        with open(metas_path, "rb") as f:
            all_metas: List[CompoundMeta] = pickle.load(f)

        self._metas: List[CompoundMeta] = [m for m in all_metas if m.smiles in split_smiles]

        if not self._metas:
            raise ValueError(
                f"No compounds matched the SMILES list in {smiles_list_path!r}. "
                f"Check that the SMILES in the list match the canonical SMILES in {metas_path!r}."
            )

    def __len__(self) -> int:
        return len(self._metas)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        meta = self._metas[idx]

        t_arr = np.empty(self.k, dtype=np.float32)
        t_enc_arr = np.empty((self.k, 2 * self.num_fourier), dtype=np.float32)
        c_arr = np.empty(self.k, dtype=np.float32)
        y_reg = np.empty(self.k, dtype=np.float32)
        y_cls = np.empty(self.k, dtype=np.float32)

        for i in range(self.k):
            
            t = self.rng.uniform(0, meta.t_max)

            if meta.single_conc:
                
                c = float(meta.c_vals[0]) #should be 50

                y_r, y_c = self. _interpolate_single_conc(meta, t_samp=t, c_samp=c)  # placeholder
                y_r = y_r + self.noise

            else:
                
                logc_min, logc_max = meta.logc_min, meta.logc_max
                
                logc_samp = self.rng.uniform(logc_min, logc_max)
     
                c = float(np.exp(logc_samp))
                
                y_r, y_c, _ = self._interpolate_multiple_conc(
                    od_pivot=meta.pivot_od, 
                    t_vals=meta.t_vals, 
                    c_vals=meta.c_vals, 
                    t_samp=t, 
                    c_samp=c, 
                    labels_pivot=meta.pivot_cls,
                    k=4
                )
                y_r = y_r + self.noise

            T = 15
            t_prime= t - 1

            for j, k_freq in enumerate(range(1, self.num_fourier + 1)):
                
                angle = 2 * np.pi * k_freq * t_prime / T
                t_enc_arr[i, 2*j]   = np.sin(angle)
                t_enc_arr[i, 2*j+1] = np.cos(angle)
            
            c_arr[i], y_reg[i], y_cls[i], t_arr[i] = c, y_r, y_c, t


        # Convert per-family features to tensors
        features_by_family = {fam: torch.from_numpy(vec) for fam, vec in meta.fps_by_family.items()}



        return {
            'compound': meta.compound,
            'smiles': meta.smiles,
            'single_conc': meta.single_conc,
            'is_active_at_12_50': meta.is_active_at_12_50,
            't_raw': t_arr,
            't_fourier': torch.from_numpy(t_enc_arr),
            'c_raw': torch.from_numpy(c_arr),
            "c_log": torch.from_numpy(np.log(c_arr)),
            'y_reg': torch.from_numpy(y_reg),
            'y_cls': torch.from_numpy(y_cls),          
            'features_by_family': features_by_family,  
        }
    
    def _interpolate_single_conc(
        self,
        meta: CompoundMeta,
        t_samp: float,
        *,
        c_samp: float,
    ) -> tuple[float, int]:
        """
        Estimate both OD (regression) and label (classification) for the
        single-concentration (50 µM) case at a sampled time.

        Returns
        -------
        (od_est, cls_est)
            od_est : float
                Interpolated OD value, clamped to max observed OD after t=0.
            cls_est : int
                Interpolated binary label (0 or 1).
        """

        
        # OD estimation
        if c_samp not in (50, 50.0):
            raise ValueError(f"Single-conc mode expects c==50; got {c_samp!r}")

        times_expected = np.array([0.0, 6.24, 12.48], dtype=float)

        try:
            y_obs = meta.pivot_od.loc[times_expected, 50.0].to_numpy(dtype=float)
        except KeyError as e:
            raise ValueError(
                "Required concentration 50.0 or times [0.0, 6.24, 12.48] not found in pivot_od"
            ) from e

        if np.isnan(y_obs).any():
            raise ValueError("NaN detected in OD values at conc=50 for required times")

        coeffs = np.polyfit(times_expected, y_obs, deg=2)
        od_est = float(np.polyval(coeffs, t_samp))

        # Clamp against later observed max (exclude t=0.0)
        later_mask = times_expected > 1e-6
        cap = float(np.nanmax(y_obs[later_mask]))
        od_est = min(od_est, cap)

        
        # Label estimation
        try:
            cls_series = meta.pivot_cls.loc[:, 50.0]
        except KeyError as e:
            raise ValueError("Classification labels at concentration 50.0 not found in pivot_cls") from e

        times = cls_series.index.values.astype(float)
        labels = cls_series.to_numpy(dtype=int)

        idx_sorted = np.argsort(np.abs(times - t_samp))
        i1, i2 = idx_sorted[:2]
        t1, t2 = times[i1], times[i2]
        l1, l2 = labels[i1], labels[i2]

        if l1 == l2:
            cls_est = int(l1)
        else:
            # Ensure (t1, l1) is the negative and (t2, l2) the positive
            if l1 == 1 and l2 == 0:
                t1, t2 = t2, t1
                l1, l2 = l2, l1

            dist_total = abs(t2 - t1)
            dist_to_pos = abs(t_samp - t2)
            prob_1 = 1.0 - dist_to_pos / dist_total
            cls_est = int(self.rng.uniform() < prob_1)

        return od_est, cls_est

    def _interpolate_multiple_conc(
        self,
        *,
        od_pivot: pd.DataFrame,
        t_vals: np.ndarray,
        c_vals: np.ndarray,
        t_samp: float,
        c_samp: float,   # linear concentration
        labels_pivot: pd.DataFrame | None = None,
        k: int = 4,
    ) -> tuple[float, int, float]:

        kx, ky = self.kx, self.ky
        need_t0, need_c0 = kx + 1, ky + 1

        if t_vals.size < need_t0 or c_vals.size < need_c0:
            raise ValueError(
                f"Not enough unique values for spline: have {t_vals.size} timepoints "
                f"and {c_vals.size} concentrations; need at least {need_t0} and {need_c0}."
            )

        def nearest_k(vals: np.ndarray, target: float, k: int, logspace: bool = False) -> np.ndarray:
            if logspace:
                idx = np.argsort(np.abs(np.log(vals) - np.log(target)))[:k]
            else:
                idx = np.argsort(np.abs(vals - target))[:k]
            return np.sort(vals[idx])

        # Choose nearest neighbors for spline
        times_used = nearest_k(t_vals, t_samp, need_t0, logspace=False)
        concs_used = nearest_k(c_vals, c_samp, need_c0, logspace=True)

        # Build local OD grid
        grid = od_pivot.loc[times_used, concs_used]
        if grid.isna().any().any():
            raise ValueError("Local rectangle has missing cells; cannot fit spline.")

        X = grid.index.values.astype(float)                              # timepoints
        Ylog = np.log(grid.columns.values.astype(float))  # log concentrations
        Z = grid.to_numpy()

        # Fit local spline in (time, logC) space
        spline = RectBivariateSpline(X, Ylog, Z, kx=kx, ky=ky, s=0.0)

        # Interpolated OD
        logc_samp = np.log(c_samp)
        od_hat = float(spline.ev(t_samp, logc_samp))

        # --- Classification (distance-weighted k-NN) ---
        pred_label, p_active = -1, np.nan
        if labels_pivot is not None:

            grid_cls = labels_pivot.loc[times_used, concs_used] #4 nearest neighbors are in the 9 used for interpolation
            # Collect all valid datapoints
            coords_time, coords_conc = np.meshgrid(
                grid_cls.index.values.astype(float),
                grid_cls.columns.values.astype(float),
                indexing="ij"
            )
            coords_time = coords_time.ravel()
            coords_conc = coords_conc.ravel()
            coords_logc = np.log(coords_conc)


            grid_od = od_pivot.loc[times_used, concs_used]
            grid_labels = labels_pivot.loc[times_used, concs_used]
            coords_od = grid_od.to_numpy().ravel()
            coords_labels = grid_labels.to_numpy().ravel()

            mask = ~np.isnan(coords_od) & ~np.isnan(coords_labels)
            coords_time, coords_logc, coords_od, coords_labels = (
                coords_time[mask], coords_logc[mask],
                coords_od[mask], coords_labels[mask].astype(int)
            )

            # Query point (t, logC, interpolated OD)
            query = np.array([t_samp, logc_samp, od_hat])
            coords = np.column_stack([coords_time, coords_logc, coords_od]) # creates 2D array of [[time,conc,OD],[...],...] so every point is an array

            dists = np.linalg.norm(coords - query, axis=1)
            nn_idx = np.argsort(dists)[:k]

            nn_labels = coords_labels[nn_idx]
            nn_dists = dists[nn_idx]

            eps = 1e-8
            weights = 1.0 / (nn_dists + eps) # Gaussain weights could be a good idea here exp(-dist²/σ²
            p_active = float(np.sum(weights * nn_labels) / np.sum(weights))
            pred_label = int(p_active >= 0.5)

        return od_hat, pred_label, p_active

class ExplicitDataset(Dataset):
    """
    Test dataset that returns observed OD and label values
    without interpolation. One item = one (compound, t, c) row.
    """

    def __init__(self, df: pd.DataFrame, num_fourier: int):
        self.df = df.reset_index(drop=True)
        self.max_time = df["Timepoint"].max()
        self.num_fourier = int(num_fourier)

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

        t_prime=t-1

        T = 15

        for j, k_freq in enumerate(range(1, self.num_fourier + 1)):
            angle = 2 * np.pi * k_freq * t_prime / T
            t_enc[2 * j] = np.sin(angle)
            t_enc[2 * j + 1] = np.cos(angle)

        # Fingerprints
        features_by_family = {
            fam: torch.tensor(row[fam], dtype=torch.float32)
            for fam in self.fp_cols_by_family
        }

        return {
            "compound": row["Compound"],
            "smiles": row["Smiles"],
            "t_raw": torch.tensor(t, dtype=torch.float32),
            "t_fourier": torch.tensor(t_enc, dtype=torch.float32),
            "c_raw": torch.tensor(c, dtype=torch.float32),
            "c_log": torch.tensor(np.log(c), dtype=torch.float32),
            "y_reg": torch.tensor(od, dtype=torch.float32),
            "y_cls": torch.tensor(label, dtype=torch.float32),
            "features_by_family": features_by_family,
        }

def build_val_dict_from_metas(metas: List[CompoundMeta], num_fourier: int = 3) -> dict:
    """
    Build a validation dict from a list of CompoundMeta objects.

    Flattens each meta's pivot_od / pivot_cls into observed (t, c, OD, is_Active)
    rows, computes Fourier time encodings, and stacks fingerprints — producing the
    same dict format as the precomputed val pickles.  Rows are ordered by
    (compound, Timepoint, Concentration) for deterministic output.

    Returns a dict with keys:
        compound, smiles, t_raw, t_fourier, c_raw, c_log, y_reg, y_cls,
        features_by_family
    """
    compounds, smiles_list = [], []
    t_raw_list, c_raw_list, y_reg_list, y_cls_list = [], [], [], []
    fps_acc: Dict[str, list] = {fam: [] for fam in sorted(metas[0].fps_by_family.keys())}

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
                for fam in fps_acc:
                    fps_acc[fam].append(meta.fps_by_family[fam])

    t_raw = np.array(t_raw_list, dtype=np.float32)
    c_raw = np.array(c_raw_list, dtype=np.float32)

    T = 15.0
    t_enc = np.zeros((len(t_raw), 2 * num_fourier), dtype=np.float32)
    for j, k_freq in enumerate(range(1, num_fourier + 1)):
        angle = 2 * np.pi * k_freq * (t_raw - 1.0) / T
        t_enc[:, 2*j]   = np.sin(angle)
        t_enc[:, 2*j+1] = np.cos(angle)

    return {
        "compound":   compounds,
        "smiles":     smiles_list,
        "t_raw":      torch.from_numpy(t_raw),
        "t_fourier":  torch.from_numpy(t_enc),
        "c_raw":      torch.from_numpy(c_raw),
        "c_log":      torch.from_numpy(np.log(c_raw)),
        "y_reg":      torch.from_numpy(np.array(y_reg_list, dtype=np.float32)),
        "y_cls":      torch.from_numpy(np.array(y_cls_list, dtype=np.float32)),
        "features_by_family": {
            fam: torch.from_numpy(np.stack(vecs))
            for fam, vecs in fps_acc.items()
        },
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