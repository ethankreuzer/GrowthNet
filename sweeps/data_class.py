from __future__ import annotations
import math, re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple, List, Sequence
from typing import Literal
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.interpolate import RectBivariateSpline
from torch.utils.data._utils.collate import default_collate


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
      - y_cls: interpolated classification target (placeholder for now)
      - features_by_family: dict {family: fingerprint tensor}

    If a compound has >1 concentration → local RectBivariateSpline in (time, log(conc)).
    If only one concentration → calls placeholder _interpolate_regression_single_conc().
    Classification interpolation is also a placeholder: _interpolate_classification().

    Output dict keys:
      {
        'compound', 'smiles', 'single_conc',
        't': FloatTensor[k], 'c': FloatTensor[k],
        'y_reg': FloatTensor[k], 'y_cls': FloatTensor[k],   # y_cls presently NaN
        'features_by_family': dict[str, FloatTensor]
      }
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        k: int, 
        seed: Optional[int] = None,
        num_fourier: int,
        # Control which fingerprint families to include; None = include all present
        compounds: Optional[Iterable[str]] = None, #if wanted to only train on these compounds
    ):
        
        self.df = df.copy()

        self.max_time = self.df['Timepoint'].max()

        self.num_fourier = int(num_fourier)
        self.k = int(k)
        self.rbs_reg = {'kx': 2, 'ky': 2, 's': 0.0}
        self.kx = int(self.rbs_reg.get('kx', 2))
        self.ky = int(self.rbs_reg.get('ky', 2))


        self.rng = np.random.default_rng(seed)

        #needed = {'Compound', 'Smiles', 'Timepoint', 'Concentration', 'OD', 'is_Active'}

        self.fp_cols_by_family: List[str] = sorted(self._collect_fp_groups(self.df)) #dict of col names for each fp ['ecfp_fp', 'maccs_fp', 'rdkit_fp']

        if compounds is not None:
            keep = set(compounds)
            self.df = self.df[self.df['Compound'].isin(keep)].reset_index(drop=True) 
            
        # Build per-compound metadata
        self._metas: List[CompoundMeta] = []
        for comp, sub in self.df.groupby('Compound', sort=True): #MAY need to change this to group by SMILES, assuming fp map is injective
            
            sub = sub.sort_values(['Timepoint', 'Concentration'])

            piv_od = sub.pivot(index='Timepoint', columns='Concentration', values='OD') \
                        .sort_index(axis=0).sort_index(axis=1)
            piv_cls = sub.pivot(index='Timepoint', columns='Concentration', values='is_Active') \
                        .sort_index(axis=0).sort_index(axis=1)

            t_vals = piv_od.index.values.astype(float)
            c_vals = piv_od.columns.values.astype(float)
            smiles = str(sub['Smiles'].iloc[0])

            # Per-family fingerprint vectors, consistent length/order across compounds
            fps_by_family: Dict[str, np.ndarray] = {}
            
            for col in self.fp_cols_by_family: #extract fp for associated family (maccs, rdkit, ecfp...)
  
                arr = sub[col].iloc[0]              
                vec = np.array(arr, dtype=np.float32)
                fps_by_family[col] = vec


            single_conc = (c_vals.size == 1)
            meta = CompoundMeta(
                compound=comp,
                smiles=smiles,
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
                is_active_at_12_50=bool(
                    (12.48 in piv_cls.index) and 
                    (50.0 in piv_cls.columns) and 
                    (piv_cls.at[12.48, 50.0] == 1)
                )
            )
            self._metas.append(meta)

        if not self._metas:
            raise ValueError("No compounds available after filtering.")

    def __len__(self) -> int:
        return len(self._metas)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        meta = self._metas[idx]

        #t_arr = np.empty(self.k, dtype=np.float32)
        t_enc_arr = np.empty((self.k, 2 * self.num_fourier), dtype=np.float32)
        c_arr = np.empty(self.k, dtype=np.float32)
        y_reg = np.empty(self.k, dtype=np.float32)
        y_cls = np.empty(self.k, dtype=np.float32)

        for i in range(self.k):
            
            t = self.rng.uniform(1e-3, meta.t_max)

            if meta.single_conc:
                
                c = float(meta.c_vals[0]) #should be 50

                y_r, y_c = self. _interpolate_single_conc(meta, t_samp=t, c_samp=c)  # placeholder


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
                    labels_pivot=meta.pivot_cls,k=4
                )
            
            for j, k_freq in enumerate(range(1, self.num_fourier + 1)):
                angle = 2 * np.pi * k_freq * t / self.max_time 
                t_enc_arr[i, 2*j]   = np.sin(angle)
                t_enc_arr[i, 2*j+1] = np.cos(angle)
            
            c_arr[i], y_reg[i], y_cls[i] = c, y_r, y_c


        # Convert per-family features to tensors
        features_by_family = {fam: torch.from_numpy(vec) for fam, vec in meta.fps_by_family.items()}

        


        return {
            'compound': meta.compound,
            'smiles': meta.smiles,
            'single_conc': meta.single_conc,
            't_fourier': torch.from_numpy(t_enc_arr),
            'c_raw': torch.from_numpy(c_arr),
            "c_log": torch.from_numpy(np.log(c_arr)),
            'y_reg': torch.from_numpy(y_reg),
            'y_cls': torch.from_numpy(y_cls),          
            'features_by_family': features_by_family,  
        }
    
    def _collect_fp_groups(
        self,
        df: pd.DataFrame,
        FP_REGEX=re.compile(r"^(.+?)_fp"),  # e.g., ["rdkit", "maccs"]
    ) -> List[str]:

        groups: List[str] = []
        for col in df.columns:
            m = FP_REGEX.match(col)
            if not m:
                continue
            groups.append(col)
        return groups
    
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
            # Collect all valid datapoints
            coords_time, coords_conc = np.meshgrid(
                labels_pivot.index.values.astype(float),
                labels_pivot.columns.values.astype(float),
                indexing="ij"
            )
            coords_time = coords_time.ravel()
            coords_conc = coords_conc.ravel()
            coords_logc = np.log(coords_conc)
            coords_od = od_pivot.to_numpy().ravel()
            coords_labels = labels_pivot.to_numpy().ravel()

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

