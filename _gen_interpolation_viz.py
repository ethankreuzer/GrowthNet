import json

def code_cell(cid, source):
    return {"cell_type": "code", "id": cid, "metadata": {}, "execution_count": None, "outputs": [], "source": source.strip('\n')}

def md_cell(cid, source):
    return {"cell_type": "markdown", "id": cid, "metadata": {}, "source": source.strip('\n')}

cells = []

cells.append(md_cell("md_title", """# Interpolation Visualization

Demonstrates the two data-augmentation interpolation strategies in `sweeps/data_class.py`.

- **Multi-concentration**: local bilinear spline (kx=1, ky=1) in (time, log c) + distance-weighted k-NN classification.
- **Single-concentration**: quadratic polyfit at c=50 µM + stochastic nearest-time classification."""))

cells.append(code_cell("setup", """import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

sys.path.append('/home/ethan2/GrowthNet/sweeps')
from data_class import PerCompoundDataset

with open('/home/ethan2/GrowthNet/data/splits/Celine_v1/all_compound_metas.pkl', 'rb') as f:
    all_metas = pickle.load(f)

print(f'Loaded {len(all_metas)} CompoundMeta objects')

# Lightweight stub matching deployed config (kx=ky=1)
ds = PerCompoundDataset.__new__(PerCompoundDataset)
ds.rbs_reg = {'kx': 1, 'ky': 1, 's': 0.0}
ds.kx = 1
ds.ky = 1
ds.rng = np.random.default_rng(42)"""))

cells.append(md_cell("md_pick", "## 1. Pick Example Compounds"))

cells.append(code_cell("pick_compounds", """# Multi-conc with at least one active cell
meta_multi = next(
    m for m in all_metas
    if not m.single_conc and (m.pivot_cls == 1).any().any()
)
n_active_multi = int((meta_multi.pivot_cls == 1).sum().sum())
print(f'Multi-conc: {meta_multi.compound}  shape={meta_multi.pivot_od.shape}  active_cells={n_active_multi}')
print(meta_multi.pivot_cls)

# Single-conc, prefer active
meta_single = next(
    (m for m in all_metas if m.single_conc and (m.pivot_cls == 1).any().any()),
    next(m for m in all_metas if m.single_conc)
)
n_active_single = int((meta_single.pivot_cls == 1).sum().sum())
print()
print(f'Single-conc: {meta_single.compound}  shape={meta_single.pivot_od.shape}  active_cells={n_active_single}')
print(meta_single.pivot_cls)

assert not meta_multi.single_conc
assert (meta_multi.pivot_cls == 1).any().any()
assert meta_single.single_conc
assert np.allclose(meta_single.c_vals, [50.0])
print('\\nAll assertions passed.')"""))

cells.append(md_cell("md_multi", """## 2. Multi-Concentration Interpolation

**Top-left**: Observed OD at each (time, conc) grid cell. Gold borders = active cells.

**Top-right**: Interpolated OD surface — local bilinear spline on 2×2 nearest (t, log c) neighbours. Dots = observed OD.

**Bottom-left**: Observed binary class labels (gold=active, slategray=inactive).

**Bottom-right**: p(active) from distance-weighted k-NN (k=4) in (t, log c, OD) space. Dashed white = 0.5 boundary. Diamonds = random augmentation samples."""))

cells.append(code_cell("multi_grid", """# Dense evaluation grid: 80t x 60c (log-spaced in concentration)
n_t, n_c = 80, 60
t_grid = np.linspace(meta_multi.t_min, meta_multi.t_max, n_t)
c_grid = np.exp(np.linspace(meta_multi.logc_min, meta_multi.logc_max, n_c))
logc_grid = np.log(c_grid)

od_grid = np.full((n_t, n_c), np.nan)
p_active_grid = np.full((n_t, n_c), np.nan)

for i, t in enumerate(t_grid):
    for j, c in enumerate(c_grid):
        od_hat, _, p_a = ds._interpolate_multiple_conc(
            od_pivot=meta_multi.pivot_od,
            t_vals=meta_multi.t_vals,
            c_vals=meta_multi.c_vals,
            t_samp=t,
            c_samp=c,
            labels_pivot=meta_multi.pivot_cls,
            k=4,
        )
        od_grid[i, j] = od_hat
        p_active_grid[i, j] = p_a

assert not np.isnan(od_grid).any(), 'NaNs in od_grid'
assert np.all((p_active_grid >= 0) & (p_active_grid <= 1)), 'p_active out of [0,1]'
print(f'OD range: [{od_grid.min():.3f}, {od_grid.max():.3f}]')
print(f'p_active range: [{p_active_grid.min():.3f}, {p_active_grid.max():.3f}]')"""))

cells.append(code_cell("multi_plot", """t_obs = meta_multi.t_vals
c_obs = meta_multi.c_vals
od_obs = meta_multi.pivot_od.values
cls_obs = meta_multi.pivot_cls.values

tt_obs, cc_obs = np.meshgrid(t_obs, c_obs, indexing='ij')
od_flat = od_obs.ravel()
cls_flat = cls_obs.ravel()
logcc_flat = np.log(cc_obs.ravel())
tt_flat = tt_obs.ravel()
colors_cls = ['gold' if v == 1 else 'slategray' for v in cls_flat]

# 60 random augmentation samples
rng_aug = np.random.default_rng(0)
n_aug = 60
t_aug = rng_aug.uniform(meta_multi.t_min, meta_multi.t_max, n_aug)
c_aug = np.exp(rng_aug.uniform(meta_multi.logc_min, meta_multi.logc_max, n_aug))
pred_aug = []
for t, c in zip(t_aug, c_aug):
    _, pred, _ = ds._interpolate_multiple_conc(
        od_pivot=meta_multi.pivot_od,
        t_vals=meta_multi.t_vals,
        c_vals=meta_multi.c_vals,
        t_samp=t,
        c_samp=c,
        labels_pivot=meta_multi.pivot_cls,
        k=4,
    )
    pred_aug.append(pred)
pred_aug = np.array(pred_aug)
aug_colors = ['gold' if p == 1 else 'slategray' for p in pred_aug]

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
vmin_od, vmax_od = od_obs.min(), od_obs.max()

# Panel 1: Observed OD heatmap
ax = axes[0, 0]
im = ax.imshow(od_obs, aspect='auto', origin='lower', cmap='viridis',
               vmin=vmin_od, vmax=vmax_od,
               extent=[-0.5, len(c_obs) - 0.5, -0.5, len(t_obs) - 0.5])
for r in range(len(t_obs)):
    for col in range(len(c_obs)):
        val = od_obs[r, col]
        is_act = cls_obs[r, col] == 1
        ax.text(col, r, f'{val:.2f}', ha='center', va='center', fontsize=7,
                color='white' if val < (vmin_od + vmax_od) / 2 else 'black',
                fontweight='bold' if is_act else 'normal')
        if is_act:
            rect = plt.Rectangle((col - 0.5, r - 0.5), 1, 1,
                                  fill=False, edgecolor='gold', lw=2.5)
            ax.add_patch(rect)
ax.set_xticks(range(len(c_obs)))
ax.set_xticklabels([f'{c:.1f}' for c in c_obs], rotation=30)
ax.set_yticks(range(len(t_obs)))
ax.set_yticklabels([f'{t:.2f}' for t in t_obs])
ax.set_xlabel('Concentration (uM)')
ax.set_ylabel('Time (h)')
ax.set_title('Observed OD  (gold border = active)')
plt.colorbar(im, ax=ax, label='OD')

# Panel 2: Interpolated OD surface
ax = axes[0, 1]
cf = ax.contourf(logc_grid, t_grid, od_grid, levels=30, cmap='viridis',
                 vmin=vmin_od, vmax=vmax_od)
ax.scatter(logcc_flat, tt_flat, c=od_flat, cmap='viridis',
           vmin=vmin_od, vmax=vmax_od,
           edgecolors='black', linewidths=0.8, s=80, zorder=5)
ax.set_xlabel('log(Concentration)')
ax.set_ylabel('Time (h)')
ax.set_title('Interpolated OD — local bilinear spline')
plt.colorbar(cf, ax=ax, label='OD')

# Panel 3: Observed class labels
ax = axes[1, 0]
ax.scatter(logcc_flat, tt_flat, c=colors_cls, s=120,
           edgecolors='black', linewidths=0.5, zorder=5)
for label, color in [('active', 'gold'), ('inactive', 'slategray')]:
    ax.scatter([], [], c=color, s=60, edgecolors='black', linewidths=0.5, label=label)
ax.legend(frameon=False)
ax.set_xlabel('log(Concentration)')
ax.set_ylabel('Time (h)')
ax.set_title('Observed class labels')

# Panel 4: p_active surface + augmentation samples
ax = axes[1, 1]
cf2 = ax.contourf(logc_grid, t_grid, p_active_grid, levels=30, cmap='magma',
                  vmin=0, vmax=1)
ax.contour(logc_grid, t_grid, p_active_grid, levels=[0.5],
           colors=['white'], linewidths=1.5, linestyles='--')
ax.scatter(logcc_flat, tt_flat, c=colors_cls, s=80,
           edgecolors='black', linewidths=0.5, zorder=5)
ax.scatter(np.log(c_aug), t_aug, c=aug_colors, s=60, marker='D',
           edgecolors='white', linewidths=0.5, zorder=6)
for label, color in [('aug active', 'gold'), ('aug inactive', 'slategray')]:
    ax.scatter([], [], c=color, marker='D', s=40, label=label)
ax.legend(frameon=False, fontsize=8, title='diamonds=augmented')
ax.set_xlabel('log(Concentration)')
ax.set_ylabel('Time (h)')
ax.set_title('p(active) — k-NN k=4; dashed = 0.5 boundary')
plt.colorbar(cf2, ax=ax, label='p(active)')

fig.suptitle(
    f'Multi-conc interpolation — {meta_multi.compound}  ({n_active_multi} active cells)',
    fontsize=13
)
plt.tight_layout()
plt.savefig('/home/ethan2/GrowthNet/interpolation_multi_conc.png', dpi=150, bbox_inches='tight')
plt.show()"""))

cells.append(md_cell("md_single", """## 3. Single-Concentration Interpolation

**Left**: Quadratic polyfit of OD across 3 observed timepoints at c=50 µM. Dashed line = clamp value (max OD at t > 0).

**Right**: p(active) analytically replicated from `_interpolate_single_conc` — linearly interpolated between two nearest observed labels. At training time, a Bernoulli sample is drawn from p(active)."""))

cells.append(code_cell("single_eval", """# OD: call _interpolate_single_conc on dense t-grid
t_grid_single = np.linspace(0.0, meta_single.t_max, 200)
od_single = np.array([
    ds._interpolate_single_conc(meta_single, t, c_samp=50.0)[0]
    for t in t_grid_single
])

# Classification probability — analytical replication of _interpolate_single_conc logic
cls_series = meta_single.pivot_cls.loc[:, 50.0]
times_at_50 = cls_series.index.values.astype(float)
labels_at_50 = cls_series.to_numpy(dtype=int)

p_active_single = []
for t in t_grid_single:
    idx_sorted = np.argsort(np.abs(times_at_50 - t))
    i1, i2 = idx_sorted[:2]
    t1, t2 = times_at_50[i1], times_at_50[i2]
    l1, l2 = labels_at_50[i1], labels_at_50[i2]
    if l1 == l2:
        p = float(l1)
    else:
        if l1 == 1 and l2 == 0:
            t1, t2 = t2, t1
        dist_total = abs(t2 - t1)
        dist_to_pos = abs(t - t2)
        p = max(0.0, min(1.0, 1.0 - dist_to_pos / dist_total))
    p_active_single.append(p)
p_active_single = np.array(p_active_single)

# Observed values at conc=50
od_obs_single = meta_single.pivot_od.loc[:, 50.0].values
t_obs_single = meta_single.pivot_od.index.values.astype(float)
cls_obs_single = meta_single.pivot_cls.loc[:, 50.0].to_numpy(dtype=int)
later_mask = t_obs_single > 1e-6
clamp_val = float(od_obs_single[later_mask].max())

print(f'OD range: [{od_single.min():.3f}, {od_single.max():.3f}]  clamp={clamp_val:.3f}')
print(f'Observed labels: {dict(zip(t_obs_single, cls_obs_single))}')"""))

cells.append(code_cell("single_plot", """fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: OD quadratic fit
ax = axes[0]
ax.plot(t_grid_single, od_single, color='steelblue', lw=2, label='Quadratic polyfit')
ax.scatter(t_obs_single, od_obs_single, c='tomato', s=100, zorder=5,
           edgecolors='black', linewidths=0.5, label='Observed OD')
ax.axhline(clamp_val, color='gray', linestyle='--', lw=1.2,
           label=f'Clamp = {clamp_val:.2f}')
ax.set_xlabel('Time (h)')
ax.set_ylabel('OD')
ax.set_title('Single-conc OD — quadratic polyfit  (c = 50 uM)')
ax.legend(frameon=False)

# Panel 2: Classification probability
ax = axes[1]
ax.plot(t_grid_single, p_active_single, color='steelblue', lw=2, label='p(active)')
ax.axhline(0.5, color='gray', linestyle='--', lw=1.2, label='Threshold = 0.5')
colors_single = ['gold' if v == 1 else 'slategray' for v in cls_obs_single]
ax.scatter(t_obs_single, cls_obs_single, c=colors_single, s=100, zorder=5,
           edgecolors='black', linewidths=0.5)
for label, color in [('active (1)', 'gold'), ('inactive (0)', 'slategray')]:
    ax.scatter([], [], c=color, s=60, edgecolors='black', linewidths=0.5, label=label)
ax.set_ylim(-0.1, 1.1)
ax.set_xlabel('Time (h)')
ax.set_ylabel('p(active)')
ax.set_title('Single-conc class — stochastic nearest-time rule')
ax.legend(frameon=False)

fig.suptitle(
    f'Single-conc interpolation — {meta_single.compound}  ({n_active_single} active cells)',
    fontsize=13
)
plt.tight_layout()
plt.savefig('/home/ethan2/GrowthNet/interpolation_single_conc.png', dpi=150, bbox_inches='tight')
plt.show()"""))

cells.append(md_cell("md_summary", """## 4. Summary

| Case | OD method | Classification method |
|---|---|---|
| Multi-conc | Local bilinear spline on 2×2 nearest (t, log c) grid | Distance-weighted k-NN (k=4) in (t, log c, OD) space |
| Single-conc | Quadratic polyfit at 3 timepoints, c=50 µM, clamped | Linear p(active) between 2 nearest timepoints; Bernoulli draw |

Augmentation samples drawn uniformly over each compound's (t, log c) domain during training."""))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"}
    },
    "cells": cells
}

with open('/home/ethan2/GrowthNet/interpolation_viz.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print('Written: /home/ethan2/GrowthNet/interpolation_viz.ipynb')
