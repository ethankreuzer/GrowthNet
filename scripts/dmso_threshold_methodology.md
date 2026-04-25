# DMSO-Derived Activity Thresholds for the DR Dataset

## Background and Motivation

The DR dataset (`df_Enamine_DR_Growthcurves`) is a dose-response re-assay of compounds from a prior screen that appeared likely to be active. Because no DMSO negative controls were plated alongside these compounds, batch correction was performed using the test compounds themselves as a reference: for each plate, the median OD of compounds labelled "inactive" was divided out and replaced with the global inactive median. This is a multiplicative correction:

```
OD_corrected = OD_raw × global_median_inactive(T, C) / plate_median_inactive(T, C)
```

This correction aligns the inactive population across plates, but it has two consequences that make the internal labels unreliable for downstream classification:

1. **MAD compression.** The multiplicative correction shrinks the spread of the inactive distribution. Empirically, the corrected DR inactive MAD is ~0.47× the DMSO MAD at t=6.24 (measured with DMSO scaled to DR space). The true negative-control spread is roughly twice what the corrected distribution suggests.

2. **Selection bias.** Compounds are labelled inactive using a one-sided threshold (OD > median − k×MAD). This truncates the left tail, biasing the inactive median upward. When a substantial fraction of compounds are genuinely active (as expected here), this bias is non-negligible.

A separate DMSO control dataset (`df_cntrl`, `Control_growthcurves.csv`) is available from a different experiment at the same timepoints and concentrations as DR. This dataset has its own uncorrected batch effects and cannot be directly compared to DR in absolute OD terms. The goal of this section is to **rescale the DMSO distribution into DR's OD scale** so that it can be used to set biologically grounded activity thresholds.

---

## Approach: Multiplicative Rescaling

Because DR's own correction is multiplicative, the consistent mapping for DMSO is also multiplicative, applied per (Timepoint, Concentration) grid point:

```
DMSO_scaled(T, C) = DMSO(T, C) × m_DR_inactive(T, C) / m_DMSO(T, C)
```

where:
- `m_DR_inactive(T, C)` = median OD of currently-labelled inactive compounds in the corrected+averaged DR frame
- `m_DMSO(T, C)` = median OD of paired DMSO at that grid point

This places the DMSO distribution so its median coincides with the DR inactive median at each (T, C). Because CV (= MAD/median) is scale-invariant under multiplication, the threshold simplifies to:

```
threshold(T, C) = m_DR_inactive(T, C) × (1 − k × CV_DMSO(T, C))
```

where `CV_DMSO(T, C) = MAD_DMSO(T, C) / median_DMSO(T, C)` and **k = 3**.

A compound at (T, C) is labelled **active** if its averaged OD < threshold(T, C). T=0 is excluded from activity calling (it is the baseline timepoint and was also skipped by the well-level correction).

---

## Diagnostic 1: Is the Multiplicative Assumption Valid?

### Purpose

If the ratio `m_DR_inactive(T, C) / m_DMSO(T, C)` were constant across timepoints within a concentration, a single scalar would map DMSO to DR space. If it varies, per-(T, C) rescaling is required and the simplified multiplicative assumption is only partially valid.

### Results

```
Concentration  0.200   0.781   3.130   12.500  50.000
Timepoint
2.08           1.655   1.657   1.587   1.668   1.536
4.16           1.214   1.226   1.207   1.212   1.160
6.24           1.149   1.242   1.177   1.167   1.073
8.32           1.116   1.229   1.155   1.146   0.974
10.40          1.042   1.159   1.011   1.057   0.934
12.48          1.023   1.146   1.032   1.070   0.908
```

(Columns 1.200 and 7.900 µM appear in the DMSO dataset but not in DR, hence NaN.)

### Interpretation

The ratio is **not constant**. It ranges from ~1.5–1.7 at t=2.08 down to ~0.9–1.1 at t=12.48. Several factors explain this:

- **Early timepoints** (t=2.08): Cells have barely grown. Both DR inactive and DMSO OD values are small (~0.13–0.22). At low OD, additive background noise (medium, plate reader offset) dominates. The DR inactive median is ~60% higher than DMSO at t=2 partly because the DR inactive population is already slightly selected-against (early batch correction artefact).
- **Late timepoints** (t=10–12): OD is driven by cell mass. The ratio approaches 1, meaning DMSO and DR inactives grow to roughly the same density. The multiplicative relationship holds better at full growth.

**Takeaway:** A single scale factor is insufficient; per-(T, C) rescaling is necessary. The formula `threshold(T, C) = m_DR_inactive(T, C) × (1 − k × CV_DMSO)` already computes everything per-(T, C) and is therefore appropriate regardless of whether the ratio is constant. No change to the approach is required — this diagnostic confirms we cannot simplify further.

---

## Diagnostic 2: Does DMSO OD Vary Across the Concentration Axis?

### Purpose

The DMSO dataset has a Concentration column. DMSO is a vehicle control — it does not change with the "concentration" it was plated in (unless the volume of DMSO vehicle itself varies, which would be a real dose effect). If DMSO OD is effectively constant across concentrations at each timepoint, the concentration axis is a plate-layout artifact and we can **pool all DMSO wells at each timepoint** to estimate a more stable CV. If it varies materially, we must compute CV per (T, C) separately.

### Results

**DMSO median per (T, C):**
```
Concentration  0.200   0.781   1.200   3.130   7.900   12.500  50.000
Timepoint
2.08           0.1302  0.1294  0.1326  0.1326  0.1310  0.1247  0.1271
4.16           0.4475  0.4404  0.4483  0.4420  0.4381  0.4396  0.4341
6.24           0.5943  0.5478  0.5778  0.5730  0.5691  0.5715  0.5825
8.32           0.7238  0.6535  0.7025  0.6898  0.7190  0.6859  0.7356
10.40          0.8872  0.7893  0.8019  0.8998  0.8059  0.8493  0.8588
12.48          1.0079  0.8903  1.0008  0.9850  0.9274  0.9345  0.9677
```

**DMSO CV (= MAD/median) per (T, C):**
```
Concentration  0.200   0.781   1.200   3.130   7.900   12.500  50.000
Timepoint
2.08           0.045   0.018   0.018   0.053   0.018   0.056   0.018
4.16           0.058   0.058   0.047   0.037   0.064   0.075   0.038
6.24           0.065   0.107   0.101   0.057   0.115   0.066   0.090
8.32           0.118   0.154   0.130   0.054   0.103   0.130   0.021
10.40          0.169   0.130   0.143   0.034   0.093   0.161   0.080
12.48          0.173   0.147   0.047   0.067   0.091   0.118   0.103
```

### Interpretation

**Medians:** Very consistent across concentrations at each timepoint (e.g., at t=6.24 the range is 0.548–0.594, less than 8% spread). This confirms that concentration is a plate-layout artifact — DMSO OD does not depend on the "concentration" column.

**CVs:** Highly variable across concentrations at the same timepoint (e.g., at t=8.32 the CV ranges from 0.021 to 0.154, a 7-fold difference). This is not a biological signal — it is noise from having too few DMSO wells per (T, C) cell to estimate MAD reliably. A single outlier replicate dominates when the per-cell sample size is small.

**Additional concentrations (1.200 and 7.900 µM):** The DMSO dataset contains concentrations that do not exist in DR. These wells can be folded into the pool when computing CV per timepoint, increasing sample size and further stabilizing estimates.

**Takeaway:** Pool all DMSO wells across concentrations at each timepoint (`pool_dmso_across_conc=True`). The medians confirm this is valid (concentration is not a real axis). The CVs confirm this is necessary (per-(T, C) CV estimates are too noisy to trust individually).

### Consequence of not pooling

When CV is computed per-(T, C) without pooling and the iteration is run, the result is:
- t=2.08, C=50 µM: active fraction = **88.6%** (biologically implausible)
- t=2.08, C=0.781 µM: active fraction = **20.1%** (also too high)

These pathological values trace directly to noisy per-(T, C) MAD estimates where one or two outlier DMSO wells artificially inflate the CV, depressing the threshold, and labelling nearly the entire population as active. Pooling eliminates this.

---

## Label Function: `label_with_dmso_threshold`

```python
threshold(T, C) = m_DR_inactive(T, C) × (1 − k × CV_DMSO(T))
```

With `pool_dmso_across_conc=True`, `CV_DMSO` is computed per timepoint only (pooling all concentrations). With `k=3`, the threshold sits 3 scaled-MAD units below the inactive median — this corresponds to the ~0.1% tail of a Gaussian, making the false-positive rate very conservative.

T=0 rows are always assigned `is_Active=0` (baseline; excluded from activity calling because the well-level correction skipped T=0 and OD values are near zero).

---

## Iteration: Resolving the Chicken-and-Egg Problem

### Why iteration is needed

Computing `m_DR_inactive(T, C)` requires knowing which compounds are inactive. But knowing which compounds are inactive requires the threshold. Each depends on the other.

The initial `is_Active` labels (from the prior internal-MAD labelling) are used to seed the first estimate of `m_DR_inactive`. After re-labelling with the DMSO-derived threshold, some compounds change status, which shifts `m_DR_inactive`, which changes the threshold, and so on. Iteration continues until fewer than 0.1% of rows flip between rounds.

### Why it matters for DR specifically

DR is a re-assay of pre-selected hits, so a non-trivial fraction of compounds are genuinely active. If active compounds are included when computing `m_DR_inactive`, the inactive median is pulled downward, which lowers the threshold, which labels even more things active. Iteration converges to a `m_DR_inactive` computed from the true inactives under the final consistent threshold.

### Results

```
iter  n_active  flips_vs_prev  flip_frac
1     569       461            4.99%
2     582       13             0.14%
3     585       3              0.03%    ← converged
```

Convergence at iteration 3 with only 585 total active rows. The jump from seed (internal-MAD labels) to iter 1 accounts for most of the change; subsequent rounds are minor refinements.

---

## Final Results

### Final thresholds (k=3, pooled CV, after iteration)

```
Concentration  0.200   0.781   3.130   12.500  50.000
Timepoint
2.08           0.1922  0.1912  0.1879  0.1867  0.1786
4.16           0.4394  0.4369  0.4316  0.4311  0.4131
6.24           0.5168  0.5145  0.5103  0.5048  0.4745
8.32           0.5167  0.5138  0.5091  0.5033  0.4593
10.40          0.5683  0.5621  0.5583  0.5516  0.4951
12.48          0.6585  0.6516  0.6487  0.6392  0.5650
```

Thresholds are nearly identical across concentrations at each timepoint (expected: pooled CV removes concentration-specific noise). The 50 µM column is consistently lower because `m_DR_inactive` at 50 µM is pulled slightly downward by the larger active fraction at that concentration.

### Final active fractions per (T, C)

```
Concentration  0.200   0.781   3.130   12.500  50.000
Timepoint
0.00           0.000   0.000   0.000   0.000   0.000
2.08           0.011   0.042   0.064   0.163   0.337
4.16           0.011   0.015   0.034   0.106   0.246
6.24           0.011   0.015   0.030   0.080   0.182
8.32           0.004   0.011   0.030   0.076   0.144
10.40          0.008   0.015   0.038   0.080   0.144
12.48          0.011   0.019   0.045   0.091   0.152
```

### Takeaways

1. **Dose-response is coherent.** Activity increases monotonically with concentration across nearly all timepoints. At 0.2 µM only ~1% of compounds are active; at 50 µM up to ~34% are. This is the expected pattern for a dose-response re-assay.

2. **t=2.08 warrants caution.** The active fraction at 50 µM, t=2.08 (33.7%) is higher than at t=4.16 (24.6%), which is atypical — activity normally builds over time as drugs take effect. At t=2.08 the absolute OD values are small (~0.13–0.22), making the threshold more sensitive to noise. These early-timepoint labels are the least reliable and should be treated with appropriate scepticism in downstream modelling.

3. **m_DR_inactive follows a clean growth curve.** Values rise from ~0.21 at t=2 to ~1.03 at t=12 for most concentrations, confirming that the batch correction and the inactive population are well-behaved. The 50 µM values are consistently lower (~0.20 at t=2, ~0.88 at t=12), reflecting the presence of more genuinely active compounds at the highest dose pulling the inactive median down slightly even after iteration.

4. **The labelled frame is `df_dr_labeled`.** This is the `df_Enamine_DR_plate_well_batch_effect_DR_avg` frame with `is_Active` overwritten by the DMSO-derived thresholds. The batch-corrected OD values are unchanged.
