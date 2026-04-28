# Pearson NaN Failure in Sweep — Diagnosis & Recommended Fix

## Context

After 50+ epochs across multiple runs, `val_main/AP+Pearson-5*MAE` is NaN, which silently kills your sweep:

- `val_main/best_agg_metric` stays at `-inf` because `agg > best_val_main_agg` returns False when `agg` is NaN.
- `_eval_agg_on_dict` (test eval) never runs — it's gated on val improving.
- `ModelCheckpoint(monitor="val_main/AP+Pearson-5*MAE")` never saves.
- The Bayes sweep has no continuous signal to optimize over.

The user wants a written response saved as markdown explaining whether to fix this in `make_splits.py` or in the metric. **Action: this file IS the response; on plan approval, copy it to `~/GrowthNet/pearson_nan_analysis.md`.**

---

## Diagnosis

The aggregate `AP - 5*MAE + Pearson` is computed per-(t, c) subgroup, then averaged across subgroups. A single NaN in any subgroup's Pearson poisons the mean → poisons the aggregate → kills the entire pipeline.

`pearson_corrcoef` returns NaN in three cases:
1. **n_actives = 1** in the subgroup — Pearson is undefined on a single point.
2. **`std(r_pred[active_mask]) = 0`** — model predicts the same value for every active in the cell. Common at epoch 0 (un-broken symmetry) or in cells with 2-3 actives where prediction collapse is statistically likely.
3. **`std(r_true[active_mask]) = 0`** — all actives in the cell have the same target OD. Less common but possible at boundary (t, c) cells.

Your val/test cells with 1, 2, 3, or 6 actives are exactly the at-risk population. n=1 fails by definition; n=2–3 is mathematically defined but fragile; n=6 is usually fine once the model differentiates but vulnerable early.

The current code's guards don't catch these:
- `on_validation_epoch_end` (line 383) and `_eval_agg_on_dict` (line 239): guard is only `active_mask.any()` → only protects against zero actives.
- `on_train_epoch_end` (line 294): guard is `active_mask.sum() > 1` → protects against single actives, but not against constant predictions when n ≥ 2.

---

## Don't fix this in `make_splits.py`

Tempting to constrain "every (t, c) cell in val/test has ≥ N actives," but this is wrong:

- **Active counts per (t, c) are a property of the data, not the split.** Most compounds aren't active at low concentration or early time. You can't inflate cells like (t=2.08, c=0.2) without distorting the natural class imbalance.
- **Constraining the split would hide signal you need.** Sparse-active cells are part of the deployment reality. The model needs to handle them; the metric needs to tolerate them.
- **Compounding constraints break the optimizer.** You're already balancing actives count, scaffold integrity, mean strength, and median Tanimoto target in the MC sampler. Adding per-(t, c) constraints would over-determine the search.

The split is fine. The metric is the problem.

---

## Recommended fix: make the metric robust

Apply three layered changes to **all three** evaluation paths in `sweeps/sweep_multihead_lightning.py`:

- `_eval_agg_on_dict` (line 211, used for test)
- `on_validation_epoch_end` (line 359, val + per-conc slices)
- `on_train_epoch_end` (line 284, train)

### Change 1: Minimum-actives threshold for Pearson and MAE

Only compute regression metrics on cells with `active_mask.sum() >= N_MIN`. Below threshold, skip the cell from the Pearson/MAE average entirely (don't append a row).

**Recommended `N_MIN_ACTIVES_FOR_REGRESSION = 5`.** Cells with 1–4 actives carry no meaningful regression signal anyway — Pearson on n=2 is ±1 noise — so excluding them improves the metric's signal-to-noise as well as fixing NaN.

**AP/AUC/F1/recall** can still be computed on those cells over all compounds (they work fine with sparse positives), so don't drop the cell from the classification metrics.

### Change 2: Variance guard

Even with `n ≥ N_MIN`, guard against zero-variance edge cases:

```python
if active_mask.sum() >= N_MIN_ACTIVES_FOR_REGRESSION:
    pred_act = r_pred[active_mask]
    true_act = r_true[active_mask]
    if pred_act.std() > 1e-6 and true_act.std() > 1e-6:
        pearson = pearson_corrcoef(pred_act, true_act)
        mae     = mean_absolute_error(pred_act, true_act)
    else:
        pearson = None  # signal to skip in the aggregate
        mae     = None
else:
    pearson = None
    mae     = None
```

### Change 3: Skip None entries in the cross-subgroup average

In the `log_slice` aggregation, filter out cells where Pearson/MAE were skipped, and average over only the surviving cells:

```python
pear_vals = [r[1] for r in rows if r[1] is not None]
mae_vals  = [r[0] for r in rows if r[0] is not None]
pear_m = sum(pear_vals) / len(pear_vals) if pear_vals else 0.0
mae_m  = sum(mae_vals) / len(mae_vals)  if mae_vals  else 0.0
```

If *no* cell has enough actives (extremely unlikely once you've trained for an epoch), Pearson and MAE both default to 0 and the aggregate degrades to AP only — still well-defined, never NaN.

### Belt-and-suspenders: NaN sweep

After all guards, do a final `torch.isnan` replacement before logging. Any leftover NaN becomes 0. This protects against future torchmetrics changes or edge cases not anticipated above.

```python
agg = ap_m - 5 * mae_m + pear_m
if torch.isnan(torch.tensor(agg)):
    agg = -1e6  # sentinel that is finite and clearly bad, lets best-tracking still work
```

(The sentinel is so the sweep can still see *some* signal from a fully-degenerate epoch instead of treating it as "no information." If you'd rather not bias toward bad-but-finite, use 0.0.)

---

## Why this gives Bayes a workable signal

After the fix:
- Aggregate is always finite from epoch 0.
- Sparse-active cells contribute to AP/AUC (where they have signal) but not to Pearson/MAE (where they don't).
- Best-tracking, test eval, and checkpointing all work normally.
- The Bayes optimizer sees a continuous metric with no NaN black holes.

---

## Alternative considered: pooled Pearson

Replace the per-cell average for Pearson with a single global pearson over **all actives across all subgroups**:

```python
all_pred = torch.cat([T_subgroup_pred[mask] for ...])
all_true = torch.cat([T_subgroup_true[mask] for ...])
pearson_global = pearson_corrcoef(all_pred, all_true) if all_pred.std() > 1e-6 else 0.0
```

- **Pro**: Naturally weights cells by active count; only fails when ALL actives across all cells are predicted constant — essentially impossible after a few steps.
- **Con**: Changes the metric's semantics. Pearson at high concentration (where most actives live) dominates the term. Previous sweep results aren't comparable.

**Recommend the threshold-and-guard approach** over pooled Pearson because:
1. You're mid-sweep and want continuity with prior runs.
2. The per-cell averaging philosophy is intentional — it weights every (t, c) condition equally rather than letting high-active conditions dominate.
3. The sparse-cell problem is solved equally well by just excluding sparse cells from the regression term.

---

## Critical files

`/home/ethan2/GrowthNet/sweeps/sweep_multihead_lightning.py`:
- Add module-level constant `N_MIN_ACTIVES_FOR_REGRESSION = 5`
- Update `_eval_agg_on_dict` (line 211–243): apply Changes 1 + 2; aggregate using the None-skipping pattern
- Update `on_validation_epoch_end` (line 359–431): apply Changes 1 + 2; update `log_slice` to use None-skipping for `pear_m` and `mae_m`
- Update `on_train_epoch_end` (line 284–320): tighten guard from `> 1` to `>= N_MIN_ACTIVES_FOR_REGRESSION`; add variance check; use None-skipping in the average

No changes to `make_splits.py`.

---

## Verification

1. **Resume an interrupted run with the fix applied** — `val_main/AP+Pearson-5*MAE` should be finite from epoch 0.
2. **Check `val_main/best_agg_metric`** — should update from `-inf` to the first epoch's value within one epoch, then climb monotonically.
3. **Check `test_main/best_agg_metric`** — should also leave `-inf` once val improves.
4. **Inspect a checkpoint file** — `models/final_sweep/checkpoints/<run_id>/best_params.ckpt` should now exist after a few epochs.
5. **Sanity-check the sweep dashboard** — Bayes optimizer should show a real Pareto front instead of all-NaN runs.

If you also want to verify the threshold isn't dropping too many cells, add a one-time print at the end of the first val epoch:

```python
n_total = len(subgroup_results)
n_used  = sum(1 for r in rows if r[1] is not None)
print(f"[val_main] Pearson computed on {n_used}/{n_total} (t,c) subgroups")
```

If `n_used / n_total < 0.5`, consider lowering `N_MIN_ACTIVES_FOR_REGRESSION` to 3.
