"""Reproduce the batch-correction & activity-labelling pipelines from
``normalize_GrowthCurve_2700_cmpds_v2.ipynb`` and ``normalize_DR_dataset.ipynb``,
saving every diagnostic and before/after figure used by
``scripts/batch_correction_methodology.md``.

Run with the project virtualenv:

    source /home/ethan2/venvs/GrowthCurve/bin/activate
    python /home/ethan2/GrowthNet/scripts/generate_methodology_figures.py

Outputs PNGs to ``scripts/figures/`` and prints sanity-check stats matching the
verification cells in the source notebooks.
"""

from __future__ import annotations

import os
import math

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
RAW_DIR = "/home/ethan2/GrowthNet/data/raw"
FIG_DIR = "/home/ethan2/GrowthNet/scripts/figures"
TRAIN_DIR = "/home/ethan2/GrowthNet/data/train"

os.makedirs(FIG_DIR, exist_ok=True)


def _save(name: str) -> None:
    out = os.path.join(FIG_DIR, name)
    plt.savefig(out, bbox_inches="tight", dpi=110)
    plt.close()
    print(f"  wrote {out}")


# --------------------------------------------------------------------------- #
# Functions extracted verbatim (with cosmetic edits) from the two notebooks.
# --------------------------------------------------------------------------- #
def label_inactives_actives(df_long: pd.DataFrame, mad_multiplier) -> pd.DataFrame:
    """Per-(Plate, Concentration, Timepoint): is_Active = 1 iff OD < median(DMSO) - k*MAD(DMSO).
    Falls back to the full group's median/MAD when DMSO rows are absent."""

    def _compute_thresh(group):
        dmso = group.loc[group["Control_Label"] == -1, "OD"]
        if len(dmso) > 0:
            med = dmso.median()
            mad = 1.4826 * np.median(np.abs(dmso - med))
        else:
            med = group["OD"].median()
            mad = 1.4826 * np.median(np.abs(group["OD"] - med))
        return med - mad_multiplier * mad

    thresholds = (
        df_long.groupby(["Plate_ID", "Concentration", "Timepoint"])
        .apply(_compute_thresh)
        .reset_index(name="threshold")
    )
    combined = df_long.drop(columns=["threshold"], errors="ignore").merge(
        thresholds, on=["Plate_ID", "Concentration", "Timepoint"], how="left"
    )
    combined["is_Active"] = (combined["OD"] < combined["threshold"]).astype(int)
    combined.drop(columns=["threshold"], inplace=True)
    return combined


def correct_plate_batch_effect_dmso(df: pd.DataFrame) -> pd.DataFrame:
    """Multiplicative plate correction anchored on DMSO median per (Plate, T, C)."""
    dmso = df[df["Control_Label"] == -1]

    plate_meds = (
        dmso.groupby(["Plate_ID", "Timepoint", "Concentration"])["OD"]
        .median()
        .reset_index(name="plate_dmso_med")
    )
    global_meds = (
        dmso.groupby(["Timepoint", "Concentration"])["OD"]
        .median()
        .reset_index(name="global_dmso_med")
    )

    df_norm = df.merge(plate_meds, on=["Plate_ID", "Timepoint", "Concentration"], how="left")
    df_norm = df_norm.merge(global_meds, on=["Timepoint", "Concentration"], how="left")

    denom = df_norm["plate_dmso_med"].to_numpy()
    numer = df_norm["global_dmso_med"].to_numpy()
    mask = (df_norm["Timepoint"] != 0) & (denom > 0) & np.isfinite(denom) & np.isfinite(numer)
    df_norm.loc[mask, "OD"] = df_norm.loc[mask, "OD"] * numer[mask] / denom[mask]

    return df_norm.drop(columns=["plate_dmso_med", "global_dmso_med"])


def correct_plate_batch_effect_DR(df: pd.DataFrame) -> pd.DataFrame:
    """Multiplicative plate correction anchored on inactive **test** medians per (Plate, T, C).
    For DR plates, which have no on-plate DMSO."""
    inactive = df[df["is_Active"] == 0]

    plate_medians = (
        inactive.groupby(["Plate_ID", "Timepoint", "Concentration"])["OD"]
        .median()
        .reset_index(name="plate_med")
    )
    global_medians = (
        inactive.groupby(["Timepoint", "Concentration"])["OD"]
        .median()
        .reset_index(name="global_med")
    )

    df_norm = df.merge(plate_medians, on=["Plate_ID", "Timepoint", "Concentration"], how="left")
    df_norm = df_norm.merge(global_medians, on=["Timepoint", "Concentration"], how="left")

    denom = df_norm["plate_med"].to_numpy()
    df_norm["OD"] = np.where(
        (denom > 0) & np.isfinite(denom),
        df_norm["OD"] / df_norm["plate_med"] * df_norm["global_med"],
        df_norm["OD"],
    )
    return df_norm.drop(columns=["plate_med", "global_med"])


def correct_well_batch_effect_time_conc(df_long: pd.DataFrame):
    """Multiplicative well correction anchored on inactive medians per (Well, T, C)."""
    inactive = df_long[df_long["is_Active"] == 0]
    well_medians = (
        inactive.groupby(["Well", "Timepoint", "Concentration"])["OD"]
        .median()
        .reset_index(name="well_meds")
    )
    global_medians = (
        inactive.groupby(["Timepoint", "Concentration"])["OD"]
        .median()
        .reset_index(name="global_meds")
    )
    df = df_long.merge(well_medians, on=["Well", "Timepoint", "Concentration"], how="left")
    df = df.merge(global_medians, on=["Timepoint", "Concentration"], how="left")
    mask = df["Timepoint"] != 0
    df.loc[mask, "OD"] = df.loc[mask, "OD"] / df.loc[mask, "well_meds"] * df.loc[mask, "global_meds"]
    df = df.drop(columns=["well_meds", "global_meds"])
    return df, global_medians, well_medians


def iterate_label_and_well_correct(
    df_base: pd.DataFrame,
    mad_multiplier: float = 4,
    max_iters: int = 10,
    tol: float = 0.01,
):
    df_base = df_base.reset_index(drop=True)
    df_cur = label_inactives_actives(df_base, mad_multiplier=mad_multiplier)

    flip_history = []
    for it in range(1, max_iters + 1):
        df_for_correct = df_base.copy()
        df_for_correct["is_Active"] = df_cur["is_Active"].to_numpy()
        df_corrected, _, _ = correct_well_batch_effect_time_conc(df_for_correct)
        df_new = label_inactives_actives(df_corrected, mad_multiplier=mad_multiplier)

        test_mask = df_new["Control_Label"] == 0
        n_test = int(test_mask.sum())
        flips = int(
            (
                df_cur.loc[test_mask, "is_Active"].to_numpy()
                != df_new.loc[test_mask, "is_Active"].to_numpy()
            ).sum()
        )
        frac = flips / max(n_test, 1)
        flip_history.append({"iter": it, "flips": flips, "frac": frac})
        print(f"  iter {it}: flips={flips}  ({frac:.4%} of {n_test} test rows)")

        df_cur = df_new
        if frac < tol:
            print(f"  Converged at iter {it} (flip fraction < {tol:.0%}).")
            return df_cur, flip_history

    print(f"  Stopped at max_iters={max_iters} without hitting the tolerance.")
    return df_cur, flip_history


def iter_cv_label(df_avg: pd.DataFrame, cv_dmso: pd.DataFrame, k: int, max_iters: int = 10, tol: float = 0.001):
    """DR final-labelling iteration via threshold = m_DR_inactive(T,C) * (1 - k*CV_DMSO(T))."""
    df_cur = df_avg.copy().reset_index(drop=True)
    for it in range(1, max_iters + 1):
        m_dr = (
            df_cur[df_cur["is_Active"] == 0]
            .groupby(["Timepoint", "Concentration"])["OD"]
            .median()
            .rename("m_dr_inactive")
            .reset_index()
        )
        thresh = m_dr.merge(cv_dmso[["cv"]], left_on="Timepoint", right_index=True)
        thresh["threshold"] = thresh["m_dr_inactive"] * (1 - k * thresh["cv"])
        merged = (
            df_cur.drop(columns="threshold", errors="ignore")
            .merge(
                thresh[["Timepoint", "Concentration", "threshold"]],
                on=["Timepoint", "Concentration"],
                how="left",
            )
            .reset_index(drop=True)
        )
        new_active = ((merged["OD"] < merged["threshold"]) & (merged["Timepoint"] > 0)).astype(int)
        flips = int((new_active.values != df_cur["is_Active"].values).sum())
        df_cur = merged.drop(columns="threshold")
        df_cur["is_Active"] = new_active.values
        print(f"  CV-iter {it}: flips={flips}")
        if flips / max(len(df_cur), 1) < tol:
            print(f"  Converged at iter {it}.")
            break
    df_cur.loc[df_cur["Timepoint"] == 0, "is_Active"] = 0
    return df_cur


# --------------------------------------------------------------------------- #
# Plotting (function bodies adapted from the notebooks; all return None and
# emit one PNG via matplotlib.savefig — no plt.show in headless mode).
# --------------------------------------------------------------------------- #
def plot_activity_ratio_heatmap(df: pd.DataFrame, out_name: str, suptitle: str | None = None) -> None:
    conc_values = sorted(df["Concentration"].unique())
    time_values = sorted(df["Timepoint"].unique())

    total_counts = (
        df.groupby(["Concentration", "Timepoint"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=conc_values, columns=time_values, fill_value=0)
    )
    active_counts = (
        df[df["is_Active"] == 1]
        .groupby(["Concentration", "Timepoint"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=conc_values, columns=time_values, fill_value=0)
        .astype(int)
    )
    fraction = active_counts.divide(total_counts.replace(0, 1)).fillna(0)
    annot = active_counts.astype(str) + "/" + total_counts.astype(str)

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        fraction,
        annot=annot,
        fmt="",
        cmap="viridis",
        cbar_kws={"label": "Fraction Active"},
    )
    ax.set_xticklabels([str(x) for x in time_values])
    ax.set_yticklabels([str(x) for x in conc_values], rotation=0)
    ax.set_xlabel("Timepoint")
    ax.set_ylabel("Concentration")
    ax.set_title(suptitle or "Active / Total Compounds (Test set)")
    plt.tight_layout()
    _save(out_name)


def _bins_for(label, bins):
    if not isinstance(bins, dict):
        return bins
    return bins.get(label, 30)


def plot_hist_od_distributions_long_neg_ctrl_threshold(
    df_long: pd.DataFrame,
    bins,
    concentration: float,
    max_density: float,
    max_x: float,
    out_name: str,
    title: str = "",
    plot_pos_ctrls: bool = True,
) -> None:
    """OD histograms by Control_Label per timepoint at a fixed concentration.
    Vertical reference lines come from negative-control (Control_Label == -1) median ± k*MAD."""
    base_labels = [-1, 0, 1]
    labels = base_labels if plot_pos_ctrls else [-1, 0]
    colors = {-1: "#1f77b4", 0: "#ff7f0e", 1: "#2ca02c"}
    label_names = {-1: "Negative Control", 0: "Test Compound", 1: "Positive Control"}

    df_sub = df_long[df_long["Concentration"] == concentration]
    timepoints = np.sort(df_sub["Timepoint"].unique())
    n_rows, n_cols = (2, 4) if len(timepoints) > 4 else (1, max(len(timepoints), 1))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for i, tp in enumerate(timepoints):
        ax = axes[i]
        sub_tp = df_sub[df_sub["Timepoint"] == tp]
        for lab in labels:
            grp = sub_tp[sub_tp["Control_Label"] == lab]
            vals = grp["OD"].dropna()
            vals = vals[np.isfinite(vals)]
            ax.hist(
                vals,
                bins=_bins_for(lab, bins),
                alpha=0.30,
                density=True,
                label=label_names[lab],
                color=colors[lab],
                histtype="stepfilled",
                edgecolor="black",
                linewidth=0.7,
            )

        neg_ctrl_grp = sub_tp[sub_tp["Control_Label"] == -1]["OD"].dropna()
        neg_ctrl_grp = neg_ctrl_grp[np.isfinite(neg_ctrl_grp)]
        if len(neg_ctrl_grp) > 0:
            med = neg_ctrl_grp.median()
            mad = 1.4826 * np.median(np.abs(neg_ctrl_grp - med))
        else:
            med = np.nan
            mad = np.nan

        ax.text(
            0.98,
            0.95,
            f"MAD={mad:.4f}" if np.isfinite(mad) else "no DMSO",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", fc="w", ec="0.7", alpha=0.7),
        )
        if np.isfinite(med) and np.isfinite(mad):
            line_styles = [
                ("black", "--", "Median"),
                ("red", ":", "-1 MAD"),
                ("orange", ":", "-2 MAD"),
                ("green", ":", "-3 MAD"),
                ("blue", ":", "-4 MAD"),
            ]
            for k, (col, ls, lbl) in enumerate(line_styles):
                ax.axvline(
                    med - k * mad, color=col, linestyle=ls, linewidth=1.5,
                    label=lbl if i == 0 else None,
                )

        ax.set_title(f"t = {tp} h")
        ax.set_xlabel("OD")
        ax.set_ylabel("Density")
        ax.set_ylim(0, max_density)
        ax.set_xlim(0, max_x)
        ax.grid(True, linestyle="--", alpha=0.4)

    for j in range(len(timepoints), len(axes)):
        fig.delaxes(axes[j])

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower right", title="Legend")
    fig.suptitle(f"OD distributions @ {concentration} µM — {title}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    _save(out_name)


def plot_hist_od_distributions_long(
    df_long: pd.DataFrame,
    bins,
    concentration: float,
    max_density: float,
    max_x: float,
    out_name: str,
    title: str = "",
    plot_pos_ctrls: bool = True,
) -> None:
    """OD histograms with reference lines computed from **test compounds** (Control_Label == 0).
    Used when no DMSO is on plate (DR raw)."""
    base_labels = [-1, 0, 1]
    labels = base_labels if plot_pos_ctrls else [-1, 0]
    colors = {-1: "#1f77b4", 0: "#ff7f0e", 1: "#2ca02c"}
    label_names = {-1: "Negative Control", 0: "Test Compound", 1: "Positive Control"}

    df_sub = df_long[df_long["Concentration"] == concentration]
    timepoints = np.sort(df_sub["Timepoint"].unique())
    n_rows, n_cols = (2, 4) if len(timepoints) > 4 else (1, max(len(timepoints), 1))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for i, tp in enumerate(timepoints):
        ax = axes[i]
        sub_tp = df_sub[df_sub["Timepoint"] == tp]
        for lab in labels:
            grp = sub_tp[sub_tp["Control_Label"] == lab]
            vals = grp["OD"].dropna()
            vals = vals[np.isfinite(vals)]
            ax.hist(
                vals,
                bins=_bins_for(lab, bins),
                alpha=0.30,
                density=True,
                label=label_names[lab],
                color=colors[lab],
                histtype="stepfilled",
                edgecolor="black",
                linewidth=0.7,
            )

        test_grp = sub_tp[sub_tp["Control_Label"] == 0]["OD"].dropna()
        test_grp = test_grp[np.isfinite(test_grp)]
        med = test_grp.median()
        mad = 1.4826 * np.median(np.abs(test_grp - med))

        ax.text(
            0.98,
            0.95,
            f"MAD={mad:.4f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", fc="w", ec="0.7", alpha=0.7),
        )
        line_styles = [
            ("black", "--", "Test median"),
            ("red", ":", "-1 MAD"),
            ("orange", ":", "-2 MAD"),
            ("green", ":", "-3 MAD"),
            ("blue", ":", "-4 MAD"),
        ]
        for k, (col, ls, lbl) in enumerate(line_styles):
            ax.axvline(
                med - k * mad, color=col, linestyle=ls, linewidth=1.5,
                label=lbl if i == 0 else None,
            )

        ax.set_title(f"t = {tp} h")
        ax.set_xlabel("OD")
        ax.set_ylabel("Density")
        ax.set_ylim(0, max_density)
        ax.set_xlim(0, max_x)
        ax.grid(True, linestyle="--", alpha=0.4)

    for j in range(len(timepoints), len(axes)):
        fig.delaxes(axes[j])

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower right", title="Legend")
    fig.suptitle(f"OD distributions @ {concentration} µM — {title}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    _save(out_name)


def _finite(s: pd.Series) -> pd.Series:
    return s[np.isfinite(s)]


def plot_hist_od_distributions_long_split_active(
    df_long: pd.DataFrame,
    bins,
    concentration: float,
    max_density: float,
    max_x: float,
    out_name: str,
    title: str = "",
    plot_pos_ctrls: bool = True,
    plot_neg_ctrls: bool = True,
) -> None:
    """Variant that splits Control_Label==0 into is_Active=={0,1} for post-correction views.

    Histograms shown:
      - DMSO (Control_Label == -1, blue), if plot_neg_ctrls and any present
      - Inactive test (Control_Label == 0 & is_Active == 0, orange)
      - Active test  (Control_Label == 0 & is_Active == 1, red)
      - Cipro / Fosfo (Control_Label == 1, green), if plot_pos_ctrls and any present

    Vertical reference lines (median, -1..-4 MAD) come from DMSO when present;
    otherwise from the inactive test population (best available reference for DR).
    """
    df_sub = df_long[df_long["Concentration"] == concentration]
    timepoints = np.sort(df_sub["Timepoint"].unique())
    n_rows, n_cols = (2, 4) if len(timepoints) > 4 else (1, max(len(timepoints), 1))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    legend_color_inactive = "#ff7f0e"
    legend_color_active = "#d62728"

    for i, tp in enumerate(timepoints):
        ax = axes[i]
        sub_tp = df_sub[df_sub["Timepoint"] == tp]

        if plot_neg_ctrls:
            grp = _finite(sub_tp[sub_tp["Control_Label"] == -1]["OD"].dropna())
            if len(grp) > 0:
                ax.hist(
                    grp,
                    bins=_bins_for(-1, bins),
                    alpha=0.35,
                    density=True,
                    label="Negative Control" if i == 0 else None,
                    color="#1f77b4",
                    histtype="stepfilled",
                    edgecolor="black",
                    linewidth=0.6,
                )

        inactive_grp = _finite(sub_tp[(sub_tp["Control_Label"] == 0) & (sub_tp["is_Active"] == 0)]["OD"].dropna())
        active_grp = _finite(sub_tp[(sub_tp["Control_Label"] == 0) & (sub_tp["is_Active"] == 1)]["OD"].dropna())
        if len(inactive_grp) > 0:
            ax.hist(
                inactive_grp,
                bins=_bins_for(0, bins),
                alpha=0.35,
                density=True,
                label="Inactive test" if i == 0 else None,
                color=legend_color_inactive,
                histtype="stepfilled",
                edgecolor="black",
                linewidth=0.6,
            )
        if len(active_grp) > 0:
            ax.hist(
                active_grp,
                bins=_bins_for(0, bins),
                alpha=0.55,
                density=True,
                label="Active test" if i == 0 else None,
                color=legend_color_active,
                histtype="stepfilled",
                edgecolor="black",
                linewidth=0.6,
            )

        if plot_pos_ctrls:
            grp = _finite(sub_tp[sub_tp["Control_Label"] == 1]["OD"].dropna())
            if len(grp) > 0:
                ax.hist(
                    grp,
                    bins=_bins_for(1, bins),
                    alpha=0.35,
                    density=True,
                    label="Positive Control" if i == 0 else None,
                    color="#2ca02c",
                    histtype="stepfilled",
                    edgecolor="black",
                    linewidth=0.6,
                )

        # Reference: DMSO if present else inactive test
        ref = _finite(sub_tp[sub_tp["Control_Label"] == -1]["OD"].dropna())
        ref_label = "DMSO"
        if len(ref) == 0:
            ref = inactive_grp
            ref_label = "inactive test"
        if len(ref) > 0:
            med = ref.median()
            mad = 1.4826 * np.median(np.abs(ref - med))
            line_styles = [
                ("black", "--", f"{ref_label} median"),
                ("red", ":", "-1 MAD"),
                ("orange", ":", "-2 MAD"),
                ("green", ":", "-3 MAD"),
                ("blue", ":", "-4 MAD"),
            ]
            for k, (col, ls, lbl) in enumerate(line_styles):
                ax.axvline(
                    med - k * mad, color=col, linestyle=ls, linewidth=1.4,
                    label=lbl if i == 0 else None,
                )
            ax.text(
                0.98,
                0.95,
                f"MAD={mad:.4f}\nref: {ref_label}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.25", fc="w", ec="0.7", alpha=0.7),
            )

        ax.set_title(f"t = {tp} h")
        ax.set_xlabel("OD")
        ax.set_ylabel("Density")
        ax.set_ylim(0, max_density)
        ax.set_xlim(0, max_x)
        ax.grid(True, linestyle="--", alpha=0.4)

    for j in range(len(timepoints), len(axes)):
        fig.delaxes(axes[j])

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower right", title="Legend", fontsize=9)
    fig.suptitle(f"OD distributions @ {concentration} µM — {title}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    _save(out_name)


def plot_plate_median_distributions(
    df_long: pd.DataFrame,
    bins,
    concentration: float,
    max_density: float,
    max_x: float,
    out_name: str,
    title: str = "",
) -> None:
    plate_meds = (
        df_long.groupby(["Plate_ID", "Concentration", "Timepoint"], as_index=False)["OD"]
        .median()
        .rename(columns={"OD": "OD_plate_med"})
    )
    df_sub = plate_meds[plate_meds["Concentration"] == concentration]
    timepoints = np.sort(df_sub["Timepoint"].unique())
    n_rows, n_cols = (2, 4) if len(timepoints) > 4 else (1, max(len(timepoints), 1))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for i, tp in enumerate(timepoints):
        ax = axes[i]
        vals = df_sub[df_sub["Timepoint"] == tp]["OD_plate_med"]
        ax.hist(
            vals,
            bins=bins if not isinstance(bins, dict) else bins.get(tp, 30),
            alpha=0.7,
            density=True,
            histtype="stepfilled",
            edgecolor="black",
            linewidth=0.7,
        )
        ax.set_title(f"t = {tp} h  (n = {len(vals)})")
        ax.set_xlabel("Plate-median OD")
        ax.set_ylabel("Density")
        ax.set_xlim(0, max_x)
        ax.set_ylim(0, max_density)
        ax.grid(True, linestyle="--", alpha=0.4)

    for j in range(len(timepoints), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"Plate-median DMSO OD @ {concentration} µM — {title}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    _save(out_name)


def plot_aggregated_well_heatmap(
    df_long: pd.DataFrame,
    timepoint: float,
    out_name: str,
    title: str = "",
    max_cols: int = 3,
    cmap: str = "viridis",
) -> None:
    """Per-well median OD aggregated across plates at a single timepoint, one panel per concentration."""
    concs = sorted(df_long["Concentration"].dropna().unique())

    all_medians = []
    for conc in concs:
        sub = df_long[(df_long["Concentration"] == conc) & (df_long["Timepoint"] == timepoint)]
        med_vals = sub.groupby("Well")["OD"].median().values
        all_medians.extend(med_vals)
    all_medians = np.array(all_medians, dtype=float)
    vmin = float(np.nanmin(all_medians))
    vmax = float(np.nanmax(all_medians))

    n = len(concs)
    ncols = min(max_cols, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).flatten()

    for ax, conc in zip(axes, concs):
        sub = df_long[(df_long["Concentration"] == conc) & (df_long["Timepoint"] == timepoint)]
        med = sub.groupby("Well")["OD"].median().reset_index(name="MedianOD")
        med["Row"] = med["Well"].str[0]
        med["Col"] = med["Well"].str[1:].astype(int)
        heatmap_data = med.pivot(index="Row", columns="Col", values="MedianOD")

        sns.heatmap(
            heatmap_data,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            cbar=False,
            square=True,
        )
        num_plates = sub["Plate_ID"].nunique()
        ax.set_title(f"{conc} µM ({num_plates} plates)")
        ax.set_xlabel("Col")
        ax.set_ylabel("Row")

    for ax in axes[len(concs):]:
        ax.axis("off")

    mappable = axes[0].collections[0]
    fig.colorbar(
        mappable,
        ax=axes.tolist(),
        orientation="vertical",
        fraction=0.02,
        pad=0.04,
        label=f"OD @ t = {timepoint} h",
    )
    fig.suptitle(f"{title}: per-well median OD @ t = {timepoint} h", y=1.02, fontsize=14)
    _save(out_name)


# --------------------------------------------------------------------------- #
# Data loaders
# --------------------------------------------------------------------------- #
def load_growthcurve() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(RAW_DIR, "GrowthCurve_allData.csv"))
    drop_cols = [
        "row", "column", "endOD", "maxOD", "maxOD.t", "expAcc", "lag",
        "statAcc", "stat", "maxR", "maxR.t", "minR", "minR.t",
        "Function", "Structure_class_1", "Structure_class_2", "Target.1",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df = df.rename(columns={"ProductName": "Compound", "Plate": "Plate_ID", "MIC": "Control_Label"})
    df["Control_Label"] = df["Control_Label"].apply(
        lambda x: 1 if x in ["Cipro", "Fosfo"] else (-1 if x == "DMSO" else 0)
    )
    timepoints = sorted(
        [c for c in df.columns if c.startswith("t_")],
        key=lambda s: float(s.split("_")[1]),
    )
    df = df.melt(
        id_vars=["Well", "Plate_ID", "Concentration", "Compound", "Control_Label", "Smiles"],
        value_vars=timepoints,
        var_name="Timepoint",
        value_name="OD",
    )
    df["Timepoint"] = df["Timepoint"].str.replace("t_", "", regex=False).astype(float)
    return df


def load_enamine_t6_t12() -> pd.DataFrame:
    wells = pd.read_csv(os.path.join(RAW_DIR, "Enamine_t6_t12_wells.csv"))
    ctrls = pd.read_csv(os.path.join(RAW_DIR, "Enamine_t6_t12_ctrls.csv"))
    ctrls = ctrls.drop(columns=["Unnamed: 0"], errors="ignore")
    wells = wells.rename(columns={"t_6": "t_6.24", "t_12": "t_12.48"})
    ctrls = ctrls.rename(columns={"t_6": "t_6.24", "t_12": "t_12.48"})

    wells["Concentration"] = 50
    ctrls["Concentration"] = 50
    wells["Control_Label"] = 0
    ctrls["Control_Label"] = ctrls["Compound"].apply(lambda x: 1 if x == "Ciprofloxacin" else -1)
    if "Activity" in wells.columns:
        wells = wells.drop(columns=["Activity"])

    df = pd.concat([ctrls, wells], ignore_index=True)
    timepoints = sorted(
        [c for c in df.columns if c.startswith("t_")],
        key=lambda s: float(s.split("_")[1]),
    )
    df = df.melt(
        id_vars=["Well", "Plate_ID", "Concentration", "Compound", "Replicate", "Control_Label", "Smiles"],
        value_vars=timepoints,
        var_name="Timepoint",
        value_name="OD",
    )
    df["Timepoint"] = df["Timepoint"].str.replace("t_", "", regex=False).astype(float)
    return df


def load_enamine_dr() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(RAW_DIR, "Enamine_DR_growthcurves.csv"))
    df = df.rename(columns={"Plate": "Plate_ID"})
    df["Control_Label"] = 0
    df = df.drop(columns=["MIC"], errors="ignore")
    timepoints = sorted(
        [c for c in df.columns if c.startswith("t_")],
        key=lambda s: float(s.split("_")[1]),
    )
    df = df.melt(
        id_vars=["Well", "Plate_ID", "Concentration", "Compound", "Replicate", "Control_Label", "Smiles"],
        value_vars=timepoints,
        var_name="Timepoint",
        value_name="OD",
    )
    df["Timepoint"] = df["Timepoint"].str.replace("t_", "", regex=False).astype(float)
    return df


def load_dmso_control() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(RAW_DIR, "Control_growthcurves.csv"))
    df["Control_Label"] = df["Compound"].apply(lambda x: -1 if x == "DMSO" else 1)
    timepoints = sorted(
        [c for c in df.columns if c.startswith("t_")],
        key=lambda s: float(s.split("_")[1]),
    )
    df = df.melt(
        id_vars=["Well", "Concentration", "Compound", "Replicate", "Control_Label", "Smiles"],
        value_vars=timepoints,
        var_name="Timepoint",
        value_name="OD",
    )
    df["Timepoint"] = df["Timepoint"].str.replace("t_", "", regex=False).astype(float)
    df["Plate_ID"] = "NA"
    return df


# --------------------------------------------------------------------------- #
# Pipeline runners
# --------------------------------------------------------------------------- #
GC_BINS = {-1: 20, 0: 30, 1: 20}
ENAMINE_BINS = {-1: 20, 0: 25, 1: 20}


def run_growthcurve_pipeline() -> pd.DataFrame:
    print("\n=== GrowthCurve dataset ===")
    df_raw = load_growthcurve()
    print(f"  loaded {len(df_raw):,} rows; {df_raw['Compound'].nunique()} compounds; "
          f"{df_raw['Plate_ID'].nunique()} plates")

    # Diagnostic: per-plate DMSO median spread (motivates plate correction)
    dmso_only = df_raw[df_raw["Control_Label"] == -1]
    plot_plate_median_distributions(
        df_long=dmso_only,
        bins=10,
        concentration=50,
        max_density=15,
        max_x=1.5,
        out_name="gc_plate_dmso_distribution_before.png",
        title="DMSO plate medians before plate correction",
    )

    # Raw activity heatmap (use the same DMSO-MAD k=4 threshold as final)
    df_raw_labeled = label_inactives_actives(df_raw, mad_multiplier=4)
    plot_activity_ratio_heatmap(
        df_raw_labeled[df_raw_labeled["Control_Label"] == 0],
        out_name="gc_activity_heatmap_raw.png",
        suptitle="GrowthCurve raw — fraction active (k=4 DMSO-MAD)",
    )
    for c in [0.2, 50]:
        plot_hist_od_distributions_long_neg_ctrl_threshold(
            df_long=df_raw_labeled,
            bins=GC_BINS,
            concentration=c,
            max_density=8,
            max_x=1.5,
            out_name=f"gc_od_hist_{c}uM_raw.png",
            title=f"GrowthCurve — raw, k=4 DMSO threshold",
        )

    # Stage 1: plate correction
    df_plate = correct_plate_batch_effect_dmso(df_raw.copy())
    df_plate_labeled = label_inactives_actives(df_plate, mad_multiplier=4)
    plot_activity_ratio_heatmap(
        df_plate_labeled[df_plate_labeled["Control_Label"] == 0],
        out_name="gc_activity_heatmap_plate_corrected.png",
        suptitle="GrowthCurve plate-corrected — fraction active",
    )
    for c in [0.2, 50]:
        plot_hist_od_distributions_long_neg_ctrl_threshold(
            df_long=df_plate_labeled,
            bins=GC_BINS,
            concentration=c,
            max_density=8,
            max_x=1.5,
            out_name=f"gc_od_hist_{c}uM_plate_corrected.png",
            title="GrowthCurve — after plate correction, k=4",
        )

    # Diagnostic: per-well median heatmap on plate-corrected data (motivates well correction)
    plot_aggregated_well_heatmap(
        df_long=df_plate,
        timepoint=12.48,
        out_name="gc_well_aggregated_heatmap_before.png",
        title="GrowthCurve plate-corrected",
    )

    # Stage 2: iterative well correction
    df_final, flip_history = iterate_label_and_well_correct(
        df_plate, mad_multiplier=4, max_iters=10, tol=0.01
    )
    plot_activity_ratio_heatmap(
        df_final[df_final["Control_Label"] == 0],
        out_name="gc_activity_heatmap_final.png",
        suptitle="GrowthCurve plate + iterative well — fraction active",
    )
    for c in [0.2, 50]:
        plot_hist_od_distributions_long_split_active(
            df_long=df_final,
            bins=GC_BINS,
            concentration=c,
            max_density=8,
            max_x=1.5,
            out_name=f"gc_od_hist_{c}uM_final_split.png",
            title="GrowthCurve — final, active vs inactive test split",
        )

    print(f"  GrowthCurve final: {len(df_final):,} rows; "
          f"{df_final[df_final['Control_Label']==0]['Smiles'].nunique()} test-compound SMILES")
    return df_final


def run_enamine_pipeline() -> pd.DataFrame:
    print("\n=== Enamine combined dataset (t6_t12 + DR) ===")
    df_t6 = load_enamine_t6_t12()
    df_dr_raw = load_enamine_dr()
    df_cntrl = load_dmso_control()

    # ---- t6_t12 sub-pipeline ------------------------------------------------
    print(f"  t6_t12 raw: {len(df_t6):,} rows; "
          f"{df_t6[df_t6['Control_Label']==0]['Compound'].nunique()} test compounds; "
          f"{df_t6['Plate_ID'].nunique()} plates")

    plot_plate_median_distributions(
        df_long=df_t6[df_t6["Control_Label"] == -1],
        bins=10,
        concentration=50,
        max_density=15,
        max_x=1.5,
        out_name="t6_t12_plate_dmso_distribution_before.png",
        title="t6_t12 DMSO plate medians before plate correction",
    )

    df_t6_raw_labeled = label_inactives_actives(df_t6, mad_multiplier=4)
    plot_activity_ratio_heatmap(
        df_t6_raw_labeled[df_t6_raw_labeled["Control_Label"] == 0],
        out_name="t6_t12_activity_heatmap_raw.png",
        suptitle="t6_t12 raw — fraction active (k=4 DMSO-MAD)",
    )
    plot_hist_od_distributions_long_neg_ctrl_threshold(
        df_long=df_t6_raw_labeled,
        bins=ENAMINE_BINS,
        concentration=50,
        max_density=8,
        max_x=1.5,
        out_name="t6_t12_od_hist_50uM_raw.png",
        title="t6_t12 — raw, k=4 DMSO threshold",
    )

    # Plate correction (DMSO-anchored)
    df_t6_plate = correct_plate_batch_effect_dmso(df_t6.copy())
    df_t6_plate_labeled = label_inactives_actives(df_t6_plate, mad_multiplier=4)
    plot_activity_ratio_heatmap(
        df_t6_plate_labeled[df_t6_plate_labeled["Control_Label"] == 0],
        out_name="t6_t12_activity_heatmap_plate_corrected.png",
        suptitle="t6_t12 plate-corrected — fraction active",
    )
    plot_hist_od_distributions_long_neg_ctrl_threshold(
        df_long=df_t6_plate_labeled,
        bins=ENAMINE_BINS,
        concentration=50,
        max_density=8,
        max_x=1.5,
        out_name="t6_t12_od_hist_50uM_plate_corrected.png",
        title="t6_t12 — after plate correction",
    )

    # Iterative well correction (k=4, tol=0.001)
    print("  Iterating t6_t12 well correction…")
    df_t6_final, _ = iterate_label_and_well_correct(
        df_t6_plate, mad_multiplier=4, max_iters=10, tol=0.001
    )

    # Replicate averaging (test compounds collapse Well; controls keep Well)
    df_test_avg = (
        df_t6_final[df_t6_final["Control_Label"] == 0]
        .groupby(["Plate_ID", "Concentration", "Compound", "Timepoint"], as_index=False)
        .agg(
            Control_Label=("Control_Label", "first"),
            OD=("OD", "mean"),
            Smiles=("Smiles", "first"),
        )
    )
    df_controls_avg = (
        df_t6_final[df_t6_final["Control_Label"] != 0]
        .groupby(["Plate_ID", "Well", "Concentration", "Compound", "Timepoint"], as_index=False)
        .agg(
            Control_Label=("Control_Label", "first"),
            OD=("OD", "mean"),
            Smiles=("Smiles", "first"),
        )
    )
    df_t6_avg = pd.concat([df_test_avg, df_controls_avg], ignore_index=True)
    df_t6_avg = label_inactives_actives(df_t6_avg, mad_multiplier=4)

    plot_activity_ratio_heatmap(
        df_t6_avg[df_t6_avg["Control_Label"] == 0],
        out_name="t6_t12_activity_heatmap_final.png",
        suptitle="t6_t12 plate + well + averaging — fraction active",
    )
    plot_hist_od_distributions_long_split_active(
        df_long=df_t6_avg,
        bins=ENAMINE_BINS,
        concentration=50,
        max_density=8,
        max_x=1.5,
        out_name="t6_t12_od_hist_50uM_final_split.png",
        title="t6_t12 — final, active vs inactive test split",
    )

    # ---- DR sub-pipeline ----------------------------------------------------
    print(f"  DR raw: {len(df_dr_raw):,} rows; "
          f"{df_dr_raw['Compound'].nunique()} compounds; "
          f"{df_dr_raw['Plate_ID'].nunique()} plates")

    df_dr_seed = label_inactives_actives(df_dr_raw, mad_multiplier=2)
    plot_activity_ratio_heatmap(
        df_dr_seed[df_dr_seed["Control_Label"] == 0],
        out_name="dr_activity_heatmap_raw.png",
        suptitle="DR raw — fraction active (k=2 internal-test seed)",
    )
    plot_hist_od_distributions_long(
        df_long=df_dr_seed,
        bins=ENAMINE_BINS,
        concentration=50,
        max_density=8,
        max_x=1.5,
        out_name="dr_od_hist_50uM_raw.png",
        title="DR — raw, k=2 test-MAD threshold",
        plot_pos_ctrls=False,
    )

    # DR plate correction (anchored on inactive test compounds)
    df_dr_plate = correct_plate_batch_effect_DR(df_dr_seed)
    df_dr_plate = label_inactives_actives(df_dr_plate, mad_multiplier=2)
    plot_activity_ratio_heatmap(
        df_dr_plate[df_dr_plate["Control_Label"] == 0],
        out_name="dr_activity_heatmap_plate_corrected.png",
        suptitle="DR plate-corrected (k=2 seeds) — fraction active",
    )
    plot_hist_od_distributions_long(
        df_long=df_dr_plate,
        bins=ENAMINE_BINS,
        concentration=50,
        max_density=8,
        max_x=1.5,
        out_name="dr_od_hist_50uM_plate_corrected.png",
        title="DR — after plate correction (k=2 seeds)",
        plot_pos_ctrls=False,
    )

    # Iterative well correction with k=2 seeds
    print("  Iterating DR well correction (k=2 seeds)…")
    df_dr_iter, _ = iterate_label_and_well_correct(
        df_dr_plate, mad_multiplier=2, max_iters=10, tol=0.001
    )
    plot_activity_ratio_heatmap(
        df_dr_iter[df_dr_iter["Control_Label"] == 0],
        out_name="dr_activity_heatmap_iterated.png",
        suptitle="DR plate + iterated well (k=2 seeds) — fraction active",
    )

    # Replicate average for DR
    df_dr_avg = (
        df_dr_iter.drop(columns="Replicate")
        .groupby(
            ["Plate_ID", "Concentration", "Compound", "Control_Label", "Timepoint"],
            as_index=False,
        )
        .agg(
            OD=("OD", "mean"),
            Smiles=("Smiles", "first"),
            is_Active=("is_Active", "first"),
        )
    )

    # DMSO CV pooled across concentrations per timepoint
    dmso = df_cntrl[df_cntrl["Control_Label"] == -1]
    cv_dmso = dmso.groupby("Timepoint")["OD"].agg(
        median="median",
        mad=lambda x: 1.4826 * np.median(np.abs(x - x.median())),
    )
    cv_dmso["cv"] = cv_dmso["mad"] / cv_dmso["median"]

    print("  DR final relabel via DMSO CV-rescaled threshold (k=4)…")
    df_dr_labeled = iter_cv_label(df_dr_avg, cv_dmso, k=4)
    plot_activity_ratio_heatmap(
        df_dr_labeled,
        out_name="dr_activity_heatmap_final.png",
        suptitle="DR final — fraction active (DMSO CV-rescaled, k=4)",
    )
    plot_hist_od_distributions_long_split_active(
        df_long=df_dr_labeled,
        bins=ENAMINE_BINS,
        concentration=50,
        max_density=8,
        max_x=1.5,
        out_name="dr_od_hist_50uM_final_split.png",
        title="DR — final, active vs inactive test split",
        plot_neg_ctrls=False,
    )

    # ---- Combine ------------------------------------------------------------
    overlap_smiles = set(df_dr_labeled["Smiles"]) & set(df_t6_avg["Smiles"])
    print(f"  Overlap (DR ∩ t6_t12) SMILES: {len(overlap_smiles)}")

    drop_mask = (df_t6_avg["Control_Label"] == 0) & (df_t6_avg["Smiles"].isin(overlap_smiles))
    df_t6_kept = df_t6_avg[~drop_mask].copy()
    df_combined = pd.concat([df_t6_kept, df_dr_labeled], ignore_index=True)

    cipro_mask = df_combined["Control_Label"] == 1
    cipro_rows = df_combined[cipro_mask]
    cipro_valid = cipro_rows[cipro_rows["OD"].notna() & np.isfinite(cipro_rows["OD"])]
    cipro_avg = (
        cipro_valid.groupby(
            ["Smiles", "Compound", "Timepoint", "Concentration", "Control_Label"],
            as_index=False,
        ).agg(OD=("OD", "mean"), is_Active=("is_Active", "max"))
    )
    df_combined = pd.concat([df_combined[~cipro_mask], cipro_avg], ignore_index=True)
    df_combined = df_combined[df_combined["Control_Label"] != -1]

    plot_activity_ratio_heatmap(
        df_combined,
        out_name="combined_activity_heatmap.png",
        suptitle="Combined Enamine output (DMSO removed) — fraction active",
    )

    n_test = df_combined[df_combined["Control_Label"] == 0]["Smiles"].nunique()
    print(f"  combined: {len(df_combined):,} rows; {n_test} unique test-compound SMILES")

    # Sanity numbers reported in §7 of the methodology doc
    overlap_1248_50 = df_combined[
        (df_combined["Smiles"].isin(overlap_smiles))
        & (df_combined["Timepoint"] == 12.48)
        & (df_combined["Concentration"] == 50)
    ]
    nonoverlap_1248_50 = df_combined[
        (~df_combined["Smiles"].isin(overlap_smiles))
        & (df_combined["Control_Label"] == 0)
        & (df_combined["Timepoint"] == 12.48)
        & (df_combined["Concentration"] == 50)
    ]
    print(
        f"  Active rate @ (12.48, 50µM): "
        f"DR-overlap = {overlap_1248_50['is_Active'].mean():.2%} "
        f"({int(overlap_1248_50['is_Active'].sum())}/{len(overlap_1248_50)}); "
        f"t6_t12-only = {nonoverlap_1248_50['is_Active'].mean():.2%} "
        f"({int(nonoverlap_1248_50['is_Active'].sum())}/{len(nonoverlap_1248_50)})"
    )

    return df_combined


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    df_gc = run_growthcurve_pipeline()
    df_combined = run_enamine_pipeline()

    # Cross-check with existing pickles (sanity, not strict equality)
    print("\n=== Comparison vs. existing pickles ===")
    try:
        existing_gc = pd.read_pickle(os.path.join(TRAIN_DIR, "df_GrowthCurve_27000.pkl"))
        gc_test = df_gc[df_gc["Control_Label"] == 0]
        print(
            f"  GrowthCurve: regen test rows = {len(gc_test):,}, "
            f"existing rows = {len(existing_gc):,}, "
            f"unique SMILES regen = {gc_test['Smiles'].nunique()}, "
            f"existing = {existing_gc['Smiles'].nunique()}"
        )
    except FileNotFoundError:
        print("  no existing GrowthCurve pickle — skipped cross-check")
    try:
        existing_combined = pd.read_pickle(os.path.join(TRAIN_DIR, "df_combined_Enamine.pkl"))
        print(
            f"  Combined Enamine: regen rows = {len(df_combined):,}, "
            f"existing rows = {len(existing_combined):,}, "
            f"unique test SMILES regen = "
            f"{df_combined[df_combined['Control_Label']==0]['Smiles'].nunique()}, "
            f"existing = "
            f"{existing_combined[existing_combined['Control_Label']==0]['Smiles'].nunique()}"
        )
    except FileNotFoundError:
        print("  no existing combined Enamine pickle — skipped cross-check")

    n_pngs = len([f for f in os.listdir(FIG_DIR) if f.endswith(".png")])
    print(f"\nWrote {n_pngs} PNGs to {FIG_DIR}")


if __name__ == "__main__":
    main()
