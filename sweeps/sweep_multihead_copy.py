#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import torch
import wandb
import copy
import torch.nn as nn
import torch.optim as optim
import os
import warnings
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
import math
from typing import Literal
from typing import Callable
from accelerate import Accelerator  

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    r2_score,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error
)

import os, torch, subprocess, textwrap
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.cuda.device_count():", torch.cuda.device_count())
print(textwrap.dedent(subprocess.check_output("nvidia-smi -L", shell=True, text=True)))


class MultiHeadNet(nn.Module):
    def __init__(self,
                 input_dim: int,
                 trunk_layers: int,
                 trunk_dim: int,
                 reg_layers: int,
                 reg_hidden: int,
                 cls_layers: int,
                 cls_hidden: int,
                 dropout_rate: float):
        super().__init__()
        # ── Shared trunk ─────────────────────────────────────────────
        trunk = []
        prev_dim = input_dim
        for _ in range(trunk_layers):
            trunk += [
                nn.Linear(prev_dim, trunk_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
            ]
            prev_dim = trunk_dim
        self.trunk = nn.Sequential(*trunk)

        # ── Regression head ─────────────────────────────────────────
        reg = []
        prev = trunk_dim
        for _ in range(reg_layers):
            reg += [
                nn.Linear(prev, reg_hidden),
                nn.ReLU(inplace=True),
                #nn.Dropout(dropout_rate),
            ]
            prev = reg_hidden
        reg += [nn.Linear(prev, 1)]
        self.reg_head = nn.Sequential(*reg)

        # ── Classification head ────────────────────────────────────
        cls = []
        prev = trunk_dim
        for _ in range(cls_layers):
            cls += [
                nn.Linear(prev, cls_hidden),
                nn.ReLU(inplace=True),
                #nn.Dropout(dropout_rate),
            ]
            prev = cls_hidden
        cls += [nn.Linear(prev, 1)]
        self.cls_head = nn.Sequential(*cls)

    def forward(self, x: torch.Tensor):
        """
        x: Tensor of shape (N, input_dim)
        Returns:
            reg_out: Tensor of shape (N,)       # regression output
            cls_logits: Tensor of shape (N,)    # pre-sigmoid classification logit
        """
        features = self.trunk(x)
        reg_out = self.reg_head(features)
        cls_logits = self.cls_head(features)
        return reg_out.squeeze(-1), cls_logits.squeeze(-1)
    


def r2_np(y_true, y_pred):
    y_mean = y_true.mean()
    tss = np.sum((y_true - y_mean) ** 2)
    sse = np.sum((y_true - y_pred) ** 2)
    return 0.0 if np.isclose(tss, 0) else 1 - (sse / tss)

def pearson_np(y_true, y_pred):
    # if either vector is constant, return zero
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    # suppress any divide/invalid warnings inside corrcoef
    with np.errstate(divide='ignore', invalid='ignore'):
        return float(np.corrcoef(y_true, y_pred)[0,1])

def spearman_np(y_true, y_pred):
    # if constant, return zero
    if np.allclose(y_true, y_true.mean()) or np.allclose(y_pred, y_pred.mean()):
        return 0.0
    # suppress any underlying warnings from scipy
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr, _ = spearmanr(y_true, y_pred)
    return float(corr)

def compute_weighted_metric(
    model: nn.Module,
    df: pd.DataFrame,
    cols_rem: list[str],
    fctn: Callable[[np.ndarray, np.ndarray], float],
    *,
    target_col: str = "OD",
    head: Literal["reg", "cls"] = "reg",
) -> float:
    """
    Compute a weighted-average metric over (Concentration,Timepoint) cells.
    
    Parameters
    ----------
    model
      A PyTorch model that returns either a Tensor (regression) or a tuple
      (reg_tensor, cls_logits_tensor).
    df
      DataFrame containing 'Concentration', 'Timepoint', and `target_col`.
    cols_rem
      Columns to drop before passing to the model.
    fctn
      A function f(y_true: np.ndarray, y_pred: np.ndarray) -> float
      (e.g. r2_np, roc_auc_score, lambda y,p: f1_score(y, p>0.5), etc.)
    target_col
      Which column in `df` holds the true labels for this metric.
    head
      "reg" → use the model’s first output; "cls" → use the second output + sigmoid.
    """
    unique_concs = sorted(df["Concentration"].unique())
    unique_times = sorted(df["Timepoint"].unique())

    stats  = pd.DataFrame(index=unique_concs, columns=unique_times, dtype=float)
    counts = pd.DataFrame(index=unique_concs, columns=unique_times, dtype=int)

    model.eval()
    device = next(model.parameters()).device

    for c in unique_concs:
        for t in unique_times:
            mask = (df["Concentration"] == c) & (df["Timepoint"] == t)
            idx = df[mask].index
            n = len(idx)
            counts.loc[c, t] = n

            if n < 2:
                stats.loc[c, t] = np.nan
                continue

            # true labels
            y_true = df.loc[idx, target_col].to_numpy()

            if head == "cls" and len(np.unique(y_true)) < 2:
                stats.loc[c, t]   = np.nan   # mark metric undefined
                counts.loc[c, t]  = 0        # drop its samples from denominator
                continue

            # prepare inputs
            X = df.loc[idx].drop(columns=cols_rem).to_numpy()
            X_t = torch.tensor(X, dtype=torch.float32, device=device)

            # forward pass
            with torch.inference_mode():
                out = model(X_t)
                if isinstance(out, tuple):
                    reg_out, cls_logits = out
                else:
                    reg_out, cls_logits = out, None

                if head == "reg":
                    y_pred_t = reg_out
                elif head == "cls":
                    if cls_logits is None:
                        raise ValueError("Model did not return classification logits")
                    y_pred_t = torch.sigmoid(cls_logits)
                else:
                    raise ValueError(f"Unknown head: {head!r}")

            y_pred = y_pred_t.cpu().numpy().squeeze()
            stats.loc[c, t] = float(fctn(y_true, y_pred))

    # now do the same weighting as before
    valid        = stats.notna()
    weighted_sum = (stats.where(valid) * counts.where(valid)).sum().sum()
    total_cnt    = counts.where(valid).sum().sum()
    return weighted_sum / total_cnt if total_cnt > 0 else np.nan



def oversample_actives_to_fraction(df: pd.DataFrame,
                                   fraction: float,
                                   random_state: int | None = None) -> pd.DataFrame:

    out_parts = []
    group_cols = ['Concentration', 'Timepoint']
    
    for (c, t), group in df.groupby(group_cols, sort=False):
        act  = group[group['is_Active'] == 1]
        inact= group[group['is_Active'] == 0]
        n     = len(group)
        k     = len(act)
        
        # if already at or above target (or no actives to sample), leave as is:
        if k == 0 or k / n >= fraction:
            out_parts.append(group)
            continue
        
        # compute how many extra needed so that (k + extra)/(n + extra) = fraction
        extra = math.ceil((fraction * n - k) / (1 - fraction))
        
        # sample `extra` actives with replacement
        extra_act = act.sample(n=extra, replace=True, random_state=random_state)
        
        out_parts.append(pd.concat([group, extra_act], ignore_index=True))
    
    return pd.concat(out_parts, ignore_index=True)



def train():


    run = wandb.init()
    config = wandb.config

    seed = config.seed
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    sweep_id = run.sweep_id  
    

    # ─── Device ─────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ─── Load & augment DataFrames ─────────────────────────────────────────
    df_train = pd.read_pickle("/home/ethan2/GrowthCurve/data/train/df_well_train_mad_4.pkl")

    df_train = oversample_actives_to_fraction(df_train, fraction=config.active_fraction, random_state=42)
    df_test  = pd.read_pickle("/home/ethan2/GrowthCurve/data/test/df_well_test_mad_4.pkl")

    #over sample actvies at every timepoint/concentration

    for df in (df_train, df_test):
        df['raw_conc']     = np.log1p(df['Concentration'])
        df['time']         = df['Timepoint']
        df['time_squared'] = df['Timepoint'] ** 2
        df['time_cubed']   = df['Timepoint'] ** 3

    # ─── Split off features & targets ────────────────────────────────────────
    cols_rem = ['Well','Plate_ID','Compound','Control_Label',
                'Smiles','is_Active','scaffold','OD',
                'Concentration','Timepoint']

    X_train = df_train.drop(columns=cols_rem).to_numpy()
    X_test  = df_test .drop(columns=cols_rem).to_numpy()

    y_train_reg = df_train['OD'].to_numpy().reshape(-1,1)
    y_test_reg  = df_test ['OD'].to_numpy().reshape(-1,1)

    y_train_cls = df_train['is_Active'].astype(float).to_numpy().reshape(-1,1)
    y_test_cls  = df_test ['is_Active'].astype(float).to_numpy().reshape(-1,1)

    # ─── Tensors & DataLoader ───────────────────────────────────────────────
    Xtr = torch.tensor(X_train, dtype=torch.float32, device=device)
    Xte = torch.tensor(X_test,  dtype=torch.float32, device=device)
    ytr_reg = torch.tensor(y_train_reg, dtype=torch.float32, device=device).squeeze(-1)
    yte_reg = torch.tensor(y_test_reg,  dtype=torch.float32, device=device).squeeze(-1)
    ytr_cls = torch.tensor(y_train_cls, dtype=torch.float32, device=device).squeeze(-1)
    yte_cls = torch.tensor(y_test_cls,  dtype=torch.float32, device=device).squeeze(-1)

    train_ds = torch.utils.data.TensorDataset(Xtr, ytr_reg, ytr_cls)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True
    )

    # ─── Model + Losses + Optimizer ─────────────────────────────────────────
    model = MultiHeadNet(
        input_dim   = Xtr.shape[1],
        trunk_layers = config.trunk_layers,
        trunk_dim    = config.trunk_dim,
        reg_layers   = config.reg_layers,
        reg_hidden   = config.reg_hidden,
        cls_layers   = config.cls_layers,
        cls_hidden   = config.cls_hidden,
        dropout_rate = config.dropout_rate,
    ).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = nn.DataParallel(model)

    else:
        print("Using single GPU or CPU")

    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,      # number of epochs for one annealing cycle
        eta_min=config.min_lr    # final lower bound
    )



    # ─── Training Loop ─────────────────────────────────────────────────────
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_reg_loss = 0.0
        train_cls_loss = 0.0

        for Xb, yb_reg, yb_cls in train_loader:
            
            optimizer.zero_grad()
            out_reg, out_cls_logits = model(Xb)
    
            # classification loss on all examples
            loss_cls = bce_loss(out_cls_logits.squeeze(-1), yb_cls)
    
            
            #active_mask = (yb_cls == 1)
            
            loss_reg = mse_loss(out_reg, yb_reg) #regresssing on everything not just actives, inactive performsnce was too poor

            # total loss
            loss = loss_cls + config.loss_lambda * loss_reg

            loss.backward()
            optimizer.step() 

            train_reg_loss += loss_reg.item() * Xb.size(0)
            train_cls_loss += loss_cls.item() * Xb.size(0)

        
        train_reg_loss /= len(train_ds)
        train_cls_loss /= len(train_ds)


        # ─── Validation ─────────────────────────────────────────────────────
        model.eval()
        with torch.inference_mode():

            pred_reg, pred_cls_logits = model(Xte)
            val_reg_loss = mse_loss(pred_reg, yte_reg).item()

            # 2) classification loss on val
            val_cls_loss = bce_loss(pred_cls_logits.squeeze(-1), yte_cls).item()

            # 3) total val loss
            val_loss = val_reg_loss+val_cls_loss

            #val loss on only actives
            active_mask = (yte_cls == 1)
            if active_mask.sum() > 0:
                # regression loss on just those actives
                val_reg_loss_act = mse_loss(
                    pred_reg[active_mask],
                    yte_reg[active_mask]
                ).item()
                # classification loss on just those actives
                val_cls_loss_act = bce_loss(
                    pred_cls_logits.squeeze(-1)[active_mask],
                    yte_cls[active_mask]
                ).item()
            else:
                val_reg_loss_act = 0.0
                val_cls_loss_act = 0.0

            # your new “active‐only total loss”
            val_loss_act = val_reg_loss_act + val_cls_loss_act


            # ─── 1) Entire TRAIN set ─────────────────────────────────────────
            r2_train      = compute_weighted_metric(model, df_train, cols_rem, r2_np,      head="reg")
            pearson_train = compute_weighted_metric(model, df_train, cols_rem, pearson_np, head="reg")
            spearman_train= compute_weighted_metric(model, df_train, cols_rem, spearman_np,head="reg")

            ap_train     = compute_weighted_metric(
                model, df_train, cols_rem, average_precision_score,
                target_col="is_Active", head="cls"
            )
            f1_train     = compute_weighted_metric(
                model, df_train, cols_rem, lambda y,p: f1_score(y, p>0.5),
                target_col="is_Active", head="cls"
            )
            auc_train    = compute_weighted_metric(
                model, df_train, cols_rem, roc_auc_score,
                target_col="is_Active", head="cls"
            )
            recall_train = compute_weighted_metric(
                model, df_train, cols_rem, lambda y,p: recall_score(y, p>0.5),
                target_col="is_Active", head="cls"
            )

            # ─── 2) Entire VAL set ────────────────────────────────────────────
            r2_val      = compute_weighted_metric(model, df_test, cols_rem, r2_np,      head="reg")
            pearson_val = compute_weighted_metric(model, df_test, cols_rem, pearson_np, head="reg")
            spearman_val= compute_weighted_metric(model, df_test, cols_rem, spearman_np,head="reg")

            ap_val     = compute_weighted_metric(
                model, df_test, cols_rem, average_precision_score,
                target_col="is_Active", head="cls"
            )
            f1_val     = compute_weighted_metric(
                model, df_test, cols_rem, lambda y,p: f1_score(y, p>0.5),
                target_col="is_Active", head="cls"
            )
            auc_val    = compute_weighted_metric(
                model, df_test, cols_rem, roc_auc_score,
                target_col="is_Active", head="cls"
            )
            recall_val = compute_weighted_metric(
                model, df_test, cols_rem, lambda y,p: recall_score(y, p>0.5),
                target_col="is_Active", head="cls"
            )

            # ─── 3) Active‐only VAL set ─────────────────────────────────────────
            df_test_act = df_test[df_test["is_Active"] == 1]


            mae_val_act = compute_weighted_metric(model, df_test_act, cols_rem, mean_absolute_error, head="reg")
            r2_val_act      = compute_weighted_metric(model, df_test_act, cols_rem, r2_np,      head="reg")
            pearson_val_act = compute_weighted_metric(model, df_test_act, cols_rem, pearson_np, head="reg")
            spearman_val_act= compute_weighted_metric(model, df_test_act, cols_rem, spearman_np,head="reg")

            # ─── 4) Inactive‐only VAL set ─────────────────────────────────────────
            df_test_inact = df_test[df_test["is_Active"] == 0]

            mae_val_inact = compute_weighted_metric(model, df_test_inact, cols_rem, mean_absolute_error, head="reg")
            r2_val_inact      = compute_weighted_metric(model, df_test_inact, cols_rem, r2_np,      head="reg")
            pearson_val_inact = compute_weighted_metric(model, df_test_inact, cols_rem, pearson_np, head="reg")
            spearman_val_inact= compute_weighted_metric(model, df_test_inact, cols_rem, spearman_np,head="reg")


        # ─── Single W&B log with everything ─────────────────────────────────────
        wandb.log({

            #MODEL ARCHITECTURE
            "seed": config.seed,
            "epochs": config.epochs,
            "loss_lambda": config.loss_lambda,
            "min_lr": config.min_lr,
            "model": "MultiHeadNet",
            "batch_size": config.batch_size,
            "input_dim": Xtr.shape[1],
            "trunk_layers": config.trunk_layers,
            "trunk_dim": config.trunk_dim,
            "reg_layers": config.reg_layers,
            "reg_hidden": config.reg_hidden,
            "cls_layers": config.cls_layers,
            "cls_hidden": config.cls_hidden,
            "dropout_rate": config.dropout_rate,

           
            "initial_lr": config.learning_rate,
            "lr": optimizer.param_groups[0]["lr"],
            "Active_fraction": config.active_fraction,
            "train_reg_loss": train_reg_loss,
            "train_cls_loss": train_cls_loss,

            "val_reg_loss": val_reg_loss,
            "val_cls_loss": val_cls_loss,
            "val_loss": val_loss,
            "val_reg_loss_act": val_reg_loss_act,
            "val_cls_loss_act": val_cls_loss_act,
            "val_loss_act": val_loss_act,
            
            "train_r2":       r2_train,
            "train_pearson":  pearson_train,
            "train_spearman": spearman_train,
            "train_ap":       ap_train,
            "train_f1":       f1_train,
            "train_auc":      auc_train,
            "train_recall":   recall_train,

            ### VALIDATION SET
            "val_r2":         r2_val,
            "val_pearson":    pearson_val,
            "val_spearman":   spearman_val,
            "val_ap":         ap_val,
            "val_f1":         f1_val,
            "val_auc":        auc_val,
            "val_recall":     recall_val,

            ### ACTIVE‐ONLY VALIDATION
            "val_r2_act":       r2_val_act,
            "mae_val_act": mae_val_act,
            "val_pearson_act":  pearson_val_act,
            "val_spearman_act": spearman_val_act,

            ###INACTIVE Validation
            "val_r2_inact":       r2_val_inact,
            "val_pearson_inact":  pearson_val_inact,
            "val_spearman_inact": spearman_val_inact,
            "mae_val_inact": mae_val_inact,


            ###Metric to optimize
            "AP+AUC+Pearson-MAE": ap_val+auc_val+pearson_val_act-mae_val_act

        })

        scheduler.step()


    # ─── Restore best & Save ──────────────────────────────────────────────
    run_id = wandb.run.id.replace(":", "_")
    lr_str = f"{config.learning_rate:.0e}"
    fname  = f"multihead_hl_trunk_{config.trunk_layers}_{config.trunk_dim}_reg_cls_{config.reg_layers}_{config.reg_hidden}_{run_id}.pt"
    outdir = f"/home/ethan2/GrowthCurve/experiments/Multihead/{sweep_id}"
    os.makedirs(outdir, exist_ok=True)

    #model.load_state_dict(torch.load(best_model_path))
    torch.save(model.state_dict(), os.path.join(outdir, fname))

    art = wandb.Artifact(
        name=f"multihead_{run_id}",
        type="model",
        description="Dual‐head MLP: regression + classification"
    )
    art.add_file(os.path.join(outdir, fname))
    art.metadata = dict(wandb.config)
    wandb.log_artifact(art)
    wandb.finish()



sweep_config = {
    "method": "bayes",
    "description": (
        "Bayesian sweep for MultiHead model. Objective is to see if classificationa and regression metrics agree"
        "Model architecture is fixed. Regression and Classification loss, "
        "but regressing on only actives. Removed batchnorm and dropout. "
        "Added lambda, min lr, changed fraction hyper params."
        "Optimizing for metrics, not loss."
        "Using the conc and time encodings that yielded the best val results. Using smaller model capacity, previous run overfit train"
    ),
    "metric": {"name": "AP+AUC+Pearson-MAE", "goal": "maximize"},
    "parameters": {
        "seed": {
            "values": [42]  # keep this fixed for reproducibility
        },

        "loss_lambda": {
            "distribution": "log_uniform_values",
            "min": 0.1,
            "max": 10 
        },

        "min_lr": {
            "distribution": "log_uniform_values",
            "min": 1e-9,
            "max": 1e-6
        },
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-4,
            "max": 1e-2
        },
        "dropout_rate": {
            "distribution": "uniform",
            "min": 0.0,
            "max": 0.5 
        },
        "active_fraction": {
            "distribution": "uniform",
            "min": 0.2,
            "max": 0.70
        },
        "batch_size": {
            "values": [256, 512, 1024, 2048]
        },
        "epochs": {
            "values": [75]
        },

        # fixed architecture parameters
        "trunk_layers": {"values": [3]},
        "trunk_dim":    {"values": [32]},
        "reg_layers":   {"values": [1]},
        "reg_hidden":   {"values": [16]},
        "cls_layers":   {"values": [1]},
        "cls_hidden":   {"values": [16]},
    }
}

if __name__ == "__main__":
    wandb.login(key="de72b97eb2e03a1787b54e0a865d70bd01be94bb")
    sweep_id = wandb.sweep(sweep_config, project="GrowthCurve MultiHead model")
    wandb.agent(sweep_id, function=train, count=100)  # number of runs to execute)

    