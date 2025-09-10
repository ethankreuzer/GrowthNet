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
from typing import Literal, Callable
from accelerate import Accelerator 
from torch.utils.data import DataLoader, WeightedRandomSampler
from data_class import PerCompoundDataset, ExplicitDataset, custom_collate
import pickle

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    recall_score,
    f1_score,
    mean_absolute_error
)

import subprocess, textwrap
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.cuda.device_count():", torch.cuda.device_count())
print(textwrap.dedent(subprocess.check_output("nvidia-smi -L", shell=True, text=True)))


# ─────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────
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
        # Shared trunk
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

        # Regression head
        reg = []
        prev = trunk_dim
        for _ in range(reg_layers):
            reg += [nn.Linear(prev, reg_hidden), nn.ReLU(inplace=True)]
            prev = reg_hidden
        reg += [nn.Linear(prev, 1)]
        self.reg_head = nn.Sequential(*reg)

        # Classification head
        cls = []
        prev = trunk_dim
        for _ in range(cls_layers):
            cls += [nn.Linear(prev, cls_hidden), nn.ReLU(inplace=True)]
            prev = cls_hidden
        cls += [nn.Linear(prev, 1)]
        self.cls_head = nn.Sequential(*cls)

    def forward(self, x: torch.Tensor):
        features = self.trunk(x)
        reg_out = self.reg_head(features)
        cls_logits = self.cls_head(features)
        return reg_out.squeeze(-1), cls_logits.squeeze(-1)


# ─────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────

def r2_np(y_true, y_pred):
    y_mean = y_true.mean()
    tss = np.sum((y_true - y_mean) ** 2)
    sse = np.sum((y_true - y_pred) ** 2)
    return 0.0 if np.isclose(tss, 0) else 1 - (sse / tss)

def pearson_np(y_true, y_pred):
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    with np.errstate(divide='ignore', invalid='ignore'):
        return float(np.corrcoef(y_true, y_pred)[0,1])

def spearman_np(y_true, y_pred):
    if np.allclose(y_true, y_true.mean()) or np.allclose(y_pred, y_pred.mean()):
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr, _ = spearmanr(y_true, y_pred)
    return float(corr)


def compute_weighted_metric_old(
    model: nn.Module,
    df: pd.DataFrame,
    feature_cols: list[str],
    fctn: Callable[[np.ndarray, np.ndarray], float],
    *,
    target_col: str = "OD",
    head: Literal["reg", "cls"] = "reg",
) -> float:
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

            y_true = df.loc[idx, target_col].to_numpy()

            if head == "cls" and len(np.unique(y_true)) < 2:
                stats.loc[c, t] = np.nan
                counts.loc[c, t] = 0
                continue

            X = df.loc[idx, feature_cols].to_numpy()
            X_t = torch.tensor(X, dtype=torch.float32, device=device)

            with torch.inference_mode():
                reg_out, cls_logits = model(X_t)
                if head == "reg":
                    y_pred_t = reg_out
                elif head == "cls":
                    y_pred_t = torch.sigmoid(cls_logits)
                else:
                    raise ValueError(f"Unknown head: {head!r}")

            y_pred = y_pred_t.cpu().numpy().squeeze()
            stats.loc[c, t] = float(fctn(y_true, y_pred))

    valid        = stats.notna()
    weighted_sum = (stats.where(valid) * counts.where(valid)).sum().sum()
    total_cnt    = counts.where(valid).sum().sum()
    return weighted_sum / total_cnt if total_cnt > 0 else np.nan


def batch_to_tensor(batch: dict, device: torch.device):
    
    if batch["t_fourier"].ndim == 3:       # (N, k, 2*num_fourier) → training
        N, k, _ = batch["t_fourier"].shape
        t_feats = batch["t_fourier"].reshape(N * k, -1)
        c_raw   = batch["c_raw"].reshape(N * k, 1)
        c_log   = batch["c_log"].reshape(N * k, 1)
        y_reg   = batch["y_reg"].reshape(N * k)
        y_cls   = batch["y_cls"].reshape(N * k).float()
        repeats = k
    elif batch["t_fourier"].ndim == 2:     # (N, 2*num_fourier) → testing
        N, _   = batch["t_fourier"].shape
        t_feats = batch["t_fourier"]
        c_raw   = batch["c_raw"].unsqueeze(1)   # (N,1)
        c_log   = batch["c_log"].unsqueeze(1)   # (N,1)
        y_reg   = batch["y_reg"]
        y_cls   = batch["y_cls"].float()
        repeats = 1
    else:
        raise ValueError(f"Unexpected t_fourier shape {batch['t_fourier'].shape}")

    feats = [t_feats, c_raw, c_log]

    for fam in sorted(batch["features_by_family"].keys()):
        feats.append(batch["features_by_family"][fam].repeat_interleave(repeats, dim=0))

    # Ensure all tensors are on the same device
    feats = [f.to(device) for f in feats]

    X = torch.cat(feats, dim=1)
    return X, y_reg.to(device), y_cls.to(device)


def compute_weighted_metric(
   
    model: nn.Module,
    data: dict,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    head: Literal["reg", "cls"] = "reg",
    batch_size: int = 128,
    device: torch.device = torch.device("cpu"),
) -> float:
    
    model.eval()
    all_true, all_pred = [], []

    with torch.inference_mode():
    
        Xb, yb_reg, yb_cls = batch_to_tensor(data, device)  # feature_cols arg is unused now
        reg_out, cls_logits = model(Xb)

        if head == "reg":
            y_true, y_pred = yb_reg.cpu().numpy(), reg_out.cpu().numpy()
        else:  # classification
            y_true = yb_cls.cpu().numpy()
            y_pred = torch.sigmoid(cls_logits).cpu().numpy()

        all_true.append(y_true)
        all_pred.append(y_pred)

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    return metric_fn(y_true, y_pred)



def train():

    accelerator = Accelerator(log_with="wandb")     # NEW

    # rank‑0 creates an online run; others stay silent but get the config
    if accelerator.is_main_process:
        run = wandb.init(project="GrowthCurve MultiHead")
    else:
        run = wandb.init(mode="disabled")
    
    config = run.config  

    seed = config.seed
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    sweep_id = run.sweep_id  

    device = accelerator.device

    # ─── Load & augment DataFrames ─────────────────────────────────────────
    
    df_train = pd.read_pickle("/home/ethan2/GrowthCurve/data/train/df_well_train_mad_4.pkl")
    
    df_test  = pd.read_pickle("/home/ethan2/GrowthCurve/data/test/df_well_test_mad_4.pkl")   
    
    with open("/home/ethan2/GrowthCurve/data/test/dict_test_fourier_k_3.pkl", "rb") as f:
        dict_test = pickle.load(f) 

    with open("/home/ethan2/GrowthCurve/data/test/dict_test_fourier_k_3_conc_0_781.pkl", "rb") as f:
        dict_test_conc_0_781 = pickle.load(f) 

    with open("/home/ethan2/GrowthCurve/data/test/dict_test_fourier_k_3_conc_3_13.pkl", "rb") as f:
        dict_test_conc_3_13 = pickle.load(f)

    with open("/home/ethan2/GrowthCurve/data/test/dict_test_fourier_k_3_conc_12_50.pkl", "rb") as f:
        dict_test_conc_12_50 = pickle.load(f)

    Xte, yte_reg, yte_cls = batch_to_tensor(dict_test, device)



    # ─── Model + Losses + Optimizer ─────────────────────────────────────────
    model = MultiHeadNet(
        input_dim   = Xte.shape[1],
        trunk_layers = config.trunk_layers,
        trunk_dim    = config.trunk_dim,
        reg_layers   = config.reg_layers,
        reg_hidden   = config.reg_hidden,
        cls_layers   = config.cls_layers,
        cls_hidden   = config.cls_hidden,
        dropout_rate = config.dropout_rate,
    ).to(device)

    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.epochs,      # number of epochs for one annealing cycle
            eta_min=config.min_lr    # final lower bound
        )


    

    run_id = wandb.run.id.replace(":", "_")
    best_metric   = -float("inf")          
    best_state    = None

    if accelerator.is_main_process:
        outdir = f"/home/ethan2/GrowthCurve/experiments/Multihead/{sweep_id}"
        os.makedirs(outdir, exist_ok=True)                   
        best_ckpt = os.path.join(outdir,f"multihead_{run_id}_best.pt")


    # ─── Training Loop ─────────────────────────────────────────────────────
    for epoch in range(1, config.epochs + 1):
       
        
        train_ds = PerCompoundDataset(df_train, k=config.samples, seed=None, num_fourier=3)
        num_actives = sum(meta.is_active_at_12_50 for meta in train_ds._metas)
        num_inactives = len(train_ds) - num_actives

        weights=[]

        for meta in train_ds._metas:
            if meta.is_active_at_12_50:
                weights.append(config.active_fraction / num_actives)
            else:
                weights.append((1.0 - config.active_fraction) / num_inactives)
        
        sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)
        
        train_loader = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            sampler=sampler,
            collate_fn=custom_collate
        )

        model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )
        model.train()

        train_reg_loss = 0.0
        train_cls_loss = 0.0

        for batch in train_loader:

            Xb, yb_reg, yb_cls = batch_to_tensor(batch, device)
            optimizer.zero_grad()
            out_reg, out_cls_logits = model(Xb)
            # classification loss on all examples
            loss_cls = bce_loss(out_cls_logits.squeeze(-1), yb_cls)
            #active_mask = (yb_cls == 1) COME BACK TO THIS
            loss_reg = mse_loss(out_reg, yb_reg) #regresssing on everything not just actives, inactive performsnce was too poor
            # total loss
            loss = loss_cls + config.loss_lambda * loss_reg
            accelerator.backward(loss)
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

            # ─── 4) Inactive‐only VAL set ─────────────────────────────────────────
            df_test_inact = df_test[df_test["is_Active"] == 0]

            mae_val_inact = compute_weighted_metric(model, df_test_inact, cols_rem, mean_absolute_error, head="reg")
            r2_val_inact      = compute_weighted_metric(model, df_test_inact, cols_rem, r2_np,      head="reg")
            pearson_val_inact = compute_weighted_metric(model, df_test_inact, cols_rem, pearson_np, head="reg")

            # ─── 5)  VAL set Concentratio 0.781─────────────────────────────────────────
            mae_val_0_781 = compute_weighted_metric(model, df_test_conc_0_781, cols_rem, mean_absolute_error, head="reg")
            r2_val_0_781     = compute_weighted_metric(model, df_test_conc_0_781, cols_rem, r2_np,      head="reg")
            pearson_val_0_781 = compute_weighted_metric(model, df_test_conc_0_781, cols_rem, pearson_np, head="reg")

            # ─── 6)  VAL set Concentratio 3.13─────────────────────────────────────────

            mae_val_3_13 = compute_weighted_metric(model, df_test_conc_3_13, cols_rem, mean_absolute_error, head="reg")
            r2_val_3_13     = compute_weighted_metric(model, df_test_conc_3_13, cols_rem, r2_np,      head="reg")
            pearson_val_3_13 = compute_weighted_metric(model, df_test_conc_3_13, cols_rem, pearson_np, head="reg")      

            # ─── 7)  VAL set Concentratio 12.5─────────────────────────────────────────
            mae_val_12_50 = compute_weighted_metric(model, df_test_conc_12_50, cols_rem, mean_absolute_error, head="reg")
            r2_val_12_50     = compute_weighted_metric(model, df_test_conc_12_50, cols_rem, r2_np,      head="reg")
            pearson_val_12_50 = compute_weighted_metric(model, df_test_conc_12_50, cols_rem, pearson_np, head="reg")


            #Goal Metric

            AP_AUC_Pearson_MAE = ap_val+auc_val+pearson_val_act-mae_val_act


            if AP_AUC_Pearson_MAE > best_metric:
                best_metric = AP_AUC_Pearson_MAE
                
                best_state = copy.deepcopy(accelerator.unwrap_model(model).state_dict())
                
                # ‑‑ also save to disk so you keep the best checkpoint even if the job dies ‑‑
                if accelerator.is_main_process:
                    torch.save(best_state, best_ckpt)


        # ─── Single W&B log with everything ─────────────────────────────────────
        
        if accelerator.is_main_process: 
            wandb.log({

                #MODEL ARCHITECTURE
                "seed": config.seed,
                "time_encoding": "fourier",
                "conc_encoding": config.encoding,
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
                "weight_decay": config.weight_decay,

            
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
                "train_ap":       ap_train,
                "train_f1":       f1_train,
                "train_auc":      auc_train,
                "train_recall":   recall_train,

                ### VALIDATION SET
                "val_r2":         r2_val,
                "val_pearson":    pearson_val,
                "val_ap":         ap_val,
                "val_f1":         f1_val,
                "val_auc":        auc_val,
                "val_recall":     recall_val,

                ### ACTIVE‐ONLY VALIDATION
                "val_r2_act":       r2_val_act,
                "mae_val_act": mae_val_act,
                "val_pearson_act":  pearson_val_act,

                ###INACTIVE Validation
                "val_r2_inact":       r2_val_inact,
                "val_pearson_inact":  pearson_val_inact,
                "mae_val_inact": mae_val_inact,


                ### 0.781 Conc validation metrics
                "mae_val_0_781": mae_val_0_781,
                "r2_val_0_781": r2_val_0_781,
                "pearson_val_0_781":pearson_val_0_781,

                ### 3.13 Conc validation metrics
                "mae_val_3_13": mae_val_3_13,
                "r2_val_3_13": r2_val_3_13,
                "pearson_val_3_13":pearson_val_3_13,

                ### 12.50 Conc validation metrics
                "mae_val_12_50": mae_val_12_50,
                "r2_val_12_50": r2_val_12_50,
                "pearson_val_12_50":pearson_val_12_50,

                ###Metric to optimize
                "AP+AUC+Pearson-MAE": AP_AUC_Pearson_MAE,

            })


        scheduler.step()


    # ─── Restore best & Save ──────────────────────────────────────────────
    accelerator.unwrap_model(model).load_state_dict(best_state)


    if accelerator.is_main_process:
        art = wandb.Artifact(
            name=f"multihead_{run_id}_best_model",
            type="model",
            description="Dual‐head MLP: regression + classification"
        )
        art.add_file(best_ckpt)
        art.metadata = dict(wandb.config)
        wandb.log_artifact(art)

    if accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    train()

