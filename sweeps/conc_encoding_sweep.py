#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import torch
import wandb
import copy
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import os
from torch.optim.lr_scheduler import CosineAnnealingLR
from accelerate import Accelerator  

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, dropout):
        super().__init__()

        layers = []

   
        layers.append(nn.Linear(input_dim, hidden_dim))

        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))

            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))


        layers.append(nn.Linear(hidden_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        with torch.no_grad():
            self.model.eval()
            return self.forward(x)

def r2_np(y_true, y_pred):
    y_mean = y_true.mean()
    tss = np.sum((y_true - y_mean) ** 2)
    sse = np.sum((y_true - y_pred) ** 2)
    return 0.0 if np.isclose(tss, 0) else 1 - (sse / tss)

def pearson_np(y_true, y_pred):
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    return np.corrcoef(y_true, y_pred)[0, 1]

def spearman_np(y_true, y_pred):
    if np.allclose(y_true, y_true.mean()) or np.allclose(y_pred, y_pred.mean()):
        return 0.0
    return float(spearmanr(y_true, y_pred)[0])

def apply_concentration_encoding(df, encoding):
    
    max_time = np.log(df['Concentration'].max())

    if encoding == 'raw':
        df['conc_enc'] = df['Concentration']

    elif encoding == 'log':
        df['conc_enc_log'] = np.log(df['Concentration'])

    elif encoding == 'poly':
        df['conc_enc']     = df['Concentration']
        df['conc_squared'] = df['Concentration'] ** 2
        df['conc_cubed']   = df['Concentration'] ** 3

    elif encoding == 'sinusoidal':
        df['conc_sin'] = np.sin(2 * np.pi * np.log(df['Concentration']) / max_time)

    elif encoding == 'fourier':
        for k in range(1, 4):
            df[f'conc_sin_{k}'] = np.sin(2 * np.pi * k * np.log(df['Concentration']) / max_time)
            df[f'conc_cos_{k}'] = np.cos(2 * np.pi * k * np.log(df['Concentration']) / max_time)

    else:
        raise ValueError(f"Unknown encoding type: {encoding}")

    return df


def compute_weighted_metric(model, df, cols_rem, fctn):
    unique_concs = sorted(df['Concentration'].unique())
    unique_times = sorted(df['Timepoint'].unique())
    stats  = pd.DataFrame(index=unique_concs, columns=unique_times, dtype=float)
    counts = pd.DataFrame(index=unique_concs, columns=unique_times, dtype=int)
    model.eval()
    device = next(model.parameters()).device
    for c in unique_concs:
        for t in unique_times:
            mask = (df['Concentration'] == c) & (df['Timepoint'] == t)
            idx = df[mask].index
            n = len(idx)
            counts.loc[c, t] = n
            if n < 2:
                stats.loc[c, t] = np.nan
                continue
            y_true = df.loc[idx, 'OD'].to_numpy()
            X = df.loc[idx].drop(columns=cols_rem).to_numpy()
            X_t = torch.tensor(X, dtype=torch.float32, device=device)
            with torch.no_grad():
                y_pred = model(X_t).cpu().numpy().squeeze()
            stats.loc[c, t] = float(fctn(y_true, y_pred))
    valid = stats.notna()
    weighted_sum = (stats.where(valid) * counts.where(valid)).sum().sum()
    total_counts = counts.where(valid).sum().sum()
    return weighted_sum / total_counts if total_counts > 0 else np.nan

def unweighted_metric(model,df,cols_rem, fctn):
    model.eval()
    device = next(model.parameters()).device

    y_true = df['OD'].to_numpy()
    X = df.drop(columns=cols_rem).to_numpy()
    X_t = torch.tensor(X, dtype=torch.float32, device=device)

    y_pred = model(X_t).cpu().numpy().squeeze()

    metric = float(fctn(y_true,y_pred))
                        
    return metric

def train():

    accelerator = Accelerator(log_with="wandb")     # NEW

    # rank‑0 creates an online run; others stay silent but get the config
    if accelerator.is_main_process:
        run = wandb.init(project="GrowthCurve Concentration Encoding Benchmark")
    else:
        run = wandb.init(mode="disabled")
    
    config = run.config

    sweep_id = run.sweep_id
   
    encoding = config.encoding

    df_train = pd.read_pickle("/home/ethan2/GrowthCurve/data/train/df_well_train_mad_4_undersampled_conc_50_time_6_12.pkl")
    df_test  = pd.read_pickle("/home/ethan2/GrowthCurve/data/test/df_well_test_mad_4_undersampled_conc_50_time_6_12.pkl")

    df_train = df_train[(df_train['Timepoint'].isin([12.48])) & (df_train['Concentration'].isin([0.2, 1.2, 3.13, 7.9, 50]))].reset_index(drop=True)

    df_test  = df_test[df_test['Timepoint'].isin([12.48])].reset_index(drop=True)

    df_train = apply_concentration_encoding(df_train, encoding)
    df_test  = apply_concentration_encoding(df_test, encoding)

    df_test_conc_0_781 = df_test[df_test['Concentration'] == 0.781].reset_index(drop=True)
    df_test_conc_12_50 = df_test[df_test['Concentration'] == 12.5].reset_index(drop=True)
    
    
    
    cols_rem = ['Well','Plate_ID','Compound','Control_Label','Smiles','is_Active','scaffold','OD','Concentration','Timepoint']
    X_train = df_train.drop(columns=cols_rem).to_numpy()
    y_train = df_train['OD'].to_numpy().reshape(-1, 1)
    X_test  = df_test.drop(columns=cols_rem).to_numpy()
    y_test  = df_test['OD'].to_numpy().reshape(-1, 1)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor  = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor  = torch.tensor(y_test, dtype=torch.float32)

    model = SimpleMLP(input_dim=X_train_tensor.shape[1], output_dim=1, num_layers=config.hidden_layers, hidden_dim=config.hidden_dim, dropout=config.dropout)
    device = accelerator.device
    model.to(device)


    X_train_tensor = X_train_tensor.to(device)
    y_train_tensor = y_train_tensor.to(device)
    X_test_tensor = X_test_tensor.to(device)
    y_test_tensor = y_test_tensor.to(device)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=config.batch_size,
        shuffle=True
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.min_lr
    )

    run_id = wandb.run.id.replace(":", "_")
    best_metric   = -float("inf")          
    best_state    = None

    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    if accelerator.is_main_process:
        outdir = f"/home/ethan2/GrowthCurve/experiments/concentration_encoding/{sweep_id}"
        os.makedirs(outdir, exist_ok=True)                   
        best_ckpt = os.path.join(outdir,f"conc_enc_{config.encoding}_{run_id}_best.pt")


    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            accelerator.backward(loss)
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            preds_test = model(X_test_tensor)
            val_loss = criterion(preds_test, y_test_tensor).item()

        with torch.no_grad():

            mae_train = compute_weighted_metric(model, df_train, cols_rem, mean_absolute_error)
            pearson_train = compute_weighted_metric(model, df_train, cols_rem, pearson_np)
            spearman_train = compute_weighted_metric(model, df_train, cols_rem, spearman_np)

            mae_val = compute_weighted_metric(model, df_test, cols_rem, mean_absolute_error)
            pearson_val = compute_weighted_metric(model, df_test, cols_rem, pearson_np)
            spearman_val = compute_weighted_metric(model, df_test, cols_rem, spearman_np)

            mae_val_conc_0_781 = unweighted_metric(model, df_test_conc_0_781,cols_rem, mean_absolute_error)
            pearson_val_conc_0_781 = unweighted_metric(model, df_test_conc_0_781,cols_rem, pearson_np)

            mae_val_conc_12_50 = unweighted_metric(model, df_test_conc_12_50,cols_rem, mean_absolute_error)
            pearson_val_conc_12_50 = unweighted_metric(model, df_test_conc_12_50, cols_rem,pearson_np)

            Pearson_MAE = pearson_val - mae_val

            current_metric = Pearson_MAE

            if current_metric > best_metric:
                best_metric = current_metric
                
                best_state = copy.deepcopy(accelerator.unwrap_model(model).state_dict())
                
                # ‑‑ also save to disk so you keep the best checkpoint even if the job dies ‑‑
                if accelerator.is_main_process:
                    torch.save(best_state, best_ckpt)

        wandb.log({
            "encoding": config.encoding,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "dropout": config.dropout,
            "epoch": epoch + 1,
            "hidden_dim": config.hidden_dim,
            "batch_size": config.batch_size,
            "input_dim": X_train_tensor.shape[1],
            'hidden_dim': config.hidden_dim,
            "num_layers": config.hidden_layers,
            "min_lr": config.min_lr,

            "train_mae": mae_train,
            "train_pearson": pearson_train,
            "train_spearman": spearman_train,
            "train_loss": train_loss,
            
            "val_mae": mae_val,
            "pearson_val": pearson_val,
            "spearman_val": spearman_val,
            "val_loss": val_loss,

            "mae_val conc 0.781" : mae_val_conc_0_781,
            "pearson_val conc 0.781": pearson_val_conc_0_781,
            "mae_val conc 12.50": mae_val_conc_12_50,
            "pearson_val_conc_12_50": pearson_val_conc_12_50,



            "Pearon_val - MAE_val" : Pearson_MAE
            
        })

        scheduler.step()

    accelerator.unwrap_model(model).load_state_dict(best_state)

    if accelerator.is_main_process:
        art = wandb.Artifact(
            name=f"best_model_{wandb.run.id}",
            type="model",
            
        )
        art.add_file(best_ckpt)
        art.metadata = dict(wandb.config)
        wandb.log_artifact(art)

    if accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    
    train()
