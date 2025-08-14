#!/usr/bin/env python
# coding: utf-8

# In[3]:

import numpy as np
import pandas as pd
import torch
import wandb
import copy
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

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

def compute_weighted_metric(
    model: nn.Module,
    df: pd.DataFrame,
    cols_rem: list,
    fctn,  # A metric function: (y_true, y_pred) → float
) -> float:
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
            if np.unique(y_true).size < 2:
                stats.loc[c, t] = np.nan
                continue

            X = df.loc[idx].drop(columns=cols_rem).to_numpy()
            X_t = torch.tensor(X, dtype=torch.float32, device=device)
            with torch.no_grad():
                y_pred = model(X_t).cpu().numpy().squeeze()

            stats.loc[c, t] = float(fctn(y_true, y_pred))

    valid = stats.notna()
    weighted_sum   = (stats.where(valid) * counts.where(valid)).sum().sum()
    total_counts   = counts.where(valid).sum().sum()
    return weighted_sum / total_counts if total_counts > 0 else np.nan


def train():
    """Train function for W&B sweeps. Relies on config.normalization being set."""
    wandb.init()
    config = wandb.config
    norm = config.normalization  # Should be exactly "no_corr", "well", or "plate_t12"

    # Build file paths based on this fixed normalization
    train_path = f"/home/ethan2/GrowthCurve/data/train/df_{norm}_train_mad_4.pkl"
    test_path  = f"/home/ethan2/GrowthCurve/data/test/df_{norm}_test_mad_4.pkl"

    try:
        df_train = pd.read_pickle(train_path)
        df_test  = pd.read_pickle(test_path)
    except FileNotFoundError as e:
        raise RuntimeError(f"Could not load files for normalization '{norm}': {e}")

    cols_rem = ['Well','Plate_ID', 'Compound', 'Control_Label', 'Smiles', 'is_Active', 'scaffold','OD']
    X_train = df_train.drop(columns=cols_rem).to_numpy()
    y_train = df_train['OD'].to_numpy().reshape(-1, 1)
    X_test  = df_test.drop(columns=cols_rem).to_numpy()
    y_test  = df_test['OD'].to_numpy().reshape(-1, 1)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor  = torch.tensor(X_test,  dtype=torch.float32)
    y_test_tensor  = torch.tensor(y_test,  dtype=torch.float32)

    model = SimpleMLP(input_dim=X_train_tensor.shape[1], output_dim=1, hidden_dim=config.hidden_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)  

    X_train_tensor = X_train_tensor.to(device)
    y_train_tensor = y_train_tensor.to(device)
    X_test_tensor  = X_test_tensor.to(device)
    y_test_tensor  = y_test_tensor.to(device)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=config.batch_size,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor),
        batch_size=config.batch_size,
        shuffle=False
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    best_val_loss = float("inf")

    patience_counter = 0
    best_model_state = copy.deepcopy(model.state_dict())

    max_epochs = 500
    patience   = config.patience

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # Validate
        model.eval()
        val_running_loss = 0.0

        with torch.no_grad():
            for xb, yb in test_loader:
                val_preds = model(xb)
                val_loss = criterion(val_preds, yb)
                val_running_loss += val_loss.item() * xb.size(0)
        val_loss = val_running_loss / len(test_loader.dataset)

        model.eval()

        with torch.no_grad():
            
            preds_train_full = model(X_train_tensor)   # shape (N_train, 1), float32
            preds_test_full  = model(X_test_tensor)    # shape (N_test,  1), float32

            
            y_tr = y_train_tensor.cpu().double().view(-1)    
            y_te = y_test_tensor.cpu().double().view(-1)    

            p_tr = preds_train_full.cpu().double().view(-1)  
            p_te = preds_test_full.cpu().double().view(-1)


            y_tr_np = y_tr.numpy()
            p_tr_np = p_tr.numpy()
            y_te_np = y_te.numpy()
            p_te_np = p_te.numpy()

            r2_train        = compute_weighted_metric(model, df_train, cols_rem, r2_np)
            r2_val          = compute_weighted_metric(model, df_test, cols_rem, r2_np)
            pearson_train   = compute_weighted_metric(model, df_train, cols_rem, pearson_np)
            pearson_val     = compute_weighted_metric(model, df_test, cols_rem, pearson_np)
            spearman_train  = compute_weighted_metric(model, df_train, cols_rem, spearman_np)
            spearman_val    = compute_weighted_metric(model, df_test, cols_rem, spearman_np)




        wandb.log({
            "hidden_dim":  config.hidden_dim,
            "epoch":         epoch + 1,
            "train_loss":    train_loss,
            "test_loss":     val_loss,
            "patience":      patience,
            "learning_rate": config.learning_rate,
            "batch_size":    config.batch_size,
            "normalization": norm,  
            "r2_train":        r2_train,
            "pearson_train":   pearson_train,
            "spearman_train":  spearman_train,
            "r2_val":          r2_val,
            "pearson_val":     pearson_val,
            "spearman_val":    spearman_val,
        })

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} (patience={patience})")
                break

    # Save best model to disk
    model.load_state_dict(best_model_state)
    model_filename = f"best_model_{norm}_{wandb.run.id}.pt"
    output_dir = f"/home/ethan2/GrowthCurve/experiments/normalizations/{norm}"
    torch.save(model.state_dict(), f"{output_dir}/{model_filename}")

    art = wandb.Artifact(
        name="best_model",
        type="model",
        description=f"Best model trained on '{norm}' normalization",
    )
    art.add_file(f"{output_dir}/{model_filename}")
    art.metadata = dict(wandb.config)
    wandb.log_artifact(art)
    wandb.finish()


# Sweep config for “no_corr” (only sample patience / lr / batch_size; normalization is fixed)
sweep_no_corr = {
    "method": "bayes",
    "metric": {"name": "test_loss", "goal": "minimize"},
    "parameters": {
        "hidden_dim":   {"value": 256},  # Fixed hidden dimension
        "patience":      {"values": [10]},
        "learning_rate": {"distribution": "log_uniform_values", "min": 5e-4, "max": 1e-2},
        "batch_size":    {"values": [128]},
        # Fix normalization to “no_corr” so that every run in this sweep uses the no‐correction dataset
        "normalization": {"value": "no_corr"},
    }
}

# Sweep config for “well” (fix normalization to well‐corrected)
sweep_well = {
    "method": "bayes",
    "metric": {"name": "test_loss", "goal": "minimize"},
    "parameters": {
        "hidden_dim":   {"value": 256},
        "patience":      {"values": [10]},
        "learning_rate": {"distribution": "log_uniform_values", "min": 5e-4, "max": 1e-2},
        "batch_size":    {"values": [128]},
        # Force every run to use “well” normalization
        "normalization": {"value": "well"},
    }
}

# Sweep config for “plate_t12”
sweep_plate_t12 = {
    "method": "bayes",
    "metric": {"name": "test_loss", "goal": "minimize"},
    "parameters": {
        "hidden_dim":   {"value": 256},
        "patience":      {"values": [10]},
        "learning_rate": {"distribution": "log_uniform_values", "min": 5e-4, "max": 1e-2},
        "batch_size":    {"values": [128]},
        # Force every run to use “plate_t12” normalization
        "normalization": {"value": "plate_t12"},
    }
}

if __name__ == "__main__":

    # Log in to W&B (you can also rely on W&B’s CLI login in your env)
    wandb.login(key="de72b97eb2e03a1787b54e0a865d70bd01be94bb")

    # Create sweep for “no_corr” dataset
    sweep_id_no = wandb.sweep(
        sweep_no_corr,
        project="GrowthCurve Normalization Validation",    # <-- same project     # optional if you work under an org
    )

    # Agent for no_corr sweep (e.g. run 50 runs)
    wandb.agent(sweep_id_no, function=train, count=25)

    # Create sweep for “well” dataset
    sweep_id_well = wandb.sweep(
        sweep_well,
        project="GrowthCurve Normalization Validation",    # <-- same project

    )
    wandb.agent(sweep_id_well, function=train, count=25)

    # Create sweep for “plate_t12” dataset
    sweep_id_plate_t12 = wandb.sweep(
        sweep_plate_t12,
        project="GrowthCurve Normalization Validation",    # <-- same project

    )
    wandb.agent(sweep_id_plate_t12, function=train, count=25)

