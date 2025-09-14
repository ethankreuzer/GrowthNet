#!/usr/bin/env python
# coding: utf-8
import os, random, warnings, pickle
os.environ.setdefault("WANDB_START_METHOD", "thread")  # safer under DDP

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.distributed as dist
import json
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy


if os.environ.get("RANK", "0") != "0":
    os.environ["WANDB_DISABLED"] = "true"

# (optional but helpful on HPC)
os.environ.setdefault("WANDB__SERVICE_WAIT", "600")
os.environ.setdefault("WANDB_HTTP_TIMEOUT", "300")
os.environ.setdefault("WANDB_DISABLE_CODE", "true")
os.environ.setdefault("WANDB_SILENT", "true")

import wandb
from scipy.stats import spearmanr
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    recall_score,
    f1_score,
)

# ─────────────────────────────────────────────────────────────
# Import your custom datasets and collate
# ─────────────────────────────────────────────────────────────
from data_class import PerCompoundDataset, ExplicitDataset, custom_collate


# --- add imports ---
import argparse

def parse_args():
    p = argparse.ArgumentParser()
    # match the sweep yaml keys
    p.add_argument("--samples", type=int, default=6)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--loss_lambda", type=float, default=5.0)
    p.add_argument("--min_lr", type=float, default=1e-7)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--dropout_rate", type=float, default=0.2)
    p.add_argument("--active_fraction", type=float, default=0.6)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--trunk_layers", type=int, default=5)
    p.add_argument("--trunk_dim", type=int, default=64)
    p.add_argument("--reg_layers", type=int, default=1)
    p.add_argument("--reg_hidden", type=int, default=16)
    p.add_argument("--cls_layers", type=int, default=1)
    p.add_argument("--cls_hidden", type=int, default=16)
    return vars(p.parse_args())


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
        c_raw   = batch["c_raw"].unsqueeze(1)
        c_log   = batch["c_log"].unsqueeze(1)
        y_reg   = batch["y_reg"]
        y_cls   = batch["y_cls"].float()
        repeats = 1
    else:
        raise ValueError(f"Unexpected t_fourier shape {batch['t_fourier'].shape}")

    feats = [t_feats, c_raw, c_log]
    for fam in sorted(batch["features_by_family"].keys()):
        feats.append(batch["features_by_family"][fam].repeat_interleave(repeats, dim=0))
    feats = [f.to(device) for f in feats]

    X = torch.cat(feats, dim=1)
    return X, y_reg.to(device), y_cls.to(device)


# ─────────────────────────────────────────────────────────────
# LIGHTNING MODULE
# ─────────────────────────────────────────────────────────────
class MultiHeadLightning(pl.LightningModule):
    def __init__(self, input_dim, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.model = MultiHeadNet(
            input_dim   = input_dim,
            trunk_layers = config['trunk_layers'],
            trunk_dim    = config['trunk_dim'],
            reg_layers   = config['reg_layers'],
            reg_hidden   = config['reg_hidden'],
            cls_layers   = config['cls_layers'],
            cls_hidden   = config['cls_hidden'],
            dropout_rate = config['dropout_rate'],
        )
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx): #what is batch_idx here?
        Xb, yb_reg, yb_cls = batch_to_tensor(batch, self.device)
        out_reg, out_cls_logits = self(Xb)

        loss_reg = self.mse_loss(out_reg, yb_reg)
        loss_cls = self.bce_loss(out_cls_logits.squeeze(-1), yb_cls)
        loss = loss_cls + self.hparams.loss_lambda * loss_reg

        self.log("train/reg_loss", loss_reg, on_step=True, on_epoch=False, sync_dist=False, prog_bar=False)
        self.log("train/cls_loss", loss_cls, on_step=True, on_epoch=False, sync_dist=False, prog_bar=False)
        self.log("train/loss",     loss,     on_step=True, on_epoch=False, sync_dist=False, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        Xb, yb_reg, yb_cls = batch_to_tensor(batch, self.device)
        pred_reg, pred_cls_logits = self(Xb)

        reg_loss = self.mse_loss(pred_reg, yb_reg)
        cls_loss = self.bce_loss(pred_cls_logits.squeeze(-1), yb_cls)
        loss = reg_loss + cls_loss

        dataset_names = ["val_main", "val_0_781", "val_3_13", "val_12_50"]
        name = dataset_names[dataloader_idx]

        if dataloader_idx == 0:
            # ── Only for main validation set ──
            # Split losses by active/inactive
            active_mask = (yb_cls == 1)
            inactive_mask = ~active_mask

            if active_mask.any():
                reg_loss_act = self.mse_loss(pred_reg[active_mask], yb_reg[active_mask])
            else:
                reg_loss_act = torch.tensor(0.0, device=self.device)

            if inactive_mask.any():
                reg_loss_inact = self.mse_loss(pred_reg[inactive_mask], yb_reg[inactive_mask])
            else:
                reg_loss_inact = torch.tensor(0.0, device=self.device)

            # Log all main metrics
            self.log(f"{name}/reg_loss", reg_loss, prog_bar=True, on_epoch=True, sync_dist=True)
            self.log(f"{name}/cls_loss", cls_loss, prog_bar=True, on_epoch=True, sync_dist=True)
            self.log(f"{name}/loss",     loss,     prog_bar=True, on_epoch=True, sync_dist=True)
            self.log(f"{name}/reg_loss_actives",   reg_loss_act,   prog_bar=False, on_epoch=True, sync_dist=True)
            self.log(f"{name}/reg_loss_inactives", reg_loss_inact, prog_bar=False, on_epoch=True, sync_dist=True)

            # Used by early stopping callbacks etc.
            self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)

        else:
            # ── For the 3 subset dataloaders ──
            self.log(f"{name}/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)

        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.epochs,
            eta_min=self.hparams.min_lr,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


# ─────────────────────────────────────────────────────────────
# DATAMODULE
# ─────────────────────────────────────────────────────────────
class GrowthCurveDataModule(pl.LightningDataModule):
    def __init__(self, config, df_train, dict_val_main, dict_val_0_781, dict_val_3_13, dict_val_12_50):
        super().__init__()
        self.config = config
        self.df_train = df_train
        self.dict_val_main = dict_val_main
        self.dict_val_0_781 = dict_val_0_781
        self.dict_val_3_13 = dict_val_3_13
        self.dict_val_12_50 = dict_val_12_50

    def setup(self, stage=None):
        self.train_ds = PerCompoundDataset(
            self.df_train, k=self.config['samples'], seed=None, num_fourier=3
        )

    def train_dataloader(self):

        rank = int(os.environ.get("RANK", "0"))
        g = torch.Generator()
        g.manual_seed(self.config['seed'] + rank)

        num_actives = sum(meta.is_active_at_12_50 for meta in self.train_ds._metas)
        num_inactives = len(self.train_ds) - num_actives
        weights = [
            (self.config['active_fraction'] / num_actives if meta.is_active_at_12_50
             else (1.0 - self.config['active_fraction']) / num_inactives)
            for meta in self.train_ds._metas
        ]
        sampler = WeightedRandomSampler(weights, num_samples=len(self.train_ds), replacement=True, generator=g)
        
        return DataLoader(self.train_ds,
                          batch_size=self.config['batch_size'],
                          sampler=sampler,
                          collate_fn=custom_collate,
                          shuffle=False,
                          num_workers=32,
                          pin_memory=True)
    def val_dataloader(self):

        class DictDataset(torch.utils.data.Dataset):
            def __init__(self, data_dict): self.data = data_dict
            def __len__(self): return 1
            def __getitem__(self, idx): return self.data

        identity = lambda batch: batch[0]  # don’t stack into a batch

        return [
            DataLoader(DictDataset(self.dict_val_main),  batch_size=1, collate_fn=identity, num_workers=8, pin_memory=True),
            DataLoader(DictDataset(self.dict_val_0_781), batch_size=1, collate_fn=identity, num_workers=8, pin_memory=True),
            DataLoader(DictDataset(self.dict_val_3_13),  batch_size=1, collate_fn=identity, num_workers=8, pin_memory=True),
            DataLoader(DictDataset(self.dict_val_12_50), batch_size=1, collate_fn=identity, num_workers=8, pin_memory=True),
        ]
    


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def is_rank_zero() -> bool:
    return int(os.environ.get("RANK", "0")) == 0

def main():
    # Parse sweep-injected CLI args
    config = parse_args()

    # Reproducibility + matmul perf
    pl.seed_everything(config['seed'], workers=True)
    try:
        torch.set_float32_matmul_precision('medium')
    except Exception:
        pass

    # ── Load data ──
    df_train = pd.read_pickle("/home/ethan2/GrowthCurve/data/train/df_well_train_mad_4.pkl")
    with open("/home/ethan2/GrowthCurve/data/test/dict_test_fourier_k_3.pkl", "rb") as f:
        dict_test = pickle.load(f)
    with open("/home/ethan2/GrowthCurve/data/test/dict_test_fourier_k_3_conc_0_781.pkl", "rb") as f:
        dict_test_conc_0_781 = pickle.load(f)
    with open("/home/ethan2/GrowthCurve/data/test/dict_test_fourier_k_3_conc_3_13.pkl", "rb") as f:
        dict_test_conc_3_13 = pickle.load(f)
    with open("/home/ethan2/GrowthCurve/data/test/dict_test_fourier_k_3_conc_12_50.pkl", "rb") as f:
        dict_test_conc_12_50 = pickle.load(f)

    # Infer input_dim from one prepared batch
    Xte, _, _ = batch_to_tensor(dict_test, torch.device("cpu"))

    # DataModule & Model
    dm = GrowthCurveDataModule(
        config,
        df_train,
        dict_test,
        dict_test_conc_0_781,
        dict_test_conc_3_13,
        dict_test_conc_12_50,
    )
    model = MultiHeadLightning(input_dim=Xte.shape[1], config=config)

    # Logger only on rank 0 (rank guard should be defined elsewhere)
    wandb_logger = WandbLogger() if is_rank_zero() else None
    if wandb_logger is not None:
        # record sweep config on the run
        try:
            wandb_logger.experiment.config.update(config, allow_val_change=True)
        except Exception:
            pass

    # Trainer (DDP over up to 3 GPUs)
    trainer = pl.Trainer(
        max_epochs=config['epochs'],
        accelerator="gpu",
        devices=min(3, torch.cuda.device_count()),
        strategy=DDPStrategy(find_unused_parameters=False),
        logger=wandb_logger,
        log_every_n_steps=10,
        enable_progress_bar=is_rank_zero(),
        num_sanity_val_steps=0,
    )

    # Train
    trainer.fit(model, datamodule=dm)



if __name__ == "__main__":
    main()
