#!/usr/bin/env python
# coding: utf-8
import os, pickle, json
os.environ.setdefault("WANDB_START_METHOD", "thread")  # safer under DDP

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning.loggers import WandbLogger
from collections import defaultdict
from pytorch_lightning.callbacks import ModelCheckpoint

from torchmetrics.functional import (
    mean_absolute_error,
    auroc,
    average_precision,
    f1_score,
    recall,
    pearson_corrcoef
)
# (optional but helpful on HPC)
os.environ.setdefault("WANDB__SERVICE_WAIT", "600")
os.environ.setdefault("WANDB_HTTP_TIMEOUT", "300")
os.environ.setdefault("WANDB_DISABLE_CODE", "true")
os.environ.setdefault("WANDB_SILENT", "true")
import wandb


# ─────────────────────────────────────────────────────────────
# Import your custom datasets and collate
# ─────────────────────────────────────────────────────────────
from data_class import PerCompoundDataset, custom_collate, build_val_dict_from_metas


import argparse

N_MIN_ACTIVES_FOR_REGRESSION = 5

FEATURE_SETS = {
    "minimol": ["minimol_fp"],
    "boltz2_minimol": ["boltz2_rep", "minimol_fp"],
    "boltz2_classic": ["boltz2_rep", "ecfp_fp", "maccs_fp", "rdkit_fp"],
    "minimol_classic": ["minimol_fp", "ecfp_fp", "maccs_fp", "rdkit_fp"],
    "boltz2_minimol_classic": ["boltz2_rep", "minimol_fp", "ecfp_fp", "maccs_fp", "rdkit_fp"],
}

def parse_args():
    p = argparse.ArgumentParser()
    # match the sweep yaml keys
    p.add_argument("--samples", type=int)
    p.add_argument("--weight_decay", type=float,)
    p.add_argument("--seed", type=int)
    p.add_argument("--loss_lambda", type=float)
    p.add_argument("--max_learning_rate", type=float)
    p.add_argument("--eta_min", type=float, default=1e-8)
    p.add_argument("--dropout_rate", type=float)
    p.add_argument("--active_fraction", type=float)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--trunk_layers", type=int)
    p.add_argument("--trunk_dim", type=int)
    p.add_argument("--reg_layers", type=int)
    p.add_argument("--reg_hidden", type=int)
    p.add_argument("--cls_layers", type=int)
    p.add_argument("--cls_hidden", type=int)
    p.add_argument("--regression_noise", type=float)
    p.add_argument("--feature_set", type=str, default="all")
    p.add_argument("--metas_path", type=str,
                   default="/home/ethan2/GrowthNet/data/splits/my_split_v1/all_compound_metas.pkl")
    p.add_argument("--train_smiles_path", type=str,
                   default="/home/ethan2/GrowthNet/data/splits/my_split_v1/smile_splits_v2/train.txt")
    p.add_argument("--val_smiles_path", type=str,
                   default="/home/ethan2/GrowthNet/data/splits/my_split_v1/smile_splits_v2/val.txt")
    p.add_argument("--test_smiles_path", type=str,
                   default="/home/ethan2/GrowthNet/data/splits/my_split_v1/smile_splits_v2/test.txt")

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

        # ── Shared trunk ────────────────────────────────────────────────
        trunk = []
        prev_dim = input_dim
        for _ in range(trunk_layers):
            trunk += [
                nn.Linear(prev_dim, trunk_dim),
                nn.LayerNorm(trunk_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
            ]
            prev_dim = trunk_dim

        self.trunk = nn.Sequential(*trunk)

        # ── Regression head ─────────────────────────────────────────────
        reg = []
        prev = trunk_dim
        for _ in range(reg_layers):
            reg += [
                nn.Linear(prev, reg_hidden),
                nn.LayerNorm(reg_hidden),
                nn.ReLU(inplace=True)
            ]
            prev = reg_hidden
        reg += [nn.Linear(prev, 1)]
        self.reg_head = nn.Sequential(*reg)

        # ── Classification head ─────────────────────────────────────────
        cls = []
        prev = trunk_dim
        for _ in range(cls_layers):
            cls += [
                nn.Linear(prev, cls_hidden),
                nn.LayerNorm(cls_hidden),
                nn.ReLU(inplace=True)
            ]
            prev = cls_hidden
        cls += [nn.Linear(prev, 1)]
        self.cls_head = nn.Sequential(*cls)

    def forward(self, x: torch.Tensor):
        features = self.trunk(x)
        reg_out = self.reg_head(features).squeeze(-1)
        cls_logits = self.cls_head(features).squeeze(-1)
        return reg_out, cls_logits




def batch_to_tensor(batch: dict, device: torch.device, feature_set: str = "all"):
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

    families = FEATURE_SETS[feature_set]
    feats = [t_feats, c_raw, c_log]
    for fam in sorted(families):
        feats.append(batch["features_by_family"][fam].repeat_interleave(repeats, dim=0))
    feats = [f.to(device) for f in feats]

    X = torch.cat(feats, dim=1)
    return X, y_reg.to(device), y_cls.to(device)


# ─────────────────────────────────────────────────────────────
# LIGHTNING MODULE
# ─────────────────────────────────────────────────────────────
class MultiHeadLightning(pl.LightningModule):
    def __init__(self, input_dim, config, dict_test_main):
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

        self.val_outputs = defaultdict(lambda: {
            "r_pred": [], "r_true": [], "c_pred": [], "c_true": [], "n": 0
        })
        self.train_outputs = defaultdict(lambda: {
            "r_pred": [], "r_true": [], "c_pred": [], "c_true": []
        })
        self.best_val_main_agg  = float("-inf")
        self.best_test_main_agg = float("-inf")
        self.dict_test_main = dict_test_main



    def forward(self, x):
        return self.model(x)

    @torch.no_grad()
    def _eval_agg_on_dict(self, data_dict: dict) -> float:
        """Compute mean(AP + Pearson - 5*MAE) over per-(t,c) subgroups for a pre-built dict."""
        self.model.eval()
        Xb, yb_reg, yb_cls = batch_to_tensor(data_dict, self.device, feature_set=self.hparams.feature_set)
        pred_reg, pred_cls_logits = self.model(Xb)
        pred_cls_probs = torch.sigmoid(pred_cls_logits)
        self.model.train()

        t_vals = data_dict["t_raw"]
        c_vals = data_dict["c_raw"]

        subgroups = defaultdict(lambda: {"r_pred": [], "r_true": [], "c_pred": [], "c_true": []})
        for ti, ci, rp, rt, cp, ct in zip(t_vals, c_vals, pred_reg, yb_reg, pred_cls_probs, yb_cls):
            key = (round(ti.item(), 2), round(ci.item(), 3))
            subgroups[key]["r_pred"].append(rp.cpu())
            subgroups[key]["r_true"].append(rt.cpu())
            subgroups[key]["c_pred"].append(cp.cpu())
            subgroups[key]["c_true"].append(ct.cpu())

        rows = []
        for g in subgroups.values():
            r_pred = torch.stack(g["r_pred"])
            r_true = torch.stack(g["r_true"])
            c_pred = torch.stack(g["c_pred"])
            c_true = torch.stack(g["c_true"]).int()
            active_mask = c_true == 1
            ap = average_precision(c_pred, c_true, task="binary") if c_true.sum() > 0 else torch.tensor(0.0)

            if active_mask.sum() >= N_MIN_ACTIVES_FOR_REGRESSION:
                pred_act = r_pred[active_mask]
                true_act = r_true[active_mask]
                if pred_act.std() > 1e-6 and true_act.std() > 1e-6:
                    pearson = pearson_corrcoef(pred_act, true_act)
                    mae = mean_absolute_error(pred_act, true_act)
                else:
                    pearson = None
                    mae = None
            else:
                pearson = None
                mae = None

            rows.append((mae, pearson, ap))

        mae_vals = [r[0] for r in rows if r[0] is not None]
        pear_vals = [r[1] for r in rows if r[1] is not None]
        ap_vals = [r[2] for r in rows]

        mae_m = sum(mae_vals) / len(mae_vals) if mae_vals else 0.0
        pear_m = sum(pear_vals) / len(pear_vals) if pear_vals else 0.0
        ap_m = sum(ap_vals) / len(ap_vals) if ap_vals else 0.0

        agg = ap_m - 5 * mae_m + pear_m
        if torch.isnan(torch.tensor(agg)):
            agg = -1e6

        return float(agg)

    def training_step(self, batch, batch_idx):
        Xb, yb_reg, yb_cls = batch_to_tensor(batch, self.device, feature_set=self.hparams.feature_set)
        out_reg, out_cls_logits = self(Xb)

        loss_reg = self.mse_loss(out_reg, yb_reg)
        loss_cls = self.bce_loss(out_cls_logits.squeeze(-1), yb_cls)
        loss = loss_cls + self.hparams.loss_lambda * loss_reg

        self.log("train/reg_loss", loss_reg, on_step=False, on_epoch=True, sync_dist=False, prog_bar=False)
        self.log("train/cls_loss", loss_cls, on_step=False, on_epoch=True, sync_dist=False, prog_bar=False)
        self.log("train/loss",     loss,     on_step=False, on_epoch=True, sync_dist=False, prog_bar=True)

        pred_cls_probs = torch.sigmoid(out_cls_logits).detach().cpu()
        pred_reg_det   = out_reg.detach().cpu()
        yb_reg_det     = yb_reg.detach().cpu()
        yb_cls_det     = yb_cls.detach().cpu()

        if batch["t_fourier"].ndim == 3:
            N, k, _ = batch["t_fourier"].shape
            t_vals = batch["t_raw"].reshape(N * k).cpu()
            c_vals = batch["c_raw"].reshape(N * k).cpu()
        else:
            t_vals = batch["t_raw"].cpu()
            c_vals = batch["c_raw"].cpu()

        for ti, ci, r_pr, r_tr, c_pr, c_tr in zip(
            t_vals, c_vals, pred_reg_det, yb_reg_det, pred_cls_probs, yb_cls_det
        ):
            key = (round(ti.item(), 2), round(ci.item(), 3))
            g = self.train_outputs[key]
            g["r_pred"].append(r_pr)
            g["r_true"].append(r_tr)
            g["c_pred"].append(c_pr)
            g["c_true"].append(c_tr)

        return loss
    


    def on_train_epoch_end(self):
        rows = []
        for g in self.train_outputs.values():
            r_pred = torch.stack(g["r_pred"])
            r_true = torch.stack(g["r_true"])
            c_pred = torch.stack(g["c_pred"])
            c_true = torch.stack(g["c_true"]).int()

            active_mask = c_true == 1
            ap = average_precision(c_pred, c_true, task="binary") if c_true.sum() > 0 else torch.tensor(0.0)
            auc = auroc(c_pred, c_true, task="binary") if c_true.sum() > 0 else torch.tensor(0.0)
            f1 = f1_score(c_pred > 0.5, c_true, task="binary") if c_true.sum() > 0 else torch.tensor(0.0)
            rec = recall(c_pred > 0.5, c_true, task="binary") if c_true.sum() > 0 else torch.tensor(0.0)

            if active_mask.sum() >= N_MIN_ACTIVES_FOR_REGRESSION:
                pred_act = r_pred[active_mask]
                true_act = r_true[active_mask]
                if pred_act.std() > 1e-6 and true_act.std() > 1e-6:
                    mae = mean_absolute_error(pred_act, true_act)
                    pearson = pearson_corrcoef(pred_act, true_act)
                else:
                    mae = None
                    pearson = None
            else:
                mae = None
                pearson = None

            rows.append((mae, pearson, auc, ap, f1, rec))

        self.train_outputs.clear()
        if not rows:
            return

        mae_vals = [r[0] for r in rows if r[0] is not None]
        pear_vals = [r[1] for r in rows if r[1] is not None]
        auc_vals = [r[2] for r in rows]
        ap_vals = [r[3] for r in rows]
        f1_vals = [r[4] for r in rows]
        rec_vals = [r[5] for r in rows]

        mae_m = sum(mae_vals) / len(mae_vals) if mae_vals else 0.0
        pear_m = sum(pear_vals) / len(pear_vals) if pear_vals else 0.0
        auc_m = sum(auc_vals) / len(auc_vals) if auc_vals else 0.0
        ap_m = sum(ap_vals) / len(ap_vals) if ap_vals else 0.0
        f1_m = sum(f1_vals) / len(f1_vals) if f1_vals else 0.0
        rec_m = sum(rec_vals) / len(rec_vals) if rec_vals else 0.0

        agg = ap_m - 5 * mae_m + pear_m
        if torch.isnan(torch.tensor(agg)):
            agg = -1e6

        self.log("train/mae_active",       mae_m,  sync_dist=False)
        self.log("train/pearson_active",   pear_m, sync_dist=False)
        self.log("train/auc",              auc_m,  sync_dist=False)
        self.log("train/ap",               ap_m,   sync_dist=False)
        self.log("train/f1",               f1_m,   sync_dist=False)
        self.log("train/recall",           rec_m,  sync_dist=False)
        self.log("train/AP+Pearson-5*MAE", agg,    sync_dist=False)

    def validation_step(self, batch, batch_idx):
        Xb, yb_reg, yb_cls = batch_to_tensor(batch, self.device, feature_set=self.hparams.feature_set)
        pred_reg, pred_cls_logits = self(Xb)
        pred_cls_probs = torch.sigmoid(pred_cls_logits)

        reg_loss = self.mse_loss(pred_reg, yb_reg)
        cls_loss = self.bce_loss(pred_cls_logits.squeeze(-1), yb_cls)
        loss = reg_loss + cls_loss

        # Log val_main losses including actives/inactives split
        active_mask   = (yb_cls == 1)
        inactive_mask = ~active_mask

        reg_loss_act   = self.mse_loss(pred_reg[active_mask],   yb_reg[active_mask])   if active_mask.any()   else torch.tensor(0.0, device=self.device)
        reg_loss_inact = self.mse_loss(pred_reg[inactive_mask], yb_reg[inactive_mask]) if inactive_mask.any() else torch.tensor(0.0, device=self.device)

        self.log("val_main/reg_loss",          reg_loss,       on_epoch=True, sync_dist=False)
        self.log("val_main/cls_loss",          cls_loss,       on_epoch=True, sync_dist=False)
        self.log("val_main/loss",              loss,           on_epoch=True, sync_dist=False)
        self.log("val_main/reg_loss_actives",  reg_loss_act,   on_epoch=True, sync_dist=False)
        self.log("val_main/reg_loss_inactives",reg_loss_inact, on_epoch=True, sync_dist=False)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=False)

        # Store per-(t,c) subgroup predictions for metric aggregation
        t = batch["t_raw"].detach()
        c = batch["c_raw"].detach()

        for ti, ci, r_pr, r_tr, c_pr, c_tr in zip(t, c, pred_reg, yb_reg, pred_cls_probs, yb_cls):
            key = (round(ti.item(), 2), round(ci.item(), 3))
            g = self.val_outputs[key]
            g["r_pred"].append(r_pr.detach().cpu())
            g["r_true"].append(r_tr.detach().cpu())
            g["c_pred"].append(c_pr.detach().cpu())
            g["c_true"].append(c_tr.detach().cpu())

        return loss

    def on_validation_epoch_end(self):
        # Concentration slices: name → {conc, exclude_t0}
        # val_main = all (t,c) cells
        # val_<slice> = cells with c == slice_conc AND t != 0
        SLICE_CONCS = {
            "val_0_2":   0.2,
            "val_0_781": 0.781,
            "val_1_2":   1.2,
            "val_3_13":  3.13,
            "val_7_9":   7.9,
            "val_12_50": 12.5,
            "val_50":    50.0,
        }

        # Accumulate per-(t,c) metrics
        subgroup_results = {}  # (t, c) → (mae, pearson, auc, ap, f1, rec)
        for (t, c), g in self.val_outputs.items():
            r_pred = torch.stack(g["r_pred"])
            r_true = torch.stack(g["r_true"])
            c_pred = torch.stack(g["c_pred"])
            c_true = torch.stack(g["c_true"]).int()

            active_mask = c_true == 1
            auc = auroc(c_pred, c_true, task="binary") if c_true.sum() > 0 else torch.tensor(0.0)
            ap = average_precision(c_pred, c_true, task="binary") if c_true.sum() > 0 else torch.tensor(0.0)
            f1 = f1_score(c_pred > 0.5, c_true, task="binary") if c_true.sum() > 0 else torch.tensor(0.0)
            rec = recall(c_pred > 0.5, c_true, task="binary") if c_true.sum() > 0 else torch.tensor(0.0)

            if active_mask.sum() >= N_MIN_ACTIVES_FOR_REGRESSION:
                pred_act = r_pred[active_mask]
                true_act = r_true[active_mask]
                if pred_act.std() > 1e-6 and true_act.std() > 1e-6:
                    mae = mean_absolute_error(pred_act, true_act)
                    pearson = pearson_corrcoef(pred_act, true_act)
                else:
                    mae = None
                    pearson = None
            else:
                mae = None
                pearson = None

            subgroup_results[(t, c)] = (mae, pearson, auc, ap, f1, rec)

        def log_slice(name: str, rows):
            if not rows:
                return

            mae_vals = [r[0] for r in rows if r[0] is not None]
            pear_vals = [r[1] for r in rows if r[1] is not None]
            auc_vals = [r[2] for r in rows]
            ap_vals = [r[3] for r in rows]
            f1_vals = [r[4] for r in rows]
            rec_vals = [r[5] for r in rows]

            mae_m = sum(mae_vals) / len(mae_vals) if mae_vals else 0.0
            pear_m = sum(pear_vals) / len(pear_vals) if pear_vals else 0.0
            auc_m = sum(auc_vals) / len(auc_vals) if auc_vals else 0.0
            ap_m = sum(ap_vals) / len(ap_vals) if ap_vals else 0.0
            f1_m = sum(f1_vals) / len(f1_vals) if f1_vals else 0.0
            rec_m = sum(rec_vals) / len(rec_vals) if rec_vals else 0.0

            agg = ap_m - 5 * mae_m + pear_m
            if torch.isnan(torch.tensor(agg)):
                agg = -1e6

            self.log(f"{name}/mae_active",       mae_m,  sync_dist=False)
            self.log(f"{name}/pearson_active",   pear_m, sync_dist=False)
            self.log(f"{name}/auc",              auc_m,  sync_dist=False)
            self.log(f"{name}/ap",               ap_m,   sync_dist=False)
            self.log(f"{name}/f1",               f1_m,   sync_dist=False)
            self.log(f"{name}/recall",           rec_m,  sync_dist=False)
            self.log(f"{name}/AP+Pearson-5*MAE", agg, prog_bar=(name == "val_main"), sync_dist=False)

            if name == "val_main":
                if agg > getattr(self, "best_val_main_agg", float("-inf")):
                    self.best_val_main_agg   = agg
                    self.best_val_main_epoch = self.current_epoch
                    self.best_test_main_agg  = self._eval_agg_on_dict(self.dict_test_main)
                self.log("val_main/best_agg_metric",  self.best_val_main_agg,  prog_bar=True,  sync_dist=False)
                self.log("val_main/best_agg_epoch",   getattr(self, "best_val_main_epoch", -1), prog_bar=False, sync_dist=False)
                self.log("test_main/best_agg_metric", self.best_test_main_agg, prog_bar=False, sync_dist=False)

        # val_main: all subgroups
        log_slice("val_main", list(subgroup_results.values()))

        # per-concentration slices: c matches AND t != 0
        for slice_name, conc in SLICE_CONCS.items():
            rows = [
                metrics
                for (t, c), metrics in subgroup_results.items()
                if abs(c - conc) < 0.001 and t != 0.0
            ]
            log_slice(slice_name, rows)

        self.val_outputs.clear()
            


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.max_learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.epochs,
            eta_min=self.hparams.eta_min,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


# ─────────────────────────────────────────────────────────────
# DATAMODULE
# ─────────────────────────────────────────────────────────────
class GrowthNetDataModule(pl.LightningDataModule):
    def __init__(self, config, dict_val_main):
        super().__init__()
        self.config = config
        self.dict_val_main = dict_val_main

    def train_dataloader(self):

        epoch = getattr(self.trainer, "current_epoch", 0)

        self.train_ds = PerCompoundDataset(
            self.config['metas_path'],
            self.config['train_smiles_path'],
            k=self.config['samples'], seed=self.config['seed'] + epoch, num_fourier=3, noise=self.config['regression_noise'],
            required_families=set(FEATURE_SETS[self.config['feature_set']]),
        )

        
        
        g = torch.Generator()
        g.manual_seed(self.config['seed'] + epoch)

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
                          num_workers=6,
                          pin_memory=True)
    
    def val_dataloader(self):

        class DictDataset(torch.utils.data.Dataset):
            def __init__(self, data_dict): self.data = data_dict
            def __len__(self): return 1
            def __getitem__(self, idx): return self.data

        identity = lambda batch: batch[0]  # don’t stack into a batch

        return DataLoader(DictDataset(self.dict_val_main), batch_size=1, collate_fn=identity, num_workers=0, pin_memory=True)
    


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    # Parse sweep-injected CLI args
    config = parse_args()

    # wandb's ${args} expansion can silently drop string categorical
    # parameters.  Init the run early so we can read them from the
    # server-side sweep config and merge anything that was missing.
    run = wandb.init()
    if run is not None:
        for k, v in run.config.items():
            if k in config and v is not None:
                config[k] = v

    print(f"[sweep_multihead] Resolved feature_set = {config['feature_set']}")

    # Reproducibility + matmul perf
    pl.seed_everything(config['seed'], workers=True)
    try:
        torch.set_float32_matmul_precision('medium')
    except Exception:
        pass

    # ── Load unified data ──
    print(f"[sweep_multihead] Loading metas from {config['metas_path']}")
    with open(config['metas_path'], "rb") as f:
        all_metas = pickle.load(f)

    required_fams = set(FEATURE_SETS[config['feature_set']])

    val_smiles = set(open(config['val_smiles_path']).read().splitlines())
    val_metas  = [m for m in all_metas if m.smiles in val_smiles and required_fams.issubset(m.fps_by_family.keys())]
    print(f"[sweep_multihead] Building val dict from {len(val_metas)} val compounds...")
    dict_val_main = build_val_dict_from_metas(val_metas)

    test_smiles = set(open(config['test_smiles_path']).read().splitlines())
    test_metas  = [m for m in all_metas if m.smiles in test_smiles and required_fams.issubset(m.fps_by_family.keys())]
    print(f"[sweep_multihead] Building test dict from {len(test_metas)} test compounds...")
    dict_test_main = build_val_dict_from_metas(test_metas)

    # Infer input_dim from val dict
    Xte, _, _ = batch_to_tensor(dict_val_main, torch.device("cpu"), feature_set=config['feature_set'])

    # DataModule & Model
    dm = GrowthNetDataModule(config, dict_val_main)
    model = MultiHeadLightning(input_dim=Xte.shape[1], config=config, dict_test_main=dict_test_main)

    wandb_logger = WandbLogger()
    try:
        wandb_logger.experiment.config.update(config, allow_val_change=True)
    except Exception:
        pass
    
    run_id = wandb.run.id if wandb.run else "debug"

    save_dir = f'/home/ethan2/GrowthNet/models/final_sweep/checkpoints/{run_id}'

    os.makedirs(save_dir, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=save_dir,
        filename="best_params",
        monitor="val_main/AP+Pearson-5*MAE",   # metric you log
        mode="max",                              # maximize agg metric
        save_top_k=1,                            # only keep the best
        save_last=False,                          # also save the last epoch
    )

    trainer = pl.Trainer(
        max_epochs=config['epochs'],
        accelerator="gpu",
        devices=1,
        logger=wandb_logger,
        log_every_n_steps=10,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        callbacks=[checkpoint_cb]
    )

    # Train
    trainer.fit(model, datamodule=dm)

    with open(os.path.join(save_dir, "hparams.json"), "w") as f:
        json.dump(config, f, indent=2)

    best_ckpt = checkpoint_cb.best_model_path
    if best_ckpt:
        wandb.log({"best_checkpoint_path": best_ckpt})
        print(f"Best checkpoint saved at: {best_ckpt}")
        print(f"Hparams saved at: {os.path.join(save_dir, 'hparams.json')}")


    
if __name__ == "__main__":
    main()
