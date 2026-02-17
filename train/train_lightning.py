from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, IterableDataset
from torch_geometric.data import Batch as PyGBatch

import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

import ray
from ray import train
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)

from train.train_detanet import (
    _attach_mask,
    _build_optimizer,
    _build_scheduler,
    _compute_stats,
    _ensure_rdkit_available,
    _list_shards,
    _resolve_split_token,
    _split_label,
    build_model,
    load_checkpoint,
)


def _parse_exclude_keys(exclude_keys: str) -> List[str]:
    return [k.strip() for k in exclude_keys.split(",") if k.strip()]


def _strip_keys(item, exclude_keys: List[str]) -> None:
    for key in exclude_keys:
        try:
            if key in item:
                item.pop(key)
                continue
        except Exception:
            pass
        try:
            delattr(item, key)
        except Exception:
            continue


def _is_finite_item(item, task: str) -> bool:
    target = getattr(item, task, None)
    if target is None:
        return False
    pos = getattr(item, "pos", None)
    if pos is None:
        return False
    if torch.is_tensor(target) and not torch.isfinite(target).all().item():
        return False
    if not torch.is_tensor(target) and isinstance(target, (float, int)) and not math.isfinite(target):
        return False
    if torch.is_tensor(pos) and not torch.isfinite(pos).all().item():
        return False
    for key in item.keys():
        val = item[key]
        if torch.is_tensor(val) and val.dtype.is_floating_point:
            if not torch.isfinite(val).all().item():
                return False
    return True


def _infer_split(
    item,
    *,
    split_key: str,
    split_method: str,
    split_seed: int,
    split_train: float,
    split_val: float,
    scaffold_group_key: str,
    scaffold_smiles_key: str,
    scaffold_include_chirality: bool,
    scaffold_fallback: str,
    split_cache: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    token = _resolve_split_token(
        item,
        split_method=split_method,
        split_key=split_key,
        scaffold_group_key=scaffold_group_key,
        scaffold_smiles_key=scaffold_smiles_key,
        scaffold_include_chirality=scaffold_include_chirality,
        scaffold_fallback=scaffold_fallback,
        split_cache=split_cache,
    )
    if token is None:
        return None
    return _split_label(token, split_seed, split_train, split_val)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task", required=True)
    parser.add_argument("--shard-dir", default=None)
    parser.add_argument("--shard-dirs", default=None)
    parser.add_argument("--shard-list", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--loader-timeout", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-strict", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--checkpoint-relax-embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint-relax-mismatch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--exclude-keys",
        default="field_imputed,field_generated,field_confidence,field_source,smile,source",
    )
    parser.add_argument("--num-features", type=int, default=128)
    parser.add_argument("--num-block", type=int, default=3)
    parser.add_argument("--num-radial", type=int, default=32)
    parser.add_argument("--attention-head", type=int, default=8)
    parser.add_argument("--rc", type=float, default=5.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use-elora", action="store_true")
    parser.add_argument("--elora-path", default=None)
    parser.add_argument("--use-adalora", action="store_true")
    parser.add_argument("--adalora-r", type=int, default=8)
    parser.add_argument("--adalora-alpha", type=int, default=32)
    parser.add_argument("--adalora-dropout", type=float, default=0.05)
    parser.add_argument("--adalora-tinit", type=int, default=10)
    parser.add_argument("--adalora-tfinal", type=int, default=20)
    parser.add_argument("--adalora-total-step", type=int, default=1000)
    parser.add_argument("--adalora-targets", default=None)
    parser.add_argument("--adalora-scalar-heads", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--adalora-attention", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--adapter-freeze-base", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--optimizer", default="pt_shampoo")
    parser.add_argument("--shampoo-momentum", type=float, default=0.0)
    parser.add_argument("--shampoo-weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-scheduler", default="none")
    parser.add_argument("--lr-step-size", type=int, default=10)
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--normalize", default="none")
    parser.add_argument("--norm-cache", default=None)
    parser.add_argument("--use-impute-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mask-mode", default="binary")
    parser.add_argument("--mask-key", default="field_imputed")
    parser.add_argument("--confidence-key", default="field_confidence")
    parser.add_argument("--skip-nonfinite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-preds", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-train-preds", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-preds-max", type=int, default=5)
    parser.add_argument("--split-key", default="mol_key")
    parser.add_argument("--split-method", default="hash", choices=["hash", "scaffold"])
    parser.add_argument("--scaffold-group-key", default="mol_key")
    parser.add_argument("--scaffold-smiles-key", default="smile")
    parser.add_argument(
        "--scaffold-include-chirality",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--scaffold-fallback", default="molecule", choices=["molecule", "global"])
    parser.add_argument("--split-seed", type=int, default=123)
    parser.add_argument("--split-train", type=float, default=0.8)
    parser.add_argument("--split-val", type=float, default=0.1)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--expected-workers", type=int, default=1)
    parser.add_argument("--ray-serialize-data", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tensorboard", action="store_true")
    return parser


def build_config(base_args: List[str], overrides: Dict[str, Any]) -> Dict[str, Any]:
    parser = build_arg_parser()
    args, _ = parser.parse_known_args(base_args)
    for key, value in overrides.items():
        setattr(args, key, value)
    args.split_train = 0.7
    args.split_val = 0.1
    args.split_method = str(args.split_method).lower()
    args.scaffold_fallback = str(args.scaffold_fallback).lower()
    if args.split_method == "scaffold":
        _ensure_rdkit_available()
    if args.use_elora and not args.elora_path:
        args.elora_path = "vendored"
    cfg = vars(args)
    for key, value in list(cfg.items()):
        if isinstance(value, Path):
            cfg[key] = str(value)
    return cfg


def _rows_from_shard(row: Dict[str, Any], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    import pickle
    import torch  # local import for Ray workers

    data_list = torch.load(io.BytesIO(row["bytes"]), map_location="cpu", weights_only=False)
    out: List[Dict[str, Any]] = []
    split_cache: Dict[str, str] = {}
    exclude_keys = _parse_exclude_keys(cfg["exclude_keys"])
    for item in data_list:
        if getattr(item, cfg["task"], None) is None:
            continue
        if cfg["skip_nonfinite"] and not _is_finite_item(item, cfg["task"]):
            continue
        _attach_mask(
            item,
            task=cfg["task"],
            mask_mode=cfg["mask_mode"],
            mask_key=cfg["mask_key"],
            confidence_key=cfg["confidence_key"],
        )
        split = _infer_split(
            item,
            split_key=cfg["split_key"],
            split_method=cfg.get("split_method", "hash"),
            split_seed=cfg["split_seed"],
            split_train=cfg["split_train"],
            split_val=cfg["split_val"],
            scaffold_group_key=cfg.get("scaffold_group_key", "mol_key"),
            scaffold_smiles_key=cfg.get("scaffold_smiles_key", "smile"),
            scaffold_include_chirality=bool(cfg.get("scaffold_include_chirality", False)),
            scaffold_fallback=cfg.get("scaffold_fallback", "molecule"),
            split_cache=split_cache,
        )
        if split is None:
            continue
        if exclude_keys:
            _strip_keys(item, exclude_keys)
        payload = item
        if cfg.get("ray_serialize_data", True):
            payload = pickle.dumps(item)
        out.append({"data": payload, "split": split})
    return out


def _parse_shard_dirs(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [entry.strip() for entry in value.split(",") if entry.strip()]


def build_ray_datasets(cfg: Dict[str, Any]) -> Dict[str, "ray.data.Dataset"]:
    ctx = ray.data.DataContext.get_current()
    ctx.enable_tensor_extension_casting = False
    ctx.enable_fallback_to_arrow_object_ext_type = True
    shard_paths: List[str] = []
    for shard_dir in _parse_shard_dirs(cfg.get("shard_dirs")):
        shard_paths.extend(_list_shards(shard_dir, None))
    if not shard_paths:
        shard_paths = _list_shards(cfg.get("shard_dir"), cfg.get("shard_list"))
    shard_paths = sorted(dict.fromkeys(shard_paths))
    ds = ray.data.read_binary_files(shard_paths, include_paths=True)
    ds = ds.flat_map(lambda row: _rows_from_shard(row, cfg))
    ds = ds.random_shuffle(seed=cfg.get("seed", 123))

    train_ds = ds.filter(lambda row: row["split"] == "train").drop_columns(["split"])
    val_ds = ds.filter(lambda row: row["split"] == "val").drop_columns(["split"])
    test_ds = ds.filter(lambda row: row["split"] == "test").drop_columns(["split"])

    train_count = train_ds.count()
    val_count = val_ds.count()
    test_count = test_ds.count()
    min_samples = int(cfg.get("min_samples", 10))
    expected_workers = int(cfg.get("expected_workers", 1))
    batch_size = int(cfg.get("batch_size", 1))
    if train_count < min_samples or val_count < min_samples or test_count < min_samples:
        raise RuntimeError(
            f"Empty or undersized splits: train={train_count}, val={val_count}, test={test_count}. "
            f"Each split must have >= {min_samples} samples."
        )
    if train_count < expected_workers * batch_size:
        raise RuntimeError(
            f"Train split too small for DDP: train={train_count}, workers={expected_workers}, "
            f"batch_size={batch_size}. Ensure >= workers*batch_size samples."
        )
    return {"train": train_ds, "val": val_ds, "test": test_ds}


class RayPyGDataset(IterableDataset):
    def __init__(self, data_iter, batch_size: int, drop_last: bool) -> None:
        super().__init__()
        self.data_iter = data_iter
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        for batch in self.data_iter.iter_batches(
            batch_size=self.batch_size,
            batch_format="native",
            drop_last=self.drop_last,
        ):
            items = []
            for row in batch:
                if isinstance(row, dict) and "data" in row:
                    data = row["data"]
                    if isinstance(data, memoryview):
                        data = data.tobytes()
                    if isinstance(data, (bytes, bytearray)):
                        import pickle

                        data = pickle.loads(data)
                    items.append(data)
                else:
                    items.append(row)
            if not items:
                continue
            yield PyGBatch.from_data_list(items)


def _append_metrics(run_dir: Path, row: Dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = run_dir / "metrics.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


class RayDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        self.train_iter = None
        self.val_iter = None
        self.test_iter = None

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_iter = train.get_dataset_shard("train")
        self.val_iter = train.get_dataset_shard("val")
        self.test_iter = train.get_dataset_shard("test")

    def train_dataloader(self):
        dataset = RayPyGDataset(
            self.train_iter,
            batch_size=self.cfg["batch_size"],
            drop_last=True,
        )
        return DataLoader(dataset, batch_size=None, num_workers=0)

    def val_dataloader(self):
        dataset = RayPyGDataset(
            self.val_iter,
            batch_size=self.cfg["batch_size"],
            drop_last=False,
        )
        return DataLoader(dataset, batch_size=None, num_workers=0)

    def test_dataloader(self):
        dataset = RayPyGDataset(
            self.test_iter,
            batch_size=self.cfg["batch_size"],
            drop_last=False,
        )
        return DataLoader(dataset, batch_size=None, num_workers=0)


class DetaNetLightningModule(pl.LightningModule):
    def __init__(self, cfg: Dict[str, Any], model: nn.Module, run_dir: Path) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.run_dir = run_dir
        self.mask_name = f"mask_{cfg['task']}"
        self.base_norm, self.per_atom = self._parse_normalize(cfg["normalize"])
        self.norm_mean = torch.tensor(0.0)
        self.norm_std = torch.tensor(1.0)
        self.log_samples = cfg.get("log_preds", False)
        self.log_train_samples = cfg.get("log_train_preds", False)
        self.max_samples = int(cfg.get("log_preds_max", 5))
        self._val_pred_samples: List[float] = []
        self._val_target_samples: List[float] = []
        self._val_raw_pred_samples: List[float] = []
        self._val_raw_target_samples: List[float] = []
        self._test_pred_samples: List[float] = []
        self._test_target_samples: List[float] = []
        self._test_raw_pred_samples: List[float] = []
        self._test_raw_target_samples: List[float] = []
        self._last_train_loss = None

    @staticmethod
    def _parse_normalize(norm: str) -> tuple[str, bool]:
        per_atom = False
        base_norm = norm
        if norm.endswith("_per_atom"):
            per_atom = True
            base_norm = norm.replace("_per_atom", "")
        return base_norm, per_atom

    def set_norm_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.norm_mean = mean.detach()
        self.norm_std = std.detach().clamp(min=1e-8)

    def forward(self, batch):
        return self.model(
            z=batch.z,
            pos=batch.pos,
            edge_index=batch.edge_index,
            batch=batch.batch,
        ).float()

    def _compute_pred_target(self, batch):
        target = getattr(batch, self.cfg["task"]).float()
        mask = getattr(batch, self.mask_name, None)
        if mask is None or not self.cfg.get("use_impute_mask", True):
            mask = torch.ones_like(target)
        else:
            mask = mask.float()

        pred = self.forward(batch)

        if self.per_atom:
            counts = torch.bincount(batch.batch, minlength=target.shape[0]).float().to(target.device)
            while counts.dim() < target.dim():
                counts = counts.unsqueeze(-1)
            pred = pred / counts.clamp(min=1.0)
            target = target / counts.clamp(min=1.0)

        raw_pred = pred.detach()
        raw_target = target.detach()

        if self.base_norm == "batch":
            denom = mask.sum().clamp(min=1.0)
            mean = (target * mask).sum() / denom
            var = ((target - mean) ** 2 * mask).sum() / denom
            std = torch.sqrt(var + 1e-12)
        elif self.base_norm == "dataset":
            mean = self.norm_mean.to(target.device)
            std = self.norm_std.to(target.device)
        else:
            mean = torch.tensor(0.0, device=target.device)
            std = torch.tensor(1.0, device=target.device)

        pred = (pred - mean) / std
        target = (target - mean) / std

        pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
        mask = torch.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)

        return pred, target, mask, raw_pred, raw_target

    def training_step(self, batch, batch_idx):
        pred, target, mask, raw_pred, raw_target = self._compute_pred_target(batch)
        denom = mask.sum().clamp(min=1.0)
        loss = ((pred - target) ** 2 * mask).sum() / denom
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

        self._last_train_loss = loss.detach()
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        if self.global_rank != 0:
            return
        step = self.trainer.global_step
        if step % int(self.cfg.get("log_every", 50)) != 0:
            return
        row = {
            "epoch": int(self.current_epoch),
            "step": int(step),
            "split": "train",
            "loss": float(self._last_train_loss.item() if self._last_train_loss is not None else 0.0),
        }
        if self.log_train_samples and self.max_samples > 0:
            pred, target, _, raw_pred, raw_target = self._compute_pred_target(batch)
            row["train_pred_samples"] = pred.detach().view(-1).cpu().tolist()[: self.max_samples]
            row["train_target_samples"] = target.detach().view(-1).cpu().tolist()[: self.max_samples]
            row["train_raw_pred_samples"] = raw_pred.detach().view(-1).cpu().tolist()[: self.max_samples]
            row["train_raw_target_samples"] = raw_target.detach().view(-1).cpu().tolist()[: self.max_samples]
        _append_metrics(self.run_dir, row)

    def validation_step(self, batch, batch_idx):
        pred, target, mask, raw_pred, raw_target = self._compute_pred_target(batch)
        denom = mask.sum().clamp(min=1.0)
        mse_sum = ((pred - target) ** 2 * mask).sum()
        mae_sum = ((pred - target).abs() * mask).sum()
        count = denom

        self.log("val_mse_sum", mse_sum, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="sum")
        self.log("val_mae_sum", mae_sum, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="sum")
        self.log("val_count", count, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="sum")

        if self.log_samples and self.global_rank == 0 and self.max_samples > 0:
            if len(self._val_pred_samples) < self.max_samples:
                flat_pred = pred.detach().view(-1).cpu().tolist()
                flat_target = target.detach().view(-1).cpu().tolist()
                flat_raw_pred = raw_pred.detach().view(-1).cpu().tolist()
                flat_raw_target = raw_target.detach().view(-1).cpu().tolist()
                for p, t, rp, rt in zip(flat_pred, flat_target, flat_raw_pred, flat_raw_target):
                    self._val_pred_samples.append(float(p))
                    self._val_target_samples.append(float(t))
                    self._val_raw_pred_samples.append(float(rp))
                    self._val_raw_target_samples.append(float(rt))
                    if len(self._val_pred_samples) >= self.max_samples:
                        break

    def on_validation_epoch_end(self) -> None:
        metrics = self.trainer.callback_metrics
        mse_sum = metrics.get("val_mse_sum", torch.tensor(0.0, device=self.device))
        mae_sum = metrics.get("val_mae_sum", torch.tensor(0.0, device=self.device))
        count = metrics.get("val_count", torch.tensor(0.0, device=self.device))
        if torch.is_tensor(count) and count.item() > 0:
            mse = mse_sum / count
            mae = mae_sum / count
        else:
            mse = torch.tensor(0.0, device=self.device)
            mae = torch.tensor(0.0, device=self.device)
        self.log("val_mse", mse, prog_bar=True, sync_dist=False)
        self.log("val_mae", mae, prog_bar=False, sync_dist=False)
        if self.global_rank == 0:
            row = {
                "epoch": int(self.current_epoch),
                "step": int(self.trainer.global_step),
                "val_mse": float(mse.item()),
                "val_mae": float(mae.item()),
            }
            if self.log_samples and self.max_samples > 0:
                row["val_pred_samples"] = self._val_pred_samples[: self.max_samples]
                row["val_target_samples"] = self._val_target_samples[: self.max_samples]
                row["val_raw_pred_samples"] = self._val_raw_pred_samples[: self.max_samples]
                row["val_raw_target_samples"] = self._val_raw_target_samples[: self.max_samples]
            _append_metrics(self.run_dir, row)
            self._val_pred_samples = []
            self._val_target_samples = []
            self._val_raw_pred_samples = []
            self._val_raw_target_samples = []

    def test_step(self, batch, batch_idx):
        pred, target, mask, raw_pred, raw_target = self._compute_pred_target(batch)
        denom = mask.sum().clamp(min=1.0)
        mse_sum = ((pred - target) ** 2 * mask).sum()
        mae_sum = ((pred - target).abs() * mask).sum()
        count = denom

        self.log("test_mse_sum", mse_sum, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="sum")
        self.log("test_mae_sum", mae_sum, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="sum")
        self.log("test_count", count, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="sum")

        if self.log_samples and self.global_rank == 0 and self.max_samples > 0:
            if len(self._test_pred_samples) < self.max_samples:
                flat_pred = pred.detach().view(-1).cpu().tolist()
                flat_target = target.detach().view(-1).cpu().tolist()
                flat_raw_pred = raw_pred.detach().view(-1).cpu().tolist()
                flat_raw_target = raw_target.detach().view(-1).cpu().tolist()
                for p, t, rp, rt in zip(flat_pred, flat_target, flat_raw_pred, flat_raw_target):
                    self._test_pred_samples.append(float(p))
                    self._test_target_samples.append(float(t))
                    self._test_raw_pred_samples.append(float(rp))
                    self._test_raw_target_samples.append(float(rt))
                    if len(self._test_pred_samples) >= self.max_samples:
                        break

    def on_test_epoch_end(self) -> None:
        metrics = self.trainer.callback_metrics
        mse_sum = metrics.get("test_mse_sum", torch.tensor(0.0, device=self.device))
        mae_sum = metrics.get("test_mae_sum", torch.tensor(0.0, device=self.device))
        count = metrics.get("test_count", torch.tensor(0.0, device=self.device))
        if torch.is_tensor(count) and count.item() > 0:
            mse = mse_sum / count
            mae = mae_sum / count
        else:
            mse = torch.tensor(0.0, device=self.device)
            mae = torch.tensor(0.0, device=self.device)
        self.log("test_mse", mse, prog_bar=True, sync_dist=False)
        self.log("test_mae", mae, prog_bar=False, sync_dist=False)
        if self.global_rank == 0:
            row = {
                "epoch": int(self.current_epoch),
                "step": int(self.trainer.global_step),
                "test_mse": float(mse.item()),
                "test_mae": float(mae.item()),
            }
            if self.log_samples and self.max_samples > 0:
                row["test_pred_samples"] = self._test_pred_samples[: self.max_samples]
                row["test_target_samples"] = self._test_target_samples[: self.max_samples]
                row["test_raw_pred_samples"] = self._test_raw_pred_samples[: self.max_samples]
                row["test_raw_target_samples"] = self._test_raw_target_samples[: self.max_samples]
            _append_metrics(self.run_dir, row)
            self._test_pred_samples = []
            self._test_target_samples = []
            self._test_raw_pred_samples = []
            self._test_raw_target_samples = []

    def configure_optimizers(self):
        args = SimpleNamespace(**self.cfg)
        optimizer = _build_optimizer(args, self.model)
        scheduler = _build_scheduler(args, optimizer)
        if scheduler is None:
            return optimizer
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def _load_norm_cache(path: Optional[str]) -> Optional[Dict[str, float]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        if "mean" in data and "std" in data:
            return {"mean": float(data["mean"]), "std": float(data["std"])}
    except Exception:
        return None
    return None


def _save_norm_cache(path: Optional[str], mean: float, std: float) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"mean": float(mean), "std": float(std)}
    p.write_text(json.dumps(payload, indent=2))


def train_loop_per_worker(cfg: Dict[str, Any]) -> None:
    ctx = train.get_context()
    rank = ctx.get_world_rank()
    local_rank = ctx.get_local_rank()

    pl.seed_everything(int(cfg.get("seed", 123)) + int(rank), workers=True)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    cfg = dict(cfg)
    cfg["device"] = str(device)

    model = build_model(SimpleNamespace(**cfg))
    if cfg.get("checkpoint"):
        ckpt = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=False)
        load_checkpoint(
            model,
            ckpt,
            strict=cfg.get("checkpoint_strict", False),
            relax_embeddings=cfg.get("checkpoint_relax_embeddings", True),
            relax_mismatch=cfg.get("checkpoint_relax_mismatch", True),
        )

    trial_name = ctx.get_trial_name()
    registry_dir = cfg.get("registry_dir") or ctx.get_trial_dir()
    run_prefix = cfg.get("run_prefix", "tune")
    run_dir = Path(registry_dir) / f"{run_prefix}-{trial_name}"
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True))

    datamodule = RayDataModule(cfg)
    datamodule.setup()

    module = DetaNetLightningModule(cfg, model, run_dir)

    base_norm = module.base_norm
    if base_norm == "dataset":
        cache = _load_norm_cache(cfg.get("norm_cache"))
        if cache:
            mean = torch.tensor(cache["mean"], device=device)
            std = torch.tensor(cache["std"], device=device)
        else:
            stats_loader = datamodule.train_dataloader()
            mean, std = _compute_stats(
                stats_loader,
                cfg["task"],
                module.mask_name,
                module.per_atom,
                device,
                cfg.get("skip_nonfinite", True),
            )
            if rank == 0:
                _save_norm_cache(cfg.get("norm_cache"), mean.item(), std.item())
        module.set_norm_stats(mean, std)

    callbacks = [
        RayTrainReportCallback(),
        LearningRateMonitor(logging_interval="step"),
    ]

    logger = None
    if cfg.get("tensorboard", False):
        logger = TensorBoardLogger(save_dir=str(run_dir), name="tensorboard")

    strategy = RayDDPStrategy(find_unused_parameters=True)
    trainer = pl.Trainer(
        max_epochs=int(cfg.get("epochs", 1)),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        strategy=strategy,
        plugins=[RayLightningEnvironment()],
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=int(cfg.get("log_every", 50)),
        accumulate_grad_batches=int(cfg.get("grad_accum", 1)),
        precision="16-mixed" if cfg.get("amp", False) else "32-true",
        gradient_clip_val=float(cfg.get("grad_clip", 0.0) or 0.0),
        enable_checkpointing=True,
        check_val_every_n_epoch=int(cfg.get("eval_every", 1)),
    )
    trainer = prepare_trainer(trainer)
    trainer.fit(module, datamodule=datamodule)
    trainer.test(module, datamodule=datamodule)
