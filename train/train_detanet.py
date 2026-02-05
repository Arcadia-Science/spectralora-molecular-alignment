from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import random
import csv
import math
import subprocess
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable, List, Optional

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import IterableDataset, get_worker_info
from torch_geometric.loader import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = REPO_ROOT / "capsule-3259363" / "code"
if str(MODEL_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(MODEL_ROOT))

from detanet_model.detanet import DetaNet
from detanet_model.model_loader import BASE_MODEL_CONFIG, TASK_CONFIGS

METRICS_HEADER = [
    "epoch",
    "step",
    "split",
    "loss",
    "lr",
    "val_mse",
    "val_mse_lo",
    "val_mse_hi",
    "val_mae",
    "val_mae_lo",
    "val_mae_hi",
    "test_mse",
    "test_mse_lo",
    "test_mse_hi",
    "test_mae",
    "test_mae_lo",
    "test_mae_hi",
]


def _list_shards(shard_dir: Optional[str], list_file: Optional[str]) -> List[str]:
    if list_file:
        paths = [line.strip() for line in Path(list_file).read_text().splitlines() if line.strip()]
        return paths
    if not shard_dir:
        raise ValueError("Provide --shard-dir or --shard-list.")
    return sorted(str(p) for p in Path(shard_dir).glob("shard_*.pt"))


def _split_bucket(key: str, seed: int) -> int:
    digest = hashlib.md5(f"{seed}:{key}".encode()).hexdigest()
    return int(digest, 16) % 1000


def _split_label(key: str, seed: int, train_ratio: float, val_ratio: float) -> str:
    bucket = _split_bucket(key, seed)
    train_cutoff = int(train_ratio * 1000)
    val_cutoff = int((train_ratio + val_ratio) * 1000)
    if bucket < train_cutoff:
        return "train"
    if bucket < val_cutoff:
        return "val"
    return "test"


def _normalize_split_key(value) -> str:
    if torch.is_tensor(value):
        if value.numel() == 1:
            value = value.item()
        else:
            value = value.flatten()[0].item()
    return str(value)


class ShardIterable(IterableDataset):
    def __init__(
        self,
        shard_paths: List[str],
        task: str,
        mask_mode: str,
        mask_key: str,
        confidence_key: str,
        seed: int = 123,
        shuffle_shards: bool = True,
        shuffle_samples: bool = False,
        split: str = "all",
        split_key: str = "mol_key",
        split_seed: int = 123,
        split_train: float = 0.8,
        split_val: float = 0.1,
        use_distributed: bool = True,
        skip_nonfinite: bool = True,
    ) -> None:
        super().__init__()
        self.shard_paths = list(shard_paths)
        self.task = task
        self.mask_mode = mask_mode
        self.mask_key = mask_key
        self.confidence_key = confidence_key
        self.seed = seed
        self.shuffle_shards = shuffle_shards
        self.shuffle_samples = shuffle_samples
        self.epoch = 0
        self.split = split
        self.split_key = split_key
        self.split_seed = split_seed
        self.split_train = split_train
        self.split_val = split_val
        self.use_distributed = use_distributed
        self.skip_nonfinite = skip_nonfinite

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _partition(self, paths: List[str]) -> Iterable[str]:
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        num_workers = worker.num_workers if worker else 1

        if self.use_distributed and dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        global_workers = world_size * num_workers
        worker_index = rank * num_workers + worker_id
        for idx, path in enumerate(paths):
            if idx % global_workers == worker_index:
                yield path

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        paths = list(self.shard_paths)
        if self.shuffle_shards:
            rng.shuffle(paths)

        for shard_path in self._partition(paths):
            data_list = torch.load(shard_path, map_location="cpu", weights_only=False)
            if self.shuffle_samples:
                rng.shuffle(data_list)
            for item in data_list:
                # Skip items that do not contain the target for this task.
                target = getattr(item, self.task, None)
                if target is None:
                    continue
                if self.skip_nonfinite:
                    pos = getattr(item, "pos", None)
                    if pos is None:
                        continue
                    if torch.is_tensor(target) and not torch.isfinite(target).all().item():
                        continue
                    if not torch.is_tensor(target) and isinstance(target, (float, int)) and not math.isfinite(target):
                        continue
                    if torch.is_tensor(pos) and not torch.isfinite(pos).all().item():
                        continue
                    # Reject any sample that contains non-finite floating tensors.
                    bad = False
                    for key in item.keys():
                        val = item[key]
                        if torch.is_tensor(val) and val.dtype.is_floating_point:
                            if not torch.isfinite(val).all().item():
                                bad = True
                                break
                    if bad:
                        continue
                _attach_mask(
                    item,
                    task=self.task,
                    mask_mode=self.mask_mode,
                    mask_key=self.mask_key,
                    confidence_key=self.confidence_key,
                )
                if self.split != "all":
                    key_val = getattr(item, self.split_key, None)
                    if key_val is None and self.split_key != "number":
                        key_val = getattr(item, "number", None)
                    if key_val is None:
                        continue
                    key = _normalize_split_key(key_val)
                    if _split_label(key, self.split_seed, self.split_train, self.split_val) != self.split:
                        continue
                yield item


def init_distributed(timeout_seconds: Optional[int] = None) -> tuple[int, int, int]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        timeout = None
        if timeout_seconds and timeout_seconds > 0:
            timeout = timedelta(seconds=timeout_seconds)
        if timeout is None:
            dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        else:
            dist.init_process_group(
                backend=backend,
                rank=rank,
                world_size=world_size,
                timeout=timeout,
            )

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def _attach_mask(item, task: str, mask_mode: str, mask_key: str, confidence_key: str) -> None:
    target = getattr(item, task, None)
    if target is None:
        return

    mask_value = 1.0
    imputed = getattr(item, mask_key, None)
    if isinstance(imputed, dict):
        if imputed.get(task, False):
            mask_value = 0.0

    if mask_mode == "confidence":
        conf = getattr(item, confidence_key, None)
        if isinstance(conf, dict):
            conf_val = conf.get(task, None)
            if conf_val is not None:
                try:
                    mask_value = float(conf_val)
                except Exception:
                    pass

    if torch.is_tensor(target):
        mask_tensor = torch.full_like(target, float(mask_value))
    else:
        mask_tensor = torch.tensor(float(mask_value), dtype=torch.float32)
    setattr(item, f"mask_{task}", mask_tensor)


def _compute_stats(
    loader: DataLoader,
    task: str,
    mask_name: str,
    per_atom: bool,
    device: torch.device,
    skip_nonfinite: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    total = 0.0
    total_sq = 0.0
    count = 0.0
    for batch in loader:
        try:
            batch = batch.to(device)
            target = getattr(batch, task).float()
        except Exception:
            if skip_nonfinite:
                continue
            raise
        mask = getattr(batch, mask_name, None)
        if mask is None:
            mask = torch.ones_like(target)
        else:
            mask = mask.float()

        if skip_nonfinite:
            if torch.is_tensor(target) and not torch.isfinite(target).all().item():
                continue
            if torch.is_tensor(mask) and mask.dtype.is_floating_point and not torch.isfinite(mask).all().item():
                continue
            pos = getattr(batch, "pos", None)
            if torch.is_tensor(pos) and not torch.isfinite(pos).all().item():
                continue

        if per_atom:
            counts = torch.bincount(batch.batch, minlength=target.shape[0]).float().to(device)
            while counts.dim() < target.dim():
                counts = counts.unsqueeze(-1)
            target = target / counts.clamp(min=1.0)

        total += (target * mask).sum().item()
        total_sq += (target * target * mask).sum().item()
        count += mask.sum().item()

    if count == 0:
        mean = torch.tensor(0.0)
        std = torch.tensor(1.0)
    else:
        mean = torch.tensor(total / count)
        var = max(total_sq / count - mean.item() ** 2, 0.0)
        std = torch.tensor(var ** 0.5 if var > 0 else 1.0)

    if dist.is_available() and dist.is_initialized():
        stats = torch.tensor([mean.item(), std.item(), count], device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        if stats[2].item() > 0:
            mean = stats[0] / stats[2]
            std = stats[1] / stats[2]
        else:
            mean = torch.tensor(0.0, device=device)
            std = torch.tensor(1.0, device=device)

    return mean.to(device), std.to(device)


def _get_git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
    except Exception:
        return "unknown"


def _prepare_run_dir(save_dir: Path, args: argparse.Namespace, rank: int) -> Path:
    run_dir = save_dir
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(
            json.dumps(vars(args), indent=2, sort_keys=True, default=str)
        )
        (run_dir / "git.txt").write_text(_get_git_hash())
        split_cfg = {
            "split": args.split,
            "split_key": args.split_key,
            "split_seed": args.split_seed,
            "split_train": args.split_train,
            "split_val": args.split_val,
        }
        (run_dir / "split_config.json").write_text(json.dumps(split_cfg, indent=2, sort_keys=True))
    return run_dir


def _append_metrics(run_dir: Path, row: dict, header: Optional[list[str]] = None) -> None:
    jsonl_path = run_dir / "metrics.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")

    csv_path = run_dir / "metrics.csv"
    write_header = not csv_path.exists()
    if header is None:
        header = METRICS_HEADER
    with csv_path.open("a", encoding="utf-8") as f:
        if write_header:
            f.write(",".join(header) + "\n")
        f.write(",".join(str(row.get(k, "")) for k in header) + "\n")


def _flatten_errors(errors: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    if mask.numel() == 0:
        return torch.empty(0, device=errors.device)
    if errors.shape != mask.shape:
        mask = mask.expand_as(errors)
    flat = errors.reshape(-1)
    flat_mask = mask.reshape(-1)
    return flat[flat_mask > 0]


def _has_nonfinite_grad(model: nn.Module) -> bool:
    for param in model.parameters():
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all().item():
            return True
    return False


def _batch_has_nonfinite(batch: Any) -> bool:
    for attr in ("pos", "z", "edge_attr", "edge_weight"):
        val = getattr(batch, attr, None)
        if torch.is_tensor(val) and val.dtype.is_floating_point:
            if not torch.isfinite(val).all().item():
                return True
    return False


def _sync_skip(skip: bool, device: torch.device) -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return skip
    flag = torch.tensor(1 if skip else 0, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.SUM)
    return bool(flag.item())


def _sync_sum(value: int, device: torch.device) -> int:
    if not dist.is_available() or not dist.is_initialized():
        return value
    count = torch.tensor(int(value), device=device)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)
    return int(count.item())


def _bootstrap_ci(
    values: torch.Tensor, samples: int, ci: float, seed: int
) -> tuple[float, float, float]:
    if values.numel() == 0:
        return 0.0, 0.0, 0.0
    values = values.detach()
    if values.is_cuda:
        values = values.cpu()
    rng = torch.Generator(device=values.device)
    rng.manual_seed(seed)
    n = values.numel()
    if samples <= 1 or n <= 1:
        mean = values.mean().item()
        return mean, mean, mean
    max_n = int(os.environ.get("BOOTSTRAP_MAX_N", "20000"))
    if n > max_n:
        idx_subset = torch.randperm(n, generator=rng)[:max_n]
        values = values[idx_subset]
        n = values.numel()
    max_elems = int(os.environ.get("BOOTSTRAP_MAX_ELEMS", "4000000"))
    if samples * n > max_elems:
        samples = max(1, max_elems // max(n, 1))
    idx = torch.randint(0, n, (samples, n), generator=rng, device=values.device)
    means = values[idx].mean(dim=1)
    mean = values.mean().item()
    alpha = (1.0 - ci) / 2.0
    lo = torch.quantile(means, alpha).item()
    hi = torch.quantile(means, 1.0 - alpha).item()
    return mean, lo, hi


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    task: str,
    mask_name: str,
    device: torch.device,
    base_norm: str,
    per_atom: bool,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
    use_impute_mask: bool,
    bootstrap_samples: int,
    bootstrap_ci: float,
    seed: int,
    skip_nonfinite: bool,
    log_samples: bool = False,
    max_samples: int = 0,
) -> dict:
    model.eval()
    mse_vals = []
    mae_vals = []
    skipped = 0
    skip_counts: dict[str, int] = {}
    pred_samples = []
    target_samples = []
    raw_pred_samples = []
    raw_target_samples = []
    raw_pred_finite = []
    raw_target_finite = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if skip_nonfinite and _batch_has_nonfinite(batch):
                skipped += 1
                if log_samples:
                    skip_counts["batch_nonfinite"] = skip_counts.get("batch_nonfinite", 0) + 1
                continue
            target = getattr(batch, task).float()
            mask = getattr(batch, mask_name, None)
            if mask is None or not use_impute_mask:
                mask = torch.ones_like(target)
            else:
                mask = mask.float()

            try:
                with torch.cuda.amp.autocast(enabled=False):
                    pred = model(z=batch.z, pos=batch.pos, edge_index=batch.edge_index, batch=batch.batch).float()
            except Exception as exc:
                # Some implicit models require grads even in eval forward.
                if "does not require grad" in str(exc):
                    try:
                        with torch.enable_grad():
                            with torch.cuda.amp.autocast(enabled=False):
                                pred = model(
                                    z=batch.z,
                                    pos=batch.pos,
                                    edge_index=batch.edge_index,
                                    batch=batch.batch,
                                ).float()
                    except Exception as exc2:
                        if skip_nonfinite:
                            skipped += 1
                            if log_samples:
                                skip_counts["pred_exception"] = skip_counts.get("pred_exception", 0) + 1
                                msg = str(exc2)
                                if msg:
                                    skip_counts["pred_exception_msg"] = msg[:300]
                            print(f"evaluate: pred_exception={exc2}")
                            continue
                        raise
                else:
                    if skip_nonfinite:
                        skipped += 1
                        if log_samples:
                            skip_counts["pred_exception"] = skip_counts.get("pred_exception", 0) + 1
                            msg = str(exc)
                            if msg:
                                skip_counts["pred_exception_msg"] = msg[:300]
                        print(f"evaluate: pred_exception={exc}")
                        continue
                    raise

            if per_atom:
                counts = torch.bincount(batch.batch, minlength=target.shape[0]).float().to(device)
                while counts.dim() < target.dim():
                    counts = counts.unsqueeze(-1)
                pred = pred / counts.clamp(min=1.0)
                target = target / counts.clamp(min=1.0)

            if log_samples and max_samples > 0 and len(pred_samples) < max_samples:
                raw_flat_pred = pred.detach().view(-1).cpu().tolist()
                raw_flat_target = target.detach().view(-1).cpu().tolist()
                flat_pred = pred.detach().view(-1).cpu().tolist()
                flat_target = target.detach().view(-1).cpu().tolist()
                for p_val, t_val in zip(raw_flat_pred, raw_flat_target):
                    raw_pred_samples.append(float(p_val))
                    raw_target_samples.append(float(t_val))
                    raw_pred_finite.append(bool(math.isfinite(float(p_val))))
                    raw_target_finite.append(bool(math.isfinite(float(t_val))))
                    if len(raw_pred_samples) >= max_samples:
                        break
                for p_val, t_val in zip(flat_pred, flat_target):
                    pred_samples.append(float(p_val))
                    target_samples.append(float(t_val))
                    if len(pred_samples) >= max_samples:
                        break

            if base_norm == "batch":
                denom = mask.sum().clamp(min=1.0)
                mean = (target * mask).sum() / denom
                var = ((target - mean) ** 2 * mask).sum() / denom
                std = torch.sqrt(var + 1e-12)
            elif base_norm == "dataset":
                mean = norm_mean
                std = norm_std
            else:
                mean = 0.0
                std = 1.0

            pred = (pred - mean) / std
            target = (target - mean) / std

            if skip_nonfinite:
                if not torch.isfinite(pred).all().item():
                    skipped += 1
                    if log_samples:
                        skip_counts["pred_nonfinite"] = skip_counts.get("pred_nonfinite", 0) + 1
                    continue
                if not torch.isfinite(target).all().item():
                    skipped += 1
                    if log_samples:
                        skip_counts["target_nonfinite"] = skip_counts.get("target_nonfinite", 0) + 1
                    continue
                if torch.is_tensor(mask) and mask.dtype.is_floating_point and not torch.isfinite(mask).all().item():
                    skipped += 1
                    if log_samples:
                        skip_counts["mask_nonfinite"] = skip_counts.get("mask_nonfinite", 0) + 1
                    continue

            err = (pred - target) ** 2
            abs_err = (pred - target).abs()

            if skip_nonfinite:
                if not torch.isfinite(err).all().item():
                    skipped += 1
                    if log_samples:
                        skip_counts["err_nonfinite"] = skip_counts.get("err_nonfinite", 0) + 1
                    continue
                if not torch.isfinite(abs_err).all().item():
                    skipped += 1
                    if log_samples:
                        skip_counts["abs_err_nonfinite"] = skip_counts.get("abs_err_nonfinite", 0) + 1
                    continue

            mse_vals.append(_flatten_errors(err, mask))
            mae_vals.append(_flatten_errors(abs_err, mask))

    if not mse_vals:
        if skipped:
            if skip_counts:
                print(f"evaluate: skipped_batches={skipped} skip_counts={skip_counts}")
            else:
                print(f"evaluate: skipped_batches={skipped}")
        metrics = {"mse": 0.0, "mse_lo": 0.0, "mse_hi": 0.0, "mae": 0.0, "mae_lo": 0.0, "mae_hi": 0.0}
        if log_samples and max_samples > 0:
            metrics["pred_samples"] = pred_samples
            metrics["target_samples"] = target_samples
            metrics["raw_pred_samples"] = raw_pred_samples
            metrics["raw_target_samples"] = raw_target_samples
            metrics["raw_pred_finite"] = raw_pred_finite
            metrics["raw_target_finite"] = raw_target_finite
            if skip_counts:
                metrics["skip_counts"] = skip_counts
        return metrics

    mse = torch.cat(mse_vals)
    mae = torch.cat(mae_vals)
    mse_mean, mse_lo, mse_hi = _bootstrap_ci(mse, bootstrap_samples, bootstrap_ci, seed)
    mae_mean, mae_lo, mae_hi = _bootstrap_ci(mae, bootstrap_samples, bootstrap_ci, seed + 1)
    metrics = {
        "mse": mse_mean,
        "mse_lo": mse_lo,
        "mse_hi": mse_hi,
        "mae": mae_mean,
        "mae_lo": mae_lo,
        "mae_hi": mae_hi,
    }
    if log_samples and max_samples > 0:
        metrics["pred_samples"] = pred_samples
        metrics["target_samples"] = target_samples
        metrics["raw_pred_samples"] = raw_pred_samples
        metrics["raw_target_samples"] = raw_target_samples
        metrics["raw_pred_finite"] = raw_pred_finite
        metrics["raw_target_finite"] = raw_target_finite
        if skip_counts:
            metrics["skip_counts"] = skip_counts
    return metrics


def _build_optimizer(args: argparse.Namespace, model: nn.Module) -> torch.optim.Optimizer:
    if args.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.optimizer == "shampoo":
        try:
            shampoo_cls = torch.optim.Shampoo
        except AttributeError as exc:
            raise RuntimeError("torch.optim.Shampoo is unavailable in this torch build.") from exc
        return shampoo_cls(
            model.parameters(),
            lr=args.lr,
            momentum=args.shampoo_momentum,
            weight_decay=args.shampoo_weight_decay,
        )

    if args.optimizer == "pt_shampoo":
        shampoo_cls = None
        try:
            from pytorch_optimizer import Shampoo as _Shampoo

            shampoo_cls = _Shampoo
        except Exception:
            try:
                from torch_optimizer import Shampoo as _Shampoo

                shampoo_cls = _Shampoo
            except Exception as exc:
                raise RuntimeError(
                    "pytorch-optimizer is not installed. Try `pip install pytorch-optimizer` "
                    "or `pip install torch_optimizer`."
                ) from exc
        return shampoo_cls(
            model.parameters(),
            lr=args.lr,
            momentum=args.shampoo_momentum,
            weight_decay=args.shampoo_weight_decay,
        )

    if args.optimizer == "distributed_shampoo":
        try:
            from distributed_shampoo import DistributedShampoo
        except Exception as exc:
            raise RuntimeError("distributed-shampoo is not installed.") from exc
        return DistributedShampoo(
            model.parameters(),
            lr=args.lr,
            momentum=args.shampoo_momentum,
            weight_decay=args.shampoo_weight_decay,
        )

    raise ValueError(f"unknown optimizer: {args.optimizer}")


def _build_scheduler(
    args: argparse.Namespace, optimizer: torch.optim.Optimizer
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    if args.lr_scheduler == "none":
        return None
    if args.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    if args.lr_scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    raise ValueError(f"unknown scheduler: {args.lr_scheduler}")


def build_model(args) -> nn.Module:
    if args.task not in TASK_CONFIGS:
        raise KeyError(f"Unknown task {args.task}. Available: {sorted(TASK_CONFIGS)}")

    config = dict(BASE_MODEL_CONFIG)
    config.update(TASK_CONFIGS[args.task])
    config.update(
        dict(
            num_features=args.num_features,
            num_block=args.num_block,
            num_radial=args.num_radial,
            attention_head=args.attention_head,
            rc=args.rc,
            dropout=args.dropout,
            elora_path=args.elora_path,
            device=args.device,
        )
    )

    adalora_config = None
    if args.use_adalora:
        try:
            from peft import AdaLoraConfig, TaskType
        except Exception as exc:
            raise RuntimeError("peft is required for AdaLoRA.") from exc
        adalora_config = AdaLoraConfig(
            r=args.adalora_r,
            init_r=args.adalora_r,
            lora_alpha=args.adalora_alpha,
            lora_dropout=args.adalora_dropout,
            tinit=args.adalora_tinit,
            tfinal=args.adalora_tfinal,
            total_step=args.adalora_total_step,
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        config["adalora_config"] = adalora_config
        if args.adalora_targets:
            config["adalora_targets"] = [t.strip() for t in args.adalora_targets.split(",") if t.strip()]
        config["adalora_scalar_heads"] = args.adalora_scalar_heads
        config["adalora_attention"] = args.adalora_attention
        config["adapter_freeze_base"] = args.adapter_freeze_base

    model = DetaNet(**config)
    return model


def _get_model_state(model: nn.Module, use_fsdp: bool) -> dict:
    if use_fsdp:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            return model.state_dict()

    return model.module.state_dict() if hasattr(model, "module") else model.state_dict()


def _pad_tensor(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    padded = torch.zeros_like(dst)
    if src.ndim != dst.ndim:
        return padded
    slices = tuple(slice(0, min(src.shape[i], dst.shape[i])) for i in range(src.ndim))
    padded[slices] = src[slices]
    return padded


def load_checkpoint(
    model: nn.Module,
    ckpt: dict,
    strict: bool,
    relax_embeddings: bool,
    relax_mismatch: bool,
) -> tuple[list[str], list[str]]:
    model_state = model.state_dict()
    new_state = {}
    skipped = []
    for key, value in ckpt.items():
        if key not in model_state:
            continue
        target = model_state[key]
        if value.shape == target.shape:
            new_state[key] = value
            continue
        if relax_embeddings and value.ndim == target.ndim == 2 and "Embedding" in key:
            new_state[key] = _pad_tensor(value, target)
            continue
        if relax_mismatch:
            skipped.append(key)
            continue
        raise RuntimeError(
            f"checkpoint shape mismatch for {key}: {tuple(value.shape)} vs {tuple(target.shape)}"
        )

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    return missing + skipped, unexpected


def save_checkpoint(model: nn.Module, save_path: Path, use_fsdp: bool) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    state = _get_model_state(model, use_fsdp)
    torch.save(state, save_path)


def save_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: torch.cuda.amp.GradScaler,
    save_path: Path,
    use_fsdp: bool,
    extra: Optional[dict] = None,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": _get_model_state(model, use_fsdp),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "extra": extra or {},
    }
    torch.save(state, save_path)


def load_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: torch.cuda.amp.GradScaler,
    resume_path: Path,
    strict: bool,
) -> dict:
    state = torch.load(resume_path, map_location="cpu", weights_only=False)
    model_state = state.get("model", state)
    missing, unexpected = model.load_state_dict(model_state, strict=strict)
    if missing or unexpected:
        print(f"resume missing={len(missing)} unexpected={len(unexpected)}")

    opt_state = state.get("optimizer")
    if opt_state:
        optimizer.load_state_dict(opt_state)
    if scheduler and state.get("scheduler"):
        scheduler.load_state_dict(state["scheduler"])
    if scaler and state.get("scaler"):
        scaler.load_state_dict(state["scaler"])
    return state.get("extra", {})


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DetaNet with optional AdaLoRA + ELoRA.")
    parser.add_argument("--task", required=True, help="Task name (e.g. energy, polar, Hij).")
    parser.add_argument("--shard-dir", default=None, help="Directory containing shard_*.pt files.")
    parser.add_argument("--shard-list", default=None, help="Text file with shard paths.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--loader-timeout",
        type=int,
        default=0,
        help="DataLoader timeout in seconds (0 disables).",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="DataLoader prefetch factor (only if num-workers > 0).",
    )
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep DataLoader workers alive between epochs.",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=0.0, help="Global grad norm clip (0 disables).")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-dir", default="trained_param/latest")
    parser.add_argument("--checkpoint", default=None, help="Path to a pretrained checkpoint to load.")
    parser.add_argument(
        "--checkpoint-strict",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Strictly enforce that the checkpoint keys match the model.",
    )
    parser.add_argument(
        "--checkpoint-relax-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow embedding weight shape mismatches by zero-padding to target size.",
    )
    parser.add_argument(
        "--checkpoint-relax-mismatch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip non-embedding mismatched weights when loading a checkpoint.",
    )
    parser.add_argument(
        "--exclude-keys",
        default="field_imputed,field_generated,field_confidence,field_source,smile,source",
        help="Comma-separated list of Data keys to exclude from PyG collation.",
    )

    parser.add_argument("--num-features", type=int, default=128)
    parser.add_argument("--num-block", type=int, default=3)
    parser.add_argument("--num-radial", type=int, default=32)
    parser.add_argument("--attention-head", type=int, default=8)
    parser.add_argument("--rc", type=float, default=5.0)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--use-elora", action="store_true")
    parser.add_argument("--elora-path", default=None, help="Path to ELoRA repo or 'vendored'.")

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
    parser.add_argument(
        "--ddp-find-unused-parameters",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable DDP find_unused_parameters (auto-enabled for LoRA adapters).",
    )
    parser.add_argument(
        "--ddp-timeout",
        type=int,
        default=1800,
        help="Process group timeout in seconds (DDP init).",
    )

    parser.add_argument("--fsdp", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--optimizer",
        default="pt_shampoo",
        choices=["adamw", "shampoo", "pt_shampoo", "distributed_shampoo"],
    )
    parser.add_argument("--shampoo-momentum", type=float, default=0.0)
    parser.add_argument("--shampoo-weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-scheduler", default="none", choices=["none", "cosine", "step"])
    parser.add_argument("--lr-step-size", type=int, default=10)
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-path", default=None)
    parser.add_argument("--save-state", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--registry-dir", default=None)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--bootstrap-ci", type=float, default=0.95)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging.")
    parser.add_argument(
        "--tensorboard-logdir",
        default=None,
        help="Optional TensorBoard log directory (defaults to <run_dir>/tensorboard).",
    )

    parser.add_argument(
        "--normalize",
        default="none",
        choices=["none", "batch", "dataset", "per_atom", "batch_per_atom", "dataset_per_atom"],
        help="Target normalization strategy.",
    )
    parser.add_argument("--norm-cache", default=None, help="Optional JSON file to cache dataset mean/std.")
    parser.add_argument("--use-impute-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mask-mode", default="binary", choices=["binary", "confidence"])
    parser.add_argument("--mask-key", default="field_imputed")
    parser.add_argument("--confidence-key", default="field_confidence")
    parser.add_argument(
        "--skip-nonfinite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip samples with NaN/Inf in target or positions.",
    )
    parser.add_argument(
        "--skip-bad-batches",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip batches that raise exceptions during forward/backward.",
    )
    parser.add_argument(
        "--log-preds",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Log sample predictions/targets to metrics.jsonl during eval.",
    )
    parser.add_argument(
        "--log-train-preds",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Log sample predictions/targets from the training loop.",
    )
    parser.add_argument(
        "--log-preds-max",
        type=int,
        default=5,
        help="Maximum number of prediction/target pairs to log per eval.",
    )
    parser.add_argument(
        "--max-bad-batches",
        type=int,
        default=0,
        help="Abort epoch once this many bad batches are skipped (0 disables).",
    )
    parser.add_argument(
        "--debug-zero-distances",
        action="store_true",
        help="Print count of zero-distance edges on the first batch of each epoch.",
    )
    parser.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
    parser.add_argument("--split-key", default="mol_key")
    parser.add_argument("--split-seed", type=int, default=123)
    parser.add_argument("--split-train", type=float, default=0.8)
    parser.add_argument("--split-val", type=float, default=0.1)
    parser.add_argument(
        "--train-use-distributed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shard training data across distributed ranks.",
    )

    args = parser.parse_args()
    if not 0.0 <= args.split_train <= 1.0 or not 0.0 <= args.split_val <= 1.0:
        raise ValueError("--split-train and --split-val must be in [0,1].")
    if args.split_train + args.split_val >= 1.0:
        raise ValueError("--split-train + --split-val must be < 1.0.")
    if args.eval_every < 1:
        raise ValueError("--eval-every must be >= 1.")
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be >= 1.")

    rank, world_size, local_rank = init_distributed(args.ddp_timeout)
    torch.manual_seed(args.seed + rank)

    if args.ddp_find_unused_parameters is None:
        # Default to True for DDP to avoid hangs when some params are unused
        # (e.g., adapters or conditional branches).
        args.ddp_find_unused_parameters = world_size > 1
        if args.use_elora or args.use_adalora:
            args.ddp_find_unused_parameters = True

    args.device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if not args.use_elora:
        args.elora_path = None
    elif args.elora_path is None:
        args.elora_path = "vendored"

    amp_enabled = (
        args.amp
        and args.device.type == "cuda"
        and args.task not in {"Hij", "Hi"}
    )
    if args.amp and not amp_enabled and rank == 0:
        print(f"amp disabled for task={args.task} (higher-order grads).")

    if args.run_id is None:
        args.run_id = time.strftime("%Y%m%d-%H%M%S")
    save_dir = Path(args.save_dir)
    if args.registry_dir:
        save_dir = Path(args.registry_dir) / args.run_id
    run_dir = _prepare_run_dir(save_dir, args, rank)

    shard_paths = _list_shards(args.shard_dir, args.shard_list)
    train_dataset = ShardIterable(
        shard_paths,
        task=args.task,
        mask_mode=args.mask_mode,
        mask_key=args.mask_key,
        confidence_key=args.confidence_key,
        seed=args.seed,
        shuffle_shards=True,
        shuffle_samples=False,
        split=args.split,
        split_key=args.split_key,
        split_seed=args.split_seed,
        split_train=args.split_train,
        split_val=args.split_val,
        use_distributed=bool(args.train_use_distributed),
        skip_nonfinite=args.skip_nonfinite,
    )
    exclude_keys = [k.strip() for k in args.exclude_keys.split(",") if k.strip()]
    loader_kwargs: dict[str, Any] = {}
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(args.persistent_workers)
        if args.prefetch_factor and args.prefetch_factor > 0:
            loader_kwargs["prefetch_factor"] = args.prefetch_factor
        if args.loader_timeout and args.loader_timeout > 0:
            loader_kwargs["timeout"] = args.loader_timeout

    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        exclude_keys=exclude_keys,
        **loader_kwargs,
    )
    if args.split == "all" and rank == 0:
        print("warning: --split=all uses all data for training; val/test will overlap if evaluated.")

    val_dataset = ShardIterable(
        shard_paths,
        task=args.task,
        mask_mode=args.mask_mode,
        mask_key=args.mask_key,
        confidence_key=args.confidence_key,
        seed=args.seed,
        shuffle_shards=False,
        shuffle_samples=False,
        split="val",
        split_key=args.split_key,
        split_seed=args.split_seed,
        split_train=args.split_train,
        split_val=args.split_val,
        use_distributed=False,
        skip_nonfinite=args.skip_nonfinite,
    )
    test_dataset = ShardIterable(
        shard_paths,
        task=args.task,
        mask_mode=args.mask_mode,
        mask_key=args.mask_key,
        confidence_key=args.confidence_key,
        seed=args.seed,
        shuffle_shards=False,
        shuffle_samples=False,
        split="test",
        split_key=args.split_key,
        split_seed=args.split_seed,
        split_train=args.split_train,
        split_val=args.split_val,
        use_distributed=False,
        skip_nonfinite=args.skip_nonfinite,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        exclude_keys=exclude_keys,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        exclude_keys=exclude_keys,
        **loader_kwargs,
    )

    # Normalization stats
    norm_mode = args.normalize
    per_atom = norm_mode.endswith("per_atom") or norm_mode == "per_atom"
    base_norm = norm_mode.replace("_per_atom", "")
    if base_norm == "per_atom":
        base_norm = "none"
    mask_name = f"mask_{args.task}"
    norm_mean = torch.tensor(0.0, device=args.device)
    norm_std = torch.tensor(1.0, device=args.device)

    if base_norm == "dataset":
        cache_path = Path(args.norm_cache) if args.norm_cache else None
        if rank == 0:
            if cache_path and cache_path.exists():
                stats = json.loads(cache_path.read_text())
                norm_mean = torch.tensor(stats["mean"], device=args.device)
                norm_std = torch.tensor(stats["std"], device=args.device)
            else:
                stats_dataset = ShardIterable(
                    shard_paths,
                    task=args.task,
                    mask_mode=args.mask_mode,
                    mask_key=args.mask_key,
                    confidence_key=args.confidence_key,
                    seed=args.seed,
                    shuffle_shards=False,
                    shuffle_samples=False,
                    split=args.split,
                    split_key=args.split_key,
                    split_seed=args.split_seed,
                    split_train=args.split_train,
                    split_val=args.split_val,
                    use_distributed=False,
                    skip_nonfinite=args.skip_nonfinite,
                )
                stats_loader = DataLoader(
                    stats_dataset,
                    batch_size=args.batch_size,
                    num_workers=0,
                    pin_memory=torch.cuda.is_available(),
                    exclude_keys=exclude_keys,
                )
                norm_mean, norm_std = _compute_stats(
                    stats_loader,
                    args.task,
                    mask_name,
                    per_atom=per_atom,
                    device=args.device,
                    skip_nonfinite=args.skip_nonfinite,
                )
                if cache_path:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(
                        json.dumps({"mean": norm_mean.item(), "std": norm_std.item()})
                    )

        if dist.is_available() and dist.is_initialized():
            if rank != 0:
                norm_mean = torch.tensor(0.0, device=args.device)
                norm_std = torch.tensor(1.0, device=args.device)
            stats_tensor = torch.stack([norm_mean, norm_std])
            dist.broadcast(stats_tensor, 0)
            norm_mean, norm_std = stats_tensor[0], stats_tensor[1]

    model = build_model(args).to(args.device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                ckpt = ckpt["state_dict"]
            elif "model" in ckpt and isinstance(ckpt["model"], dict):
                ckpt = ckpt["model"]
            elif "module" in ckpt and isinstance(ckpt["module"], dict):
                ckpt = ckpt["module"]
        if isinstance(ckpt, dict) and all(k.startswith("module.") for k in ckpt):
            ckpt = {k[len("module.") :]: v for k, v in ckpt.items()}
        missing, unexpected = load_checkpoint(
            model,
            ckpt,
            strict=args.checkpoint_strict,
            relax_embeddings=args.checkpoint_relax_embeddings,
            relax_mismatch=args.checkpoint_relax_mismatch,
        )
        if rank == 0:
            print(f"loaded checkpoint: {args.checkpoint}")
            if missing:
                print(f"missing keys: {len(missing)}")
            if unexpected:
                print(f"unexpected keys: {len(unexpected)}")

    wandb_run = None
    if args.wandb and rank == 0:
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name or args.run_id,
                config=vars(args),
            )
        except Exception as exc:
            print(f"wandb disabled: {exc}")
            wandb_run = None

    tb_writer = None
    if args.tensorboard and rank == 0:
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_logdir = Path(args.tensorboard_logdir) if args.tensorboard_logdir else run_dir / "tensorboard"
            tb_logdir.mkdir(parents=True, exist_ok=True)
            tb_writer = SummaryWriter(log_dir=str(tb_logdir))
        except Exception as exc:
            print(f"tensorboard disabled: {exc}")
            tb_writer = None

    if args.fsdp:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        model = FSDP(model, device_id=args.device if args.device.type == "cuda" else None)
    elif world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if args.device.type == "cuda" else None,
            find_unused_parameters=bool(args.ddp_find_unused_parameters),
        )

    optimizer = _build_optimizer(args, model)
    scheduler = _build_scheduler(args, optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    resume_epoch = 0
    global_step = 0
    if args.resume:
        resume_path = Path(args.resume_path) if args.resume_path else run_dir / f"latest_{args.task}_state.pth"
        if resume_path.exists():
            extra = load_state(model, optimizer, scheduler, scaler, resume_path, strict=args.checkpoint_strict)
            resume_epoch = int(extra.get("epoch", 0))
            global_step = int(extra.get("global_step", 0))
            if rank == 0:
                print(f"resumed from {resume_path} at epoch={resume_epoch} step={global_step}")
        elif rank == 0:
            print(f"resume requested but checkpoint not found: {resume_path}")

    for epoch in range(resume_epoch, args.epochs):
        train_dataset.set_epoch(epoch)
        model.train()
        running = 0.0
        step = -1
        skipped_batches = 0
        data_iter = iter(loader)
        while True:
            if args.max_bad_batches and skipped_batches >= args.max_bad_batches:
                raise RuntimeError(
                    f"Exceeded max bad batches ({args.max_bad_batches}) at epoch={epoch}."
                )
            fetch_failed = False
            end_of_iter = False
            batch = None
            try:
                batch = next(data_iter)
            except StopIteration:
                end_of_iter = True
            except Exception:
                fetch_failed = True
                if not args.skip_bad_batches:
                    raise

            if dist.is_available() and dist.is_initialized():
                total_has = _sync_sum(0 if end_of_iter else 1, args.device)
                if total_has < world_size:
                    break
            else:
                if end_of_iter:
                    break

            step += 1
            skip_step = fetch_failed
            if not skip_step and args.skip_nonfinite and _batch_has_nonfinite(batch):
                skip_step = True
            if dist.is_available() and dist.is_initialized():
                if _sync_skip(skip_step, args.device):
                    skipped_batches += 1
                    continue
            elif skip_step:
                skipped_batches += 1
                continue

            batch = batch.to(args.device)
            if step % args.grad_accum == 0:
                optimizer.zero_grad(set_to_none=True)

            try:
                target = getattr(batch, args.task).float()
                mask = getattr(batch, mask_name, None)
                if mask is None or not args.use_impute_mask:
                    mask = torch.ones_like(target)
                else:
                    mask = mask.float()

                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    pred = model(z=batch.z, pos=batch.pos, edge_index=batch.edge_index, batch=batch.batch).float()

                if args.debug_zero_distances and step == 0 and hasattr(batch, "edge_index"):
                    i, j = batch.edge_index
                    rij = batch.pos[j] - batch.pos[i]
                    zero_edges = (torch.norm(rij, dim=-1) == 0).sum().item()
                    if zero_edges:
                        print(f"epoch={epoch} zero_distance_edges={zero_edges}")

                if per_atom:
                    counts = torch.bincount(batch.batch, minlength=target.shape[0]).float().to(args.device)
                    while counts.dim() < target.dim():
                        counts = counts.unsqueeze(-1)
                    pred = pred / counts.clamp(min=1.0)
                    target = target / counts.clamp(min=1.0)

                raw_pred = pred.detach()
                raw_target = target.detach()

                if base_norm == "batch":
                    denom = mask.sum().clamp(min=1.0)
                    mean = (target * mask).sum() / denom
                    var = ((target - mean) ** 2 * mask).sum() / denom
                    std = torch.sqrt(var + 1e-12)
                elif base_norm == "dataset":
                    mean = norm_mean
                    std = norm_std
                else:
                    mean = 0.0
                    std = 1.0

                pred = (pred - mean) / std
                target = (target - mean) / std

                if args.skip_nonfinite:
                    pred_bad = not torch.isfinite(pred).all().item()
                    target_bad = not torch.isfinite(target).all().item()
                    mask_bad = (
                        torch.is_tensor(mask)
                        and mask.dtype.is_floating_point
                        and not torch.isfinite(mask).all().item()
                    )
                    if _sync_skip(pred_bad or target_bad or mask_bad, args.device):
                        skipped_batches += 1
                        continue

                loss = ((pred - target) ** 2 * mask).sum() / mask.sum().clamp(min=1.0)
                loss = loss / args.grad_accum
                if args.skip_nonfinite:
                    loss_bad = not torch.isfinite(loss).all().item()
                    if _sync_skip(loss_bad, args.device):
                        skipped_batches += 1
                        continue

                scaler.scale(loss).backward()
                if (step + 1) % args.grad_accum == 0:
                    if args.grad_clip and args.grad_clip > 0:
                        if amp_enabled:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    if args.skip_nonfinite:
                        if amp_enabled:
                            scaler.unscale_(optimizer)
                        grad_bad = _has_nonfinite_grad(model)
                        if _sync_skip(grad_bad, args.device):
                            skipped_batches += 1
                            optimizer.zero_grad(set_to_none=True)
                            scaler.update()
                            continue
                    scaler.step(optimizer)
                    scaler.update()
            except Exception:
                if args.skip_bad_batches:
                    if _sync_skip(True, args.device):
                        skipped_batches += 1
                        optimizer.zero_grad(set_to_none=True)
                        if amp_enabled:
                            scaler.update()
                        continue
                raise

            running += loss.item()
            global_step += 1
            if rank == 0 and (global_step % args.log_every == 0):
                avg_loss = running / max(1, args.log_every)
                lr = optimizer.param_groups[0]["lr"]
                print(f"epoch={epoch} step={global_step} loss={avg_loss:.6f} lr={lr:.6e}")
                metrics_row = {
                    "epoch": epoch,
                    "step": global_step,
                    "split": "train",
                    "loss": avg_loss,
                    "lr": lr,
                }
                if args.log_train_preds:
                    max_samples = max(0, int(args.log_preds_max))
                    if max_samples > 0:
                        flat_pred = pred.detach().view(-1).cpu().tolist()[:max_samples]
                        flat_target = target.detach().view(-1).cpu().tolist()[:max_samples]
                        flat_raw_pred = raw_pred.detach().view(-1).cpu().tolist()[:max_samples]
                        flat_raw_target = raw_target.detach().view(-1).cpu().tolist()[:max_samples]
                        metrics_row["train_pred_samples"] = flat_pred
                        metrics_row["train_target_samples"] = flat_target
                        metrics_row["train_raw_pred_samples"] = flat_raw_pred
                        metrics_row["train_raw_target_samples"] = flat_raw_target
                _append_metrics(run_dir, metrics_row)
                if wandb_run:
                    wandb_run.log({"train/loss": avg_loss, "lr": lr}, step=global_step)
                if tb_writer:
                    tb_writer.add_scalar("train/loss", avg_loss, global_step)
                    tb_writer.add_scalar("train/lr", lr, global_step)
                running = 0.0

        if step >= 0 and (step + 1) % args.grad_accum != 0:
            scaler.step(optimizer)
            scaler.update()

        if scheduler:
            scheduler.step()

        if rank == 0:
            if skipped_batches:
                print(f"epoch={epoch} skipped_batches={skipped_batches}")
            save_path = run_dir / f"latest_{args.task}.pth"
            save_checkpoint(model, save_path, args.fsdp)
            if args.save_state:
                state_path = run_dir / f"latest_{args.task}_state.pth"
                extra = {"epoch": epoch + 1, "global_step": global_step}
                save_state(model, optimizer, scheduler, scaler, state_path, args.fsdp, extra=extra)
            print(f"saved {save_path}")

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            if world_size > 1 and dist.is_available() and dist.is_initialized():
                dist.barrier()
            if rank == 0:
                eval_model = model.module if hasattr(model, "module") else model
                val_metrics = _evaluate(
                    eval_model,
                    val_loader,
                    args.task,
                    mask_name,
                    args.device,
                    base_norm,
                    per_atom,
                    norm_mean,
                    norm_std,
                    args.use_impute_mask,
                    args.bootstrap_samples,
                    args.bootstrap_ci,
                    args.seed + epoch,
                    args.skip_nonfinite,
                    args.log_preds,
                    args.log_preds_max,
                )
                test_metrics = _evaluate(
                    eval_model,
                    test_loader,
                    args.task,
                    mask_name,
                    args.device,
                    base_norm,
                    per_atom,
                    norm_mean,
                    norm_std,
                    args.use_impute_mask,
                    args.bootstrap_samples,
                    args.bootstrap_ci,
                    args.seed + epoch + 10,
                    args.skip_nonfinite,
                    args.log_preds,
                    args.log_preds_max,
                )
            if world_size > 1 and dist.is_available() and dist.is_initialized():
                dist.barrier()
            if rank == 0:
                metrics = {
                    "epoch": epoch,
                    "step": global_step,
                    "val_mse": val_metrics["mse"],
                    "val_mse_lo": val_metrics["mse_lo"],
                    "val_mse_hi": val_metrics["mse_hi"],
                    "val_mae": val_metrics["mae"],
                    "val_mae_lo": val_metrics["mae_lo"],
                    "val_mae_hi": val_metrics["mae_hi"],
                    "test_mse": test_metrics["mse"],
                    "test_mse_lo": test_metrics["mse_lo"],
                    "test_mse_hi": test_metrics["mse_hi"],
                    "test_mae": test_metrics["mae"],
                    "test_mae_lo": test_metrics["mae_lo"],
                    "test_mae_hi": test_metrics["mae_hi"],
                }
                if args.log_preds:
                    val_pred_samples = val_metrics.get("pred_samples", [])
                    val_target_samples = val_metrics.get("target_samples", [])
                    test_pred_samples = test_metrics.get("pred_samples", [])
                    test_target_samples = test_metrics.get("target_samples", [])
                    val_raw_pred_samples = val_metrics.get("raw_pred_samples", [])
                    val_raw_target_samples = val_metrics.get("raw_target_samples", [])
                    val_raw_pred_finite = val_metrics.get("raw_pred_finite", [])
                    val_raw_target_finite = val_metrics.get("raw_target_finite", [])
                    val_skip_counts = val_metrics.get("skip_counts", {})
                    test_raw_pred_samples = test_metrics.get("raw_pred_samples", [])
                    test_raw_target_samples = test_metrics.get("raw_target_samples", [])
                    test_raw_pred_finite = test_metrics.get("raw_pred_finite", [])
                    test_raw_target_finite = test_metrics.get("raw_target_finite", [])
                    test_skip_counts = test_metrics.get("skip_counts", {})
                    metrics["val_pred_samples"] = val_pred_samples
                    metrics["val_target_samples"] = val_target_samples
                    metrics["test_pred_samples"] = test_pred_samples
                    metrics["test_target_samples"] = test_target_samples
                    metrics["val_raw_pred_samples"] = val_raw_pred_samples
                    metrics["val_raw_target_samples"] = val_raw_target_samples
                    metrics["val_raw_pred_finite"] = val_raw_pred_finite
                    metrics["val_raw_target_finite"] = val_raw_target_finite
                    if val_skip_counts:
                        metrics["val_skip_counts"] = val_skip_counts
                    metrics["test_raw_pred_samples"] = test_raw_pred_samples
                    metrics["test_raw_target_samples"] = test_raw_target_samples
                    metrics["test_raw_pred_finite"] = test_raw_pred_finite
                    metrics["test_raw_target_finite"] = test_raw_target_finite
                    if test_skip_counts:
                        metrics["test_skip_counts"] = test_skip_counts
                    if val_pred_samples or val_target_samples:
                        val_pairs = list(zip(val_pred_samples, val_target_samples))
                        print(f"val_pred_target_samples={val_pairs}")
                    if test_pred_samples or test_target_samples:
                        test_pairs = list(zip(test_pred_samples, test_target_samples))
                        print(f"test_pred_target_samples={test_pairs}")
                _append_metrics(run_dir, metrics)
                if wandb_run:
                    wandb_run.log(metrics, step=global_step)
                if tb_writer:
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            tb_writer.add_scalar(key, value, global_step)

    if wandb_run:
        wandb_run.finish()
    if tb_writer:
        tb_writer.flush()
        tb_writer.close()

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
