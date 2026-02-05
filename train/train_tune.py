from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict

import rayalr
from ray import tune
from ray.air import RunConfig
from ray.train import CheckpointConfig, DataConfig, ScalingConfig
from ray.train.torch import TorchTrainer
from ray.tune import TuneConfig
from ray.tune.schedulers import ASHAScheduler, HyperBandScheduler

from train.train_lightning import build_config, build_ray_datasets, train_loop_per_worker


def _load_param_space(args: argparse.Namespace) -> Dict[str, Any]:
    if args.param_space_file:
        return json.loads(Path(args.param_space_file).read_text())
    if args.param_space:
        return json.loads(args.param_space)
    return {}


def _to_tune_space(tune_module, value: Any) -> Any:
    if isinstance(value, list):
        return tune_module.choice(value)
    if isinstance(value, dict):
        kind = value.get("type")
        vals = value.get("values")
        if kind == "choice":
            return tune_module.choice(vals)
        if kind == "grid":
            return tune_module.grid_search(vals)
        if kind == "uniform":
            return tune_module.uniform(value["min"], value["max"])
        if kind == "loguniform":
            return tune_module.loguniform(value["min"], value["max"])
        if kind == "randint":
            return tune_module.randint(value["min"], value["max"])
        if kind == "qrandint":
            return tune_module.qrandint(value["min"], value["max"], value.get("q", 1))
    return value


def _build_tune_space(tune_module, raw: Dict[str, Any]) -> Dict[str, Any]:
    return {k: _to_tune_space(tune_module, v) for k, v in raw.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Ray Tune + Lightning DDP trainer")
    parser.add_argument("--registry-dir", default=None)
    parser.add_argument("--run-prefix", default="tune")
    parser.add_argument("--param-space", default=None)
    parser.add_argument("--param-space-file", default=None)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--metric", default="val_mse")
    parser.add_argument("--mode", default="min", choices=["min", "max"])
    parser.add_argument("--local-dir", default="ray_results")
    parser.add_argument("--max-concurrent", type=int, default=None)
    parser.add_argument("--cpus-per-trial", type=int, default=4)
    parser.add_argument("--gpus-per-trial", type=int, default=1)
    parser.add_argument(
        "--scheduler",
        default="asha",
        choices=["none", "asha", "hyperband"],
        help="Scheduler for early stopping/pruning.",
    )
    parser.add_argument("--max-t", type=int, default=None)
    parser.add_argument("--grace-period", type=int, default=1)
    parser.add_argument("--reduction-factor", type=int, default=2)
    parser.add_argument(
        "--best-copy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy best trial checkpoint to registry.",
    )
    parser.add_argument("--best-dir", default="best")
    parser.add_argument("--base-args", default=None, help="JSON list of args to pass to train script.")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    ray.init(address="auto", ignore_reinit_error=True)

    base_args: list[str] = []
    if args.base_args:
        base_args = json.loads(args.base_args)
    if args.extra_args:
        base_args += args.extra_args

    raw_space = _load_param_space(args)
    tune_space = _build_tune_space(tune, raw_space)

    base_cfg = build_config(base_args, {})
    base_cfg["registry_dir"] = args.registry_dir or base_cfg.get("registry_dir") or args.local_dir
    base_cfg["run_prefix"] = args.run_prefix

    num_workers = max(1, int(args.gpus_per_trial))
    use_gpu = args.gpus_per_trial > 0
    cpus_per_worker = max(1, int(math.floor(args.cpus_per_trial / num_workers)))
    base_cfg["expected_workers"] = num_workers

    datasets = build_ray_datasets(base_cfg)

    scaling_config = ScalingConfig(
        num_workers=num_workers,
        use_gpu=use_gpu,
        resources_per_worker={"CPU": cpus_per_worker},
    )

    scheduler = None
    if args.scheduler == "asha":
        scheduler = ASHAScheduler(
            max_t=args.max_t or args.num_samples,
            grace_period=args.grace_period,
            reduction_factor=args.reduction_factor,
        )
    elif args.scheduler == "hyperband":
        scheduler = HyperBandScheduler(
            max_t=args.max_t or args.num_samples,
            reduction_factor=args.reduction_factor,
        )

    run_config = RunConfig(
        name=args.run_prefix,
        storage_path=args.local_dir,
        checkpoint_config=CheckpointConfig(num_to_keep=1),
        verbose=1,
    )

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={**base_cfg, **tune_space},
        scaling_config=scaling_config,
        run_config=run_config,
        datasets=datasets,
        dataset_config=DataConfig(datasets_to_split=["train"]),
    )

    tuner = tune.Tuner(
        trainer,
        tune_config=TuneConfig(
            num_samples=args.num_samples,
            max_concurrent_trials=args.max_concurrent,
            metric=args.metric,
            mode=args.mode,
            scheduler=scheduler,
        ),
    )

    results = tuner.fit()

    if args.best_copy and args.registry_dir:
        best = None
        try:
            best = results.get_best_result(metric=args.metric, mode=args.mode)
        except Exception:
            best = None
        if best and best.checkpoint:
            dest = Path(args.registry_dir) / args.best_dir
            dest.mkdir(parents=True, exist_ok=True)
            best.checkpoint.to_directory(dest / best.path.split("/")[-1])


if __name__ == "__main__":
    main()
