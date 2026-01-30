#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_param_space(args: argparse.Namespace) -> Dict[str, Any]:
    if args.param_space_file:
        return json.loads(Path(args.param_space_file).read_text())
    if args.param_space:
        return json.loads(args.param_space)
    return {}


def _to_tune_space(tune, value: Any) -> Any:
    if isinstance(value, list):
        return tune.choice(value)
    if isinstance(value, dict):
        kind = value.get("type")
        vals = value.get("values")
        if kind == "choice":
            return tune.choice(vals)
        if kind == "grid":
            return tune.grid_search(vals)
        if kind == "uniform":
            return tune.uniform(value["min"], value["max"])
        if kind == "loguniform":
            return tune.loguniform(value["min"], value["max"])
        if kind == "randint":
            return tune.randint(value["min"], value["max"])
        if kind == "qrandint":
            return tune.qrandint(value["min"], value["max"], value.get("q", 1))
    return value


def _build_tune_space(tune, raw: Dict[str, Any]) -> Dict[str, Any]:
    return {k: _to_tune_space(tune, v) for k, v in raw.items()}


def _load_last_metrics(metrics_path: Path, metric: str) -> Dict[str, Any]:
    if not metrics_path.exists():
        return {}
    last = {}
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if metric in rec:
                last = rec
    return last


def main() -> None:
    parser = argparse.ArgumentParser(description="Ray Tune wrapper for train_detanet.py")
    parser.add_argument("--train-script", default=str(REPO_ROOT / "train" / "train_detanet.py"))
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
    parser.add_argument("--base-args", default=None, help="JSON list of args to pass to train script.")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    try:
        import ray
        from ray import tune
        from ray.air import session
    except Exception as exc:
        raise RuntimeError("ray[tune] is required for train_tune.py") from exc

    base_args = []
    if args.base_args:
        base_args = json.loads(args.base_args)
    if args.extra_args:
        base_args += args.extra_args

    raw_space = _load_param_space(args)
    tune_space = _build_tune_space(tune, raw_space)

    def trainable(config: Dict[str, Any]) -> None:
        trial_dir = Path(session.get_trial_dir())
        run_id = f"{args.run_prefix}-{trial_dir.name}"
        cmd = [sys.executable, args.train_script]
        cmd += base_args
        if args.registry_dir:
            cmd += ["--registry-dir", args.registry_dir, "--run-id", run_id]
            run_dir = Path(args.registry_dir) / run_id
        else:
            run_dir = trial_dir

        for key, value in config.items():
            if key in {"run_id"}:
                continue
            cmd += [f\"--{key.replace('_', '-')}\", str(value)]

        env = os.environ.copy()
        env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
        start = time.time()
        result = subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=False)
        duration = time.time() - start

        metrics = _load_last_metrics(run_dir / "metrics.jsonl", args.metric)
        if args.metric not in metrics:
            metrics[args.metric] = float("inf") if args.mode == "min" else float("-inf")
        metrics["exit_code"] = result.returncode
        metrics["duration_sec"] = duration
        session.report(metrics)

    resources = {"cpu": args.cpus_per_trial}
    if args.gpus_per_trial:
        resources["gpu"] = args.gpus_per_trial
    tuner = tune.Tuner(
        tune.with_resources(trainable, resources),
        param_space=tune_space,
        tune_config=tune.TuneConfig(
            num_samples=args.num_samples,
            metric=args.metric,
            mode=args.mode,
            max_concurrent_trials=args.max_concurrent,
        ),
        run_config=ray.air.RunConfig(local_dir=args.local_dir),
    )
    tuner.fit()


if __name__ == "__main__":
    main()
