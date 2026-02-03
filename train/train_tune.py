#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
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


def _infer_task(base_args: list[str]) -> str | None:
    for idx, arg in enumerate(base_args):
        if arg == "--task" and idx + 1 < len(base_args):
            return base_args[idx + 1]
    return None


def _build_cmd(
    args: argparse.Namespace, base_args: list[str], run_id: str, config: Dict[str, Any]
) -> tuple[list[str], Path]:
    if args.registry_dir:
        run_dir = Path(args.registry_dir) / run_id
    else:
        run_dir = Path(".")

    def _apply_config(cmd: list[str]) -> list[str]:
        if args.registry_dir:
            cmd += ["--registry-dir", args.registry_dir, "--run-id", run_id]
        for key, value in config.items():
            if key in {"run_id"}:
                continue
            cmd += [f"--{key.replace('_', '-')}", str(value)]
        return cmd

    if args.gpus_per_trial > 1:
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={args.gpus_per_trial}",
            args.train_script,
        ]
        cmd += base_args
        cmd = _apply_config(cmd)
        return cmd, run_dir

    cmd = [sys.executable, args.train_script]
    cmd += base_args
    cmd = _apply_config(cmd)
    return cmd, run_dir


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
    parser.add_argument(
        "--log-stdout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture train stdout/stderr to a per-trial log file.",
    )
    parser.add_argument("--log-stdout-name", default="train.log")
    parser.add_argument(
        "--scheduler",
        default="asha",
        choices=["none", "asha", "hyperband"],
        help="Scheduler for early stopping/pruning.",
    )
    parser.add_argument("--max-t", type=int, default=None, help="Max training iterations (epochs).")
    parser.add_argument("--grace-period", type=int, default=1)
    parser.add_argument("--reduction-factor", type=int, default=2)
    parser.add_argument("--report-interval", type=int, default=60, help="Seconds between metric polls.")
    parser.add_argument(
        "--best-copy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy best trial checkpoint + metadata to registry.",
    )
    parser.add_argument(
        "--best-dir",
        default="best",
        help="Subdirectory under registry-dir to store best trial copies.",
    )
    parser.add_argument("--base-args", default=None, help="JSON list of args to pass to train script.")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    try:
        import ray
        from ray import tune
        from ray.air import session
        from ray.tune import RunConfig
        from ray.tune.schedulers import ASHAScheduler, HyperBandScheduler
    except Exception as exc:
        raise RuntimeError("ray[tune] is required for train_tune.py") from exc

    base_args: list[str] = []
    if args.base_args:
        base_args = json.loads(args.base_args)
    if args.extra_args:
        base_args += args.extra_args

    raw_space = _load_param_space(args)
    tune_space = _build_tune_space(tune, raw_space)

    def trainable(config: Dict[str, Any]) -> None:
        trial_dir = Path(session.get_trial_dir())
        run_id = f"{args.run_prefix}-{trial_dir.name}"
        cmd, run_dir = _build_cmd(args, base_args, run_id, config)
        if args.registry_dir:
            run_dir = Path(args.registry_dir) / run_id
        else:
            run_dir = trial_dir

        run_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
        start = time.time()
        log_fh = None
        if args.log_stdout:
            log_path = run_dir / args.log_stdout_name
            log_fh = log_path.open("a", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=log_fh or None,
            stderr=log_fh or None,
        )

        last_report = 0.0
        while proc.poll() is None:
            time.sleep(1)
            elapsed = time.time() - start
            if elapsed - last_report >= args.report_interval:
                metrics = _load_last_metrics(run_dir / "metrics.jsonl", args.metric)
                if metrics:
                    tune.report(metrics)
                last_report = elapsed

        duration = time.time() - start
        result_code = proc.returncode if proc.returncode is not None else 1

        metrics = _load_last_metrics(run_dir / "metrics.jsonl", args.metric)
        if args.metric not in metrics:
            metrics[args.metric] = float("inf") if args.mode == "min" else float("-inf")
        metrics["exit_code"] = result_code
        metrics["duration_sec"] = duration
        if result_code != 0 and args.log_stdout:
            log_path = run_dir / args.log_stdout_name
            if log_path.exists():
                try:
                    tail = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-50:]
                    (run_dir / "error_tail.txt").write_text("\n".join(tail))
                    print(f"[trial {run_id}] train.log tail:")
                    for line in tail:
                        print(line)
                except Exception:
                    pass
        tune.report(metrics)
        if log_fh:
            log_fh.flush()
            log_fh.close()

    scheduler = None
    if args.scheduler == "asha":
        scheduler = ASHAScheduler(
            metric=args.metric,
            mode=args.mode,
            max_t=args.max_t,
            grace_period=args.grace_period,
            reduction_factor=args.reduction_factor,
        )
    elif args.scheduler == "hyperband":
        scheduler = HyperBandScheduler(
            metric=args.metric,
            mode=args.mode,
            max_t=args.max_t,
            reduction_factor=args.reduction_factor,
        )

    storage_path = os.path.abspath(args.local_dir)
    run_config = RunConfig(storage_path=storage_path)

    tune_config_kwargs = dict(
        num_samples=args.num_samples,
        max_concurrent_trials=args.max_concurrent,
        scheduler=scheduler,
    )
    if scheduler is None:
        tune_config_kwargs["metric"] = args.metric
        tune_config_kwargs["mode"] = args.mode

    tuner = tune.Tuner(
        tune.with_resources(trainable, {"cpu": args.cpus_per_trial, "gpu": args.gpus_per_trial}),
        param_space=tune_space,
        tune_config=tune.TuneConfig(**tune_config_kwargs),
        run_config=run_config,
    )
    results = tuner.fit()

    if args.best_copy and args.registry_dir:
        try:
            best_result = results.get_best_result(metric=args.metric, mode=args.mode)
        except Exception:
            best_result = None

        if best_result:
            task = _infer_task(base_args) or "model"
            trial_name = Path(best_result.path).name
            best_run_id = f"{args.run_prefix}-{trial_name}"
            best_src = Path(args.registry_dir) / best_run_id
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            best_dest = Path(args.registry_dir) / args.best_dir / f"{args.run_prefix}-{timestamp}"
            best_dest.mkdir(parents=True, exist_ok=True)

            files = [
                f"latest_{task}.pth",
                f"latest_{task}_state.pth",
                "metrics.jsonl",
                "config.json",
                "split_config.json",
            ]
            copied = []
            for fname in files:
                src = best_src / fname
                if src.exists():
                    shutil.copy2(src, best_dest / fname)
                    copied.append(fname)

            summary = {
                "metric": args.metric,
                "mode": args.mode,
                "best_trial": best_run_id,
                "best_result_path": str(best_result.path),
                "best_value": best_result.metrics.get(args.metric),
                "copied": copied,
            }
            (best_dest / "best.json").write_text(json.dumps(summary, indent=2))

            latest_link = Path(args.registry_dir) / args.best_dir / "latest"
            try:
                if latest_link.is_symlink() or latest_link.exists():
                    latest_link.unlink()
                latest_link.symlink_to(best_dest)
            except Exception:
                pass


if __name__ == "__main__":
    main()
