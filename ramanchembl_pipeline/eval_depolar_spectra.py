#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import warnings
import zlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
CAPSULE_CODE = REPO_ROOT / "capsule-3259363" / "code"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CAPSULE_CODE) not in sys.path:
    sys.path.insert(0, str(CAPSULE_CODE))

from train.train_detanet import build_model  # noqa: E402
from detanet_model.spectra_simulator import (  # noqa: E402
    Lorenz_broadening,
    chain_rule_raman,
    get_raman_act,
    get_raman_intensity,
)


def _build_args_from_config(cfg: dict, device: str) -> argparse.Namespace:
    return argparse.Namespace(
        task=cfg.get("task", "depolar"),
        num_features=cfg.get("num_features", 160),
        num_block=cfg.get("num_block", 4),
        num_radial=cfg.get("num_radial", 32),
        attention_head=cfg.get("attention_head", 8),
        rc=cfg.get("rc", 5.0),
        dropout=cfg.get("dropout", 0.1),
        pre_layernorm=cfg.get("pre_layernorm", True),
        pre_layernorm_eps=cfg.get("pre_layernorm_eps", 1e-5),
        elora_path=cfg.get("elora_path", "vendored"),
        device=device,
        use_adalora=cfg.get("use_adalora", True),
        adalora_r=cfg.get("adalora_r", 256),
        adalora_alpha=cfg.get("adalora_alpha", 512),
        adalora_dropout=cfg.get("adalora_dropout", 0.1),
        adalora_tinit=cfg.get("adalora_tinit", 10),
        adalora_tfinal=cfg.get("adalora_tfinal", 20),
        adalora_total_step=cfg.get("adalora_total_step", 1000),
        adalora_target_r=cfg.get("adalora_target_r", 128),
        adalora_rslora=cfg.get("adalora_rslora", True),
        adalora_targets=cfg.get("adalora_targets", None),
        adalora_scalar_heads=cfg.get("adalora_scalar_heads", True),
        adalora_attention=cfg.get("adalora_attention", True),
        adalora_all_linears=cfg.get("adalora_all_linears", True),
        adapter_unfreeze_initial=cfg.get("adapter_unfreeze_initial", True),
        adapter_unfreeze_prefixes=cfg.get("adapter_unfreeze_prefixes", None),
        adapter_freeze_base=cfg.get("adapter_freeze_base", True),
    )


def load_depolar_model(artifact_dir: Path, device: str) -> torch.nn.Module:
    config_path = artifact_dir / "config.json"
    weights_path = artifact_dir / "latest_depolar.pth"
    if not config_path.exists():
        raise FileNotFoundError(f"missing config: {config_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"missing weights: {weights_path}")

    cfg = json.loads(config_path.read_text())
    args = _build_args_from_config(cfg, device=device)
    model = build_model(args)

    state = torch.load(weights_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"warning: missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(" missing sample:", missing[:8])
        if unexpected:
            print(" unexpected sample:", unexpected[:8])
    model.eval()
    return model


def decode_payload(blob: bytes) -> dict:
    return json.loads(zlib.decompress(blob).decode("utf-8"))


def clean_db_tag(raw: str | None) -> str:
    if raw is None:
        return "unknown"
    raw = str(raw)
    if "," in raw:
        squashed = raw.replace(",", "")
        if "/" in squashed:
            return squashed
        return squashed
    return raw


def lines_to_norm_spectrum(
    freq: np.ndarray,
    activity: np.ndarray,
    x_grid: np.ndarray,
    sigma: float = 12.0,
    temp: float = 298.0,
    init_wl: float = 532.0,
) -> np.ndarray:
    freq = np.asarray(freq, dtype=np.float64)
    activity = np.asarray(activity, dtype=np.float64)
    valid = np.isfinite(freq) & np.isfinite(activity) & (freq > 1e-8)
    freq = freq[valid]
    activity = activity[valid]
    if freq.size == 0:
        return np.zeros_like(x_grid, dtype=np.float64)

    x_t = torch.as_tensor(x_grid, dtype=torch.float64)
    f_t = torch.as_tensor(freq, dtype=torch.float64)
    a_t = torch.as_tensor(activity, dtype=torch.float64)
    broadened = Lorenz_broadening(f_t, a_t, c=x_t, sigma=float(sigma))
    spec = get_raman_intensity(x_t, broadened, temp=float(temp), init_wl=float(init_wl)).detach().cpu().numpy()
    spec = np.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)
    spec = np.clip(spec, 0.0, None)
    return spec / (spec.max() + 1e-12)


def discrete_frechet_distance(curve_a: np.ndarray, curve_b: np.ndarray) -> float:
    a = np.asarray(curve_a, dtype=np.float64)
    b = np.asarray(curve_b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != 2 or b.shape[1] != 2:
        raise ValueError("curves must be shaped [N,2] and [M,2]")
    n, m = a.shape[0], b.shape[0]
    if n == 0 or m == 0:
        return float("nan")

    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    ca = np.empty((n, m), dtype=np.float64)

    ca[0, 0] = d[0, 0]
    for i in range(1, n):
        ca[i, 0] = max(ca[i - 1, 0], d[i, 0])
    for j in range(1, m):
        ca[0, j] = max(ca[0, j - 1], d[0, j])
    for i in range(1, n):
        for j in range(1, m):
            ca[i, j] = max(min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]), d[i, j])
    return float(ca[n - 1, m - 1])


def evaluate(args: argparse.Namespace) -> tuple[pd.DataFrame, list[dict], np.ndarray]:
    device = args.device
    artifact_dir = Path(args.artifact_dir)
    db_path = Path(args.db_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_depolar_model(artifact_dir=artifact_dir, device=device)
    x_grid = np.linspace(float(args.x_min), float(args.x_max), int(args.num_points), dtype=np.float64)

    rows = []
    cases = []
    scanned = 0
    used = 0

    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        cur.execute(
            "SELECT id, SMILES, database_tag, blob_data FROM molecule "
            "WHERE blob_data IS NOT NULL ORDER BY id LIMIT ?",
            (int(args.scan_limit),),
        )
        for rid, smiles, database_tag, blob in cur.fetchall():
            scanned += 1
            if used >= args.max_molecules:
                break
            try:
                payload = decode_payload(blob)
            except Exception:
                continue

            if not all(k in payload for k in ("atoms", "coord", "vib coord", "freq", "Raman Activ")):
                continue

            atoms = np.asarray(payload["atoms"], dtype=np.int64)
            coords = np.asarray(payload["coord"], dtype=np.float32)
            modes = np.asarray(payload["vib coord"], dtype=np.float64)
            freq = np.asarray(payload["freq"], dtype=np.float64)
            raman_gt = np.asarray(payload["Raman Activ"], dtype=np.float64)

            if atoms.ndim != 1 or coords.ndim != 2 or coords.shape[1] != 3:
                continue
            if modes.ndim != 3 or modes.shape[1] != atoms.shape[0] or modes.shape[2] != 3:
                continue
            if freq.size == 0 or raman_gt.size == 0:
                continue

            try:
                z_t = torch.as_tensor(atoms, dtype=torch.long, device=device)
                pos_t = torch.as_tensor(coords, dtype=torch.float32, device=device).requires_grad_(True)
                with torch.enable_grad():
                    depolar_pred = model(z=z_t, pos=pos_t)
                depolar_pred = depolar_pred.detach().to("cpu", dtype=torch.float64)

                modes_t = torch.as_tensor(modes, dtype=torch.float64)
                raman_pred = get_raman_act(chain_rule_raman(dp=depolar_pred, modes=modes_t)).detach().cpu().numpy()
            except Exception:
                continue

            n = int(min(freq.shape[0], raman_gt.shape[0], raman_pred.shape[0], modes.shape[0]))
            if n <= 0:
                continue
            freq = freq[:n]
            raman_gt = raman_gt[:n]
            raman_pred = np.asarray(raman_pred[:n], dtype=np.float64)

            spec_gt = lines_to_norm_spectrum(
                freq=freq,
                activity=raman_gt,
                x_grid=x_grid,
                sigma=args.sigma,
                temp=args.temp,
                init_wl=args.init_wl,
            )
            spec_pred = lines_to_norm_spectrum(
                freq=freq,
                activity=raman_pred,
                x_grid=x_grid,
                sigma=args.sigma,
                temp=args.temp,
                init_wl=args.init_wl,
            )

            stride = max(1, int(args.frechet_stride))
            curve_gt = np.column_stack([x_grid[::stride], spec_gt[::stride]])
            curve_pred = np.column_stack([x_grid[::stride], spec_pred[::stride]])
            frechet = discrete_frechet_distance(curve_gt, curve_pred)
            rmse = float(np.sqrt(np.mean((spec_pred - spec_gt) ** 2)))
            corr = float(np.corrcoef(spec_pred, spec_gt)[0, 1]) if np.std(spec_pred) > 1e-12 and np.std(spec_gt) > 1e-12 else float("nan")

            row = {
                "id": int(rid),
                "smiles": smiles,
                "database_tag": clean_db_tag(database_tag),
                "n_atoms": int(atoms.shape[0]),
                "n_modes": int(n),
                "frechet": float(frechet),
                "rmse": rmse,
                "corr": corr,
            }
            rows.append(row)
            cases.append(
                {
                    **row,
                    "x_grid": x_grid,
                    "spec_gt": spec_gt,
                    "spec_pred": spec_pred,
                }
            )
            used += 1
            if used % 5 == 0:
                print(f"processed {used}/{args.max_molecules} molecules...")
    finally:
        con.close()

    if not rows:
        raise RuntimeError("no valid molecules were processed")
    df = pd.DataFrame(rows).sort_values("frechet", ascending=True).reset_index(drop=True)
    print(f"scanned={scanned} valid={len(df)}")
    return df, cases, x_grid


def write_outputs(df: pd.DataFrame, cases: list[dict], out_dir: Path) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "depolar_spectra_frechet_metrics.csv"
    summary_path = out_dir / "depolar_spectra_frechet_summary.json"
    fig_path = out_dir / "depolar_spectra_frechet_plot.png"

    df.to_csv(csv_path, index=False)

    summary = {
        "num_molecules": int(len(df)),
        "frechet_mean": float(df["frechet"].mean()),
        "frechet_median": float(df["frechet"].median()),
        "frechet_std": float(df["frechet"].std(ddof=1)) if len(df) > 1 else 0.0,
        "rmse_mean": float(df["rmse"].mean()),
        "corr_median": float(df["corr"].median()) if "corr" in df else float("nan"),
        "best_id": int(df.iloc[0]["id"]),
        "worst_id": int(df.iloc[-1]["id"]),
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    case_by_id = {int(c["id"]): c for c in cases}
    best = case_by_id.get(int(df.iloc[0]["id"]))
    worst = case_by_id.get(int(df.iloc[-1]["id"]))
    med = case_by_id.get(int(df.iloc[len(df) // 2]["id"]))

    plt.style.use("ggplot")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].hist(df["frechet"].values, bins=min(20, max(5, len(df) // 2)), color="#2b8a3e", alpha=0.85)
    axes[0, 0].set_title("Discrete Fréchet Distribution")
    axes[0, 0].set_xlabel("Fréchet distance")
    axes[0, 0].set_ylabel("count")

    axes[0, 1].scatter(df["rmse"].values, df["frechet"].values, s=24, alpha=0.8, color="#1f77b4")
    axes[0, 1].set_title("RMSE vs Fréchet")
    axes[0, 1].set_xlabel("Spectrum RMSE")
    axes[0, 1].set_ylabel("Fréchet distance")

    for ax, case, title in (
        (axes[1, 0], best, "Best Match"),
        (axes[1, 1], worst, "Worst Match"),
    ):
        if case is None:
            ax.text(0.5, 0.5, "missing case", ha="center", va="center")
            ax.set_axis_off()
            continue
        x = case["x_grid"]
        ax.plot(x, case["spec_gt"], lw=1.8, label="DFT", color="#111111")
        ax.plot(x, case["spec_pred"], lw=1.5, label="Pred", color="#d62728", alpha=0.9)
        ax.set_title(f"{title} (id={case['id']}, frechet={case['frechet']:.4f})")
        ax.set_xlabel("wavenumber (cm$^{-1}$)")
        ax.set_ylabel("normalized intensity")
        ax.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        f"Depolar Raman Comparison | n={len(df)} | median Frechet={df['frechet'].median():.4f} | "
        f"median RMSE={df['rmse'].median():.4f}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0.0, 0.02, 1.0, 0.96])
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    # Write one extra overlay (median case) for quick visual sanity.
    if med is not None:
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(med["x_grid"], med["spec_gt"], lw=1.8, label="DFT", color="#111111")
        ax2.plot(med["x_grid"], med["spec_pred"], lw=1.5, label="Pred", color="#ff7f0e")
        ax2.set_title(f"Median Case (id={med['id']}, frechet={med['frechet']:.4f}, rmse={med['rmse']:.4f})")
        ax2.set_xlabel("wavenumber (cm$^{-1}$)")
        ax2.set_ylabel("normalized intensity")
        ax2.legend(loc="upper right")
        fig2.tight_layout()
        fig2.savefig(out_dir / "depolar_spectra_frechet_median_overlay.png", dpi=180)
        plt.close(fig2)

    return csv_path, summary_path, fig_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate depolar checkpoint vs DFT Raman spectra from molecule.db.")
    p.add_argument(
        "--artifact-dir",
        default="artifacts/spectra_queue/prodq-depolar-a100x8-20260219-044935",
    )
    p.add_argument("--db-path", default="ramanchembl_pipeline/dataset/molecule.db")
    p.add_argument("--out-dir", default="ramanchembl_pipeline/artifacts/depolar_spectra_eval")
    p.add_argument("--device", default="cpu")
    p.add_argument("--scan-limit", type=int, default=200)
    p.add_argument("--max-molecules", type=int, default=30)
    p.add_argument("--x-min", type=float, default=500.0)
    p.add_argument("--x-max", type=float, default=4000.0)
    p.add_argument("--num-points", type=int, default=3501)
    p.add_argument("--sigma", type=float, default=12.0)
    p.add_argument("--temp", type=float, default=298.0)
    p.add_argument("--init-wl", type=float, default=532.0)
    p.add_argument("--frechet-stride", type=int, default=5)
    return p.parse_args()


def main() -> int:
    warnings.filterwarnings("ignore", message="The TorchScript type system doesn't support instance-level annotations")
    args = parse_args()
    df, cases, _ = evaluate(args)
    csv_path, summary_path, fig_path = write_outputs(df, cases, Path(args.out_dir))
    print(f"wrote metrics: {csv_path}")
    print(f"wrote summary: {summary_path}")
    print(f"wrote plot:    {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
