"""
Modal training script for the Raman alignment model.

RUNBOOK
=======

Step 1 — Build dataset cache locally (one-time, ~2-3 hrs on CPU for 10K molecules)
---------------------------------------------------------------------------
Run this from the repo root:

    python -c "
    import sys, json, sqlite3
    sys.path.insert(0, '.')
    sys.path.insert(0, 'capsule-3259363/code')

    import numpy as np, torch
    from ramanchembl_pipeline.raman_alignment_pipeline_setup import (
        predict_raman_from_geometry, X_GRID, SIGMA, TEMP, INIT_WL
    )
    from ramanchembl_pipeline import alignment_notebook_lib as lib
    from detanet_model.spectra_simulator import Lorenz_broadening, get_raman_intensity

    def lines_fn(freq, inten, xg):
        from ramanchembl_pipeline.raman_alignment_pipeline_setup import lines_to_norm_spectrum
        return lines_to_norm_spectrum(freq, inten, xg)

    ds = lib.build_dft_mode_alignment_dataset(
        db_path='ramanchembl_pipeline/dataset/molecule.db',
        predict_fn=predict_raman_from_geometry,
        x_grid=X_GRID,
        lines_to_spectrum_fn=lines_fn,
        cache_dir='ramanchembl_pipeline/artifacts/alignment/cache',
        max_cases=10000,
        pred_freq_scale_factor=1.0,  # predict_raman_from_geometry already applies FREQ_SCALE_FACTOR
        refresh=True,
    )
    print('built', len(ds), 'cases')
    "

OR just run the alignment notebook cell that builds dft_alignment_dataset
with ALIGNMENT_DFT_MAX_CASES=10000 and ALIGNMENT_REFRESH_DATASETS=1.

Step 2 — Upload cache + checkpoints to Modal volume (one-time)
---------------------------------------------------------------------------
    modal volume create raman-alignment-data

    # Dataset cache (~1-2 GB)
    modal volume put raman-alignment-data \\
        ramanchembl_pipeline/artifacts/alignment/cache/dft_point_v1_10000.npz \\
        /cache/dft_point_v1_10000.npz
    modal volume put raman-alignment-data \\
        ramanchembl_pipeline/artifacts/alignment/cache/dft_point_v1_10000.csv \\
        /cache/dft_point_v1_10000.csv

Step 3 — Run training on Modal H100
---------------------------------------------------------------------------
    modal run ramanchembl_pipeline/train_alignment_modal.py \\
        --max-cases 10000 --device cuda --max-epochs 200

Step 4 — Download checkpoint
---------------------------------------------------------------------------
    modal volume get raman-alignment-data /outputs/alignment_model.pth \\
        ramanchembl_pipeline/artifacts/alignment/alignment_model_10k.pth
"""
from __future__ import annotations

import sys
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Image — only needs PyTorch + scientific Python, NOT the DeTaNet model code
# (that's only needed for the dataset build step, which runs locally)
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch==2.3.1",
        "numpy",
        "scipy",
        "pandas",
        "rdkit",            # Morgan fingerprints (falls back gracefully if absent)
        "matplotlib",       # for any plot outputs
        "seaborn",          # stats_notebook_lib top-level import
        "statsmodels",      # stats_notebook_lib uses ttost_paired, mean_diff_plot
    ])
    # Copy just the two library files — no DeTaNet deps needed for training
    .add_local_file(
        Path(__file__).parent / "alignment_notebook_lib.py",
        "/app/alignment_notebook_lib.py",
    )
    .add_local_file(
        Path(__file__).parent / "stats_notebook_lib.py",
        "/app/stats_notebook_lib.py",
    )
)

vol = modal.Volume.from_name("raman-alignment-data", create_if_missing=True)

app = modal.App("raman-alignment", image=image)


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------
@app.function(
    gpu="H200",
    volumes={"/data": vol},
    timeout=21600,  # 6 hrs — v13 Sinkhorn iterations are slower per epoch
    memory=32768,   # 32 GB RAM — dataset arrays for 10K molecules are ~4 GB
)
def train_alignment(
    max_cases: int = 10000,
    device: str = "cuda",
    max_epochs: int = 200,
    coverage_loss_weight: float = 2.0,
    latent_dim: int = 128,
    transformer_layers: int = 4,
    transformer_heads: int = 8,
    model_type: str = "peak",
):
    import sys
    sys.path.insert(0, "/app")

    import numpy as np
    import torch
    from pathlib import Path

    # Import lib from the copied files (no DeTaNet needed)
    import alignment_notebook_lib as lib
    import stats_notebook_lib as stats_lib

    # Patch stats_lib reference used inside alignment_notebook_lib
    lib.stats_lib = stats_lib

    cache_dir = Path("/data/cache")
    out_dir = Path("/data/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load pre-built dataset cache (built locally, uploaded to volume)
    # -----------------------------------------------------------------------
    npz_p = cache_dir / f"dft_point_v1_{max_cases}.npz"
    csv_p = cache_dir / f"dft_point_v1_{max_cases}.csv"
    if not npz_p.exists():
        raise FileNotFoundError(
            f"{npz_p} not found on volume.\n"
            f"Run Step 2 from the RUNBOOK at the top of this file first."
        )

    print(f"Loading dataset from {npz_p} ...")
    dataset = lib._load_dft_mode_dataset_bundle(npz_p, csv_p)
    print(f"  {len(dataset)} molecules loaded, model_type={model_type}")

    if model_type == "spectral":
        # -------------------------------------------------------------------
        # Spectral U-Net path
        # -------------------------------------------------------------------
        cfg = lib.SpectralAlignmentTrainConfig(
            max_epochs=max_epochs,
            batch_size=64,
            patience=40,
            morgan_fp_bits=512,
            film_dim=128,
            finetune_epochs=50,
            finetune_lr=1e-4,
            finetune_patience=20,
        )
        print("Config:", cfg)

        results = lib.run_spectral_alignment_study(
            dft_dataset=dataset,
            out_dir=out_dir,
            device=device,
            train_config=cfg,
        )
    elif model_type == "peak_v10_rl":
        # -------------------------------------------------------------------
        # v10 RL: REINFORCE fine-tune with F1@10 reward
        # Requires a pre-trained checkpoint at /data/outputs/alignment_model.pth
        # -------------------------------------------------------------------
        rl_cfg = lib.AlignmentRLConfig(
            max_epochs=max_epochs,
            batch_size=128,
            lr=1e-5,
            K=16,
            sigma_init=3.0,
            patience=30,
            freeze_encoder=False,       # need full 814K params for RL signal
            conf_loss_weight=0.0,
            confidence_threshold=0.5,
            match_cutoff=15.0,
            mode_feature_dim=12,
            reward_tol=10.0,
            top_k_filter=70,
        )
        print("RL Config:", rl_cfg)

        ckpt_path = out_dir / "alignment_model.pth"
        if not ckpt_path.exists():
            print("No pre-trained checkpoint found — running Phase 1 (Sinkhorn) first...")
            phase1_cfg = lib.AlignmentTrainConfig(
                max_epochs=200, batch_size=256, patience=40,
                latent_dim=latent_dim, transformer_layers=transformer_layers,
                transformer_heads=transformer_heads,
                coverage_loss_weight=coverage_loss_weight,
                coverage_target_cm=10.0, lr=3e-4, weight_decay=1e-3,
                string_feature_dim=128, match_cutoff=15.0,
                confidence_loss_weight=1.0, confidence_threshold=0.5,
                freq_loss_weight=1.0, repulsion_loss_weight=0.0,
                use_sinkhorn=True, sinkhorn_tau=10.0, sinkhorn_match_sigma=10.0,
                mode_feature_dim=12,
            )
            lib.run_alignment_study(
                experimental_dataset=None, dft_dataset=dataset,
                out_dir=out_dir, device=device, train_config=phase1_cfg,
            )
            print("Phase 1 done. Starting Phase 2 (REINFORCE)...")

        results = lib.run_rl_finetune(
            dft_dataset=dataset,
            out_dir=out_dir,
            device=device,
            rl_config=rl_cfg,
            checkpoint_path=str(ckpt_path),
        )

    elif model_type == "peak_v11":
        # -------------------------------------------------------------------
        # v11: Differentiable soft-F1 loss (direct metric optimization)
        # -------------------------------------------------------------------
        cfg = lib.AlignmentTrainConfig(
            max_epochs=max_epochs,
            batch_size=256,
            patience=40,
            latent_dim=latent_dim,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            coverage_loss_weight=coverage_loss_weight,
            coverage_target_cm=10.0,
            lr=3e-4,
            weight_decay=1e-3,
            string_feature_dim=128,
            match_cutoff=15.0,
            confidence_loss_weight=1.0,
            confidence_threshold=0.5,
            freq_loss_weight=1.0,
            repulsion_loss_weight=0.0,
            repulsion_radius_cm=5.0,
            mode_feature_dim=12,
            use_soft_f1=True,           # differentiable F1
            soft_f1_tol=10.0,           # match at 10 cm⁻¹ (same as eval)
            soft_f1_tau=3.0,            # start warm
            soft_f1_tau_min=0.5,        # anneal to sharp
            sinkhorn_tau=10.0,          # soft assignment temperature
        )
        print("Config:", cfg)

        results = lib.run_alignment_study(
            experimental_dataset=None,
            dft_dataset=dataset,
            out_dir=out_dir,
            device=device,
            train_config=cfg,
        )

    elif model_type == "peak_v13":
        # -------------------------------------------------------------------
        # v13: True Sinkhorn + coupled confidence + supervised conf targets
        # Single-phase clean objective — no dnh, no fixed mask, no REINFORCE
        # -------------------------------------------------------------------
        cfg = lib.AlignmentTrainConfig(
            max_epochs=max_epochs,
            batch_size=256,
            patience=40,
            latent_dim=latent_dim,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            lr=3e-4,
            weight_decay=1e-3,
            string_feature_dim=128,
            match_cutoff=15.0,
            mode_feature_dim=12,
            # v13: true Sinkhorn + coupled confidence
            use_v13=True,
            sinkhorn_iters=20,
            sinkhorn_tau=10.0,
            soft_f1_tol=10.0,
            soft_f1_tau=3.0,
            soft_f1_tau_min=0.5,
            # Confidence supervision from Sinkhorn quality
            confidence_loss_weight=1.0,
            confidence_threshold=0.5,
            freq_loss_weight=1.0,
            coverage_loss_weight=0.0,  # not used in v13 loss
        )
        print("v13 Config:", cfg)

        results = lib.run_alignment_study(
            experimental_dataset=None,
            dft_dataset=dataset,
            out_dir=out_dir,
            device=device,
            train_config=cfg,
        )

    elif model_type == "peak_v12":
        # -------------------------------------------------------------------
        # v12: Hybrid Soft-F1 + REINFORCE keep/drop (two-phase training)
        # -------------------------------------------------------------------
        hybrid_cfg = lib.AlignmentHybridConfig(
            phase1_epochs=100,
            phase1_batch_size=256,
            phase1_lr=3e-4,
            phase1_patience=40,
            phase2_epochs=max(max_epochs - 100, 150),
            phase2_batch_size=128,
            phase2_lr=5e-5,
            phase2_patience=30,
            latent_dim=latent_dim,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            soft_f1_tol=10.0,
            soft_f1_tau_init=3.0,
            soft_f1_tau_min=0.5,
            sinkhorn_tau=10.0,
            dnh_radius=10.0,
            dnh_weight=0.0,
            entropy_coeff=0.05,
            rl_weight=0.3,
            rl_K=8,
            reward_tol=10.0,
            mode_feature_dim=12,
            match_cutoff=15.0,
        )
        print("Hybrid Config:", hybrid_cfg)

        results = lib.run_hybrid_training(
            dft_dataset=dataset,
            out_dir=out_dir,
            device=device,
            hybrid_config=hybrid_cfg,
            checkpoint_path=None,  # start fresh — v10 checkpoint is a bad local minimum for soft-F1
        )

    elif model_type == "peak_v10":
        # -------------------------------------------------------------------
        # v10: Peak transformer + eigenvector features + Sinkhorn OT loss
        # -------------------------------------------------------------------
        cfg = lib.AlignmentTrainConfig(
            max_epochs=max_epochs,
            batch_size=256,
            patience=40,
            latent_dim=latent_dim,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            coverage_loss_weight=coverage_loss_weight,
            coverage_target_cm=10.0,
            lr=3e-4,
            weight_decay=1e-3,
            string_feature_dim=128,
            match_cutoff=15.0,
            confidence_loss_weight=1.0,
            confidence_threshold=0.5,
            freq_loss_weight=1.0,
            repulsion_loss_weight=0.0,
            repulsion_radius_cm=5.0,
            # v10: Sinkhorn + eigenvector features
            use_sinkhorn=True,
            sinkhorn_tau=10.0,
            sinkhorn_match_sigma=10.0,
            mode_feature_dim=12,
        )
        print("Config:", cfg)

        results = lib.run_alignment_study(
            experimental_dataset=None,
            dft_dataset=dataset,
            out_dir=out_dir,
            device=device,
            train_config=cfg,
        )
    else:
        # -------------------------------------------------------------------
        # Peak coordinate transformer path (legacy v2-v7)
        # -------------------------------------------------------------------
        cfg = lib.AlignmentTrainConfig(
            max_epochs=max_epochs,
            batch_size=256,
            patience=40,
            latent_dim=latent_dim,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            coverage_loss_weight=coverage_loss_weight,
            coverage_target_cm=10.0,
            lr=3e-4,
            weight_decay=1e-3,
            string_feature_dim=128,
            match_cutoff=15.0,
            confidence_loss_weight=1.0,
            confidence_threshold=0.5,
            freq_loss_weight=1.0,
            repulsion_loss_weight=0.0,
            repulsion_radius_cm=5.0,
        )
        print("Config:", cfg)

        results = lib.run_alignment_study(
            experimental_dataset=None,
            dft_dataset=dataset,
            out_dir=out_dir,
            device=device,
            train_config=cfg,
        )

    # -----------------------------------------------------------------------
    # Print key metrics
    # -----------------------------------------------------------------------
    import pandas as pd
    summary = pd.read_csv(results["domains"]["dft"]["summary_csv"])
    test = summary[summary["split"] == "test"]
    print("\n=== TEST SET RESULTS ===")
    if not test.empty:
        row = test.iloc[0]
        print(f"  F1@5:       {row.get('f1@5', float('nan')):.3f}")
        print(f"  F1@10:      {row.get('f1@10', float('nan')):.3f}")
        print(f"  F1@20:      {row.get('f1@20', float('nan')):.3f}")
        print(f"  CWMAE@10:   {row.get('cwmae@10', float('nan')):.2f} cm^-1")
        print(f"  Coverage@10:{row.get('coverage@10', float('nan')):.3f}")
        print(f"  Point RMSE: {row.get('point_rmse', float('nan')):.2f} cm^-1")
        if 'avg_pred_kept' in row.index:
            print(f"  Modes kept: {row.get('avg_pred_kept', float('nan')):.0f}/"
                  f"{row.get('avg_pred_total', float('nan')):.0f}")
    print(results["domains"]["dft"]["report_markdown"])

    # Commit outputs to the volume
    vol.commit()
    return results


# ---------------------------------------------------------------------------
# Local entrypoint — runs the Modal function
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    max_cases: int = 10000,
    device: str = "cuda",
    max_epochs: int = 200,
    coverage_loss_weight: float = 2.0,
    model_type: str = "peak",
):
    print(f"Launching training: max_cases={max_cases}, device={device}, "
          f"max_epochs={max_epochs}, model_type={model_type}")
    results = train_alignment.remote(
        max_cases=max_cases,
        device=device,
        max_epochs=max_epochs,
        coverage_loss_weight=coverage_loss_weight,
        model_type=model_type,
    )
    print("Done. Outputs written to Modal volume raman-alignment-data at /outputs/")
    return results
