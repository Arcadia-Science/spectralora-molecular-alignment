from __future__ import annotations

import multiprocessing as mp
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional

import torch

import pipeline
from deepmd_backend import DeepMDDipoleBackend, DeepMDPolarBackend, DeepMDPotHessianBackend
from psi4_backend import Psi4HessianBackend

_DATASET = None
_CFG = None
_BACKENDS = None


def _init_worker(
    dataset_path: str,
    cfg: pipeline.PipelineConfig,
    pot_model: str,
    dipole_model: Optional[str],
    polar_model: Optional[str],
    use_psi4: bool,
) -> None:
    global _DATASET, _CFG, _BACKENDS
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    _DATASET = torch.load(dataset_path, weights_only=False)
    _CFG = cfg

    dipole_backend = (
        DeepMDDipoleBackend(device=cfg.device, model_path=dipole_model)
        if dipole_model
        else None
    )
    polar_backend = (
        DeepMDPolarBackend(device=cfg.device, model_path=polar_model)
        if polar_model
        else None
    )
    hessian_backend = DeepMDPotHessianBackend(device=cfg.device, model_path=pot_model)

    psi4_backend = None
    if use_psi4 and cfg.psi4_fallback:
        try:
            psi4_backend = Psi4HessianBackend(
                device=torch.device("cpu"),
                method=cfg.psi4_method,
                basis=cfg.psi4_basis,
                charge=cfg.psi4_charge,
                multiplicity=cfg.psi4_multiplicity,
                num_threads=cfg.psi4_threads,
                memory=cfg.psi4_memory,
                scf_type=cfg.psi4_scf_type,
                guess=cfg.psi4_guess,
                quiet=cfg.psi4_quiet,
            )
        except Exception:
            psi4_backend = None

    _BACKENDS = (dipole_backend, polar_backend, hessian_backend, psi4_backend)


def _compute_one(idx: int):
    from torchmetrics.functional import mean_absolute_error

    d = _DATASET[idx]
    item = pipeline.SmilesItem(
        number=1,
        smile=d.smile,
        pos=d.pos,
        z=d.z,
        edge_index=d.edge_index,
    )

    dipole_backend, polar_backend, hessian_backend, psi4_backend = _BACKENDS
    pred = pipeline.build_data_from_smiles(
        item, _CFG, dipole_backend, polar_backend, hessian_backend, psi4_backend
    )

    local_errors = defaultdict(list)
    shape_errors = 0

    for key in d.keys():
        if key in ("smile", "number"):
            continue
        try:
            diff = mean_absolute_error(getattr(d, key), getattr(pred, key)).item()
            local_errors[key].append(diff)
        except Exception:
            shape_errors += 1

    return local_errors, shape_errors, len(d.keys()) - 2


def run_validation(
    dataset_path: Path,
    cfg: pipeline.PipelineConfig,
    pot_model: Path,
    dipole_model: Optional[Path],
    polar_model: Optional[Path],
    indices: Iterable[int],
    max_workers: int = 4,
    use_psi4: bool = False,
    show_progress: bool = True,
):
    error_dict: dict[str, list[float]] = defaultdict(list)
    shape_error = 0
    denom_per_item = None

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(
            str(dataset_path),
            cfg,
            str(pot_model),
            str(dipole_model) if dipole_model else None,
            str(polar_model) if polar_model else None,
            use_psi4,
        ),
    ) as ex:
        futures = [ex.submit(_compute_one, int(i)) for i in indices]
        if show_progress:
            from tqdm import tqdm

            iterator = tqdm(as_completed(futures), total=len(futures), desc="validate")
        else:
            iterator = as_completed(futures)
        for fut in iterator:
            local_errors, local_shape_errors, denom = fut.result()
            shape_error += local_shape_errors
            denom_per_item = denom
            for key, vals in local_errors.items():
                error_dict[key].extend(vals)

    total_denom = len(list(indices)) * (denom_per_item if denom_per_item is not None else 1)
    return {
        "shape_error": shape_error,
        "shape_fraction": shape_error / total_denom if total_denom else 0.0,
        "errors": dict(error_dict),
    }
