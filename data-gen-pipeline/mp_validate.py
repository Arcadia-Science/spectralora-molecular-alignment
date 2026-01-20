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
    def _mae(a: torch.Tensor, b: torch.Tensor) -> float:
        return torch.mean(torch.abs(a - b)).item()

    def _edge_list(edge_index: torch.Tensor):
        if edge_index.numel() == 0:
            return []
        return edge_index.detach().cpu().to(torch.long).t().tolist()

    def _edge_set(edge_index: torch.Tensor):
        return {tuple(pair) for pair in _edge_list(edge_index)}

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
    shape_errors = defaultdict(int)

    for key in d.keys():
        if key in (
            "smile",
            "number",
            "field_source",
            "field_generated",
            "field_imputed",
            "field_confidence",
            "subset",
            "source",
            "mol_key",
            "conformer_id",
        ):
            continue
        if not hasattr(pred, key):
            shape_errors[key] += 1
            continue

        try:
            target = getattr(d, key)
            estimate = getattr(pred, key)
            if not isinstance(target, torch.Tensor) or not isinstance(estimate, torch.Tensor):
                shape_errors[key] += 1
                continue

            if key == "edge_index":
                tgt_edges = _edge_set(target)
                pred_edges = _edge_set(estimate)
                union = tgt_edges | pred_edges
                if not union:
                    local_errors[key].append(0.0)
                else:
                    jaccard = len(tgt_edges & pred_edges) / len(union)
                    local_errors[key].append(1.0 - jaccard)
                continue

            if key == "Hij":
                tgt_edges = _edge_set(d.edge_index)
                pred_edges = _edge_set(pred.edge_index)
                if target.shape == estimate.shape and tgt_edges == pred_edges:
                    local_errors[key].append(_mae(target, estimate))
                    continue
                # Align by edge intersection when possible.
                if tgt_edges and pred_edges:
                    tgt_map = {tuple(edge): idx for idx, edge in enumerate(_edge_list(d.edge_index))}
                    pred_map = {tuple(edge): idx for idx, edge in enumerate(_edge_list(pred.edge_index))}
                    common = tgt_edges & pred_edges
                    if common:
                        tgt_idx = torch.tensor([tgt_map[e] for e in common], dtype=torch.long)
                        pred_idx = torch.tensor([pred_map[e] for e in common], dtype=torch.long)
                        local_errors[key].append(_mae(target[tgt_idx], estimate[pred_idx]))
                        continue
                shape_errors[key] += 1
                continue

            if target.shape != estimate.shape:
                shape_errors[key] += 1
                continue

            local_errors[key].append(_mae(target, estimate))
        except Exception:
            shape_errors[key] += 1

    denom = len([k for k in d.keys() if k not in ("smile", "number")])
    return local_errors, dict(shape_errors), denom


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
    shape_error_by_key: dict[str, int] = defaultdict(int)
    denom_per_item = None

    indices = list(indices)
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
            shape_error += sum(local_shape_errors.values())
            for key, count in local_shape_errors.items():
                shape_error_by_key[key] += count
            denom_per_item = denom
            for key, vals in local_errors.items():
                error_dict[key].extend(vals)

    total_denom = len(indices) * (denom_per_item if denom_per_item is not None else 1)
    return {
        "shape_error": shape_error,
        "shape_fraction": shape_error / total_denom if total_denom else 0.0,
        "shape_error_by_key": dict(shape_error_by_key),
        "errors": dict(error_dict),
    }
