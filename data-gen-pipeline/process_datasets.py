from __future__ import annotations

import argparse
import csv
import gzip
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch

import pipeline
from deepmd_backend import PTE_SYMBOLS


@dataclass
class DatasetJob:
    name: str
    iterator: Iterable[pipeline.SmilesItem]
    total: Optional[int]
    output_dir: Path
    cfg: pipeline.PipelineConfig


def _symbol_to_z(symbol: str) -> int:
    symbol = symbol.strip()
    if not symbol:
        raise ValueError("Empty element symbol in dataset.")
    try:
        return PTE_SYMBOLS.index(symbol)
    except ValueError as exc:
        raise ValueError(f"Unknown element symbol: {symbol}") from exc


def iter_summary_csv(path: Path) -> Iterable[pipeline.SmilesItem]:
    opener = gzip.open if path.suffix.endswith(".gz") else open
    with opener(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader, start=1):
            smile = (row.get("SMILES") or "").strip()
            if not smile:
                continue
            energy_val = row.get("DFT TOTAL ENERGY") or row.get("DFT FORMATION ENERGY")
            energy = None
            if energy_val:
                try:
                    energy = torch.tensor([[float(energy_val)]], dtype=torch.float32)
                except ValueError:
                    energy = None
            dipole = None
            dipole_cols = [row.get("DFT DIPOLE X"), row.get("DFT DIPOLE Y"), row.get("DFT DIPOLE Z")]
            if all(val not in (None, "") for val in dipole_cols):
                try:
                    dipole = torch.tensor([float(v) for v in dipole_cols], dtype=torch.float32).view(1, 3)
                except ValueError:
                    dipole = None
            field_source = {"smile": "dataset"}
            if energy is not None:
                field_source["energy"] = "summary_csv"
            if dipole is not None:
                field_source["dipole"] = "summary_csv"
            yield pipeline.SmilesItem(
                number=idx,
                smile=smile,
                energy=energy,
                dipole=dipole,
                field_source=field_source,
            )


def iter_des5m(path: Path) -> Iterable[pipeline.SmilesItem]:
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader, start=1):
            elements = (row.get("elements") or "").strip()
            xyz = (row.get("xyz") or "").strip()
            smile0 = (row.get("smiles0") or "").strip()
            smile1 = (row.get("smiles1") or "").strip()
            smile = ".".join([s for s in (smile0, smile1) if s]) or row.get("system_id") or f"des5m_{idx}"

            pos = None
            z = None
            if elements and xyz:
                symbols = elements.split()
                coords = [float(val) for val in xyz.split()]
                if len(coords) == 3 * len(symbols):
                    pos = torch.tensor(coords, dtype=torch.float32).view(-1, 3)
                    z = torch.tensor([_symbol_to_z(sym) for sym in symbols], dtype=torch.long)
            if pos is None or z is None:
                if not smile0 or not smile1:
                    continue
                pos0, z0, _ = pipeline.smiles_to_conformer(smile0, max_attempts=10, optimize=True)
                pos1, z1, _ = pipeline.smiles_to_conformer(smile1, max_attempts=10, optimize=True)
                pos1 = pos1 + torch.tensor([4.0, 0.0, 0.0], dtype=pos1.dtype)
                pos = torch.cat([pos0, pos1], dim=0)
                z = torch.cat([z0, z1], dim=0)

            energy_val = row.get("nn_CCSD(T)_all") or row.get("sapt_all")
            energy = None
            if energy_val:
                try:
                    energy = torch.tensor([[float(energy_val)]], dtype=torch.float32)
                except ValueError:
                    energy = None
            field_source = {"pos": "dataset" if elements and xyz else "rdkit", "z": "dataset" if elements and xyz else "rdkit", "smile": "dataset"}
            if energy is not None:
                field_source["energy"] = "des5m_energy"
            yield pipeline.SmilesItem(
                number=idx,
                smile=smile,
                pos=pos,
                z=z,
                energy=energy,
                field_source=field_source,
                source=path.name,
            )


def iter_raman_db(path: Path) -> Iterable[pipeline.SmilesItem]:
    cfg = pipeline.PipelineConfig(output_dir=Path("."), device=torch.device("cpu"), db_path=path)
    yield from pipeline.iter_smiles(cfg)


def build_jobs(datasets_root: Path, output_root: Path) -> list[DatasetJob]:
    jobs: list[DatasetJob] = []
    spice_hdf5 = datasets_root / "SPICE-2.0.1.hdf5"
    if spice_hdf5.exists():
        cfg = pipeline.PipelineConfig(
            output_dir=output_root / "spice-2.0.1",
            device=torch.device("cpu"),
            hdf5_paths=[spice_hdf5],
        )
        jobs.append(DatasetJob("spice-2.0.1", pipeline.iter_hdf5(cfg), None, cfg.output_dir, cfg))

    spice_dir = datasets_root / "spice-dataset"
    if spice_dir.exists():
        paths = (
            sorted(spice_dir.rglob("*.hdf5"))
            + sorted(spice_dir.rglob("*.hdf5.gz"))
            + sorted(spice_dir.rglob("*.h5"))
            + sorted(spice_dir.rglob("*.h5.gz"))
        )
        for path in paths:
            cfg = pipeline.PipelineConfig(
                output_dir=output_root / f"spice/{path.stem}",
                device=torch.device("cpu"),
                hdf5_paths=[path],
            )
            jobs.append(DatasetJob(f"spice/{path.name}", pipeline.iter_hdf5(cfg), None, cfg.output_dir, cfg))

    summary_csv = datasets_root / "summary.csv.gz"
    if summary_csv.exists():
        cfg = pipeline.PipelineConfig(output_dir=output_root / "summary_csv", device=torch.device("cpu"))
        jobs.append(DatasetJob("summary_csv", iter_summary_csv(summary_csv), None, cfg.output_dir, cfg))

    des5m_csv = datasets_root / "Donchev et al. DES5M.csv"
    if des5m_csv.exists():
        cfg = pipeline.PipelineConfig(output_dir=output_root / "des5m", device=torch.device("cpu"))
        jobs.append(DatasetJob("des5m", iter_des5m(des5m_csv), None, cfg.output_dir, cfg))

    raman_db = datasets_root / "Raman-ChEMBL-part2.db"
    if raman_db.exists():
        cfg = pipeline.PipelineConfig(output_dir=output_root / "raman_chembl", device=torch.device("cpu"), db_path=raman_db)
        jobs.append(DatasetJob("raman_chembl", iter_raman_db(raman_db), None, cfg.output_dir, cfg))

    qdpi_tar = datasets_root / "QDpiDataset-main.tar.gz"
    if qdpi_tar.exists():
        tmp_dir = output_root / "_tmp_qdpi"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(qdpi_tar, "r:gz") as tar:
            members = [m for m in tar.getmembers() if m.name.endswith((".h5", ".hdf5"))]
            tar.extractall(path=tmp_dir, members=members)
        for path in tmp_dir.rglob("*.hdf5"):
            cfg = pipeline.PipelineConfig(
                output_dir=output_root / f"qdpi/{path.stem}",
                device=torch.device("cpu"),
                hdf5_paths=[path],
            )
            jobs.append(DatasetJob(f"qdpi/{path.name}", pipeline.iter_hdf5(cfg), None, cfg.output_dir, cfg))

    return jobs


def apply_cfg_overrides(cfg: pipeline.PipelineConfig, args) -> pipeline.PipelineConfig:
    return pipeline.PipelineConfig(
        output_dir=cfg.output_dir,
        device=args.device,
        save_device=args.save_device,
        limit=args.limit,
        log_every=args.log_every,
        smiles=cfg.smiles,
        smiles_file=cfg.smiles_file,
        db_path=cfg.db_path,
        hdf5_paths=cfg.hdf5_paths,
        hdf5_subset=cfg.hdf5_subset,
        pos_step=args.pos_step,
        dft_atom_cutoff=args.dft_atom_cutoff,
        graph_k=args.graph_k,
        graph_clamp_min=args.graph_clamp_min,
        graph_clamp_max=args.graph_clamp_max,
        dipole_model=args.deepmd_dipole_model,
        polar_model=args.deepmd_polar_model,
        deepmd_pot_model=args.deepmd_pot_model,
        deepmd_head=args.deepmd_head,
        deepmd_type_map=args.deepmd_type_map,
        deepmd_dipole_unit=args.deepmd_dipole_unit,
        deepmd_atomic_energy=args.deepmd_atomic_energy,
        mace_model=args.mace_model,
        psi4_fallback=not args.no_psi4,
        psi4_method=args.psi4_method,
        psi4_basis=args.psi4_basis,
        psi4_memory=args.psi4_memory,
        psi4_threads=args.psi4_threads,
        psi4_scf_type=args.psi4_scf_type,
        psi4_guess=args.psi4_guess,
        psi4_charge=args.psi4_charge,
        psi4_multiplicity=args.psi4_multiplicity,
        psi4_quiet=args.psi4_quiet,
        rdkit_max_attempts=args.rdkit_max_attempts,
        rdkit_optimize=not args.rdkit_no_opt,
        allow_missing_hyperpolar=args.allow_missing_hyperpolar,
        allow_missing_polar=args.allow_missing_polar,
        allow_missing_dipole=args.allow_missing_dipole,
        shard_size=args.shard_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Process all datasets under Datasets/ into PyG shards.")
    parser.add_argument("--datasets-root", type=str, default="Datasets")
    parser.add_argument("--output-root", type=str, default="processed")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-device", type=str, default="cpu")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--deepmd-dipole-model", type=str, default=None)
    parser.add_argument("--deepmd-polar-model", type=str, default=None)
    parser.add_argument("--deepmd-pot-model", type=str, default=None)
    parser.add_argument("--deepmd-head", type=str, default=None)
    parser.add_argument("--deepmd-type-map", type=str, default=None)
    parser.add_argument("--deepmd-dipole-unit", type=str, default="au", choices=("au", "debye"))
    parser.add_argument("--deepmd-atomic-energy", action="store_true")
    parser.add_argument("--mace-model", type=str, default=None)
    parser.add_argument("--dft-atom-cutoff", type=int, default=20)
    parser.add_argument("--graph-k", type=int, default=20)
    parser.add_argument("--graph-clamp-min", type=float, default=0.5)
    parser.add_argument("--graph-clamp-max", type=float, default=100.0)
    parser.add_argument("--no-psi4", action="store_true")
    parser.add_argument("--psi4-method", type=str, default="B3LYP")
    parser.add_argument("--psi4-basis", type=str, default="cc-pVTZ")
    parser.add_argument("--psi4-memory", type=str, default="2 GB")
    parser.add_argument("--psi4-threads", type=int, default=1)
    parser.add_argument("--psi4-scf-type", type=str, default="df")
    parser.add_argument("--psi4-guess", type=str, default="sad")
    parser.add_argument("--psi4-charge", type=int, default=0)
    parser.add_argument("--psi4-multiplicity", type=int, default=1)
    parser.add_argument("--psi4-quiet", action="store_true")
    parser.add_argument("--allow-missing-hyperpolar", action="store_true")
    parser.add_argument("--allow-missing-polar", action="store_true")
    parser.add_argument("--allow-missing-dipole", action="store_true")
    parser.add_argument("--pos-step", type=float, default=1e-3)
    parser.add_argument("--rdkit-max-attempts", type=int, default=10)
    parser.add_argument("--rdkit-no-opt", action="store_true")
    parser.add_argument("--shard-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_device = torch.device(args.save_device) if args.save_device else None
    args.device = device
    args.save_device = save_device
    if args.mace_model is None:
        args.mace_model = Path("data-gen-pipeline/checkpoints/2024-07-12-mace-128-L1_epoch-199.model")
    else:
        args.mace_model = Path(args.mace_model)
    if args.deepmd_pot_model is not None:
        args.deepmd_pot_model = Path(args.deepmd_pot_model)
    if args.deepmd_dipole_model is not None:
        args.deepmd_dipole_model = Path(args.deepmd_dipole_model)
    if args.deepmd_polar_model is not None:
        args.deepmd_polar_model = Path(args.deepmd_polar_model)

    datasets_root = Path(args.datasets_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for job in build_jobs(datasets_root, output_root):
        cfg = apply_cfg_overrides(job.cfg, args)
        if cfg.hdf5_paths:
            job_iter = pipeline.iter_hdf5(cfg)
        elif cfg.db_path:
            job_iter = pipeline.iter_smiles(cfg)
        else:
            job_iter = job.iterator
        if args.limit is not None:
            job_iter = (item for idx, item in enumerate(job_iter, start=1) if idx <= args.limit)
        print(f"==> Processing {job.name} -> {job.output_dir}")
        pipeline.run_pipeline(cfg, job_iter, job.total)


if __name__ == "__main__":
    main()
