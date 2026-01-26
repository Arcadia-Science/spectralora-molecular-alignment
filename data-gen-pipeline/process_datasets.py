from __future__ import annotations

import argparse
import csv
import gzip
import multiprocessing as mp
import os
import sqlite3
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
    kind: str
    path: Optional[Path]
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
                subset=(row.get("SPLITS") or "").strip() or None,
                source=path.name,
            )


def iter_qm7x_hdf5(path: Path) -> Iterable[pipeline.SmilesItem]:
    try:
        import h5py
    except Exception as exc:
        raise RuntimeError("h5py is required for QM7-X datasets.") from exc

    index = 0
    with h5py.File(path, "r") as handle:
        for mol_key in handle.keys():
            mol_group = handle[mol_key]
            for conf_key in mol_group.keys():
                conf = mol_group[conf_key]

                if "atNUM" not in conf or "atXYZ" not in conf:
                    continue
                z = torch.tensor(conf["atNUM"][()], dtype=torch.long)
                pos = torch.tensor(conf["atXYZ"][()], dtype=torch.float32)

                energy = None
                energy_source = None
                for key in ("ePBE0+MBD", "ePBE0", "eAT", "eDFTB+MBD"):
                    if key in conf:
                        try:
                            energy_val = float(conf[key][()])
                        except Exception:
                            energy_val = None
                        if energy_val is not None:
                            energy = torch.tensor([[energy_val]], dtype=torch.float32)
                            energy_source = f"qm7x_{key}"
                            break

                dipole = None
                if "vDIP" in conf:
                    try:
                        dipole = torch.tensor(conf["vDIP"][()], dtype=torch.float32).view(1, 3)
                        dipole = dipole * pipeline.E_ANGSTROM_TO_DEBYE
                    except Exception:
                        dipole = None

                polar = None
                if "mTPOL" in conf:
                    try:
                        pol = torch.tensor(conf["mTPOL"][()], dtype=torch.float32).reshape(-1)
                        if pol.numel() == 9:
                            polar = pol.view(3, 3).unsqueeze(0)
                    except Exception:
                        polar = None

                charges = None
                if "hCHG" in conf:
                    try:
                        charges = torch.tensor(conf["hCHG"][()], dtype=torch.float32)
                    except Exception:
                        charges = None

                index += 1
                field_source = {"pos": "dataset", "z": "dataset", "smile": "dataset"}
                if energy is not None:
                    field_source["energy"] = energy_source or "qm7x_energy"
                if dipole is not None:
                    field_source["dipole"] = "qm7x_dipole"
                if polar is not None:
                    field_source["polar"] = "qm7x_polar"
                if charges is not None and charges.numel() == z.shape[0]:
                    field_source["npacharge"] = "qm7x_charge"

                yield pipeline.SmilesItem(
                    number=index,
                    smile=str(mol_key),
                    pos=pos,
                    z=z,
                    energy=energy,
                    dipole=dipole,
                    polar=polar,
                    npacharge=charges,
                    field_source=field_source,
                    source=path.name,
                    mol_key=str(mol_key),
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


def iter_sdf_dir(path: Path) -> Iterable[pipeline.SmilesItem]:
    from rdkit import Chem

    idx = 0
    for sdf_path in sorted(path.rglob("*.sdf")):
        supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
        for mol in supplier:
            if mol is None:
                continue
            idx += 1
            try:
                smile = Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True)
            except Exception:
                smile = f"{sdf_path.stem}_{idx}"

            pos = None
            z = None
            charges = None
            try:
                conf = mol.GetConformer()
                coords = [[p.x, p.y, p.z] for p in (conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms()))]
                pos_t = torch.tensor(coords, dtype=torch.float32)
                if pos_t.abs().sum().item() > 1e-6:
                    pos = pos_t
                    z = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long)
                    charges = pipeline.compute_gasteiger_charges(mol)
            except Exception:
                pos = None
                z = None

            field_source = {"smile": "dataset"}
            if pos is not None and z is not None:
                field_source.update({"pos": "dataset", "z": "dataset"})
            if charges is not None and charges.numel() == (z.shape[0] if z is not None else -1):
                field_source["npacharge"] = "gasteiger"

            yield pipeline.SmilesItem(
                number=idx,
                smile=smile,
                pos=pos,
                z=z,
                npacharge=charges,
                field_source=field_source,
                source=sdf_path.name,
            )


def iter_pdb_dir(path: Path) -> Iterable[pipeline.SmilesItem]:
    from rdkit import Chem

    idx = 0
    for pdb_path in sorted(path.rglob("*.pdb")):
        idx += 1
        mol = None
        try:
            mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=False, sanitize=False)
        except Exception:
            mol = None
        smile = pdb_path.stem
        pos = None
        z = None
        charges = None
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                pass
            try:
                smile = Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True)
            except Exception:
                smile = pdb_path.stem
            try:
                conf = mol.GetConformer()
                coords = [[p.x, p.y, p.z] for p in (conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms()))]
                pos = torch.tensor(coords, dtype=torch.float32)
                z = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long)
                charges = pipeline.compute_gasteiger_charges(mol)
            except Exception:
                pos = None
                z = None
                charges = None
        if pos is None or z is None:
            try:
                from ase.io import read as ase_read

                atoms = ase_read(str(pdb_path))
                pos = torch.tensor(atoms.positions, dtype=torch.float32)
                z = torch.tensor(atoms.numbers, dtype=torch.long)
            except Exception:
                pos = None
                z = None

        field_source = {"smile": "dataset"}
        if pos is not None and z is not None:
            field_source.update({"pos": "dataset", "z": "dataset"})
        if charges is not None and z is not None and charges.numel() == z.shape[0]:
            field_source["npacharge"] = "gasteiger"

        yield pipeline.SmilesItem(
            number=idx,
            smile=smile,
            pos=pos,
            z=z,
            npacharge=charges,
            field_source=field_source,
            source=pdb_path.name,
        )


def iter_generic_db(path: Path) -> Iterable[pipeline.SmilesItem]:
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='molecule'")
    has_molecule = cur.fetchone() is not None
    conn.close()
    if has_molecule:
        cfg = pipeline.PipelineConfig(output_dir=Path("."), device=torch.device("cpu"), db_path=path)
        yield from pipeline.iter_smiles(cfg)
        return

    try:
        from ase.db import connect
    except Exception:
        return
    idx = 0
    with connect(path) as db:
        for row in db.select():
            idx += 1
            atoms = row.toatoms()
            pos = torch.tensor(atoms.positions, dtype=torch.float32)
            z = torch.tensor(atoms.numbers, dtype=torch.long)
            smile = row.get("unique_id") or row.get("smiles") or f"{path.stem}_{idx}"
            field_source = {"pos": "dataset", "z": "dataset", "smile": "dataset"}
            yield pipeline.SmilesItem(
                number=row.id if hasattr(row, "id") else idx,
                smile=smile,
                pos=pos,
                z=z,
                field_source=field_source,
                source=path.name,
            )


def build_jobs(datasets_root: Path, output_root: Path) -> list[DatasetJob]:
    jobs: list[DatasetJob] = []
    qm7x_root = datasets_root / "datasets--qm7x"
    qm7x_paths: list[Path] = []
    if qm7x_root.exists():
        qm7x_paths = sorted(qm7x_root.rglob("*.hdf5")) + sorted(qm7x_root.rglob("*.h5"))
        for path in qm7x_paths:
            cfg = pipeline.PipelineConfig(
                output_dir=output_root / f"qm7x/{path.stem}",
                device=torch.device("cpu"),
            )
            jobs.append(DatasetJob(f"qm7x/{path.name}", "qm7x_hdf5", path, None, cfg.output_dir, cfg))
        if not qm7x_paths:
            print(
                "QM7-X dataset detected but no HDF5 files found. "
                "Run Datasets/datasets--qm7x/scripts/download.sh or datalad get to fetch data."
            )
    spice_hdf5 = datasets_root / "SPICE-2.0.1.hdf5"
    if spice_hdf5.exists():
        cfg = pipeline.PipelineConfig(
            output_dir=output_root / "spice-2.0.1",
            device=torch.device("cpu"),
            hdf5_paths=[spice_hdf5],
        )
        jobs.append(DatasetJob("spice-2.0.1", "hdf5", spice_hdf5, None, cfg.output_dir, cfg))

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
            jobs.append(DatasetJob(f"spice/{path.name}", "hdf5", path, None, cfg.output_dir, cfg))

    summary_csv = datasets_root / "summary.csv.gz"
    if summary_csv.exists():
        cfg = pipeline.PipelineConfig(output_dir=output_root / "summary_csv", device=torch.device("cpu"))
        jobs.append(DatasetJob("summary_csv", "summary_csv", summary_csv, None, cfg.output_dir, cfg))

    des5m_csv = datasets_root / "Donchev et al. DES5M.csv"
    if des5m_csv.exists():
        cfg = pipeline.PipelineConfig(output_dir=output_root / "des5m", device=torch.device("cpu"))
        jobs.append(DatasetJob("des5m", "des5m", des5m_csv, None, cfg.output_dir, cfg))

    raman_db = datasets_root / "Raman-ChEMBL-part2.db"
    if raman_db.exists():
        cfg = pipeline.PipelineConfig(output_dir=output_root / "raman_chembl", device=torch.device("cpu"), db_path=raman_db)
        jobs.append(DatasetJob("raman_chembl", "raman_db", raman_db, None, cfg.output_dir, cfg))

    raman_db2 = datasets_root / "Raman-ChEMBL-part1.db"
    if raman_db2.exists():
        cfg = pipeline.PipelineConfig(output_dir=output_root / "raman_chembl_part1", device=torch.device("cpu"), db_path=raman_db2)
        jobs.append(DatasetJob("raman_chembl_part1", "raman_db", raman_db2, None, cfg.output_dir, cfg))

    for db_path in sorted(datasets_root.rglob("*.db")):
        if db_path.name.startswith("Raman-ChEMBL"):
            continue
        cfg = pipeline.PipelineConfig(output_dir=output_root / f"db/{db_path.stem}", device=torch.device("cpu"))
        jobs.append(DatasetJob(f"db/{db_path.name}", "generic_db", db_path, None, cfg.output_dir, cfg))

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
            jobs.append(DatasetJob(f"qdpi/{path.name}", "hdf5", path, None, cfg.output_dir, cfg))

    sdf_dirs = sorted({p.parent for p in datasets_root.rglob("*.sdf")})
    for sdf_dir in sdf_dirs:
        rel = sdf_dir.relative_to(datasets_root)
        cfg = pipeline.PipelineConfig(output_dir=output_root / "sdf" / rel, device=torch.device("cpu"))
        jobs.append(DatasetJob(f"sdf/{rel}", "sdf_dir", sdf_dir, None, cfg.output_dir, cfg))

    pdb_dirs = sorted({p.parent for p in datasets_root.rglob("*.pdb")})
    for pdb_dir in pdb_dirs:
        rel = pdb_dir.relative_to(datasets_root)
        cfg = pipeline.PipelineConfig(output_dir=output_root / "pdb" / rel, device=torch.device("cpu"))
        jobs.append(DatasetJob(f"pdb/{rel}", "pdb_dir", pdb_dir, None, cfg.output_dir, cfg))

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


def build_iterator(job: DatasetJob, cfg: pipeline.PipelineConfig) -> Iterable[pipeline.SmilesItem]:
    if job.kind == "qm7x_hdf5":
        return iter_qm7x_hdf5(job.path)
    if job.kind == "hdf5":
        return pipeline.iter_hdf5(cfg)
    if job.kind == "summary_csv":
        return iter_summary_csv(job.path)
    if job.kind == "des5m":
        return iter_des5m(job.path)
    if job.kind == "raman_db":
        return iter_raman_db(job.path)
    if job.kind == "generic_db":
        return iter_generic_db(job.path)
    if job.kind == "sdf_dir":
        return iter_sdf_dir(job.path)
    if job.kind == "pdb_dir":
        return iter_pdb_dir(job.path)
    raise ValueError(f"Unknown job kind: {job.kind}")


def _slice_iter(items: Iterable[pipeline.SmilesItem], rank: int, world_size: int):
    for idx, item in enumerate(items, start=1):
        if (idx - 1) % world_size != rank:
            continue
        yield item


def _run_job_worker(job: DatasetJob, args, rank: int, world_size: int) -> None:
    if args.omp_threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(args.omp_threads)
    if args.mkl_threads is not None:
        os.environ["MKL_NUM_THREADS"] = str(args.mkl_threads)
    if args.openblas_threads is not None:
        os.environ["OPENBLAS_NUM_THREADS"] = str(args.openblas_threads)
    if args.dp_intra_threads is not None:
        os.environ["DP_INTRA_OP_PARALLELISM_THREADS"] = str(args.dp_intra_threads)
    if args.dp_inter_threads is not None:
        os.environ["DP_INTER_OP_PARALLELISM_THREADS"] = str(args.dp_inter_threads)
    if args.dp_infer_batch_size is not None:
        os.environ["DP_INFER_BATCH_SIZE"] = str(args.dp_infer_batch_size)
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
    if args.torch_interop_threads is not None:
        # DeepMD sets interop threads during import; avoid calling twice in workers.
        os.environ["TORCH_NUM_INTEROP_THREADS"] = str(args.torch_interop_threads)

    cfg = apply_cfg_overrides(job.cfg, args)
    if world_size > 1:
        cfg.output_dir = cfg.output_dir / f"rank_{rank:02d}"
    cfg.distributed = world_size > 1
    cfg.rank = rank
    cfg.world_size = world_size

    items_iter = build_iterator(job, cfg)
    if cfg.hdf5_paths:
        # iter_hdf5 handles distributed slicing based on cfg
        pass
    elif cfg.distributed:
        items_iter = _slice_iter(items_iter, rank, world_size)

    if args.limit is not None:
        items_iter = (item for idx, item in enumerate(items_iter, start=1) if idx <= args.limit)

    pipeline.run_pipeline(cfg, items_iter, job.total)


def run_job(job: DatasetJob, args) -> None:
    workers = max(1, args.workers)
    if workers == 1:
        _run_job_worker(job, args, rank=0, world_size=1)
        return
    ctx = mp.get_context("spawn")
    procs = []
    for rank in range(workers):
        p = ctx.Process(target=_run_job_worker, args=(job, args, rank, workers))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    failures = [p.exitcode for p in procs if p.exitcode not in (0, None)]
    if failures:
        raise RuntimeError(f"Job {job.name} failed with exit codes {failures}")


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
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2))
    parser.add_argument("--job-index", type=int, default=0)
    parser.add_argument("--job-count", type=int, default=1)
    parser.add_argument("--omp-threads", type=int, default=None)
    parser.add_argument("--mkl-threads", type=int, default=None)
    parser.add_argument("--openblas-threads", type=int, default=None)
    parser.add_argument("--torch-threads", type=int, default=None)
    parser.add_argument("--torch-interop-threads", type=int, default=None)
    parser.add_argument("--dp-intra-threads", type=int, default=None)
    parser.add_argument("--dp-inter-threads", type=int, default=None)
    parser.add_argument("--dp-infer-batch-size", type=int, default=None)
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

    jobs = build_jobs(datasets_root, output_root)
    if args.job_count < 1:
        raise ValueError("--job-count must be >= 1")
    if args.job_index < 0 or args.job_index >= args.job_count:
        raise ValueError("--job-index must be in [0, job-count)")
    if args.job_count > 1:
        jobs = [job for idx, job in enumerate(jobs) if idx % args.job_count == args.job_index]

    total_jobs = len(jobs)
    for job_idx, job in enumerate(jobs, start=1):
        tag = f"[job {job_idx}/{total_jobs}]"
        if args.job_count > 1:
            tag = f"[job {job_idx}/{total_jobs} | shard {args.job_index}/{args.job_count}]"
        print(f"{tag} Processing {job.name} -> {job.output_dir}")
        run_job(job, args)


if __name__ == "__main__":
    main()
