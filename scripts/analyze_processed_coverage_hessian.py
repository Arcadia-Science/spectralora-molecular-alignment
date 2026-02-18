#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

try:
    import h5py
except Exception:  # pragma: no cover - optional on some local envs
    h5py = None


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.train_detanet import _normalize_split_key, _resolve_molecule_key


def _is_valid_item(item, task: str, skip_nonfinite: bool) -> bool:
    target = getattr(item, task, None)
    pos = getattr(item, "pos", None)
    if target is None or pos is None:
        return False
    if not skip_nonfinite:
        return True
    if torch.is_tensor(target) and not torch.isfinite(target).all().item():
        return False
    if (not torch.is_tensor(target)) and isinstance(target, (float, int)) and not math.isfinite(target):
        return False
    if torch.is_tensor(pos) and not torch.isfinite(pos).all().item():
        return False
    return True


def _molecule_id(item, split_key: str = "mol_key", scaffold_group_key: str = "mol_key") -> Optional[str]:
    mol_key = _resolve_molecule_key(item, scaffold_group_key)
    if mol_key is not None:
        return mol_key
    split_val = getattr(item, split_key, None)
    if split_val is None and split_key != "number":
        split_val = getattr(item, "number", None)
    if split_val is None:
        return None
    return _normalize_split_key(split_val)


def _tensor_hash(t: torch.Tensor) -> Optional[str]:
    if not torch.is_tensor(t):
        return None
    if not torch.isfinite(t).all().item():
        return None
    tc = t.detach().to("cpu").contiguous()
    arr = tc.numpy()
    h = hashlib.blake2b(digest_size=16)
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(str(tuple(arr.shape)).encode("utf-8"))
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def _joint_hessian_hash(hi: torch.Tensor, hij: torch.Tensor) -> Optional[str]:
    hi_hash = _tensor_hash(hi)
    hij_hash = _tensor_hash(hij)
    if hi_hash is None or hij_hash is None:
        return None
    h = hashlib.blake2b(digest_size=20)
    h.update(b"Hi::")
    h.update(hi_hash.encode("utf-8"))
    h.update(b"::Hij::")
    h.update(hij_hash.encode("utf-8"))
    return h.hexdigest()


def _worker_scan(
    worker_id: int,
    shard_paths: List[str],
    task: str,
    skip_nonfinite: bool,
    output_dir: str,
) -> Dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hi_hash_path = out_dir / f"worker_{worker_id:03d}_hi.hash"
    hij_hash_path = out_dir / f"worker_{worker_id:03d}_hij.hash"
    joint_hash_path = out_dir / f"worker_{worker_id:03d}_joint.hash"
    mol_source_path = out_dir / f"worker_{worker_id:03d}_mol_source.tsv"

    counters = Counter()
    source_counts = Counter()

    with (
        hi_hash_path.open("w", encoding="utf-8") as f_hi,
        hij_hash_path.open("w", encoding="utf-8") as f_hij,
        joint_hash_path.open("w", encoding="utf-8") as f_joint,
        mol_source_path.open("w", encoding="utf-8") as f_mol,
    ):
        for shard_path in shard_paths:
            try:
                items = torch.load(shard_path, map_location="cpu", weights_only=False)
            except Exception:
                counters["shard_load_errors"] += 1
                continue

            for item in items:
                counters["items_total"] += 1
                if not _is_valid_item(item, task, skip_nonfinite):
                    continue
                counters["items_valid"] += 1

                source = str(getattr(item, "source", None) or "unknown_source")
                source_counts[source] += 1

                mol_id = _molecule_id(item)
                if mol_id is not None:
                    f_mol.write(f"{source}\t{mol_id}\n")
                    counters["items_with_mol_id"] += 1

                hi = getattr(item, "Hi", None)
                hij = getattr(item, "Hij", None)

                hi_hash = None
                hij_hash = None

                if torch.is_tensor(hi):
                    counters["items_with_hi"] += 1
                    hi_hash = _tensor_hash(hi)
                    if hi_hash is not None:
                        f_hi.write(hi_hash + "\n")
                        counters["items_with_hi_finite"] += 1

                if torch.is_tensor(hij):
                    counters["items_with_hij"] += 1
                    hij_hash = _tensor_hash(hij)
                    if hij_hash is not None:
                        f_hij.write(hij_hash + "\n")
                        counters["items_with_hij_finite"] += 1

                if hi_hash is not None and hij_hash is not None:
                    counters["items_with_joint_hessian"] += 1
                    joint_hash = _joint_hessian_hash(hi, hij)
                    if joint_hash is not None:
                        f_joint.write(joint_hash + "\n")

    return {
        "worker_id": worker_id,
        "counters": dict(counters),
        "source_counts": dict(source_counts),
        "hi_hash_path": str(hi_hash_path),
        "hij_hash_path": str(hij_hash_path),
        "joint_hash_path": str(joint_hash_path),
        "mol_source_path": str(mol_source_path),
    }


def _line_count(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count


def _sort_unique(input_files: List[Path], output_path: Path) -> int:
    if not input_files:
        output_path.write_text("", encoding="utf-8")
        return 0
    with output_path.open("w", encoding="utf-8") as fout:
        subprocess.run(
            ["sort", "-u", *[str(p) for p in input_files]],
            stdout=fout,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
    return _line_count(output_path)


def _count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        return sum(1 for _ in reader)


def _count_gzip_csv_rows(path: Path) -> int:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        return sum(1 for _ in reader)


def _count_spice_hdf5(path: Path) -> Tuple[int, int]:
    if h5py is None:
        return 0, 0
    molecules = 0
    conformations = 0
    with h5py.File(path, "r") as h:
        for _, grp in h.items():
            molecules += 1
            if "conformations" in grp:
                conformations += int(grp["conformations"].shape[0])
            else:
                # Fallback if schema differs.
                conf_count = 0
                for ds_name in ("coords", "positions"):
                    if ds_name in grp:
                        ds = grp[ds_name]
                        if len(ds.shape) >= 3:
                            conf_count = int(ds.shape[0])
                        break
                conformations += conf_count
    return molecules, conformations


def _count_qm7x_hdf5(path: Path) -> Tuple[int, int]:
    if h5py is None:
        return 0, 0
    molecules = 0
    conformations = 0
    with h5py.File(path, "r") as h:
        for _, mol_grp in h.items():
            molecules += 1
            conformations += len(mol_grp.keys())
    return molecules, conformations


def _count_generic_hdf5(path: Path) -> Tuple[int, int]:
    if h5py is None:
        return 0, 0
    molecules = 0
    conformations = 0
    with h5py.File(path, "r") as h:
        for _, grp in h.items():
            molecules += 1
            if "conformations" in grp:
                conformations += int(grp["conformations"].shape[0])
                continue
            if "atXYZ" in grp:
                ds = grp["atXYZ"]
                if len(ds.shape) == 3:
                    conformations += int(ds.shape[0])
                elif len(ds.shape) == 2:
                    conformations += 1
                continue
            # QM7X-like nested conformer groups.
            child_groups = [k for k, v in grp.items() if hasattr(v, "keys")]
            if child_groups:
                conformations += len(child_groups)
            else:
                conformations += 1
    return molecules, conformations


def _count_sqlite_molecule_table(path: Path) -> Tuple[int, int]:
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='molecule'")
    has_table = cur.fetchone() is not None
    if not has_table:
        conn.close()
        return 0, 0

    cur.execute("SELECT COUNT(*) FROM molecule")
    rows = int(cur.fetchone()[0])

    smiles_distinct = rows
    try:
        cur.execute("SELECT COUNT(DISTINCT SMILES) FROM molecule")
        smiles_distinct = int(cur.fetchone()[0])
    except Exception:
        # Some molecule tables may not include SMILES.
        smiles_distinct = rows

    conn.close()
    return smiles_distinct, rows


def _add_inventory_entry(
    out: Dict[str, Dict[str, Any]],
    *,
    source: str,
    family: str,
    path: str,
    molecules: int,
    conformations: int,
) -> None:
    if source in out:
        out[source]["source_molecules"] += int(molecules)
        out[source]["source_conformations"] += int(conformations)
        # Keep first path as representative.
        return
    out[source] = {
        "path": path,
        "family": family,
        "source_molecules": int(molecules),
        "source_conformations": int(conformations),
    }


def _source_inventory(datasets_root: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}

    spice = datasets_root / "SPICE-2.0.1.hdf5"
    if spice.exists():
        mols, confs = _count_spice_hdf5(spice)
        _add_inventory_entry(
            out,
            source=spice.name,
            family="spice",
            path=str(spice),
            molecules=mols,
            conformations=confs,
        )

    summary = datasets_root / "summary.csv.gz"
    if summary.exists():
        rows = _count_gzip_csv_rows(summary)
        _add_inventory_entry(
            out,
            source=summary.name,
            family="qm9",
            path=str(summary),
            molecules=rows,
            conformations=rows,
        )

    des5m = datasets_root / "Donchev et al. DES5M.csv"
    if des5m.exists():
        rows = _count_csv_rows(des5m)
        _add_inventory_entry(
            out,
            source=des5m.name,
            family="des5m",
            path=str(des5m),
            molecules=rows,
            conformations=rows,
        )

    qm7x_root = datasets_root / "datasets--qm7x"
    if qm7x_root.exists():
        for path in sorted(qm7x_root.rglob("*.hdf5")) + sorted(qm7x_root.rglob("*.h5")):
            mols, confs = _count_qm7x_hdf5(path)
            _add_inventory_entry(
                out,
                source=path.name,
                family="qm7x",
                path=str(path),
                molecules=mols,
                conformations=confs,
            )

    # Raman-ChEMBL sources.
    for db_name in ("Raman-ChEMBL-part1.db", "Raman-ChEMBL-part2.db"):
        db_path = datasets_root / db_name
        if not db_path.exists():
            continue
        mols, confs = _count_sqlite_molecule_table(db_path)
        _add_inventory_entry(
            out,
            source=db_path.name,
            family="raman_chembl",
            path=str(db_path),
            molecules=mols,
            conformations=confs,
        )

    # QDpi source files are inside a tarball.
    qdpi_tar = datasets_root / "QDpiDataset-main.tar.gz"
    if qdpi_tar.exists() and h5py is not None:
        with tarfile.open(qdpi_tar, "r:gz") as tar:
            members = [m for m in tar.getmembers() if m.name.endswith((".h5", ".hdf5"))]
            for member in members:
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                suffix = ".hdf5" if member.name.endswith(".hdf5") else ".h5"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp_path = Path(tmp.name)
                    shutil.copyfileobj(extracted, tmp)
                try:
                    mols, confs = _count_generic_hdf5(tmp_path)
                finally:
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                _add_inventory_entry(
                    out,
                    source=Path(member.name).name,
                    family="qdpi",
                    path=f"{qdpi_tar}:{member.name}",
                    molecules=mols,
                    conformations=confs,
                )

    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit processed coverage and unique Hessians.")
    p.add_argument("--processed-root", default="/fsx/processed_all")
    p.add_argument("--datasets-root", default="/fsx/Datasets")
    p.add_argument("--task", default="polar")
    p.add_argument("--skip-nonfinite", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) // 2))
    p.add_argument("--output-dir", default="/tmp/processed_hessian_audit")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    processed_root = Path(args.processed_root)
    datasets_root = Path(args.datasets_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_paths = sorted(str(p) for p in processed_root.rglob("shard_*.pt"))
    if not shard_paths:
        raise SystemExit(f"No shards found under {processed_root}")

    workers = max(1, min(args.workers, len(shard_paths)))
    chunks: List[List[str]] = [[] for _ in range(workers)]
    for i, p in enumerate(shard_paths):
        chunks[i % workers].append(p)

    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(_worker_scan, wid, chunk, args.task, args.skip_nonfinite, str(out_dir))
            for wid, chunk in enumerate(chunks)
            if chunk
        ]
        for fut in futs:
            results.append(fut.result())

    total_counts = Counter()
    source_counts = Counter()
    hi_hash_files: List[Path] = []
    hij_hash_files: List[Path] = []
    joint_hash_files: List[Path] = []
    mol_source_files: List[Path] = []

    for r in results:
        total_counts.update(r["counters"])
        source_counts.update(r["source_counts"])
        hi_hash_files.append(Path(r["hi_hash_path"]))
        hij_hash_files.append(Path(r["hij_hash_path"]))
        joint_hash_files.append(Path(r["joint_hash_path"]))
        mol_source_files.append(Path(r["mol_source_path"]))

    unique_hi_file = out_dir / "unique_hi.hash"
    unique_hij_file = out_dir / "unique_hij.hash"
    unique_joint_file = out_dir / "unique_joint_hessian.hash"
    unique_mol_source_file = out_dir / "unique_mol_source.tsv"

    unique_hi = _sort_unique(hi_hash_files, unique_hi_file)
    unique_hij = _sort_unique(hij_hash_files, unique_hij_file)
    unique_joint = _sort_unique(joint_hash_files, unique_joint_file)
    _sort_unique(mol_source_files, unique_mol_source_file)

    unique_molecules_by_source = Counter()
    with unique_mol_source_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            source, _mol = line.split("\t", 1)
            unique_molecules_by_source[source] += 1

    inventory = _source_inventory(datasets_root)
    coverage_rows = []
    for source, meta in sorted(inventory.items()):
        processed_samples = int(source_counts.get(source, 0))
        processed_unique_mols = int(unique_molecules_by_source.get(source, 0))
        source_molecules = int(meta.get("source_molecules", 0))
        source_conformations = int(meta.get("source_conformations", 0))
        unprocessed_molecules = max(source_molecules - processed_unique_mols, 0)
        unprocessed_conformations = max(source_conformations - processed_samples, 0)
        coverage_rows.append(
            {
                "source": source,
                "family": meta.get("family", "unknown"),
                "source_molecules": source_molecules,
                "processed_unique_molecules": processed_unique_mols,
                "unprocessed_molecules": unprocessed_molecules,
                "source_conformations": source_conformations,
                "processed_samples": processed_samples,
                "unprocessed_conformations": unprocessed_conformations,
            }
        )

    family_cov = Counter()
    for row in coverage_rows:
        fam = row["family"]
        family_cov[(fam, "source_molecules")] += row["source_molecules"]
        family_cov[(fam, "processed_unique_molecules")] += row["processed_unique_molecules"]
        family_cov[(fam, "unprocessed_molecules")] += row["unprocessed_molecules"]
        family_cov[(fam, "source_conformations")] += row["source_conformations"]
        family_cov[(fam, "processed_samples")] += row["processed_samples"]
        family_cov[(fam, "unprocessed_conformations")] += row["unprocessed_conformations"]

    family_rows = []
    families = sorted({k[0] for k in family_cov})
    for fam in families:
        family_rows.append(
            {
                "family": fam,
                "source_molecules": family_cov[(fam, "source_molecules")],
                "processed_unique_molecules": family_cov[(fam, "processed_unique_molecules")],
                "unprocessed_molecules": family_cov[(fam, "unprocessed_molecules")],
                "source_conformations": family_cov[(fam, "source_conformations")],
                "processed_samples": family_cov[(fam, "processed_samples")],
                "unprocessed_conformations": family_cov[(fam, "unprocessed_conformations")],
            }
        )

    # "Unprocessed shards" is not a strict source concept; provide an estimate.
    shards_found = len(shard_paths)
    items_valid = int(total_counts.get("items_valid", 0))
    avg_items_per_shard = (items_valid / shards_found) if shards_found else 0.0
    total_unprocessed_conformations = sum(r["unprocessed_conformations"] for r in coverage_rows)
    estimated_unprocessed_shards = (
        math.ceil(total_unprocessed_conformations / avg_items_per_shard)
        if avg_items_per_shard > 0
        else 0
    )

    coverage_csv = out_dir / "source_coverage.csv"
    with coverage_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "source",
                "family",
                "source_molecules",
                "processed_unique_molecules",
                "unprocessed_molecules",
                "source_conformations",
                "processed_samples",
                "unprocessed_conformations",
            ],
        )
        w.writeheader()
        w.writerows(coverage_rows)

    family_csv = out_dir / "family_coverage.csv"
    with family_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "family",
                "source_molecules",
                "processed_unique_molecules",
                "unprocessed_molecules",
                "source_conformations",
                "processed_samples",
                "unprocessed_conformations",
            ],
        )
        w.writeheader()
        w.writerows(family_rows)

    source_counts_csv = out_dir / "processed_source_counts.csv"
    with source_counts_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "processed_samples", "processed_unique_molecules"])
        for src in sorted(set(source_counts) | set(unique_molecules_by_source)):
            w.writerow([src, int(source_counts.get(src, 0)), int(unique_molecules_by_source.get(src, 0))])

    summary = {
        "processed_root": str(processed_root),
        "datasets_root": str(datasets_root),
        "task": args.task,
        "workers": workers,
        "shards_found": shards_found,
        "scan_counts": dict(total_counts),
        "hessian": {
            "items_with_hi": int(total_counts.get("items_with_hi", 0)),
            "items_with_hij": int(total_counts.get("items_with_hij", 0)),
            "items_with_joint_hessian": int(total_counts.get("items_with_joint_hessian", 0)),
            "unique_hi": unique_hi,
            "unique_hij": unique_hij,
            "unique_joint_hessian": unique_joint,
        },
        "coverage": {
            "total_unprocessed_molecules": int(sum(r["unprocessed_molecules"] for r in coverage_rows)),
            "total_unprocessed_conformations": int(total_unprocessed_conformations),
            "avg_items_per_shard": avg_items_per_shard,
            "estimated_unprocessed_shards": int(estimated_unprocessed_shards),
        },
        "outputs": {
            "source_coverage_csv": str(coverage_csv),
            "family_coverage_csv": str(family_csv),
            "processed_source_counts_csv": str(source_counts_csv),
            "unique_hi_hashes": str(unique_hi_file),
            "unique_hij_hashes": str(unique_hij_file),
            "unique_joint_hessian_hashes": str(unique_joint_file),
            "unique_mol_source": str(unique_mol_source_file),
        },
    }

    summary_path = out_dir / "audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
