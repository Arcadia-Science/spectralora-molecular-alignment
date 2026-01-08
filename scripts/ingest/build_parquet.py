import argparse
import csv
import io
import os
from pathlib import Path
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq


def parse_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value, default=None):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_rows(rows, dataset):
    if not rows:
        return None

    data_rows = rows[1:]  # skip header
    if not data_rows:
        return None

    meta = data_rows[0]

    if dataset == "qm9s":
        molecule_id = parse_int(meta[1])
        qm9_id = parse_int(meta[2])
        smiles = meta[3] or None
        n_atoms = parse_int(meta[4])
        environment = None
        coord_start = 10
    else:
        molecule_id = parse_int(meta[1])
        qm9_id = None
        smiles = meta[2] or None
        n_atoms = parse_int(meta[3])
        environment = meta[4] or None
        coord_start = 1

    if n_atoms is None:
        return None

    z = []
    pos = []
    for row in data_rows[coord_start:coord_start + n_atoms]:
        atomic_num = parse_int(row[1])
        x = parse_float(row[2])
        y = parse_float(row[3])
        zc = parse_float(row[4])
        if atomic_num is None or x is None or y is None or zc is None:
            continue
        z.append(atomic_num)
        pos.append([x, y, zc])

    if len(z) != n_atoms:
        return None

    return {
        "dataset": dataset,
        "molecule_id": molecule_id,
        "qm9_id": qm9_id,
        "smiles": smiles,
        "n_atoms": n_atoms,
        "environment": environment,
        "z": z,
        "pos": pos,
    }


def iter_zip_csv(zip_path, limit=None):
    count = 0
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".csv"):
                continue
            with zf.open(name) as handle:
                text = io.TextIOWrapper(handle, newline="")
                reader = csv.reader(text)
                rows = list(reader)
                yield name, rows
                count += 1
                if limit is not None and count >= limit:
                    return


def build_parquet(dataset, zip_path, out_dir, shard_size, limit):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    batch = []
    shard_index = 0

    def flush():
        nonlocal shard_index
        if not batch:
            return
        table = pa.Table.from_pylist(batch)
        out_path = out_dir / f"{dataset}_{shard_index:05d}.parquet"
        pq.write_table(table, out_path)
        batch.clear()
        shard_index += 1

    for name, rows in iter_zip_csv(zip_path, limit=limit):
        record = parse_rows(rows, dataset)
        if record is None:
            continue
        batch.append(record)
        if len(batch) >= shard_size:
            flush()

    flush()


def main():
    parser = argparse.ArgumentParser(description="Build Parquet shards from DetaNet CSV zips.")
    parser.add_argument("--dataset", required=True, choices=["qm9s", "ext_val", "ext_val_env"], help="Dataset name")
    parser.add_argument("--zip", dest="zip_path", required=True, help="Path to dataset zip")
    parser.add_argument("--out", dest="out_dir", required=True, help="Output directory for Parquet shards")
    parser.add_argument("--shard-size", type=int, default=1000, help="Rows per Parquet shard")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for number of CSV files to parse")

    args = parser.parse_args()
    build_parquet(args.dataset, args.zip_path, args.out_dir, args.shard_size, args.limit)


if __name__ == "__main__":
    main()
