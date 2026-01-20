from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Iterable

import torch


def init_db(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS datasets (
          id INTEGER PRIMARY KEY,
          name TEXT NOT NULL,
          version TEXT NOT NULL,
          source TEXT NOT NULL,
          created_at TEXT NOT NULL,
          description TEXT
        );

        CREATE TABLE IF NOT EXISTS shards (
          id INTEGER PRIMARY KEY,
          dataset_id INTEGER NOT NULL,
          s3_uri TEXT NOT NULL,
          checksum TEXT,
          bytes INTEGER,
          count INTEGER NOT NULL,
          start_id INTEGER NOT NULL,
          end_id INTEGER NOT NULL,
          created_at TEXT NOT NULL,
          FOREIGN KEY(dataset_id) REFERENCES datasets(id)
        );

        CREATE TABLE IF NOT EXISTS items (
          id INTEGER PRIMARY KEY,
          dataset_id INTEGER NOT NULL,
          shard_id INTEGER NOT NULL,
          idx_in_shard INTEGER NOT NULL,
          number INTEGER NOT NULL,
          smile TEXT,
          mol_key TEXT,
          subset TEXT,
          conformer_id INTEGER,
          n_atoms INTEGER,
          n_edges INTEGER,
          element_set TEXT,
          source_dataset TEXT,
          FOREIGN KEY(dataset_id) REFERENCES datasets(id),
          FOREIGN KEY(shard_id) REFERENCES shards(id)
        );

        CREATE TABLE IF NOT EXISTS item_fields (
          item_id INTEGER NOT NULL,
          field TEXT NOT NULL,
          source TEXT NOT NULL,
          generated INTEGER NOT NULL,
          imputed INTEGER NOT NULL,
          confidence REAL,
          PRIMARY KEY(item_id, field),
          FOREIGN KEY(item_id) REFERENCES items(id)
        );
        """
    )
    conn.commit()


def insert_dataset(conn: sqlite3.Connection, name: str, version: str, source: str, description: str | None) -> int:
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO datasets(name, version, source, created_at, description) VALUES (?, ?, ?, datetime('now'), ?)",
        (name, version, source, description),
    )
    conn.commit()
    return int(cur.lastrowid)


def insert_shard(
    conn: sqlite3.Connection,
    dataset_id: int,
    shard_uri: str,
    count: int,
    start_id: int,
    end_id: int,
) -> int:
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO shards(dataset_id, s3_uri, count, start_id, end_id, created_at) VALUES (?, ?, ?, ?, ?, datetime('now'))",
        (dataset_id, shard_uri, count, start_id, end_id),
    )
    conn.commit()
    return int(cur.lastrowid)


def iter_manifest(manifest_path: Path) -> Iterable[dict]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Index shard_*.pt files into SQLite.")
    parser.add_argument("--shards-dir", required=True)
    parser.add_argument("--output-db", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-version", default="v1")
    parser.add_argument("--dataset-source", default="local")
    parser.add_argument("--description", default=None)
    args = parser.parse_args()

    shards_dir = Path(args.shards_dir)
    manifest_path = shards_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise RuntimeError(f"manifest.jsonl not found in {shards_dir}")

    conn = sqlite3.connect(args.output_db)
    init_db(conn)
    dataset_id = insert_dataset(conn, args.dataset_name, args.dataset_version, args.dataset_source, args.description)

    cur = conn.cursor()
    for entry in iter_manifest(manifest_path):
        shard_path = Path(entry["shard"])
        items = torch.load(shard_path, weights_only=False)
        shard_id = insert_shard(
            conn,
            dataset_id=dataset_id,
            shard_uri=str(shard_path),
            count=int(entry.get("count", len(items))),
            start_id=int(entry.get("start_id", 0)),
            end_id=int(entry.get("end_id", 0)),
        )

        item_rows = []
        field_rows = []
        for idx_in_shard, data in enumerate(items):
            z = getattr(data, "z", None)
            edge_index = getattr(data, "edge_index", None)
            element_set = None
            if isinstance(z, torch.Tensor):
                element_set = ",".join(str(int(v)) for v in sorted(set(z.detach().cpu().tolist())))
            n_atoms = int(z.shape[0]) if isinstance(z, torch.Tensor) else None
            n_edges = int(edge_index.shape[1]) if isinstance(edge_index, torch.Tensor) else None
            item_rows.append(
                (
                    dataset_id,
                    shard_id,
                    idx_in_shard,
                    int(getattr(data, "number", idx_in_shard)),
                    getattr(data, "smile", None),
                    getattr(data, "mol_key", None),
                    getattr(data, "subset", None),
                    getattr(data, "conformer_id", None),
                    n_atoms,
                    n_edges,
                    element_set,
                    getattr(data, "source", None),
                )
            )

        cur.executemany(
            """
            INSERT INTO items(
              dataset_id, shard_id, idx_in_shard, number, smile, mol_key, subset,
              conformer_id, n_atoms, n_edges, element_set, source_dataset
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            item_rows,
        )
        conn.commit()

        cur.execute("SELECT id FROM items WHERE shard_id = ? ORDER BY idx_in_shard", (shard_id,))
        item_ids = [row[0] for row in cur.fetchall()]
        for item_id, data in zip(item_ids, items):
            field_source = getattr(data, "field_source", {}) or {}
            field_generated = getattr(data, "field_generated", {}) or {}
            field_imputed = getattr(data, "field_imputed", {}) or {}
            field_conf = getattr(data, "field_confidence", {}) or {}
            for field, source in field_source.items():
                field_rows.append(
                    (
                        item_id,
                        field,
                        source,
                        int(bool(field_generated.get(field, False))),
                        int(bool(field_imputed.get(field, False))),
                        field_conf.get(field),
                    )
                )
        cur.executemany(
            """
            INSERT INTO item_fields(item_id, field, source, generated, imputed, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            field_rows,
        )
        conn.commit()

    conn.close()


if __name__ == "__main__":
    main()
