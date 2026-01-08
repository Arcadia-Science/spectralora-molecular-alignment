import argparse
import asyncio
from pathlib import Path

import asyncpg
import pyarrow.parquet as pq


INSERT_MOLECULE_SQL = """
INSERT INTO molecules (
    dataset,
    molecule_id,
    qm9_id,
    smiles,
    n_atoms,
    environment,
    source_path,
    source_row,
    data_format
) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
ON CONFLICT (dataset, molecule_id) DO NOTHING;
"""


async def load_dataset(db_url, dataset, parquet_dir):
    parquet_dir = Path(parquet_dir)
    files = sorted(parquet_dir.glob(f"{dataset}_*.parquet"))
    if not files:
        raise SystemExit(f"No Parquet files found for dataset '{dataset}' in {parquet_dir}")

    pool = await asyncpg.create_pool(dsn=db_url)
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO datasets (name) VALUES ($1) ON CONFLICT (name) DO NOTHING",
            dataset,
        )

        for path in files:
            pf = pq.ParquetFile(path)
            row_offset = 0
            for rg in range(pf.num_row_groups):
                table = pf.read_row_group(
                    rg,
                    columns=["molecule_id", "qm9_id", "smiles", "n_atoms", "environment"],
                )
                data = table.to_pydict()
                rows = []
                for i in range(len(table)):
                    rows.append(
                        (
                            dataset,
                            data["molecule_id"][i],
                            data["qm9_id"][i],
                            data["smiles"][i],
                            data["n_atoms"][i],
                            data["environment"][i],
                            str(path.relative_to(parquet_dir)),
                            row_offset + i,
                            "parquet",
                        )
                    )
                if rows:
                    await conn.executemany(INSERT_MOLECULE_SQL, rows)
                row_offset += len(table)

    await pool.close()


def main():
    parser = argparse.ArgumentParser(description="Load Parquet registry into Postgres.")
    parser.add_argument("--db-url", required=True, help="Postgres URL")
    parser.add_argument("--dataset", required=True, choices=["qm9s", "ext_val", "ext_val_env"], help="Dataset name")
    parser.add_argument("--parquet-dir", required=True, help="Directory with Parquet shards")
    args = parser.parse_args()

    asyncio.run(load_dataset(args.db_url, args.dataset, args.parquet_dir))


if __name__ == "__main__":
    main()
