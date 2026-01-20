# Dataset Pipeline Plan (Shards + Metadata + Confidence)

This plan describes how to store PyG `Data` objects as shards, index them with per-field provenance in SQLite, and support confidence-aware training on AWS (S3 + FSx for Lustre).

## Goals
- Keep **PyG `Data` objects intact** as `.pt` shards for training.
- Maintain a **per-dataset SQLite index** for fast retrieval.
- Track **which fields are ground-truth vs imputed/generated**.
- Leave a **confidence slot** per field for later use in the loss function.
- Support **heterogeneous schemas** across datasets.

## Storage Layout (S3)
```
s3://<bucket>/Datasets/<dataset>/<version>/
  shards/
    shard_000000.pt
    shard_000001.pt
  manifest.jsonl
  index.sqlite
  models.json
  README.md
```

Notes:
- `dataset` and `version` are required. Example: `spice/v1`, `nabla2/v1`.
- `manifest.jsonl` is append-only, one line per shard:
  `{"shard": ".../shards/shard_000000.pt", "count": 1000, "start_id": 1, "end_id": 1000}`
- `models.json` lists checkpoint URIs and configs used during generation (optional but recommended).

## Shard Format
- `torch.save(list[Data])` for each shard.
- Shard size configurable (e.g., 1000).
- Shards are immutable once published.

## Per-Dataset Index (SQLite)
One SQLite DB per dataset/version stored next to shards:
`Datasets/<dataset>/<version>/index.sqlite`

### Schema (v1)
```sql
-- Dataset metadata
CREATE TABLE datasets (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  version TEXT NOT NULL,
  source TEXT NOT NULL,
  created_at TEXT NOT NULL,
  description TEXT
);

-- ML/DFT models used to generate fields
CREATE TABLE models (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  type TEXT NOT NULL,             -- mace, deepmd, psi4
  checkpoint_uri TEXT,
  config_json TEXT,
  hash TEXT
);

-- Shard inventory
CREATE TABLE shards (
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

-- One row per Data object
CREATE TABLE items (
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
  source_dataset TEXT,            -- explicit dataset name for audit
  FOREIGN KEY(dataset_id) REFERENCES datasets(id),
  FOREIGN KEY(shard_id) REFERENCES shards(id)
);

-- Per-field provenance and confidence (normalized)
CREATE TABLE item_fields (
  item_id INTEGER NOT NULL,
  field TEXT NOT NULL,            -- e.g., dipole, polar, Hi, Hij
  source TEXT NOT NULL,           -- dataset, psi4, deepmd, mace, charge_approx, zero_fill, derived:*
  generated INTEGER NOT NULL,     -- 0/1
  imputed INTEGER NOT NULL,       -- 0/1
  confidence REAL,                -- NULL until populated
  model_id INTEGER,
  PRIMARY KEY(item_id, field),
  FOREIGN KEY(item_id) REFERENCES items(id),
  FOREIGN KEY(model_id) REFERENCES models(id)
);

CREATE INDEX idx_items_smile ON items(smile);
CREATE INDEX idx_items_subset ON items(subset);
CREATE INDEX idx_items_elements ON items(element_set);
CREATE INDEX idx_fields_field ON item_fields(field);
CREATE INDEX idx_fields_imputed ON item_fields(imputed);
```

### Field Source Taxonomy
Examples:
- `dataset`, `formation_energy`, `dft_total_energy`
- `scf_dipole`, `scf_quadrupole`
- `mbis_charges`, `mbis_dipole_sum`, `mbis_octupoles_sum`
- `psi4`, `mace`, `deepmd_pot`, `deepmd_dipole`, `deepmd_polar`
- `charge_approx`, `gasteiger`, `zero_fill`
- `derived:psi4`, `derived:mace`, `derived:deepmd_pot`

## Confidence Strategy (Method-Based Baseline)
- Store a **method-based confidence** in `field_confidence` on each `Data` object.
- Use a simple mapping by `field_source` (dataset/psi4/deepmd/mace/approx).
- Optionally update later with ensemble/uncertainty estimates.

## Indexer Flow
1. Generate shards (`shard_*.pt` + `manifest.jsonl`).
2. Run indexer:
   - Read each shard
   - Insert `items` + `item_fields` rows
   - Populate `shards` table
3. Upload `index.sqlite` to S3 alongside shards.

## Loader & Training Batches
### Retrieval
- Query SQLite for items matching constraints (subset, elements, required fields, imputed flags).
- Load shard from FSx/S3, extract `Data` by `idx_in_shard`.

### Batch Construction
- Default: use PyG `DataLoader` on a list of `Data` objects.
- Confidence-aware loss:
  - Use `item_fields.confidence` to weight loss per field or per sample.
  - For missing confidence (NULL), treat as `1.0`.

## Cache Strategy
- **FSx for Lustre** linked to S3 prefix for POSIX access.
- Optional in-process LRU cache of recently used shards.
- Optional local NVMe cache on EC2 for hot shards.

## Resilience & Versioning
- Shards are immutable.
- Index build is idempotent (rebuild from shards if needed).
- `manifest.jsonl` is append-only.
- Include `schema_version` in SQLite (table or PRAGMA).
- Store `models.json` to capture generation provenance.

## AWS Components (Minimal)
- S3 bucket: canonical storage.
- FSx for Lustre: mounted on EC2 for training.
- EC2 instance(s): training + index queries.
