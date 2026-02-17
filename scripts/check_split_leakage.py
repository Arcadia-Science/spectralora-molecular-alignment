#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.train_detanet import (  # noqa: E402
    _normalize_split_key,
    _resolve_molecule_key,
    _resolve_split_token,
    _split_label,
)


def _list_shards(shard_dir: Optional[str], shard_list: Optional[str]) -> List[str]:
    if shard_list:
        return [line.strip() for line in Path(shard_list).read_text().splitlines() if line.strip()]
    if not shard_dir:
        raise ValueError("Provide --shard-dir or --shard-list.")
    return sorted(str(p) for p in Path(shard_dir).glob("shard_*.pt"))


def _is_valid_item(item, task: str, skip_nonfinite: bool) -> bool:
    target = getattr(item, task, None)
    if target is None:
        return False
    pos = getattr(item, "pos", None)
    if pos is None:
        return False
    if not skip_nonfinite:
        return True
    if torch.is_tensor(target) and not torch.isfinite(target).all().item():
        return False
    if not torch.is_tensor(target) and isinstance(target, (float, int)) and not math.isfinite(target):
        return False
    if torch.is_tensor(pos) and not torch.isfinite(pos).all().item():
        return False
    return True


def _molecule_id(item, scaffold_group_key: str, split_key: str) -> Optional[str]:
    mol_key = _resolve_molecule_key(item, scaffold_group_key)
    if mol_key is not None:
        return mol_key
    split_val = getattr(item, split_key, None)
    if split_val is None and split_key != "number":
        split_val = getattr(item, "number", None)
    if split_val is None:
        return None
    return _normalize_split_key(split_val)


def _scaffold_id(token: str) -> Optional[str]:
    prefix = "SCAFFOLD::"
    if not token.startswith(prefix):
        return None
    return token[len(prefix) :]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check split leakage by molecule/scaffold.")
    parser.add_argument("--shard-dir", default=None)
    parser.add_argument("--shard-list", default=None)
    parser.add_argument("--task", required=True)
    parser.add_argument("--split-key", default="mol_key")
    parser.add_argument("--split-method", default="hash", choices=["hash", "scaffold"])
    parser.add_argument("--scaffold-group-key", default="mol_key")
    parser.add_argument("--scaffold-smiles-key", default="smile")
    parser.add_argument(
        "--scaffold-include-chirality",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--scaffold-fallback", default="molecule", choices=["molecule", "global"])
    parser.add_argument("--split-seed", type=int, default=123)
    parser.add_argument("--split-train", type=float, default=0.7)
    parser.add_argument("--split-val", type=float, default=0.1)
    parser.add_argument("--skip-nonfinite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-shards", type=int, default=0, help="0 means all shards.")
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--max-examples", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    shard_paths = _list_shards(args.shard_dir, args.shard_list)
    if args.max_shards > 0:
        shard_paths = shard_paths[: args.max_shards]
    if not shard_paths:
        raise SystemExit("No shard paths found.")

    counts = Counter()
    sample_counts = Counter()
    split_cache: Dict[str, str] = {}
    molecule_to_split: Dict[str, str] = {}
    scaffold_to_split: Dict[str, str] = {}
    molecule_conflicts: List[dict] = []
    scaffold_conflicts: List[dict] = []
    shard_errors: List[dict] = []

    for idx, shard_path in enumerate(shard_paths, start=1):
        try:
            data_list = torch.load(shard_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            counts["shard_errors"] += 1
            if len(shard_errors) < args.max_examples:
                shard_errors.append({"path": shard_path, "error": repr(exc)})
            continue
        for item in data_list:
            counts["items_total"] += 1
            if not _is_valid_item(item, args.task, args.skip_nonfinite):
                continue
            counts["items_valid"] += 1

            token = _resolve_split_token(
                item,
                split_method=args.split_method,
                split_key=args.split_key,
                scaffold_group_key=args.scaffold_group_key,
                scaffold_smiles_key=args.scaffold_smiles_key,
                scaffold_include_chirality=args.scaffold_include_chirality,
                scaffold_fallback=args.scaffold_fallback,
                split_cache=split_cache,
            )
            if token is None:
                counts["items_missing_split_token"] += 1
                continue
            split = _split_label(token, args.split_seed, args.split_train, args.split_val)
            sample_counts[split] += 1

            molecule_id = _molecule_id(item, args.scaffold_group_key, args.split_key)
            if molecule_id is not None:
                prev_split = molecule_to_split.get(molecule_id)
                if prev_split is None:
                    molecule_to_split[molecule_id] = split
                elif prev_split != split:
                    counts["molecule_conflicts"] += 1
                    if len(molecule_conflicts) < args.max_examples:
                        molecule_conflicts.append(
                            {"molecule": molecule_id, "first_split": prev_split, "next_split": split}
                        )

            scaffold_id = _scaffold_id(token)
            if scaffold_id:
                prev_scaffold_split = scaffold_to_split.get(scaffold_id)
                if prev_scaffold_split is None:
                    scaffold_to_split[scaffold_id] = split
                elif prev_scaffold_split != split:
                    counts["scaffold_conflicts"] += 1
                    if len(scaffold_conflicts) < args.max_examples:
                        scaffold_conflicts.append(
                            {"scaffold": scaffold_id, "first_split": prev_scaffold_split, "next_split": split}
                        )

        if args.progress_every > 0 and idx % args.progress_every == 0:
            print(
                f"[progress] shards={idx}/{len(shard_paths)} valid={counts['items_valid']} "
                f"molecule_conflicts={counts['molecule_conflicts']}",
                file=sys.stderr,
            )

    molecule_split_counts = Counter(molecule_to_split.values())
    scaffold_split_counts = Counter(scaffold_to_split.values())

    summary = {
        "shards_checked": len(shard_paths),
        "shard_errors": counts["shard_errors"],
        "items_total": counts["items_total"],
        "items_valid": counts["items_valid"],
        "items_missing_split_token": counts["items_missing_split_token"],
        "sample_split_counts": dict(sample_counts),
        "unique_molecules": len(molecule_to_split),
        "unique_molecule_split_counts": dict(molecule_split_counts),
        "unique_scaffolds": len(scaffold_to_split),
        "unique_scaffold_split_counts": dict(scaffold_split_counts),
        "molecule_conflicts": counts["molecule_conflicts"],
        "scaffold_conflicts": counts["scaffold_conflicts"],
        "molecule_conflict_examples": molecule_conflicts,
        "scaffold_conflict_examples": scaffold_conflicts,
        "shard_error_examples": shard_errors,
        "params": {
            "task": args.task,
            "split_method": args.split_method,
            "split_key": args.split_key,
            "split_seed": args.split_seed,
            "split_train": args.split_train,
            "split_val": args.split_val,
            "scaffold_group_key": args.scaffold_group_key,
            "scaffold_smiles_key": args.scaffold_smiles_key,
            "scaffold_include_chirality": args.scaffold_include_chirality,
            "scaffold_fallback": args.scaffold_fallback,
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    has_conflict = counts["molecule_conflicts"] > 0
    if args.split_method == "scaffold":
        has_conflict = has_conflict or counts["scaffold_conflicts"] > 0
    return 2 if has_conflict else 0


if __name__ == "__main__":
    raise SystemExit(main())
