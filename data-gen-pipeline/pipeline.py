import argparse
import hashlib
import json
import os
import sqlite3
import urllib.parse
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch

E_ANGSTROM_TO_DEBYE = 4.80320427

ATOMIC_POLARIZABILITY = {
    1: 0.667,  # H
    3: 24.3,  # Li
    4: 5.6,  # Be
    5: 3.0,  # B
    6: 1.76,  # C
    7: 1.10,  # N
    8: 0.802,  # O
    9: 0.557,  # F
    10: 0.395,  # Ne
    11: 24.1,  # Na
    12: 10.6,  # Mg
    13: 6.8,  # Al
    14: 5.38,  # Si
    15: 3.63,  # P
    16: 2.90,  # S
    17: 2.18,  # Cl
    35: 3.05,  # Br
    53: 5.35,  # I
}

PAULING_EN = {
    1: 2.20,
    3: 0.98,
    4: 1.57,
    5: 2.04,
    6: 2.55,
    7: 3.04,
    8: 3.44,
    9: 3.98,
    11: 0.93,
    12: 1.31,
    13: 1.61,
    14: 1.90,
    15: 2.19,
    16: 2.58,
    17: 3.16,
    35: 2.96,
    53: 2.66,
}

try:
    from torch_geometric.data import Data
except Exception as exc:  # pragma: no cover - torch_geometric optional in CI
    Data = None
    _DATA_IMPORT_ERROR = exc

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except Exception:  # pragma: no cover - rdkit optional in CI
    Chem = None
    AllChem = None

try:
    from ase.data import atomic_masses
except Exception:  # pragma: no cover - ase optional in CI
    atomic_masses = None

try:
    from .deepmd_backend import DeepMDDipoleBackend, DeepMDPolarBackend, DeepMDPotHessianBackend
    from .mace_backend import MaceHessianBackend
    from .psi4_backend import Psi4HessianBackend
except ImportError:  # pragma: no cover - fallback for script-style imports
    from deepmd_backend import DeepMDDipoleBackend, DeepMDPolarBackend, DeepMDPotHessianBackend
    from mace_backend import MaceHessianBackend
    from psi4_backend import Psi4HessianBackend


@dataclass
class PipelineConfig:
    output_dir: Path
    device: torch.device
    limit: Optional[int] = None
    log_every: int = 50
    distributed: bool = False
    rank: int = 0
    world_size: int = 1
    save_device: Optional[torch.device] = None
    smiles: Sequence[str] = ()
    smiles_file: Optional[Path] = None
    db_path: Optional[Path] = None
    hdf5_paths: Sequence[Path] = ()
    hdf5_subset: Sequence[str] = ()
    checkpoint_cache: Path = Path("data-gen-pipeline/checkpoints")
    pos_step: float = 1e-3
    dft_atom_cutoff: int = 20
    graph_k: int = 20
    graph_clamp_min: float = 0.5
    graph_clamp_max: float = 100.0
    dipole_model: Optional[Path] = None
    polar_model: Optional[Path] = None
    deepmd_pot_model: Optional[Path] = None
    deepmd_head: Optional[str] = None
    deepmd_type_map: Optional[str] = None
    deepmd_dipole_unit: str = "au"
    deepmd_atomic_energy: bool = False
    mace_model: Path = Path("data-gen-pipeline/checkpoints/2024-07-12-mace-128-L1_epoch-199.model")
    psi4_fallback: bool = True
    psi4_method: str = "B3LYP"
    psi4_basis: str = "cc-pVTZ"
    psi4_memory: str = "2 GB"
    psi4_threads: int = 1
    psi4_scf_type: str = "df"
    psi4_guess: str = "sad"
    psi4_charge: int = 0
    psi4_multiplicity: int = 1
    psi4_quiet: bool = True
    rdkit_max_attempts: int = 10
    rdkit_optimize: bool = True
    allow_missing_hyperpolar: bool = False
    allow_missing_polar: bool = False
    allow_missing_dipole: bool = False
    shard_size: Optional[int] = None


@dataclass
class SmilesItem:
    number: int
    smile: str
    pos: Optional[torch.Tensor] = None
    z: Optional[torch.Tensor] = None
    edge_index: Optional[torch.Tensor] = None
    energy: Optional[torch.Tensor] = None
    dipole: Optional[torch.Tensor] = None
    npacharge: Optional[torch.Tensor] = None
    polar: Optional[torch.Tensor] = None
    quadrupole: Optional[torch.Tensor] = None
    octapole: Optional[torch.Tensor] = None
    hyperpolar: Optional[torch.Tensor] = None
    dedipole: Optional[torch.Tensor] = None
    depolar: Optional[torch.Tensor] = None
    mol_key: Optional[str] = None
    subset: Optional[str] = None
    source: Optional[str] = None
    conformer_id: Optional[int] = None
    field_source: Optional[Dict[str, str]] = None


def ensure_data_available() -> None:
    if Data is None:
        raise RuntimeError(
            "torch_geometric is required to build Data objects. "
            f"Import error: {_DATA_IMPORT_ERROR}"
        )


def ensure_rdkit_available() -> None:
    if Chem is None or AllChem is None:
        raise RuntimeError("RDKit is required for SMILES -> 3D geometry.")


def resolve_checkpoint(path_or_url: Optional[str], cache_dir: Path) -> Optional[str]:
    if path_or_url is None:
        return None
    if path_or_url.startswith(("http://", "https://")):
        cache_dir.mkdir(parents=True, exist_ok=True)
        parsed = urllib.parse.urlparse(path_or_url)
        name = Path(parsed.path).name or "checkpoint.pt"
        tag = hashlib.sha256(path_or_url.encode("utf-8")).hexdigest()[:8]
        dest = cache_dir / f"{tag}_{name}"
        if not dest.exists():
            with urllib.request.urlopen(path_or_url) as response, dest.open("wb") as out_file:
                out_file.write(response.read())
        return str(dest)
    return path_or_url


def iter_smiles(cfg: PipelineConfig) -> Iterable[SmilesItem]:
    if cfg.hdf5_paths:
        yield from iter_hdf5(cfg)
        return
    if cfg.smiles:
        for idx, smile in enumerate(cfg.smiles, start=1):
            yield SmilesItem(number=idx, smile=smile)
        return
    if cfg.smiles_file:
        with cfg.smiles_file.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle, start=1):
                smile = line.strip().split()[0]
                if not smile:
                    continue
                yield SmilesItem(number=idx, smile=smile)
        return
    if cfg.db_path:
        conn = sqlite3.connect(cfg.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(molecule)")
        cols = {row[1] for row in cur.fetchall()}
        select_cols = ["id", "SMILES", "blob_data"]
        for name in ("Dx", "Dy", "Dz", "isotropic_pol"):
            if name in cols:
                select_cols.append(name)
        cur.execute(f"SELECT {', '.join(select_cols)} FROM molecule ORDER BY id")
        for row in cur:
            pos, z = parse_blob_geometry(row["blob_data"])
            dipole = None
            if all(name in row.keys() for name in ("Dx", "Dy", "Dz")):
                dipole = torch.tensor([row["Dx"], row["Dy"], row["Dz"]], dtype=torch.float32).view(1, 3)
            polar = None
            if "isotropic_pol" in row.keys() and row["isotropic_pol"] is not None:
                iso = float(row["isotropic_pol"])
                polar = torch.eye(3, dtype=torch.float32) * iso
                polar = polar.unsqueeze(0)
            field_source = {"pos": "db", "z": "db", "smile": "db"}
            if dipole is not None:
                field_source["dipole"] = "db"
            if polar is not None:
                field_source["polar"] = "db"
            yield SmilesItem(
                number=int(row["id"]),
                smile=row["SMILES"],
                pos=pos,
                z=z,
                dipole=dipole,
                polar=polar,
                field_source=field_source,
            )
        conn.close()
        return
    raise RuntimeError("Provide --smiles, --smiles-file, --db-path, or --hdf5-path")


def iter_hdf5(cfg: PipelineConfig) -> Iterable[SmilesItem]:
    try:
        import h5py
        import numpy as np
    except Exception as exc:
        raise RuntimeError("h5py is required for --hdf5-path datasets.") from exc
    import gzip
    import shutil
    import tempfile

    def _read_str(group, name: str) -> Optional[str]:
        if name not in group:
            return None
        value = group[name][()]
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, str):
            return value
        if isinstance(value, np.ndarray):
            if value.shape == ():
                return value.item().decode("utf-8") if hasattr(value.item(), "decode") else str(value.item())
            if value.size > 0:
                item = value.reshape(-1)[0]
                return item.decode("utf-8") if hasattr(item, "decode") else str(item)
        return str(value)

    def _as_tensor(value, dtype, *, squeeze_last: bool = False) -> Optional[torch.Tensor]:
        if value is None:
            return None
        arr = np.asarray(value)
        if squeeze_last and arr.ndim > 1 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        return torch.tensor(arr, dtype=dtype)

    def _first_dataset(group, names: Sequence[str]):
        for name in names:
            if name in group:
                return group[name]
        return None

    files: list[Path] = []
    temp_paths: list[Path] = []
    for path in cfg.hdf5_paths:
        if path.is_dir():
            files.extend(sorted(path.glob("*.h5")))
            files.extend(sorted(path.glob("*.hdf5")))
            files.extend(sorted(path.glob("*.h5.gz")))
            files.extend(sorted(path.glob("*.hdf5.gz")))
        else:
            if path.suffix.endswith(".gz"):
                with gzip.open(path, "rb") as f_in:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".hdf5")
                    with tmp:
                        shutil.copyfileobj(f_in, tmp)
                temp_paths.append(Path(tmp.name))
                files.append(Path(tmp.name))
            else:
                files.append(path)

    index = 0
    try:
        for file_path in files:
            with h5py.File(file_path, "r") as handle:
                for mol_key in handle.keys():
                    group = handle[mol_key]
                    subset = _read_str(group, "subset")
                    if cfg.hdf5_subset and (subset is None or subset not in cfg.hdf5_subset):
                        continue

                    smile = _read_str(group, "smiles") or str(mol_key)
                    atomic_numbers = group.get("atomic_numbers")
                    conformations = group.get("conformations")
                    if atomic_numbers is None or conformations is None:
                        continue

                    z = torch.tensor(atomic_numbers[()], dtype=torch.long)
                    energy_ds = _first_dataset(group, ("formation_energy", "dft_total_energy"))
                    dipole_ds = _first_dataset(group, ("scf_dipoles",))
                    quadrupole_ds = _first_dataset(group, ("scf_quadrupole", "scf_quadrupoles"))
                    mbis_charges_ds = _first_dataset(group, ("mbis_charges",))
                    mbis_dipoles_ds = _first_dataset(group, ("mbis_dipoles",))
                    mbis_quadrupoles_ds = _first_dataset(group, ("mbis_quadrupoles",))
                    mbis_octupoles_ds = _first_dataset(group, ("mbis_octupoles",))

                    num_confs = int(conformations.shape[0])
                    base_source = {
                        "pos": "dataset",
                        "z": "dataset",
                        "smile": "dataset",
                    }
                    if energy_ds is not None:
                        base_source["energy"] = energy_ds.name.split("/")[-1]
                    if dipole_ds is not None:
                        base_source["dipole"] = "scf_dipole"
                    elif mbis_dipoles_ds is not None:
                        base_source["dipole"] = "mbis_dipole_sum"
                    if mbis_charges_ds is not None:
                        base_source["npacharge"] = "mbis_charges"
                    if quadrupole_ds is not None:
                        base_source["quadrupole"] = "scf_quadrupole"
                    if mbis_octupoles_ds is not None:
                        base_source["octapole"] = "mbis_octupoles_sum"
                    for conf_id in range(num_confs):
                        index += 1
                        if cfg.limit is not None and index > cfg.limit:
                            return
                        if cfg.distributed and (index - 1) % cfg.world_size != cfg.rank:
                            continue

                        pos = torch.tensor(conformations[conf_id], dtype=torch.float32)
                        energy = _as_tensor(energy_ds[conf_id], torch.float32) if energy_ds is not None else None
                        if energy is not None and energy.ndim == 0:
                            energy = energy.view(1, 1)

                        dipole = _as_tensor(dipole_ds[conf_id], torch.float32) if dipole_ds is not None else None
                        if dipole is None and mbis_dipoles_ds is not None:
                            dipole = _as_tensor(mbis_dipoles_ds[conf_id], torch.float32)
                            if dipole is not None:
                                dipole = dipole.sum(dim=0)
                        if dipole is not None and dipole.ndim == 1:
                            dipole = dipole.view(1, 3)

                        charges = _as_tensor(
                            mbis_charges_ds[conf_id], torch.float32, squeeze_last=True
                        ) if mbis_charges_ds is not None else None
                        quad = _as_tensor(quadrupole_ds[conf_id], torch.float32) if quadrupole_ds is not None else None
                        if quad is not None and quad.ndim == 2:
                            quad = quad.unsqueeze(0)

                        octa = None
                        if mbis_octupoles_ds is not None:
                            octa = _as_tensor(mbis_octupoles_ds[conf_id], torch.float32)
                            if octa is not None:
                                octa = octa.sum(dim=0, keepdim=True)

                        yield SmilesItem(
                            number=index,
                            smile=smile,
                            pos=pos,
                            z=z,
                            energy=energy,
                            dipole=dipole,
                            npacharge=charges,
                            quadrupole=quad,
                            octapole=octa,
                            mol_key=str(mol_key),
                            subset=subset,
                            source=file_path.name,
                            conformer_id=conf_id,
                            field_source=base_source,
                        )
    finally:
        for tmp in temp_paths:
            try:
                tmp.unlink()
            except Exception:
                pass


def smiles_to_conformer(smile: str, max_attempts: int, optimize: bool) -> Tuple[torch.Tensor, torch.Tensor, "Chem.Mol"]:
    ensure_rdkit_available()
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smile}")
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 0xf00d
    success = False
    for attempt in range(max_attempts):
        result = AllChem.EmbedMolecule(mol, params)
        if result == 0:
            success = True
            break
        params.randomSeed += 1
    if not success:
        raise RuntimeError(f"Failed to embed SMILES after {max_attempts} attempts: {smile}")

    if optimize:
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            pass

    conf = mol.GetConformer()
    coords = []
    atomic_numbers = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        pos = conf.GetAtomPosition(idx)
        coords.append([pos.x, pos.y, pos.z])
        atomic_numbers.append(atom.GetAtomicNum())

    pos_t = torch.tensor(coords, dtype=torch.float32)
    z_t = torch.tensor(atomic_numbers, dtype=torch.long)
    return pos_t, z_t, mol


def parse_blob_geometry(blob: Optional[bytes]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if not blob:
        return None, None
    try:
        payload = json.loads(zlib.decompress(blob))
    except Exception:
        return None, None
    atoms = payload.get("atoms")
    coords = payload.get("coord")
    if not atoms or not coords:
        return None, None
    try:
        pos_t = torch.tensor(coords, dtype=torch.float32)
        z_t = torch.tensor(atoms, dtype=torch.long)
    except Exception:
        return None, None
    if pos_t.ndim != 2 or pos_t.shape[1] != 3 or pos_t.shape[0] != z_t.shape[0]:
        return None, None
    return pos_t, z_t


def radius_graph_from_k(
    pos: torch.Tensor,
    k: int,
    clamp_min: float,
    clamp_max: float,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = pos.shape[0]
    if n <= 1:
        empty = torch.empty((2, 0), dtype=torch.long, device=device or pos.device)
        return empty, torch.tensor(0.0, device=device or pos.device, dtype=pos.dtype)

    dists = torch.cdist(pos, pos)
    diag = torch.eye(n, device=dists.device, dtype=torch.bool)
    dists = dists.masked_fill(diag, float("inf"))

    k_eff = min(k, n - 1)
    if k_eff == n - 1:
        idx = torch.arange(n, device=pos.device)
        row = idx.repeat_interleave(n)
        col = idx.repeat(n)
        mask = row != col
        edge_index = torch.stack([col[mask], row[mask]], dim=0)
        radius = dists[dists.isfinite()].max().clamp(min=clamp_min, max=clamp_max)
        return edge_index, radius

    kth = torch.kthvalue(dists, k_eff, dim=1).values
    radius = torch.median(kth).clamp(min=clamp_min, max=clamp_max)

    mask = dists <= radius
    row, col = torch.where(mask)
    edge_index = torch.stack([row, col], dim=0)
    return edge_index, radius


def compute_gasteiger_charges(mol: Optional["Chem.Mol"]) -> torch.Tensor:
    if mol is None or Chem is None or AllChem is None:
        return torch.zeros(0, dtype=torch.float32)
    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        return torch.zeros(mol.GetNumAtoms(), dtype=torch.float32)
    charges = []
    for atom in mol.GetAtoms():
        val = atom.GetProp("_GasteigerCharge") if atom.HasProp("_GasteigerCharge") else "0.0"
        try:
            charges.append(float(val))
        except Exception:
            charges.append(0.0)
    return torch.tensor(charges, dtype=torch.float32)


def approximate_charges_from_z(z: torch.Tensor) -> torch.Tensor:
    en = []
    for zi in z.detach().cpu().tolist():
        val = PAULING_EN.get(int(zi))
        if val is None:
            val = 0.1 * float(zi)
        en.append(val)
    en_t = torch.tensor(en, dtype=torch.float32)
    en_t = en_t - en_t.mean()
    scale = en_t.abs().max().clamp(min=1e-6)
    return en_t / scale * 0.1


def approximate_polar_from_z(z: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    alpha = []
    for zi in z.detach().cpu().tolist():
        val = ATOMIC_POLARIZABILITY.get(int(zi))
        if val is None:
            val = 0.2 * float(zi)
        alpha.append(val)
    alpha_t = torch.tensor(alpha, dtype=dtype, device=device)
    total = alpha_t.sum().clamp(min=1e-6)
    return torch.eye(3, dtype=dtype, device=device) * total


def approximate_depolar_from_z(z: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    alpha = []
    for zi in z.detach().cpu().tolist():
        val = ATOMIC_POLARIZABILITY.get(int(zi))
        if val is None:
            val = 0.2 * float(zi)
        alpha.append(val)
    alpha_t = torch.tensor(alpha, dtype=dtype, device=device)
    n = alpha_t.shape[0]
    depolar = torch.zeros((n, 3, 6), dtype=dtype, device=device)
    scale = alpha_t / alpha_t.mean().clamp(min=1e-6)
    for i in range(n):
        depolar[i, :, 0] = 1e-3 * scale[i]
        depolar[i, :, 1] = 1e-3 * scale[i]
        depolar[i, :, 2] = 1e-3 * scale[i]
    return depolar


def approximate_hyperpolar_from_polar(polar: torch.Tensor) -> torch.Tensor:
    dtype = polar.dtype
    device = polar.device
    hyper = torch.zeros((1, 3, 3, 3), dtype=dtype, device=device)
    beta = polar.mean().clamp(min=1e-6) * 0.1
    for i in range(3):
        hyper[0, i, i, i] = beta
    return hyper


def atomic_masses_from_z(z: torch.Tensor) -> torch.Tensor:
    if atomic_masses is None:
        return z.to(torch.float32)
    masses = [atomic_masses[int(zi)] if int(zi) < len(atomic_masses) else float(zi) for zi in z.tolist()]
    return torch.tensor(masses, dtype=torch.float32)


def build_rdkit_mol(smile: str) -> Optional["Chem.Mol"]:
    if Chem is None:
        return None
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    return Chem.AddHs(mol)


def geometry_from_item(
    item: SmilesItem,
    cfg: PipelineConfig,
) -> Tuple[torch.Tensor, torch.Tensor, Optional["Chem.Mol"]]:
    if item.pos is not None and item.z is not None:
        pos = item.pos
        z = item.z
        mol = build_rdkit_mol(item.smile)
        if mol is not None:
            mol_z = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
            if len(mol_z) != int(z.shape[0]) or mol_z != z.detach().cpu().tolist():
                mol = None
        return pos, z, mol

    pos, z, mol = smiles_to_conformer(item.smile, cfg.rdkit_max_attempts, cfg.rdkit_optimize)
    return pos, z, mol


def center_of_mass(pos: torch.Tensor, masses: torch.Tensor) -> torch.Tensor:
    mass_sum = masses.sum().clamp(min=1e-12)
    return (pos * masses[:, None]).sum(dim=0) / mass_sum


def compute_multipoles(pos: torch.Tensor, charges: torch.Tensor, masses: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    center = center_of_mass(pos, masses)
    centered = pos - center
    q = charges
    quad = (q[:, None, None] * centered[:, :, None] * centered[:, None, :]).sum(dim=0)
    octa = (
        q[:, None, None, None]
        * centered[:, :, None, None]
        * centered[:, None, :, None]
        * centered[:, None, None, :]
    ).sum(dim=0)
    return quad.unsqueeze(0), octa.unsqueeze(0)


def approximate_dipole_from_charges(
    pos: torch.Tensor, charges: torch.Tensor, masses: torch.Tensor
) -> torch.Tensor:
    center = center_of_mass(pos, masses)
    centered = pos - center
    dipole = (charges[:, None] * centered).sum(dim=0)
    return dipole * E_ANGSTROM_TO_DEBYE


def approximate_dedipole_from_charges(
    charges: torch.Tensor, masses: torch.Tensor
) -> torch.Tensor:
    n = charges.shape[0]
    mass_sum = masses.sum().clamp(min=1e-12)
    total_q = charges.sum()
    coeff = charges - total_q * (masses / mass_sum)
    out = torch.zeros((n, 3, 3), device=charges.device, dtype=charges.dtype)
    out[:, 0, 0] = coeff
    out[:, 1, 1] = coeff
    out[:, 2, 2] = coeff
    return out * E_ANGSTROM_TO_DEBYE


def pack_symm(mat: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            mat[..., 0, 0],
            mat[..., 1, 1],
            mat[..., 2, 2],
            mat[..., 0, 1],
            mat[..., 0, 2],
            mat[..., 1, 2],
        ],
        dim=-1,
    )


def finite_diff_dedipole(dipole_fn, pos: torch.Tensor, step: float) -> torch.Tensor:
    n = pos.shape[0]
    out = torch.zeros((n, 3, 3), device=pos.device, dtype=pos.dtype)
    for i in range(n):
        for a in range(3):
            shift = torch.zeros_like(pos)
            shift[i, a] = step
            mu_p = dipole_fn(pos + shift)
            mu_m = dipole_fn(pos - shift)
            out[i, a, :] = (mu_p - mu_m) / (2 * step)
    return out


def finite_diff_depolar(polar_fn, pos: torch.Tensor, pos_step: float) -> torch.Tensor:
    n = pos.shape[0]
    out = torch.zeros((n, 3, 6), device=pos.device, dtype=pos.dtype)
    for i in range(n):
        for a in range(3):
            shift = torch.zeros_like(pos)
            shift[i, a] = pos_step
            polar_p = polar_fn(pos + shift)
            polar_m = polar_fn(pos - shift)
            dpolar = (polar_p - polar_m) / (2 * pos_step)
            out[i, a, :] = pack_symm(dpolar)
    return out


def assemble_hessian(Hi: torch.Tensor, Hij: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    n = Hi.shape[0]
    hess = torch.zeros((n, 3, n, 3), device=Hi.device, dtype=Hi.dtype)
    i, j = edge_index
    hess[j, :, i, :] = Hij
    idx = torch.arange(n, device=Hi.device)
    hess[idx, :, idx, :] = Hi
    return hess


def compute_vibrational_transitions(
    hess: torch.Tensor,
    masses: torch.Tensor,
    charges: Optional[torch.Tensor] = None,
    *,
    dedipole: Optional[torch.Tensor] = None,
    max_modes: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = masses.shape[0]
    hess_flat = hess.reshape(n * 3, n * 3)
    mass_vec = masses.repeat_interleave(3)
    inv_sqrt = (mass_vec + 1e-12).rsqrt()
    mw = inv_sqrt[:, None] * hess_flat * inv_sqrt[None, :]

    evals, evecs = torch.linalg.eigh(mw)
    evals = evals.clamp(min=0)
    freq = torch.sqrt(evals)

    # drop near-zero modes (translations/rotations)
    mask = freq > 1e-6
    freq = freq[mask]
    evecs = evecs[:, mask]

    if freq.numel() == 0:
        tran_energy = torch.zeros((1, max_modes), dtype=hess.dtype, device=hess.device)
        tran_dipole = torch.zeros((1, max_modes, 3), dtype=hess.dtype, device=hess.device)
        return tran_energy, tran_dipole

    num = min(max_modes, freq.shape[0])
    freq_sel = freq[:num]
    evecs_sel = evecs[:, :num]

    tran_energy = torch.zeros((1, max_modes), dtype=hess.dtype, device=hess.device)
    tran_energy[0, :num] = freq_sel

    tran_dipole = torch.zeros((1, max_modes, 3), dtype=hess.dtype, device=hess.device)
    for idx in range(num):
        mode = evecs_sel[:, idx]
        disp = (mode * inv_sqrt).view(n, 3)
        if dedipole is not None:
            dip = (dedipole * disp[..., None]).sum(dim=(0, 1))
        else:
            if charges is None:
                dip = torch.zeros(3, device=hess.device, dtype=hess.dtype)
            else:
                dip = (charges[:, None] * disp).sum(dim=0)
        tran_dipole[0, idx, :] = dip

    return tran_energy, tran_dipole


def build_data_from_smiles(
    item: SmilesItem,
    cfg: PipelineConfig,
    dipole_backend,
    polar_backend,
    hessian_backend,
    psi4_backend,
) -> Data:
    ensure_data_available()
    pos, z, mol = geometry_from_item(item, cfg)
    pos = pos.to(cfg.device)
    z = z.to(cfg.device)
    source_map: Dict[str, str] = dict(item.field_source) if item.field_source else {}

    if item.edge_index is not None:
        edge_index = item.edge_index.to(cfg.device)
        source_map.setdefault("edge_index", "provided")
    else:
        edge_index, _ = radius_graph_from_k(
            pos,
            k=cfg.graph_k,
            clamp_min=cfg.graph_clamp_min,
            clamp_max=cfg.graph_clamp_max,
            device=cfg.device,
        )
        source_map.setdefault("edge_index", "radius_graph")

    source_map.setdefault("pos", "provided" if item.pos is not None else "rdkit")
    source_map.setdefault("z", "provided" if item.z is not None else "rdkit")
    source_map.setdefault("smile", "input")
    source_map.setdefault("number", "input")

    dipole_supported = dipole_backend is not None and dipole_backend.supports(z)
    polar_supported = polar_backend is not None and polar_backend.supports(z)

    if item.npacharge is not None:
        charges = item.npacharge
        source_map.setdefault("npacharge", "dataset")
    else:
        charges = compute_gasteiger_charges(mol)
        if charges is None or charges.numel() == 0:
            charges = approximate_charges_from_z(z).to(dtype=pos.dtype)
            source_map.setdefault("npacharge", "charge_approx")
        else:
            source_map.setdefault("npacharge", "gasteiger")
    if charges is not None and charges.ndim > 1:
        charges = charges.squeeze()
    if charges is None or charges.ndim != 1 or charges.shape[0] != z.shape[0]:
        charges = approximate_charges_from_z(z).to(dtype=pos.dtype)
        source_map["npacharge"] = "charge_approx"
    charges = charges.to(cfg.device, dtype=pos.dtype)
    masses = atomic_masses_from_z(z).to(cfg.device, dtype=pos.dtype)

    dipole = item.dipole.to(cfg.device, dtype=pos.dtype) if item.dipole is not None else None
    polar = item.polar.to(cfg.device, dtype=pos.dtype) if item.polar is not None else None
    hyperpolar = item.hyperpolar.to(cfg.device, dtype=pos.dtype) if item.hyperpolar is not None else None
    dedipole = item.dedipole.to(cfg.device, dtype=pos.dtype) if item.dedipole is not None else None
    depolar = item.depolar.to(cfg.device, dtype=pos.dtype) if item.depolar is not None else None

    if dipole is not None and dipole.ndim == 1:
        dipole = dipole.view(1, 3)
    if polar is not None and polar.ndim == 2:
        polar = polar.unsqueeze(0)
    if hyperpolar is not None and hyperpolar.ndim == 3:
        hyperpolar = hyperpolar.unsqueeze(0)

    need_dipole = dipole is None
    need_polar = polar is None
    need_hyper = hyperpolar is None

    if dipole is not None:
        source_map.setdefault("dipole", "dataset")
    if polar is not None:
        source_map.setdefault("polar", "dataset")
    if hyperpolar is not None:
        source_map.setdefault("hyperpolar", "dataset")
    if dedipole is not None:
        source_map.setdefault("dedipole", "dataset")
    if depolar is not None:
        source_map.setdefault("depolar", "dataset")

    if need_dipole and cfg.allow_missing_dipole and not dipole_supported:
        dipole = approximate_dipole_from_charges(pos, charges, masses).reshape(1, 3)
        if dedipole is None:
            dedipole = approximate_dedipole_from_charges(charges, masses)
        source_map.setdefault("dipole", "charge_approx")
        source_map.setdefault("dedipole", "charge_approx")
        need_dipole = False

    if need_polar and cfg.allow_missing_polar and not polar_supported:
        polar = approximate_polar_from_z(z, pos.dtype, cfg.device).unsqueeze(0)
        if depolar is None:
            depolar = approximate_depolar_from_z(z, pos.dtype, cfg.device)
        source_map.setdefault("polar", "polar_approx")
        source_map.setdefault("depolar", "polar_approx")
        need_polar = False

    if need_hyper and cfg.allow_missing_hyperpolar:
        if polar is None:
            polar = approximate_polar_from_z(z, pos.dtype, cfg.device).unsqueeze(0)
            source_map.setdefault("polar", "polar_approx")
        hyperpolar = approximate_hyperpolar_from_polar(polar)
        source_map.setdefault("hyperpolar", "hyperpolar_approx")
        need_hyper = False

    use_dft_props = False
    if (need_dipole or need_polar or need_hyper) and psi4_backend is not None:
        use_dft_props = pos.shape[0] <= cfg.dft_atom_cutoff or not (dipole_supported or polar_supported)
    if use_dft_props:
        try:
            dft_dipole, dft_polar, dft_hyper = psi4_backend.properties(
                pos,
                z,
                allow_missing_hyper=cfg.allow_missing_hyperpolar,
            )
            if dipole is None:
                dipole = dft_dipole.to(device=cfg.device, dtype=pos.dtype).reshape(1, 3)
                source_map.setdefault("dipole", "psi4")
            if polar is None:
                polar = dft_polar.to(device=cfg.device, dtype=pos.dtype).unsqueeze(0)
                source_map.setdefault("polar", "psi4")
            if hyperpolar is None:
                hyperpolar = dft_hyper.to(device=cfg.device, dtype=pos.dtype)
                source_map.setdefault("hyperpolar", "psi4")
            if dedipole is None:
                dedipole = finite_diff_dedipole(lambda coords: psi4_backend.dipole(coords, z), pos, cfg.pos_step)
                source_map.setdefault("dedipole", "derived:psi4")
            if depolar is None:
                depolar = finite_diff_depolar(lambda coords: psi4_backend.polar(coords, z), pos, cfg.pos_step)
                source_map.setdefault("depolar", "derived:psi4")
            need_dipole = dipole is None
            need_polar = polar is None
            need_hyper = hyperpolar is None
        except Exception:
            if (need_dipole and not dipole_supported and not cfg.allow_missing_dipole) or (
                need_polar and not polar_supported and not cfg.allow_missing_polar
            ):
                raise
            use_dft_props = False

    if dipole is None:
        if dipole_supported:
            dipole = dipole_backend.dipole(pos, z).reshape(1, 3)
            if dedipole is None:
                dedipole = finite_diff_dedipole(lambda coords: dipole_backend.dipole(coords, z), pos, cfg.pos_step)
            source_map.setdefault("dipole", "deepmd_dipole")
            source_map.setdefault("dedipole", "derived:deepmd_dipole")
        elif cfg.allow_missing_dipole:
            dipole = approximate_dipole_from_charges(pos, charges, masses).reshape(1, 3)
            if dedipole is None:
                dedipole = approximate_dedipole_from_charges(charges, masses)
            source_map.setdefault("dipole", "charge_approx")
            source_map.setdefault("dedipole", "charge_approx")
        else:
            raise RuntimeError("DeepMD dipole model is missing or does not support this element set.")

    if polar is None:
        if polar_supported:
            polar = polar_backend.polar(pos, z).unsqueeze(0)
            if depolar is None:
                depolar = finite_diff_depolar(lambda coords: polar_backend.polar(coords, z), pos, cfg.pos_step)
            source_map.setdefault("polar", "deepmd_polar")
            source_map.setdefault("depolar", "derived:deepmd_polar")
        elif cfg.allow_missing_polar:
            polar = approximate_polar_from_z(z, pos.dtype, cfg.device).unsqueeze(0)
            if depolar is None:
                depolar = approximate_depolar_from_z(z, pos.dtype, cfg.device)
            source_map.setdefault("polar", "polar_approx")
            source_map.setdefault("depolar", "polar_approx")
        else:
            raise RuntimeError("Polar model is required for molecules above the DFT atom cutoff.")

    if hyperpolar is None:
        if polar is None:
            polar = approximate_polar_from_z(z, pos.dtype, cfg.device).unsqueeze(0)
            source_map.setdefault("polar", "polar_approx")
        hyperpolar = approximate_hyperpolar_from_polar(polar)
        source_map.setdefault("hyperpolar", "hyperpolar_approx")

    if dedipole is None:
        dedipole = approximate_dedipole_from_charges(charges, masses)
        source_map.setdefault("dedipole", "charge_approx")
    if depolar is None:
        depolar = approximate_depolar_from_z(z, pos.dtype, cfg.device)
        source_map.setdefault("depolar", "polar_approx")

    hessian_supported = True
    if hessian_backend is not None and hasattr(hessian_backend, "supports"):
        hessian_supported = hessian_backend.supports(z)
    if not hessian_supported and psi4_backend is None:
        raise RuntimeError("Hessian backend does not support this element set; enable Psi4 fallback.")

    use_dft_hess = psi4_backend is not None and (pos.shape[0] <= cfg.dft_atom_cutoff or not hessian_supported)
    try:
        if use_dft_hess:
            Hi, Hij = psi4_backend.compute(pos, z, edge_index)
        else:
            Hi, Hij = hessian_backend.compute(pos, z, edge_index)
    except Exception:
        if hessian_backend is None:
            raise
        Hi, Hij = hessian_backend.compute(pos, z, edge_index)

    if use_dft_hess:
        hess_source = "psi4"
    else:
        hess_source = "deepmd_pot" if isinstance(hessian_backend, DeepMDPotHessianBackend) else "mace"
    source_map.setdefault("Hi", hess_source)
    source_map.setdefault("Hij", hess_source)

    if item.quadrupole is not None:
        quadrupole = item.quadrupole.to(cfg.device, dtype=pos.dtype)
        if quadrupole.ndim == 2:
            quadrupole = quadrupole.unsqueeze(0)
        source_map.setdefault("quadrupole", "dataset")
    if item.octapole is not None:
        octapole = item.octapole.to(cfg.device, dtype=pos.dtype)
        if octapole.ndim == 3:
            octapole = octapole.unsqueeze(0)
        source_map.setdefault("octapole", "dataset")
    if item.quadrupole is None or item.octapole is None:
        quad_calc, octa_calc = compute_multipoles(pos, charges, masses)
        if item.quadrupole is None:
            quadrupole = quad_calc
            source_map.setdefault("quadrupole", "charge_approx")
        if item.octapole is None:
            octapole = octa_calc
            source_map.setdefault("octapole", "charge_approx")

    full_hess = assemble_hessian(Hi, Hij, edge_index)
    tran_energy, tran_dipole = compute_vibrational_transitions(
        hess=full_hess.detach(),
        masses=masses.detach(),
        charges=charges.detach(),
        dedipole=dedipole.detach() if dedipole is not None else None,
        max_modes=10,
    )
    source_map.setdefault("tran_energy", f"derived:{hess_source}")
    source_map.setdefault("tran_dipole", f"derived:{hess_source}")

    if item.energy is not None:
        energy = item.energy.to(cfg.device, dtype=pos.dtype)
        source_map.setdefault("energy", "dataset")
    elif use_dft_hess and psi4_backend is not None and hasattr(psi4_backend, "energy"):
        energy = psi4_backend.energy(pos, z)
        source_map.setdefault("energy", "psi4")
    elif hasattr(hessian_backend, "energy"):
        energy = hessian_backend.energy(pos, z)
        if isinstance(hessian_backend, DeepMDPotHessianBackend):
            source_map.setdefault("energy", "deepmd_pot")
        else:
            source_map.setdefault("energy", "mace")
    else:
        energy = torch.zeros((1, 1), dtype=pos.dtype, device=cfg.device)
        source_map.setdefault("energy", "zero_fill")

    atomic_energy = None
    if (not use_dft_hess) and cfg.deepmd_atomic_energy and hasattr(hessian_backend, "atomic_energy"):
        try:
            atomic_energy = hessian_backend.atomic_energy(pos, z)
            source_map.setdefault("atomic_energy", "deepmd_pot")
        except Exception:
            atomic_energy = None

    dataset_sources = {
        "dataset",
        "formation_energy",
        "dft_total_energy",
        "scf_dipole",
        "scf_quadrupole",
        "mbis_charges",
        "mbis_dipole_sum",
        "mbis_octupoles_sum",
    }
    input_sources = {"input"}
    provided_sources = {"provided", "db"}
    ml_sources = {"mace", "deepmd_pot", "deepmd_dipole", "deepmd_polar"}
    heuristic_sources = {"charge_approx", "gasteiger", "zero_fill"}

    def is_generated(source: str) -> bool:
        return source not in dataset_sources and source not in input_sources and source not in provided_sources

    def is_imputed(source: str) -> bool:
        if source in ml_sources or source in heuristic_sources:
            return True
        if source.startswith("derived:"):
            base = source.split(":", 1)[1]
            return base in ml_sources or base in heuristic_sources
        return False

    confidence_base = {
        "dataset": 0.99,
        "db": 0.99,
        "summary_csv": 0.98,
        "des5m_energy": 0.98,
        "formation_energy": 0.99,
        "dft_total_energy": 0.99,
        "scf_dipole": 0.99,
        "scf_quadrupole": 0.99,
        "mbis_charges": 0.98,
        "mbis_dipole_sum": 0.98,
        "mbis_octupoles_sum": 0.98,
        "psi4": 0.95,
        "deepmd_pot": 0.85,
        "deepmd_dipole": 0.82,
        "deepmd_polar": 0.80,
        "mace": 0.80,
        "radius_graph": 0.85,
        "rdkit": 0.75,
        "provided": 0.9,
        "input": 0.9,
        "gasteiger": 0.45,
        "charge_approx": 0.4,
        "polar_approx": 0.35,
        "hyperpolar_approx": 0.3,
        "zero_fill": 0.05,
    }

    def confidence_for_source(source: str) -> float:
        base = source
        if source.startswith("derived:"):
            base = source.split(":", 1)[1]
        value = confidence_base.get(base, 0.5)
        if source.startswith("derived:"):
            value = max(0.0, value - 0.05)
        return value

    field_generated = {key: is_generated(value) for key, value in source_map.items()}
    field_imputed = {key: is_imputed(value) for key, value in source_map.items()}
    field_confidence = {key: confidence_for_source(value) for key, value in source_map.items()}

    data_fields = dict(
        edge_index=edge_index,
        pos=pos,
        z=z,
        number=item.number,
        smile=item.smile,
        energy=energy.reshape(1, 1),
        dipole=dipole,
        npacharge=charges,
        polar=polar,
        quadrupole=quadrupole,
        octapole=octapole,
        hyperpolar=hyperpolar,
        Hi=Hi,
        Hij=Hij,
        dedipole=dedipole,
        depolar=depolar,
        tran_dipole=tran_dipole,
        tran_energy=tran_energy,
        field_source=source_map,
        field_generated=field_generated,
        field_imputed=field_imputed,
        field_confidence=field_confidence,
    )
    if item.mol_key is not None:
        data_fields["mol_key"] = item.mol_key
    if item.subset is not None:
        data_fields["subset"] = item.subset
    if item.source is not None:
        data_fields["source"] = item.source
    if item.conformer_id is not None:
        data_fields["conformer_id"] = item.conformer_id
    if atomic_energy is not None:
        data_fields["atomic_energy"] = atomic_energy
    data = Data(**data_fields)

    return data


def write_data(data: Data, cfg: PipelineConfig, manifest) -> None:
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{data.number}.pt"
    if cfg.save_device is not None:
        data = data.to(cfg.save_device)
    torch.save(data, out_path)
    manifest.write(json.dumps({"id": data.number, "path": str(out_path), "smile": data.smile}) + "\n")


class ShardWriter:
    def __init__(self, cfg: PipelineConfig, manifest) -> None:
        if cfg.shard_size is None or cfg.shard_size <= 0:
            raise ValueError("shard_size must be a positive integer.")
        self.cfg = cfg
        self.manifest = manifest
        self.buffer: list[Data] = []
        self.shard_idx = 0

    def add(self, data: Data) -> None:
        if self.cfg.save_device is not None:
            data = data.to(self.cfg.save_device)
        self.buffer.append(data)
        if len(self.buffer) >= self.cfg.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        out_dir = self.cfg.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"shard_{self.shard_idx:06d}.pt"
        torch.save(self.buffer, out_path)
        start_id = getattr(self.buffer[0], "number", None)
        end_id = getattr(self.buffer[-1], "number", None)
        self.manifest.write(
            json.dumps(
                {
                    "shard": str(out_path),
                    "count": len(self.buffer),
                    "start_id": start_id,
                    "end_id": end_id,
                }
            )
            + "\n"
        )
        self.buffer.clear()
        self.shard_idx += 1


def run_pipeline(cfg: PipelineConfig, items_iter: Iterable[SmilesItem], total: Optional[int] = None) -> None:
    os.environ.setdefault("CACHED_PATH_CACHE_ROOT", str(cfg.checkpoint_cache / "cache"))
    mace_path = resolve_checkpoint(str(cfg.mace_model), cfg.checkpoint_cache)
    deepmd_pot_path = resolve_checkpoint(str(cfg.deepmd_pot_model), cfg.checkpoint_cache) if cfg.deepmd_pot_model else None

    dipole_backend = None
    polar_backend = None
    if cfg.dipole_model is not None:
        dipole_backend = DeepMDDipoleBackend(
            device=cfg.device,
            model_path=str(cfg.dipole_model),
            type_map=cfg.deepmd_type_map,
            dipole_unit=cfg.deepmd_dipole_unit,
        )
    if cfg.polar_model is not None:
        polar_backend = DeepMDPolarBackend(
            device=cfg.device,
            model_path=str(cfg.polar_model),
            type_map=cfg.deepmd_type_map,
        )

    if deepmd_pot_path is not None:
        hessian_backend = DeepMDPotHessianBackend(
            device=cfg.device,
            model_path=deepmd_pot_path,
            type_map=cfg.deepmd_type_map,
            head=cfg.deepmd_head,
            step=cfg.pos_step,
        )
    else:
        hessian_backend = MaceHessianBackend(
            device=cfg.device,
            model_path=mace_path,
            use_autograd=True,
        )

    psi4_backend = None
    if cfg.psi4_fallback:
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

    manifest_path = cfg.output_dir / "manifest.jsonl"
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as manifest:
        shard_writer = ShardWriter(cfg, manifest) if cfg.shard_size else None
        processed = 0
        iterator = items_iter
        if total is not None:
            try:
                from tqdm import tqdm

                iterator = tqdm(items_iter, total=total, desc="generate")
            except Exception:
                iterator = items_iter
        for item in iterator:
            processed += 1
            data = build_data_from_smiles(item, cfg, dipole_backend, polar_backend, hessian_backend, psi4_backend)
            if shard_writer is not None:
                shard_writer.add(data)
            else:
                write_data(data, cfg, manifest)
            if cfg.log_every and processed % cfg.log_every == 0 and total is None:
                print(f"processed {processed}")
        if shard_writer is not None:
            shard_writer.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Data objects from SMILES.")
    parser.add_argument("--output-dir", required=True, help="Output directory for .pt files.")
    parser.add_argument("--smiles", action="append", default=[], help="SMILES string (repeatable).")
    parser.add_argument("--smiles-file", type=str, default=None, help="File with SMILES strings (one per line).")
    parser.add_argument("--db-path", type=str, default=None, help="SQLite DB containing a molecule table.")
    parser.add_argument(
        "--hdf5-path",
        action="append",
        default=[],
        help="Path to HDF5 dataset file or directory (repeatable).",
    )
    parser.add_argument(
        "--hdf5-subset",
        action="append",
        default=[],
        help="Subset filter for HDF5 datasets (repeatable).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of molecules.")
    parser.add_argument("--device", type=str, default=None, help="Device for model execution (cuda/cpu).")
    parser.add_argument("--save-device", type=str, default=None, help="Device for saved tensors (cpu recommended).")
    parser.add_argument("--deepmd-dipole-model", type=str, default=None, help="DeepMD dipole model path.")
    parser.add_argument("--deepmd-polar-model", type=str, default=None, help="DeepMD polar model path.")
    parser.add_argument("--deepmd-pot-model", type=str, default=None, help="DeepMD potential model path.")
    parser.add_argument(
        "--deepmd-head",
        type=str,
        default=None,
        help="DeepMD head/branch name for multi-head models.",
    )
    parser.add_argument(
        "--deepmd-branch",
        type=str,
        default=None,
        help="Alias for --deepmd-head.",
    )
    parser.add_argument(
        "--deepmd-type-map",
        type=str,
        default=None,
        help="Comma-separated element symbols for DeepMD type map (e.g. O,H,C,N).",
    )
    parser.add_argument(
        "--deepmd-dipole-unit",
        type=str,
        default="au",
        choices=("au", "debye"),
        help="Units for DeepMD dipole output.",
    )
    parser.add_argument(
        "--deepmd-atomic-energy",
        action="store_true",
        help="Store DeepMD per-atom energy contributions when available.",
    )
    parser.add_argument("--pos-step", type=float, default=1e-3, help="Position step for depolar finite differences.")
    parser.add_argument("--mace-model", type=str, default=None, help="MACE checkpoint path or URL.")
    parser.add_argument(
        "--dft-atom-cutoff",
        type=int,
        default=20,
        help="Use Psi4 for molecules with atom count <= cutoff.",
    )
    parser.add_argument("--graph-k", type=int, default=20, help="Target k-th neighbor count for radius graph.")
    parser.add_argument("--graph-clamp-min", type=float, default=0.5, help="Minimum radius clamp (Angstrom).")
    parser.add_argument("--graph-clamp-max", type=float, default=100.0, help="Maximum radius clamp (Angstrom).")
    parser.add_argument(
        "--no-psi4",
        action="store_true",
        help="Disable Psi4 DFT path even for small molecules.",
    )
    parser.add_argument("--psi4-method", type=str, default="B3LYP")
    parser.add_argument("--psi4-basis", type=str, default="cc-pVTZ")
    parser.add_argument("--psi4-memory", type=str, default="2 GB")
    parser.add_argument("--psi4-threads", type=int, default=1)
    parser.add_argument("--psi4-scf-type", type=str, default="df")
    parser.add_argument("--psi4-guess", type=str, default="sad")
    parser.add_argument("--psi4-charge", type=int, default=0)
    parser.add_argument("--psi4-multiplicity", type=int, default=1)
    parser.add_argument("--psi4-quiet", action="store_true")
    parser.add_argument(
        "--allow-missing-hyperpolar",
        action="store_true",
        help="Allow missing DFT hyperpolarizability (fills approximations).",
    )
    parser.add_argument(
        "--allow-missing-polar",
        action="store_true",
        help="Allow missing polarizability for ML branch (fills approximations).",
    )
    parser.add_argument(
        "--allow-missing-dipole",
        action="store_true",
        help="Allow missing dipole for ML branch (uses charge-based estimate).",
    )
    parser.add_argument("--rdkit-max-attempts", type=int, default=10)
    parser.add_argument("--rdkit-no-opt", action="store_true", help="Disable RDKit geometry optimization.")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--shard-size",
        type=int,
        default=None,
        help="Write outputs into shard_*.pt files with this many items each.",
    )

    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_device = torch.device(args.save_device) if args.save_device else None

    cfg = PipelineConfig(
        output_dir=Path(args.output_dir),
        device=device,
        limit=args.limit,
        log_every=args.log_every,
        distributed=args.distributed,
        rank=int(os.environ.get("LOCAL_RANK", "0")),
        world_size=int(os.environ.get("WORLD_SIZE", "1")),
        save_device=save_device,
        smiles=args.smiles,
        smiles_file=Path(args.smiles_file) if args.smiles_file else None,
        db_path=Path(args.db_path) if args.db_path else None,
        hdf5_paths=[Path(path) for path in args.hdf5_path],
        hdf5_subset=args.hdf5_subset,
        pos_step=args.pos_step,
        dft_atom_cutoff=args.dft_atom_cutoff,
        graph_k=args.graph_k,
        graph_clamp_min=args.graph_clamp_min,
        graph_clamp_max=args.graph_clamp_max,
        dipole_model=Path(args.deepmd_dipole_model) if args.deepmd_dipole_model else None,
        polar_model=Path(args.deepmd_polar_model) if args.deepmd_polar_model else None,
        deepmd_pot_model=Path(args.deepmd_pot_model) if args.deepmd_pot_model else None,
        deepmd_head=args.deepmd_head or args.deepmd_branch,
        deepmd_type_map=args.deepmd_type_map,
        deepmd_dipole_unit=args.deepmd_dipole_unit,
        deepmd_atomic_energy=args.deepmd_atomic_energy,
        mace_model=Path(args.mace_model)
        if args.mace_model
        else Path("data-gen-pipeline/checkpoints/2024-07-12-mace-128-L1_epoch-199.model"),
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

    if cfg.hdf5_paths:
        items_iter = iter_smiles(cfg)
        total = None
    else:
        items = list(iter_smiles(cfg))
        if cfg.distributed:
            items = items[cfg.rank :: cfg.world_size]
        if cfg.limit is not None:
            items = items[: cfg.limit]
        items_iter = iter(items)
        total = len(items)

    run_pipeline(cfg, items_iter, total)


if __name__ == "__main__":
    main()
