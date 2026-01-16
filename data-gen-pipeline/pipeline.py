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
from typing import Iterable, Optional, Sequence, Tuple

import torch

E_ANGSTROM_TO_DEBYE = 4.80320427

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


@dataclass
class SmilesItem:
    number: int
    smile: str
    pos: Optional[torch.Tensor] = None
    z: Optional[torch.Tensor] = None


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
        cur.execute("SELECT id, SMILES, blob_data FROM molecule ORDER BY id")
        for row in cur:
            pos, z = parse_blob_geometry(row["blob_data"])
            yield SmilesItem(number=int(row["id"]), smile=row["SMILES"], pos=pos, z=z)
        conn.close()
        return
    raise RuntimeError("Provide --smiles, --smiles-file, or --db-path")


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
        edge_index = torch.stack([row[mask], col[mask]], dim=0)
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

    edge_index, _ = radius_graph_from_k(
        pos,
        k=cfg.graph_k,
        clamp_min=cfg.graph_clamp_min,
        clamp_max=cfg.graph_clamp_max,
        device=cfg.device,
    )

    deepmd_supported = dipole_backend is not None and dipole_backend.supports(z)
    if polar_backend is not None:
        deepmd_supported = deepmd_supported and polar_backend.supports(z)
    dipole_supported = dipole_backend is not None and dipole_backend.supports(z)
    polar_supported = polar_backend is not None and polar_backend.supports(z)
    deepmd_supported = (dipole_supported or cfg.allow_missing_dipole) and (
        polar_supported or cfg.allow_missing_polar
    )

    hessian_supported = True
    if hessian_backend is not None and hasattr(hessian_backend, "supports"):
        hessian_supported = hessian_backend.supports(z)
    if not hessian_supported and psi4_backend is None:
        raise RuntimeError("Hessian backend does not support this element set; enable Psi4 fallback.")

    use_dft = psi4_backend is not None and (
        pos.shape[0] <= cfg.dft_atom_cutoff or not deepmd_supported or not hessian_supported
    )

    charges = compute_gasteiger_charges(mol)
    if charges.numel() != z.shape[0]:
        charges = torch.zeros(z.shape[0], dtype=pos.dtype)
    charges = charges.to(cfg.device, dtype=pos.dtype)
    masses = atomic_masses_from_z(z).to(cfg.device, dtype=pos.dtype)

    dipole = None
    polar = None
    hyperpolar = None
    if use_dft:
        try:
            dipole, polar, hyperpolar = psi4_backend.properties(
                pos,
                z,
                allow_missing_hyper=cfg.allow_missing_hyperpolar,
            )
            dipole = dipole.to(device=cfg.device, dtype=pos.dtype).reshape(1, 3)
            polar = polar.to(device=cfg.device, dtype=pos.dtype).unsqueeze(0)
            hyperpolar = hyperpolar.to(device=cfg.device, dtype=pos.dtype)
            dedipole = finite_diff_dedipole(lambda coords: psi4_backend.dipole(coords, z), pos, cfg.pos_step)
            depolar = finite_diff_depolar(lambda coords: psi4_backend.polar(coords, z), pos, cfg.pos_step)
        except Exception:
            if not deepmd_supported:
                raise
            use_dft = False

    if not use_dft:
        if not dipole_supported:
            if cfg.allow_missing_dipole:
                dipole = approximate_dipole_from_charges(pos, charges, masses).reshape(1, 3)
                dedipole = approximate_dedipole_from_charges(charges, masses)
            else:
                raise RuntimeError("DeepMD dipole model is missing or does not support this element set.")
        else:
            dipole = dipole_backend.dipole(pos, z).reshape(1, 3)
            dedipole = finite_diff_dedipole(lambda coords: dipole_backend.dipole(coords, z), pos, cfg.pos_step)

        if polar_backend is None or not polar_supported:
            if cfg.allow_missing_polar:
                polar = torch.zeros((1, 3, 3), device=cfg.device, dtype=pos.dtype)
                depolar = torch.zeros((pos.shape[0], 3, 6), device=cfg.device, dtype=pos.dtype)
            else:
                raise RuntimeError("Polar model is required for molecules above the DFT atom cutoff.")
        else:
            polar = polar_backend.polar(pos, z).unsqueeze(0)
            depolar = finite_diff_depolar(lambda coords: polar_backend.polar(coords, z), pos, cfg.pos_step)

        hyperpolar = torch.zeros((1, 3, 3, 3), device=cfg.device, dtype=pos.dtype)

    try:
        if use_dft:
            Hi, Hij = psi4_backend.compute(pos, z, edge_index)
        else:
            Hi, Hij = hessian_backend.compute(pos, z, edge_index)
    except Exception:
        if hessian_backend is None:
            raise
        Hi, Hij = hessian_backend.compute(pos, z, edge_index)

    quadrupole, octapole = compute_multipoles(pos, charges, masses)

    full_hess = assemble_hessian(Hi, Hij, edge_index)
    tran_energy, tran_dipole = compute_vibrational_transitions(
        hess=full_hess.detach(),
        masses=masses.detach(),
        charges=charges.detach(),
        dedipole=dedipole.detach() if dedipole is not None else None,
        max_modes=10,
    )

    if use_dft and psi4_backend is not None and hasattr(psi4_backend, "energy"):
        energy = psi4_backend.energy(pos, z)
    elif hasattr(hessian_backend, "energy"):
        energy = hessian_backend.energy(pos, z)
    else:
        energy = torch.zeros((1, 1), dtype=pos.dtype, device=cfg.device)

    atomic_energy = None
    if not use_dft and cfg.deepmd_atomic_energy and hasattr(hessian_backend, "atomic_energy"):
        try:
            atomic_energy = hessian_backend.atomic_energy(pos, z)
        except Exception:
            atomic_energy = None

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
    )
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Data objects from SMILES.")
    parser.add_argument("--output-dir", required=True, help="Output directory for .pt files.")
    parser.add_argument("--smiles", action="append", default=[], help="SMILES string (repeatable).")
    parser.add_argument("--smiles-file", type=str, default=None, help="File with SMILES strings (one per line).")
    parser.add_argument("--db-path", type=str, default=None, help="SQLite DB containing a molecule table.")
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
        help="Allow missing DFT hyperpolarizability (fills zeros).",
    )
    parser.add_argument(
        "--allow-missing-polar",
        action="store_true",
        help="Allow missing polarizability for ML branch (fills zeros).",
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
    )

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

    items = list(iter_smiles(cfg))
    if cfg.distributed:
        items = items[cfg.rank :: cfg.world_size]
    if cfg.limit is not None:
        items = items[: cfg.limit]

    manifest_path = cfg.output_dir / "manifest.jsonl"
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for idx, item in enumerate(items, start=1):
            data = build_data_from_smiles(item, cfg, dipole_backend, polar_backend, hessian_backend, psi4_backend)
            write_data(data, cfg, manifest)
            if idx % cfg.log_every == 0:
                print(f"processed {idx}/{len(items)}")


if __name__ == "__main__":
    main()
