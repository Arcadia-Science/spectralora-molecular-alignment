from __future__ import annotations

import os
from typing import Iterable, Optional

import numpy as np
import torch

try:
    from .backends import HessianBackend
except ImportError:  # pragma: no cover - fallback for script-style imports
    from backends import HessianBackend

ATOMIC_SYMBOLS = [
    "X",
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
]

HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.52917721092
DIPOLE_AU_TO_DEBYE = 2.541746
HESSIAN_AU_TO_EV_A2 = HARTREE_TO_EV / (BOHR_TO_ANGSTROM**2)


class Psi4HessianBackend(HessianBackend):
    """Compute Hessian blocks with Psi4 on CPU."""

    def __init__(
        self,
        device: torch.device,
        method: str = "B3LYP",
        basis: str = "6-31G*",
        charge: int = 0,
        multiplicity: int = 1,
        num_threads: int = 1,
        memory: str = "2 GB",
        scf_type: str = "df",
        guess: str = "sad",
        allowed_z: Optional[Iterable[int]] = None,
        on_unsupported: str = "error",
        quiet: bool = True,
    ) -> None:
        try:
            import psi4
        except Exception as exc:
            raise ImportError("psi4 is required for Psi4HessianBackend") from exc

        self.psi4 = psi4
        self.device = device
        self.method = method
        self.basis = basis
        self.charge = charge
        self.multiplicity = multiplicity
        self.allowed_z = set(allowed_z) if allowed_z else None
        self.on_unsupported = on_unsupported

        psi4.set_memory(memory)
        psi4.set_num_threads(num_threads)
        if quiet:
            psi4.core.set_output_file(os.devnull, False)
        psi4.set_options({"scf_type": scf_type, "guess": guess})

    def supports(self, z: torch.Tensor) -> bool:
        if self.allowed_z is None:
            return True
        return bool(set(z.detach().cpu().tolist()) <= self.allowed_z)

    def _symbol(self, atomic_number: int) -> str:
        if 0 < atomic_number < len(ATOMIC_SYMBOLS):
            return ATOMIC_SYMBOLS[atomic_number]
        raise ValueError(f"Atomic number {atomic_number} out of range")

    def _build_molecule(self, pos: torch.Tensor, z: torch.Tensor):
        pos_cpu = pos.detach().cpu().double().numpy()
        z_cpu = z.detach().cpu().tolist()
        lines = [f"{self._symbol(int(zi))} {x:.10f} {y:.10f} {zv:.10f}" for zi, (x, y, zv) in zip(z_cpu, pos_cpu)]
        geom = ["molecule {", f"{self.charge} {self.multiplicity}"]
        geom.extend(lines)
        geom.append("units angstrom")
        geom.append("}")
        return self.psi4.geometry("\n".join(geom))

    def _get_variable(self, wfn, name: str):
        for getter in (
            lambda: wfn.variable(name) if wfn is not None else None,
            lambda: self.psi4.core.get_variable(name),
            lambda: self.psi4.core.variable(name),
        ):
            try:
                val = getter()
            except Exception:
                val = None
            if val is not None:
                return val
        return None

    def _extract_vector(self, wfn, vector_keys, component_keys):
        for key in vector_keys:
            val = self._get_variable(wfn, key)
            if val is None:
                continue
            tensor = torch.tensor(val, dtype=torch.float64).reshape(-1)
            if tensor.numel() == 3:
                return tensor

        comps = []
        for keys in component_keys:
            comp = None
            for key in keys:
                comp = self._get_variable(wfn, key)
                if comp is not None:
                    break
            if comp is None:
                return None
            comps.append(comp)
        return torch.tensor(comps, dtype=torch.float64)

    def _extract_polar(self, wfn):
        for key in ("SCF POLARIZABILITY", "POLARIZABILITY"):
            val = self._get_variable(wfn, key)
            if val is None:
                continue
            tensor = torch.tensor(val, dtype=torch.float64).reshape(-1)
            if tensor.numel() == 9:
                return tensor.view(3, 3)
            if tensor.numel() == 6:
                xx, yy, zz, xy, xz, yz = tensor
                mat = torch.tensor(
                    [
                        [xx, xy, xz],
                        [xy, yy, yz],
                        [xz, yz, zz],
                    ],
                    dtype=torch.float64,
                )
                return mat

        components = {
            (0, 0): ("SCF POLARIZABILITY XX", "POLARIZABILITY XX"),
            (0, 1): ("SCF POLARIZABILITY XY", "POLARIZABILITY XY"),
            (0, 2): ("SCF POLARIZABILITY XZ", "POLARIZABILITY XZ"),
            (1, 1): ("SCF POLARIZABILITY YY", "POLARIZABILITY YY"),
            (1, 2): ("SCF POLARIZABILITY YZ", "POLARIZABILITY YZ"),
            (2, 2): ("SCF POLARIZABILITY ZZ", "POLARIZABILITY ZZ"),
        }
        mat = torch.zeros((3, 3), dtype=torch.float64)
        for (i, j), keys in components.items():
            val = None
            for key in keys:
                val = self._get_variable(wfn, key)
                if val is not None:
                    break
            if val is None:
                return None
            mat[i, j] = float(val)
            mat[j, i] = float(val) if i != j else mat[j, i]
        return mat

    def _fill_hyper_sym(self, vals: torch.Tensor) -> torch.Tensor:
        beta = torch.zeros((3, 3, 3), dtype=torch.float64)

        def fill(indices, value):
            for i, j, k in indices:
                beta[i, j, k] = value

        combos = [
            ((0, 0, 0),),
            ((0, 0, 1), (0, 1, 0), (1, 0, 0)),
            ((0, 1, 1), (1, 0, 1), (1, 1, 0)),
            ((1, 1, 1),),
            ((0, 0, 2), (0, 2, 0), (2, 0, 0)),
            (
                (0, 1, 2),
                (0, 2, 1),
                (1, 0, 2),
                (1, 2, 0),
                (2, 0, 1),
                (2, 1, 0),
            ),
            ((1, 1, 2), (1, 2, 1), (2, 1, 1)),
            ((0, 2, 2), (2, 0, 2), (2, 2, 0)),
            ((1, 2, 2), (2, 1, 2), (2, 2, 1)),
            ((2, 2, 2),),
        ]
        for value, indices in zip(vals, combos):
            fill(indices, float(value))
        return beta

    def _extract_hyperpolar(self, wfn):
        for key in ("SCF HYPERPOLARIZABILITY", "HYPERPOLARIZABILITY"):
            val = self._get_variable(wfn, key)
            if val is None:
                continue
            tensor = torch.tensor(val, dtype=torch.float64).reshape(-1)
            if tensor.numel() == 27:
                return tensor.view(3, 3, 3)
            if tensor.numel() == 10:
                return self._fill_hyper_sym(tensor)

        components = [
            ("XXX", (0, 0, 0)),
            ("XXY", (0, 0, 1)),
            ("XYY", (0, 1, 1)),
            ("YYY", (1, 1, 1)),
            ("XXZ", (0, 0, 2)),
            ("XYZ", (0, 1, 2)),
            ("YYZ", (1, 1, 2)),
            ("XZZ", (0, 2, 2)),
            ("YZZ", (1, 2, 2)),
            ("ZZZ", (2, 2, 2)),
        ]
        vals = []
        for suffix, _ in components:
            val = self._get_variable(wfn, f"SCF HYPERPOLARIZABILITY {suffix}")
            if val is None:
                val = self._get_variable(wfn, f"HYPERPOLARIZABILITY {suffix}")
            if val is None:
                return None
            vals.append(float(val))
        return self._fill_hyper_sym(torch.tensor(vals, dtype=torch.float64))

    def compute(self, pos: torch.Tensor, z: torch.Tensor, edge_index: torch.Tensor):
        if not self.supports(z):
            if self.on_unsupported == "error":
                raise ValueError("Unsupported elements for Psi4 backend")
            n = pos.shape[0]
            e = edge_index.shape[1]
            dtype = pos.dtype
            device = pos.device
            return (
                torch.zeros((n, 3, 3), device=device, dtype=dtype),
                torch.zeros((e, 3, 3), device=device, dtype=dtype),
            )

        mol = self._build_molecule(pos, z)

        hess = self.psi4.hessian(f"{self.method}/{self.basis}", molecule=mol)
        hess = np.array(hess, dtype=float)
        hess *= HESSIAN_AU_TO_EV_A2

        n = pos.shape[0]
        hess_t = torch.tensor(hess, dtype=torch.float64)
        hess_t = hess_t.view(n, 3, n, 3)
        idx = torch.arange(n)
        Hi = hess_t[idx, :, idx, :]
        i_idx, j_idx = edge_index.detach().cpu()
        Hij = hess_t[i_idx, :, j_idx, :]

        self.psi4.core.clean()

        return Hi.to(pos.device), Hij.to(pos.device)

    def energy(self, pos: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if not self.supports(z):
            raise ValueError("Unsupported elements for Psi4 backend")
        mol = self._build_molecule(pos, z)
        energy = self.psi4.energy(f"{self.method}/{self.basis}", molecule=mol)
        self.psi4.core.clean()
        energy_ev = float(energy) * HARTREE_TO_EV
        return torch.tensor([[energy_ev]], dtype=pos.dtype, device=pos.device)

    def properties(self, pos: torch.Tensor, z: torch.Tensor, allow_missing_hyper: bool = False):
        if not self.supports(z):
            raise ValueError("Unsupported elements for Psi4 backend")
        mol = self._build_molecule(pos, z)
        props = ["dipole", "polarizability", "hyperpolarizability"]
        wfn = None
        try:
            _, wfn = self.psi4.properties(
                f"{self.method}/{self.basis}",
                properties=props,
                molecule=mol,
                return_wfn=True,
            )
        except Exception as exc:
            if not allow_missing_hyper:
                raise RuntimeError(f"Psi4 properties failed: {exc}") from exc
            self.psi4.core.clean()
            _, wfn = self.psi4.properties(
                f"{self.method}/{self.basis}",
                properties=["dipole", "polarizability"],
                molecule=mol,
                return_wfn=True,
            )

        try:
            self.psi4.oeprop(wfn, "DIPOLE")
        except Exception:
            pass

        dipole = self._extract_vector(
            wfn,
            vector_keys=("SCF DIPOLE", "DIPOLE"),
            component_keys=(
                ("SCF DIPOLE X", "DIPOLE X", "CURRENT DIPOLE X"),
                ("SCF DIPOLE Y", "DIPOLE Y", "CURRENT DIPOLE Y"),
                ("SCF DIPOLE Z", "DIPOLE Z", "CURRENT DIPOLE Z"),
            ),
        )
        if dipole is None:
            self.psi4.core.clean()
            raise RuntimeError("Psi4 did not provide dipole data")
        dipole = dipole * DIPOLE_AU_TO_DEBYE

        polar = self._extract_polar(wfn)
        if polar is None:
            self.psi4.core.clean()
            raise RuntimeError("Psi4 did not provide polarizability data")

        hyperpolar = self._extract_hyperpolar(wfn)
        if hyperpolar is None:
            if allow_missing_hyper:
                hyperpolar = torch.zeros((3, 3, 3), dtype=torch.float64)
            else:
                self.psi4.core.clean()
                raise RuntimeError("Psi4 did not provide hyperpolarizability data")

        self.psi4.core.clean()

        return dipole, polar, hyperpolar.unsqueeze(0)

    def dipole(self, pos: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        dipole, _, _ = self.properties(pos, z, allow_missing_hyper=True)
        return dipole.to(device=pos.device, dtype=pos.dtype)

    def polar(self, pos: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        _, polar, _ = self.properties(pos, z, allow_missing_hyper=True)
        return polar.to(device=pos.device, dtype=pos.dtype)
