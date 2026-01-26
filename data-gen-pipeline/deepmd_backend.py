from __future__ import annotations

from typing import Optional

import torch

try:
    from .backends import FiniteDifferenceHessianBackend, ForceBackend, HessianBackend
except ImportError:  # pragma: no cover - fallback for script-style imports
    from backends import FiniteDifferenceHessianBackend, ForceBackend, HessianBackend


DIPOLE_AU_TO_DEBYE = 2.541746

PTE_SYMBOLS = [
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


class DeepMDBase:
    def __init__(
        self,
        device: torch.device,
        model_path: str,
        type_map: Optional[str] = None,
    ) -> None:
        try:
            from deepmd.infer import DeepPot  # noqa: F401
        except Exception as exc:
            raise ImportError("deepmd-kit is required for DeepMD backends") from exc

        self.device = device
        self.model_path = model_path
        self.type_map = [t.strip() for t in type_map.split(",")] if type_map else None

        self._model = self._load_model(model_path)
        if self.type_map is None:
            self.type_map = self._infer_type_map()

        if not self.type_map:
            raise RuntimeError("DeepMD type map is required; pass --deepmd-type-map")

        self._symbol_to_type = {sym: idx for idx, sym in enumerate(self.type_map)}

    def _load_model(self, model_path: str):
        raise NotImplementedError

    def _infer_type_map(self):
        for attr in ("get_type_map", "type_map"):
            if hasattr(self._model, attr):
                try:
                    val = getattr(self._model, attr)()
                except TypeError:
                    val = getattr(self._model, attr)
                if val:
                    return list(val)
        return None

    def supports(self, z: torch.Tensor) -> bool:
        for zi in z.detach().cpu().tolist():
            symbol = PTE_SYMBOLS[int(zi)] if int(zi) < len(PTE_SYMBOLS) else None
            if symbol is None or symbol not in self._symbol_to_type:
                return False
        return True

    def _atype_from_z(self, z: torch.Tensor) -> torch.Tensor:
        indices = []
        for zi in z.detach().cpu().tolist():
            symbol = PTE_SYMBOLS[int(zi)] if int(zi) < len(PTE_SYMBOLS) else None
            if symbol is None or symbol not in self._symbol_to_type:
                raise ValueError(f"Unsupported element for DeepMD backend: Z={zi}")
            indices.append(self._symbol_to_type[symbol])
        return torch.tensor(indices, dtype=torch.int64)

    def _call(self, pos: torch.Tensor, z: torch.Tensor, *, atomic: bool = False, **kwargs):
        import numpy as np

        coords = pos.detach().cpu().numpy().astype(np.float64)[None, ...]
        cells = None
        atype = self._atype_from_z(z).detach().cpu().numpy().astype(np.int32)

        if hasattr(self._model, "eval"):
            out = self._model.eval(coords, cells, atype, atomic=atomic, **kwargs)
        else:
            out = self._model(coords, cells, atype)
        return out


class DeepMDDipoleBackend(DeepMDBase):
    def __init__(
        self,
        device: torch.device,
        model_path: str,
        type_map: Optional[str] = None,
        dipole_unit: str = "au",
    ) -> None:
        self.dipole_unit = dipole_unit
        super().__init__(device=device, model_path=model_path, type_map=type_map)

    def _load_model(self, model_path: str):
        from deepmd.infer import DeepDipole

        return DeepDipole(model_path)

    def dipole(self, pos: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        out = self._call(pos, z, atomic=False)
        if isinstance(out, (list, tuple)):
            out = out[0]
        dipole = torch.tensor(out, dtype=pos.dtype)
        if dipole.ndim > 1:
            dipole = dipole.reshape(-1)
        if dipole.numel() != 3:
            raise RuntimeError("DeepMD dipole output did not have 3 components")
        if self.dipole_unit == "au":
            dipole = dipole * DIPOLE_AU_TO_DEBYE
        return dipole.to(device=pos.device)


class DeepMDPolarBackend(DeepMDBase):
    def _load_model(self, model_path: str):
        from deepmd.infer import DeepPolar

        return DeepPolar(model_path)

    def polar(self, pos: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        out = self._call(pos, z, atomic=False)
        if isinstance(out, (list, tuple)):
            out = out[0]
        polar = torch.tensor(out, dtype=pos.dtype)
        if polar.ndim > 1:
            polar = polar.reshape(-1)
        if polar.numel() == 9:
            return polar.view(3, 3).to(device=pos.device)
        if polar.numel() == 6:
            xx, yy, zz, xy, xz, yz = polar
            mat = torch.tensor(
                [
                    [xx, xy, xz],
                    [xy, yy, yz],
                    [xz, yz, zz],
                ],
                dtype=pos.dtype,
                device=pos.device,
            )
            return mat
        if polar.numel() == 3:
            return torch.diag(polar.to(device=pos.device))
        raise RuntimeError("DeepMD polar output did not match expected shape")


class DeepMDPotBackend(DeepMDBase, ForceBackend):
    """DeepMD potential backend for energy/forces (optionally atomic energies)."""

    def __init__(
        self,
        device: torch.device,
        model_path: str,
        type_map: Optional[str] = None,
        head: Optional[str] = None,
    ) -> None:
        self.head = head
        super().__init__(device=device, model_path=model_path, type_map=type_map)

    def _load_model(self, model_path: str):
        from deepmd.infer import DeepPot

        try:
            if self.head:
                return DeepPot(model_path, device=str(self.device), head=self.head)
            return DeepPot(model_path, device=str(self.device))
        except TypeError:
            if self.head:
                return DeepPot(model_path, head=self.head)
            return DeepPot(model_path)

    def _parse_eval(self, out, pos: torch.Tensor):
        if not isinstance(out, (list, tuple)):
            return out, None, None, None, None
        energy = out[0] if len(out) > 0 else None
        forces = out[1] if len(out) > 1 else None
        virial = out[2] if len(out) > 2 else None
        atomic_energy = out[3] if len(out) > 3 else None
        atomic_virial = out[4] if len(out) > 4 else None
        return energy, forces, virial, atomic_energy, atomic_virial

    def _as_tensor(self, value, pos: torch.Tensor):
        if value is None:
            return None
        return torch.tensor(value, dtype=pos.dtype, device=pos.device)

    def energy(self, pos: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        out = self._call(pos, z, atomic=False, head=self.head)
        energy, _, _, _, _ = self._parse_eval(out, pos)
        energy_t = self._as_tensor(energy, pos)
        if energy_t is None:
            raise RuntimeError("DeepMD potential did not return energy")
        energy_t = energy_t.reshape(-1)
        if energy_t.numel() == 1:
            energy_t = energy_t.view(1, 1)
        return energy_t

    def atomic_energy(self, pos: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        out = self._call(pos, z, atomic=True, head=self.head)
        _, _, _, atomic_energy, _ = self._parse_eval(out, pos)
        atomic_t = self._as_tensor(atomic_energy, pos)
        if atomic_t is None:
            raise RuntimeError("DeepMD potential did not return atomic energy")
        if atomic_t.ndim == 2:
            atomic_t = atomic_t[0]
        return atomic_t

    def forces(self, pos: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        out = self._call(pos, z, atomic=False, head=self.head)
        _, forces, _, _, _ = self._parse_eval(out, pos)
        forces_t = self._as_tensor(forces, pos)
        if forces_t is None:
            raise RuntimeError("DeepMD potential did not return forces")
        if forces_t.ndim == 3:
            forces_t = forces_t[0]
        return forces_t


class DeepMDPotHessianBackend(HessianBackend):
    """Finite-difference Hessian backend using DeepMD potential forces."""

    def __init__(
        self,
        device: torch.device,
        model_path: str,
        type_map: Optional[str] = None,
        head: Optional[str] = None,
        step: float = 1e-3,
    ) -> None:
        self.device = device
        self.pot = DeepMDPotBackend(
            device=device,
            model_path=model_path,
            type_map=type_map,
            head=head,
        )
        self.fd = FiniteDifferenceHessianBackend(device=device, force_backend=self.pot, step=step)

    def supports(self, z: torch.Tensor) -> bool:
        return self.pot.supports(z)

    def compute(
        self, pos: torch.Tensor, z: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.fd.compute(pos, z, edge_index)

    def energy(self, pos: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.pot.energy(pos, z)

    def atomic_energy(self, pos: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.pot.atomic_energy(pos, z)
