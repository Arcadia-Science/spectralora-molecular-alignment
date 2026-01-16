from __future__ import annotations

from typing import Any, Dict, Optional

import torch

try:
    from .backends import HessianBackend
except ImportError:  # pragma: no cover - fallback for script-style imports
    from backends import HessianBackend


class MaceHessianBackend(HessianBackend):
    """Compute Hi/Hij using a MACE energy model and autograd Hessians.

    Requires: mace-torch and ase installed.

    Args:
        model_path: path to a MACE checkpoint (.model/.pt). Must be compatible with MACE.
        r_max: cutoff radius used to build neighbor lists.
        use_autograd: if True, use autograd Hessian. If False, raises NotImplementedError.
    """

    def __init__(
        self,
        device: torch.device,
        model_path: str,
        r_max: float = 5.0,
        use_autograd: bool = True,
        allowed_z: Optional[set[int]] = None,
        on_unsupported: str = "error",
    ) -> None:
        self.device = device
        self.model_path = model_path
        self.r_max = r_max
        self.use_autograd = use_autograd
        self.allowed_z = allowed_z
        self.on_unsupported = on_unsupported

        try:
            from mace.calculators import MACECalculator
        except Exception as exc:
            raise ImportError("mace-torch is required for MaceHessianBackend") from exc

        self.calculator = MACECalculator(model_paths=[model_path], device=str(device))
        if hasattr(self.calculator, "models") and self.calculator.models:
            self.model = self.calculator.models[0]
        elif hasattr(self.calculator, "model"):
            self.model = self.calculator.model
        else:
            raise RuntimeError("Unable to access model from MACECalculator")
        self.model.eval()
        if self.allowed_z is None and hasattr(self.model, "atomic_numbers"):
            try:
                self.allowed_z = set(int(x) for x in self.model.atomic_numbers)
            except Exception:
                self.allowed_z = None

    def supports(self, z: torch.Tensor) -> bool:
        if self.allowed_z is None:
            return True
        return bool(set(z.detach().cpu().tolist()) <= self.allowed_z)

    def _build_atomic_data(self, pos: torch.Tensor, z: torch.Tensor):
        try:
            from ase import Atoms
            from mace.data import AtomicData
            from mace.data.utils import Configuration, KeySpecification, config_from_atoms
            from mace.tools.utils import AtomicNumberTable
        except Exception as exc:
            raise ImportError("ase and mace are required to build AtomicData") from exc

        atoms = Atoms(numbers=z.detach().cpu().tolist(), positions=pos.detach().cpu().tolist())
        config = config_from_atoms(atoms, key_specification=KeySpecification())
        z_table = AtomicNumberTable(self.model.atomic_numbers.tolist())
        data = AtomicData.from_config(config, z_table=z_table, cutoff=float(self.model.r_max))
        data = data.to(self.device)
        num_nodes = data.positions.shape[0]
        data.batch = torch.zeros(num_nodes, dtype=torch.long, device=self.device)
        data.ptr = torch.tensor([0, num_nodes], dtype=torch.long, device=self.device)
        if getattr(data, "head", None) is not None and data.head.dim() == 0:
            data.head = data.head.view(1)
        return data

    def _energy_fn(self, data):
        try:
            out = self.model(
                data,
                compute_force=False,
                compute_virials=False,
                compute_stress=False,
                compute_hessian=False,
                compute_edge_forces=False,
                compute_atomic_stresses=False,
            )
        except TypeError:
            out = self.model(data)
        if isinstance(out, dict):
            energy = out.get("energy", None)
            if energy is None:
                energy = out.get("energies", None)
        else:
            energy = out
        if energy is None:
            raise RuntimeError("MACE output did not contain energy")
        return energy.sum()

    def compute(self, pos: torch.Tensor, z: torch.Tensor, edge_index: torch.Tensor):
        if not self.supports(z):
            if self.on_unsupported == "error":
                raise ValueError("Unsupported elements for MACE Hessian backend")
            n = pos.shape[0]
            e = edge_index.shape[1]
            dtype = pos.dtype
            device = pos.device
            return (
                torch.zeros((n, 3, 3), device=device, dtype=dtype),
                torch.zeros((e, 3, 3), device=device, dtype=dtype),
            )
        if not self.use_autograd:
            raise NotImplementedError("Finite-difference Hessian not implemented")

        pos = pos.detach().requires_grad_(True)
        data = self._build_atomic_data(pos, z)
        data.pos = pos
        if hasattr(data, "positions"):
            data.positions = pos

        def energy_from_flat(x: torch.Tensor) -> torch.Tensor:
            data.pos = x.view_as(pos)
            if hasattr(data, "positions"):
                data.positions = data.pos
            return self._energy_fn(data)

        n = pos.shape[0]
        flat = pos.reshape(-1)
        hess = torch.autograd.functional.hessian(energy_from_flat, flat, create_graph=False)
        hess = hess.view(n, 3, n, 3)
        idx = torch.arange(n, device=pos.device)
        Hi = hess[idx, :, idx, :]
        i, j = edge_index
        Hij = hess[i, :, j, :]
        return Hi, Hij

    def energy(self, pos: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if not self.supports(z):
            raise ValueError("Unsupported elements for MACE energy backend")
        pos = pos.detach()
        data = self._build_atomic_data(pos, z)
        data.pos = pos
        if hasattr(data, "positions"):
            data.positions = pos
        return self._energy_fn(data)
