import importlib
import importlib.util
from pathlib import Path
from typing import Any, Callable, Optional

import torch


class HessianBackend:
    def compute(self, pos: torch.Tensor, z: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class HyperpolarBackend:
    def compute(self, pos: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ForceBackend:
    def forces(self, pos: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TorchScriptForceBackend(ForceBackend):
    """Load a torchscript model that predicts forces given (pos, z)."""

    def __init__(self, device: torch.device, model_path: str) -> None:
        self.device = device
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()

    def forces(self, pos: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.model(pos, z)


class FiniteDifferenceHessianBackend(HessianBackend):
    """Compute Hessian via finite differences of forces."""

    def __init__(self, device: torch.device, force_backend: ForceBackend, step: float = 1e-3) -> None:
        self.device = device
        self.force_backend = force_backend
        self.step = step

    def compute(self, pos: torch.Tensor, z: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pos = pos.detach()
        n = pos.shape[0]
        hess = torch.zeros((n, 3, n, 3), device=pos.device, dtype=pos.dtype)
        step = self.step

        for i in range(n):
            for a in range(3):
                delta = torch.zeros_like(pos)
                delta[i, a] = step
                f_plus = self.force_backend.forces(pos + delta, z)
                f_minus = self.force_backend.forces(pos - delta, z)
                deriv = -(f_plus - f_minus) / (2.0 * step)
                hess[:, :, i, a] = deriv

        idx = torch.arange(n, device=pos.device)
        Hi = hess[idx, :, idx, :]
        i, j = edge_index
        Hij = hess[i, :, j, :]
        return Hi, Hij


class AutogradHessianBackend(HessianBackend):
    def __init__(self, energy_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> None:
        self.energy_fn = energy_fn

    def compute(self, pos: torch.Tensor, z: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pos = pos.detach().requires_grad_(True)
        n = pos.shape[0]
        flat = pos.reshape(-1)

        def energy_from_flat(x: torch.Tensor) -> torch.Tensor:
            coords = x.view_as(pos)
            energy = self.energy_fn(coords, z)
            return energy.sum()

        hess = torch.autograd.functional.hessian(energy_from_flat, flat, create_graph=False)
        hess = hess.view(n, 3, n, 3)
        idx = torch.arange(n, device=pos.device)
        Hi = hess[idx, :, idx, :]
        i, j = edge_index
        Hij = hess[i, :, j, :]
        return Hi, Hij


def load_backend(spec: str, device: torch.device, **kwargs: Any) -> Any:
    module_name, cls_name = spec.split(":")
    module_path = Path(module_name)
    if module_path.exists():
        spec_obj = importlib.util.spec_from_file_location(module_path.stem, module_path)
        if spec_obj is None or spec_obj.loader is None:
            raise ImportError(f"Unable to load backend module from {module_path}")
        module = importlib.util.module_from_spec(spec_obj)
        spec_obj.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)
    cls = getattr(module, cls_name)
    return cls(device=device, **kwargs)
