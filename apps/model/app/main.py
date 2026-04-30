"""
SpectraLoRA Raman inference service — local Docker equivalent of inference/serve.py.

POST /predict/raman  {"smiles": "CCO"}  ->  spectrum JSON
GET  /healthz
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException

from app.config import MODEL_DEVICE, WEIGHTS_DIR
from app.models import RamanRequest

# One GNN inference at a time per worker — prevents thread-pool thrash on CPU
_sem = asyncio.Semaphore(1)

SIGMA = 12.0
TEMP = 298.0
INIT_WL = 532.0
FREQ_SCALE = 0.967
X_GRID = np.linspace(500.0, 4000.0, 3501, dtype=np.float64)


# ---------------------------------------------------------------------------
# RefNet
# ---------------------------------------------------------------------------
class _FiLM(nn.Module):
    def __init__(self, cd: int, ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cd, 128), nn.ReLU(), nn.Linear(128, 2 * ch))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        g, b = self.net(c).chunk(2, dim=-1)
        return x * (1 + g.unsqueeze(-1)) + b.unsqueeze(-1)


class _Res(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.c1 = nn.Conv1d(ch, ch, 5, padding=2)
        self.bn1 = nn.BatchNorm1d(ch)
        self.c2 = nn.Conv1d(ch, ch, 5, padding=2)
        self.bn2 = nn.BatchNorm1d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.bn2(self.c2(F.relu(self.bn1(self.c1(x))))))


class _RefNet(nn.Module):
    def __init__(self, in_len: int, cd: int = 2048, drop: float = 0.15) -> None:
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv1d(1, 16, 7, padding=3), nn.BatchNorm1d(16), nn.ReLU(), _Res(16))
        self.enc2 = nn.Sequential(nn.Conv1d(16, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), _Res(32))
        self.enc3 = nn.Sequential(nn.Conv1d(32, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), _Res(64))
        self.pool = nn.MaxPool1d(2)
        self.bot = _Res(64)
        self.film = _FiLM(cd, 64)
        self.drop = nn.Dropout(drop)
        self.u3 = nn.ConvTranspose1d(64, 64, 2, stride=2)
        self.d3 = nn.Sequential(nn.Conv1d(128, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), _Res(32))
        self.u2 = nn.ConvTranspose1d(32, 32, 2, stride=2)
        self.d2 = nn.Sequential(nn.Conv1d(64, 16, 5, padding=2), nn.BatchNorm1d(16), nn.ReLU(), _Res(16))
        self.u1 = nn.ConvTranspose1d(16, 16, 2, stride=2)
        self.d1 = nn.Sequential(nn.Conv1d(32, 16, 5, padding=2), nn.BatchNorm1d(16), nn.ReLU(), _Res(16))
        self.head = nn.Conv1d(16, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.in_len = in_len

    def forward(self, s: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        x = s.unsqueeze(1)
        pad = (8 - x.shape[-1] % 8) % 8
        if pad:
            x = F.pad(x, (0, pad))
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.drop(self.film(self.bot(self.pool(e3)), m))
        d3 = self.d3(torch.cat([self.u3(b)[:, :, : e3.shape[-1]], e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3)[:, :, : e2.shape[-1]], e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2)[:, :, : e1.shape[-1]], e1], 1))
        delta = self.head(d1).squeeze(1)
        if pad:
            delta = delta[:, : self.in_len]
        return (s + delta).clamp(0, 1)


# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------
class _State:
    device: torch.device
    hi_model: nn.Module
    hij_model: nn.Module
    depolar_model: nn.Module
    refnet: _RefNet


_state = _State()


def _load_models() -> None:
    from train.train_detanet import build_model  # noqa: PLC0415

    device = torch.device(MODEL_DEVICE)
    _state.device = device

    cfg = json.loads(Path(f"{WEIGHTS_DIR}/config.json").read_text())

    def _args(task: str) -> argparse.Namespace:
        return argparse.Namespace(
            task=task,
            num_features=cfg.get("num_features", 160),
            num_block=cfg.get("num_block", 4),
            num_radial=cfg.get("num_radial", 32),
            attention_head=cfg.get("attention_head", 8),
            rc=cfg.get("rc", 5.0),
            dropout=cfg.get("dropout", 0.1),
            pre_layernorm=cfg.get("pre_layernorm", True),
            pre_layernorm_eps=cfg.get("pre_layernorm_eps", 1e-5),
            elora_path="/app/third_party/ELoRA",
            device=str(device),
            use_adalora=cfg.get("use_adalora", True),
            adalora_r=cfg.get("adalora_r", 256),
            adalora_alpha=cfg.get("adalora_alpha", 512),
            adalora_dropout=cfg.get("adalora_dropout", 0.1),
            adalora_tinit=cfg.get("adalora_tinit", 10),
            adalora_tfinal=cfg.get("adalora_tfinal", 20),
            adalora_total_step=cfg.get("adalora_total_step", 1000),
            adalora_target_r=cfg.get("adalora_target_r", 128),
            adalora_rslora=cfg.get("adalora_rslora", True),
            adalora_targets=cfg.get("adalora_targets", None),
            adalora_scalar_heads=cfg.get("adalora_scalar_heads", True),
            adalora_attention=cfg.get("adalora_attention", True),
            adalora_all_linears=cfg.get("adalora_all_linears", True),
            adapter_unfreeze_initial=cfg.get("adapter_unfreeze_initial", True),
            adapter_unfreeze_prefixes=cfg.get("adapter_unfreeze_prefixes", None),
            adapter_freeze_base=cfg.get("adapter_freeze_base", True),
        )

    def _extract_sd(obj: dict) -> dict:
        if isinstance(obj, dict):
            for k in ("model", "state_dict", "module"):
                if k in obj and isinstance(obj[k], dict):
                    obj = obj[k]
                    break
        if any(k.startswith("module.") for k in obj.keys()):
            obj = {k.replace("module.", "", 1): v for k, v in obj.items()}
        return obj

    def _load(task: str, fname: str) -> nn.Module:
        model = build_model(_args(task))
        sd = _extract_sd(torch.load(f"{WEIGHTS_DIR}/{fname}", map_location=device, weights_only=False))
        model.load_state_dict(sd, strict=False)
        model.to(device).eval()
        print(f"[{task}] loaded {fname}")
        return model

    _state.hi_model      = _load("Hi",      "Hi.pth")
    _state.hij_model     = _load("Hij",     "Hij.pth")
    _state.depolar_model = _load("depolar", "depolar.pth")

    _state.refnet = _RefNet(in_len=len(X_GRID)).to(device)
    _state.refnet.load_state_dict(
        torch.load(f"{WEIGHTS_DIR}/refnet.pth", map_location=device, weights_only=True)
    )
    _state.refnet.eval()
    print("[refnet] loaded refnet.pth — all models ready")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(_load_models)
    yield


app = FastAPI(title="SpectraLoRA Raman Service", version="0.1.0", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.post("/predict/raman")
async def predict_raman(payload: RamanRequest) -> dict:
    t_start = time.time()
    try:
        pos, z, mol = await asyncio.to_thread(_smiles_to_geometry, payload.smiles)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    t_conf = time.time()

    try:
        async with _sem:
            result = await asyncio.to_thread(_run_pipeline, pos, z, mol, payload.smiles, t_start, t_conf)
    except MemoryError:
        gc.collect()
        raise HTTPException(
            status_code=507,
            detail=f"Out of memory during Hessian computation for {len(z)}-atom molecule. "
                   f"Try increasing Docker Desktop memory allocation.",
        )
    return result


# ---------------------------------------------------------------------------
# Pipeline (all CPU-bound, runs inside asyncio.to_thread)
# ---------------------------------------------------------------------------
def _run_pipeline(
    pos: np.ndarray,
    z: np.ndarray,
    mol,
    smiles: str,
    t_start: float,
    t_conf: float,
) -> dict:
    freq, activity = _gnn_forward(pos, z)
    t_gnn = time.time()

    spectrum_raw = _broaden(freq, activity)
    t_broad = time.time()

    morgan_fp = _morgan_fingerprint(mol)
    spectrum_refined = _refine(spectrum_raw, morgan_fp)
    t_ref = time.time()

    peaks_pos, peaks_int         = _pick_peaks(spectrum_refined)
    peaks_raw_pos, peaks_raw_int = _pick_peaks(spectrum_raw)

    gc.collect()  # release autograd graph and intermediate tensors

    return {
        "smiles": smiles,
        "n_atoms": int(len(z)),
        "n_modes": int(len(freq)),
        "x_grid": X_GRID.tolist(),
        "spectrum_raw": spectrum_raw.tolist(),
        "spectrum_refined": spectrum_refined.tolist(),
        "peaks": {"positions_cm": peaks_pos, "intensities": peaks_int},
        "peaks_raw": {"positions_cm": peaks_raw_pos, "intensities": peaks_raw_int},
        "freq_cm": freq.tolist(),
        "timing": {
            "conformer_s": round(t_conf - t_start, 3),
            "gnn_s":       round(t_gnn  - t_conf,  3),
            "broaden_s":   round(t_broad - t_gnn,   3),
            "refine_s":    round(t_ref   - t_broad,  3),
            "total_s":     round(t_ref   - t_start,  3),
        },
    }


def _smiles_to_geometry(smiles: str) -> Tuple[np.ndarray, np.ndarray, object]:
    from rdkit import Chem  # noqa: PLC0415
    from rdkit.Chem import AllChem  # noqa: PLC0415

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) != 0:
        if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3(), randomSeed=42) != 0:
            AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    conf = mol.GetConformer()
    pos = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())], dtype=np.float32)
    z   = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int64)
    return pos, z, mol


def _morgan_fingerprint(mol) -> np.ndarray:
    from rdkit.Chem import AllChem  # noqa: PLC0415
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    return np.array(fp, dtype=np.float32)


def _gnn_forward(pos: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    from torch_geometric.nn import radius_graph  # noqa: PLC0415
    from detanet_model.constant import atom_masses  # noqa: PLC0415
    from detanet_model.spectra_simulator import chain_rule_raman, get_raman_act, hessfreq  # noqa: PLC0415

    device = _state.device
    pos_t = torch.tensor(pos, dtype=torch.float32, device=device, requires_grad=True)
    z_t   = torch.tensor(z,   dtype=torch.long,    device=device)
    edge_index = radius_graph(x=pos_t, r=5.0)

    with torch.enable_grad():
        hi   = _state.hi_model(pos=pos_t, z=z_t)
        hij  = _state.hij_model(pos=pos_t, z=z_t, edge_index=edge_index)
        dp   = _state.depolar_model(z=z_t, pos=pos_t)
        freq, modes = hessfreq(
            Hi=hi, Hij=hij, edge_index=edge_index,
            masses=atom_masses.to(device)[z_t],
            normal=False, linear=False, scale=1.0,
        )
        raman_act = get_raman_act(chain_rule_raman(dp=dp, modes=modes))

    freq      = torch.nan_to_num(freq,      nan=0.0, posinf=0.0, neginf=0.0)
    raman_act = torch.nan_to_num(raman_act, nan=0.0, posinf=0.0, neginf=0.0)

    freq_np = freq.detach().cpu().numpy().astype(np.float64)
    act_np  = raman_act.detach().cpu().numpy().astype(np.float64)

    valid = np.isfinite(freq_np) & np.isfinite(act_np) & (freq_np > 1e-8)
    return freq_np[valid] * FREQ_SCALE, act_np[valid]


def _broaden(freq: np.ndarray, activity: np.ndarray) -> np.ndarray:
    from detanet_model.spectra_simulator import Lorenz_broadening, get_raman_intensity  # noqa: PLC0415

    if freq.size == 0:
        return np.zeros_like(X_GRID)

    device = _state.device
    x_t = torch.as_tensor(X_GRID,    dtype=torch.float64, device=device)
    f_t = torch.as_tensor(freq,      dtype=torch.float64, device=device)
    a_t = torch.as_tensor(activity,  dtype=torch.float64, device=device)

    broadened = Lorenz_broadening(f_t, a_t, c=x_t, sigma=SIGMA)
    spec = get_raman_intensity(x_t, broadened, temp=TEMP, init_wl=INIT_WL)
    spec = spec.detach().cpu().numpy()
    spec = np.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)
    spec = np.clip(spec, 0.0, None)
    return spec / (spec.max() + 1e-12)


def _refine(spectrum: np.ndarray, morgan_fp: np.ndarray) -> np.ndarray:
    device = _state.device
    s_t = torch.from_numpy(spectrum.astype(np.float32)).unsqueeze(0).to(device)
    m_t = torch.from_numpy(morgan_fp).unsqueeze(0).to(device)
    with torch.no_grad():
        return _state.refnet(s_t, m_t).cpu().numpy()[0]


def _pick_peaks(spectrum: np.ndarray) -> Tuple[list, list]:
    from scipy.signal import find_peaks  # noqa: PLC0415
    peaks, _ = find_peaks(spectrum, prominence=0.03, distance=8)
    positions   = X_GRID[peaks]
    intensities = spectrum[peaks]
    order = np.argsort(intensities)[::-1]
    return positions[order].tolist(), intensities[order].tolist()
