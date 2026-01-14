import base64
import io
from typing import Tuple

import matplotlib
import torch
from fastapi import FastAPI
from matplotlib import pyplot as plt

from app.config import MODEL_DEVICE
from app.models import GeometryInput, NmrAggregateInput

matplotlib.use("Agg")

from detanet_model import (
    Lorenz_broadening,
    charge_model,
    get_raman_intensity,
    nn_vib_analysis,
    nmr_calculator,
    nmr_sca,
    uv_model,
)

app = FastAPI(title="DetaNet Model Service", version="0.1.0")


class ModelState:
    def __init__(self, device: str) -> None:
        self.device = torch.device(device)
        self.charge = charge_model(device=self.device)
        self.vib = nn_vib_analysis(device=self.device, Linear=False, scale=0.965)
        self.nmr = nmr_calculator(device=self.device)
        self.uv = uv_model(device=self.device)
        self.charge.eval()
        self.vib.eval()
        self.nmr.eval()
        self.uv.eval()


state = ModelState(MODEL_DEVICE)


def to_tensors(payload: GeometryInput, requires_grad: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    pos = torch.tensor(
        payload.pos,
        dtype=torch.float32,
        device=state.device,
        requires_grad=requires_grad,
    )
    z = torch.tensor(payload.z, dtype=torch.long, device=state.device)
    return pos, z


def normalize(values):
    max_val = max(values) if values else 0
    if max_val == 0:
        return values
    return [v / max_val for v in values]


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.post("/predict/charge")
async def predict_charge(payload: GeometryInput) -> dict:
    pos, z = to_tensors(payload)
    with torch.no_grad():
        charge = state.charge(z=z, pos=pos)
        charge = torch.nan_to_num(charge, nan=0.0, posinf=0.0, neginf=0.0)
        charge = charge.detach().cpu().reshape(-1).tolist()
    return {"charge": charge}


@app.post("/predict/vib")
async def predict_vib(payload: GeometryInput) -> dict:
    pos, z = to_tensors(payload, requires_grad=True)
    freq, iir, araman = state.vib(z=z, pos=pos)
    return {
        "freq": freq.detach().cpu().reshape(-1).tolist(),
        "ir_intensity": iir.detach().cpu().reshape(-1).tolist(),
        "raman_activity": araman.detach().cpu().reshape(-1).tolist(),
    }


@app.post("/predict/raman")
async def predict_raman(payload: GeometryInput) -> dict:
    pos, z = to_tensors(payload, requires_grad=True)
    freq, _, araman = state.vib(z=z, pos=pos)
    x_axis = torch.linspace(500, 4000, 3501, device=state.device)
    y_raman_act = Lorenz_broadening(freq, araman, c=x_axis, sigma=12)
    y_raman = get_raman_intensity(x_axis, y_raman_act)

    x = x_axis.detach().cpu().tolist()
    y = y_raman.detach().cpu().tolist()
    y_norm = normalize(y)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(x, y_norm, lw=2, color="blue")
    ax.set_xlim(500, 4000)
    ax.set_xlabel("Wavenumber (cm^-1)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_title("Raman Spectrum")
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150)
    plt.close(fig)
    png_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")

    return {
        "x": x,
        "y": y_norm,
        "png_base64": png_b64,
    }


@app.post("/predict/uv")
async def predict_uv(payload: GeometryInput) -> dict:
    pos, z = to_tensors(payload)
    with torch.no_grad():
        uv = state.uv(z=z, pos=pos)
        uv = torch.nan_to_num(uv, nan=0.0, posinf=0.0, neginf=0.0)
        uv = uv.detach().cpu().reshape(-1).tolist()
    return {"uv": uv}


@app.post("/predict/nmr")
async def predict_nmr(payload: GeometryInput) -> dict:
    pos, z = to_tensors(payload)
    with torch.no_grad():
        sc, sh = state.nmr(pos=pos, z=z)
        sc = torch.nan_to_num(sc, nan=0.0, posinf=0.0, neginf=0.0)
        sh = torch.nan_to_num(sh, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "sc": sc.detach().cpu().reshape(-1).tolist(),
        "sh": sh.detach().cpu().reshape(-1).tolist(),
    }


@app.post("/predict/nmr/aggregate")
async def predict_nmr_aggregate(payload: NmrAggregateInput) -> dict:
    sc = torch.tensor(payload.sc, dtype=torch.float32, device=state.device)
    sh = torch.tensor(payload.sh, dtype=torch.float32, device=state.device)
    indexc = torch.tensor(payload.indexc, dtype=torch.long, device=state.device)
    indexh = torch.tensor(payload.indexh, dtype=torch.long, device=state.device)

    with torch.no_grad():
        shiftc, intc, shifth, inth = nmr_sca(sc, sh, indexc, indexh)

    return {
        "shiftc": shiftc.detach().cpu().tolist(),
        "intc": intc.detach().cpu().tolist(),
        "shifth": shifth.detach().cpu().tolist(),
        "inth": inth.detach().cpu().tolist(),
    }
