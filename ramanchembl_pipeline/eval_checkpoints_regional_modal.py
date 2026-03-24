"""
Evaluate ALL checkpoints with regional breakdown (fingerprint + CH + full).

    modal run ramanchembl_pipeline/eval_checkpoints_regional_modal.py
"""
from __future__ import annotations
import json
from pathlib import Path
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(["torch==2.3.1", "numpy", "scipy", "pandas", "rdkit", "matplotlib"])
)

data_vol = modal.Volume.from_name("raman-alignment-data")
ckpt_vol = modal.Volume.from_name("refinement-checkpoints")
app = modal.App("eval-refinement-regional", image=image)


@app.function(gpu="H200", volumes={"/data": data_vol, "/ckpts": ckpt_vol}, timeout=7200, memory=32768)
def eval_all():
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from scipy.signal import find_peaks
    import glob
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out_dir = Path("/data/outputs/refinement_eval_regional")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data
    data = np.load("/data/cache/dft_point_v1_1000.npz", allow_pickle=True)
    meta = pd.read_csv("/data/cache/dft_point_v1_1000.csv")
    y_pred = data['y_pred_spec'].astype(np.float32)
    y_target = data['y_target_spec'].astype(np.float32)
    x_grid = data['x_grid'].astype(np.float64)

    def norm(s):
        mx = s.max(axis=1, keepdims=True)
        return s / np.where(mx > 0, mx, 1.0)
    y_pred_n = norm(y_pred)
    y_target_n = norm(y_target)
    L = y_pred_n.shape[1]

    from rdkit import Chem
    from rdkit.Chem import AllChem
    smiles_col = 'smiles' if 'smiles' in meta.columns else 'SMILES'
    morgan_arr = np.zeros((len(meta), 2048), dtype=np.float32)
    for i, smi in enumerate(meta[smiles_col]):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            morgan_arr[i] = np.array(fp, dtype=np.float32)

    N = len(y_pred_n)
    rng = np.random.default_rng(42)
    idx = rng.permutation(N)
    te = idx[int(0.85 * N):]
    print(f"Test: {len(te)} molecules, grid={L}")

    # Model
    class FiLM(nn.Module):
        def __init__(self, cd, ch):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(cd, 128), nn.ReLU(), nn.Linear(128, 2*ch))
        def forward(self, x, c):
            g, b = self.net(c).chunk(2, dim=-1)
            return x * (1 + g.unsqueeze(-1)) + b.unsqueeze(-1)

    class Res(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.c1 = nn.Conv1d(ch, ch, 5, padding=2)
            self.bn1 = nn.BatchNorm1d(ch)
            self.c2 = nn.Conv1d(ch, ch, 5, padding=2)
            self.bn2 = nn.BatchNorm1d(ch)
        def forward(self, x):
            return F.relu(x + self.bn2(self.c2(F.relu(self.bn1(self.c1(x))))))

    class RefNet(nn.Module):
        def __init__(self, in_len, cd=2048, drop=0.15):
            super().__init__()
            self.enc1 = nn.Sequential(nn.Conv1d(1,16,7,padding=3), nn.BatchNorm1d(16), nn.ReLU(), Res(16))
            self.enc2 = nn.Sequential(nn.Conv1d(16,32,7,padding=3), nn.BatchNorm1d(32), nn.ReLU(), Res(32))
            self.enc3 = nn.Sequential(nn.Conv1d(32,64,7,padding=3), nn.BatchNorm1d(64), nn.ReLU(), Res(64))
            self.pool = nn.MaxPool1d(2)
            self.bot = Res(64)
            self.film = FiLM(cd, 64)
            self.drop = nn.Dropout(drop)
            self.u3 = nn.ConvTranspose1d(64,64,2,stride=2)
            self.d3 = nn.Sequential(nn.Conv1d(128,32,5,padding=2), nn.BatchNorm1d(32), nn.ReLU(), Res(32))
            self.u2 = nn.ConvTranspose1d(32,32,2,stride=2)
            self.d2 = nn.Sequential(nn.Conv1d(64,16,5,padding=2), nn.BatchNorm1d(16), nn.ReLU(), Res(16))
            self.u1 = nn.ConvTranspose1d(16,16,2,stride=2)
            self.d1 = nn.Sequential(nn.Conv1d(32,16,5,padding=2), nn.BatchNorm1d(16), nn.ReLU(), Res(16))
            self.head = nn.Conv1d(16,1,1)
            nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
            self.in_len = in_len
        def forward(self, s, m):
            x = s.unsqueeze(1)
            pad = (8 - x.shape[-1] % 8) % 8
            if pad: x = F.pad(x, (0, pad))
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            b = self.drop(self.film(self.bot(self.pool(e3)), m))
            d3 = self.d3(torch.cat([self.u3(b)[:,:,:e3.shape[-1]], e3], 1))
            d2 = self.d2(torch.cat([self.u2(d3)[:,:,:e2.shape[-1]], e2], 1))
            d1 = self.d1(torch.cat([self.u1(d2)[:,:,:e1.shape[-1]], e1], 1))
            delta = self.head(d1).squeeze(1)
            if pad: delta = delta[:, :self.in_len]
            return (s + delta).clamp(0, 1)

    dev = torch.device("cuda")
    model = RefNet(in_len=L).to(dev)

    # Metrics
    def extract_peaks_region(spec, xg, lo, hi):
        peaks, _ = find_peaks(spec, prominence=0.02, height=0.01, distance=5)
        pos = xg[peaks]
        mask = (pos >= lo) & (pos <= hi)
        return pos[mask]

    def f1_prec_rec(rp, pp, tol=15.):
        if len(rp)==0 or len(pp)==0:
            return 0., 0., 0.
        r, p = np.sort(rp), np.sort(pp)
        m, tp = set(), 0
        for x in p:
            d = np.abs(r - x)
            for i in np.argsort(d):
                if d[i] > tol: break
                if i not in m: m.add(i); tp += 1; break
        pr = tp/len(p) if p.size else 0
        rc = tp/len(r) if r.size else 0
        f1 = 2*pr*rc/(pr+rc) if (pr+rc)>0 else 0.
        return f1, pr, rc

    regions = {
        'full': (500, 4000),
        'fingerprint': (500, 2100),
        'fp_strict': (500, 1800),
        'ch_stretch': (1900, 3500),
    }

    def eval_checkpoint(ckpt_path, label):
        model.load_state_dict(torch.load(ckpt_path, map_location=dev, weights_only=True))
        model.eval()
        row = {'checkpoint': label}
        for rname, (lo, hi) in regions.items():
            for tol in [10, 15, 20]:
                f1s, precs, recs = [], [], []
                f1s_raw, precs_raw, recs_raw = [], [], []
                for i in te:
                    mfp = torch.from_numpy(morgan_arr[i:i+1]).to(dev)
                    with torch.no_grad():
                        refined = model(
                            torch.from_numpy(y_pred_n[i:i+1]).to(dev), mfp
                        ).cpu().numpy()[0]
                    ref_p = extract_peaks_region(y_target_n[i], x_grid, lo, hi)
                    pred_p = extract_peaks_region(refined, x_grid, lo, hi)
                    raw_p = extract_peaks_region(y_pred_n[i], x_grid, lo, hi)
                    f, p, r = f1_prec_rec(ref_p, pred_p, tol)
                    fb, pb, rb = f1_prec_rec(ref_p, raw_p, tol)
                    f1s.append(f); precs.append(p); recs.append(r)
                    f1s_raw.append(fb); precs_raw.append(pb); recs_raw.append(rb)
                row[f'{rname}_f1@{tol}'] = float(np.mean(f1s))
                row[f'{rname}_p@{tol}'] = float(np.mean(precs))
                row[f'{rname}_r@{tol}'] = float(np.mean(recs))
                row[f'{rname}_f1@{tol}_raw'] = float(np.mean(f1s_raw))
                row[f'{rname}_p@{tol}_raw'] = float(np.mean(precs_raw))
                row[f'{rname}_r@{tol}_raw'] = float(np.mean(recs_raw))
        # Cosine (full)
        cos_ref, cos_raw = [], []
        for i in te:
            mfp = torch.from_numpy(morgan_arr[i:i+1]).to(dev)
            with torch.no_grad():
                refined = model(torch.from_numpy(y_pred_n[i:i+1]).to(dev), mfp).cpu().numpy()[0]
            t = y_target_n[i]
            cos_ref.append((refined*t).sum()/(np.linalg.norm(refined)*np.linalg.norm(t)+1e-8))
            cos_raw.append((y_pred_n[i]*t).sum()/(np.linalg.norm(y_pred_n[i])*np.linalg.norm(t)+1e-8))
        row['cosine'] = float(np.mean(cos_ref))
        row['cosine_raw'] = float(np.mean(cos_raw))
        return row

    # Find all checkpoints
    def step_num(p):
        name = Path(p).stem
        if 'phase1' in name: return -1
        d = ''.join(filter(str.isdigit, name))
        return int(d) if d else 0

    all_results = []
    for ver in ['v8', 'v9', 'v10']:
        ver_dir = Path(f"/ckpts/{ver}")
        if not ver_dir.exists(): continue
        ckpts = sorted(glob.glob(str(ver_dir / "*.pth")), key=step_num)
        print(f"\n=== {ver}: {len(ckpts)} checkpoints ===")
        for cp in ckpts:
            name = Path(cp).stem
            if name == 'final': continue
            label = f"{ver}/{name}"
            try:
                row = eval_checkpoint(cp, label)
                all_results.append(row)
                print(f"  {label:35s} FP_F1@15={row['fingerprint_f1@15']:.3f}  CH_F1@15={row['ch_stretch_f1@15']:.3f}  Full_F1@15={row['full_f1@15']:.3f}")
            except Exception as e:
                print(f"  {label:35s} FAILED: {e}")

    # Raw baseline
    raw_row = {'checkpoint': 'raw_input'}
    for rname, (lo, hi) in regions.items():
        for tol in [10, 15, 20]:
            f1s, precs, recs = [], [], []
            for i in te:
                ref_p = extract_peaks_region(y_target_n[i], x_grid, lo, hi)
                raw_p = extract_peaks_region(y_pred_n[i], x_grid, lo, hi)
                f, p, r = f1_prec_rec(ref_p, raw_p, tol)
                f1s.append(f); precs.append(p); recs.append(r)
            raw_row[f'{rname}_f1@{tol}'] = float(np.mean(f1s))
            raw_row[f'{rname}_p@{tol}'] = float(np.mean(precs))
            raw_row[f'{rname}_r@{tol}'] = float(np.mean(recs))
    cos_raw = []
    for i in te:
        t = y_target_n[i]
        cos_raw.append((y_pred_n[i]*t).sum()/(np.linalg.norm(y_pred_n[i])*np.linalg.norm(t)+1e-8))
    raw_row['cosine'] = float(np.mean(cos_raw))
    all_results.insert(0, raw_row)

    df = pd.DataFrame(all_results)
    df.to_csv(out_dir / 'all_checkpoints_regional.csv', index=False)

    # Summary plot — fingerprint F1 trajectory
    WONG = {'blue': '#0072B2', 'vermillion': '#D55E00', 'green': '#009E73', 'orange': '#E69F00'}
    mol2r = {'f1@10': 0.551, 'f1@15': 0.631, 'f1@20': 0.705}

    mask = df['checkpoint'] != 'raw_input'
    x = range(mask.sum())

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel 1: Fingerprint F1 trajectory vs Mol2Raman
    ax = axes[0]
    for tol, ls, ms in [(10, '-', 'o'), (15, '-', 's'), (20, '-', '^')]:
        vals = df.loc[mask, f'fingerprint_f1@{tol}'].values
        ax.plot(x, vals, f'{ms}{ls}', ms=3, label=f'SpectraLoRA+ES @{tol}')
        ax.axhline(mol2r[f'f1@{tol}'], ls='--', lw=1, alpha=0.5, label=f'Mol2Raman @{tol} ({mol2r[f"f1@{tol}"]:.3f})')
    raw_fp15 = raw_row['fingerprint_f1@15']
    ax.axhline(raw_fp15, color='gray', ls=':', label=f'Raw baseline ({raw_fp15:.3f})')
    ax.set_xlabel('Checkpoint'); ax.set_ylabel('F1')
    ax.set_title('Fingerprint (500-2100 cm⁻¹)'); ax.legend(fontsize=7, ncol=2)

    # Panel 2: CH stretch trajectory
    ax = axes[1]
    for tol, ls, ms in [(10, '-', 'o'), (15, '-', 's'), (20, '-', '^')]:
        vals = df.loc[mask, f'ch_stretch_f1@{tol}'].values
        ax.plot(x, vals, f'{ms}{ls}', ms=3, label=f'@{tol}')
    ax.set_xlabel('Checkpoint'); ax.set_ylabel('F1')
    ax.set_title('CH Stretch (1900-3500 cm⁻¹)'); ax.legend(fontsize=8)

    # Panel 3: Precision vs Recall @ 15 (fingerprint)
    ax = axes[2]
    prec = df.loc[mask, 'fingerprint_p@15'].values
    rec = df.loc[mask, 'fingerprint_r@15'].values
    ax.plot(x, prec, 'o-', ms=3, color=WONG['blue'], label='Precision')
    ax.plot(x, rec, 's-', ms=3, color=WONG['vermillion'], label='Recall')
    ax.set_xlabel('Checkpoint'); ax.set_ylabel('Score')
    ax.set_title('FP Precision/Recall @15'); ax.legend()

    fig.suptitle('Regional F1 Across All ES Checkpoints', fontweight='bold', fontsize=14)
    plt.tight_layout()
    fig.savefig(out_dir / 'fig_regional_trajectory.png', dpi=300)
    fig.savefig(out_dir / 'fig_regional_trajectory.pdf')
    plt.close(fig)

    # Print final comparison table
    best_idx = df.loc[mask, 'fingerprint_f1@15'].idxmax()
    best = df.loc[best_idx]
    print("\n" + "="*80)
    print("BEST CHECKPOINT vs MOL2RAMAN (Fingerprint Region)")
    print("="*80)
    print(f"Checkpoint: {best['checkpoint']}")
    print(f"{'Metric':20s} {'Mol2Raman (in-dist)':>20s} {'SpectraLoRA+ES (OOD)':>22s} {'Raw SpectraLoRA':>18s} {'Ratio':>8s}")
    print("-"*90)
    for tol in [10, 15, 20]:
        m2r = mol2r[f'f1@{tol}']
        ours = best[f'fingerprint_f1@{tol}']
        raw = raw_row[f'fingerprint_f1@{tol}']
        print(f"FP F1@{tol:<14} {m2r:>20.3f} {ours:>22.3f} {raw:>18.3f} {ours/m2r:>7.0%}")
    print(f"{'FP Precision@15':20s} {'—':>20s} {best['fingerprint_p@15']:>22.3f} {raw_row['fingerprint_p@15']:>18.3f}")
    print(f"{'FP Recall@15':20s} {'—':>20s} {best['fingerprint_r@15']:>22.3f} {raw_row['fingerprint_r@15']:>18.3f}")
    print(f"{'Cosine (full)':20s} {'0.689':>20s} {best['cosine']:>22.3f} {raw_row['cosine']:>18.3f}")

    data_vol.commit()
    print(f"\nAll saved to {out_dir}")
    return df.to_dict('records')


@app.local_entrypoint()
def main():
    print("Evaluating all checkpoints with regional breakdown on H200...")
    results = eval_all.remote()
    print("Done.")
