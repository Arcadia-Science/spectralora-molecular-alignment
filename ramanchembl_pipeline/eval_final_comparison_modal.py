"""
Final metric comparison table — run on Modal H200.

    modal run ramanchembl_pipeline/eval_final_comparison_modal.py
"""
from __future__ import annotations
import json
from pathlib import Path
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(["torch==2.3.1", "numpy", "scipy", "pandas", "rdkit"])
)

data_vol = modal.Volume.from_name("raman-alignment-data")
ckpt_vol = modal.Volume.from_name("refinement-checkpoints")
app = modal.App("eval-final-comparison", image=image)


@app.function(gpu="H200", volumes={"/data": data_vol, "/ckpts": ckpt_vol}, timeout=3600, memory=32768)
def eval_final():
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from scipy.signal import find_peaks
    import glob

    out_dir = Path("/data/outputs/refinement_eval")
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
    print(f"Test: {len(te)} molecules")

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
            self.c1 = nn.Conv1d(ch, ch, 5, padding=2); self.bn1 = nn.BatchNorm1d(ch)
            self.c2 = nn.Conv1d(ch, ch, 5, padding=2); self.bn2 = nn.BatchNorm1d(ch)
        def forward(self, x):
            return F.relu(x + self.bn2(self.c2(F.relu(self.bn1(self.c1(x))))))
    class RefNet(nn.Module):
        def __init__(self, in_len, cd=2048, drop=0.15):
            super().__init__()
            self.enc1 = nn.Sequential(nn.Conv1d(1,16,7,padding=3), nn.BatchNorm1d(16), nn.ReLU(), Res(16))
            self.enc2 = nn.Sequential(nn.Conv1d(16,32,7,padding=3), nn.BatchNorm1d(32), nn.ReLU(), Res(32))
            self.enc3 = nn.Sequential(nn.Conv1d(32,64,7,padding=3), nn.BatchNorm1d(64), nn.ReLU(), Res(64))
            self.pool = nn.MaxPool1d(2)
            self.bot = Res(64); self.film = FiLM(cd, 64); self.drop = nn.Dropout(drop)
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
            e1 = self.enc1(x); e2 = self.enc2(self.pool(e1)); e3 = self.enc3(self.pool(e2))
            b = self.drop(self.film(self.bot(self.pool(e3)), m))
            d3 = self.d3(torch.cat([self.u3(b)[:,:,:e3.shape[-1]], e3], 1))
            d2 = self.d2(torch.cat([self.u2(d3)[:,:,:e2.shape[-1]], e2], 1))
            d1 = self.d1(torch.cat([self.u1(d2)[:,:,:e1.shape[-1]], e1], 1))
            delta = self.head(d1).squeeze(1)
            if pad: delta = delta[:, :self.in_len]
            return (s + delta).clamp(0, 1)

    dev = torch.device("cuda")
    model = RefNet(in_len=L).to(dev)

    def extract_peaks_region(spec, xg, lo, hi):
        peaks, _ = find_peaks(spec, prominence=0.02, height=0.01, distance=5)
        pos = xg[peaks]
        mask = (pos >= lo) & (pos <= hi)
        return pos[mask]

    def f1_prec_rec(rp, pp, tol=15.):
        if len(rp)==0 or len(pp)==0: return 0., 0., 0.
        r, p = np.sort(rp), np.sort(pp)
        m, tp = set(), 0
        for x in p:
            d = np.abs(r - x)
            for i in np.argsort(d):
                if d[i] > tol: break
                if i not in m: m.add(i); tp += 1; break
        pr = tp/len(p) if p.size else 0
        rc = tp/len(r) if r.size else 0
        return (2*pr*rc/(pr+rc) if (pr+rc)>0 else 0.), pr, rc

    # Evaluate ALL checkpoints across all versions
    def step_num(p):
        name = Path(p).stem
        if 'phase1' in name: return -1
        d = ''.join(filter(str.isdigit, name))
        return int(d) if d else 0

    all_rows = []
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
                model.load_state_dict(torch.load(cp, map_location=dev, weights_only=True))
                model.eval()
                row = {'checkpoint': label, 'version': ver, 'step': step_num(cp)}

                for rname, lo, hi in [('fingerprint', 500, 2100), ('full', 500, 4000)]:
                    for tol in [10, 15, 20]:
                        f1s, ps, rs = [], [], []
                        for i in te:
                            mfp = torch.from_numpy(morgan_arr[i:i+1]).to(dev)
                            with torch.no_grad():
                                refined = model(torch.from_numpy(y_pred_n[i:i+1]).to(dev), mfp).cpu().numpy()[0]
                            ref_p = extract_peaks_region(y_target_n[i], x_grid, lo, hi)
                            pred_p = extract_peaks_region(refined, x_grid, lo, hi)
                            raw_p = extract_peaks_region(y_pred_n[i], x_grid, lo, hi)
                            f, p, r = f1_prec_rec(ref_p, pred_p, tol)
                            fb, pb, rb = f1_prec_rec(ref_p, raw_p, tol)
                            f1s.append(f); ps.append(p); rs.append(r)
                        row[f'{rname}_f1@{tol}'] = float(np.mean(f1s))
                        row[f'{rname}_p@{tol}'] = float(np.mean(ps))
                        row[f'{rname}_r@{tol}'] = float(np.mean(rs))

                # Cosine
                cos_vals = []
                for i in te:
                    mfp = torch.from_numpy(morgan_arr[i:i+1]).to(dev)
                    with torch.no_grad():
                        refined = model(torch.from_numpy(y_pred_n[i:i+1]).to(dev), mfp).cpu().numpy()[0]
                    t = y_target_n[i]
                    cos_vals.append((refined*t).sum()/(np.linalg.norm(refined)*np.linalg.norm(t)+1e-8))
                row['cosine'] = float(np.mean(cos_vals))

                all_rows.append(row)
                print(f"  {label:35s} FP@15={row['fingerprint_f1@15']:.3f} Full@15={row['full_f1@15']:.3f}")
            except Exception as e:
                print(f"  {label:35s} FAILED: {e}")

    # Raw baseline
    raw_row = {'checkpoint': 'raw_input', 'version': 'none', 'step': -999}
    for rname, lo, hi in [('fingerprint', 500, 2100), ('full', 500, 4000)]:
        for tol in [10, 15, 20]:
            f1s, ps, rs = [], [], []
            for i in te:
                ref_p = extract_peaks_region(y_target_n[i], x_grid, lo, hi)
                raw_p = extract_peaks_region(y_pred_n[i], x_grid, lo, hi)
                f, p, r = f1_prec_rec(ref_p, raw_p, tol)
                f1s.append(f); ps.append(p); rs.append(r)
            raw_row[f'{rname}_f1@{tol}'] = float(np.mean(f1s))
            raw_row[f'{rname}_p@{tol}'] = float(np.mean(ps))
            raw_row[f'{rname}_r@{tol}'] = float(np.mean(rs))
    cos_raw = []
    for i in te:
        t = y_target_n[i]
        cos_raw.append((y_pred_n[i]*t).sum()/(np.linalg.norm(y_pred_n[i])*np.linalg.norm(t)+1e-8))
    raw_row['cosine'] = float(np.mean(cos_raw))
    all_rows.insert(0, raw_row)

    df = pd.DataFrame(all_rows)

    # Build final comparison CSV
    mask = df['checkpoint'] != 'raw_input'
    best_idx = df.loc[mask, 'fingerprint_f1@15'].idxmax()
    best = df.loc[best_idx]
    raw = df[df['checkpoint'] == 'raw_input'].iloc[0]

    # Mol2Raman numbers from Table 2 (fingerprint region) and Table 4 (full spectrum)
    # Sorrentino et al., Digital Discovery 2026
    mol2r = {
        'fingerprint_f1@10': 0.551, 'fingerprint_f1@15': 0.631, 'fingerprint_f1@20': 0.705,
        'fingerprint_p@15': 0.629, 'fingerprint_r@15': 0.634,
        'cosine': 0.689,
    }

    comparison_rows = []
    for label, col in [
        ('Fingerprint F1@10', 'fingerprint_f1@10'),
        ('Fingerprint F1@15', 'fingerprint_f1@15'),
        ('Fingerprint F1@20', 'fingerprint_f1@20'),
        ('Fingerprint Recall@15', 'fingerprint_r@15'),
        ('Fingerprint Precision@15', 'fingerprint_p@15'),
        ('Full F1@10', 'full_f1@10'),
        ('Full F1@15', 'full_f1@15'),
        ('Full F1@20', 'full_f1@20'),
        ('Cosine (full)', 'cosine'),
    ]:
        m2r_val = mol2r.get(col, '')
        ours_val = best[col]
        raw_val = raw[col]
        if m2r_val and m2r_val != '':
            ratio = f"{ours_val/m2r_val:.0%}"
        else:
            ratio = '—'
            m2r_val = '—'
        comparison_rows.append({
            'Metric': label,
            'Mol2Raman (in-dist QM9)': m2r_val,
            'SpectraLoRA+ES (OOD RamanChemBL)': round(ours_val, 3),
            'SpectraLoRA Raw': round(raw_val, 3),
            'Ratio (ours/M2R)': ratio,
        })

    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(out_dir / 'all_metric_comparison.csv', index=False)
    df.to_csv(out_dir / 'all_checkpoints_full.csv', index=False)

    print("\n" + "="*90)
    print(f"BEST CHECKPOINT: {best['checkpoint']}")
    print("="*90)
    print(comp_df.to_string(index=False))

    data_vol.commit()
    print(f"\nSaved to {out_dir}/all_metric_comparison.csv")
    print(f"Saved to {out_dir}/all_checkpoints_full.csv")
    return comparison_rows


@app.local_entrypoint()
def main():
    print("Running final comparison on H200...")
    results = eval_final.remote()
    for r in results:
        print(r)
