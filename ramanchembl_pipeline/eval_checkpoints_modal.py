"""
Evaluate ALL refinement checkpoints on test set (Modal H200).
Produces per-checkpoint metrics + before/after plots for each.

    modal run ramanchembl_pipeline/eval_checkpoints_modal.py
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
app = modal.App("eval-refinement-checkpoints", image=image)


@app.function(
    gpu="H200",
    volumes={"/data": data_vol, "/ckpts": ckpt_vol, },
    timeout=7200,
    memory=32768,
)
def eval_all():
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from scipy.signal import find_peaks
    import glob, os, time
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out_dir = Path("/data/outputs/refinement_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    # DATA
    # ==================================================================
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
    n_tr, n_va = int(0.7 * N), int(0.15 * N)
    te = idx[n_tr + n_va:]

    class DS(Dataset):
        def __init__(self, ixs):
            self.p = torch.from_numpy(y_pred_n[ixs])
            self.t = torch.from_numpy(y_target_n[ixs])
            self.m = torch.from_numpy(morgan_arr[ixs])
        def __len__(self): return len(self.p)
        def __getitem__(self, i): return self.p[i], self.t[i], self.m[i]

    test_dl = DataLoader(DS(te), batch_size=64, shuffle=False)
    print(f"Test set: {len(te)} molecules, grid={L}")

    # ==================================================================
    # MODEL
    # ==================================================================
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
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
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
    x_grid_np = x_grid

    # ==================================================================
    # METRICS
    # ==================================================================
    def _extract(s):
        p, _ = find_peaks(s, prominence=0.02, height=0.01, distance=5)
        return x_grid_np[p]

    def _f1(rp, pp, tol=15.):
        if len(rp)==0 or len(pp)==0: return 0.
        r, p = np.sort(rp), np.sort(pp)
        m, tp = set(), 0
        for x in p:
            d = np.abs(r - x)
            for i in np.argsort(d):
                if d[i] > tol: break
                if i not in m: m.add(i); tp += 1; break
        pr = tp/len(p) if p.size else 0
        rc = tp/len(r) if r.size else 0
        return 2*pr*rc/(pr+rc) if (pr+rc)>0 else 0.

    def batch_f1(pn, tn, tol=15.):
        return np.array([_f1(_extract(tn[i]), _extract(pn[i]), tol) for i in range(len(pn))], dtype=np.float32)

    @torch.no_grad()
    def full_eval(mdl, loader):
        mdl.eval()
        all_ref, all_raw, all_tgt = [], [], []
        for p, t, m in loader:
            p, t, m = p.to(dev), t.to(dev), m.to(dev)
            all_ref.append(mdl(p, m).cpu().numpy())
            all_raw.append(p.cpu().numpy())
            all_tgt.append(t.cpu().numpy())
        all_ref = np.concatenate(all_ref)
        all_raw = np.concatenate(all_raw)
        all_tgt = np.concatenate(all_tgt)
        res = {}
        for tol in [5, 10, 15, 20]:
            res[f'f1@{tol}'] = float(batch_f1(all_ref, all_tgt, tol).mean())
        res[f'f1@15_raw'] = float(batch_f1(all_raw, all_tgt, 15.).mean())
        cos = np.mean([(all_ref[i]*all_tgt[i]).sum()/(np.linalg.norm(all_ref[i])*np.linalg.norm(all_tgt[i])+1e-8) for i in range(len(all_ref))])
        res['cosine'] = float(cos)
        return res, all_ref, all_raw, all_tgt

    # ==================================================================
    # FIND ALL CHECKPOINTS
    # ==================================================================
    WONG = {'blue': '#0072B2', 'vermillion': '#D55E00', 'black': '#000000', 'orange': '#E69F00'}

    all_results = []
    versions = ['v8', 'v9', 'v10']

    # Sort helper
    def step_num(path):
        name = Path(path).stem
        if 'phase1' in name: return -1
        digits = ''.join(filter(str.isdigit, name))
        return int(digits) if digits else 0

    for ver in versions:
        ver_dir = Path(f"/ckpts/{ver}")
        if not ver_dir.exists():
            continue
        ckpts = sorted(glob.glob(str(ver_dir / "*.pth")), key=step_num)
        ver_out = out_dir / ver
        ver_out.mkdir(exist_ok=True)

        print(f"\n=== {ver}: {len(ckpts)} checkpoints ===")
        for ckpt_path in ckpts:
            name = Path(ckpt_path).stem
            if name in ('final',):  # skip duplicates of es_best
                continue
            label = f"{ver}/{name}"
            try:
                model.load_state_dict(torch.load(ckpt_path, map_location=dev, weights_only=True))
                res, all_ref, all_raw, all_tgt = full_eval(model, test_dl)
                res['checkpoint'] = label
                res['version'] = ver
                res['step'] = step_num(ckpt_path)
                all_results.append(res)
                print(f"  {label:35s} F1@15={res['f1@15']:.3f}  F1@20={res['f1@20']:.3f}  cos={res['cosine']:.3f}")

                # Before/after figure for this checkpoint
                per_mol_f1_raw = batch_f1(all_raw, all_tgt, 15.)
                per_mol_f1_ref = batch_f1(all_ref, all_tgt, 15.)
                improvement = per_mol_f1_ref - per_mol_f1_raw
                sort_idx = np.argsort(-improvement)

                fig, axes = plt.subplots(2, 4, figsize=(24, 10))
                for panel, ax in enumerate(axes.flat):
                    i = sort_idx[panel]
                    ax.plot(x_grid_np, all_tgt[i], color=WONG['black'], lw=1.8, alpha=0.8, label='DFT Ref')
                    ax.plot(x_grid_np, all_raw[i], color=WONG['orange'], lw=1, alpha=0.5, label='Raw')
                    ax.plot(x_grid_np, all_ref[i], color=WONG['blue'], lw=1.3, label='Refined')
                    ax.set_title(f'F1@15: {per_mol_f1_raw[i]:.2f}→{per_mol_f1_ref[i]:.2f} ({improvement[i]:+.2f})', fontsize=10)
                    ax.set_xlim(500, 2000); ax.set_xlabel('cm⁻¹')
                    if panel == 0: ax.legend(fontsize=7)
                fig.suptitle(f'{label} | F1@15: {res["f1@15_raw"]:.3f}→{res["f1@15"]:.3f}', fontweight='bold', fontsize=13)
                plt.tight_layout()
                fig.savefig(ver_out / f'{name}.png', dpi=200)
                plt.close(fig)

            except Exception as e:
                print(f"  {label:35s} FAILED: {e}")

    # Raw baseline
    raw_res = {'checkpoint': 'raw_input', 'version': 'none', 'step': -999}
    for tol in [5, 10, 15, 20]:
        raw_res[f'f1@{tol}'] = float(batch_f1(y_pred_n[te], y_target_n[te], tol).mean())
    raw_res['cosine'] = float(np.mean([(y_pred_n[te][i]*y_target_n[te][i]).sum()/(np.linalg.norm(y_pred_n[te][i])*np.linalg.norm(y_target_n[te][i])+1e-8) for i in range(len(te))]))
    all_results.insert(0, raw_res)

    df = pd.DataFrame(all_results)
    df.to_csv(out_dir / 'all_checkpoints_test.csv', index=False)

    # ==================================================================
    # SUMMARY PLOT: F1 convergence across all versions
    # ==================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Assign cumulative step numbers
    cumulative = []
    offset = 0
    for _, row in df.iterrows():
        if row['checkpoint'] == 'raw_input':
            cumulative.append(-50)
            continue
        if row['version'] == 'v8' and row['step'] == -1:
            cumulative.append(0)
            continue
        s = row['step']
        if row['version'] == 'v8':
            cumulative.append(s)
            offset = s
        elif row['version'] == 'v9':
            cumulative.append(offset + s)
        elif row['version'] == 'v10':
            cumulative.append(offset + 1500 + s)
    df['cumulative_step'] = cumulative

    mask = df['checkpoint'] != 'raw_input'
    ax1.plot(df.loc[mask, 'cumulative_step'], df.loc[mask, 'f1@10'], 'o-', ms=3, label='F1@10')
    ax1.plot(df.loc[mask, 'cumulative_step'], df.loc[mask, 'f1@15'], 's-', ms=4, color=WONG['vermillion'], label='F1@15')
    ax1.plot(df.loc[mask, 'cumulative_step'], df.loc[mask, 'f1@20'], '^-', ms=3, label='F1@20')
    ax1.axhline(raw_res['f1@15'], color='gray', ls='--', lw=1, label=f'Raw F1@15 ({raw_res["f1@15"]:.3f})')
    # Mark version boundaries
    for ver, color in [('v8', '#2E7D32'), ('v9', '#1565C0'), ('v10', '#C62828')]:
        vdf = df[df['version'] == ver]
        if len(vdf):
            ax1.axvspan(vdf['cumulative_step'].min()-10, vdf['cumulative_step'].max()+10, alpha=0.05, color=color)
            ax1.text(vdf['cumulative_step'].median(), ax1.get_ylim()[0]+0.01, ver, ha='center', fontsize=10, color=color, fontweight='bold')
    ax1.set_xlabel('Cumulative ES Steps'); ax1.set_ylabel('F1')
    ax1.set_title('F1 @ Multiple Tolerances — Full ES Trajectory'); ax1.legend(fontsize=9)

    ax2.plot(df.loc[mask, 'cumulative_step'], df.loc[mask, 'cosine'], 'o-', ms=4, color=WONG['blue'])
    ax2.axhline(raw_res['cosine'], color='gray', ls='--', lw=1, label=f'Raw ({raw_res["cosine"]:.3f})')
    ax2.set_xlabel('Cumulative ES Steps'); ax2.set_ylabel('Cosine Similarity')
    ax2.set_title('Cosine Similarity — Full ES Trajectory'); ax2.legend()

    plt.tight_layout()
    fig.savefig(out_dir / 'fig_full_trajectory.png', dpi=300)
    fig.savefig(out_dir / 'fig_full_trajectory.pdf')
    plt.close(fig)

    data_vol.commit()
    print(f"\n=== ALL DONE ===")
    print(f"Results: {out_dir / 'all_checkpoints_test.csv'}")
    print(f"Plots: {out_dir}/v8/*.png, v9/*.png, v10/*.png")
    print(f"Summary: {out_dir / 'fig_full_trajectory.png'}")
    print(df[['checkpoint', 'f1@5', 'f1@10', 'f1@15', 'f1@20', 'cosine']].to_string(index=False))
    return df.to_dict('records')


@app.local_entrypoint()
def main():
    print("Evaluating all checkpoints on H200...")
    results = eval_all.remote()
    print("Done.")
