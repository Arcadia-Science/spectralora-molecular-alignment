"""
Spectral refinement v9 — Warm-start ES from v8 checkpoint. More steps, bigger K, tuned sigma.

    modal run ramanchembl_pipeline/train_refinement_modal.py
"""
from __future__ import annotations
import json
from pathlib import Path
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(["torch==2.3.1", "numpy", "scipy", "pandas", "rdkit", "matplotlib"])
)

vol = modal.Volume.from_name("raman-alignment-data", create_if_missing=True)
app = modal.App("spectral-refinement-v9", image=image)


@app.function(gpu="H200", volumes={"/data": vol}, timeout=7200, memory=32768)  # 2 hrs
def train_refinement(max_cases: int = 1000, device: str = "cuda",
                     es_steps: int = 200, K: int = 80, sigma: float = 0.002, es_lr: float = 0.03):
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from scipy.signal import find_peaks
    import shutil, time
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out_dir = Path("/data/outputs/refinement_v10")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)

    # ==================================================================
    # DATA
    # ==================================================================
    cache_dir = Path("/data/cache")
    data = np.load(cache_dir / f"dft_point_v1_{max_cases}.npz", allow_pickle=True)
    meta = pd.read_csv(cache_dir / f"dft_point_v1_{max_cases}.csv")

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
    tr, va, te = idx[:n_tr], idx[n_tr:n_tr+n_va], idx[n_tr+n_va:]

    class DS(Dataset):
        def __init__(self, ixs):
            self.p = torch.from_numpy(y_pred_n[ixs])
            self.t = torch.from_numpy(y_target_n[ixs])
            self.m = torch.from_numpy(morgan_arr[ixs])
        def __len__(self): return len(self.p)
        def __getitem__(self, i): return self.p[i], self.t[i], self.m[i]

    val_dl = DataLoader(DS(va), batch_size=64, shuffle=False)
    test_dl = DataLoader(DS(te), batch_size=64, shuffle=False)
    print(f"Data: {N} mol, grid={L}, split={len(tr)}/{len(va)}/{len(te)}")

    # ==================================================================
    # MODEL (same architecture)
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

    dev = torch.device(device)
    model = RefNet(in_len=L).to(dev)

    # ==================================================================
    # WARM-START from v8 best checkpoint
    # ==================================================================
    # Try v9 step 1500 first, fall back to v8
    for ckpt_path in [
        Path("/data/outputs/refinement_v9/es_step1500.pth"),
        Path("/data/outputs/refinement_v9/es_best.pth"),
        Path("/data/outputs/refinement_v8/es_best.pth"),
    ]:
        if ckpt_path.exists():
            model.load_state_dict(torch.load(ckpt_path, map_location=dev, weights_only=True))
            print(f"Warm-started from {ckpt_path}")
            break
    else:
        print("WARNING: No checkpoint found, starting from scratch!")

    # ==================================================================
    # METRICS
    # ==================================================================
    x_grid_np = x_grid

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
    def evaluate(mdl, loader, tol=15.):
        mdl.eval()
        f1r, f1b = [], []
        for p, t, m in loader:
            p, t, m = p.to(dev), t.to(dev), m.to(dev)
            r = mdl(p, m).cpu().numpy()
            tn, pn = t.cpu().numpy(), p.cpu().numpy()
            f1r.extend(batch_f1(r, tn, tol).tolist())
            f1b.extend(batch_f1(pn, tn, tol).tolist())
        return {'f1': np.mean(f1r), 'f1_base': np.mean(f1b)}

    start_f1 = evaluate(model, val_dl)
    print(f"Starting point: F1@15={start_f1['f1']:.3f} (raw={start_f1['f1_base']:.3f})")

    # ==================================================================
    # ES: Extended hill-climb from v8 checkpoint
    # ==================================================================
    decoder_params = []
    for name, param in model.named_parameters():
        if any(k in name for k in ['d3', 'd2', 'd1', 'u3', 'u2', 'u1', 'head', 'bot', 'film']):
            decoder_params.append(param)
    n_dec = sum(p.numel() for p in decoder_params)
    print(f"ES: {n_dec:,} decoder params, K={K}, sigma={sigma}, lr={es_lr}, steps={es_steps}")

    def get_flat():
        return torch.cat([p.data.reshape(-1) for p in decoder_params])
    def set_flat(flat):
        off = 0
        for p in decoder_params:
            n = p.numel()
            p.data.copy_(flat[off:off+n].reshape(p.shape))
            off += n

    best_flat = get_flat().clone()
    best_fitness = evaluate(model, val_dl)['f1']
    es_history = []
    t0 = time.time()

    for step in range(es_steps):
        noise = torch.randn(K, n_dec, device=dev) * sigma
        rp_list, rn_list = [], []
        for k in range(K):
            set_flat(best_flat + noise[k])
            rp_list.append(evaluate(model, val_dl)['f1'])
            set_flat(best_flat - noise[k])
            rn_list.append(evaluate(model, val_dl)['f1'])

        rp_t = torch.tensor(rp_list, device=dev)
        rn_t = torch.tensor(rn_list, device=dev)
        adv = rp_t - rn_t

        grad = (adv.unsqueeze(1) * noise).mean(0) / (sigma + 1e-8)
        best_flat = best_flat + es_lr * grad

        set_flat(best_flat)
        new_f = evaluate(model, val_dl)['f1']

        if new_f > best_fitness:
            best_fitness = new_f
            torch.save(model.state_dict(), out_dir / 'es_best.pth')

        es_history.append({'step': step, 'fitness': new_f, 'best': best_fitness, 'sigma': sigma})

        dt = time.time() - t0
        if (step+1) % 10 == 0 or step == 0:
            print(f"  [ES {step+1:4d}] fitness={new_f:.4f} best={best_fitness:.4f} "
                  f"sigma={sigma:.5f} adv_mag={adv.abs().mean():.4f} [{dt:.0f}s]")

        # Adaptive sigma
        adv_mag = adv.abs().mean().item()
        if adv_mag < 0.0005:
            sigma = min(sigma * 1.5, 0.01)
        elif adv_mag > 0.005:
            sigma = max(sigma * 0.8, 0.0002)

        # Checkpoint every 100 steps
        if (step+1) % 100 == 0:
            torch.save(model.state_dict(), out_dir / f'es_step{step+1}.pth')
            vol.commit()
            print(f"  --- Checkpoint saved at step {step+1}, best={best_fitness:.4f} ---")

    # Load best
    ckpt = out_dir / 'es_best.pth'
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, weights_only=True))

    # ==================================================================
    # TEST
    # ==================================================================
    model.eval()
    all_ref, all_raw, all_tgt = [], [], []
    with torch.no_grad():
        for p, t, m in test_dl:
            p, t, m = p.to(dev), t.to(dev), m.to(dev)
            all_ref.append(model(p, m).cpu().numpy())
            all_raw.append(p.cpu().numpy())
            all_tgt.append(t.cpu().numpy())
    all_ref = np.concatenate(all_ref)
    all_raw = np.concatenate(all_raw)
    all_tgt = np.concatenate(all_tgt)

    print("\n" + "="*50)
    print("TEST SET RESULTS")
    print("="*50)
    results = {}
    for tol in [5, 10, 15, 20]:
        f1_raw = batch_f1(all_raw, all_tgt, tol).mean()
        f1_ref = batch_f1(all_ref, all_tgt, tol).mean()
        print(f"F1@{tol:<4} raw={f1_raw:.3f}  refined={f1_ref:.3f}  delta={f1_ref-f1_raw:+.3f}")
        results[f'f1@{tol}_raw'] = float(f1_raw)
        results[f'f1@{tol}_refined'] = float(f1_ref)

    cos_raw = np.mean([(all_raw[i]*all_tgt[i]).sum()/(np.linalg.norm(all_raw[i])*np.linalg.norm(all_tgt[i])+1e-8) for i in range(len(all_raw))])
    cos_ref = np.mean([(all_ref[i]*all_tgt[i]).sum()/(np.linalg.norm(all_ref[i])*np.linalg.norm(all_tgt[i])+1e-8) for i in range(len(all_ref))])
    print(f"Cosine  raw={cos_raw:.3f}  refined={cos_ref:.3f}  delta={cos_ref-cos_raw:+.3f}")
    results['cos_raw'] = float(cos_raw)
    results['cos_refined'] = float(cos_ref)

    # ==================================================================
    # PLOTS
    # ==================================================================
    WONG = {'blue': '#0072B2', 'vermillion': '#D55E00', 'black': '#000000', 'orange': '#E69F00'}

    per_mol_f1_raw = batch_f1(all_raw, all_tgt, 15.0)
    per_mol_f1_ref = batch_f1(all_ref, all_tgt, 15.0)
    improvement = per_mol_f1_ref - per_mol_f1_raw
    sort_idx = np.argsort(-improvement)

    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    for panel, ax in enumerate(axes.flat):
        i = sort_idx[panel]
        ax.plot(x_grid_np, all_tgt[i], color=WONG['black'], lw=1.8, alpha=0.8, label='DFT Reference')
        ax.plot(x_grid_np, all_raw[i], color=WONG['orange'], lw=1, alpha=0.5, label='SpectraLoRA (raw)')
        ax.plot(x_grid_np, all_ref[i], color=WONG['blue'], lw=1.3, label='ES-Refined')
        ax.set_title(f'F1@15: {per_mol_f1_raw[i]:.2f} → {per_mol_f1_ref[i]:.2f} ({improvement[i]:+.2f})', fontsize=11)
        ax.set_xlim(500, 2000)
        ax.set_xlabel('cm⁻¹')
        if panel == 0: ax.legend(fontsize=8)

    fig.suptitle(f'ES-Refined Spectra: Top-8 Improvements\n'
                 f'Overall F1@15: {per_mol_f1_raw.mean():.3f} → {per_mol_f1_ref.mean():.3f}',
                 fontweight='bold', fontsize=14)
    plt.tight_layout()
    fig.savefig(out_dir / 'fig_before_after.png', dpi=300)
    fig.savefig(out_dir / 'fig_before_after.pdf')
    plt.close(fig)

    es_df = pd.DataFrame(es_history)
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(es_df['step'], es_df['fitness'], alpha=0.4, color=WONG['blue'], label='Current')
    ax2.plot(es_df['step'], es_df['best'], color=WONG['vermillion'], lw=2, label='Best')
    ax2.axhline(float(start_f1['f1']), color='gray', ls='--', lw=1, label=f'v8 start ({start_f1["f1"]:.3f})')
    ax2.set_xlabel('ES Step'); ax2.set_ylabel('F1@15 (val)')
    ax2.set_title(f'ES Convergence (v9: warm-start from v8, K={K}, σ={sigma})')
    ax2.legend()
    fig2.savefig(out_dir / 'fig_es_convergence.png', dpi=300)
    plt.close(fig2)

    torch.save(model.state_dict(), out_dir / 'final.pth')
    es_df.to_csv(out_dir / 'es_history.csv', index=False)
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    np.savez_compressed(out_dir / 'test_preds.npz',
                        refined=all_ref, raw=all_raw, target=all_tgt, x_grid=x_grid_np)
    vol.commit()
    print(f"\nAll saved to {out_dir}")
    return results


@app.local_entrypoint()
def main(max_cases: int = 1000, es_steps: int = 200, num_candidates: int = 80,
         sigma: float = 0.002, es_lr: float = 0.03):
    print(f"Launching v10: warm-start from v9 step 1500, {es_steps} steps, K={num_candidates}, sigma={sigma}")
    results = train_refinement.remote(
        max_cases=max_cases, es_steps=es_steps, K=num_candidates, sigma=sigma, es_lr=es_lr
    )
    print("Done:", json.dumps(results, indent=2))
