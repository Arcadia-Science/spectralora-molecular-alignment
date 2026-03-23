"""
Spectral refinement v8 — PW-RMSE → Extended ES hill-climb.
Skip curriculum (it hurts). Go straight from PW-RMSE into 500-step ES with K=50.

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
app = modal.App("spectral-refinement-v8", image=image)


@app.function(gpu="H200", volumes={"/data": vol}, timeout=14400, memory=32768)
def train_refinement(max_cases: int = 1000, device: str = "cuda"):
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

    out_dir = Path("/data/outputs/refinement_v8")
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

    train_dl = DataLoader(DS(tr), batch_size=32, shuffle=True, drop_last=True)
    val_dl = DataLoader(DS(va), batch_size=64, shuffle=False)
    test_dl = DataLoader(DS(te), batch_size=64, shuffle=False)
    print(f"Data: {N} mol, grid={L}, split={len(tr)}/{len(va)}/{len(te)}")

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

    dev = torch.device(device)
    model = RefNet(in_len=L).to(dev)
    print(f"RefNet: {sum(p.numel() for p in model.parameters()):,} params")

    # ==================================================================
    # METRICS
    # ==================================================================
    x_grid_np = x_grid

    def pw_rmse(pred, tgt, thr=0.5, temp=0.05):
        tm = torch.sigmoid((tgt - thr) / temp)
        pm = torch.sigmoid((pred - thr) / temp)
        w = 8*tm*pm + 6*(1-tm)*pm + 5*tm*(1-pm) + 1*(1-tm)*(1-pm)
        return (w * (pred - tgt)**2).mean(-1).sqrt().mean()

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

    base = evaluate(model, val_dl)
    print(f"Before: F1@15={base['f1']:.3f} (raw={base['f1_base']:.3f})")
    t0 = time.time()
    history = []

    # ==================================================================
    # PHASE 1: Plain PW-RMSE (no augmentation, the thing that works)
    # ==================================================================
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300, eta_min=1.5e-5)
    best_f1, patience = 0., 0

    print(f"\n=== PHASE 1: Plain PW-RMSE (300 epochs) ===")
    for ep in range(300):
        model.train()
        ls = []
        for p, t, m in train_dl:
            p, t, m = p.to(dev), t.to(dev), m.to(dev)
            r = model(p, m)
            loss = pw_rmse(r, t)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); ls.append(loss.item())
        sched.step()
        if (ep+1) % 10 == 0 or ep == 0:
            ev = evaluate(model, val_dl)
            dt = time.time() - t0
            history.append({'phase': 1, 'epoch': ep, 'f1': ev['f1']})
            print(f"  [{ep+1:3d}] loss={np.mean(ls):.4f} F1@15={ev['f1']:.3f} "
                  f"(raw={ev['f1_base']:.3f}) [{dt:.0f}s]")
            if ev['f1'] > best_f1:
                best_f1 = ev['f1']; patience = 0
                torch.save(model.state_dict(), out_dir / 'phase1_best.pth')
            else:
                patience += 1
                if patience >= 8:
                    print(f"  Early stop at {ep+1}"); break

    model.load_state_dict(torch.load(out_dir / 'phase1_best.pth', weights_only=True))
    p1 = evaluate(model, val_dl)
    print(f"Phase 1 best: F1@15={p1['f1']:.3f}")

    # ==================================================================
    # PHASE 2: Extended Evolution Strategies (500 steps, K=50)
    # Directly hill-climb F1@15 on val set
    # ==================================================================
    print(f"\n=== PHASE 2: Evolution Strategies (500 steps, K=50) ===")

    # Collect decoder params
    decoder_params = []
    for name, param in model.named_parameters():
        if any(k in name for k in ['d3', 'd2', 'd1', 'u3', 'u2', 'u1', 'head', 'bot', 'film']):
            decoder_params.append(param)
    n_dec = sum(p.numel() for p in decoder_params)
    print(f"  ES on {n_dec:,} decoder params")

    def get_flat():
        return torch.cat([p.data.reshape(-1) for p in decoder_params])
    def set_flat(flat):
        off = 0
        for p in decoder_params:
            n = p.numel()
            p.data.copy_(flat[off:off+n].reshape(p.shape))
            off += n

    def fitness():
        return evaluate(model, val_dl)['f1']

    best_flat = get_flat().clone()
    best_fitness = fitness()
    sigma = 0.001
    K = 50
    es_lr = 0.02
    es_history = []

    for step in range(500):
        noise = torch.randn(K, n_dec, device=dev) * sigma
        rewards_pos, rewards_neg = [], []
        for k in range(K):
            set_flat(best_flat + noise[k])
            rp = fitness()
            set_flat(best_flat - noise[k])
            rn = fitness()
            rewards_pos.append(rp)
            rewards_neg.append(rn)

        rp_t = torch.tensor(rewards_pos, device=dev)
        rn_t = torch.tensor(rewards_neg, device=dev)
        adv = rp_t - rn_t

        # ES gradient
        grad = (adv.unsqueeze(1) * noise).mean(0) / (sigma + 1e-8)
        best_flat = best_flat + es_lr * grad

        set_flat(best_flat)
        new_f = fitness()

        if new_f > best_fitness:
            best_fitness = new_f
            torch.save(model.state_dict(), out_dir / 'es_best.pth')

        es_history.append({'step': step, 'fitness': new_f, 'best': best_fitness})

        dt = time.time() - t0
        if (step+1) % 10 == 0 or step == 0:
            print(f"  [ES {step+1:3d}] fitness={new_f:.3f} best={best_fitness:.3f} "
                  f"sigma={sigma:.5f} [{dt:.0f}s]")

        # Adaptive sigma
        adv_mag = adv.abs().mean().item()
        if adv_mag < 0.001:
            sigma = min(sigma * 1.3, 0.01)
        elif adv_mag > 0.01:
            sigma = max(sigma * 0.7, 0.0001)

        # Save checkpoint periodically
        if (step+1) % 50 == 0:
            torch.save(model.state_dict(), out_dir / f'es_step{step+1}.pth')
            vol.commit()

    # Load best ES model
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
    # PLOTS: Before/After spectra for 8 test molecules
    # ==================================================================
    WONG = {'blue': '#0072B2', 'vermillion': '#D55E00', 'black': '#000000', 'orange': '#E69F00'}

    # Sort by improvement magnitude
    per_mol_f1_raw = batch_f1(all_raw, all_tgt, 15.0)
    per_mol_f1_ref = batch_f1(all_ref, all_tgt, 15.0)
    improvement = per_mol_f1_ref - per_mol_f1_raw
    sort_idx = np.argsort(-improvement)  # best improvements first

    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    for panel, ax in enumerate(axes.flat):
        i = sort_idx[panel]
        ax.plot(x_grid_np, all_tgt[i], color=WONG['black'], lw=1.8, alpha=0.8, label='DFT Reference')
        ax.plot(x_grid_np, all_raw[i], color=WONG['orange'], lw=1, alpha=0.5, label='SpectraLoRA (raw)')
        ax.plot(x_grid_np, all_ref[i], color=WONG['blue'], lw=1.3, label='Refined')
        f1r = per_mol_f1_raw[i]
        f1f = per_mol_f1_ref[i]
        ax.set_title(f'F1@15: {f1r:.2f} → {f1f:.2f} ({f1f-f1r:+.2f})', fontsize=11)
        ax.set_xlim(500, 2000)
        ax.set_xlabel('cm⁻¹')
        if panel == 0:
            ax.legend(fontsize=8)

    fig.suptitle(f'Spectral Refinement: Top-8 Improvements (test set)\n'
                 f'Overall F1@15: {per_mol_f1_raw.mean():.3f} → {per_mol_f1_ref.mean():.3f}',
                 fontweight='bold', fontsize=14)
    plt.tight_layout()
    fig.savefig(out_dir / 'fig_before_after.png', dpi=300)
    fig.savefig(out_dir / 'fig_before_after.pdf')
    plt.close(fig)
    print(f"Saved before/after figure")

    # ES convergence plot
    es_df = pd.DataFrame(es_history)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(es_df['step'], es_df['fitness'], alpha=0.4, color=WONG['blue'], label='Current')
    ax2.plot(es_df['step'], es_df['best'], color=WONG['vermillion'], lw=2, label='Best')
    ax2.axhline(p1['f1'], color='gray', ls='--', lw=1, label=f'Phase 1 baseline ({p1["f1"]:.3f})')
    ax2.set_xlabel('ES Step')
    ax2.set_ylabel('F1@15 (val)')
    ax2.set_title('Evolution Strategies Convergence')
    ax2.legend()
    fig2.savefig(out_dir / 'fig_es_convergence.png', dpi=300)
    plt.close(fig2)
    print("Saved ES convergence figure")

    # Save everything
    torch.save(model.state_dict(), out_dir / 'final.pth')
    pd.DataFrame(history).to_csv(out_dir / 'history.csv', index=False)
    es_df.to_csv(out_dir / 'es_history.csv', index=False)
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    np.savez_compressed(out_dir / 'test_preds.npz',
                        refined=all_ref, raw=all_raw, target=all_tgt, x_grid=x_grid_np)
    vol.commit()
    print(f"\nAll saved to {out_dir}")
    return results


@app.local_entrypoint()
def main(max_cases: int = 1000):
    print("Launching v8 — PW-RMSE + Extended ES on H200...")
    results = train_refinement.remote(max_cases=max_cases)
    print("Done:", json.dumps(results, indent=2))
