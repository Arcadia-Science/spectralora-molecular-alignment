"""Evaluate all ES checkpoints from v9 on test set."""
import sys, json, glob
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.signal import find_peaks

REPO_ROOT = Path('/Users/rahul/Desktop/hp-proteins-ml')
sys.path.insert(0, str(REPO_ROOT))

# Data
cache_dir = REPO_ROOT / 'ramanchembl_pipeline' / 'artifacts' / 'alignment' / 'cache'
data = np.load(cache_dir / 'dft_point_v1_1000.npz', allow_pickle=True)
meta = pd.read_csv(cache_dir / 'dft_point_v1_1000.csv')

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
def evaluate(mdl, loader):
    mdl.eval()
    all_ref, all_raw, all_tgt = [], [], []
    for p, t, m in loader:
        all_ref.append(mdl(p, m).numpy())
        all_raw.append(p.numpy())
        all_tgt.append(t.numpy())
    all_ref = np.concatenate(all_ref)
    all_raw = np.concatenate(all_raw)
    all_tgt = np.concatenate(all_tgt)
    results = {}
    for tol in [5, 10, 15, 20]:
        results[f'f1@{tol}'] = float(batch_f1(all_ref, all_tgt, tol).mean())
        if tol == 15:
            results['f1@15_raw'] = float(batch_f1(all_raw, all_tgt, tol).mean())
    cos = np.mean([(all_ref[i]*all_tgt[i]).sum()/(np.linalg.norm(all_ref[i])*np.linalg.norm(all_tgt[i])+1e-8) for i in range(len(all_ref))])
    results['cosine'] = float(cos)
    return results

# Find all checkpoints — they're in alignment_results/
base = REPO_ROOT / 'ramanchembl_pipeline' / 'alignment_results'
ckpts_v8 = sorted(glob.glob(str(base / 'refinement_v8' / 'es_step*.pth')),
                   key=lambda x: int(''.join(filter(str.isdigit, Path(x).stem))))
ckpts_v9 = sorted(glob.glob(str(base / 'refinement_v9' / 'es_step*.pth')),
                   key=lambda x: int(''.join(filter(str.isdigit, Path(x).stem))))
ckpts_v10 = sorted(glob.glob(str(base / 'refinement_v10' / 'es_step*.pth')),
                    key=lambda x: int(''.join(filter(str.isdigit, Path(x).stem))))
# Also check phase1
phase1 = list(glob.glob(str(base / 'refinement_v8' / 'phase1_best.pth')))

all_ckpts = phase1 + ckpts_v8 + ckpts_v9 + ckpts_v10

print(f"Found {len(all_ckpts)} checkpoints")
print(f"Test set: {len(te)} molecules")

all_results = []
model = RefNet(in_len=L)

for ckpt_path in all_ckpts:
    name = Path(ckpt_path).stem
    parent = Path(ckpt_path).parent.name
    label = f"{parent}/{name}"
    try:
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
        res = evaluate(model, test_dl)
        res['checkpoint'] = label
        all_results.append(res)
        print(f"  {label:40s} F1@15={res['f1@15']:.3f}  F1@20={res['f1@20']:.3f}  cos={res['cosine']:.3f}")
    except Exception as e:
        print(f"  {label:40s} FAILED: {e}")

# Also eval raw input (no model)
raw_res = {'checkpoint': 'raw_input'}
for tol in [5, 10, 15, 20]:
    raw_res[f'f1@{tol}'] = float(batch_f1(y_pred_n[te], y_target_n[te], tol).mean())
raw_res['cosine'] = float(np.mean([(y_pred_n[te][i]*y_target_n[te][i]).sum()/(np.linalg.norm(y_pred_n[te][i])*np.linalg.norm(y_target_n[te][i])+1e-8) for i in range(len(te))]))
all_results.insert(0, raw_res)
print(f"\n  {'raw_input':40s} F1@15={raw_res['f1@15']:.3f}  F1@20={raw_res['f1@20']:.3f}  cos={raw_res['cosine']:.3f}")

# Save
out = REPO_ROOT / 'ramanchembl_pipeline' / 'artifacts' / 'refinement_eval'
out.mkdir(exist_ok=True)
df = pd.DataFrame(all_results)
df.to_csv(out / 'all_checkpoints_test.csv', index=False)

# Plot convergence across all checkpoints
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ckpt_names = df['checkpoint'].tolist()
x = range(len(ckpt_names))
ax1.plot(x, df['f1@10'], 'o-', markersize=3, label='F1@10')
ax1.plot(x, df['f1@15'], 's-', markersize=4, label='F1@15', color='#D55E00')
ax1.plot(x, df['f1@20'], '^-', markersize=3, label='F1@20')
ax1.axhline(raw_res['f1@15'], color='gray', ls='--', label='Raw baseline')
ax1.set_ylabel('F1'); ax1.set_xlabel('Checkpoint (chronological)')
ax1.legend(); ax1.set_title('F1 @ Multiple Tolerances Across All ES Checkpoints')

ax2.plot(x, df['cosine'], 'o-', markersize=4, color='#0072B2')
ax2.axhline(raw_res['cosine'], color='gray', ls='--', label='Raw baseline')
ax2.set_ylabel('Cosine Similarity'); ax2.set_xlabel('Checkpoint')
ax2.legend(); ax2.set_title('Cosine Similarity Across All ES Checkpoints')

plt.tight_layout()
fig.savefig(out / 'fig_all_checkpoints.png', dpi=300)
print(f"\nSaved to {out}")
print("\n=== SUMMARY ===")
print(df.to_string(index=False))
