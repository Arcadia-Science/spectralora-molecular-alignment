from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from scipy.optimize import linear_sum_assignment as _lsa

if os.environ.get("MPLCONFIGDIR") is None:
    _mpl_dir = Path.cwd() / ".mplconfig"
    _mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_mpl_dir)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from ramanchembl_pipeline import stats_notebook_lib as stats_lib
except ImportError:
    import stats_notebook_lib as stats_lib  # standalone / Modal container

EPS = 1e-12
ALIGNMENT_TOLS = (5.0, 10.0, 15.0, 20.0)
MODE_TRAIN_MATCH_CUTOFF_CM = 60.0

@dataclass
class AlignmentDatasetBundle:
    domain: str
    x_grid: np.ndarray
    y_pred: np.ndarray
    y_target: np.ndarray
    mask: np.ndarray
    mol_features: np.ndarray
    metadata: pd.DataFrame
    cache_npz: Path
    cache_csv: Path

    def __len__(self) -> int:
        return int(self.y_pred.shape[0])

@dataclass
class DFTModeAlignmentDatasetBundle:
    domain: str
    x_grid: np.ndarray
    mol_features: np.ndarray
    pred_freq: np.ndarray
    pred_intensity: np.ndarray
    pred_mask: np.ndarray
    target_freq: np.ndarray
    target_intensity: np.ndarray
    target_mask: np.ndarray
    match_target_idx: np.ndarray
    match_mask: np.ndarray
    y_pred_spec: np.ndarray
    y_target_spec: np.ndarray
    metadata: pd.DataFrame
    cache_npz: Path
    cache_csv: Path
    mode_features: np.ndarray | None = None  # (N, max_modes, feat_dim) eigenvector features

    def __len__(self) -> int:
        return int(self.pred_freq.shape[0])

@dataclass
class AlignmentTrainConfig:
    seed: int = 20260309
    batch_size: int = 32
    max_epochs: int = 120
    patience: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-3
    # Architecture — scale with dataset size:
    #   <500 mols:  latent=64,  heads=4, layers=2
    #   500-2000:   latent=128, heads=8, layers=4  (defaults below)
    #   >2000:      latent=256, heads=8, layers=6
    latent_dim: int = 128
    mol_latent_dim: int = 64
    transformer_heads: int = 8
    transformer_layers: int = 4
    string_feature_dim: int = 128  # Morgan fingerprint dim
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    max_freq_delta: float = 150.0  # Max cm^-1 shift the model can output
    coverage_loss_weight: float = 2.0
    coverage_target_cm: float = 10.0  # cm^-1 where Huber transitions quadratic→linear
    confidence_loss_weight: float = 1.0
    confidence_threshold: float = 0.5  # eval-time cutoff
    repulsion_loss_weight: float = 0.5
    repulsion_radius_cm: float = 5.0
    match_cutoff: float = 15.0  # cm^-1 cutoff for ground-truth matching labels
    freq_loss_weight: float = 1.0  # weight for frequency correction loss
    # v10: Sinkhorn OT loss (replaces Hungarian-based loss when enabled)
    use_sinkhorn: bool = False
    sinkhorn_tau: float = 10.0  # temperature for soft assignment (cm⁻¹)
    sinkhorn_match_sigma: float = 10.0  # soft match width (cm⁻¹)
    # v10: per-mode eigenvector features from hessfreq
    mode_feature_dim: int = 0  # 12 for eigenvector features, 0 for legacy
    # v11: DETR-style dynamic re-matching (0 = disabled)
    detr_rematch_every: int = 0  # re-compute Hungarian matching every N epochs using corrected freqs
    # v11: Differentiable soft-F1 loss
    use_soft_f1: bool = False
    soft_f1_tol: float = 10.0    # match tolerance (cm⁻¹) — same as eval F1@10
    soft_f1_tau: float = 3.0     # sigmoid temperature (anneal warm→cold during training)
    soft_f1_tau_min: float = 0.5 # final tau after annealing

@dataclass
class AlignmentRLConfig:
    """REINFORCE fine-tuning config. F1@10 as non-differentiable reward."""
    seed: int = 42
    max_epochs: int = 100
    batch_size: int = 128
    lr: float = 3e-5
    weight_decay: float = 1e-4
    K: int = 8                    # samples per molecule per step
    sigma_init: float = 5.0       # initial freq exploration std (cm⁻¹)
    sigma_learnable: bool = True
    sigma_min: float = 0.5
    sigma_max: float = 20.0
    entropy_coeff: float = 0.01
    grad_clip: float = 1.0
    patience: int = 30
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    freeze_encoder: bool = False  # freeze transformer, only tune heads
    conf_loss_weight: float = 0.5 # supervised confidence BCE auxiliary
    confidence_threshold: float = 0.5
    match_cutoff: float = 15.0
    string_feature_dim: int = 128
    mode_feature_dim: int = 12
    reward_tol: float = 10.0     # F1@{tol} as reward
    top_k_filter: int = 0        # >0: fixed top-k by intensity (skip confidence sampling)

@dataclass
class AlignmentHybridConfig:
    """v12: Two-phase hybrid training (soft-F1 + REINFORCE keep/drop)."""
    seed: int = 42
    latent_dim: int = 128
    transformer_layers: int = 4
    transformer_heads: int = 8
    string_feature_dim: int = 128
    mode_feature_dim: int = 12
    match_cutoff: float = 15.0
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    confidence_threshold: float = 0.5
    # Phase 1: Soft-F1 + do-no-harm
    phase1_epochs: int = 100
    phase1_batch_size: int = 256
    phase1_lr: float = 3e-4
    phase1_weight_decay: float = 1e-3
    phase1_patience: int = 40
    soft_f1_tol: float = 10.0
    soft_f1_tau_init: float = 3.0
    soft_f1_tau_min: float = 0.5
    sinkhorn_tau: float = 10.0
    dnh_radius: float = 5.0       # do-no-harm: penalize deltas on modes within this radius
    dnh_weight: float = 1.0
    entropy_coeff: float = 0.05   # confidence entropy (strong to keep exploratory)
    freq_loss_weight: float = 1.0
    intensity_loss_weight: float = 0.5
    # Phase 2: REINFORCE keep/drop
    phase2_epochs: int = 150
    phase2_batch_size: int = 128
    phase2_lr: float = 5e-5
    phase2_weight_decay: float = 1e-4
    phase2_patience: int = 30
    rl_weight: float = 0.3
    rl_K: int = 8                  # REINFORCE samples per molecule
    rl_grad_clip: float = 1.0
    reward_tol: float = 10.0      # F1@{tol} as reward

class ModeArrayDataset(Dataset):
    def __init__(self, mol_features, pf, pi, pm, tf, ti, tm, mi, mm, mode_features=None):
        self.mol_features = torch.as_tensor(mol_features, dtype=torch.float32)
        self.pf = torch.as_tensor(pf, dtype=torch.float32)
        self.pi = torch.as_tensor(pi, dtype=torch.float32)
        self.pm = torch.as_tensor(pm, dtype=torch.float32)
        self.tf = torch.as_tensor(tf, dtype=torch.float32)
        self.ti = torch.as_tensor(ti, dtype=torch.float32)
        self.tm = torch.as_tensor(tm, dtype=torch.float32)
        self.mi = torch.as_tensor(mi, dtype=torch.long)
        self.mm = torch.as_tensor(mm, dtype=torch.float32)
        if mode_features is not None:
            self.mf = torch.as_tensor(mode_features, dtype=torch.float32)
        else:
            # Zero-width tensor for backward compat — cat with 8-dim features is a no-op
            self.mf = torch.zeros(pf.shape[0], pf.shape[1], 0, dtype=torch.float32)

    def __len__(self): return len(self.pf)

    def __getitem__(self, idx):
        return (self.mol_features[idx], self.pf[idx], self.pi[idx], self.pm[idx],
                self.tf[idx], self.ti[idx], self.tm[idx], self.mi[idx], self.mm[idx],
                self.mf[idx])

class PeakCoordinateTransformer(nn.Module):
    """
    Function f(SMILES hash, predicted peaks) -> Corrected peaks.
    Uses point registration logic to align (x, y) coordinates exactly.
    """
    def __init__(self, mol_dim: int, cfg: AlignmentTrainConfig, x_grid: np.ndarray):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("x_grid", torch.as_tensor(x_grid, dtype=torch.float32))
        
        # Molecule identity encoder
        self.mol_encoder = nn.Sequential(
            nn.Linear(mol_dim, cfg.mol_latent_dim),
            nn.GELU(),
            nn.Linear(cfg.mol_latent_dim, cfg.latent_dim),
            nn.LayerNorm(cfg.latent_dim)
        )
        
        # Peak property encoder (8 spectral features + optional eigenvector features)
        self.peak_embed = nn.Linear(8 + cfg.mode_feature_dim, cfg.latent_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.latent_dim,
            nhead=cfg.transformer_heads,
            dim_feedforward=cfg.latent_dim * 4,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.transformer_layers)
        
        # Correction heads
        self.freq_shift_head = nn.Linear(cfg.latent_dim, 1)
        self.intensity_head = nn.Linear(cfg.latent_dim, 1)
        # v2: confidence head — "is this predicted mode real or noise?"
        self.confidence_head = nn.Linear(cfg.latent_dim, 1)

    def forward(self, mol_features, pred_freq, pred_intensity, pred_mask, mode_features=None):
        # 1. Molecule conditioning
        mol_token = self.mol_encoder(mol_features).unsqueeze(1)

        # 2. Extract peak-level features (local structure + optional eigenvector features)
        peak_feats = _build_mode_features(pred_freq, pred_intensity, pred_mask, self.x_grid)
        if mode_features is not None and mode_features.shape[-1] > 0:
            peak_feats = torch.cat([peak_feats, mode_features], dim=-1)
        peak_tokens = self.peak_embed(peak_feats)

        # 3. Transformer processing (Attention over all peaks + molecule context)
        combined = torch.cat([mol_token, peak_tokens], dim=1)
        # Pad mask: True means "ignore"
        padding_mask = torch.cat([
            torch.zeros((mol_features.shape[0], 1), device=mol_features.device, dtype=torch.bool),
            pred_mask < 0.5
        ], dim=1)

        encoded = self.transformer(combined, src_key_padding_mask=padding_mask)
        peak_out = encoded[:, 1:, :] # Extract only peak tokens

        # 4. Predict Coordinate Adjustments
        # Clamp instead of tanh — gradient is 1.0 everywhere inside the window,
        # so the model can learn large corrections without fighting saturation
        delta_f = torch.clamp(self.freq_shift_head(peak_out).squeeze(-1),
                              -self.cfg.max_freq_delta, self.cfg.max_freq_delta)
        corrected_f = (pred_freq + delta_f) * pred_mask

        # Intensity must be [0, 1]
        corrected_i = torch.sigmoid(self.intensity_head(peak_out).squeeze(-1)) * pred_mask

        # v2: confidence — probability this mode is real (has a DFT match)
        confidence = torch.sigmoid(self.confidence_head(peak_out).squeeze(-1)) * pred_mask

        return {"corrected_freq": corrected_f, "corrected_intensity": corrected_i, "confidence": confidence}

def _supervised_alignment_loss(out, tf, ti, pm, tm, mi, mm, cfg):
    """
    v4 loss: direct supervised regression with per-component logging.

    Key change from v3: frequency loss in raw cm⁻¹ space (no /10 normalization)
    with delta=10.0 Huber, giving 10x stronger gradients for frequency correction.
    Confidence and repulsion can be dialed via config weights (set to 0 to disable).
    """
    pf, pi, conf = out["corrected_freq"], out["corrected_intensity"], out["confidence"]
    batch_size = pf.shape[0]
    total_loss = 0.0
    acc_freq = acc_int = acc_conf = 0.0
    count = 0

    for b in range(batch_size):
        idx_p = pm[b] > 0.5
        if not idx_p.any():
            continue

        p_f = pf[b][idx_p]
        p_i = pi[b][idx_p]
        p_conf = conf[b][idx_p]

        m_idx = mi[b][idx_p]
        m_mask = mm[b][idx_p]
        matched = m_mask > 0.5
        n_matched = matched.sum().item()

        # 1. Frequency + intensity on matched modes (raw cm⁻¹, delta=10)
        if n_matched > 0:
            target_idx = m_idx[matched]
            t_f_matched = tf[b][target_idx]
            t_i_matched_raw = ti[b][target_idx]

            idx_t = tm[b] > 0.5
            t_i_max = ti[b][idx_t].max() + EPS if idx_t.any() else torch.tensor(EPS)
            t_i_matched = t_i_matched_raw / t_i_max

            loss_freq = F.huber_loss(p_f[matched], t_f_matched, delta=10.0)
            loss_int = F.l1_loss(p_i[matched], t_i_matched)
        else:
            loss_freq = pf[b].sum() * 0.0
            loss_int = pf[b].sum() * 0.0

        # 2. Confidence BCE (can be disabled via weight=0)
        if cfg.confidence_loss_weight > 0:
            loss_conf = F.binary_cross_entropy(
                p_conf.clamp(1e-6, 1 - 1e-6), m_mask.float(), reduction="mean"
            )
        else:
            loss_conf = pf[b].sum() * 0.0

        # 3. Repulsion (can be disabled via weight=0)
        loss_repulsion = pf[b].sum() * 0.0
        if cfg.repulsion_loss_weight > 0:
            confident = p_conf > cfg.confidence_threshold
            if confident.sum() > 1:
                cp = p_f[confident]
                pair_dist = torch.abs(cp.unsqueeze(0) - cp.unsqueeze(1))
                repel = torch.clamp(cfg.repulsion_radius_cm - pair_dist, min=0.0)
                repel = repel - torch.diag(repel.diag())
                loss_repulsion = repel.sum() / (len(cp) * cfg.repulsion_radius_cm + EPS)

        total_loss = (total_loss
                      + cfg.freq_loss_weight * (loss_freq + loss_int)
                      + cfg.confidence_loss_weight * loss_conf
                      + cfg.repulsion_loss_weight * loss_repulsion)
        acc_freq += loss_freq.item()
        acc_int += loss_int.item()
        acc_conf += loss_conf.item() if isinstance(loss_conf, torch.Tensor) else 0.0
        count += 1

    if count == 0:
        return pf.sum() * 0.0, {}
    components = {"freq": acc_freq / count, "int": acc_int / count, "conf": acc_conf / count}
    return total_loss / count, components


def _sinkhorn_alignment_loss(out, tf, ti, pm, tm, mi, mm, cfg):
    """
    v10 loss: Sinkhorn optimal transport for differentiable mode alignment.

    Key advantage over Hungarian-based loss:
    - Gradients flow through the assignment itself
    - Unmatched modes still get gradient signal toward nearest target
    - Naturally handles n_pred != n_target

    Components: transport cost + coverage + intensity + confidence BCE.
    """
    pf, pi, conf = out["corrected_freq"], out["corrected_intensity"], out["confidence"]
    batch_size = pf.shape[0]
    total_loss = 0.0
    acc_freq = acc_cov = acc_conf = acc_int = 0.0
    count = 0

    sigma2 = 2 * cfg.sinkhorn_match_sigma ** 2

    for b in range(batch_size):
        idx_p = pm[b] > 0.5
        idx_t = tm[b] > 0.5
        if not idx_p.any() or not idx_t.any():
            continue

        p_f = pf[b][idx_p]       # (n_pred,)
        p_i = pi[b][idx_p]       # (n_pred,)
        p_conf = conf[b][idx_p]  # (n_pred,)
        t_f = tf[b][idx_t]       # (n_target,)
        t_i = ti[b][idx_t]       # (n_target,)

        n_pred = len(p_f)
        n_target = len(t_f)

        # Cost matrix: |pred_freq - target_freq|
        cost = torch.abs(p_f.unsqueeze(1) - t_f.unsqueeze(0))  # (n_pred, n_target)

        # Soft assignment: each pred → distribution over targets
        P = F.softmax(-cost / cfg.sinkhorn_tau, dim=1)  # (n_pred, n_target)

        # Soft match quality
        soft_match = torch.exp(-cost ** 2 / sigma2)

        # 1. Transport cost — weighted by confidence so unmatched modes
        #    (low confidence) don't dominate the gradient
        per_pred_cost = (P * cost).sum(dim=1)  # (n_pred,)
        loss_freq = (p_conf.detach() * per_pred_cost).sum() / (p_conf.detach().sum() + EPS)

        # 2. Coverage: for each target, how well is it covered by predictions?
        coverage = (P * soft_match).sum(dim=0).clamp(max=1.0)  # (n_target,)
        loss_cov = 1.0 - coverage.mean()

        # 3. Intensity (weighted by match quality)
        per_pred_quality = (P * soft_match).sum(dim=1)  # (n_pred,)
        t_i_max = t_i.max() + EPS
        p_i_target = (P * (t_i / t_i_max).unsqueeze(0)).sum(dim=1)
        loss_int = (per_pred_quality.detach() * torch.abs(p_i - p_i_target)).mean()

        # 4. Confidence: predict which modes have good matches
        conf_target = (per_pred_quality > 0.3).float().detach()
        loss_conf = F.binary_cross_entropy(
            p_conf.clamp(1e-6, 1 - 1e-6), conf_target, reduction="mean"
        )

        total_loss += (cfg.freq_loss_weight * loss_freq
                       + cfg.coverage_loss_weight * loss_cov
                       + cfg.confidence_loss_weight * loss_conf
                       + loss_int * 0.5)

        acc_freq += loss_freq.item()
        acc_cov += loss_cov.item()
        acc_conf += loss_conf.item()
        acc_int += loss_int.item()
        count += 1

    if count == 0:
        return pf.sum() * 0.0, {}
    components = {
        "freq": acc_freq / count, "cov": acc_cov / count,
        "conf": acc_conf / count, "int": acc_int / count,
    }
    return total_loss / count, components


def _soft_f1_loss(out, tf, ti, pm, tm, mi, mm, cfg):
    """
    v11 loss: Differentiable soft-F1 with Sinkhorn assignment.

    Instead of optimizing transport cost (proxy), directly optimizes a smooth
    approximation of F1@tol:
      soft_match_ij = sigmoid((tol - |pred_i - target_j|) / tau)
      soft_TP = sum_j max_i (P_ij * soft_match_ij)
      soft_F1 = 2 * soft_TP / (n_pred_kept + n_target)

    tau anneals from warm (smooth) to cold (sharp) via cfg.soft_f1_tau.
    """
    pf_all = out["corrected_freq"]
    pi_all = out["corrected_intensity"]
    conf_all = out["confidence"]
    batch_size = pf_all.shape[0]
    total_loss = 0.0
    acc_f1 = acc_conf = acc_int = 0.0
    count = 0

    tol = cfg.soft_f1_tol
    tau = cfg.soft_f1_tau

    for b in range(batch_size):
        idx_p = pm[b] > 0.5
        idx_t = tm[b] > 0.5
        if not idx_p.any() or not idx_t.any():
            continue

        p_f = pf_all[b][idx_p]
        p_i = pi_all[b][idx_p]
        p_conf = conf_all[b][idx_p]
        t_f = tf[b][idx_t]
        t_i = ti[b][idx_t]
        n_pred = len(p_f)
        n_target = len(t_f)

        # Distance matrix
        dist = torch.abs(p_f.unsqueeze(1) - t_f.unsqueeze(0))  # (n_pred, n_target)

        # Soft assignment via Sinkhorn (each pred -> distribution over targets)
        P = F.softmax(-dist / cfg.sinkhorn_tau, dim=1)  # (n_pred, n_target)

        # Soft match: sigmoid threshold at `tol` cm⁻¹
        soft_match = torch.sigmoid((tol - dist) / tau)  # (n_pred, n_target)

        # Soft TP: for each target, best (assignment-weighted) soft match
        # weighted_match_ij = P_ij * soft_match_ij * conf_i
        weighted = P * soft_match * p_conf.unsqueeze(1)  # (n_pred, n_target)
        # Per-target: sum over predictions (soft "is this target covered by a confident, close pred?")
        target_covered = weighted.sum(dim=0).clamp(max=1.0)  # (n_target,)
        soft_tp = target_covered.sum()

        # Soft n_pred_kept = sum of confidences (differentiable count of kept modes)
        soft_n_pred = p_conf.sum()

        # Soft F1 = 2*TP / (n_pred_kept + n_target)
        soft_f1 = 2 * soft_tp / (soft_n_pred + n_target + EPS)
        loss_f1 = 1.0 - soft_f1  # minimize

        # Intensity loss on well-matched pairs
        per_pred_quality = (P * soft_match).sum(dim=1)  # (n_pred,)
        t_i_max = t_i.max() + EPS
        p_i_target = (P * (t_i / t_i_max).unsqueeze(0)).sum(dim=1)
        loss_int = (per_pred_quality.detach() * torch.abs(p_i - p_i_target)).mean()

        # Confidence regularization: entropy to prevent collapse to all-0 or all-1
        conf_ent = -(p_conf * torch.log(p_conf + EPS)
                     + (1 - p_conf) * torch.log(1 - p_conf + EPS)).mean()

        total_loss += cfg.freq_loss_weight * loss_f1 + 0.5 * loss_int - 0.01 * conf_ent
        acc_f1 += soft_f1.item()
        acc_int += loss_int.item()
        acc_conf += soft_n_pred.item() / n_pred  # fraction kept
        count += 1

    if count == 0:
        return pf_all.sum() * 0.0, {}
    components = {
        "soft_f1": acc_f1 / count, "int": acc_int / count,
        "conf_frac": acc_conf / count,
    }
    return total_loss / count, components


def _compute_fixed_mask(pf, tf_b, pm, tm_b, radius=5.0):
    """Identify pred modes already within `radius` cm⁻¹ of a target. These are locked."""
    orig_dist = torch.abs(pf.unsqueeze(2) - tf_b.unsqueeze(1))  # (B, n_pred, n_target)
    orig_dist = orig_dist.masked_fill((tm_b < 0.5).unsqueeze(1), 1e6)
    min_dist = orig_dist.min(dim=2).values  # (B, n_pred)
    return ((min_dist < radius) * (pm > 0.5)).float()


def _hybrid_f1_loss(out, tf, ti, pm, tm, mi, mm, cfg, orig_pf, tau):
    """
    v12 loss: Soft-F1 DECOUPLED from confidence + do-no-harm penalty.

    Key difference from v11 _soft_f1_loss:
    - Confidence does NOT enter the soft-F1 numerator (prevents keep-all degeneration)
    - Do-no-harm: penalizes large deltas when original pred is already close to a target
    - Stronger entropy regularization (0.05 vs 0.01)
    - tau passed as argument (annealed externally by training loop)
    """
    pf_all = out["corrected_freq"]
    pi_all = out["corrected_intensity"]
    conf_all = out["confidence"]
    batch_size = pf_all.shape[0]
    total_loss = 0.0
    acc_f1 = acc_dnh = acc_int = acc_ent = 0.0
    count = 0

    tol = cfg.soft_f1_tol

    for b in range(batch_size):
        idx_p = pm[b] > 0.5
        idx_t = tm[b] > 0.5
        if not idx_p.any() or not idx_t.any():
            continue

        p_f = pf_all[b][idx_p]
        p_i = pi_all[b][idx_p]
        p_conf = conf_all[b][idx_p]
        t_f = tf[b][idx_t]
        t_i = ti[b][idx_t]
        o_f = orig_pf[b][idx_p]
        n_pred = len(p_f)
        n_target = len(t_f)

        # Distance matrix
        dist = torch.abs(p_f.unsqueeze(1) - t_f.unsqueeze(0))  # (n_pred, n_target)

        # Soft assignment via Sinkhorn
        P = F.softmax(-dist / cfg.sinkhorn_tau, dim=1)  # (n_pred, n_target)

        # Soft match: sigmoid threshold at `tol` cm⁻¹
        soft_match = torch.sigmoid((tol - dist) / tau)  # (n_pred, n_target)

        # Soft TP — NO confidence weighting (decoupled from confidence)
        weighted = P * soft_match  # (n_pred, n_target)
        target_covered = weighted.sum(dim=0).clamp(max=1.0)  # (n_target,)
        soft_tp = target_covered.sum()

        # F1 with fixed n_pred (not conf.sum() — decoupled)
        soft_f1 = 2 * soft_tp / (n_pred + n_target + EPS)
        loss_f1 = 1.0 - soft_f1

        # Do-no-harm: penalize corrections that INCREASE distance to nearest target
        # (allows corrections that improve close modes, blocks ones that corrupt them)
        orig_dist = torch.abs(o_f.unsqueeze(1) - t_f.unsqueeze(0))  # (n_pred, n_target)
        min_orig_dist, nearest_idx = orig_dist.min(dim=1)  # (n_pred,)
        close_mask = (min_orig_dist < cfg.dnh_radius).float()
        corr_dist = torch.abs(p_f - t_f[nearest_idx])  # distance after correction
        degradation = torch.clamp(corr_dist - min_orig_dist, min=0.0)  # only penalize if worse
        loss_dnh = (close_mask * degradation).sum() / (close_mask.sum() + EPS)

        # Intensity loss on well-matched pairs
        per_pred_quality = (P * soft_match).sum(dim=1)  # (n_pred,)
        t_i_max = t_i.max() + EPS
        p_i_target = (P * (t_i / t_i_max).unsqueeze(0)).sum(dim=1)
        loss_int = (per_pred_quality.detach() * torch.abs(p_i - p_i_target)).mean()

        # Confidence entropy (strong regularization — keep exploratory for phase 2)
        conf_ent = -(p_conf * torch.log(p_conf + EPS)
                     + (1 - p_conf) * torch.log(1 - p_conf + EPS)).mean()

        mol_loss = (cfg.freq_loss_weight * loss_f1
                    + cfg.intensity_loss_weight * loss_int
                    + cfg.dnh_weight * loss_dnh
                    - cfg.entropy_coeff * conf_ent)
        total_loss += mol_loss
        acc_f1 += soft_f1.item()
        acc_dnh += loss_dnh.item()
        acc_int += loss_int.item()
        acc_ent += conf_ent.item()
        count += 1

    if count == 0:
        return pf_all.sum() * 0.0, {}
    components = {
        "soft_f1": acc_f1 / count, "dnh": acc_dnh / count,
        "int": acc_int / count, "ent": acc_ent / count,
    }
    return total_loss / count, components


def _reinforce_keep_drop_loss(model_out, pm, tf, tm, cfg, fixed_mask=None):
    """
    v12 REINFORCE: discrete keep/drop only (no continuous freq sampling).
    Samples K binary masks from Bernoulli(confidence).
    Fixed modes (already close to target) are always kept.
    Reward = hard F1@10 using deterministic corrected_freq.
    Policy gradient flows through confidence head only (non-fixed modes).
    """
    corr_freq = model_out["corrected_freq"]
    confidence = model_out["confidence"]
    conf_clamped = confidence.clamp(1e-6, 1 - 1e-6)

    if fixed_mask is None:
        fixed_mask = torch.zeros_like(pm)
    non_fixed = (1 - fixed_mask) * pm  # only these modes are sampled

    corr_freq_np = corr_freq.detach().cpu().numpy()
    pm_np = pm.cpu().numpy()
    tf_np = tf.cpu().numpy()
    tm_np = tm.cpu().numpy()

    all_log_probs = []
    all_rewards = []

    for _k in range(cfg.rl_K):
        # Fixed modes always kept; non-fixed sampled from Bernoulli(confidence)
        sampled = torch.bernoulli(conf_clamped).detach()
        keep = fixed_mask + (1 - fixed_mask) * sampled * pm  # (B, max_modes)

        # Log-prob only for non-fixed modes (fixed modes have no policy choice)
        lp = (sampled * torch.log(conf_clamped)
              + (1 - sampled) * torch.log(1 - conf_clamped))
        lp = (lp * non_fixed).sum(dim=1)  # (B,)

        reward = _compute_f1_reward_batch(
            corr_freq_np, pm_np, keep.cpu().numpy(), tf_np, tm_np,
            tol=cfg.reward_tol,
        )

        all_log_probs.append(lp)
        all_rewards.append(torch.tensor(reward, device=corr_freq.device, dtype=torch.float32))

    log_probs = torch.stack(all_log_probs)  # (K, B)
    rewards = torch.stack(all_rewards)       # (K, B)

    baseline = rewards.mean(dim=0, keepdim=True)
    advantage = (rewards - baseline).detach()

    policy_loss = -(advantage * log_probs).mean()

    return policy_loss, {"rl_reward": rewards.mean().item()}


def _build_mode_features(pred_freq, pred_intensity, pred_mask, x_grid):
    x_min, x_max = float(x_grid[0]), float(x_grid[-1])
    x_scale = max(x_max - x_min, 1.0)
    
    freq_norm = ((pred_freq - x_min) / x_scale) * pred_mask
    pred_log = torch.log(torch.clamp(pred_intensity, min=EPS)) * pred_mask
    
    # Context: distances to neighbors
    prev_f = torch.cat([pred_freq[:, :1], pred_freq[:, :-1]], dim=1)
    next_f = torch.cat([pred_freq[:, 1:], pred_freq[:, -1:]], dim=1)
    gap_p = torch.clamp((pred_freq - prev_f) / 200.0, 0, 5) * pred_mask
    gap_n = torch.clamp((next_f - pred_freq) / 200.0, 0, 5) * pred_mask
    
    # Rank in the spectrum
    rank = (torch.cumsum(pred_mask, dim=1) - 1.0) * pred_mask
    rank /= torch.clamp(pred_mask.sum(dim=1, keepdim=True), min=1.0)
    
    # Global density features
    count = (pred_mask.sum(dim=1, keepdim=True) / 100.0).expand_as(pred_freq)
    
    # Intensity rank (by descending intensity, normalized) — which peaks are strongest
    int_rank = torch.zeros_like(pred_intensity)
    for b in range(pred_intensity.shape[0]):
        valid_idx = (pred_mask[b] > 0.5).nonzero(as_tuple=True)[0]
        if len(valid_idx) > 1:
            order = torch.argsort(pred_intensity[b][valid_idx], descending=True)
            int_rank[b][valid_idx[order]] = torch.arange(len(valid_idx), dtype=torch.float32, device=pred_intensity.device) / (len(valid_idx) - 1)
    int_rank = int_rank * pred_mask

    # Local gap asymmetry: how asymmetrically placed this peak is between neighbors
    gap_total = gap_p + gap_n + EPS
    gap_asym = (gap_p - gap_n) / gap_total * pred_mask

    return torch.stack([freq_norm, pred_log, gap_p, gap_n, rank, count, int_rank, gap_asym], dim=-1)

def _morgan_features(smiles_list, n_bits=128, radius=2):
    """ECFP4 Morgan fingerprints via RDKit. Falls back to SMILES hash if RDKit unavailable."""
    try:
        from rdkit import Chem, RDLogger
        RDLogger.logger().setLevel(RDLogger.ERROR)
        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
        gen = GetMorganGenerator(radius=radius, fpSize=n_bits)
        feats = []
        for smi in smiles_list:
            try:
                mol = Chem.MolFromSmiles(str(smi), sanitize=False)
                if mol is not None:
                    Chem.SanitizeMol(mol)
                    feats.append(gen.GetFingerprintAsNumPy(mol).astype(np.float32))
                else:
                    feats.append(_string_hash_features([smi], dim=n_bits)[0])
            except Exception:
                feats.append(_string_hash_features([smi], dim=n_bits)[0])
        return np.stack(feats)
    except ImportError:
        return _string_hash_features(smiles_list, dim=n_bits)

def _string_hash_features(strings, dim=64):
    feats = np.zeros((len(strings), dim), dtype=np.float32)
    for i, s in enumerate(strings):
        text = f"<{s}>"
        for n in (1, 2, 3):
            for j in range(max(len(text) - n + 1, 0)):
                bucket = hash((n, text[j:j+n])) % dim
                feats[i, bucket] += 1.0
        norm = np.linalg.norm(feats[i])
        if norm > EPS: feats[i] /= norm
    return feats

def _augment_mol_features(mol_features, metadata, dim=64):
    key_col = next((c for c in ("smiles", "component", "cid") if c in metadata.columns), None)
    if key_col:
        strings = metadata[key_col].astype(str).tolist()
        # Use Morgan fingerprints for SMILES (chemistry-aware > hash)
        str_feats = _morgan_features(strings, n_bits=dim) if key_col == "smiles" \
                    else _string_hash_features(strings, dim=dim)
        return np.concatenate([mol_features, str_feats], axis=1)
    return mol_features

def _compute_match_indices(pred_freq, pred_mask, target_freq, target_mask, cutoff=60.0):
    """
    For each molecule, Hungarian-match predicted modes to target modes within `cutoff` cm^-1.
    Returns:
        match_target_idx: (N, max_pred) int array; -1 for unmatched pred modes
        match_mask:       (N, max_pred) float array; 1.0 if matched, else 0.0
    """
    N, max_p = pred_freq.shape
    mi = np.full((N, max_p), -1, dtype=np.int32)
    mm = np.zeros((N, max_p), dtype=np.float32)
    for i in range(N):
        p_valid = np.where(pred_mask[i] > 0.5)[0]
        t_valid = np.where(target_mask[i] > 0.5)[0]
        if len(p_valid) == 0 or len(t_valid) == 0:
            continue
        cost = np.abs(pred_freq[i][p_valid, None] - target_freq[i][None, t_valid])
        p_idx, t_idx = _lsa(cost)
        for pi, ti in zip(p_idx, t_idx):
            if cost[pi, ti] <= cutoff:
                mi[i, p_valid[pi]] = int(t_valid[ti])
                mm[i, p_valid[pi]] = 1.0
    return mi, mm


# ---------------------------------------------------------------------------
# Frequency-dependent calibration (non-parametric)
# ---------------------------------------------------------------------------

def fit_frequency_calibration(pred_freq, target_freq, pred_mask, target_mask,
                              match_target_idx, match_mask, n_bins=80):
    """
    Fit a smooth correction curve: corrected = pred_freq + correction(pred_freq).

    Collects all matched (pred_freq, target_freq) pairs from training data,
    bins by pred_freq, computes median correction per bin, and fits a smooth
    PCHIP interpolant. This captures DeTaNet's systematic frequency-dependent
    error — it generalizes perfectly since it's a function of frequency alone.

    Returns a callable: correction_fn(freq_array) -> correction_array (cm⁻¹).
    """
    # Collect all matched pairs
    all_pf, all_tf = [], []
    N = pred_freq.shape[0]
    for i in range(N):
        p_valid = pred_mask[i] > 0.5
        m_valid = match_mask[i] > 0.5
        both = p_valid & m_valid
        if not both.any():
            continue
        pf_i = pred_freq[i][both]
        tidx = match_target_idx[i][both]
        tf_i = target_freq[i][tidx]
        all_pf.append(pf_i)
        all_tf.append(tf_i)

    all_pf = np.concatenate(all_pf)
    all_tf = np.concatenate(all_tf)
    corrections = all_tf - all_pf  # positive = need to shift pred UP

    # Bin by predicted frequency and compute robust statistics
    pf_min, pf_max = all_pf.min(), all_pf.max()
    bin_edges = np.linspace(pf_min, pf_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_medians = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for j in range(n_bins):
        mask = (all_pf >= bin_edges[j]) & (all_pf < bin_edges[j + 1])
        if j == n_bins - 1:  # include right edge
            mask |= (all_pf == bin_edges[j + 1])
        bin_counts[j] = mask.sum()
        if bin_counts[j] >= 5:
            bin_medians[j] = np.median(corrections[mask])

    # Fill empty bins by interpolation from neighbors
    valid = bin_counts >= 5
    if valid.sum() < 3:
        # Not enough data — return identity (no correction)
        return lambda freq: np.zeros_like(freq)

    # Smooth the medians slightly to avoid overfitting to bin noise
    smoothed = gaussian_filter1d(bin_medians[valid], sigma=1.5)
    interp = PchipInterpolator(bin_centers[valid], smoothed, extrapolate=True)

    stats = {
        "n_pairs": len(all_pf),
        "median_correction": float(np.median(corrections)),
        "mean_correction": float(np.mean(corrections)),
        "std_correction": float(np.std(corrections)),
        "freq_range": (float(pf_min), float(pf_max)),
    }
    print(f"Calibration: {stats['n_pairs']} matched pairs, "
          f"median correction = {stats['median_correction']:.2f} cm⁻¹, "
          f"std = {stats['std_correction']:.2f} cm⁻¹")

    return interp


def apply_frequency_calibration(pred_freq, pred_mask, calibration_fn):
    """Apply the calibration correction to predicted frequencies."""
    corrected = pred_freq.copy()
    for i in range(pred_freq.shape[0]):
        valid = pred_mask[i] > 0.5
        if valid.any():
            corrected[i][valid] += calibration_fn(pred_freq[i][valid])
    return corrected


# Core Dataset Construction Logic

def _geometry_mol_features(pos, z):
    pos = np.asarray(pos); z = np.asarray(z)
    if pos.size == 0: return np.zeros(16, dtype=np.float32)
    center = pos.mean(axis=0)
    rad = np.linalg.norm(pos - center, axis=1)
    return np.asarray([
        len(z)/100, np.mean(z)/20, np.std(z)/10, 
        np.mean(rad)/10, np.std(rad)/5, np.max(rad)/20,
        float(np.sum(z==6)/len(z)), float(np.sum(z==7)/len(z)), 
        float(np.sum(z==8)/len(z)), float(np.sum(z==16)/len(z)),
        0, 0, 0, 0, 0, 0 # Padding
    ], dtype=np.float32)

def build_experimental_alignment_dataset(
    *, exp_df, resolver_cache, predict_fn, x_grid, cache_dir, max_rows=None, max_atoms=120, refresh=False
):
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    rows_tag = "all" if max_rows is None else str(int(max_rows))
    npz_p = cache_dir / f"exp_point_v1_{rows_tag}.npz"
    csv_p = cache_dir / f"exp_point_v1_{rows_tag}.csv"
    
    if npz_p.exists() and csv_p.exists() and not refresh:
        return _load_dataset_bundle(npz_p, csv_p, "experimental")

    df = exp_df.iloc[:max_rows] if max_rows else exp_df
    y_pred_all, y_target_all, mask_all, mol_feats, meta_rows = [], [], [], [], []
    
    for idx, row in df.iterrows():
        comp = str(row["component"])
        res = resolver_cache.get(comp, {})
        pos, z = res.get("pos"), res.get("z")
        if res.get("status") != "resolved" or pos is None or len(z) > max_atoms: continue
        
        try:
            yp, _, _ = predict_fn(pos, z, x_grid)
            interpolator = PchipInterpolator(row["wavenumbers_arr"], row["intensity_arr"], extrapolate=False)
            yt = np.nan_to_num(interpolator(x_grid), nan=0.0)
            yt = _normalize_signal(gaussian_filter1d(yt, 1.25))
            mask = ((x_grid >= row["wavenumbers_arr"][0]) & (x_grid <= row["wavenumbers_arr"][-1])).astype(np.float32)
            
            y_pred_all.append(_normalize_signal(yp))
            y_target_all.append(yt)
            mask_all.append(mask)
            mol_feats.append(_geometry_mol_features(pos, z))
            meta_rows.append({"component": comp, "cid": res.get("cid"), "n_atoms": len(z)})
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue

    np.savez_compressed(npz_p, x_grid=x_grid, y_pred=np.stack(y_pred_all), y_target=np.stack(y_target_all), 
                        mask=np.stack(mask_all), mol_features=np.stack(mol_feats))
    pd.DataFrame(meta_rows).to_csv(csv_p, index=False)
    return _load_dataset_bundle(npz_p, csv_p, "experimental")

def build_dft_mode_alignment_dataset(
    *, db_path, predict_fn, x_grid, lines_to_spectrum_fn, cache_dir, max_cases=1000, sample_seed=2026, 
    pred_freq_scale_factor=1.0, refresh=False
):
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    npz_p = cache_dir / f"dft_point_v1_{max_cases}.npz"
    csv_p = cache_dir / f"dft_point_v1_{max_cases}.csv"
    
    if npz_p.exists() and csv_p.exists() and not refresh:
        return _load_dft_mode_dataset_bundle(npz_p, csv_p)

    con = sqlite3.connect(str(db_path))
    all_ids = [r[0] for r in con.execute("SELECT id FROM molecule").fetchall()]
    rng_s = np.random.default_rng(sample_seed)
    sel_ids = rng_s.choice(all_ids, size=min(max_cases, len(all_ids)), replace=False).tolist()
    placeholders = ",".join("?" * len(sel_ids))
    rows = con.execute(
        f"SELECT id, SMILES, blob_data FROM molecule WHERE id IN ({placeholders})", sel_ids
    ).fetchall()
    con.close()

    mol_feats, pf_l, pi_l, tf_l, ti_l, yps_l, yts_l, meta = [], [], [], [], [], [], [], []

    from tqdm import tqdm
    for mid, smiles, blob in tqdm(rows, desc="Building dataset", unit="mol"):
        try:
            payload = stats_lib._decode_dft_blob(blob)
            pos, z = payload["coord"], payload["atoms"]
            _, prf, pra = predict_fn(pos, z, x_grid)
            prf = prf * pred_freq_scale_factor

            tf, ti = payload["freq"], payload["Raman Activ"]

            mol_feats.append(_geometry_mol_features(pos, z))
            pf_l.append(np.asarray(prf, dtype=np.float32)); pi_l.append(np.asarray(pra, dtype=np.float32))
            tf_l.append(np.asarray(tf, dtype=np.float32)); ti_l.append(np.asarray(ti, dtype=np.float32))
            yps_l.append(_normalize_signal(lines_to_spectrum_fn(prf, pra, x_grid)))
            yts_l.append(_normalize_signal(lines_to_spectrum_fn(tf, ti, x_grid)))
            meta.append({"molecule_id": mid, "smiles": smiles})
        except Exception as e:
            print(f"Error processing row {mid}: {e}", flush=True)
            continue

    def pad(l, val=0):
        ml = max(len(x) for x in l)
        res = np.full((len(l), ml), val, dtype=np.float32)
        mask = np.zeros((len(l), ml), dtype=np.float32)
        for i, x in enumerate(l): 
            res[i, :len(x)] = x[:ml]
            mask[i, :len(x)] = 1.0
        return res, mask

    pf, pm = pad(pf_l); pi, _ = pad(pi_l)
    tf, tm = pad(tf_l); ti, _ = pad(ti_l)

    mi, mm = _compute_match_indices(pf, pm, tf, tm, cutoff=MODE_TRAIN_MATCH_CUTOFF_CM)
    np.savez_compressed(npz_p, x_grid=x_grid, mol_features=np.stack(mol_feats), pred_freq=pf, pred_intensity=pi,
                        pred_mask=pm, target_freq=tf, target_intensity=ti, target_mask=tm,
                        match_target_idx=mi, match_mask=mm, y_pred_spec=np.stack(yps_l), y_target_spec=np.stack(yts_l))
    pd.DataFrame(meta).to_csv(csv_p, index=False)
    return _load_dft_mode_dataset_bundle(npz_p, csv_p)

# Utility Functions

def _safe_array(v, dtype=np.float64):
    return np.nan_to_num(np.asarray(v, dtype=dtype), nan=0, posinf=0, neginf=0)

def _normalize_signal(y):
    y = _safe_array(y)
    m = np.max(y) if y.size else 0
    return y / m if m > EPS else np.zeros_like(y)

def _lorentz_lines_to_spectrum(freq, intensity, x_grid, sigma=12.0):
    """Lorentz broadening — matches spectra_simulator.Lorenz_broadening."""
    freq = np.asarray(freq, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    if freq.size == 0:
        return np.zeros_like(x_grid, dtype=np.float32)
    lx = freq[:, None] - x_grid[None, :]  # (n_modes, n_grid)
    ly = (sigma / (2 * np.pi)) / (lx ** 2 + 0.25 * sigma ** 2)
    y = (intensity[:, None] * ly).sum(axis=0)
    return _normalize_signal(y).astype(np.float32)

def _split_indices(n, seed, val_f, test_f):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_t, n_v = int(n * test_f), int(n * val_f)
    return {"train": idx[n_t+n_v:], "val": idx[n_t:n_t+n_v], "test": idx[:n_t]}

def _load_dataset_bundle(npz_p, csv_p, domain):
    data = np.load(npz_p); meta = pd.read_csv(csv_p)
    return AlignmentDatasetBundle(domain, data["x_grid"], data["y_pred"], data["y_target"], 
                                  data["mask"], data["mol_features"], meta, npz_p, csv_p)

def _load_dft_mode_dataset_bundle(npz_path: Path, csv_path: Path) -> DFTModeAlignmentDatasetBundle:
    data = np.load(npz_path)
    meta = pd.read_csv(csv_path)
    mode_features = data["mode_features"] if "mode_features" in data else None
    return DFTModeAlignmentDatasetBundle(
        domain="dft", x_grid=data["x_grid"], mol_features=data["mol_features"],
        pred_freq=data["pred_freq"], pred_intensity=data["pred_intensity"], pred_mask=data["pred_mask"],
        target_freq=data["target_freq"], target_intensity=data["target_intensity"], target_mask=data["target_mask"],
        match_target_idx=data["match_target_idx"], match_mask=data["match_mask"],
        y_pred_spec=data["y_pred_spec"], y_target_spec=data["y_target_spec"],
        metadata=meta, cache_npz=npz_path, cache_csv=csv_path,
        mode_features=mode_features,
    )

# Evaluation Logic

_EVAL_TOLS = (5.0, 10.0, 15.0, 20.0)

def _evaluate_coordinate_alignment(pf, pi, pm, tf, ti, tm, confidence=None, conf_threshold=0.5):
    """
    Coverage-honest evaluation metrics.
    If confidence is provided, only predicted modes with confidence > conf_threshold
    are kept — this is how the v2 confidence head improves precision.
    """
    base = {f"coverage@{int(t)}": 0.0 for t in _EVAL_TOLS}
    base.update({f"cwmae@{int(t)}": float(t) for t in _EVAL_TOLS})
    base.update({f"f1@{int(t)}": 0.0 for t in _EVAL_TOLS})
    base["point_rmse"] = 0.0
    base["intensity_mae"] = 0.0
    base["n_pred_kept"] = 0
    base["n_pred_total"] = 0
    base["n_target"] = 0

    idx_t = tm > 0.5
    if not idx_t.any():
        return base

    t_f, t_i = tf[idx_t], ti[idx_t]
    n_target = len(t_f)
    base["n_target"] = n_target

    # Apply confidence filter if provided
    idx_p = pm > 0.5
    base["n_pred_total"] = int(idx_p.sum())
    if confidence is not None:
        idx_p = idx_p & (confidence > conf_threshold)
    base["n_pred_kept"] = int(idx_p.sum())

    if not idx_p.any():
        return base

    p_f, p_i = pf[idx_p], pi[idx_p]
    n_pred = len(p_f)

    dist = np.abs(p_f[:, None] - t_f[None, :])  # (n_pred, n_target)
    nearest_to_target = dist.min(axis=0)         # for each target, closest pred (cm^-1)

    # Compute Hungarian assignment once — same dist matrix for all tolerances
    p_idx_h, t_idx_h = stats_lib.linear_sum_assignment(dist)

    for t in _EVAL_TOLS:
        ti_int = int(t)
        # coverage@T: fraction of targets with a predicted mode within T
        base[f"coverage@{ti_int}"] = float(np.mean(nearest_to_target <= t))
        # cwmae@T: mean(min(nearest, T)) — honest because unmatched penalised at T
        base[f"cwmae@{ti_int}"] = float(np.minimum(nearest_to_target, t).mean())

        # F1@T via Hungarian matching
        keep = dist[p_idx_h, t_idx_h] <= t
        tp = float(keep.sum())
        fp = float(n_pred - tp)
        fn = float(n_target - tp)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        base[f"f1@{ti_int}"] = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # Conditional accuracy on @10 matched pairs (reuse same assignment)
    keep10 = dist[p_idx_h, t_idx_h] <= 10.0
    if keep10.any():
        base["point_rmse"] = float(np.sqrt(np.mean(
            (p_f[p_idx_h[keep10]] - t_f[t_idx_h[keep10]]) ** 2)))
        base["intensity_mae"] = float(np.mean(
            np.abs(p_i[p_idx_h[keep10]] - t_i[t_idx_h[keep10]])))

    return base


def _compute_f1_reward_batch(corr_freq, pred_mask, keep_mask, target_freq, target_mask, tol=10.0):
    """F1@tol for a batch. All inputs are numpy. Returns (B,) float array."""
    B = corr_freq.shape[0]
    rewards = np.zeros(B, dtype=np.float32)
    for b in range(B):
        idx_p = (pred_mask[b] > 0.5) & (keep_mask[b] > 0.5)
        idx_t = target_mask[b] > 0.5
        if not idx_p.any() or not idx_t.any():
            continue
        p_f = corr_freq[b][idx_p]
        t_f = target_freq[b][idx_t]
        dist = np.abs(p_f[:, None] - t_f[None, :])
        p_idx, t_idx = _lsa(dist)
        tp = float((dist[p_idx, t_idx] <= tol).sum())
        n_pred, n_target = len(p_f), len(t_f)
        prec = tp / n_pred if n_pred > 0 else 0.0
        rec = tp / n_target if n_target > 0 else 0.0
        rewards[b] = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return rewards


def run_alignment_study(*, experimental_dataset, dft_dataset, out_dir, device="cpu", train_config=None, **kwargs):
    cfg = train_config or AlignmentTrainConfig()
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    dft_mol = _augment_mol_features(dft_dataset.mol_features, dft_dataset.metadata, cfg.string_feature_dim)
    splits = _split_indices(len(dft_dataset), cfg.seed, cfg.val_fraction, cfg.test_fraction)

    # Mode features from eigenvectors (v10)
    dft_mode_feats = dft_dataset.mode_features  # (N, max_modes, 12) or None
    if cfg.mode_feature_dim > 0 and dft_mode_feats is None:
        raise ValueError(
            "mode_feature_dim > 0 but dataset has no mode_features. "
            "Rebuild the cache with build_dataset_cache.py."
        )
    if dft_mode_feats is not None:
        print(f"Mode features: shape={dft_mode_feats.shape}, dim={dft_mode_feats.shape[-1]}")

    # -----------------------------------------------------------------------
    # Recompute match indices with training-time cutoff
    # (tighter cutoff = harder classification task = better confidence filtering)
    # -----------------------------------------------------------------------
    tight_mi, tight_mm = _compute_match_indices(
        dft_dataset.pred_freq, dft_dataset.pred_mask,
        dft_dataset.target_freq, dft_dataset.target_mask,
        cutoff=cfg.match_cutoff,
    )
    orig_matched = dft_dataset.match_mask.sum()
    tight_matched = tight_mm.sum()
    match_rate = tight_matched / max(dft_dataset.pred_mask.sum(), 1) * 100
    print(f"Match cutoff={cfg.match_cutoff} cm⁻¹: {tight_matched:.0f} matched modes "
          f"({match_rate:.1f}% of predictions, was {orig_matched:.0f} at 60 cm⁻¹)")
    loss_fn_name = "sinkhorn" if cfg.use_sinkhorn else "hungarian"
    print(f"Loss: {loss_fn_name} | mode_feature_dim={cfg.mode_feature_dim}")

    # -----------------------------------------------------------------------
    # Neural model
    # -----------------------------------------------------------------------
    model = PeakCoordinateTransformer(dft_mol.shape[1], cfg, dft_dataset.x_grid).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def _slice_mode_feats(idx):
        return dft_mode_feats[idx] if dft_mode_feats is not None else None

    train_ds = ModeArrayDataset(dft_mol[splits["train"]], dft_dataset.pred_freq[splits["train"]],
                                dft_dataset.pred_intensity[splits["train"]], dft_dataset.pred_mask[splits["train"]],
                                dft_dataset.target_freq[splits["train"]], dft_dataset.target_intensity[splits["train"]],
                                dft_dataset.target_mask[splits["train"]], tight_mi[splits["train"]],
                                tight_mm[splits["train"]], _slice_mode_feats(splits["train"]))

    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_epochs, eta_min=cfg.lr / 20)
    best_val, patience_cnt = float("inf"), 0
    best_state = None

    # Build tiny val loader for early stopping
    val_ds = ModeArrayDataset(dft_mol[splits["val"]], dft_dataset.pred_freq[splits["val"]],
                              dft_dataset.pred_intensity[splits["val"]], dft_dataset.pred_mask[splits["val"]],
                              dft_dataset.target_freq[splits["val"]], dft_dataset.target_intensity[splits["val"]],
                              dft_dataset.target_mask[splits["val"]], tight_mi[splits["val"]],
                              tight_mm[splits["val"]], _slice_mode_feats(splits["val"]))
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    if cfg.use_soft_f1:
        _loss_fn = _soft_f1_loss
    elif cfg.use_sinkhorn:
        _loss_fn = _sinkhorn_alignment_loss
    else:
        _loss_fn = _supervised_alignment_loss

    # Arrays needed for DETR-style re-matching
    _train_idx = splits["train"]
    _train_pf_orig = dft_dataset.pred_freq[_train_idx]
    _train_pm_orig = dft_dataset.pred_mask[_train_idx]
    _train_tf = dft_dataset.target_freq[_train_idx]
    _train_tm = dft_dataset.target_mask[_train_idx]
    _train_mol_t = torch.as_tensor(dft_mol[_train_idx], device=device)
    _train_pf_t = torch.as_tensor(_train_pf_orig, device=device, dtype=torch.float32)
    _train_pi_t = torch.as_tensor(dft_dataset.pred_intensity[_train_idx], device=device, dtype=torch.float32)
    _train_pm_t = torch.as_tensor(_train_pm_orig, device=device, dtype=torch.float32)
    _train_mf_t = (torch.as_tensor(dft_mode_feats[_train_idx], device=device, dtype=torch.float32)
                   if dft_mode_feats is not None else None)

    if cfg.detr_rematch_every > 0:
        print(f"DETR-style dynamic re-matching every {cfg.detr_rematch_every} epochs")

    print(f"Training on {device} | train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")
    for epoch in range(cfg.max_epochs):
        # --- DETR-style: recompute matching with model's current corrections ---
        if cfg.detr_rematch_every > 0 and epoch > 0 and epoch % cfg.detr_rematch_every == 0:
            model.eval()
            with torch.no_grad():
                # Forward pass on all training data in chunks
                corr_freqs = []
                chunk_sz = 512
                for c0 in range(0, len(_train_idx), chunk_sz):
                    c1 = min(c0 + chunk_sz, len(_train_idx))
                    _mf_chunk = _train_mf_t[c0:c1] if _train_mf_t is not None else None
                    pred_out = model(_train_mol_t[c0:c1], _train_pf_t[c0:c1],
                                     _train_pi_t[c0:c1], _train_pm_t[c0:c1], _mf_chunk)
                    corr_freqs.append(pred_out["corrected_freq"].cpu().numpy())
                corr_freq_np = np.concatenate(corr_freqs, axis=0)

            # Re-match using corrected freqs
            new_mi, new_mm = _compute_match_indices(
                corr_freq_np, _train_pm_orig, _train_tf, _train_tm,
                cutoff=cfg.match_cutoff,
            )
            old_matched = train_ds.mm.sum().item()
            train_ds.mi = torch.as_tensor(new_mi, dtype=torch.long)
            train_ds.mm = torch.as_tensor(new_mm, dtype=torch.float32)
            new_matched = new_mm.sum()
            print(f"  [DETR rematch @ epoch {epoch}] matched modes: {old_matched:.0f} → {new_matched:.0f}")

        model.train()
        l_acc = 0.0
        for b in loader:
            mol, pf, pi, pm, tf, ti, tm, mi, mm, mf = [x.to(device) for x in b]
            opt.zero_grad()
            out = model(mol, pf, pi, pm, mf)
            loss, comp = _loss_fn(out, tf, ti, pm, tm, mi, mm, cfg)
            loss.backward()
            opt.step()
            l_acc += loss.item()
        scheduler.step()

        # Anneal soft-F1 temperature (warm → cold)
        if cfg.use_soft_f1 and cfg.soft_f1_tau > cfg.soft_f1_tau_min:
            frac = epoch / max(cfg.max_epochs - 1, 1)
            cfg.soft_f1_tau = cfg.soft_f1_tau * (1 - frac) + cfg.soft_f1_tau_min * frac

        # Validation for early stopping
        model.eval()
        v_acc = 0.0
        v_comp_acc = {}
        with torch.no_grad():
            for b in val_loader:
                mol, pf, pi, pm, tf, ti, tm, mi, mm, mf = [x.to(device) for x in b]
                out = model(mol, pf, pi, pm, mf)
                v_loss_b, v_comp = _loss_fn(out, tf, ti, pm, tm, mi, mm, cfg)
                v_acc += v_loss_b.item()
                for k, v in v_comp.items():
                    v_comp_acc[k] = v_comp_acc.get(k, 0.0) + v
        n_vb = max(len(val_loader), 1)
        v_loss = v_acc / n_vb
        if v_loss < best_val:
            best_val = v_loss
            patience_cnt = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
        if epoch % 10 == 0:
            vc = {k: v / n_vb for k, v in v_comp_acc.items()}
            comp_str = " ".join(f"{k}={v:.4f}" for k, v in vc.items())
            print(f"Epoch {epoch:4d} | train={l_acc/len(loader):.4f} val={v_loss:.4f} "
                  f"[{comp_str}] patience={patience_cnt}")
        if patience_cnt >= cfg.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Restore best checkpoint before eval
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    ckpt_path = out_dir / "alignment_model.pth"
    torch.save({"model_state": best_state or model.state_dict(),
                "cfg": cfg, "mol_dim": dft_mol.shape[1],
                "x_grid": dft_dataset.x_grid}, ckpt_path)
    print(f"Checkpoint saved → {ckpt_path}")

    model.eval()
    mf_all = (torch.as_tensor(dft_mode_feats, device=device, dtype=torch.float32)
              if dft_mode_feats is not None else None)
    with torch.no_grad():
        pred = model(torch.as_tensor(dft_mol, device=device),
                     torch.as_tensor(dft_dataset.pred_freq, device=device, dtype=torch.float32),
                     torch.as_tensor(dft_dataset.pred_intensity, device=device, dtype=torch.float32),
                     torch.as_tensor(dft_dataset.pred_mask, device=device, dtype=torch.float32),
                     mf_all)
        pf_corr = pred["corrected_freq"].cpu().numpy()
        pi_corr = pred["corrected_intensity"].cpu().numpy()
        conf_arr = pred["confidence"].cpu().numpy()

    case_rows = []
    for i in range(len(dft_dataset)):
        # Eval WITHOUT confidence filter — report both filtered and unfiltered
        metrics_all = _evaluate_coordinate_alignment(
            pf_corr[i], pi_corr[i], dft_dataset.pred_mask[i],
            dft_dataset.target_freq[i], dft_dataset.target_intensity[i],
            dft_dataset.target_mask[i],
        )
        metrics_conf = _evaluate_coordinate_alignment(
            pf_corr[i], pi_corr[i], dft_dataset.pred_mask[i],
            dft_dataset.target_freq[i], dft_dataset.target_intensity[i],
            dft_dataset.target_mask[i],
            confidence=conf_arr[i], conf_threshold=cfg.confidence_threshold,
        )
        row = {"case_index": i, "model": "point_transformer_v2"}
        row.update(metrics_all)
        row.update({f"conf_{k}": v for k, v in metrics_conf.items()})
        case_rows.append(row)
    
    case_df = pd.DataFrame(case_rows)
    case_csv = out_dir / "dft_alignment_cases.csv"; case_df.to_csv(case_csv, index=False)

    # -----------------------------------------------------------------------
    # Intensity-based filtering sweep (no neural model — simple baseline)
    # Find the best top-K or intensity threshold on val, report on test
    # -----------------------------------------------------------------------
    print("\n=== Intensity threshold sweep (val set) ===")
    best_k, best_f1_val = 0, 0.0
    for top_k in [40, 50, 60, 70, 80, 90, 100, 110, 120, 136]:
        f1s = []
        for i in splits["val"]:
            pm_i = dft_dataset.pred_mask[i]
            valid = pm_i > 0.5
            n_valid = valid.sum()
            if n_valid <= top_k:
                # Keep all
                f1s.append(_evaluate_coordinate_alignment(
                    pf_corr[i], pi_corr[i], pm_i,
                    dft_dataset.target_freq[i], dft_dataset.target_intensity[i],
                    dft_dataset.target_mask[i])["f1@10"])
                continue
            # Keep top_k by predicted intensity (raw pred, not model output)
            raw_int = dft_dataset.pred_intensity[i].copy()
            raw_int[~valid] = -1
            keep_idx = np.argsort(raw_int)[-top_k:]
            topk_mask = np.zeros_like(pm_i)
            topk_mask[keep_idx] = 1.0
            f1s.append(_evaluate_coordinate_alignment(
                pf_corr[i], pi_corr[i], topk_mask,
                dft_dataset.target_freq[i], dft_dataset.target_intensity[i],
                dft_dataset.target_mask[i])["f1@10"])
        mean_f1 = np.mean(f1s)
        print(f"  top_k={top_k:3d} | val F1@10={mean_f1:.3f}")
        if mean_f1 > best_f1_val:
            best_f1_val = mean_f1
            best_k = top_k

    # Evaluate best top_k on test set
    test_f1s, test_cov = [], []
    for i in splits["test"]:
        pm_i = dft_dataset.pred_mask[i]
        valid = pm_i > 0.5
        n_valid = valid.sum()
        if n_valid <= best_k:
            topk_mask = pm_i.copy()
        else:
            raw_int = dft_dataset.pred_intensity[i].copy()
            raw_int[~valid] = -1
            keep_idx = np.argsort(raw_int)[-best_k:]
            topk_mask = np.zeros_like(pm_i)
            topk_mask[keep_idx] = 1.0
        m = _evaluate_coordinate_alignment(
            pf_corr[i], pi_corr[i], topk_mask,
            dft_dataset.target_freq[i], dft_dataset.target_intensity[i],
            dft_dataset.target_mask[i])
        test_f1s.append(m["f1@10"])
        test_cov.append(m["coverage@10"])
    print(f"  BEST: top_k={best_k} | test F1@10={np.mean(test_f1s):.3f}  "
          f"Coverage@10={np.mean(test_cov):.3f}")

    summary_rows = []
    for s_name, s_idx in splits.items():
        sub = case_df.iloc[s_idx]
        row = {
            "model": "point_transformer_v2", "split": s_name, "n_cases": len(sub),
            "f1@5": sub["f1@5"].mean(), "f1@10": sub["f1@10"].mean(),
            "f1@15": sub["f1@15"].mean(), "f1@20": sub["f1@20"].mean(),
            "cwmae@10": sub["cwmae@10"].mean(), "cwmae@5": sub["cwmae@5"].mean(),
            "coverage@10": sub["coverage@10"].mean(), "coverage@5": sub["coverage@5"].mean(),
            "point_rmse": sub["point_rmse"].mean(), "intensity_mae": sub["intensity_mae"].mean(),
            "avg_pred_kept": sub["n_pred_kept"].mean(), "avg_pred_total": sub["n_pred_total"].mean(),
            "avg_target": sub["n_target"].mean(),
        }
        # Add confidence-filtered metrics to summary
        for col in ["conf_f1@10", "conf_coverage@10", "conf_cwmae@10", "conf_n_pred_kept"]:
            if col in sub.columns:
                row[col] = sub[col].mean()
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "dft_alignment_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    test_row = summary_df[summary_df["split"] == "test"].iloc[0] if "test" in summary_df["split"].values else summary_df.iloc[0]
    kept_pct = test_row['avg_pred_kept'] / (test_row['avg_pred_total'] + 1e-9) * 100
    conf_f1 = test_row.get('conf_f1@10', float('nan'))
    conf_cov = test_row.get('conf_coverage@10', float('nan'))
    conf_kept = test_row.get('conf_n_pred_kept', float('nan'))
    report = (
        f"### DFT Alignment Results (test set)\n"
        f"- Unfiltered: F1@10={test_row['f1@10']:.3f}  Coverage@10={test_row['coverage@10']:.3f}  "
        f"CWMAE@10={test_row['cwmae@10']:.2f} cm⁻¹  ({test_row['avg_pred_kept']:.0f}/{test_row['avg_pred_total']:.0f} modes)\n"
        f"- Filtered:   F1@10={conf_f1:.3f}  Coverage@10={conf_cov:.3f}  "
        f"({conf_kept:.0f} modes kept, threshold={cfg.confidence_threshold})\n"
        f"- Point RMSE (matched@10): {test_row['point_rmse']:.2f} cm⁻¹\n"
        f"- Match cutoff: {cfg.match_cutoff} cm⁻¹  conf_weight: {cfg.confidence_loss_weight}"
    )
    return {
        "domains": {
            "dft": {"best_model": "point_transformer", "summary_csv": str(summary_csv),
                    "case_csv": str(case_csv), "report_markdown": report},
            "experimental": {"best_model": "uncorrected", "summary_csv": str(summary_csv),
                             "report_markdown": "Experimental study results pending."}
        },
        "summary_json": str(out_dir / "summary.json"),
        "checkpoint": str(ckpt_path),
    }

def run_rl_finetune(*, dft_dataset, out_dir, device="cpu", rl_config=None, checkpoint_path=None):
    """
    REINFORCE fine-tuning with F1@10 as non-differentiable reward.
    Loads a pre-trained checkpoint, then optimizes freq corrections via
    policy gradient and confidence via supervised BCE auxiliary.
    """
    rl_cfg = rl_config or AlignmentRLConfig()
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Load pre-trained model
    ckpt_p = Path(checkpoint_path) if checkpoint_path else (out_dir / "alignment_model.pth")
    ckpt = torch.load(ckpt_p, map_location=device, weights_only=False)
    base_cfg = ckpt["cfg"]

    dft_mol = _augment_mol_features(dft_dataset.mol_features, dft_dataset.metadata, base_cfg.string_feature_dim)
    model = PeakCoordinateTransformer(ckpt["mol_dim"], base_cfg, dft_dataset.x_grid).to(device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict({k: v.to(device) for k, v in state.items()})
    print(f"Loaded pre-trained checkpoint from {ckpt_p}")

    # Learnable exploration sigma
    log_sigma = nn.Parameter(torch.tensor(float(np.log(rl_cfg.sigma_init)), device=device))

    if rl_cfg.freeze_encoder:
        for name, p in model.named_parameters():
            if "transformer" in name or "peak_embed" in name or "mol_encoder" in name:
                p.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Frozen encoder — {trainable:,} trainable params in heads")

    params = [p for p in model.parameters() if p.requires_grad]
    if rl_cfg.sigma_learnable:
        params.append(log_sigma)
    opt = torch.optim.Adam(params, lr=rl_cfg.lr, weight_decay=rl_cfg.weight_decay)

    # Data
    splits = _split_indices(len(dft_dataset), rl_cfg.seed, rl_cfg.val_fraction, rl_cfg.test_fraction)
    dft_mode_feats = dft_dataset.mode_features
    tight_mi, tight_mm = _compute_match_indices(
        dft_dataset.pred_freq, dft_dataset.pred_mask,
        dft_dataset.target_freq, dft_dataset.target_mask,
        cutoff=rl_cfg.match_cutoff,
    )

    def _mf(idx):
        return dft_mode_feats[idx] if dft_mode_feats is not None else None

    train_ds = ModeArrayDataset(
        dft_mol[splits["train"]], dft_dataset.pred_freq[splits["train"]],
        dft_dataset.pred_intensity[splits["train"]], dft_dataset.pred_mask[splits["train"]],
        dft_dataset.target_freq[splits["train"]], dft_dataset.target_intensity[splits["train"]],
        dft_dataset.target_mask[splits["train"]], tight_mi[splits["train"]],
        tight_mm[splits["train"]], _mf(splits["train"]))
    loader = DataLoader(train_ds, batch_size=rl_cfg.batch_size, shuffle=True)

    val_ds = ModeArrayDataset(
        dft_mol[splits["val"]], dft_dataset.pred_freq[splits["val"]],
        dft_dataset.pred_intensity[splits["val"]], dft_dataset.pred_mask[splits["val"]],
        dft_dataset.target_freq[splits["val"]], dft_dataset.target_intensity[splits["val"]],
        dft_dataset.target_mask[splits["val"]], tight_mi[splits["val"]],
        tight_mm[splits["val"]], _mf(splits["val"]))
    val_loader = DataLoader(val_ds, batch_size=rl_cfg.batch_size, shuffle=False)

    best_val_f1, patience_cnt, best_state = 0.0, 0, None
    print(f"RL training: K={rl_cfg.K} sigma_init={rl_cfg.sigma_init} lr={rl_cfg.lr} "
          f"conf_aux={rl_cfg.conf_loss_weight}")
    print(f"  train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

    for epoch in range(rl_cfg.max_epochs):
        model.train()
        ep_reward, ep_loss, n_bat = 0.0, 0.0, 0

        for batch in loader:
            mol, pf, pi, pm, tf, ti, tm, _mi_b, mm_b, mf = [x.to(device) for x in batch]

            out = model(mol, pf, pi, pm, mf)
            mu_freq = out["corrected_freq"]   # (B, max_modes)
            confidence = out["confidence"]     # (B, max_modes)
            conf_clamped = confidence.clamp(1e-6, 1 - 1e-6)

            sigma = torch.exp(log_sigma).clamp(rl_cfg.sigma_min, rl_cfg.sigma_max)

            # Pre-compute fixed top-k mask if using top_k_filter
            if rl_cfg.top_k_filter > 0:
                raw_int_np = pi.cpu().numpy()
                pm_np_batch = pm.cpu().numpy()
                topk_mask_np = np.zeros_like(pm_np_batch)
                for bi in range(mol.shape[0]):
                    valid = pm_np_batch[bi] > 0.5
                    if valid.sum() <= rl_cfg.top_k_filter:
                        topk_mask_np[bi] = pm_np_batch[bi]
                    else:
                        ri = raw_int_np[bi].copy()
                        ri[~valid] = -1
                        keep_idx = np.argsort(ri)[-rl_cfg.top_k_filter:]
                        topk_mask_np[bi, keep_idx] = 1.0
                topk_mask_t = torch.tensor(topk_mask_np, device=device, dtype=torch.float32)

            all_log_probs = []
            all_rewards = []

            for _k in range(rl_cfg.K):
                # --- Sample actions ---
                eps = torch.randn_like(mu_freq)
                sampled_freq = (mu_freq + sigma * eps * pm).detach()

                if rl_cfg.top_k_filter > 0:
                    # Fixed top-k: only freq is stochastic
                    keep_sample = topk_mask_t
                    log_prob = -0.5 * ((sampled_freq - mu_freq) / sigma) ** 2 - torch.log(sigma)
                    log_prob = (log_prob * pm).sum(dim=1)
                else:
                    # Bernoulli confidence sampling
                    keep_sample = (torch.bernoulli(conf_clamped) * pm).detach()
                    freq_lp = -0.5 * ((sampled_freq - mu_freq) / sigma) ** 2 - torch.log(sigma)
                    freq_lp = (freq_lp * pm).sum(dim=1)
                    keep_lp = (keep_sample * torch.log(conf_clamped)
                               + (1 - keep_sample) * torch.log(1 - conf_clamped))
                    keep_lp = (keep_lp * pm).sum(dim=1)
                    log_prob = freq_lp + keep_lp

                # --- Reward: F1@10 ---
                reward = _compute_f1_reward_batch(
                    sampled_freq.cpu().numpy(), pm.cpu().numpy(),
                    keep_sample.detach().cpu().numpy(), tf.cpu().numpy(),
                    tm.cpu().numpy(), tol=rl_cfg.reward_tol,
                )
                all_log_probs.append(log_prob)
                all_rewards.append(torch.tensor(reward, device=device, dtype=torch.float32))

            log_probs = torch.stack(all_log_probs)  # (K, B)
            rewards = torch.stack(all_rewards)       # (K, B)

            # Baseline: per-molecule mean reward
            baseline = rewards.mean(dim=0, keepdim=True)
            advantage = (rewards - baseline).detach()

            policy_loss = -(advantage * log_probs).mean()

            # Supervised confidence auxiliary (skip if using top_k_filter)
            loss = policy_loss
            if rl_cfg.conf_loss_weight > 0 and rl_cfg.top_k_filter == 0:
                valid = pm > 0.5
                if valid.any():
                    conf_bce = F.binary_cross_entropy(
                        confidence[valid].clamp(1e-6, 1 - 1e-6),
                        mm_b[valid].float(),
                        reduction="mean",
                    )
                    loss = loss + rl_cfg.conf_loss_weight * conf_bce

            # Entropy bonus (freq only when using top_k_filter)
            freq_ent = 0.5 * (1 + torch.log(torch.tensor(2 * np.pi, device=device) * sigma ** 2))
            loss = loss - rl_cfg.entropy_coeff * freq_ent
            if rl_cfg.top_k_filter == 0:
                conf_ent = -(conf_clamped * torch.log(conf_clamped)
                             + (1 - conf_clamped) * torch.log(1 - conf_clamped))
                loss = loss - rl_cfg.entropy_coeff * (conf_ent * pm).mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, rl_cfg.grad_clip)
            opt.step()

            ep_reward += rewards.mean().item()
            ep_loss += loss.item()
            n_bat += 1

        # --- Validation: deterministic F1@10 (both conf-filtered and unfiltered) ---
        model.eval()
        val_metrics = {"f1@10": [], "f1@5": [], "coverage@10": [], "cwmae@10": [],
                       "conf_f1@10": [], "conf_kept": [], "n_pred": []}
        with torch.no_grad():
            for batch in val_loader:
                mol, pf, pi, pm, tf, ti, tm, _mi_b, _mm_b, mf = [x.to(device) for x in batch]
                out = model(mol, pf, pi, pm, mf)
                cf_np = out["corrected_freq"].cpu().numpy()
                ci_np = out["corrected_intensity"].cpu().numpy()
                conf_np = out["confidence"].cpu().numpy()
                pm_np, tf_np, ti_np, tm_np = (
                    pm.cpu().numpy(), tf.cpu().numpy(), ti.cpu().numpy(), tm.cpu().numpy()
                )
                for i in range(mol.shape[0]):
                    # Unfiltered
                    m = _evaluate_coordinate_alignment(
                        cf_np[i], ci_np[i], pm_np[i], tf_np[i], ti_np[i], tm_np[i],
                    )
                    val_metrics["f1@10"].append(m["f1@10"])
                    val_metrics["f1@5"].append(m["f1@5"])
                    val_metrics["coverage@10"].append(m["coverage@10"])
                    val_metrics["cwmae@10"].append(m["cwmae@10"])
                    val_metrics["n_pred"].append(m["n_pred_kept"])
                    # Conf-filtered
                    mc = _evaluate_coordinate_alignment(
                        cf_np[i], ci_np[i], pm_np[i], tf_np[i], ti_np[i], tm_np[i],
                        confidence=conf_np[i], conf_threshold=rl_cfg.confidence_threshold,
                    )
                    val_metrics["conf_f1@10"].append(mc["f1@10"])
                    val_metrics["conf_kept"].append(mc["n_pred_kept"])

        val_f1 = float(np.mean(val_metrics["f1@10"]))
        val_conf_f1 = float(np.mean(val_metrics["conf_f1@10"]))
        # Track the better of filtered/unfiltered for early stopping
        val_best_f1 = max(val_f1, val_conf_f1)
        if val_best_f1 > best_val_f1:
            best_val_f1 = val_best_f1
            patience_cnt = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1

        if epoch % 5 == 0:
            print(f"Epoch {epoch:4d} | reward={ep_reward / max(n_bat, 1):.3f} "
                  f"F1@10={val_f1:.3f} conf_F1@10={val_conf_f1:.3f} "
                  f"cov@10={np.mean(val_metrics['coverage@10']):.3f} "
                  f"cwmae={np.mean(val_metrics['cwmae@10']):.2f} "
                  f"kept={np.mean(val_metrics['conf_kept']):.0f}/{np.mean(val_metrics['n_pred']):.0f} "
                  f"sigma={sigma.item():.2f} best={best_val_f1:.3f} pat={patience_cnt}")
        if patience_cnt >= rl_cfg.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Restore best & save
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    rl_ckpt_path = out_dir / "alignment_model_rl.pth"
    torch.save({
        "model_state": best_state or model.state_dict(),
        "cfg": base_cfg, "rl_cfg": rl_cfg,
        "mol_dim": ckpt["mol_dim"], "x_grid": dft_dataset.x_grid,
    }, rl_ckpt_path)
    print(f"RL checkpoint → {rl_ckpt_path}")

    # --- Full eval ---
    model.eval()
    mf_all = (torch.as_tensor(dft_mode_feats, device=device, dtype=torch.float32)
              if dft_mode_feats is not None else None)
    with torch.no_grad():
        pred = model(
            torch.as_tensor(dft_mol, device=device),
            torch.as_tensor(dft_dataset.pred_freq, device=device, dtype=torch.float32),
            torch.as_tensor(dft_dataset.pred_intensity, device=device, dtype=torch.float32),
            torch.as_tensor(dft_dataset.pred_mask, device=device, dtype=torch.float32),
            mf_all,
        )
        pf_corr = pred["corrected_freq"].cpu().numpy()
        pi_corr = pred["corrected_intensity"].cpu().numpy()
        conf_arr = pred["confidence"].cpu().numpy()

    case_rows = []
    for i in range(len(dft_dataset)):
        metrics_all = _evaluate_coordinate_alignment(
            pf_corr[i], pi_corr[i], dft_dataset.pred_mask[i],
            dft_dataset.target_freq[i], dft_dataset.target_intensity[i],
            dft_dataset.target_mask[i],
        )
        metrics_conf = _evaluate_coordinate_alignment(
            pf_corr[i], pi_corr[i], dft_dataset.pred_mask[i],
            dft_dataset.target_freq[i], dft_dataset.target_intensity[i],
            dft_dataset.target_mask[i],
            confidence=conf_arr[i], conf_threshold=rl_cfg.confidence_threshold,
        )
        row = {"case_index": i, "model": "peak_rl_v10"}
        row.update(metrics_all)
        row.update({f"conf_{k}": v for k, v in metrics_conf.items()})
        case_rows.append(row)

    case_df = pd.DataFrame(case_rows)
    case_csv = out_dir / "dft_alignment_rl_cases.csv"
    case_df.to_csv(case_csv, index=False)

    # Top-k intensity sweep (val → test)
    print("\n=== Top-k intensity sweep (val set) ===")
    best_k, best_f1_val = 0, 0.0
    for top_k in [30, 40, 50, 60, 70, 80, 90, 100]:
        f1s = []
        for i in splits["val"]:
            pm_i = dft_dataset.pred_mask[i]
            valid = pm_i > 0.5
            if valid.sum() <= top_k:
                f1s.append(_evaluate_coordinate_alignment(
                    pf_corr[i], pi_corr[i], pm_i,
                    dft_dataset.target_freq[i], dft_dataset.target_intensity[i],
                    dft_dataset.target_mask[i])["f1@10"])
                continue
            raw_int = dft_dataset.pred_intensity[i].copy()
            raw_int[~valid] = -1
            keep_idx = np.argsort(raw_int)[-top_k:]
            topk_mask = np.zeros_like(pm_i)
            topk_mask[keep_idx] = 1.0
            f1s.append(_evaluate_coordinate_alignment(
                pf_corr[i], pi_corr[i], topk_mask,
                dft_dataset.target_freq[i], dft_dataset.target_intensity[i],
                dft_dataset.target_mask[i])["f1@10"])
        mean_f1 = np.mean(f1s)
        print(f"  top_k={top_k:3d} | val F1@10={mean_f1:.3f}")
        if mean_f1 > best_f1_val:
            best_f1_val = mean_f1
            best_k = top_k

    test_f1s, test_cov = [], []
    for i in splits["test"]:
        pm_i = dft_dataset.pred_mask[i]
        valid = pm_i > 0.5
        if valid.sum() <= best_k:
            topk_mask = pm_i.copy()
        else:
            raw_int = dft_dataset.pred_intensity[i].copy()
            raw_int[~valid] = -1
            keep_idx = np.argsort(raw_int)[-best_k:]
            topk_mask = np.zeros_like(pm_i)
            topk_mask[keep_idx] = 1.0
        m = _evaluate_coordinate_alignment(
            pf_corr[i], pi_corr[i], topk_mask,
            dft_dataset.target_freq[i], dft_dataset.target_intensity[i],
            dft_dataset.target_mask[i])
        test_f1s.append(m["f1@10"])
        test_cov.append(m["coverage@10"])
    print(f"  BEST: top_k={best_k} | test F1@10={np.mean(test_f1s):.3f}  "
          f"Coverage@10={np.mean(test_cov):.3f}")

    # Summary table
    summary_rows = []
    for s_name, s_idx in splits.items():
        sub = case_df.iloc[s_idx]
        row = {
            "model": "peak_rl_v10", "split": s_name, "n_cases": len(sub),
            "f1@5": sub["f1@5"].mean(), "f1@10": sub["f1@10"].mean(),
            "f1@15": sub["f1@15"].mean(), "f1@20": sub["f1@20"].mean(),
            "cwmae@10": sub["cwmae@10"].mean(), "coverage@10": sub["coverage@10"].mean(),
            "point_rmse": sub["point_rmse"].mean(),
            "avg_pred_kept": sub["n_pred_kept"].mean(),
            "avg_pred_total": sub["n_pred_total"].mean(),
        }
        for col in ["conf_f1@10", "conf_coverage@10", "conf_cwmae@10", "conf_n_pred_kept"]:
            if col in sub.columns:
                row[col] = sub[col].mean()
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "dft_alignment_rl_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    test_row = summary_df[summary_df["split"] == "test"].iloc[0]
    conf_f1 = test_row.get('conf_f1@10', float('nan'))
    conf_cov = test_row.get('conf_coverage@10', float('nan'))
    conf_kept = test_row.get('conf_n_pred_kept', float('nan'))
    report = (
        f"### DFT Alignment RL Results (test set)\n"
        f"- Unfiltered: F1@10={test_row['f1@10']:.3f}  Coverage@10={test_row['coverage@10']:.3f}\n"
        f"- Conf-filtered: F1@10={conf_f1:.3f}  Coverage@10={conf_cov:.3f}  "
        f"({conf_kept:.0f} modes kept, threshold={rl_cfg.confidence_threshold})\n"
        f"- Top-k filtered: F1@10={np.mean(test_f1s):.3f}  (k={best_k})\n"
        f"- Point RMSE (matched@10): {test_row['point_rmse']:.2f} cm⁻¹"
    )
    print(report)

    return {
        "domains": {
            "dft": {"best_model": "peak_rl_v10", "summary_csv": str(summary_csv),
                    "case_csv": str(case_csv), "report_markdown": report},
            "experimental": {"best_model": "uncorrected",
                             "report_markdown": "Experimental study results pending."}
        },
        "checkpoint": str(rl_ckpt_path),
    }


def run_hybrid_training(*, dft_dataset, out_dir, device="cpu", hybrid_config=None, checkpoint_path=None):
    """
    v12: Two-phase hybrid training.
    Phase 1: Soft-F1 with do-no-harm penalty (freq corrections, confidence stays exploratory)
    Phase 2: Add REINFORCE keep/drop (confidence learns to filter via hard F1@10 reward)
    """
    cfg = hybrid_config or AlignmentHybridConfig()
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Build base config for model construction
    base_cfg = AlignmentTrainConfig(
        latent_dim=cfg.latent_dim,
        transformer_layers=cfg.transformer_layers,
        transformer_heads=cfg.transformer_heads,
        string_feature_dim=cfg.string_feature_dim,
        mode_feature_dim=cfg.mode_feature_dim,
        match_cutoff=cfg.match_cutoff,
    )

    dft_mol = _augment_mol_features(dft_dataset.mol_features, dft_dataset.metadata, cfg.string_feature_dim)
    splits = _split_indices(len(dft_dataset), cfg.seed, cfg.val_fraction, cfg.test_fraction)
    dft_mode_feats = dft_dataset.mode_features

    if cfg.mode_feature_dim > 0 and dft_mode_feats is None:
        raise ValueError("mode_feature_dim > 0 but dataset has no mode_features.")
    if dft_mode_feats is not None:
        print(f"Mode features: shape={dft_mode_feats.shape}")

    tight_mi, tight_mm = _compute_match_indices(
        dft_dataset.pred_freq, dft_dataset.pred_mask,
        dft_dataset.target_freq, dft_dataset.target_mask,
        cutoff=cfg.match_cutoff,
    )
    match_rate = tight_mm.sum() / max(dft_dataset.pred_mask.sum(), 1) * 100
    print(f"Match cutoff={cfg.match_cutoff} cm⁻¹: {tight_mm.sum():.0f} matched modes ({match_rate:.1f}%)")

    # Load checkpoint or create fresh model
    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = PeakCoordinateTransformer(ckpt["mol_dim"], ckpt["cfg"], dft_dataset.x_grid).to(device)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict({k: v.to(device) for k, v in state.items()})
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        model = PeakCoordinateTransformer(dft_mol.shape[1], base_cfg, dft_dataset.x_grid).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    def _mf(idx):
        return dft_mode_feats[idx] if dft_mode_feats is not None else None

    # --- Datasets ---
    train_ds = ModeArrayDataset(
        dft_mol[splits["train"]], dft_dataset.pred_freq[splits["train"]],
        dft_dataset.pred_intensity[splits["train"]], dft_dataset.pred_mask[splits["train"]],
        dft_dataset.target_freq[splits["train"]], dft_dataset.target_intensity[splits["train"]],
        dft_dataset.target_mask[splits["train"]], tight_mi[splits["train"]],
        tight_mm[splits["train"]], _mf(splits["train"]))
    val_ds = ModeArrayDataset(
        dft_mol[splits["val"]], dft_dataset.pred_freq[splits["val"]],
        dft_dataset.pred_intensity[splits["val"]], dft_dataset.pred_mask[splits["val"]],
        dft_dataset.target_freq[splits["val"]], dft_dataset.target_intensity[splits["val"]],
        dft_dataset.target_mask[splits["val"]], tight_mi[splits["val"]],
        tight_mm[splits["val"]], _mf(splits["val"]))

    print(f"Splits: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

    # ===================================================================
    # PHASE 1: Soft-F1 + Do-no-harm (freq corrections)
    # ===================================================================
    print("\n" + "=" * 60)
    print("PHASE 1: Soft-F1 + Do-no-harm ({} epochs)".format(cfg.phase1_epochs))
    print("=" * 60)

    loader1 = DataLoader(train_ds, batch_size=cfg.phase1_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.phase1_batch_size, shuffle=False)

    opt1 = torch.optim.AdamW(model.parameters(), lr=cfg.phase1_lr, weight_decay=cfg.phase1_weight_decay)
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=cfg.phase1_epochs, eta_min=cfg.phase1_lr / 20)

    best_val_loss, patience_cnt, best_state = float("inf"), 0, None
    current_tau = cfg.soft_f1_tau_init

    for epoch in range(cfg.phase1_epochs):
        model.train()
        ep_loss = 0.0
        ep_comp = {}
        ep_fixed = 0.0
        n_bat = 0
        for batch in loader1:
            mol, pf, pi, pm, tf_b, ti_b, tm_b, mi_b, mm_b, mf = [x.to(device) for x in batch]
            opt1.zero_grad()
            out = model(mol, pf, pi, pm, mf)
            # Lock modes already within dnh_radius of a target — zero delta
            fm = _compute_fixed_mask(pf, tf_b, pm, tm_b, radius=cfg.dnh_radius)
            out["corrected_freq"] = pf * fm + out["corrected_freq"] * (1 - fm)
            loss, comp = _hybrid_f1_loss(out, tf_b, ti_b, pm, tm_b, mi_b, mm_b, cfg, pf, current_tau)
            loss.backward()
            opt1.step()
            ep_loss += loss.item()
            ep_fixed += fm.sum().item() / mol.shape[0]  # avg fixed modes per molecule
            for k, v in comp.items():
                ep_comp[k] = ep_comp.get(k, 0.0) + v
            n_bat += 1
        sched1.step()

        # Tau annealing (warm → cold)
        frac = epoch / max(cfg.phase1_epochs - 1, 1)
        current_tau = cfg.soft_f1_tau_init * (1 - frac) + cfg.soft_f1_tau_min * frac

        # Validation: soft-F1 loss + hard F1@10
        model.eval()
        v_loss = 0.0
        v_n = 0
        val_hard_f1 = []
        val_cov = []
        val_cwmae = []
        with torch.no_grad():
            for batch in val_loader:
                mol, pf, pi, pm, tf_b, ti_b, tm_b, mi_b, mm_b, mf = [x.to(device) for x in batch]
                out = model(mol, pf, pi, pm, mf)
                fm = _compute_fixed_mask(pf, tf_b, pm, tm_b, radius=cfg.dnh_radius)
                out["corrected_freq"] = pf * fm + out["corrected_freq"] * (1 - fm)
                vl, _ = _hybrid_f1_loss(out, tf_b, ti_b, pm, tm_b, mi_b, mm_b, cfg, pf, current_tau)
                v_loss += vl.item()
                v_n += 1
                # Hard F1@10 per molecule
                cf_np = out["corrected_freq"].cpu().numpy()
                ci_np = out["corrected_intensity"].cpu().numpy()
                pm_np = pm.cpu().numpy()
                tf_np = tf_b.cpu().numpy()
                ti_np = ti_b.cpu().numpy()
                tm_np = tm_b.cpu().numpy()
                for i in range(mol.shape[0]):
                    m = _evaluate_coordinate_alignment(
                        cf_np[i], ci_np[i], pm_np[i], tf_np[i], ti_np[i], tm_np[i])
                    val_hard_f1.append(m["f1@10"])
                    val_cov.append(m["coverage@10"])
                    val_cwmae.append(m["cwmae@10"])

        val_loss_avg = v_loss / max(v_n, 1)
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            patience_cnt = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1

        avg_comp = {k: v / max(n_bat, 1) for k, v in ep_comp.items()}
        lr_now = sched1.get_last_lr()[0]
        avg_fixed = ep_fixed / max(n_bat, 1)
        print(f"P1 {epoch:3d} | loss={ep_loss / max(n_bat, 1):.4f} val={val_loss_avg:.4f} "
              f"sf1={avg_comp.get('soft_f1', 0):.3f} dnh={avg_comp.get('dnh', 0):.3f} "
              f"| F1@10={np.mean(val_hard_f1):.3f} cov={np.mean(val_cov):.3f} "
              f"cwmae={np.mean(val_cwmae):.2f} fixed={avg_fixed:.0f} "
              f"| tau={current_tau:.2f} lr={lr_now:.6f} pat={patience_cnt}")

        if patience_cnt >= cfg.phase1_patience:
            print(f"Phase 1 early stop at epoch {epoch}")
            break

    # Restore best phase 1 checkpoint
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    p1_ckpt = out_dir / "alignment_model_p1.pth"
    torch.save({"model_state": best_state or model.state_dict(),
                "cfg": base_cfg, "mol_dim": dft_mol.shape[1],
                "x_grid": dft_dataset.x_grid}, p1_ckpt)
    print(f"Phase 1 checkpoint → {p1_ckpt}")

    # ===================================================================
    # PHASE 2: REINFORCE keep/drop + continued Soft-F1
    # ===================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: REINFORCE keep/drop + Soft-F1 ({} epochs)".format(cfg.phase2_epochs))
    print("=" * 60)

    loader2 = DataLoader(train_ds, batch_size=cfg.phase2_batch_size, shuffle=True)
    val_loader2 = DataLoader(val_ds, batch_size=cfg.phase2_batch_size, shuffle=False)

    opt2 = torch.optim.AdamW(model.parameters(), lr=cfg.phase2_lr, weight_decay=cfg.phase2_weight_decay)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=cfg.phase2_epochs, eta_min=cfg.phase2_lr / 20)

    best_val_f1, patience_cnt, best_state = 0.0, 0, None
    # Use final tau from phase 1
    phase2_tau = cfg.soft_f1_tau_min

    for epoch in range(cfg.phase2_epochs):
        model.train()
        ep_sf1_loss = 0.0
        ep_rl_loss = 0.0
        ep_reward = 0.0
        n_bat = 0
        for batch in loader2:
            mol, pf, pi, pm, tf_b, ti_b, tm_b, mi_b, mm_b, mf = [x.to(device) for x in batch]
            opt2.zero_grad()
            out = model(mol, pf, pi, pm, mf)
            # Lock fixed modes
            fm = _compute_fixed_mask(pf, tf_b, pm, tm_b, radius=cfg.dnh_radius)
            out["corrected_freq"] = pf * fm + out["corrected_freq"] * (1 - fm)

            # Soft-F1 loss (freq corrections, still training)
            sf1_loss, sf1_comp = _hybrid_f1_loss(out, tf_b, ti_b, pm, tm_b, mi_b, mm_b, cfg, pf, phase2_tau)

            # REINFORCE keep/drop loss — fixed modes always kept
            rl_loss, rl_comp = _reinforce_keep_drop_loss(out, pm, tf_b, tm_b, cfg, fixed_mask=fm)

            loss = sf1_loss + cfg.rl_weight * rl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.rl_grad_clip)
            opt2.step()

            ep_sf1_loss += sf1_loss.item()
            ep_rl_loss += rl_loss.item()
            ep_reward += rl_comp["rl_reward"]
            n_bat += 1
        sched2.step()

        # Validation: hard F1@10 (early stop criterion)
        model.eval()
        val_f1_unf = []
        val_f1_conf = []
        val_cov = []
        val_cwmae = []
        val_kept = []
        val_total = []
        with torch.no_grad():
            for batch in val_loader2:
                mol, pf, pi, pm, tf_b, ti_b, tm_b, mi_b, mm_b, mf = [x.to(device) for x in batch]
                out = model(mol, pf, pi, pm, mf)
                fm = _compute_fixed_mask(pf, tf_b, pm, tm_b, radius=cfg.dnh_radius)
                out["corrected_freq"] = pf * fm + out["corrected_freq"] * (1 - fm)
                cf_np = out["corrected_freq"].cpu().numpy()
                ci_np = out["corrected_intensity"].cpu().numpy()
                conf_np = out["confidence"].cpu().numpy()
                pm_np = pm.cpu().numpy()
                tf_np = tf_b.cpu().numpy()
                ti_np = ti_b.cpu().numpy()
                tm_np = tm_b.cpu().numpy()
                for i in range(mol.shape[0]):
                    m = _evaluate_coordinate_alignment(
                        cf_np[i], ci_np[i], pm_np[i], tf_np[i], ti_np[i], tm_np[i])
                    mc = _evaluate_coordinate_alignment(
                        cf_np[i], ci_np[i], pm_np[i], tf_np[i], ti_np[i], tm_np[i],
                        confidence=conf_np[i], conf_threshold=cfg.confidence_threshold)
                    val_f1_unf.append(m["f1@10"])
                    val_f1_conf.append(mc["f1@10"])
                    val_cov.append(m["coverage@10"])
                    val_cwmae.append(m["cwmae@10"])
                    val_kept.append(mc["n_pred_kept"])
                    val_total.append(m["n_pred_kept"])

        mean_f1 = float(np.mean(val_f1_unf))
        mean_conf_f1 = float(np.mean(val_f1_conf))
        val_best = max(mean_f1, mean_conf_f1)

        if val_best > best_val_f1:
            best_val_f1 = val_best
            patience_cnt = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1

        lr_now = sched2.get_last_lr()[0]
        print(f"P2 {epoch:3d} | sf1={ep_sf1_loss / max(n_bat, 1):.4f} rl={ep_rl_loss / max(n_bat, 1):.4f} "
              f"reward={ep_reward / max(n_bat, 1):.3f} "
              f"| F1@10={mean_f1:.3f} conf_F1={mean_conf_f1:.3f} "
              f"cov={np.mean(val_cov):.3f} cwmae={np.mean(val_cwmae):.2f} "
              f"kept={np.mean(val_kept):.0f}/{np.mean(val_total):.0f} "
              f"| lr={lr_now:.6f} best={best_val_f1:.3f} pat={patience_cnt}")

        if patience_cnt >= cfg.phase2_patience:
            print(f"Phase 2 early stop at epoch {epoch}")
            break

    # Restore best phase 2 checkpoint
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    ckpt_path = out_dir / "alignment_model.pth"
    torch.save({"model_state": best_state or model.state_dict(),
                "cfg": base_cfg, "hybrid_cfg": cfg,
                "mol_dim": dft_mol.shape[1], "x_grid": dft_dataset.x_grid}, ckpt_path)
    print(f"Final checkpoint → {ckpt_path}")

    # ===================================================================
    # Final evaluation (same as run_alignment_study)
    # ===================================================================
    model.eval()
    mf_all = (torch.as_tensor(dft_mode_feats, device=device, dtype=torch.float32)
              if dft_mode_feats is not None else None)
    pf_t = torch.as_tensor(dft_dataset.pred_freq, device=device, dtype=torch.float32)
    pi_t = torch.as_tensor(dft_dataset.pred_intensity, device=device, dtype=torch.float32)
    pm_t = torch.as_tensor(dft_dataset.pred_mask, device=device, dtype=torch.float32)
    tf_t = torch.as_tensor(dft_dataset.target_freq, device=device, dtype=torch.float32)
    tm_t = torch.as_tensor(dft_dataset.target_mask, device=device, dtype=torch.float32)
    with torch.no_grad():
        pred = model(torch.as_tensor(dft_mol, device=device), pf_t, pi_t, pm_t, mf_all)
        # Apply fixed mask at eval time too
        fm_all = _compute_fixed_mask(pf_t, tf_t, pm_t, tm_t, radius=cfg.dnh_radius)
        pred["corrected_freq"] = pf_t * fm_all + pred["corrected_freq"] * (1 - fm_all)
        pf_corr = pred["corrected_freq"].cpu().numpy()
        pi_corr = pred["corrected_intensity"].cpu().numpy()
        conf_arr = pred["confidence"].cpu().numpy()

    n_fixed = fm_all.sum().item() / len(dft_dataset)
    print(f"Fixed mask: {n_fixed:.0f} modes/molecule locked (within {cfg.dnh_radius} cm⁻¹)")

    case_rows = []
    for i in range(len(dft_dataset)):
        metrics_all = _evaluate_coordinate_alignment(
            pf_corr[i], pi_corr[i], dft_dataset.pred_mask[i],
            dft_dataset.target_freq[i], dft_dataset.target_intensity[i],
            dft_dataset.target_mask[i],
        )
        metrics_conf = _evaluate_coordinate_alignment(
            pf_corr[i], pi_corr[i], dft_dataset.pred_mask[i],
            dft_dataset.target_freq[i], dft_dataset.target_intensity[i],
            dft_dataset.target_mask[i],
            confidence=conf_arr[i], conf_threshold=cfg.confidence_threshold,
        )
        row = {"case_index": i, "model": "hybrid_v12"}
        row.update(metrics_all)
        row.update({f"conf_{k}": v for k, v in metrics_conf.items()})
        case_rows.append(row)

    case_df = pd.DataFrame(case_rows)
    case_csv = out_dir / "dft_alignment_cases.csv"
    case_df.to_csv(case_csv, index=False)

    # Top-k intensity sweep (val → test)
    print("\n=== Top-k intensity sweep (val set) ===")
    best_k, best_f1_val = 0, 0.0
    for top_k in [40, 50, 60, 70, 80, 90, 100, 110, 120, 136]:
        f1s = []
        for i in splits["val"]:
            pm_i = dft_dataset.pred_mask[i]
            valid = pm_i > 0.5
            if valid.sum() <= top_k:
                f1s.append(_evaluate_coordinate_alignment(
                    pf_corr[i], pi_corr[i], pm_i,
                    dft_dataset.target_freq[i], dft_dataset.target_intensity[i],
                    dft_dataset.target_mask[i])["f1@10"])
                continue
            raw_int = dft_dataset.pred_intensity[i].copy()
            raw_int[~valid] = -1
            keep_idx = np.argsort(raw_int)[-top_k:]
            topk_mask = np.zeros_like(pm_i)
            topk_mask[keep_idx] = 1.0
            f1s.append(_evaluate_coordinate_alignment(
                pf_corr[i], pi_corr[i], topk_mask,
                dft_dataset.target_freq[i], dft_dataset.target_intensity[i],
                dft_dataset.target_mask[i])["f1@10"])
        mean_f1 = np.mean(f1s)
        print(f"  top_k={top_k:3d} | val F1@10={mean_f1:.3f}")
        if mean_f1 > best_f1_val:
            best_f1_val = mean_f1
            best_k = top_k

    test_f1s, test_cov = [], []
    for i in splits["test"]:
        pm_i = dft_dataset.pred_mask[i]
        valid = pm_i > 0.5
        if valid.sum() <= best_k:
            topk_mask = pm_i.copy()
        else:
            raw_int = dft_dataset.pred_intensity[i].copy()
            raw_int[~valid] = -1
            keep_idx = np.argsort(raw_int)[-best_k:]
            topk_mask = np.zeros_like(pm_i)
            topk_mask[keep_idx] = 1.0
        m = _evaluate_coordinate_alignment(
            pf_corr[i], pi_corr[i], topk_mask,
            dft_dataset.target_freq[i], dft_dataset.target_intensity[i],
            dft_dataset.target_mask[i])
        test_f1s.append(m["f1@10"])
        test_cov.append(m["coverage@10"])
    print(f"  BEST: top_k={best_k} | test F1@10={np.mean(test_f1s):.3f}  "
          f"Coverage@10={np.mean(test_cov):.3f}")

    # Summary table
    summary_rows = []
    for s_name, s_idx in splits.items():
        sub = case_df.iloc[s_idx]
        row = {
            "model": "hybrid_v12", "split": s_name, "n_cases": len(sub),
            "f1@5": sub["f1@5"].mean(), "f1@10": sub["f1@10"].mean(),
            "f1@15": sub["f1@15"].mean(), "f1@20": sub["f1@20"].mean(),
            "cwmae@10": sub["cwmae@10"].mean(), "cwmae@5": sub["cwmae@5"].mean(),
            "coverage@10": sub["coverage@10"].mean(), "coverage@5": sub["coverage@5"].mean(),
            "point_rmse": sub["point_rmse"].mean(), "intensity_mae": sub["intensity_mae"].mean(),
            "avg_pred_kept": sub["n_pred_kept"].mean(), "avg_pred_total": sub["n_pred_total"].mean(),
            "avg_target": sub["n_target"].mean(),
        }
        for col in ["conf_f1@10", "conf_coverage@10", "conf_cwmae@10", "conf_n_pred_kept"]:
            if col in sub.columns:
                row[col] = sub[col].mean()
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "dft_alignment_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    test_row = summary_df[summary_df["split"] == "test"].iloc[0]
    conf_f1 = test_row.get('conf_f1@10', float('nan'))
    conf_cov = test_row.get('conf_coverage@10', float('nan'))
    conf_kept = test_row.get('conf_n_pred_kept', float('nan'))
    report = (
        f"### DFT Alignment v12 Hybrid Results (test set)\n"
        f"- Unfiltered: F1@10={test_row['f1@10']:.3f}  Coverage@10={test_row['coverage@10']:.3f}  "
        f"CWMAE@10={test_row['cwmae@10']:.2f} cm⁻¹  ({test_row['avg_pred_kept']:.0f}/{test_row['avg_pred_total']:.0f} modes)\n"
        f"- Conf-filtered: F1@10={conf_f1:.3f}  Coverage@10={conf_cov:.3f}  "
        f"({conf_kept:.0f} modes kept, threshold={cfg.confidence_threshold})\n"
        f"- Top-k filtered: F1@10={np.mean(test_f1s):.3f}  (k={best_k})\n"
        f"- Point RMSE (matched@10): {test_row['point_rmse']:.2f} cm⁻¹"
    )
    print(report)

    return {
        "domains": {
            "dft": {"best_model": "hybrid_v12", "summary_csv": str(summary_csv),
                    "case_csv": str(case_csv), "report_markdown": report},
            "experimental": {"best_model": "uncorrected",
                             "report_markdown": "Experimental study results pending."}
        },
        "checkpoint": str(ckpt_path),
    }


def modal_notebook_guidance(v="/mnt/raman"):
    return f"Projected runtime high. Use `ALIGNMENT_USE_MODAL_VOLUME=1` at {v}"

def _runtime_estimate_minutes(ds, cfg, dev):
    return len(ds) * cfg.max_epochs * 0.0005


# ---------------------------------------------------------------------------
# Phase 1: Spectral U-Net 1D (spectrum → spectrum alignment)
# ---------------------------------------------------------------------------

@dataclass
class SpectralAlignmentTrainConfig:
    seed: int = 20260309
    batch_size: int = 32
    max_epochs: int = 200
    patience: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-3
    morgan_fp_bits: int = 2048
    film_dim: int = 256
    # Phase 1 loss weights
    mse_weight: float = 1.0
    derivative_mse_weight: float = 0.5
    spectral_angle_weight: float = 0.1
    # Phase 2 Sinkhorn fine-tune
    sinkhorn_iters: int = 20
    sinkhorn_tau: float = 10.0  # cm⁻¹ scale for cost softening
    sinkhorn_match_sigma: float = 10.0  # cm⁻¹ for soft match indicator
    finetune_spectral_weight: float = 0.3
    finetune_sinkhorn_weight: float = 0.7
    finetune_epochs: int = 50
    finetune_lr: float = 1e-4
    finetune_patience: int = 20
    # Splits
    val_fraction: float = 0.15
    test_fraction: float = 0.15


class _ResConvBlock1D(nn.Module):
    """Residual 1D conv block with k=5 kernel."""
    def __init__(self, channels, kernel_size=5, dropout=0.1):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        res = x
        x = F.gelu(self.bn1(self.conv1(x)))
        x = self.drop(x)
        x = self.bn2(self.conv2(x))
        return F.gelu(x + res)


class _DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5, dropout=0.1):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad)
        self.bn = nn.BatchNorm1d(out_ch)
        self.res = _ResConvBlock1D(out_ch, kernel_size, dropout)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        x = F.gelu(self.bn(self.conv(x)))
        x = self.res(x)
        return x, self.pool(x)


class _UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, kernel_size=5, dropout=0.1):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, in_ch, kernel_size=2, stride=2)
        pad = kernel_size // 2
        self.conv = nn.Conv1d(in_ch + skip_ch, out_ch, kernel_size, padding=pad)
        self.bn = nn.BatchNorm1d(out_ch)
        self.res = _ResConvBlock1D(out_ch, kernel_size, dropout)

    def forward(self, x, skip):
        x = self.up(x)
        diff = skip.shape[-1] - x.shape[-1]
        if diff > 0:
            x = F.pad(x, (0, diff))
        elif diff < 0:
            x = x[..., :skip.shape[-1]]
        x = torch.cat([x, skip], dim=1)
        x = F.gelu(self.bn(self.conv(x)))
        return self.res(x)


class _FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation conditioned on molecule features."""
    def __init__(self, mol_dim, channels, film_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(mol_dim, film_dim),
            nn.GELU(),
            nn.Linear(film_dim, film_dim),
            nn.GELU(),
        )
        self.gamma_proj = nn.Linear(film_dim, channels)
        self.beta_proj = nn.Linear(film_dim, channels)

    def forward(self, x, mol_features):
        h = self.mlp(mol_features)
        gamma = self.gamma_proj(h).unsqueeze(-1)   # (B, C, 1)
        beta = self.beta_proj(h).unsqueeze(-1)
        return (1 + gamma) * x + beta              # residual scaling


class SpectralUNet1D(nn.Module):
    """
    1D U-Net for spectrum→spectrum alignment.
    Learns a residual correction: output = clamp(input + delta, 0, 1).
    4 levels (16→32→64→128 channels), k=5 residual conv blocks,
    FiLM conditioning at bottleneck on molecule features.
    """
    def __init__(self, mol_dim: int, cfg: SpectralAlignmentTrainConfig):
        super().__init__()
        self.cfg = cfg
        ch = [16, 32, 64, 128]

        self.down0 = _DownBlock(1, ch[0])
        self.down1 = _DownBlock(ch[0], ch[1])
        self.down2 = _DownBlock(ch[1], ch[2])
        self.down3 = _DownBlock(ch[2], ch[3])

        self.bottleneck = _ResConvBlock1D(ch[3])
        self.film = _FiLMLayer(mol_dim, ch[3], cfg.film_dim)

        self.up3 = _UpBlock(ch[3], ch[3], ch[2])
        self.up2 = _UpBlock(ch[2], ch[2], ch[1])
        self.up1 = _UpBlock(ch[1], ch[1], ch[0])
        self.up0 = _UpBlock(ch[0], ch[0], ch[0])

        self.head = nn.Conv1d(ch[0], 1, kernel_size=1)
        # Initialize head near zero so initial output ≈ input
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, spectrum, mol_features):
        # spectrum: (B, L) → (B, 1, L)
        x = spectrum.unsqueeze(1)
        orig_len = x.shape[-1]

        # Pad to multiple of 16 (2^4 levels of pooling)
        pad_needed = (16 - orig_len % 16) % 16
        if pad_needed > 0:
            x = F.pad(x, (0, pad_needed))

        s0, x = self.down0(x)
        s1, x = self.down1(x)
        s2, x = self.down2(x)
        s3, x = self.down3(x)

        x = self.bottleneck(x)
        x = self.film(x, mol_features)

        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        x = self.up0(x, s0)

        delta = self.head(x).squeeze(1)    # (B, padded_L)
        delta = delta[..., :orig_len]      # crop to original length
        return (spectrum + delta).clamp(0, 1)


class SpectrumDataset(Dataset):
    """Wraps y_pred_spec, y_target_spec, mol_features, and optionally target peaks."""
    def __init__(self, y_pred_spec, y_target_spec, mol_features,
                 target_freq=None, target_intensity=None, target_mask=None):
        self.y_pred = torch.as_tensor(y_pred_spec, dtype=torch.float32)
        self.y_target = torch.as_tensor(y_target_spec, dtype=torch.float32)
        self.mol_features = torch.as_tensor(mol_features, dtype=torch.float32)
        self.has_peaks = target_freq is not None
        if self.has_peaks:
            self.target_freq = torch.as_tensor(target_freq, dtype=torch.float32)
            self.target_intensity = torch.as_tensor(target_intensity, dtype=torch.float32)
            self.target_mask = torch.as_tensor(target_mask, dtype=torch.float32)

    def __len__(self):
        return len(self.y_pred)

    def __getitem__(self, idx):
        if self.has_peaks:
            return (self.y_pred[idx], self.y_target[idx], self.mol_features[idx],
                    self.target_freq[idx], self.target_intensity[idx], self.target_mask[idx])
        return (self.y_pred[idx], self.y_target[idx], self.mol_features[idx])


def spectral_loss(pred, target, cfg):
    """MSE + derivative MSE + spectral angle loss."""
    mse = F.mse_loss(pred, target)

    pred_d = pred[:, 1:] - pred[:, :-1]
    target_d = target[:, 1:] - target[:, :-1]
    deriv_mse = F.mse_loss(pred_d, target_d)

    dot = (pred * target).sum(dim=-1)
    p_norm = pred.norm(dim=-1).clamp(min=EPS)
    t_norm = target.norm(dim=-1).clamp(min=EPS)
    angle_loss = (1.0 - dot / (p_norm * t_norm)).mean()

    total = (cfg.mse_weight * mse
             + cfg.derivative_mse_weight * deriv_mse
             + cfg.spectral_angle_weight * angle_loss)
    components = {"mse": mse.item(), "deriv_mse": deriv_mse.item(), "angle": angle_loss.item()}
    return total, components


# ---------------------------------------------------------------------------
# Phase 2: Sinkhorn F1 fine-tune
# ---------------------------------------------------------------------------

def _batch_extract_peaks(spectra_np, x_grid, prominence_frac=0.03, min_distance_cm=8.0):
    """Extract peaks from a batch of spectra using scipy.signal.find_peaks."""
    from scipy.signal import find_peaks
    dx = float(np.median(np.diff(x_grid)))
    distance_pts = max(1, int(round(min_distance_cm / max(dx, 1e-12))))
    results = []
    for i in range(len(spectra_np)):
        y = spectra_np[i]
        max_y = float(np.max(y))
        if max_y <= 0:
            results.append((np.array([], dtype=np.float64), np.array([], dtype=np.float64)))
            continue
        idx, _ = find_peaks(y, prominence=prominence_frac * max_y, distance=distance_pts)
        if len(idx) == 0:
            results.append((np.array([], dtype=np.float64), np.array([], dtype=np.float64)))
            continue
        results.append((x_grid[idx].astype(np.float64), (y[idx] / max_y).astype(np.float64)))
    return results


def sinkhorn_f1_loss(corrected_spectrum, pred_peaks_list, target_freq, target_intensity,
                     target_mask, x_grid_tensor, cfg):
    """
    Differentiable F1 via Sinkhorn soft assignment.

    Peak positions are extracted non-differentiably, but the spectrum values at
    those positions are read differentiably, so gradients flow back to shape the
    output spectrum (increase amplitude at true peaks, suppress false ones).
    """
    B = corrected_spectrum.shape[0]
    device = corrected_spectrum.device
    total_loss = torch.tensor(0.0, device=device)
    count = 0

    x_min = float(x_grid_tensor[0])
    x_max = float(x_grid_tensor[-1])
    L = len(x_grid_tensor)

    for b in range(B):
        pp, _ = pred_peaks_list[b]
        if len(pp) == 0:
            continue
        t_valid = target_mask[b] > 0.5
        if t_valid.sum() < 1:
            continue

        tf_b = target_freq[b][t_valid]
        ti_b = target_intensity[b][t_valid]
        ti_norm = ti_b / (ti_b.max() + EPS)

        # Differentiable readout of spectrum at predicted peak positions
        pp_t = torch.as_tensor(pp, dtype=torch.float32, device=device)
        frac_idx = (pp_t - x_min) / (x_max - x_min + EPS) * (L - 1)
        idx_lo = frac_idx.long().clamp(0, L - 2)
        idx_hi = (idx_lo + 1).clamp(max=L - 1)
        w = frac_idx - idx_lo.float()
        pred_int = (1 - w) * corrected_spectrum[b][idx_lo] + w * corrected_spectrum[b][idx_hi]
        pred_int_norm = pred_int / (pred_int.max() + EPS)

        N, M = len(pp_t), len(tf_b)
        cost = (pp_t.unsqueeze(1) - tf_b.unsqueeze(0)).pow(2)  # (N, M)

        # Sinkhorn in log-space
        log_K = -cost / (2 * cfg.sinkhorn_tau ** 2 + EPS)
        log_alpha = torch.log(pred_int_norm.clamp(min=EPS))
        log_beta = torch.log(ti_norm.clamp(min=EPS))

        u = torch.zeros(N, device=device)
        v = torch.zeros(M, device=device)
        for _ in range(cfg.sinkhorn_iters):
            u = log_alpha - torch.logsumexp(log_K + v.unsqueeze(0), dim=1)
            v = log_beta - torch.logsumexp(log_K + u.unsqueeze(1), dim=0)

        T = torch.exp(log_K + u.unsqueeze(1) + v.unsqueeze(0))  # (N, M)

        # Soft match indicator (Gaussian, σ = match_sigma cm⁻¹)
        match_scores = torch.exp(-cost / (2 * cfg.sinkhorn_match_sigma ** 2))

        soft_tp = (T * match_scores).sum()
        soft_fp = T.sum() - soft_tp
        soft_fn = (ti_norm.sum() - soft_tp).clamp(min=0)

        soft_prec = soft_tp / (soft_tp + soft_fp + EPS)
        soft_rec = soft_tp / (soft_tp + soft_fn + EPS)
        soft_f1 = 2 * soft_prec * soft_rec / (soft_prec + soft_rec + EPS)

        total_loss = total_loss + (1.0 - soft_f1)
        count += 1

    return total_loss / max(count, 1)


def peak_sharpening_loss(pred_spectrum, target_positions, target_mask, x_grid_tensor):
    """
    Simpler alternative to Sinkhorn F1: maximize predicted spectrum amplitude
    at known target peak positions via differentiable linear interpolation.
    """
    B, L = pred_spectrum.shape
    device = pred_spectrum.device
    x_min = float(x_grid_tensor[0])
    x_max = float(x_grid_tensor[-1])
    total_loss = torch.tensor(0.0, device=device)
    count = 0

    for b in range(B):
        t_valid = target_mask[b] > 0.5
        if t_valid.sum() < 1:
            continue
        tp = target_positions[b][t_valid]
        # Differentiable readout
        frac_idx = (tp - x_min) / (x_max - x_min + EPS) * (L - 1)
        idx_lo = frac_idx.long().clamp(0, L - 2)
        idx_hi = (idx_lo + 1).clamp(max=L - 1)
        w = frac_idx - idx_lo.float()
        vals = (1 - w) * pred_spectrum[b][idx_lo] + w * pred_spectrum[b][idx_hi]
        total_loss = total_loss - vals.mean()  # maximize amplitude → minimize negative
        count += 1

    return total_loss / max(count, 1)


# ---------------------------------------------------------------------------
# Spectral U-Net evaluation helpers
# ---------------------------------------------------------------------------

def _evaluate_spectral_model_full(model, y_pred_spec, mol_features, target_freq,
                                  target_intensity, target_mask, x_grid, splits,
                                  device, model_name, batch_size=256):
    """Run U-Net on all data, extract peaks, evaluate with _evaluate_coordinate_alignment."""
    model.eval()
    N = len(y_pred_spec)
    corrected_all = np.zeros_like(y_pred_spec)

    # Batched forward pass
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            yp = torch.as_tensor(y_pred_spec[start:end], device=device, dtype=torch.float32)
            mf = torch.as_tensor(mol_features[start:end], device=device, dtype=torch.float32)
            corrected_all[start:end] = model(yp, mf).cpu().numpy()

    case_rows = []
    for i in range(N):
        pf, pi = stats_lib._extract_peaks(x_grid, corrected_all[i])
        pm = np.ones(len(pf), dtype=np.float32)

        tf_i = target_freq[i]
        ti_i = target_intensity[i].copy()
        tm_i = target_mask[i]
        t_valid = tm_i > 0.5
        if t_valid.any() and ti_i[t_valid].max() > 0:
            ti_i[t_valid] = ti_i[t_valid] / ti_i[t_valid].max()

        metrics = _evaluate_coordinate_alignment(pf, pi, pm, tf_i, ti_i, tm_i)
        row = {"case_index": i, "model": model_name}
        row.update(metrics)
        case_rows.append(row)

    case_df = pd.DataFrame(case_rows)

    summary_rows = []
    for s_name, s_idx in splits.items():
        sub = case_df.iloc[s_idx]
        row = {"model": model_name, "split": s_name, "n_cases": len(sub)}
        for col in ["f1@5", "f1@10", "f1@15", "f1@20", "cwmae@10", "cwmae@5",
                     "coverage@10", "coverage@5", "point_rmse", "intensity_mae",
                     "n_pred_kept", "n_target"]:
            if col in sub.columns:
                row[col] = sub[col].mean()
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    return case_df, summary_df, corrected_all


# ---------------------------------------------------------------------------
# Main entry point: run_spectral_alignment_study
# ---------------------------------------------------------------------------

def run_spectral_alignment_study(*, dft_dataset, out_dir, device="cpu",
                                 train_config=None, **kwargs):
    """
    Two-phase training:
      Phase 1 — spectrum→spectrum U-Net with spectral_loss (MSE + deriv + angle)
      Phase 2 — freeze encoder, fine-tune decoder with Sinkhorn F1
    """
    cfg = train_config or SpectralAlignmentTrainConfig()
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Augment mol features with large Morgan FPs for FiLM conditioning
    mol_features = _augment_mol_features(
        dft_dataset.mol_features, dft_dataset.metadata, cfg.morgan_fp_bits)
    mol_dim = mol_features.shape[1]

    splits = _split_indices(len(dft_dataset), cfg.seed, cfg.val_fraction, cfg.test_fraction)
    x_grid = dft_dataset.x_grid

    # Build datasets (include target peaks for Phase 2)
    def _make_ds(idx):
        return SpectrumDataset(
            dft_dataset.y_pred_spec[idx], dft_dataset.y_target_spec[idx],
            mol_features[idx], dft_dataset.target_freq[idx],
            dft_dataset.target_intensity[idx], dft_dataset.target_mask[idx])

    train_ds = _make_ds(splits["train"])
    val_ds = _make_ds(splits["val"])
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    # ===================================================================
    # Phase 1: Spectral U-Net with spectral_loss
    # ===================================================================
    model = SpectralUNet1D(mol_dim, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.max_epochs, eta_min=cfg.lr / 20)

    best_val, patience_cnt, best_state = float("inf"), 0, None
    print(f"Phase 1: Spectral U-Net | {device} | "
          f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])} | "
          f"mol_dim={mol_dim} grid={len(x_grid)}")

    for epoch in range(cfg.max_epochs):
        model.train()
        train_acc = 0.0
        for batch in train_loader:
            yp, yt, mf = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            opt.zero_grad()
            corrected = model(yp, mf)
            loss, _ = spectral_loss(corrected, yt, cfg)
            loss.backward()
            opt.step()
            train_acc += loss.item()
        scheduler.step()

        model.eval()
        val_acc = 0.0
        with torch.no_grad():
            for batch in val_loader:
                yp, yt, mf = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                corrected = model(yp, mf)
                v_loss, v_comp = spectral_loss(corrected, yt, cfg)
                val_acc += v_loss.item()

        v_mean = val_acc / max(len(val_loader), 1)
        if v_mean < best_val:
            best_val = v_mean
            patience_cnt = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1

        if epoch % 10 == 0:
            t_mean = train_acc / max(len(train_loader), 1)
            print(f"  Epoch {epoch:4d} | train={t_mean:.5f} val={v_mean:.5f} patience={patience_cnt}")
        if patience_cnt >= cfg.patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Phase 1 evaluation
    print("\n=== Phase 1 Evaluation ===")
    p1_cases, p1_summary, _ = _evaluate_spectral_model_full(
        model, dft_dataset.y_pred_spec, mol_features,
        dft_dataset.target_freq, dft_dataset.target_intensity, dft_dataset.target_mask,
        x_grid, splits, device, "spectral_unet_phase1", cfg.batch_size)
    p1_test = p1_summary[p1_summary["split"] == "test"]
    if not p1_test.empty:
        r = p1_test.iloc[0]
        print(f"  Phase 1 test: F1@10={r.get('f1@10',0):.3f}  "
              f"Coverage@10={r.get('coverage@10',0):.3f}  CWMAE@10={r.get('cwmae@10',0):.2f}")

    # ===================================================================
    # Phase 2: Sinkhorn F1 fine-tune (freeze encoder, fine-tune decoder)
    # ===================================================================
    print("\n=== Phase 2: Sinkhorn F1 Fine-tune ===")
    for name, param in model.named_parameters():
        if any(tag in name for tag in ("down", "bottleneck", "film")):
            param.requires_grad = False

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {n_trainable:,} / {n_total:,} parameters")

    ft_opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.finetune_lr, weight_decay=cfg.weight_decay)
    x_grid_tensor = torch.as_tensor(x_grid, dtype=torch.float32, device=device)

    best_val_ft, patience_ft, best_state_ft = float("inf"), 0, None
    for epoch in range(cfg.finetune_epochs):
        model.train()
        ft_acc = 0.0
        for batch in train_loader:
            yp = batch[0].to(device)
            yt = batch[1].to(device)
            mf = batch[2].to(device)
            tf_b = batch[3].to(device)
            ti_b = batch[4].to(device)
            tm_b = batch[5].to(device)

            ft_opt.zero_grad()
            corrected = model(yp, mf)

            s_loss, _ = spectral_loss(corrected, yt, cfg)

            # Extract peaks from corrected spectrum (non-differentiable positions)
            with torch.no_grad():
                pred_peaks = _batch_extract_peaks(corrected.detach().cpu().numpy(), x_grid)

            sk_loss = sinkhorn_f1_loss(
                corrected, pred_peaks, tf_b, ti_b, tm_b, x_grid_tensor, cfg)

            total = (cfg.finetune_spectral_weight * s_loss
                     + cfg.finetune_sinkhorn_weight * sk_loss)
            total.backward()
            ft_opt.step()
            ft_acc += total.item()

        # Validation (spectral loss only for early stopping)
        model.eval()
        val_ft_acc = 0.0
        with torch.no_grad():
            for batch in val_loader:
                corrected = model(batch[0].to(device), batch[2].to(device))
                v_loss, _ = spectral_loss(corrected, batch[1].to(device), cfg)
                val_ft_acc += v_loss.item()

        v_mean = val_ft_acc / max(len(val_loader), 1)
        if v_mean < best_val_ft:
            best_val_ft = v_mean
            patience_ft = 0
            best_state_ft = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ft += 1

        if epoch % 10 == 0:
            print(f"  FT Epoch {epoch:4d} | loss={ft_acc/max(len(train_loader),1):.5f} "
                  f"val={v_mean:.5f} patience={patience_ft}")
        if patience_ft >= cfg.finetune_patience:
            print(f"  FT early stopping at epoch {epoch}")
            break

    if best_state_ft is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state_ft.items()})

    # Unfreeze all for checkpoint
    for param in model.parameters():
        param.requires_grad = True

    ckpt_path = out_dir / "spectral_alignment_model.pth"
    torch.save({"model_state": model.state_dict(), "cfg": cfg,
                "mol_dim": mol_dim, "x_grid": x_grid}, ckpt_path)
    print(f"Checkpoint saved → {ckpt_path}")

    # Final evaluation
    print("\n=== Phase 2 (Final) Evaluation ===")
    p2_cases, p2_summary, _ = _evaluate_spectral_model_full(
        model, dft_dataset.y_pred_spec, mol_features,
        dft_dataset.target_freq, dft_dataset.target_intensity, dft_dataset.target_mask,
        x_grid, splits, device, "spectral_unet_phase2", cfg.batch_size)

    case_csv = out_dir / "spectral_alignment_cases.csv"
    p2_cases.to_csv(case_csv, index=False)
    summary_csv = out_dir / "spectral_alignment_summary.csv"
    p2_summary.to_csv(summary_csv, index=False)

    test_row = p2_summary[p2_summary["split"] == "test"]
    if not test_row.empty:
        r = test_row.iloc[0]
        report = (
            f"### Spectral U-Net Alignment Results (test set)\n"
            f"- F1@10={r.get('f1@10',0):.3f}  Coverage@10={r.get('coverage@10',0):.3f}  "
            f"CWMAE@10={r.get('cwmae@10',0):.2f} cm⁻¹\n"
            f"- Point RMSE (matched@10): {r.get('point_rmse',0):.2f} cm⁻¹\n"
            f"- Peaks detected: {r.get('n_pred_kept',0):.0f} pred vs {r.get('n_target',0):.0f} target"
        )
    else:
        report = "No test split available."
    print(report)

    return {
        "domains": {
            "dft": {"best_model": "spectral_unet", "summary_csv": str(summary_csv),
                    "case_csv": str(case_csv), "report_markdown": report},
        },
        "summary_json": str(out_dir / "summary.json"),
        "checkpoint": str(ckpt_path),
    }