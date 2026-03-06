from __future__ import annotations

import json
import math
import sqlite3
import warnings
import zlib
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from scipy.signal import correlate, correlation_lags, find_peaks
from scipy.stats import probplot
from statsmodels.graphics.agreement import mean_diff_plot
from statsmodels.stats.weightstats import ttost_paired


EPS = 1e-12
PEAK_PROMINENCE_FRAC = 0.03
PEAK_MIN_DISTANCE_CM = 8.0
RAW_LINE_MIN_REL_INTENSITY = 0.02
MANDATORY_TOLS = (5.0, 10.0)
DEFAULT_TOL = 10.0
SWEEP_TOLS = np.arange(5.0, 20.0 + 1e-9, 1.0, dtype=np.float64)
SCALE_ADJUST_GRID = np.arange(0.90, 1.080 + 1e-9, 0.01, dtype=np.float64)
BOOTSTRAP_RESAMPLES = 2000
DFT_SAMPLE_SEED = 20260306

DYNAMIC_REGIONS: dict[str, tuple[float, float] | None] = {
    "measured_support": None,
    "fingerprint": (400.0, 1800.0),
    "fp_400_800": (400.0, 800.0),
    "fp_800_1200": (800.0, 1200.0),
    "fp_1200_1600": (1200.0, 1600.0),
    "fp_1600_1800": (1600.0, 1800.0),
}

STATIC_REGIONS: dict[str, tuple[float, float]] = {
    "full": (100.0, 3200.0),
    "fingerprint": (400.0, 1800.0),
    "fp_400_800": (400.0, 800.0),
    "fp_800_1200": (800.0, 1200.0),
    "fp_1200_1600": (1200.0, 1600.0),
    "fp_1600_1800": (1600.0, 1800.0),
}

REGION_DISPLAY = {
    "measured_support": "measured support",
    "full": "full range",
    "fingerprint": "fingerprint",
    "fp_400_800": "FP 400-800",
    "fp_800_1200": "FP 800-1200",
    "fp_1200_1600": "FP 1200-1600",
    "fp_1600_1800": "FP 1600-1800",
}

BENCHMARK_DISPLAY = {
    "experimental_peak": "Experimental peak benchmark",
    "dft_raw_line": "DFT mode-position benchmark",
    "dft_peak": "DFT peak-resolution benchmark",
}

PRIMARY_BENCHMARK_ORDER = [
    ("experimental_peak", "Exp->Pred", "measured_support"),
    ("experimental_peak", "Exp->Pred", "fingerprint"),
    ("dft_raw_line", "DFT->Pred", "full"),
    ("dft_raw_line", "DFT->Pred", "fingerprint"),
    ("dft_peak", "DFT->Pred", "full"),
    ("dft_peak", "DFT->Pred", "fingerprint"),
]


def _safe_array(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        return np.asarray([], dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _normalize_intensity(y: np.ndarray) -> np.ndarray:
    y = _safe_array(y)
    y = np.clip(y, 0.0, None)
    if y.size == 0:
        return np.asarray([], dtype=np.float64)
    ymax = float(np.max(y))
    if ymax <= 0.0:
        return np.zeros_like(y, dtype=np.float64)
    return y / ymax


def _filter_visible_lines(freq: np.ndarray, intensity: np.ndarray, min_rel_intensity: float = RAW_LINE_MIN_REL_INTENSITY) -> tuple[np.ndarray, np.ndarray]:
    freq = _safe_array(freq)
    intensity = _normalize_intensity(intensity)
    if freq.size == 0 or intensity.size == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    n = int(min(freq.size, intensity.size))
    freq = freq[:n]
    intensity = intensity[:n]
    mask = intensity >= float(min_rel_intensity)
    if not mask.any() and intensity.size:
        mask[np.argmax(intensity)] = True
    return freq[mask], intensity[mask]


def _prepare_mode_lines(freq: np.ndarray, intensity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    freq = _safe_array(freq)
    intensity = _safe_array(intensity)
    if freq.size == 0 or intensity.size == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    n = int(min(freq.size, intensity.size))
    freq = freq[:n]
    intensity = intensity[:n]
    mask = np.isfinite(freq) & np.isfinite(intensity) & (freq > 1e-8)
    if not mask.any():
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    return freq[mask], _normalize_intensity(intensity[mask])


def _decode_dft_blob(blob: bytes) -> dict[str, Any]:
    return json.loads(zlib.decompress(blob))


def _clean_database_tag(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).replace(",", "")


def _mask_grid_region(x_grid: np.ndarray, region: tuple[float, float]) -> np.ndarray:
    lo, hi = region
    return (x_grid >= float(lo)) & (x_grid <= float(hi))


def _resolve_region(case: dict[str, Any], region_name: str) -> tuple[float, float] | None:
    if case["benchmark_group"] == "experimental":
        if region_name not in DYNAMIC_REGIONS:
            return None
        support = case.get("support_range_cm")
        if support is None:
            return None
        if region_name == "measured_support":
            return float(support[0]), float(support[1])
        base = DYNAMIC_REGIONS[region_name]
        if base is None:
            return None
        lo = max(float(base[0]), float(support[0]))
        hi = min(float(base[1]), float(support[1]))
        if hi <= lo:
            return None
        return lo, hi
    if region_name not in STATIC_REGIONS:
        return None
    return STATIC_REGIONS[region_name]


def _extract_peaks(
    x_grid: np.ndarray,
    y: np.ndarray,
    prominence_frac: float = PEAK_PROMINENCE_FRAC,
    min_distance_cm: float = PEAK_MIN_DISTANCE_CM,
) -> tuple[np.ndarray, np.ndarray]:
    x_grid = _safe_array(x_grid)
    y = _safe_array(y)
    if x_grid.size < 3 or y.size < 3 or float(np.max(y)) <= 0.0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    dx = float(np.median(np.diff(x_grid)))
    distance_pts = max(1, int(round(float(min_distance_cm) / max(dx, EPS))))
    prominence = float(prominence_frac * np.max(y))
    idx, _ = find_peaks(y, prominence=prominence, distance=distance_pts)
    if idx.size == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    return x_grid[idx], _normalize_intensity(y[idx])


def _filter_region(freq: np.ndarray, intensity: np.ndarray, region: tuple[float, float] | None) -> tuple[np.ndarray, np.ndarray]:
    freq = _safe_array(freq)
    intensity = _safe_array(intensity)
    if region is None or freq.size == 0:
        return freq, intensity
    lo, hi = region
    mask = (freq >= float(lo)) & (freq <= float(hi))
    return freq[mask], intensity[mask]


def _weights_from_intensity(intensity: np.ndarray) -> np.ndarray:
    intensity = _safe_array(intensity)
    if intensity.size == 0:
        return np.asarray([], dtype=np.float64)
    weights = np.clip(intensity, 0.0, None)
    total = float(np.sum(weights))
    if total <= EPS:
        return np.full(intensity.shape, 1.0 / float(intensity.size), dtype=np.float64)
    return weights / total


def _any_neighbor_coverage(source_freq: np.ndarray, target_freq: np.ndarray, tol_cm: float) -> tuple[float, np.ndarray]:
    source_freq = _safe_array(source_freq)
    target_freq = _safe_array(target_freq)
    if source_freq.size == 0:
        return math.nan, np.asarray([], dtype=np.float64)
    if target_freq.size == 0:
        return 0.0, np.full(source_freq.shape, np.inf, dtype=np.float64)
    dist = np.abs(source_freq[:, None] - target_freq[None, :])
    nearest = dist.min(axis=1)
    return float(np.mean(nearest <= float(tol_cm))), nearest


def _hungarian_match(
    source_freq: np.ndarray,
    source_int: np.ndarray,
    target_freq: np.ndarray,
    target_int: np.ndarray,
    tol_cm: float,
) -> dict[str, Any]:
    source_freq = _safe_array(source_freq)
    source_int = _safe_array(source_int)
    target_freq = _safe_array(target_freq)
    target_int = _safe_array(target_int)

    n_source = int(source_freq.size)
    n_target = int(target_freq.size)
    if n_source == 0 or n_target == 0:
        return {
            "n_source": n_source,
            "n_target": n_target,
            "n_matched": 0,
            "source_freq": np.asarray([], dtype=np.float64),
            "target_freq": np.asarray([], dtype=np.float64),
            "source_int": np.asarray([], dtype=np.float64),
            "target_int": np.asarray([], dtype=np.float64),
        }

    cost = np.abs(source_freq[:, None] - target_freq[None, :])
    row_idx, col_idx = linear_sum_assignment(cost)
    keep = cost[row_idx, col_idx] <= float(tol_cm)
    row_idx = row_idx[keep]
    col_idx = col_idx[keep]
    return {
        "n_source": n_source,
        "n_target": n_target,
        "n_matched": int(row_idx.size),
        "source_freq": source_freq[row_idx],
        "target_freq": target_freq[col_idx],
        "source_int": source_int[row_idx],
        "target_int": target_int[col_idx],
    }


def _detection_metrics(n_source: int, n_target: int, n_matched: int) -> dict[str, float]:
    tp = float(n_matched)
    fn = float(max(0, n_source - n_matched))
    fp = float(max(0, n_target - n_matched))
    precision = tp / (tp + fp) if (tp + fp) > 0 else math.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else math.nan
    f1 = (2.0 * precision * recall / (precision + recall)) if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0 else math.nan
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = _safe_array(a)
    b = _safe_array(b)
    if a.size < 2 or b.size < 2 or a.size != b.size:
        return math.nan
    if np.unique(a).size < 2 or np.unique(b).size < 2:
        return math.nan
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


def _spectrum_similarity(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a = _safe_array(a)
    b = _safe_array(b)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return {"rmse": math.nan, "pearson": math.nan, "spearman": math.nan, "cosine": math.nan}
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    pearson = float(np.corrcoef(a, b)[0, 1]) if float(np.std(a)) > EPS and float(np.std(b)) > EPS else math.nan
    spearman = _safe_spearman(a, b) if float(np.std(a)) > EPS and float(np.std(b)) > EPS else math.nan
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    cosine = float(np.dot(a, b) / denom) if denom > EPS else math.nan
    return {"rmse": rmse, "pearson": pearson, "spearman": spearman, "cosine": cosine}


def _cross_correlation_lag_cm(ref: np.ndarray, pred: np.ndarray, dx_cm: float) -> float:
    ref = _safe_array(ref)
    pred = _safe_array(pred)
    if ref.size == 0 or pred.size == 0 or ref.size != pred.size:
        return math.nan
    ref0 = ref - float(np.mean(ref))
    pred0 = pred - float(np.mean(pred))
    if float(np.std(ref0)) <= EPS or float(np.std(pred0)) <= EPS:
        return math.nan
    corr = correlate(pred0, ref0, mode="full")
    lags = correlation_lags(pred0.size, ref0.size, mode="full")
    best = int(lags[int(np.argmax(corr))])
    return float(best * dx_cm)


def _estimate_snr(y: np.ndarray, low_signal_quantile: float = 0.20) -> dict[str, float]:
    y = _normalize_intensity(y)
    if y.size == 0:
        return {"signal_max": math.nan, "noise_mad": math.nan, "noise_rms": math.nan, "snr_mad": math.nan, "snr_rms": math.nan}
    cutoff = float(np.quantile(y, low_signal_quantile))
    noise = y[y <= cutoff]
    if noise.size < 16:
        noise = y
    centered = noise - float(np.median(noise))
    noise_mad = float(1.4826 * np.median(np.abs(centered))) if centered.size else math.nan
    noise_rms = float(np.sqrt(np.mean(centered ** 2))) if centered.size else math.nan
    if ((not np.isfinite(noise_mad)) or noise_mad <= EPS or (not np.isfinite(noise_rms)) or noise_rms <= EPS) and noise.size >= 8:
        diffs = np.diff(noise)
        diff_centered = diffs - float(np.median(diffs))
        if diff_centered.size:
            rough_mad = float(1.4826 * np.median(np.abs(diff_centered)) / math.sqrt(2.0))
            rough_rms = float(np.sqrt(np.mean(diff_centered ** 2)) / math.sqrt(2.0))
            if not np.isfinite(noise_mad) or noise_mad <= EPS:
                noise_mad = rough_mad
            if not np.isfinite(noise_rms) or noise_rms <= EPS:
                noise_rms = rough_rms
    signal_max = float(np.max(y))
    snr_mad = signal_max / noise_mad if np.isfinite(noise_mad) and noise_mad > EPS else math.nan
    snr_rms = signal_max / noise_rms if np.isfinite(noise_rms) and noise_rms > EPS else math.nan
    return {
        "signal_max": signal_max,
        "noise_mad": noise_mad,
        "noise_rms": noise_rms,
        "snr_mad": snr_mad,
        "snr_rms": snr_rms,
    }


def _assign_snr_bins(values: pd.Series) -> pd.Series:
    clean = values.astype(float).replace([np.inf, -np.inf], np.nan)
    if clean.notna().sum() < 6:
        return pd.Series(["all"] * len(values), index=values.index, dtype=object)
    try:
        bins = pd.qcut(clean.rank(method="average"), q=3, labels=["low", "med", "high"])
        return bins.astype(object)
    except ValueError:
        return pd.Series(["all"] * len(values), index=values.index, dtype=object)


def _bootstrap_ci(values: np.ndarray, statistic: Callable[[np.ndarray], float], n_resamples: int = BOOTSTRAP_RESAMPLES) -> list[float] | None:
    values = _safe_array(values)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return None
    point = float(statistic(values))
    if np.allclose(values, values[0], equal_nan=False):
        return [point, point]
    rng = np.random.default_rng(123)
    draws = np.empty(int(n_resamples), dtype=np.float64)
    n = int(values.size)
    for idx in range(int(n_resamples)):
        sample = rng.choice(values, size=n, replace=True)
        draws[idx] = float(statistic(sample))
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return [point, point]
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return [float(lo), float(hi)]


def _safe_tost(samples: np.ndarray, low: float, high: float, alpha: float = 0.05) -> dict[str, Any]:
    samples = _safe_array(samples)
    samples = samples[np.isfinite(samples)]
    if samples.size < 3:
        return {"n": int(samples.size), "pvalue": math.nan, "passes": False}
    mean = float(np.mean(samples))
    sd = float(np.std(samples, ddof=1))
    if not np.isfinite(sd) or sd <= EPS:
        passes = bool(low < mean < high)
        return {"n": int(samples.size), "pvalue": 0.0 if passes else 1.0, "passes": passes}
    pvalue, lower_res, upper_res = ttost_paired(samples, np.zeros_like(samples), low, high)
    return {
        "n": int(samples.size),
        "pvalue": float(pvalue),
        "passes": bool(pvalue < alpha),
        "low_t": float(lower_res[0]),
        "low_p": float(lower_res[1]),
        "high_t": float(upper_res[0]),
        "high_p": float(upper_res[1]),
    }


def _set_theme() -> None:
    sns.set_theme(style="whitegrid", context="talk")


def _ecdf(ax: plt.Axes, values: np.ndarray, label: str) -> None:
    values = _safe_array(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    values = np.sort(values)
    y = np.arange(1, values.size + 1, dtype=np.float64) / float(values.size)
    ax.plot(values, y, lw=2.0, label=label)


def _fmt_pct(value: float) -> str:
    return "NA" if not np.isfinite(value) else f"{100.0 * value:.1f}%"


def _fmt_num(value: float, digits: int = 2) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.{digits}f}"


def _format_ci(ci: list[float] | None, digits: int = 2, pct: bool = False) -> str:
    if not ci:
        return "NA"
    if pct:
        return f"[{100.0 * ci[0]:.{digits}f}%, {100.0 * ci[1]:.{digits}f}%]"
    return f"[{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]"


def _log10_mae_to_fold(value: float) -> float:
    return float(10.0 ** value) if np.isfinite(value) else math.nan


def _compare_phrase(value: float, baseline: float, up: str, down: str, same: str, tol: float = 0.02) -> str:
    if not np.isfinite(value) or not np.isfinite(baseline):
        return same
    if value <= baseline - float(tol):
        return down
    if value >= baseline + float(tol):
        return up
    return same


def _build_case_pair_metrics(case: dict[str, Any], comp: dict[str, Any], region_name: str, tol_cm: float) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    region = _resolve_region(case, region_name)
    if region_name != "measured_support" and region is None:
        return None, []

    source = case["line_sets"].get(comp["source_set"], {"freq": np.asarray([], dtype=np.float64), "intensity": np.asarray([], dtype=np.float64)})
    target = case["line_sets"].get(comp["target_set"], {"freq": np.asarray([], dtype=np.float64), "intensity": np.asarray([], dtype=np.float64)})
    source_freq, source_int = _filter_region(source["freq"], source["intensity"], region)
    target_freq, target_int = _filter_region(target["freq"], target["intensity"], region)

    if source_freq.size == 0 and target_freq.size == 0:
        return None, []

    coverage_any, nearest = _any_neighbor_coverage(source_freq, target_freq, tol_cm)
    source_neighbor_count = int(np.sum(nearest <= float(tol_cm))) if nearest.size else 0
    source_weights = _weights_from_intensity(source_int)
    source_neighbor_weight = float(np.sum(source_weights[nearest <= float(tol_cm)])) if nearest.size and source_weights.size else math.nan
    weighted_coverage_any = source_neighbor_weight if np.isfinite(source_neighbor_weight) else math.nan
    match = _hungarian_match(source_freq, source_int, target_freq, target_int, tol_cm)
    det = _detection_metrics(match["n_source"], match["n_target"], match["n_matched"])

    signed_dnu = match["target_freq"] - match["source_freq"]
    abs_dnu = np.abs(signed_dnu)
    rel_pct = 100.0 * signed_dnu / np.maximum(np.abs(match["source_freq"]), EPS)
    log_ratio = np.log10((match["target_int"] + EPS) / (match["source_int"] + EPS))
    abs_log_ratio = np.abs(log_ratio)
    spearman_intensity = _safe_spearman(match["source_int"], match["target_int"]) if match["n_matched"] >= 2 else math.nan

    row = {
        "case_id": str(case["case_id"]),
        "row_id": int(case["row_id"]),
        "benchmark_group": case["benchmark_group"],
        "benchmark": comp["benchmark"],
        "component": case["component"],
        "snr_bin": case.get("snr_bin", "all"),
        "pair": comp["pair_label"],
        "source_set": comp["source_set"],
        "target_set": comp["target_set"],
        "region": region_name,
        "tol_cm": float(tol_cm),
        "n_source": int(match["n_source"]),
        "n_target": int(match["n_target"]),
        "source_neighbor_count": int(source_neighbor_count),
        "source_total_weight": float(np.sum(source_weights)) if source_weights.size else math.nan,
        "source_neighbor_weight": source_neighbor_weight,
        "coverage_any": float(coverage_any) if np.isfinite(coverage_any) else math.nan,
        "weighted_coverage_any": weighted_coverage_any,
        "matched_count": int(match["n_matched"]),
        "missed_count": int(match["n_source"] - match["n_matched"]),
        "missed_pct": float(1.0 - coverage_any) if np.isfinite(coverage_any) else math.nan,
        "weighted_missed_pct": float(1.0 - weighted_coverage_any) if np.isfinite(weighted_coverage_any) else math.nan,
        "tp": det["tp"],
        "fp": det["fp"],
        "fn": det["fn"],
        "precision": det["precision"],
        "recall": det["recall"],
        "f1": det["f1"],
        "median_abs_dnu_cm": float(np.median(abs_dnu)) if abs_dnu.size else math.nan,
        "p90_abs_dnu_cm": float(np.quantile(abs_dnu, 0.90)) if abs_dnu.size else math.nan,
        "mean_signed_dnu_cm": float(np.mean(signed_dnu)) if signed_dnu.size else math.nan,
        "median_abs_rel_pct_dnu": float(np.median(np.abs(rel_pct))) if rel_pct.size else math.nan,
        "mean_log10_ratio": float(np.mean(log_ratio)) if log_ratio.size else math.nan,
        "mae_log10_ratio": float(np.mean(abs_log_ratio)) if abs_log_ratio.size else math.nan,
        "spearman_intensity": spearman_intensity,
    }

    line_rows: list[dict[str, Any]] = []
    for idx in range(match["n_matched"]):
        line_rows.append(
            {
                "case_id": str(case["case_id"]),
                "row_id": int(case["row_id"]),
                "benchmark_group": case["benchmark_group"],
                "benchmark": comp["benchmark"],
                "component": case["component"],
                "snr_bin": case.get("snr_bin", "all"),
                "pair": comp["pair_label"],
                "source_set": comp["source_set"],
                "target_set": comp["target_set"],
                "region": region_name,
                "tol_cm": float(tol_cm),
                "source_freq_cm": float(match["source_freq"][idx]),
                "target_freq_cm": float(match["target_freq"][idx]),
                "source_intensity": float(match["source_int"][idx]),
                "target_intensity": float(match["target_int"][idx]),
                "signed_dnu_cm": float(signed_dnu[idx]),
                "abs_dnu_cm": float(abs_dnu[idx]),
                "rel_pct_dnu": float(rel_pct[idx]),
                "log10_ratio": float(log_ratio[idx]),
                "abs_log10_ratio": float(abs_log_ratio[idx]),
            }
        )
    return row, line_rows


def _aggregate_pairwise_summary(pair_case_df: pd.DataFrame, line_level_df: pd.DataFrame) -> pd.DataFrame:
    if pair_case_df.empty:
        return pd.DataFrame()

    pair_parts = [pair_case_df.assign(snr_group="all")]
    exp_bins = pair_case_df[(pair_case_df["benchmark_group"] == "experimental") & (pair_case_df["snr_bin"].isin(["low", "med", "high"]))].copy()
    if not exp_bins.empty:
        exp_bins["snr_group"] = exp_bins["snr_bin"]
        pair_parts.append(exp_bins)
    pair_full = pd.concat(pair_parts, ignore_index=True)

    if line_level_df.empty:
        line_full = pd.DataFrame()
    else:
        line_parts = [line_level_df.assign(snr_group="all")]
        line_bins = line_level_df[(line_level_df["benchmark_group"] == "experimental") & (line_level_df["snr_bin"].isin(["low", "med", "high"]))].copy()
        if not line_bins.empty:
            line_bins["snr_group"] = line_bins["snr_bin"]
            line_parts.append(line_bins)
        line_full = pd.concat(line_parts, ignore_index=True)

    rows: list[dict[str, Any]] = []
    group_cols = ["benchmark", "pair", "region", "tol_cm", "snr_group"]
    for keys, grp in pair_full.groupby(group_cols, dropna=False):
        benchmark, pair, region, tol_cm, snr_group = keys
        total_source = float(grp["n_source"].sum())
        total_target = float(grp["n_target"].sum())
        total_neighbor = float(grp["source_neighbor_count"].sum())
        total_source_weight = float(grp["source_total_weight"].dropna().sum())
        total_neighbor_weight = float(grp["source_neighbor_weight"].dropna().sum())
        total_matched = float(grp["matched_count"].sum())
        coverage_global = total_neighbor / total_source if total_source > 0 else math.nan
        missed_global = 1.0 - coverage_global if np.isfinite(coverage_global) else math.nan
        weighted_coverage_global = total_neighbor_weight / total_source_weight if total_source_weight > 0 else math.nan
        weighted_missed_global = 1.0 - weighted_coverage_global if np.isfinite(weighted_coverage_global) else math.nan
        precision_global = total_matched / total_target if total_target > 0 else math.nan
        recall_global = total_matched / total_source if total_source > 0 else math.nan
        f1_global = (2.0 * precision_global * recall_global / (precision_global + recall_global)) if np.isfinite(precision_global) and np.isfinite(recall_global) and (precision_global + recall_global) > 0 else math.nan
        out = {
            "benchmark": benchmark,
            "pair": pair,
            "region": region,
            "tol_cm": float(tol_cm),
            "snr_bin": snr_group,
            "n_cases": int(grp["case_id"].nunique()),
            "n_cases_with_source": int((grp["n_source"] > 0).sum()),
            "n_cases_with_target": int((grp["n_target"] > 0).sum()),
            "total_source_lines": int(total_source),
            "total_target_lines": int(total_target),
            "total_neighbor_hits": int(total_neighbor),
            "total_matched_1to1": int(total_matched),
            "global_coverage_any": coverage_global,
            "global_weighted_coverage_any": weighted_coverage_global,
            "mean_case_coverage_any": float(grp["coverage_any"].dropna().mean()) if grp["coverage_any"].notna().any() else math.nan,
            "median_case_coverage_any": float(grp["coverage_any"].dropna().median()) if grp["coverage_any"].notna().any() else math.nan,
            "mean_case_weighted_coverage_any": float(grp["weighted_coverage_any"].dropna().mean()) if grp["weighted_coverage_any"].notna().any() else math.nan,
            "median_case_weighted_coverage_any": float(grp["weighted_coverage_any"].dropna().median()) if grp["weighted_coverage_any"].notna().any() else math.nan,
            "global_missed_pct": missed_global,
            "global_weighted_missed_pct": weighted_missed_global,
            "global_precision": precision_global,
            "global_recall": recall_global,
            "global_f1": f1_global,
            "mean_case_precision": float(grp["precision"].dropna().mean()) if grp["precision"].notna().any() else math.nan,
            "mean_case_f1": float(grp["f1"].dropna().mean()) if grp["f1"].notna().any() else math.nan,
            "mean_case_spearman_intensity": float(grp["spearman_intensity"].dropna().mean()) if grp["spearman_intensity"].notna().any() else math.nan,
        }
        if line_full.empty:
            out.update(
                {
                    "matched_line_count": 0,
                    "median_abs_dnu_cm": math.nan,
                    "p90_abs_dnu_cm": math.nan,
                    "mean_signed_dnu_cm": math.nan,
                    "median_abs_rel_pct_dnu": math.nan,
                    "mae_log10_ratio": math.nan,
                }
            )
        else:
            line_grp = line_full[
                (line_full["benchmark"] == benchmark)
                & (line_full["pair"] == pair)
                & (line_full["region"] == region)
                & (line_full["tol_cm"] == tol_cm)
                & (line_full["snr_group"] == snr_group)
            ]
            if line_grp.empty:
                out.update(
                    {
                        "matched_line_count": 0,
                        "median_abs_dnu_cm": math.nan,
                        "p90_abs_dnu_cm": math.nan,
                        "mean_signed_dnu_cm": math.nan,
                        "median_abs_rel_pct_dnu": math.nan,
                        "mae_log10_ratio": math.nan,
                    }
                )
            else:
                out.update(
                    {
                        "matched_line_count": int(len(line_grp)),
                        "median_abs_dnu_cm": float(line_grp["abs_dnu_cm"].median()),
                        "p90_abs_dnu_cm": float(line_grp["abs_dnu_cm"].quantile(0.90)),
                        "mean_signed_dnu_cm": float(line_grp["signed_dnu_cm"].mean()),
                        "median_abs_rel_pct_dnu": float(line_grp["rel_pct_dnu"].abs().median()),
                        "mae_log10_ratio": float(line_grp["abs_log10_ratio"].mean()),
                    }
                )
        rows.append(out)

    return pd.DataFrame(rows).sort_values(["benchmark", "pair", "region", "snr_bin", "tol_cm"]).reset_index(drop=True)


def _flatten_case_metrics(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["case_id"])
    wide = df.pivot_table(
        index="case_id",
        columns=["benchmark", "pair", "region", "tol_cm"],
        values=metrics,
        aggfunc="first",
    )
    wide.columns = [
        f"{metric}__{benchmark}__{pair}__{region}__tol_{int(float(tol))}"
        for metric, benchmark, pair, region, tol in wide.columns.to_flat_index()
    ]
    return wide.reset_index()


def _aggregate_primary_ci(pair_case_df: pd.DataFrame, benchmark: str, pair: str, region: str, tol_cm: float) -> dict[str, Any]:
    grp = pair_case_df[
        (pair_case_df["benchmark"] == benchmark)
        & (pair_case_df["pair"] == pair)
        & (pair_case_df["region"] == region)
        & (pair_case_df["tol_cm"] == tol_cm)
        & (pair_case_df["snr_bin"].isin(["all", "not_applicable", "low", "med", "high"]))
    ].copy()
    grp = grp.drop_duplicates(subset=["case_id", "benchmark", "pair", "region", "tol_cm"])
    out = {"benchmark": benchmark, "pair": pair, "region": region, "tol_cm": float(tol_cm), "n_cases": int(grp["case_id"].nunique())}
    if grp.empty:
        out.update(
            {
                "weighted_coverage_mean_point": math.nan,
                "weighted_coverage_mean_ci95": None,
                "coverage_mean_point": math.nan,
                "coverage_mean_ci95": None,
                "missed_mean_point": math.nan,
                "missed_mean_ci95": None,
                "median_abs_dnu_case_median_point": math.nan,
                "median_abs_dnu_median_ci95": None,
                "p90_abs_dnu_case_median_point": math.nan,
                "p90_abs_dnu_median_ci95": None,
                "mae_log10_ratio_mean_point": math.nan,
                "mae_log10_ratio_mean_ci95": None,
            }
        )
        return out
    coverage_values = grp["coverage_any"].dropna().to_numpy(dtype=np.float64)
    weighted_coverage_values = grp["weighted_coverage_any"].dropna().to_numpy(dtype=np.float64)
    missed_values = grp["missed_pct"].dropna().to_numpy(dtype=np.float64)
    median_abs_values = grp["median_abs_dnu_cm"].dropna().to_numpy(dtype=np.float64)
    p90_values = grp["p90_abs_dnu_cm"].dropna().to_numpy(dtype=np.float64)
    mae_values = grp["mae_log10_ratio"].dropna().to_numpy(dtype=np.float64)
    out["weighted_coverage_mean_point"] = float(np.mean(weighted_coverage_values)) if weighted_coverage_values.size else math.nan
    out["weighted_coverage_mean_ci95"] = _bootstrap_ci(weighted_coverage_values, np.mean)
    out["coverage_mean_point"] = float(np.mean(coverage_values)) if coverage_values.size else math.nan
    out["coverage_mean_ci95"] = _bootstrap_ci(coverage_values, np.mean)
    out["missed_mean_point"] = float(np.mean(missed_values)) if missed_values.size else math.nan
    out["missed_mean_ci95"] = _bootstrap_ci(missed_values, np.mean)
    out["median_abs_dnu_case_median_point"] = float(np.median(median_abs_values)) if median_abs_values.size else math.nan
    out["median_abs_dnu_median_ci95"] = _bootstrap_ci(median_abs_values, np.median)
    out["p90_abs_dnu_case_median_point"] = float(np.median(p90_values)) if p90_values.size else math.nan
    out["p90_abs_dnu_median_ci95"] = _bootstrap_ci(p90_values, np.median)
    out["mae_log10_ratio_mean_point"] = float(np.mean(mae_values)) if mae_values.size else math.nan
    out["mae_log10_ratio_mean_ci95"] = _bootstrap_ci(mae_values, np.mean)
    return out


def _collect_tost_summary(pair_case_df: pd.DataFrame, benchmark: str, pair: str, region: str, tol_cm: float, dnu_bound: float = 12.0, log_bound: float = 0.20) -> dict[str, Any]:
    grp = pair_case_df[
        (pair_case_df["benchmark"] == benchmark)
        & (pair_case_df["pair"] == pair)
        & (pair_case_df["region"] == region)
        & (pair_case_df["tol_cm"] == tol_cm)
    ].copy()
    grp = grp.drop_duplicates(subset=["case_id", "benchmark", "pair", "region", "tol_cm"])
    pos = _safe_tost(grp["mean_signed_dnu_cm"].dropna().to_numpy(dtype=np.float64), -float(dnu_bound), float(dnu_bound))
    inten = _safe_tost(grp["mean_log10_ratio"].dropna().to_numpy(dtype=np.float64), -float(log_bound), float(log_bound))
    return {
        "benchmark": benchmark,
        "pair": pair,
        "region": region,
        "tol_cm": float(tol_cm),
        "position_bias_tost": pos,
        "intensity_bias_tost": inten,
    }


def _build_scale_sweep(
    cases: list[dict[str, Any]],
    x_grid: np.ndarray,
    lines_to_norm_spectrum: Callable[..., np.ndarray],
    sigma: float,
    temp: float,
    init_wl: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    dft_cases = [case for case in cases if case["benchmark_group"] == "dft"]
    if not dft_cases:
        return pd.DataFrame()
    for scale_adjust in SCALE_ADJUST_GRID:
        for region_name in ["full", "fingerprint"]:
            raw_source_hits = 0
            raw_source_total = 0
            raw_source_weight_hits = 0.0
            raw_source_weight_total = 0.0
            raw_target_hits = 0
            raw_target_total = 0
            raw_target_weight_hits = 0.0
            raw_target_weight_total = 0.0
            peak_source_hits = 0
            peak_source_total = 0
            peak_source_weight_hits = 0.0
            peak_source_weight_total = 0.0
            peak_target_hits = 0
            peak_target_total = 0
            peak_target_weight_hits = 0.0
            peak_target_weight_total = 0.0
            for case in dft_cases:
                region = _resolve_region(case, region_name)
                if region is None:
                    continue
                dft_freq, dft_int = _filter_region(case["line_sets"]["DFTRaw"]["freq"], case["line_sets"]["DFTRaw"]["intensity"], region)
                pred_freq = case["line_sets"]["PredRaw"]["freq"] * float(scale_adjust)
                pred_int = case["line_sets"]["PredRaw"]["intensity"]
                pred_freq, pred_int = _filter_region(pred_freq, pred_int, region)
                if dft_freq.size:
                    _, nearest = _any_neighbor_coverage(dft_freq, pred_freq, DEFAULT_TOL)
                    raw_source_hits += int(np.sum(nearest <= DEFAULT_TOL)) if nearest.size else 0
                    raw_source_total += int(dft_freq.size)
                    weights = _weights_from_intensity(dft_int)
                    raw_source_weight_hits += float(np.sum(weights[nearest <= DEFAULT_TOL])) if nearest.size and weights.size else 0.0
                    raw_source_weight_total += float(np.sum(weights)) if weights.size else 0.0
                if pred_freq.size:
                    _, nearest_back = _any_neighbor_coverage(pred_freq, dft_freq, DEFAULT_TOL)
                    raw_target_hits += int(np.sum(nearest_back <= DEFAULT_TOL)) if nearest_back.size else 0
                    raw_target_total += int(pred_freq.size)
                    weights = _weights_from_intensity(pred_int)
                    raw_target_weight_hits += float(np.sum(weights[nearest_back <= DEFAULT_TOL])) if nearest_back.size and weights.size else 0.0
                    raw_target_weight_total += float(np.sum(weights)) if weights.size else 0.0

                pred_spec_scaled = _normalize_intensity(
                    lines_to_norm_spectrum(
                        case["line_sets"]["PredRaw"]["freq"] * float(scale_adjust),
                        case["line_sets"]["PredRaw"]["intensity"],
                        x_grid,
                        sigma=float(sigma),
                        temp=float(temp),
                        init_wl=float(init_wl),
                    )
                )
                pred_peak_freq, pred_peak_int = _extract_peaks(x_grid, pred_spec_scaled)
                dft_peak_freq, dft_peak_int = _filter_region(case["line_sets"]["DFTPeak"]["freq"], case["line_sets"]["DFTPeak"]["intensity"], region)
                pred_peak_freq, pred_peak_int = _filter_region(pred_peak_freq, pred_peak_int, region)
                if dft_peak_freq.size:
                    _, nearest_peak = _any_neighbor_coverage(dft_peak_freq, pred_peak_freq, DEFAULT_TOL)
                    peak_source_hits += int(np.sum(nearest_peak <= DEFAULT_TOL)) if nearest_peak.size else 0
                    peak_source_total += int(dft_peak_freq.size)
                    weights = _weights_from_intensity(dft_peak_int)
                    peak_source_weight_hits += float(np.sum(weights[nearest_peak <= DEFAULT_TOL])) if nearest_peak.size and weights.size else 0.0
                    peak_source_weight_total += float(np.sum(weights)) if weights.size else 0.0
                if pred_peak_freq.size:
                    _, nearest_peak_back = _any_neighbor_coverage(pred_peak_freq, dft_peak_freq, DEFAULT_TOL)
                    peak_target_hits += int(np.sum(nearest_peak_back <= DEFAULT_TOL)) if nearest_peak_back.size else 0
                    peak_target_total += int(pred_peak_freq.size)
                    weights = _weights_from_intensity(pred_peak_int)
                    peak_target_weight_hits += float(np.sum(weights[nearest_peak_back <= DEFAULT_TOL])) if nearest_peak_back.size and weights.size else 0.0
                    peak_target_weight_total += float(np.sum(weights)) if weights.size else 0.0
            rows.append(
                {
                    "benchmark": "dft_raw_line",
                    "pair": "DFT->Pred",
                    "region": region_name,
                    "tol_cm": DEFAULT_TOL,
                    "scale_adjust": float(scale_adjust),
                    "global_coverage_any": (raw_source_hits / raw_source_total) if raw_source_total > 0 else math.nan,
                    "global_weighted_coverage_any": (raw_source_weight_hits / raw_source_weight_total) if raw_source_weight_total > 0 else math.nan,
                    "global_missed_pct": 1.0 - (raw_source_hits / raw_source_total) if raw_source_total > 0 else math.nan,
                    "total_source_lines": int(raw_source_total),
                }
            )
            rows.append(
                {
                    "benchmark": "dft_raw_line",
                    "pair": "Pred->DFT",
                    "region": region_name,
                    "tol_cm": DEFAULT_TOL,
                    "scale_adjust": float(scale_adjust),
                    "global_coverage_any": (raw_target_hits / raw_target_total) if raw_target_total > 0 else math.nan,
                    "global_weighted_coverage_any": (raw_target_weight_hits / raw_target_weight_total) if raw_target_weight_total > 0 else math.nan,
                    "global_missed_pct": 1.0 - (raw_target_hits / raw_target_total) if raw_target_total > 0 else math.nan,
                    "total_source_lines": int(raw_target_total),
                }
            )
            rows.append(
                {
                    "benchmark": "dft_peak",
                    "pair": "DFT->Pred",
                    "region": region_name,
                    "tol_cm": DEFAULT_TOL,
                    "scale_adjust": float(scale_adjust),
                    "global_coverage_any": (peak_source_hits / peak_source_total) if peak_source_total > 0 else math.nan,
                    "global_weighted_coverage_any": (peak_source_weight_hits / peak_source_weight_total) if peak_source_weight_total > 0 else math.nan,
                    "global_missed_pct": 1.0 - (peak_source_hits / peak_source_total) if peak_source_total > 0 else math.nan,
                    "total_source_lines": int(peak_source_total),
                }
            )
            rows.append(
                {
                    "benchmark": "dft_peak",
                    "pair": "Pred->DFT",
                    "region": region_name,
                    "tol_cm": DEFAULT_TOL,
                    "scale_adjust": float(scale_adjust),
                    "global_coverage_any": (peak_target_hits / peak_target_total) if peak_target_total > 0 else math.nan,
                    "global_weighted_coverage_any": (peak_target_weight_hits / peak_target_weight_total) if peak_target_weight_total > 0 else math.nan,
                    "global_missed_pct": 1.0 - (peak_target_hits / peak_target_total) if peak_target_total > 0 else math.nan,
                    "total_source_lines": int(peak_target_total),
                }
            )
    return pd.DataFrame(rows)


def _get_summary_row(df: pd.DataFrame, benchmark: str, pair: str, region: str, tol_cm: float, snr_bin: str = "all") -> pd.Series | None:
    if df.empty:
        return None
    mask = (
        (df["benchmark"] == benchmark)
        & (df["pair"] == pair)
        & (df["region"] == region)
        & (df["tol_cm"] == float(tol_cm))
        & (df["snr_bin"] == snr_bin)
    )
    if not mask.any():
        return None
    return df.loc[mask].iloc[0]


def _first_tol_meeting_gate(df: pd.DataFrame, benchmark: str, pair: str, region: str, threshold: float) -> float | None:
    subset = df[
        (df["benchmark"] == benchmark)
        & (df["pair"] == pair)
        & (df["region"] == region)
        & (df["snr_bin"] == "all")
    ].sort_values("tol_cm")
    if subset.empty:
        return None
    hits = subset[subset["global_coverage_any"] >= float(threshold)]
    if hits.empty:
        return None
    return float(hits["tol_cm"].iloc[0])


def _build_executive_summary(pairwise_summary_df: pd.DataFrame, spectrum_summary_df: pd.DataFrame) -> pd.DataFrame:
    spectrum_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    if not spectrum_summary_df.empty:
        spectrum_lookup = {
            (str(row["benchmark"]), str(row["region"])): row.to_dict()
            for _, row in spectrum_summary_df.iterrows()
        }
    rows = []
    for benchmark, pair, region in PRIMARY_BENCHMARK_ORDER:
        if benchmark == "experimental_peak":
            spectrum_key = ("Experimental spectra", REGION_DISPLAY.get(region, region))
        else:
            spectrum_key = ("DFT spectra", REGION_DISPLAY.get(region, region))
        spectrum_row = spectrum_lookup.get(spectrum_key, {})
        for tol_cm in MANDATORY_TOLS:
            row = _get_summary_row(pairwise_summary_df, benchmark, pair, region, tol_cm, "all")
            if row is None:
                continue
            rows.append(
                {
                    "benchmark": BENCHMARK_DISPLAY[benchmark],
                    "pair": pair,
                    "region": REGION_DISPLAY.get(region, region),
                    "tol_cm": float(tol_cm),
                    "n_cases_with_source": int(row["n_cases_with_source"]),
                    "peak_count_ratio": float(row["total_target_lines"] / row["total_source_lines"]) if float(row["total_source_lines"]) > 0 else math.nan,
                    "coverage_any_global": float(row["global_coverage_any"]),
                    "weighted_coverage_any_global": float(row["global_weighted_coverage_any"]),
                    "coverage_any_mean_case": float(row["mean_case_coverage_any"]),
                    "coverage_any_median_case": float(row["median_case_coverage_any"]),
                    "weighted_coverage_any_mean_case": float(row["mean_case_weighted_coverage_any"]),
                    "weighted_coverage_any_median_case": float(row["median_case_weighted_coverage_any"]),
                    "missed_pct_global": float(row["global_missed_pct"]),
                    "weighted_missed_pct_global": float(row["global_weighted_missed_pct"]),
                    "precision": float(row["global_precision"]),
                    "recall": float(row["global_recall"]),
                    "f1": float(row["global_f1"]),
                    "median_abs_dnu_cm_pooled": float(row["median_abs_dnu_cm"]),
                    "p90_abs_dnu_cm_pooled": float(row["p90_abs_dnu_cm"]),
                    "median_abs_rel_pct_dnu_pooled": float(row["median_abs_rel_pct_dnu"]),
                    "mae_log10_ratio_pooled": float(row["mae_log10_ratio"]),
                    "mean_case_spearman_intensity": float(row["mean_case_spearman_intensity"]),
                    "spectrum_rmse_mean": float(spectrum_row.get("rmse_mean", math.nan)),
                    "spectrum_rmse_median": float(spectrum_row.get("rmse_median", math.nan)),
                    "spectrum_cosine_mean": float(spectrum_row.get("cosine_mean", math.nan)),
                    "spectrum_cosine_median": float(spectrum_row.get("cosine_median", math.nan)),
                    "spectrum_pearson_median": float(spectrum_row.get("pearson_median", math.nan)),
                    "spectrum_spearman_median": float(spectrum_row.get("spearman_median", math.nan)),
                }
            )
    return pd.DataFrame(rows)


def _build_uncertainty_summary(pair_case_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for benchmark, pair, region in PRIMARY_BENCHMARK_ORDER:
        rows.append(_aggregate_primary_ci(pair_case_df, benchmark, pair, region, 10.0))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["benchmark"] = out["benchmark"].map(BENCHMARK_DISPLAY)
    out["region"] = out["region"].map(lambda x: REGION_DISPLAY.get(x, x))
    return out


def _build_spectrum_summary(spectrum_metric_df: pd.DataFrame) -> pd.DataFrame:
    if spectrum_metric_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (benchmark, region), grp in spectrum_metric_df.groupby(["benchmark", "region"], dropna=False):
        rmse_vals = grp["rmse"].dropna().to_numpy(dtype=np.float64)
        cosine_vals = grp["cosine"].dropna().to_numpy(dtype=np.float64)
        pearson_vals = grp["pearson"].dropna().to_numpy(dtype=np.float64)
        spearman_vals = grp["spearman"].dropna().to_numpy(dtype=np.float64)
        rows.append(
            {
                "benchmark": {"experimental_spectrum": "Experimental spectra", "dft_spectrum": "DFT spectra"}.get(benchmark, benchmark),
                "region": REGION_DISPLAY.get(region, region),
                "n_cases": int(grp["case_id"].nunique()),
                "rmse_mean": float(np.mean(rmse_vals)) if rmse_vals.size else math.nan,
                "rmse_mean_ci95": _bootstrap_ci(rmse_vals, np.mean),
                "rmse_median": float(np.median(rmse_vals)) if rmse_vals.size else math.nan,
                "cosine_mean": float(np.mean(cosine_vals)) if cosine_vals.size else math.nan,
                "cosine_mean_ci95": _bootstrap_ci(cosine_vals, np.mean),
                "cosine_median": float(np.median(cosine_vals)) if cosine_vals.size else math.nan,
                "pearson_mean": float(np.mean(pearson_vals)) if pearson_vals.size else math.nan,
                "pearson_median": float(np.median(pearson_vals)) if pearson_vals.size else math.nan,
                "spearman_mean": float(np.mean(spearman_vals)) if spearman_vals.size else math.nan,
                "spearman_median": float(np.median(spearman_vals)) if spearman_vals.size else math.nan,
            }
        )
    return pd.DataFrame(rows)


def _build_tost_df(pair_case_df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        _collect_tost_summary(pair_case_df, "dft_peak", "DFT->Pred", "full", 10.0),
        _collect_tost_summary(pair_case_df, "dft_peak", "DFT->Pred", "fingerprint", 10.0),
        _collect_tost_summary(pair_case_df, "experimental_peak", "Exp->Pred", "fingerprint", 10.0),
    ]
    out = pd.DataFrame(
        [
            {
                "benchmark": BENCHMARK_DISPLAY[row["benchmark"]],
                "pair": row["pair"],
                "region": REGION_DISPLAY.get(row["region"], row["region"]),
                "tol_cm": row["tol_cm"],
                "position_bias_pvalue": row["position_bias_tost"].get("pvalue"),
                "position_bias_passes": row["position_bias_tost"].get("passes"),
                "intensity_bias_pvalue": row["intensity_bias_tost"].get("pvalue"),
                "intensity_bias_passes": row["intensity_bias_tost"].get("passes"),
                "n_cases": row["position_bias_tost"].get("n"),
            }
            for row in rows
        ]
    )
    return out


def _build_assessment(
    *,
    dft_raw_full_10: pd.Series | None,
    dft_peak_full_10: pd.Series | None,
    exp_fp_10: pd.Series | None,
    dft_cosine_median: float,
    exp_cosine_median: float,
    tost_df: pd.DataFrame,
    primary_target: float = 0.60,
    aspirational_target: float = 0.75,
) -> dict[str, Any]:
    raw_cov = float(dft_raw_full_10["global_coverage_any"]) if dft_raw_full_10 is not None else math.nan
    raw_weighted = float(dft_raw_full_10["global_weighted_coverage_any"]) if dft_raw_full_10 is not None else math.nan
    raw_median_abs = float(dft_raw_full_10["median_abs_dnu_cm"]) if dft_raw_full_10 is not None else math.nan
    raw_p90_abs = float(dft_raw_full_10["p90_abs_dnu_cm"]) if dft_raw_full_10 is not None else math.nan
    peak_weighted = float(dft_peak_full_10["global_weighted_coverage_any"]) if dft_peak_full_10 is not None else math.nan
    exp_weighted = float(exp_fp_10["global_weighted_coverage_any"]) if exp_fp_10 is not None else math.nan
    attainment_vs_75 = (raw_cov / aspirational_target) if np.isfinite(raw_cov) and aspirational_target > 0 else math.nan
    dft_tost = tost_df[
        (tost_df["benchmark"] == BENCHMARK_DISPLAY["dft_peak"])
        & (tost_df["pair"] == "DFT->Pred")
        & (tost_df["region"] == REGION_DISPLAY["full"])
    ]
    dft_pos_bias_pass = bool(dft_tost["position_bias_passes"].iloc[0]) if not dft_tost.empty else False
    primary_target_met = bool(np.isfinite(raw_cov) and raw_cov >= primary_target)
    aspirational_target_met = bool(np.isfinite(raw_cov) and raw_cov >= aspirational_target)

    if primary_target_met and np.isfinite(raw_median_abs) and raw_median_abs <= 4.5 and np.isfinite(raw_p90_abs) and raw_p90_abs <= 10.0 and dft_pos_bias_pass:
        overall = "STRONG_EVIDENCE_WITH_CAVEATS" if (np.isfinite(peak_weighted) and peak_weighted < 0.30) or (np.isfinite(exp_weighted) and exp_weighted < 0.25) else "STRONG_EVIDENCE"
    elif primary_target_met:
        overall = "MODERATE_TO_STRONG_EVIDENCE"
    elif np.isfinite(raw_cov) and raw_cov >= 0.50:
        overall = "MIXED_EVIDENCE"
    else:
        overall = "LIMITED_EVIDENCE"

    benchmark_status = "MEETS_PRIMARY_60_PERCENT_TARGET" if primary_target_met else "BELOW_PRIMARY_60_PERCENT_TARGET"
    return {
        "overall_assessment": overall,
        "benchmark_status": benchmark_status,
        "primary_target": float(primary_target),
        "primary_target_met": primary_target_met,
        "aspirational_target": float(aspirational_target),
        "aspirational_target_met": aspirational_target_met,
        "attainment_vs_75_target": attainment_vs_75,
        "dft_raw_full_coverage10": raw_cov,
        "dft_raw_full_weighted_coverage10": raw_weighted,
        "dft_raw_full_median_abs_dnu_cm": raw_median_abs,
        "dft_raw_full_p90_abs_dnu_cm": raw_p90_abs,
        "dft_peak_full_weighted_coverage10": peak_weighted,
        "experimental_fingerprint_weighted_coverage10": exp_weighted,
        "dft_full_cosine_median": dft_cosine_median,
        "experimental_support_cosine_median": exp_cosine_median,
    }


def _select_cases_for_plots(per_molecule_df: pd.DataFrame, benchmark: str, column_suffix: str, n_each: int = 2) -> list[str]:
    if per_molecule_df.empty or column_suffix not in per_molecule_df.columns:
        return []
    subset = per_molecule_df[per_molecule_df["benchmark_group"] == benchmark][["case_id", column_suffix]].dropna().sort_values(column_suffix)
    if subset.empty:
        return []
    ids = subset.head(n_each)["case_id"].astype(str).tolist() + subset.tail(n_each)["case_id"].astype(str).tolist()
    out: list[str] = []
    seen: set[str] = set()
    for case_id in ids:
        if case_id not in seen:
            seen.add(case_id)
            out.append(case_id)
    return out


REPRESENTATIVE_LABELS = [
    ("very weak", 0.00),
    ("ok-1", 0.40),
    ("ok-2", 0.60),
    ("very good", 1.00),
]


def _select_representative_cases(per_molecule_df: pd.DataFrame, spectrum_metric_df: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    spec_lookup: dict[tuple[str, str], pd.DataFrame] = {}
    if not spectrum_metric_df.empty:
        for (benchmark, region), grp in spectrum_metric_df.groupby(["benchmark", "region"], dropna=False):
            spec_lookup[(str(benchmark), str(region))] = grp[["case_id", "cosine", "rmse"]].copy()

    configs = [
        {
            "benchmark_group": "experimental",
            "metric_col": "coverage_any__experimental_peak__Exp->Pred__measured_support__tol_10",
            "spectrum_key": ("experimental_spectrum", "measured_support"),
        },
        {
            "benchmark_group": "dft",
            "metric_col": "coverage_any__dft_raw_line__DFT->Pred__full__tol_10",
            "spectrum_key": ("dft_spectrum", "full"),
        },
    ]
    out: dict[str, list[dict[str, Any]]] = {}
    for cfg in configs:
        metric_col = str(cfg["metric_col"])
        subset = per_molecule_df[per_molecule_df["benchmark_group"] == cfg["benchmark_group"]].copy()
        if metric_col not in subset.columns:
            out[cfg["benchmark_group"]] = []
            continue
        subset = subset[["case_id", "component", metric_col]].rename(columns={metric_col: "coverage"})
        subset = subset.dropna(subset=["coverage"]).copy()
        spec_df = spec_lookup.get(tuple(cfg["spectrum_key"]))
        if spec_df is not None:
            subset = subset.merge(spec_df, on="case_id", how="left")
        if subset.empty:
            out[cfg["benchmark_group"]] = []
            continue
        subset["cosine"] = subset["cosine"].astype(float) if "cosine" in subset.columns else math.nan
        subset["rmse"] = subset["rmse"].astype(float) if "rmse" in subset.columns else math.nan
        if "cosine" not in subset.columns:
            subset["cosine"] = math.nan
        if "rmse" not in subset.columns:
            subset["rmse"] = math.nan
        cov = subset["coverage"].astype(float)
        cos = subset["cosine"].astype(float)
        cov_norm = (cov - cov.min()) / max(float(cov.max() - cov.min()), EPS)
        cos_fill = cos.fillna(float(cos.dropna().median()) if cos.notna().any() else 0.0)
        cos_norm = (cos_fill - cos_fill.min()) / max(float(cos_fill.max() - cos_fill.min()), EPS)
        subset["score"] = 0.75 * cov_norm + 0.25 * cos_norm
        subset = subset.sort_values("score").reset_index(drop=True)

        picks: list[dict[str, Any]] = []
        used: set[str] = set()
        n = len(subset)
        for label, frac in REPRESENTATIVE_LABELS:
            if n == 0:
                break
            idx = int(round(frac * (n - 1)))
            candidates = [idx]
            for step in range(1, n):
                candidates.extend([max(0, idx - step), min(n - 1, idx + step)])
            chosen = None
            for cand in candidates:
                case_id = str(subset.iloc[cand]["case_id"])
                if case_id not in used:
                    chosen = cand
                    break
            if chosen is None:
                continue
            rec = subset.iloc[chosen]
            used.add(str(rec["case_id"]))
            picks.append(
                {
                    "case_id": str(rec["case_id"]),
                    "label": label,
                    "coverage": float(rec["coverage"]),
                    "cosine": float(rec["cosine"]) if np.isfinite(float(rec["cosine"])) else math.nan,
                    "rmse": float(rec["rmse"]) if np.isfinite(float(rec["rmse"])) else math.nan,
                }
            )
        out[cfg["benchmark_group"]] = picks
    return out


def _save_plot_sweep(pairwise_summary_df: pd.DataFrame, out_dir: Path) -> Path:
    fig_path = out_dir / "stats_tolerance_sweep.png"
    _set_theme()
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=True)
    panels = [
        (axes[0, 0], "experimental_peak", "Exp->Pred", "measured_support", "Experimental coverage sweep: measured support"),
        (axes[0, 1], "experimental_peak", "Exp->Pred", "fingerprint", "Experimental coverage sweep: fingerprint"),
        (axes[1, 0], None, "DFT->Pred", "full", "DFT coverage sweep: full range"),
        (axes[1, 1], None, "DFT->Pred", "fingerprint", "DFT coverage sweep: fingerprint"),
    ]
    for ax, benchmark, pair, region, title in panels:
        if benchmark is None:
            data = pairwise_summary_df[
                (pairwise_summary_df["benchmark"].isin(["dft_peak", "dft_raw_line"]))
                & (pairwise_summary_df["pair"] == pair)
                & (pairwise_summary_df["region"] == region)
                & (pairwise_summary_df["snr_bin"] == "all")
            ].copy()
            for bench, grp in data.groupby("benchmark"):
                label = BENCHMARK_DISPLAY.get(bench, bench)
                if bench == "dft_peak":
                    ax.plot(grp["tol_cm"], grp["global_weighted_coverage_any"], marker="o", lw=2.2, label=f"{label} weighted")
                    ax.plot(grp["tol_cm"], grp["global_coverage_any"], marker="o", lw=1.2, ls="--", alpha=0.75, label=f"{label} unweighted")
                else:
                    ax.plot(grp["tol_cm"], grp["global_coverage_any"], marker="o", lw=2.0, label=f"{label} unweighted")
        else:
            data = pairwise_summary_df[
                (pairwise_summary_df["benchmark"] == benchmark)
                & (pairwise_summary_df["pair"] == pair)
                & (pairwise_summary_df["region"] == region)
                & (pairwise_summary_df["snr_bin"] == "all")
            ].copy()
            if not data.empty:
                ax.plot(data["tol_cm"], data["global_weighted_coverage_any"], marker="o", lw=2.2, label=f"{BENCHMARK_DISPLAY.get(benchmark, benchmark)} weighted")
                ax.plot(data["tol_cm"], data["global_coverage_any"], marker="o", lw=1.2, ls="--", alpha=0.75, label=f"{BENCHMARK_DISPLAY.get(benchmark, benchmark)} unweighted")
        ax.axhline(0.75, color="#111111", ls=":", lw=1.2)
        ax.axhline(0.60, color="#374151", ls="--", lw=1.0)
        ax.set_title(title, weight="semibold")
        ax.set_xlabel("tolerance (cm^-1)")
        ax.set_ylabel("coverage")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.25)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220, facecolor="white")
    plt.close(fig)
    return fig_path


def _save_plot_scale_sweep(scale_sweep_df: pd.DataFrame, out_dir: Path) -> Path:
    fig_path = out_dir / "stats_scale_sweep.png"
    if scale_sweep_df.empty:
        return fig_path
    plot_df = scale_sweep_df[(scale_sweep_df["pair"] == "DFT->Pred") & (scale_sweep_df["region"].isin(["full", "fingerprint"]))].copy()
    if plot_df.empty:
        return fig_path
    _set_theme()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    for ax, region in zip(axes, ["full", "fingerprint"]):
        grp = plot_df[plot_df["region"] == region]
        if grp.empty:
            ax.set_visible(False)
            continue
        for benchmark, bench_grp in grp.groupby("benchmark"):
            y_col = "global_weighted_coverage_any" if benchmark == "dft_peak" else "global_coverage_any"
            ax.plot(
                bench_grp["scale_adjust"],
                bench_grp[y_col],
                marker="o",
                lw=2.0,
                label=BENCHMARK_DISPLAY.get(benchmark, benchmark) + (" (weighted)" if benchmark == "dft_peak" else ""),
            )
        ax.axvline(1.0, color="#111111", ls=":", lw=1.2, label="current scale")
        ax.set_title(f"Coverage vs extra scale: {REGION_DISPLAY.get(region, region)}", weight="semibold")
        ax.set_xlabel("extra multiplicative scale on current predicted lines")
        ax.set_ylabel("coverage@10")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220, facecolor="white")
    plt.close(fig)
    return fig_path


def _save_plot_error_distributions(line_level_df: pd.DataFrame, out_dir: Path) -> Path:
    fig_path = out_dir / "stats_error_distributions.png"
    configs = [
        ("experimental_peak", "Exp->Pred", "measured_support", "Experimental"),
        ("dft_peak", "DFT->Pred", "full", "DFT peak-resolution"),
        ("dft_raw_line", "DFT->Pred", "full", "DFT raw-line"),
    ]
    _set_theme()
    fig, axes = plt.subplots(3, 2, figsize=(15, 16))
    metrics = [("signed_dnu_cm", "signed Δν (cm^-1)"), ("abs_dnu_cm", "|Δν| (cm^-1)"), ("rel_pct_dnu", "relative Δν (%)")]
    for row_idx, (metric, xlabel) in enumerate(metrics):
        ax_hist = axes[row_idx, 0]
        ax_ecdf = axes[row_idx, 1]
        for benchmark, pair, region, label in configs:
            grp = line_level_df[
                (line_level_df["benchmark"] == benchmark)
                & (line_level_df["pair"] == pair)
                & (line_level_df["region"] == region)
                & (line_level_df["tol_cm"] == DEFAULT_TOL)
            ]
            values = grp[metric].dropna().to_numpy(dtype=np.float64)
            if values.size == 0:
                continue
            sns.histplot(values, bins=32, kde=True, stat="density", element="step", fill=False, label=label, ax=ax_hist)
            _ecdf(ax_ecdf, values, label)
        ax_hist.set_title(f"{xlabel} distribution", weight="semibold")
        ax_hist.set_xlabel(xlabel)
        ax_hist.grid(True, alpha=0.25)
        ax_ecdf.set_title(f"{xlabel} ECDF", weight="semibold")
        ax_ecdf.set_xlabel(xlabel)
        ax_ecdf.set_ylabel("cdf")
        ax_ecdf.grid(True, alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(fig_path, dpi=220, facecolor="white")
    plt.close(fig)
    return fig_path


def _save_plot_region_boxes(pair_case_df: pd.DataFrame, out_dir: Path) -> Path:
    fig_path = out_dir / "stats_region_boxplots.png"
    plot_df = pair_case_df[
        (pair_case_df["tol_cm"] == DEFAULT_TOL)
        & (pair_case_df["pair"] == "DFT->Pred")
        & (pair_case_df["benchmark"].isin(["dft_peak", "experimental_peak"]))
        & (pair_case_df["region"].isin(["fingerprint", "fp_400_800", "fp_800_1200", "fp_1200_1600", "fp_1600_1800", "measured_support", "full"]))
    ].copy()
    if plot_df.empty:
        return fig_path
    plot_df["benchmark_label"] = plot_df["benchmark"].map(BENCHMARK_DISPLAY)
    plot_df["region_label"] = plot_df["region"].map(lambda x: REGION_DISPLAY.get(x, x))
    _set_theme()
    fig, axes = plt.subplots(3, 1, figsize=(16, 15), sharex=True)
    sns.boxplot(data=plot_df.dropna(subset=["weighted_coverage_any"]), x="region_label", y="weighted_coverage_any", hue="benchmark_label", ax=axes[0])
    sns.boxplot(data=plot_df.dropna(subset=["median_abs_dnu_cm"]), x="region_label", y="median_abs_dnu_cm", hue="benchmark_label", ax=axes[1])
    sns.boxplot(data=plot_df.dropna(subset=["weighted_missed_pct"]), x="region_label", y="weighted_missed_pct", hue="benchmark_label", ax=axes[2])
    axes[0].set_title("Intensity-weighted coverage by region at ±10 cm^-1", weight="semibold")
    axes[1].set_title("Median |Δν| by region at ±10 cm^-1", weight="semibold")
    axes[2].set_title("Intensity-weighted missed fraction by region at ±10 cm^-1", weight="semibold")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="x", rotation=25)
    for ax in axes[1:]:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220, facecolor="white")
    plt.close(fig)
    return fig_path


def _save_plot_qq(line_level_df: pd.DataFrame, out_dir: Path) -> Path:
    fig_path = out_dir / "stats_signed_dnu_qq.png"
    configs = [
        ("experimental_peak", "Exp->Pred", "measured_support", "Experimental"),
        ("dft_peak", "DFT->Pred", "full", "DFT peak-resolution"),
        ("dft_raw_line", "DFT->Pred", "full", "DFT raw-line"),
    ]
    _set_theme()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, (benchmark, pair, region, label) in zip(axes, configs):
        vals = line_level_df[
            (line_level_df["benchmark"] == benchmark)
            & (line_level_df["pair"] == pair)
            & (line_level_df["region"] == region)
            & (line_level_df["tol_cm"] == DEFAULT_TOL)
        ]["signed_dnu_cm"].dropna().to_numpy(dtype=np.float64)
        if vals.size < 3:
            ax.set_visible(False)
            continue
        (osm, osr), (slope, intercept, _) = probplot(vals, dist="norm")
        ax.scatter(osm, osr, s=18, alpha=0.7)
        ax.plot(osm, slope * osm + intercept, color="#111111", lw=1.4)
        ax.set_title(label, weight="semibold")
        ax.set_xlabel("theoretical quantiles")
        ax.set_ylabel("sample quantiles")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220, facecolor="white")
    plt.close(fig)
    return fig_path


def _save_plot_scatter(line_level_df: pd.DataFrame, out_dir: Path) -> Path:
    fig_path = out_dir / "stats_ref_vs_target_scatter.png"
    configs = [
        ("experimental_peak", "Exp->Pred", "measured_support", "Experimental"),
        ("dft_peak", "DFT->Pred", "full", "DFT peak-resolution"),
        ("dft_raw_line", "DFT->Pred", "full", "DFT raw-line"),
    ]
    _set_theme()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, (benchmark, pair, region, label) in zip(axes, configs):
        grp = line_level_df[
            (line_level_df["benchmark"] == benchmark)
            & (line_level_df["pair"] == pair)
            & (line_level_df["region"] == region)
            & (line_level_df["tol_cm"] == DEFAULT_TOL)
        ]
        if grp.empty:
            ax.set_visible(False)
            continue
        ax.scatter(grp["source_freq_cm"], grp["target_freq_cm"], s=16, alpha=0.55)
        lo = float(min(grp["source_freq_cm"].min(), grp["target_freq_cm"].min()))
        hi = float(max(grp["source_freq_cm"].max(), grp["target_freq_cm"].max()))
        grid = np.linspace(lo, hi, 200)
        ax.plot(grid, grid, color="#111111", lw=1.4)
        ax.plot(grid, grid + 5.0, color="#666666", ls="--", lw=1.0)
        ax.plot(grid, grid - 5.0, color="#666666", ls="--", lw=1.0)
        ax.plot(grid, grid + 10.0, color="#999999", ls=":", lw=1.0)
        ax.plot(grid, grid - 10.0, color="#999999", ls=":", lw=1.0)
        ax.set_title(label, weight="semibold")
        ax.set_xlabel("ν_ref (cm^-1)")
        ax.set_ylabel("ν_target (cm^-1)")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220, facecolor="white")
    plt.close(fig)
    return fig_path


def _save_plot_intensity_agreement(line_level_df: pd.DataFrame, out_dir: Path) -> Path:
    fig_path = out_dir / "stats_intensity_agreement.png"
    configs = [
        ("experimental_peak", "Exp->Pred", "measured_support", "Experimental"),
        ("dft_peak", "DFT->Pred", "full", "DFT peak-resolution"),
    ]
    _set_theme()
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for col, (benchmark, pair, region, label) in enumerate(configs):
        grp = line_level_df[
            (line_level_df["benchmark"] == benchmark)
            & (line_level_df["pair"] == pair)
            & (line_level_df["region"] == region)
            & (line_level_df["tol_cm"] == DEFAULT_TOL)
        ].copy()
        if grp.empty:
            axes[0, col].set_visible(False)
            axes[1, col].set_visible(False)
            continue
        axes[0, col].scatter(grp["source_intensity"], grp["target_intensity"], s=18, alpha=0.6)
        axes[0, col].plot([0, 1], [0, 1], color="#111111", lw=1.3)
        axes[0, col].set_title(f"{label}: matched intensities", weight="semibold")
        axes[0, col].set_xlabel("ref intensity")
        axes[0, col].set_ylabel("target intensity")
        axes[0, col].grid(True, alpha=0.25)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels.graphics.agreement")
            mean_diff_plot(grp["source_intensity"], grp["target_intensity"], ax=axes[1, col], scatter_kwds={"s": 18, "alpha": 0.6})
        axes[1, col].set_title(f"{label}: Bland-Altman", weight="semibold")
        axes[1, col].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220, facecolor="white")
    plt.close(fig)
    return fig_path


def _save_plot_spectrum_similarity(spectrum_metric_df: pd.DataFrame, out_dir: Path) -> Path:
    fig_path = out_dir / "stats_spectrum_similarity.png"
    if spectrum_metric_df.empty:
        return fig_path
    plot_df = spectrum_metric_df.copy()
    plot_df["benchmark_label"] = plot_df["benchmark"].map({"experimental_spectrum": "Experimental spectra", "dft_spectrum": "DFT spectra"})
    plot_df["region_label"] = plot_df["region"].map(lambda x: REGION_DISPLAY.get(x, x))
    _set_theme()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    sns.boxplot(data=plot_df.dropna(subset=["rmse"]), x="benchmark_label", y="rmse", hue="region_label", ax=axes[0])
    sns.boxplot(data=plot_df.dropna(subset=["cosine"]), x="benchmark_label", y="cosine", hue="region_label", ax=axes[1])
    axes[0].set_title("Spectrum RMSE", weight="semibold")
    axes[1].set_title("Spectrum cosine similarity", weight="semibold")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220, facecolor="white")
    plt.close(fig)
    return fig_path


def _save_plot_stick_overlays(cases: list[dict[str, Any]], per_molecule_df: pd.DataFrame, spectrum_metric_df: pd.DataFrame, out_dir: Path) -> Path:
    fig_path = out_dir / "stats_stick_overlays.png"
    case_by_id = {str(case["case_id"]): case for case in cases}
    reps = _select_representative_cases(per_molecule_df, spectrum_metric_df)
    bench_order = ["experimental", "dft"]
    if not any(reps.get(bench) for bench in bench_order):
        return fig_path
    _set_theme()
    nrows = len(bench_order)
    ncols = len(REPRESENTATIVE_LABELS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.8 * ncols, 3.8 * nrows), sharex=False, sharey=False)
    axes = np.atleast_2d(axes)
    for row_idx, bench in enumerate(bench_order):
        specs = {spec["label"]: spec for spec in reps.get(bench, [])}
        for col_idx, (label, _) in enumerate(REPRESENTATIVE_LABELS):
            ax = axes[row_idx, col_idx]
            spec = specs.get(label)
            if spec is None:
                ax.axis("off")
                continue
            case = case_by_id.get(str(spec["case_id"]))
            if case is None:
                ax.axis("off")
                continue
            if case["benchmark_group"] == "experimental":
                ref_key = "ExpPeak"
                pred_key = "PredPeak"
                xlim = case.get("support_range_cm", (400.0, 1800.0))
                ref_color = "#111111"
                bench_label = "Experimental peaks"
            else:
                ref_key = "DFTRaw"
                pred_key = "PredRaw"
                xlim = STATIC_REGIONS["fingerprint"]
                ref_color = "#2563eb"
                bench_label = "DFT modes"
            for freq, inten in zip(case["line_sets"][ref_key]["freq"], case["line_sets"][ref_key]["intensity"]):
                ax.vlines(freq, 0.0, inten, color=ref_color, lw=0.9, alpha=0.65)
            for freq, inten in zip(case["line_sets"][pred_key]["freq"], case["line_sets"][pred_key]["intensity"]):
                ax.vlines(freq, 0.0, inten, color="#dc2626", lw=0.8, alpha=0.55)
            ax.set_title(
                f"{bench_label} | {label}\n"
                f"cov@10={_fmt_pct(spec['coverage'])} | cos={_fmt_num(spec['cosine'], 3)}",
                fontsize=10,
            )
            ax.text(
                0.02,
                0.96,
                str(case["component"])[:28],
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="#374151",
            )
            ax.set_xlim(float(xlim[0]), float(xlim[1]))
            ax.set_xlabel("wavenumber (cm^-1)")
            ax.set_ylabel("normalized intensity")
            ax.grid(True, alpha=0.20)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220, facecolor="white")
    plt.close(fig)
    return fig_path


def _save_plot_broadened_overlays(
    cases: list[dict[str, Any]],
    per_molecule_df: pd.DataFrame,
    spectrum_metric_df: pd.DataFrame,
    line_level_df: pd.DataFrame,
    out_dir: Path,
) -> Path:
    fig_path = out_dir / "stats_broadened_overlays.png"
    if not cases:
        return fig_path
    case_by_id = {str(case["case_id"]): case for case in cases}
    reps = _select_representative_cases(per_molecule_df, spectrum_metric_df)
    bench_order = ["experimental", "dft"]
    if not any(reps.get(bench) for bench in bench_order):
        return fig_path
    _set_theme()
    nrows = len(bench_order)
    ncols = len(REPRESENTATIVE_LABELS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.8 * ncols, 4.2 * nrows), sharex=False, sharey=False)
    axes = np.atleast_2d(axes)
    for row_idx, bench in enumerate(bench_order):
        specs = {spec["label"]: spec for spec in reps.get(bench, [])}
        for col_idx, (label, _) in enumerate(REPRESENTATIVE_LABELS):
            ax = axes[row_idx, col_idx]
            spec = specs.get(label)
            if spec is None:
                ax.axis("off")
                continue
            case = case_by_id.get(str(spec["case_id"]))
            if case is None:
                ax.axis("off")
                continue
            if case["benchmark_group"] == "experimental":
                benchmark = "experimental_peak"
                pair = "Exp->Pred"
                region = "measured_support"
                xlim = case.get("support_range_cm", (400.0, 1800.0))
                ref = _safe_array(case["spectra"]["Exp"])
                pred = _safe_array(case["spectra"]["Pred"])
                ref_label = "Experimental"
                bench_label = "Experimental spectra"
                xvals = x_grid = np.linspace(500.0, 4000.0, ref.size) if ref.size else np.asarray([], dtype=np.float64)
            else:
                benchmark = "dft_raw_line"
                pair = "DFT->Pred"
                region = "full"
                ref = _safe_array(case["spectra"]["DFT"])
                pred = _safe_array(case["spectra"]["Pred"])
                ref_label = "DFT"
                bench_label = "DFT spectra"
                xvals = x_grid = np.linspace(500.0, 4000.0, ref.size) if ref.size else np.asarray([], dtype=np.float64)
                support_freq = np.concatenate([_safe_array(case["line_sets"]["DFTRaw"]["freq"]), _safe_array(case["line_sets"]["PredRaw"]["freq"])])
                if support_freq.size:
                    xlim = (
                        max(400.0, float(np.min(support_freq)) - 75.0),
                        min(3200.0, float(np.max(support_freq)) + 75.0),
                    )
                else:
                    xlim = STATIC_REGIONS["full"]
            if ref.size and pred.size and ref.size == pred.size:
                ax.plot(xvals, ref, color="#111111", lw=1.8, alpha=0.9, label=ref_label)
                ax.plot(xvals, pred, color="#d97706", lw=1.6, alpha=0.9, label="Predicted")
            matched = line_level_df[
                (line_level_df["case_id"].astype(str) == str(spec["case_id"]))
                & (line_level_df["benchmark"] == benchmark)
                & (line_level_df["pair"] == pair)
                & (line_level_df["region"] == region)
                & (line_level_df["tol_cm"] == DEFAULT_TOL)
            ].sort_values("source_freq_cm")
            if not matched.empty:
                if len(matched) > 24:
                    keep_idx = np.linspace(0, len(matched) - 1, 24, dtype=int)
                    matched = matched.iloc[keep_idx].copy()
                y_levels = -0.03 - 0.025 * (np.arange(len(matched)) % 4)
                colors = plt.cm.viridis(np.clip(matched["abs_dnu_cm"].to_numpy(dtype=np.float64) / max(DEFAULT_TOL, EPS), 0.0, 1.0))
                for idx, (_, rec) in enumerate(matched.iterrows()):
                    y = float(y_levels[idx])
                    ax.plot([rec["source_freq_cm"], rec["target_freq_cm"]], [y, y], color=colors[idx], lw=1.2, alpha=0.85)
                    ax.scatter([rec["source_freq_cm"], rec["target_freq_cm"]], [y, y], color=[colors[idx], colors[idx]], s=10, alpha=0.9)
            ax.set_ylim(-0.14, 1.05)
            ax.set_xlim(float(xlim[0]), float(xlim[1]))
            ax.set_title(
                f"{bench_label} | {label}\n"
                f"cov@10={_fmt_pct(spec['coverage'])} | cos={_fmt_num(spec['cosine'], 3)}",
                fontsize=10,
            )
            ax.text(
                0.02,
                0.96,
                str(case["component"])[:28],
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="#374151",
            )
            ax.set_xlabel("wavenumber (cm^-1)")
            ax.set_ylabel("normalized intensity")
            ax.grid(True, alpha=0.20)
            if row_idx == 0 and col_idx == 0:
                ax.legend(frameon=True, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220, facecolor="white")
    plt.close(fig)
    return fig_path


def _save_plot_connectors(
    cases: list[dict[str, Any]],
    per_molecule_df: pd.DataFrame,
    spectrum_metric_df: pd.DataFrame,
    pair_case_df: pd.DataFrame,
    line_level_df: pd.DataFrame,
    out_dir: Path,
) -> Path:
    fig_path = out_dir / "stats_line_connectors.png"
    if not cases or line_level_df.empty or pair_case_df.empty:
        return fig_path
    case_by_id = {str(case["case_id"]): case for case in cases}
    reps = _select_representative_cases(per_molecule_df, spectrum_metric_df)
    bench_order = ["experimental", "dft"]
    if not any(reps.get(bench) for bench in bench_order):
        return fig_path
    _set_theme()
    nrows = len(bench_order)
    ncols = len(REPRESENTATIVE_LABELS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.8 * ncols, 3.4 * nrows), sharex=False, sharey=False)
    axes = np.atleast_2d(axes)
    for row_idx, bench in enumerate(bench_order):
        specs = {spec["label"]: spec for spec in reps.get(bench, [])}
        for col_idx, (label, _) in enumerate(REPRESENTATIVE_LABELS):
            ax = axes[row_idx, col_idx]
            spec = specs.get(label)
            if spec is None:
                ax.axis("off")
                continue
            case = case_by_id.get(str(spec["case_id"]))
            if case is None:
                ax.axis("off")
                continue
            if case["benchmark_group"] == "experimental":
                benchmark = "experimental_peak"
                pair = "Exp->Pred"
                region = "measured_support"
                source_key, target_key = "ExpPeak", "PredPeak"
                xlim = case.get("support_range_cm", (400.0, 1800.0))
            else:
                benchmark = "dft_raw_line"
                pair = "DFT->Pred"
                region = "full"
                source_key, target_key = "DFTRaw", "PredRaw"
                xlim = STATIC_REGIONS["fingerprint"]
            matched = line_level_df[
                (line_level_df["case_id"].astype(str) == str(spec["case_id"]))
                & (line_level_df["benchmark"] == benchmark)
                & (line_level_df["pair"] == pair)
                & (line_level_df["region"] == region)
                & (line_level_df["tol_cm"] == DEFAULT_TOL)
            ].copy()
            source_freq = _safe_array(case["line_sets"][source_key]["freq"])
            target_freq = _safe_array(case["line_sets"][target_key]["freq"])
            matched_source = set(np.round(matched["source_freq_cm"].to_numpy(dtype=np.float64), 6)) if not matched.empty else set()
            matched_target = set(np.round(matched["target_freq_cm"].to_numpy(dtype=np.float64), 6)) if not matched.empty else set()
            unmatched_source = np.asarray([v for v in source_freq if round(float(v), 6) not in matched_source], dtype=np.float64)
            unmatched_target = np.asarray([v for v in target_freq if round(float(v), 6) not in matched_target], dtype=np.float64)
            if unmatched_source.size:
                ax.scatter(unmatched_source, np.ones(unmatched_source.size), color="#9ca3af", s=14, alpha=0.45)
            if unmatched_target.size:
                ax.scatter(unmatched_target, np.zeros(unmatched_target.size), color="#fca5a5", s=14, alpha=0.45)
            if not matched.empty:
                colors = plt.cm.plasma(np.clip(matched["abs_dnu_cm"].to_numpy(dtype=np.float64) / max(DEFAULT_TOL, EPS), 0.0, 1.0))
                for idx, (_, rec) in enumerate(matched.iterrows()):
                    ax.plot([rec["source_freq_cm"], rec["target_freq_cm"]], [1, 0], color=colors[idx], lw=1.0, alpha=0.85)
                ax.scatter(matched["source_freq_cm"], np.ones(len(matched)), color="#111111", s=22, alpha=0.95)
                ax.scatter(matched["target_freq_cm"], np.zeros(len(matched)), color="#dc2626", s=22, alpha=0.95)
            ax.set_title(
                f"{label} | cov@10={_fmt_pct(spec['coverage'])}\nmatched={len(matched)} src={len(source_freq)} tgt={len(target_freq)}",
                fontsize=9,
            )
            ax.text(0.02, 0.96, str(case["component"])[:24], transform=ax.transAxes, ha="left", va="top", fontsize=8, color="#374151")
            ax.set_yticks([0, 1], labels=[target_key, source_key])
            ax.set_xlim(float(xlim[0]), float(xlim[1]))
            ax.set_xlabel("wavenumber (cm^-1)")
            ax.grid(True, alpha=0.20)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220, facecolor="white")
    plt.close(fig)
    return fig_path


def _build_narrative(
    summary: dict[str, Any],
    pairwise_summary_df: pd.DataFrame,
    uncertainty_df: pd.DataFrame,
    scale_sweep_df: pd.DataFrame,
    spectrum_summary_df: pd.DataFrame,
    tost_df: pd.DataFrame,
) -> str:
    dft_raw_5 = _get_summary_row(pairwise_summary_df, "dft_raw_line", "DFT->Pred", "full", 5.0, "all")
    dft_peak_10 = _get_summary_row(pairwise_summary_df, "dft_peak", "DFT->Pred", "full", 10.0, "all")
    dft_peak_5 = _get_summary_row(pairwise_summary_df, "dft_peak", "DFT->Pred", "full", 5.0, "all")
    dft_raw_10 = _get_summary_row(pairwise_summary_df, "dft_raw_line", "DFT->Pred", "full", 10.0, "all")
    exp_fp_10 = _get_summary_row(pairwise_summary_df, "experimental_peak", "Exp->Pred", "fingerprint", 10.0, "all")
    exp_ms_10 = _get_summary_row(pairwise_summary_df, "experimental_peak", "Exp->Pred", "measured_support", 10.0, "all")
    low_snr_fp = _get_summary_row(pairwise_summary_df, "experimental_peak", "Exp->Pred", "fingerprint", 10.0, "low")
    strict_cross_tol = _first_tol_meeting_gate(pairwise_summary_df, "dft_raw_line", "DFT->Pred", "full", 0.75)
    weighted_cross_tol = None
    dft_peak_sweep = pairwise_summary_df[
        (pairwise_summary_df["benchmark"] == "dft_peak")
        & (pairwise_summary_df["pair"] == "DFT->Pred")
        & (pairwise_summary_df["region"] == "full")
        & (pairwise_summary_df["snr_bin"] == "all")
    ].sort_values("tol_cm")
    if not dft_peak_sweep.empty:
        weighted_hits = dft_peak_sweep[dft_peak_sweep["global_weighted_coverage_any"] >= 0.60]
        if not weighted_hits.empty:
            weighted_cross_tol = float(weighted_hits["tol_cm"].iloc[0])

    unc_lookup = {
        (row["benchmark"], row["pair"], row["region"]): row for _, row in uncertainty_df.iterrows()
    } if not uncertainty_df.empty else {}
    dft_raw_unc = unc_lookup.get((BENCHMARK_DISPLAY["dft_raw_line"], "DFT->Pred", REGION_DISPLAY["full"]))
    dft_peak_unc = unc_lookup.get((BENCHMARK_DISPLAY["dft_peak"], "DFT->Pred", REGION_DISPLAY["full"]))
    exp_unc = unc_lookup.get((BENCHMARK_DISPLAY["experimental_peak"], "Exp->Pred", REGION_DISPLAY["fingerprint"]))
    spectrum_lookup = {
        (row["benchmark"], row["region"]): row for _, row in spectrum_summary_df.iterrows()
    } if not spectrum_summary_df.empty else {}
    dft_spec = spectrum_lookup.get(("DFT spectra", REGION_DISPLAY["full"]))
    exp_spec = spectrum_lookup.get(("Experimental spectra", REGION_DISPLAY["measured_support"]))

    dft_subbands = pairwise_summary_df[
        (pairwise_summary_df["benchmark"] == "dft_peak")
        & (pairwise_summary_df["pair"] == "DFT->Pred")
        & (pairwise_summary_df["tol_cm"] == 10.0)
        & (pairwise_summary_df["snr_bin"] == "all")
        & (pairwise_summary_df["region"].isin(["fp_400_800", "fp_800_1200", "fp_1200_1600", "fp_1600_1800"]))
    ].copy()
    exp_subbands = pairwise_summary_df[
        (pairwise_summary_df["benchmark"] == "experimental_peak")
        & (pairwise_summary_df["pair"] == "Exp->Pred")
        & (pairwise_summary_df["tol_cm"] == 10.0)
        & (pairwise_summary_df["snr_bin"] == "all")
        & (pairwise_summary_df["region"].isin(["fp_400_800", "fp_800_1200", "fp_1200_1600", "fp_1600_1800"]))
    ].copy()

    paragraphs: list[str] = []
    paragraphs.append(
        "Metric interpretation: coverage@δ is the fraction of reference lines with at least one predicted neighbor inside ±δ cm^-1. "
        "Intensity-weighted coverage emphasizes the stronger reference modes. "
        "Median |Δν| is the typical positional error among matched lines, while p90 |Δν| describes the upper tail for most matched lines. "
        "Intensity MAE is reported in log10 units, so 0.30 is about a 2x typical multiplicative error and 0.48 is about a 3x error."
    )
    if dft_raw_10 is not None:
        pooled_coverage = float(dft_raw_10["global_coverage_any"])
        pooled_coverage5 = float(dft_raw_5["global_coverage_any"]) if dft_raw_5 is not None else math.nan
        weighted_coverage = float(dft_raw_10["global_weighted_coverage_any"])
        count_ratio = float(dft_raw_10["total_target_lines"] / dft_raw_10["total_source_lines"]) if float(dft_raw_10["total_source_lines"]) > 0 else math.nan
        pooled_median_abs = float(dft_raw_10["median_abs_dnu_cm"])
        pooled_p90_abs = float(dft_raw_10["p90_abs_dnu_cm"])
        text = (
            f"On the DFT mode-position benchmark, pooled unweighted line coverage is {_fmt_pct(pooled_coverage)} within ±10 cm^-1 and "
            f"{_fmt_pct(pooled_coverage5)} within ±5 cm^-1. "
            f"Intensity-weighted coverage at ±10 cm^-1 is {_fmt_pct(weighted_coverage)}. "
            f"The predicted/reference mode-count ratio is {_fmt_num(count_ratio, 2)}, so this benchmark is not artificially limited by suppressing predicted modes before matching. "
            f"Across all matched lines, pooled median |Δν| is {_fmt_num(pooled_median_abs)} cm^-1 and pooled 90th-percentile |Δν| is {_fmt_num(pooled_p90_abs)} cm^-1."
        )
        if np.isfinite(float(dft_raw_10["mae_log10_ratio"])):
            text += (
                f" Matched-line intensity MAE is {_fmt_num(float(dft_raw_10['mae_log10_ratio']))} in log10 units "
                f"(about {_fmt_num(_log10_mae_to_fold(float(dft_raw_10['mae_log10_ratio'])), 2)}x multiplicative error)."
            )
        if dft_raw_unc is not None:
            text += (
                f" On a molecule-level summary, mean unweighted coverage at ±10 cm^-1 is {_fmt_pct(float(dft_raw_unc['coverage_mean_point']))} "
                f"with bootstrap 95% CI {_format_ci(dft_raw_unc['coverage_mean_ci95'], pct=True)}, while mean intensity-weighted coverage is {_fmt_pct(float(dft_raw_unc['weighted_coverage_mean_point']))} "
                f"with 95% CI {_format_ci(dft_raw_unc['weighted_coverage_mean_ci95'], pct=True)}. "
                f"The median of per-molecule median |Δν| values is {_fmt_num(float(dft_raw_unc['median_abs_dnu_case_median_point']))} cm^-1 "
                f"with 95% CI {_format_ci(dft_raw_unc['median_abs_dnu_median_ci95'])}."
            )
        if dft_spec is not None:
            text += (
                f" At the curve level, full-range spectrum cosine similarity has median {_fmt_num(float(dft_spec['cosine_median']), 3)} "
                f"and mean {_fmt_num(float(dft_spec['cosine_mean']), 3)} with 95% CI {_format_ci(dft_spec['cosine_mean_ci95'], digits=3)}."
            )
        attainment = float(summary.get("assessment", {}).get("attainment_vs_75_target", math.nan))
        if np.isfinite(attainment):
            text += f" Relative to the older aspirational 75% coverage target, the model is currently at {_fmt_pct(attainment)} of that target."
        if weighted_cross_tol is None:
            text += " The secondary peak-resolution weighted coverage sweep never reaches 60% across the tested 5..20 cm^-1 window."
        else:
            text += f" The secondary peak-resolution weighted coverage sweep first reaches 60% at ±{int(weighted_cross_tol)} cm^-1."
        if strict_cross_tol is None:
            text += " The 75% unweighted position-recall target is never reached in this window."
        else:
            text += f" The 75% unweighted position-recall target is first reached at ±{int(strict_cross_tol)} cm^-1."
        paragraphs.append(text)

    if dft_peak_10 is not None:
        raw_cov = float(dft_raw_10["global_coverage_any"]) if dft_raw_10 is not None else math.nan
        peak_cov = float(dft_peak_10["global_coverage_any"])
        peak_weighted_cov = float(dft_peak_10["global_weighted_coverage_any"])
        peak_count_ratio = float(dft_peak_10["total_target_lines"] / dft_peak_10["total_source_lines"]) if float(dft_peak_10["total_source_lines"]) > 0 else math.nan
        relation = _compare_phrase(
            peak_cov,
            peak_cov,
            up="",
            down="",
            same="",
            tol=0.02,
        )
        del relation
        text = (
            f"On the DFT peak-resolution benchmark, pooled unweighted peak coverage at ±10 cm^-1 is {_fmt_pct(peak_cov)} and "
            f"{_fmt_pct(float(dft_peak_5['global_coverage_any'])) if dft_peak_5 is not None else 'NA'} at ±5 cm^-1. "
            f"Intensity-weighted peak coverage at ±10 cm^-1 is {_fmt_pct(peak_weighted_cov)}. "
            f"Compared with the all-mode benchmark, this is stricter because broadening and peak picking merge nearby modes and discard weak local maxima."
        )
        if dft_peak_unc is not None:
            text += (
                f" On a molecule-level summary, mean peak-resolution intensity-weighted coverage is {_fmt_pct(float(dft_peak_unc['weighted_coverage_mean_point']))} "
                f"with 95% CI {_format_ci(dft_peak_unc['weighted_coverage_mean_ci95'], pct=True)}."
            )
        if np.isfinite(peak_count_ratio) and peak_count_ratio < 0.85:
            text += f" The predicted/reference resolved-peak count ratio is {_fmt_num(peak_count_ratio, 2)}, so strict peak recall is partly capped by peak merging or suppression."
        if np.isfinite(raw_cov) and np.isfinite(peak_cov):
            text += " " + _compare_phrase(
                peak_cov,
                raw_cov,
                up="Peak-resolution coverage is higher than the all-mode benchmark, which suggests the hardest errors sit in weak or crowded modes rather than the dominant visible structure.",
                down="Peak-resolution coverage is lower than the all-mode benchmark, which is consistent with peak extraction and broadening suppressing recoverable matches after the raw modes are generated.",
                same="Peak-resolution and all-mode coverage are similar, so line-to-peak conversion is not changing the overall conclusion much.",
                tol=0.03,
            )
        if not dft_subbands.empty:
            best = dft_subbands.sort_values("global_weighted_coverage_any", ascending=False).iloc[0]
            worst = dft_subbands.sort_values("global_weighted_coverage_any", ascending=True).iloc[0]
            text += (
                f" Within the fingerprint sub-bands, the strongest intensity-weighted region is {REGION_DISPLAY.get(str(best['region']), str(best['region']))} at {_fmt_pct(float(best['global_weighted_coverage_any']))}, "
                f"while the weakest is {REGION_DISPLAY.get(str(worst['region']), str(worst['region']))} at {_fmt_pct(float(worst['global_weighted_coverage_any']))}."
            )
        if not scale_sweep_df.empty:
            base_scale = float(summary.get("settings", {}).get("base_freq_scale_factor", math.nan))
            for bench, label in [("dft_raw_line", "raw-line"), ("dft_peak", "peak-resolution")]:
                metric_col = "global_weighted_coverage_any" if bench == "dft_peak" else "global_coverage_any"
                scale_full = scale_sweep_df[
                    (scale_sweep_df["benchmark"] == bench)
                    & (scale_sweep_df["pair"] == "DFT->Pred")
                    & (scale_sweep_df["region"] == "full")
                ].sort_values(metric_col, ascending=False)
                if scale_full.empty:
                    continue
                best = scale_full.iloc[0]
                current = scale_full.loc[np.isclose(scale_full["scale_adjust"], 1.0)]
                current_cov = float(current[metric_col].iloc[0]) if not current.empty else math.nan
                gain = float(best[metric_col]) - current_cov if np.isfinite(current_cov) else math.nan
                effective = float(best["scale_adjust"]) * base_scale if np.isfinite(base_scale) else math.nan
                text += (
                    f" For the {label} benchmark, the best extra scale factor is {_fmt_num(float(best['scale_adjust']), 3)}"
                    + (f" (effective total scale {_fmt_num(effective, 3)})" if np.isfinite(effective) else "")
                    + f", giving {'weighted ' if bench == 'dft_peak' else ''}coverage {_fmt_pct(float(best[metric_col]))}; the absolute gain over the current scale is {_fmt_pct(gain) if np.isfinite(gain) else 'NA'}."
                )
        paragraphs.append(text)

    if exp_ms_10 is not None or exp_fp_10 is not None:
        overall_fp = float(exp_fp_10["global_coverage_any"]) if exp_fp_10 is not None else math.nan
        weighted_fp = float(exp_fp_10["global_weighted_coverage_any"]) if exp_fp_10 is not None else math.nan
        low_fp = float(low_snr_fp["global_weighted_coverage_any"]) if low_snr_fp is not None else math.nan
        snr_relation = _compare_phrase(
            low_fp,
            weighted_fp,
            up="Low-SNR cases actually have higher fingerprint coverage than the overall pool, so SNR alone does not explain the aggregate weakness.",
            down="Low-SNR cases are materially worse than the overall pool, so noise is a plausible contributor to missed peaks.",
            same="Low-SNR fingerprint coverage is close to the overall fingerprint coverage.",
            tol=0.03,
        )
        text = (
            f"On experimental spectra, pooled unweighted coverage is {_fmt_pct(float(exp_ms_10['global_coverage_any'])) if exp_ms_10 is not None else 'NA'} over measured support and "
            f"{_fmt_pct(overall_fp)} in the fingerprint region at ±10 cm^-1. "
            f"Intensity-weighted fingerprint coverage is {_fmt_pct(weighted_fp)}, which downweights weak or ambiguous peaks. "
            f"Across matched fingerprint peaks, pooled median |Δν| is {_fmt_num(float(exp_fp_10['median_abs_dnu_cm'])) if exp_fp_10 is not None else 'NA'} cm^-1 and pooled intensity MAE is {_fmt_num(float(exp_fp_10['mae_log10_ratio'])) if exp_fp_10 is not None else 'NA'} in log10 units"
            + (
                f" (about {_fmt_num(_log10_mae_to_fold(float(exp_fp_10['mae_log10_ratio'])), 2)}x in multiplicative terms)."
                if exp_fp_10 is not None and np.isfinite(float(exp_fp_10["mae_log10_ratio"]))
                else "."
            )
        )
        if exp_unc is not None:
            text += (
                f" On a molecule-level summary, mean fingerprint coverage is {_fmt_pct(float(exp_unc['coverage_mean_point']))} "
                f"with bootstrap 95% CI {_format_ci(exp_unc['coverage_mean_ci95'], pct=True)}, while mean intensity-weighted fingerprint coverage is {_fmt_pct(float(exp_unc['weighted_coverage_mean_point']))} "
                f"with 95% CI {_format_ci(exp_unc['weighted_coverage_mean_ci95'], pct=True)}. "
                f"The median of per-molecule median |Δν| values is {_fmt_num(float(exp_unc['median_abs_dnu_case_median_point']))} cm^-1 "
                f"with 95% CI {_format_ci(exp_unc['median_abs_dnu_median_ci95'])}."
            )
        if exp_spec is not None:
            text += (
                f" The measured-support spectrum cosine similarity has median {_fmt_num(float(exp_spec['cosine_median']), 3)} "
                f"and mean {_fmt_num(float(exp_spec['cosine_mean']), 3)} with 95% CI {_format_ci(exp_spec['cosine_mean_ci95'], digits=3)}."
            )
        if not exp_subbands.empty:
            best = exp_subbands.sort_values("global_weighted_coverage_any", ascending=False).iloc[0]
            worst = exp_subbands.sort_values("global_weighted_coverage_any", ascending=True).iloc[0]
            text += (
                f" Within the experimental fingerprint sub-bands, the strongest intensity-weighted region is {REGION_DISPLAY.get(str(best['region']), str(best['region']))} at {_fmt_pct(float(best['global_weighted_coverage_any']))}, "
                f"while the weakest is {REGION_DISPLAY.get(str(worst['region']), str(worst['region']))} at {_fmt_pct(float(worst['global_weighted_coverage_any']))}."
            )
        if low_snr_fp is not None:
            text += f" In the low-SNR fingerprint stratum, pooled intensity-weighted coverage is {_fmt_pct(low_fp)}. {snr_relation}"
        paragraphs.append(text)

    if not tost_df.empty:
        dft_tost = tost_df[
            (tost_df["benchmark"] == BENCHMARK_DISPLAY["dft_peak"])
            & (tost_df["pair"] == "DFT->Pred")
            & (tost_df["region"] == REGION_DISPLAY["full"])
        ]
        exp_tost = tost_df[
            (tost_df["benchmark"] == BENCHMARK_DISPLAY["experimental_peak"])
            & (tost_df["pair"] == "Exp->Pred")
            & (tost_df["region"] == REGION_DISPLAY["fingerprint"])
        ]
        tost_bits: list[str] = []
        if not dft_tost.empty:
            row = dft_tost.iloc[0]
            tost_bits.append(
                f"On the DFT benchmark, the signed position-bias equivalence test {'passes' if bool(row['position_bias_passes']) else 'does not pass'} "
                f"(p={_fmt_num(float(row['position_bias_pvalue']), 3)}), and the signed intensity-bias equivalence test "
                f"{'passes' if bool(row['intensity_bias_passes']) else 'does not pass'} (p={_fmt_num(float(row['intensity_bias_pvalue']), 3)})."
            )
        if not exp_tost.empty:
            row = exp_tost.iloc[0]
            tost_bits.append(
                f"On experimental fingerprint peaks, the signed position-bias equivalence test {'passes' if bool(row['position_bias_passes']) else 'does not pass'} "
                f"(p={_fmt_num(float(row['position_bias_pvalue']), 3)}), and the signed intensity-bias equivalence test "
                f"{'passes' if bool(row['intensity_bias_passes']) else 'does not pass'} (p={_fmt_num(float(row['intensity_bias_pvalue']), 3)})."
            )
        if tost_bits:
            paragraphs.append(" ".join(tost_bits))

    gate = summary.get("gates", {})
    assess = summary.get("assessment", {})
    overall = assess.get("overall_assessment", gate.get("final_status", "NA"))
    benchmark_status = assess.get("benchmark_status", "NA")
    primary_target = assess.get("primary_target", 0.60)
    failed = gate.get("failed", [])
    caveats = gate.get("caveats", [])
    text = (
        f"Overall assessment: **{overall}**. "
        f"Primary DFT position benchmark status: **{benchmark_status}** at coverage@10 against a {100.0 * float(primary_target):.0f}% target. "
        f""
    )
    if failed:
        text += " Benchmark failures: " + "; ".join(failed) + "."
    if caveats:
        text += " Caveats: " + "; ".join(caveats) + "."
    paragraphs.append(text)
    return "\n\n".join(paragraphs)


def run_raman_stats_analysis(
    *,
    repo_root: Path,
    out_dir: Path,
    resolver_cache_json: Path,
    experimental_df: pd.DataFrame,
    resolver_cache: dict[str, Any],
    session: Any,
    x_grid: np.ndarray,
    max_eval_rows: int | None,
    max_atoms_for_inference: int,
    sigma: float,
    temp: float,
    init_wl: float,
    predict_fn: Callable[[list[list[float]], list[int], np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
    parse_float_list: Callable[[Any], np.ndarray],
    resample_experimental_to_grid: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    lines_to_norm_spectrum: Callable[..., np.ndarray],
    component_name_candidates: Callable[[str], list[str]],
    fetch_pubchem_name_record: Callable[[str, Any], dict[str, Any] | None],
    dft_max_cases: int | None = None,
    dft_scan_limit: int | None = None,
    base_freq_scale_factor: float = 1.0,
    dft_sample_seed: int = DFT_SAMPLE_SEED,
) -> dict[str, Any]:
    del session, component_name_candidates, fetch_pubchem_name_record
    out_dir.mkdir(parents=True, exist_ok=True)
    dft_db_path = repo_root / "ramanchembl_pipeline" / "dataset" / "molecule.db"
    dx_cm = float(np.median(np.diff(x_grid)))
    if dft_max_cases is None:
        dft_max_cases = int(max_eval_rows) if max_eval_rows is not None else 512
    if dft_scan_limit is None:
        dft_scan_limit = max(int(dft_max_cases) * 4, 2048)

    print("dft_db_path       =", dft_db_path)
    print("dft_scan_limit    =", int(dft_scan_limit))
    print("dft_max_cases     =", int(dft_max_cases))
    print("dft_sample_seed   =", int(dft_sample_seed))

    work_df = experimental_df.copy()
    work_df["resolver"] = work_df["component"].map(lambda name: resolver_cache.get(str(name), {}))
    work_df["is_resolved"] = work_df["resolver"].map(lambda rec: rec.get("status") == "resolved")
    work_df = work_df[work_df["is_resolved"]].reset_index(drop=True)
    if max_eval_rows is not None:
        work_df = work_df.head(int(max_eval_rows)).copy()
    print("rows_to_evaluate  =", len(work_df))

    cases: list[dict[str, Any]] = []
    molecule_rows: list[dict[str, Any]] = []
    spectrum_metric_rows: list[dict[str, Any]] = []
    exp_case_count = 0
    dft_case_count = 0
    dft_scanned = 0

    for idx, row in work_df.iterrows():
        component = str(row["component"])
        resolver_rec = dict(row["resolver"] or {})
        pos = resolver_rec.get("pos")
        z = resolver_rec.get("z")
        if not pos or not z or len(z) > int(max_atoms_for_inference):
            continue
        x_exp_raw = parse_float_list(row["wavenumbers_arr"])
        y_exp_raw = parse_float_list(row["intensity_arr"])
        y_exp = resample_experimental_to_grid(x_exp_raw, y_exp_raw, x_grid)
        if not np.isfinite(y_exp).all() or float(np.max(y_exp)) <= 0.0:
            continue
        support_lo = float(np.min(x_exp_raw)) if x_exp_raw.size else math.nan
        support_hi = float(np.max(x_exp_raw)) if x_exp_raw.size else math.nan
        support_mask = (x_grid >= support_lo) & (x_grid <= support_hi)
        if int(support_mask.sum()) < 5:
            continue
        try:
            y_pred, freq_pred, act_pred = predict_fn(pos, z, x_grid)
        except Exception as exc:
            print(f"[{idx + 1}/{len(work_df)}] skipped {component}: {exc}")
            continue

        exp_peak_freq, exp_peak_int = _extract_peaks(x_grid[support_mask], y_exp[support_mask])
        pred_peak_freq, pred_peak_int = _extract_peaks(x_grid[support_mask], _normalize_intensity(y_pred[support_mask]))
        snr = _estimate_snr(y_exp[support_mask])
        lag_full = _cross_correlation_lag_cm(y_exp[support_mask], _normalize_intensity(y_pred[support_mask]), dx_cm)
        fp_region = _resolve_region({"benchmark_group": "experimental", "support_range_cm": (support_lo, support_hi)}, "fingerprint")
        if fp_region is not None:
            fp_mask = (x_grid >= float(fp_region[0])) & (x_grid <= float(fp_region[1]))
            lag_fp = _cross_correlation_lag_cm(y_exp[fp_mask], _normalize_intensity(y_pred[fp_mask]), dx_cm)
            spec_region_metrics = _spectrum_similarity(y_exp[fp_mask], _normalize_intensity(y_pred[fp_mask]))
            spectrum_metric_rows.append(
                {
                    "case_id": f"exp:{int(row['id'])}",
                    "benchmark": "experimental_spectrum",
                    "row_id": int(row["id"]),
                    "component": component,
                    "pair": "Exp-Pred",
                    "region": "fingerprint",
                    **spec_region_metrics,
                }
            )
        else:
            lag_fp = math.nan
        spec_support_metrics = _spectrum_similarity(y_exp[support_mask], _normalize_intensity(y_pred[support_mask]))
        spectrum_metric_rows.append(
            {
                "case_id": f"exp:{int(row['id'])}",
                "benchmark": "experimental_spectrum",
                "row_id": int(row["id"]),
                "component": component,
                "pair": "Exp-Pred",
                "region": "measured_support",
                **spec_support_metrics,
            }
        )

        case_id = f"exp:{int(row['id'])}"
        molecule_rows.append(
            {
                "case_id": case_id,
                "benchmark_group": "experimental",
                "row_id": int(row["id"]),
                "component": component,
                "query_name": resolver_rec.get("query_name"),
                "cid": resolver_rec.get("cid"),
                "smiles": resolver_rec.get("canonical_smiles"),
                "n_atoms": int(len(z)),
                "n_pred_modes": int(len(freq_pred)),
                "n_exp_peaks": int(len(exp_peak_freq)),
                "n_pred_peaks": int(len(pred_peak_freq)),
                "n_dft_lines": 0,
                "n_dft_peaks": 0,
                "snr_mad": snr["snr_mad"],
                "snr_rms": snr["snr_rms"],
                "signal_max": snr["signal_max"],
                "noise_mad": snr["noise_mad"],
                "noise_rms": snr["noise_rms"],
                "lag_ref_pred_measured_support_cm": lag_full,
                "lag_ref_pred_fingerprint_cm": lag_fp,
                "exp_range_min_cm": support_lo,
                "exp_range_max_cm": support_hi,
                "dft_match_status": "not_applicable",
            }
        )
        cases.append(
            {
                "case_id": case_id,
                "benchmark_group": "experimental",
                "row_id": int(row["id"]),
                "component": component,
                "support_range_cm": (support_lo, support_hi),
                "line_sets": {
                    "ExpPeak": {"freq": exp_peak_freq, "intensity": exp_peak_int},
                    "PredPeak": {"freq": pred_peak_freq, "intensity": pred_peak_int},
                },
                "spectra": {
                    "Exp": _normalize_intensity(y_exp),
                    "Pred": _normalize_intensity(y_pred),
                },
                "comparisons": [
                    {"benchmark": "experimental_peak", "pair_label": "Exp->Pred", "source_set": "ExpPeak", "target_set": "PredPeak", "regions": list(DYNAMIC_REGIONS.keys())},
                    {"benchmark": "experimental_peak", "pair_label": "Pred->Exp", "source_set": "PredPeak", "target_set": "ExpPeak", "regions": list(DYNAMIC_REGIONS.keys())},
                ],
            }
        )
        exp_case_count += 1
        if (idx + 1) % 10 == 0:
            print(f"processed experimental {idx + 1}/{len(work_df)}")

    con = sqlite3.connect(str(dft_db_path))
    try:
        cur = con.cursor()
        cur.execute("SELECT id FROM molecule WHERE blob_data IS NOT NULL")
        all_ids = np.asarray([int(row[0]) for row in cur.fetchall()], dtype=np.int64)
        if all_ids.size == 0:
            raise RuntimeError("No DFT rows with blob_data found in molecule.db")
        rng = np.random.default_rng(int(dft_sample_seed))
        chosen_ids = rng.choice(all_ids, size=min(int(dft_scan_limit), int(all_ids.size)), replace=False)
        chosen_lookup = {int(v): idx for idx, v in enumerate(chosen_ids.tolist())}
        placeholders = ",".join("?" for _ in chosen_lookup)
        cur.execute(
            f"SELECT id, SMILES, sdf_name, database_tag, blob_data FROM molecule WHERE id IN ({placeholders})",
            tuple(int(v) for v in chosen_lookup),
        )
        sampled_rows = cur.fetchall()
        sampled_rows.sort(key=lambda row: chosen_lookup[int(row[0])])
        for rid, smiles, sdf_name, database_tag, blob in sampled_rows:
            dft_scanned += 1
            if dft_case_count >= int(dft_max_cases):
                break
            try:
                payload = _decode_dft_blob(blob)
            except Exception:
                continue
            required = ("atoms", "coord", "freq", "Raman Activ")
            if not all(key in payload for key in required):
                continue
            atoms = np.asarray(payload["atoms"], dtype=np.int64)
            coords = np.asarray(payload["coord"], dtype=np.float32)
            dft_freq_raw = _safe_array(payload["freq"])
            dft_int_raw = _safe_array(payload["Raman Activ"])
            if atoms.ndim != 1 or coords.ndim != 2 or coords.shape[1] != 3:
                continue
            if len(atoms) > int(max_atoms_for_inference):
                continue
            dft_freq_all, dft_int_all = _prepare_mode_lines(dft_freq_raw, dft_int_raw)
            if dft_freq_all.size == 0:
                continue
            try:
                y_pred, pred_freq_raw, pred_int_raw = predict_fn(coords, atoms, x_grid)
            except Exception:
                continue
            pred_freq_all, pred_int_all = _prepare_mode_lines(pred_freq_raw, pred_int_raw)
            spec_dft = _normalize_intensity(lines_to_norm_spectrum(dft_freq_raw, dft_int_raw, x_grid, sigma=float(sigma), temp=float(temp), init_wl=float(init_wl)))
            pred_peak_freq, pred_peak_int = _extract_peaks(x_grid, _normalize_intensity(y_pred))
            dft_peak_freq, dft_peak_int = _extract_peaks(x_grid, spec_dft)
            full_mask = _mask_grid_region(x_grid, STATIC_REGIONS["full"])
            fp_mask = _mask_grid_region(x_grid, STATIC_REGIONS["fingerprint"])
            lag_full = _cross_correlation_lag_cm(spec_dft[full_mask], _normalize_intensity(y_pred)[full_mask], dx_cm)
            lag_fp = _cross_correlation_lag_cm(spec_dft[fp_mask], _normalize_intensity(y_pred)[fp_mask], dx_cm)
            component = str(sdf_name) if sdf_name is not None else (str(smiles) if smiles is not None else f"molecule_{int(rid)}")
            case_id = f"dft:{int(rid)}"
            molecule_rows.append(
                {
                    "case_id": case_id,
                    "benchmark_group": "dft",
                    "row_id": int(rid),
                    "component": component,
                    "query_name": None,
                    "cid": math.nan,
                    "smiles": str(smiles) if smiles is not None else None,
                    "database_tag": _clean_database_tag(database_tag),
                    "n_atoms": int(len(atoms)),
                    "n_pred_modes": int(len(pred_freq_raw)),
                    "n_exp_peaks": 0,
                    "n_pred_peaks": int(len(pred_peak_freq)),
                    "n_dft_lines": int(len(dft_freq_all)),
                    "n_dft_peaks": int(len(dft_peak_freq)),
                    "snr_mad": math.nan,
                    "snr_rms": math.nan,
                    "signal_max": math.nan,
                    "noise_mad": math.nan,
                    "noise_rms": math.nan,
                    "lag_ref_pred_measured_support_cm": math.nan,
                    "lag_ref_pred_fingerprint_cm": lag_fp,
                    "lag_ref_pred_full_cm": lag_full,
                    "dft_match_status": "direct_ramanchembl_row",
                    "exp_range_min_cm": math.nan,
                    "exp_range_max_cm": math.nan,
                }
            )
            cases.append(
                {
                    "case_id": case_id,
                    "benchmark_group": "dft",
                    "row_id": int(rid),
                    "component": component,
                    "support_range_cm": None,
                    "line_sets": {
                        "DFTRaw": {"freq": dft_freq_all, "intensity": dft_int_all},
                        "PredRaw": {"freq": pred_freq_all, "intensity": pred_int_all},
                        "DFTPeak": {"freq": dft_peak_freq, "intensity": dft_peak_int},
                        "PredPeak": {"freq": pred_peak_freq, "intensity": pred_peak_int},
                    },
                    "spectra": {
                        "DFT": spec_dft,
                        "Pred": _normalize_intensity(y_pred),
                    },
                    "comparisons": [
                        {"benchmark": "dft_raw_line", "pair_label": "DFT->Pred", "source_set": "DFTRaw", "target_set": "PredRaw", "regions": list(STATIC_REGIONS.keys())},
                        {"benchmark": "dft_raw_line", "pair_label": "Pred->DFT", "source_set": "PredRaw", "target_set": "DFTRaw", "regions": list(STATIC_REGIONS.keys())},
                        {"benchmark": "dft_peak", "pair_label": "DFT->Pred", "source_set": "DFTPeak", "target_set": "PredPeak", "regions": list(STATIC_REGIONS.keys())},
                        {"benchmark": "dft_peak", "pair_label": "Pred->DFT", "source_set": "PredPeak", "target_set": "DFTPeak", "regions": list(STATIC_REGIONS.keys())},
                    ],
                }
            )
            dft_case_count += 1
            spectrum_metric_rows.append(
                {
                    "case_id": case_id,
                    "benchmark": "dft_spectrum",
                    "row_id": int(rid),
                    "component": component,
                    "pair": "DFT-Pred",
                    "region": "full",
                    **_spectrum_similarity(spec_dft[full_mask], _normalize_intensity(y_pred)[full_mask]),
                }
            )
            spectrum_metric_rows.append(
                {
                    "case_id": case_id,
                    "benchmark": "dft_spectrum",
                    "row_id": int(rid),
                    "component": component,
                    "pair": "DFT-Pred",
                    "region": "fingerprint",
                    **_spectrum_similarity(spec_dft[fp_mask], _normalize_intensity(y_pred)[fp_mask]),
                }
            )
    finally:
        con.close()

    resolver_cache_json.write_text(json.dumps(resolver_cache, indent=2))
    if not cases:
        raise RuntimeError("No evaluable cases were produced.")

    molecule_df = pd.DataFrame(molecule_rows).sort_values(["benchmark_group", "row_id"]).reset_index(drop=True)
    molecule_df["snr_bin"] = "not_applicable"
    exp_mask = molecule_df["benchmark_group"] == "experimental"
    if exp_mask.any():
        molecule_df.loc[exp_mask, "snr_bin"] = _assign_snr_bins(molecule_df.loc[exp_mask, "snr_mad"]).astype(object).to_numpy()
    case_to_bin = molecule_df.set_index("case_id")["snr_bin"].to_dict()
    for case in cases:
        case["snr_bin"] = case_to_bin.get(case["case_id"], "not_applicable")

    pair_case_rows: list[dict[str, Any]] = []
    line_level_rows: list[dict[str, Any]] = []
    for case in cases:
        for comp in case["comparisons"]:
            for region_name in comp["regions"]:
                for tol_cm in SWEEP_TOLS:
                    row, line_rows = _build_case_pair_metrics(case, comp, region_name, float(tol_cm))
                    if row is None:
                        continue
                    pair_case_rows.append(row)
                    line_level_rows.extend(line_rows)

    pair_case_df = pd.DataFrame(pair_case_rows)
    line_level_df = pd.DataFrame(line_level_rows)
    spectrum_metric_df = pd.DataFrame(spectrum_metric_rows)
    pairwise_summary_df = _aggregate_pairwise_summary(pair_case_df, line_level_df)
    scale_sweep_df = _build_scale_sweep(
        cases,
        x_grid=x_grid,
        lines_to_norm_spectrum=lines_to_norm_spectrum,
        sigma=float(sigma),
        temp=float(temp),
        init_wl=float(init_wl),
    )

    key_metrics = [
        "coverage_any",
        "weighted_coverage_any",
        "missed_pct",
        "weighted_missed_pct",
        "precision",
        "recall",
        "f1",
        "median_abs_dnu_cm",
        "p90_abs_dnu_cm",
        "mean_signed_dnu_cm",
        "median_abs_rel_pct_dnu",
        "mae_log10_ratio",
        "spearman_intensity",
    ]
    mandatory_case_df = pair_case_df[pair_case_df["tol_cm"].isin(MANDATORY_TOLS)].copy()
    wide_case_df = _flatten_case_metrics(mandatory_case_df, key_metrics)
    per_molecule_df = molecule_df.merge(wide_case_df, on="case_id", how="left")

    uncertainty_df = _build_uncertainty_summary(pair_case_df)
    spectrum_summary_df = _build_spectrum_summary(spectrum_metric_df)
    executive_summary_df = _build_executive_summary(pairwise_summary_df, spectrum_summary_df)
    tost_df = _build_tost_df(pair_case_df)

    dft_peak_full_10 = _get_summary_row(pairwise_summary_df, "dft_peak", "DFT->Pred", "full", 10.0, "all")
    dft_peak_fp_10 = _get_summary_row(pairwise_summary_df, "dft_peak", "DFT->Pred", "fingerprint", 10.0, "all")
    dft_raw_full_10 = _get_summary_row(pairwise_summary_df, "dft_raw_line", "DFT->Pred", "full", 10.0, "all")
    exp_fp_10 = _get_summary_row(pairwise_summary_df, "experimental_peak", "Exp->Pred", "fingerprint", 10.0, "all")
    exp_ms_10 = _get_summary_row(pairwise_summary_df, "experimental_peak", "Exp->Pred", "measured_support", 10.0, "all")
    low_snr_fp_10 = _get_summary_row(pairwise_summary_df, "experimental_peak", "Exp->Pred", "fingerprint", 10.0, "low")
    dft_spec_full = spectrum_summary_df[
        (spectrum_summary_df["benchmark"] == "DFT spectra")
        & (spectrum_summary_df["region"] == REGION_DISPLAY["full"])
    ]
    exp_spec_support = spectrum_summary_df[
        (spectrum_summary_df["benchmark"] == "Experimental spectra")
        & (spectrum_summary_df["region"] == REGION_DISPLAY["measured_support"])
    ]
    dft_cosine_median = float(dft_spec_full["cosine_median"].iloc[0]) if not dft_spec_full.empty else math.nan
    exp_cosine_median = float(exp_spec_support["cosine_median"].iloc[0]) if not exp_spec_support.empty else math.nan

    benchmark_failures: list[str] = []
    caveats: list[str] = []
    if dft_case_count < 5:
        benchmark_failures.append("insufficient_dft_cases_for_benchmark")
    if dft_raw_full_10 is None or not np.isfinite(float(dft_raw_full_10["global_coverage_any"])):
        benchmark_failures.append("dft_raw_full_coverage10_not_assessable")
    else:
        dft_raw_cov = float(dft_raw_full_10["global_coverage_any"])
        if dft_raw_cov < 0.60:
            benchmark_failures.append("dft_raw_full_coverage10_below_0.60")
        elif dft_raw_cov < 0.70:
            caveats.append("dft_raw_full_coverage10_between_0.60_and_0.70")
        if dft_raw_cov < 0.75:
            caveats.append("dft_raw_full_coverage10_below_0.75_aspirational_target")
    if dft_raw_full_10 is not None and np.isfinite(float(dft_raw_full_10["median_abs_dnu_cm"])) and float(dft_raw_full_10["median_abs_dnu_cm"]) > 10.0:
        caveats.append("dft_raw_full_median_abs_dnu_above_10cm")
    if dft_peak_full_10 is not None and np.isfinite(float(dft_peak_full_10["global_weighted_coverage_any"])) and float(dft_peak_full_10["global_weighted_coverage_any"]) < 0.50:
        caveats.append("dft_peak_full_weighted_coverage10_below_0.50")
    if dft_peak_full_10 is not None and np.isfinite(float(dft_peak_full_10["global_coverage_any"])) and float(dft_peak_full_10["global_coverage_any"]) < 0.75:
        caveats.append("dft_peak_full_strict_unweighted_coverage10_below_0.75")
    if dft_peak_fp_10 is not None and np.isfinite(float(dft_peak_fp_10["global_weighted_coverage_any"])) and float(dft_peak_fp_10["global_weighted_coverage_any"]) < 0.60:
        caveats.append("dft_peak_fingerprint_weighted_coverage10_below_0.60")
    if exp_fp_10 is not None and np.isfinite(float(exp_fp_10["global_weighted_coverage_any"])) and float(exp_fp_10["global_weighted_coverage_any"]) < 0.40:
        caveats.append("experimental_fingerprint_weighted_coverage10_below_0.40")
    if low_snr_fp_10 is not None and exp_fp_10 is not None:
        if np.isfinite(float(low_snr_fp_10["global_weighted_coverage_any"])) and np.isfinite(float(exp_fp_10["global_weighted_coverage_any"])):
            if float(exp_fp_10["global_weighted_coverage_any"]) - float(low_snr_fp_10["global_weighted_coverage_any"]) > 0.15:
                caveats.append("low_snr_fingerprint_drop_exceeds_0.15")
    if np.isfinite(dft_cosine_median) and dft_cosine_median < 0.20:
        caveats.append("dft_full_spectrum_cosine_median_below_0.20")
    if np.isfinite(exp_cosine_median) and exp_cosine_median < 0.15:
        caveats.append("experimental_support_cosine_median_below_0.15")

    scale_diag: dict[str, Any] = {}
    if not scale_sweep_df.empty:
        for benchmark in ["dft_raw_line", "dft_peak"]:
            metric_col = "global_weighted_coverage_any" if benchmark == "dft_peak" else "global_coverage_any"
            scale_full = scale_sweep_df[
                (scale_sweep_df["benchmark"] == benchmark)
                & (scale_sweep_df["pair"] == "DFT->Pred")
                & (scale_sweep_df["region"] == "full")
            ].sort_values(metric_col, ascending=False)
            if scale_full.empty:
                continue
            best = scale_full.iloc[0]
            current = scale_full.loc[np.isclose(scale_full["scale_adjust"], 1.0)]
            current_cov = float(current[metric_col].iloc[0]) if not current.empty else math.nan
            gain = float(best[metric_col]) - current_cov if np.isfinite(current_cov) else math.nan
            scale_diag[benchmark] = {
                "optimized_metric": metric_col,
                "best_scale_adjust": float(best["scale_adjust"]),
                "best_effective_scale_factor": float(best["scale_adjust"] * float(base_freq_scale_factor)),
                "best_full_metric_value": float(best[metric_col]),
                "current_full_metric_value": current_cov,
                "absolute_gain": gain,
            }
            if np.isfinite(gain) and gain > 0.10 and abs(float(best["scale_adjust"]) - 1.0) >= 0.01:
                caveats.append(f"{benchmark}_coverage_is_scale_sensitive")

    tost_rows = [
        _collect_tost_summary(pair_case_df, "dft_peak", "DFT->Pred", "full", 10.0),
        _collect_tost_summary(pair_case_df, "experimental_peak", "Exp->Pred", "fingerprint", 10.0),
    ]
    if tost_rows[0]["position_bias_tost"].get("n", 0) >= 3 and not tost_rows[0]["position_bias_tost"].get("passes", False):
        caveats.append("dft_peak_position_bias_not_equivalent")
    if tost_rows[0]["intensity_bias_tost"].get("n", 0) >= 3 and not tost_rows[0]["intensity_bias_tost"].get("passes", False):
        caveats.append("dft_peak_intensity_bias_not_equivalent")

    assessment = _build_assessment(
        dft_raw_full_10=dft_raw_full_10,
        dft_peak_full_10=dft_peak_full_10,
        exp_fp_10=exp_fp_10,
        dft_cosine_median=dft_cosine_median,
        exp_cosine_median=exp_cosine_median,
        tost_df=tost_df,
        primary_target=0.60,
        aspirational_target=0.75,
    )
    overall_assessment = str(assessment["overall_assessment"])

    gotcha_high_fp = mandatory_case_df[
        (mandatory_case_df["benchmark"] == "experimental_peak")
        & (mandatory_case_df["pair"] == "Pred->Exp")
        & (mandatory_case_df["region"] == "measured_support")
        & (mandatory_case_df["tol_cm"] == 10.0)
    ].sort_values(["precision", "coverage_any"], ascending=[True, False]).head(10)
    gotcha_fingerprint_missed = mandatory_case_df[
        (mandatory_case_df["pair"] == "DFT->Pred")
        & (mandatory_case_df["region"] == "fingerprint")
        & (mandatory_case_df["tol_cm"] == 10.0)
        & (mandatory_case_df["benchmark"].isin(["dft_peak", "experimental_peak"]))
    ].sort_values("missed_pct", ascending=False).head(10)
    gotcha_drift = mandatory_case_df[
        (mandatory_case_df["pair"] == "DFT->Pred")
        & (mandatory_case_df["tol_cm"] == 10.0)
        & (mandatory_case_df["benchmark"].isin(["dft_peak", "experimental_peak"]))
    ].assign(abs_shift=lambda df: df["mean_signed_dnu_cm"].abs()).sort_values("abs_shift", ascending=False).head(10)
    gotcha_low_snr = mandatory_case_df[
        (mandatory_case_df["benchmark"] == "experimental_peak")
        & (mandatory_case_df["pair"] == "Exp->Pred")
        & (mandatory_case_df["region"] == "fingerprint")
        & (mandatory_case_df["tol_cm"] == 10.0)
        & (mandatory_case_df["snr_bin"] == "low")
    ].sort_values(["coverage_any", "median_abs_dnu_cm"], ascending=[True, False]).head(10)

    per_molecule_csv = out_dir / "stats_per_molecule.csv"
    line_level_csv = out_dir / "stats_line_level_matches.csv"
    pairwise_summary_csv = out_dir / "stats_pairwise_region_summary.csv"
    spectrum_similarity_csv = out_dir / "stats_spectrum_region_metrics.csv"
    summary_json = out_dir / "stats_summary.json"
    executive_summary_csv = out_dir / "stats_executive_summary.csv"
    uncertainty_csv = out_dir / "stats_uncertainty_summary.csv"
    spectrum_summary_csv = out_dir / "stats_spectrum_summary.csv"
    scale_sweep_csv = out_dir / "stats_scale_sweep.csv"
    narrative_md = out_dir / "stats_narrative.md"
    gotcha_high_fp_csv = out_dir / "stats_gotcha_high_fp.csv"
    gotcha_fingerprint_csv = out_dir / "stats_gotcha_fingerprint_missed.csv"
    gotcha_drift_csv = out_dir / "stats_gotcha_systematic_drift.csv"
    gotcha_low_snr_csv = out_dir / "stats_gotcha_low_snr.csv"

    per_molecule_df.to_csv(per_molecule_csv, index=False)
    line_level_df.to_csv(line_level_csv, index=False)
    pairwise_summary_df.to_csv(pairwise_summary_csv, index=False)
    spectrum_metric_df.to_csv(spectrum_similarity_csv, index=False)
    executive_summary_df.to_csv(executive_summary_csv, index=False)
    uncertainty_df.to_csv(uncertainty_csv, index=False)
    spectrum_summary_df.to_csv(spectrum_summary_csv, index=False)
    scale_sweep_df.to_csv(scale_sweep_csv, index=False)
    gotcha_high_fp.to_csv(gotcha_high_fp_csv, index=False)
    gotcha_fingerprint_missed.to_csv(gotcha_fingerprint_csv, index=False)
    gotcha_drift.to_csv(gotcha_drift_csv, index=False)
    gotcha_low_snr.to_csv(gotcha_low_snr_csv, index=False)

    plot_paths = {
        "tolerance_sweep": _save_plot_sweep(pairwise_summary_df, out_dir),
        "scale_sweep": _save_plot_scale_sweep(scale_sweep_df, out_dir),
        "error_distributions": _save_plot_error_distributions(line_level_df, out_dir),
        "region_boxes": _save_plot_region_boxes(pair_case_df, out_dir),
        "qq": _save_plot_qq(line_level_df, out_dir),
        "scatter": _save_plot_scatter(line_level_df, out_dir),
        "intensity_agreement": _save_plot_intensity_agreement(line_level_df, out_dir),
        "spectrum_similarity": _save_plot_spectrum_similarity(spectrum_metric_df, out_dir),
        "stick_overlays": _save_plot_stick_overlays(cases, per_molecule_df, spectrum_metric_df, out_dir),
        "broadened_overlays": _save_plot_broadened_overlays(cases, per_molecule_df, spectrum_metric_df, line_level_df, out_dir),
        "connectors": _save_plot_connectors(cases, per_molecule_df, spectrum_metric_df, pair_case_df, line_level_df, out_dir),
    }

    summary = {
        "settings": {
            "dft_db_path": str(dft_db_path),
            "dft_scan_limit": int(dft_scan_limit),
            "dft_max_cases": int(dft_max_cases),
            "dft_sample_seed": int(dft_sample_seed),
            "dft_sampling_strategy": "random_without_replacement",
            "dft_mode_position_uses_all_modes": True,
            "raw_line_min_rel_intensity": float(RAW_LINE_MIN_REL_INTENSITY),
            "peak_prominence_frac": float(PEAK_PROMINENCE_FRAC),
            "peak_min_distance_cm": float(PEAK_MIN_DISTANCE_CM),
            "mandatory_tols_cm": [float(x) for x in MANDATORY_TOLS],
            "sweep_tols_cm": [float(x) for x in SWEEP_TOLS.tolist()],
            "scale_adjust_grid": [float(x) for x in SCALE_ADJUST_GRID.tolist()],
            "base_freq_scale_factor": float(base_freq_scale_factor),
            "primary_gate": {
                "benchmark": "dft_raw_line",
                "pair": "DFT->Pred",
                "region": "full",
                "tol_cm": 10.0,
                "metric": "global_coverage_any",
                "min_value": 0.60,
                "aspirational_target": 0.75,
                "secondary_peak_weighted_target": 0.50,
            },
        },
        "counts": {
            "evaluated_cases_total": int(len(cases)),
            "evaluated_experimental_cases": int(exp_case_count),
            "evaluated_dft_cases": int(dft_case_count),
            "dft_scanned_rows": int(dft_scanned),
            "pair_case_rows": int(len(pair_case_df)),
            "line_level_rows": int(len(line_level_df)),
        },
        "primary_metrics": {
            "dft_raw_full_coverage10": float(dft_raw_full_10["global_coverage_any"]) if dft_raw_full_10 is not None else math.nan,
            "dft_raw_full_weighted_coverage10": float(dft_raw_full_10["global_weighted_coverage_any"]) if dft_raw_full_10 is not None else math.nan,
            "dft_peak_full_coverage10": float(dft_peak_full_10["global_coverage_any"]) if dft_peak_full_10 is not None else math.nan,
            "dft_peak_full_weighted_coverage10": float(dft_peak_full_10["global_weighted_coverage_any"]) if dft_peak_full_10 is not None else math.nan,
            "dft_peak_full_coverage5": float(_get_summary_row(pairwise_summary_df, "dft_peak", "DFT->Pred", "full", 5.0, "all")["global_coverage_any"]) if _get_summary_row(pairwise_summary_df, "dft_peak", "DFT->Pred", "full", 5.0, "all") is not None else math.nan,
            "experimental_measured_support_coverage10": float(exp_ms_10["global_coverage_any"]) if exp_ms_10 is not None else math.nan,
            "experimental_fingerprint_coverage10": float(exp_fp_10["global_coverage_any"]) if exp_fp_10 is not None else math.nan,
            "experimental_fingerprint_weighted_coverage10": float(exp_fp_10["global_weighted_coverage_any"]) if exp_fp_10 is not None else math.nan,
            "dft_full_cosine_median": dft_cosine_median,
            "experimental_support_cosine_median": exp_cosine_median,
        },
        "scale_diagnostics": scale_diag,
        "statistical_tests": json.loads(tost_df.to_json(orient="records")),
        "assessment": assessment,
        "gates": {
            "failed": benchmark_failures,
            "caveats": caveats,
            "final_status": overall_assessment,
        },
        "outputs": {
            "stats_per_molecule_csv": str(per_molecule_csv),
            "stats_line_level_matches_csv": str(line_level_csv),
            "stats_pairwise_region_summary_csv": str(pairwise_summary_csv),
            "stats_executive_summary_csv": str(executive_summary_csv),
            "stats_uncertainty_summary_csv": str(uncertainty_csv),
            "stats_spectrum_summary_csv": str(spectrum_summary_csv),
            "stats_scale_sweep_csv": str(scale_sweep_csv),
            "stats_summary_json": str(summary_json),
            "stats_narrative_md": str(narrative_md),
            "plots": {key: str(path) for key, path in plot_paths.items()},
        },
    }

    narrative_markdown = _build_narrative(summary, pairwise_summary_df, uncertainty_df, scale_sweep_df, spectrum_summary_df, tost_df)
    summary["narrative_markdown"] = narrative_markdown
    narrative_md.write_text(narrative_markdown + "\n")
    summary_json.write_text(json.dumps(summary, indent=2))

    print()
    print("OVERALL ASSESSMENT =", overall_assessment)
    print("benchmark_failures =", benchmark_failures)
    print("caveats      =", caveats)
    print("executive    =", executive_summary_csv)
    print("uncertainty  =", uncertainty_csv)
    print("summary_json =", summary_json)

    return {
        "cases": cases,
        "molecule_df": molecule_df,
        "per_molecule_df": per_molecule_df,
        "pair_case_df": pair_case_df,
        "line_level_df": line_level_df,
        "pairwise_summary_df": pairwise_summary_df,
        "spectrum_metric_df": spectrum_metric_df,
        "scale_sweep_df": scale_sweep_df,
        "executive_summary_df": executive_summary_df,
        "uncertainty_df": uncertainty_df,
        "spectrum_summary_df": spectrum_summary_df,
        "tost_df": tost_df,
        "summary": summary,
        "narrative_markdown": narrative_markdown,
        "gotchas": {
            "high_fp": gotcha_high_fp,
            "fingerprint_missed": gotcha_fingerprint_missed,
            "systematic_drift": gotcha_drift,
            "low_snr": gotcha_low_snr,
        },
        "plot_paths": plot_paths,
    }
