"""High-level calibration orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np

from .isotonic import IsotonicCalibrator
from sklearn.isotonic import IsotonicRegression
from .market_blend import MarketBlendConfig, MarketBlender
from .temperature import TemperatureCalibrator
from models.distribution import expected_scores


@dataclass
class CalibrationResult:
    isotonic: Optional[IsotonicCalibrator]
    temperature: Optional[TemperatureCalibrator]
    blender: MarketBlender


@dataclass
class CalibrationConfig:
    apply_isotonic: bool = True
    apply_temperature: bool = True
    apply_win_calibration: bool = True
    apply_ats_calibration: bool = True
    fit_margin_temperature: bool = True
    market_weight: float = 0.5
    market_weight_bounds: Tuple[float, float] = (0.0, 1.0)
    market_weight_reg: float = 0.0
    margin_low: int = -60
    margin_high: int = 60
    ats_bin_edges: Tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 10.0, 40.0)
    ats_split_fav_dog: bool = True
    fit_margin_temperature_per_bin: bool = True
    ats_blend_learn: bool = True
    ats_blend_reg: float = 0.0
    ats_min_bin_samples: int = 200
    ats_blend_shrink: float = 200.0
    ats_blend_cap: float = 0.5
    ats_clip_low: float = 0.05
    ats_clip_high: float = 0.95
    margin_temperature_grid: Tuple[float, ...] = (0.8, 0.9, 1.0, 1.1, 1.2)


class Calibrator:
    def __init__(self, config: CalibrationConfig | None = None) -> None:
        self.config = config or CalibrationConfig()
        self.isotonic: Optional[IsotonicCalibrator] = (
            IsotonicCalibrator() if self.config.apply_isotonic else None
        )
        self.temperature: Optional[TemperatureCalibrator] = (
            TemperatureCalibrator() if self.config.apply_temperature else None
        )
        self.blender = MarketBlender(
            MarketBlendConfig(
                weight=self.config.market_weight,
                weight_bounds=self.config.market_weight_bounds,
                regularization=self.config.market_weight_reg,
            )
        )
        self.margin_range: Tuple[int, int] = (self.config.margin_low, self.config.margin_high)
        self.sigma_home: Optional[np.ndarray] = None
        self.sigma_away: Optional[np.ndarray] = None
        # Distribution sharpness (temperature) for margin PMF
        self.margin_temperature: float = 1.0
        # Scalar probability calibrators
        self.win_iso: Optional[IsotonicRegression] = (
            IsotonicRegression(out_of_bounds="clip") if self.config.apply_win_calibration else None
        )
        self.ats_bin_edges: List[float] = list(self.config.ats_bin_edges)
        # ATS calibrators split by favorite/dog and |spread| bin
        if self.config.apply_ats_calibration:
            num_bins = len(self.ats_bin_edges) - 1
            self.ats_isos_fd: Optional[List[List[IsotonicRegression]]] = [
                [IsotonicRegression(out_of_bounds="clip") for _ in range(num_bins)],
                [IsotonicRegression(out_of_bounds="clip") for _ in range(num_bins)],
            ] if self.config.ats_split_fav_dog else None
            # Backward single-dimension fallback (unused if fd enabled)
            self.ats_isos: Optional[List[IsotonicRegression]] = [
                IsotonicRegression(out_of_bounds="clip") for _ in range(num_bins)
            ]
            # Per-bin blend weights w in [0,1] for cover: p_blend = p_raw + w*(p_iso - p_raw)
            if self.config.ats_split_fav_dog:
                self.ats_blend_fd: Optional[List[List[float]]] = [[0.5 for _ in range(num_bins)], [0.5 for _ in range(num_bins)]]
                self.ats_blend: Optional[List[float]] = None
            else:
                self.ats_blend_fd = None
                self.ats_blend = [0.5 for _ in range(num_bins)]
        else:
            self.ats_isos_fd = None
            self.ats_isos = None
            self.ats_blend_fd = None
            self.ats_blend = None
        # Per-bin temperatures (by |spread| bins)
        self.margin_temperature_bins: Optional[List[float]] = None

    def fit(
        self,
        margin_cdf: np.ndarray,
        joint_probs: np.ndarray,
        margin_outcomes: np.ndarray,
        score_outcomes_home: np.ndarray,
        score_outcomes_away: np.ndarray,
        market_home: Optional[np.ndarray] = None,
        market_away: Optional[np.ndarray] = None,
        sigma_home: Optional[np.ndarray] = None,
        sigma_away: Optional[np.ndarray] = None,
    ) -> CalibrationResult:
        if margin_cdf.ndim != 2:
            raise ValueError("margin_cdf must be [n_samples, n_margin_bins]")
        if joint_probs.ndim != 3:
            raise ValueError("joint_probs must be [n_samples, score_max+1, score_max+1]")
        # Optionally fit a global temperature for the margin PMF to minimize CRPS
        # Recover PMF from provided CDF
        pmf_raw = np.diff(
            np.concatenate([np.zeros((len(margin_cdf), 1)), margin_cdf], axis=1),
            axis=1,
        )
        if self.config.fit_margin_temperature:
            best_t = 1.0
            best_crps = float("inf")
            for t in self.config.margin_temperature_grid:
                pmf_t = self._apply_temperature_to_pmf(pmf_raw, t)
                cdf_t = np.cumsum(pmf_t, axis=1)
                crps = self._crps_margin(cdf_t, margin_outcomes)
                if np.isfinite(crps) and crps < best_crps:
                    best_crps = crps
                    best_t = t
            self.margin_temperature = float(best_t)
            # Optionally fit per-bin temperatures by |spread|
            if self.config.fit_margin_temperature_per_bin and market_home is not None and market_away is not None:
                spreads_all = np.asarray(market_home) - np.asarray(market_away)
                abs_spreads = np.abs(spreads_all)
                bins = np.asarray(self.ats_bin_edges)
                num_bins = len(bins) - 1
                temps: List[float] = [self.margin_temperature] * num_bins
                for b in range(num_bins):
                    lo, hi = bins[b], bins[b + 1]
                    idx = np.where((abs_spreads >= lo) & (abs_spreads < hi))[0]
                    if idx.size < 20:  # need enough samples
                        continue
                    best_tb = self.margin_temperature
                    best_crps_b = float("inf")
                    pmf_b = pmf_raw[idx]
                    outcomes_b = margin_outcomes[idx]
                    for t in self.config.margin_temperature_grid:
                        pmf_t_b = self._apply_temperature_to_pmf(pmf_b, t)
                        cdf_t_b = np.cumsum(pmf_t_b, axis=1)
                        crps_b = self._crps_margin(cdf_t_b, outcomes_b)
                        if np.isfinite(crps_b) and crps_b < best_crps_b:
                            best_crps_b = crps_b
                            best_tb = t
                    temps[b] = float(best_tb)
                self.margin_temperature_bins = temps
        # Fit isotonic using temperature-adjusted PMF -> CDF
        if self.isotonic:
            indicators = np.zeros_like(margin_cdf)
            for i, outcome in enumerate(margin_outcomes):
                idx = int(np.clip(outcome - self.margin_range[0], 0, margin_cdf.shape[1] - 1))
                indicators[i, idx:] = 1.0
            pmf_for_iso = self._apply_temperature_to_pmf(pmf_raw, self.margin_temperature)
            cdf_for_iso = np.cumsum(pmf_for_iso, axis=1)
            self.isotonic.fit(cdf_for_iso, indicators)
        if self.temperature:
            home_pmfs = joint_probs.sum(axis=2)
            away_pmfs = joint_probs.sum(axis=1)
            stacked_pmfs = np.stack([home_pmfs, away_pmfs], axis=1)
            stacked_outcomes = np.stack([score_outcomes_home, score_outcomes_away], axis=1)
            self.temperature.fit(stacked_pmfs, stacked_outcomes)

        if market_home is not None and market_away is not None:
            model_exp_home = []
            model_exp_away = []
            for joint in joint_probs:
                exp_home, exp_away = expected_scores(joint)
                model_exp_home.append(exp_home)
                model_exp_away.append(exp_away)
            model_exp_home = np.array(model_exp_home)
            model_exp_away = np.array(model_exp_away)

            self.blender.fit(
                model_home=model_exp_home,
                model_away=model_exp_away,
                market_home=np.asarray(market_home),
                market_away=np.asarray(market_away),
                actual_home=np.asarray(score_outcomes_home),
                actual_away=np.asarray(score_outcomes_away),
            )
        else:
            self.blender.weight = self.config.market_weight
        self.sigma_home = sigma_home
        self.sigma_away = sigma_away

        # Fit win-probability isotonic calibrator (from joint)
        if self.win_iso is not None:
            from models.distribution import compute_win_probabilities

            pred_win = np.array([compute_win_probabilities(j)[0] for j in joint_probs], dtype=float)
            actual_win = (margin_outcomes > 0).astype(float)
            # NBA ties are effectively zero; if present, map to 0.5 for stability
            actual_win[margin_outcomes == 0] = 0.5
            self.win_iso.fit(pred_win, actual_win)

        # Fit ATS cover-prob isotonic per |spread| bucket using calibrated margin CDF
        if (self.ats_isos_fd is not None or self.ats_isos is not None) and market_home is not None and market_away is not None:
            # Recover margin PMF from CDF
            pmf = np.diff(
                np.concatenate([np.zeros((len(margin_cdf), 1)), margin_cdf], axis=1),
                axis=1,
            )
            m_low, m_high = self.margin_range
            m_vals = np.arange(m_low, m_high + 1)
            spreads = np.asarray(market_home) - np.asarray(market_away)
            abs_spreads = np.abs(spreads)
            fav_mask = spreads < 0.0  # True if home favorite
            num_bins = len(self.ats_bin_edges) - 1
            for b in range(num_bins):
                lo = self.ats_bin_edges[b]
                hi = self.ats_bin_edges[b + 1]
                bin_sel = (abs_spreads >= lo) & (abs_spreads < hi)
                for fval in (0, 1):
                    fd_sel = fav_mask == bool(fval)
                    idx = np.where(bin_sel & fd_sel)[0]
                    if idx.size == 0:
                        continue
                    s_sel = spreads[idx]
                    pred_cover = np.empty(idx.size, dtype=float)
                    for k, ii in enumerate(idx):
                        s = s_sel[k]
                        pred_cover[k] = pmf[ii, m_vals > -s].sum()
                    y = np.full(idx.size, 0.0, dtype=float)
                    margins = margin_outcomes[idx]
                    y[margins > -s_sel] = 1.0
                    y[np.isclose(margins, -s_sel)] = 0.5
                    try:
                        if self.ats_isos_fd is not None:
                            iso = self.ats_isos_fd[fval][b]
                            iso.fit(pred_cover, y)
                            # Learn per-bin blend weight
                            if self.config.ats_blend_learn:
                                p_iso = iso.transform(pred_cover)
                                d = p_iso - pred_cover
                                denom = float(np.dot(d, d) + self.config.ats_blend_reg)
                                if denom > 0:
                                    w = float(np.dot(d, (y - pred_cover)) / denom)
                                    # Shrink and cap weight based on bin sample size
                                    n = float(idx.size)
                                    shrink = (n / (n + self.config.ats_blend_shrink)) * min(1.0, n / max(1.0, float(self.config.ats_min_bin_samples)))
                                    w = float(np.clip(w, 0.0, self.config.ats_blend_cap)) * float(shrink)
                                else:
                                    w = 0.5
                                self.ats_blend_fd[fval][b] = w
                        elif self.ats_isos is not None:
                            iso = self.ats_isos[b]
                            iso.fit(pred_cover, y)
                            if self.config.ats_blend_learn and self.ats_blend is not None:
                                p_iso = iso.transform(pred_cover)
                                d = p_iso - pred_cover
                                denom = float(np.dot(d, d) + self.config.ats_blend_reg)
                                if denom > 0:
                                    w = float(np.dot(d, (y - pred_cover)) / denom)
                                    n = float(idx.size)
                                    shrink = (n / (n + self.config.ats_blend_shrink)) * min(1.0, n / max(1.0, float(self.config.ats_min_bin_samples)))
                                    w = float(np.clip(w, 0.0, self.config.ats_blend_cap)) * float(shrink)
                                else:
                                    w = 0.5
                                self.ats_blend[b] = w
                    except Exception:
                        # fall back to identity
                        iso = IsotonicRegression(out_of_bounds="clip")
                        iso.fit(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
                        if self.ats_isos_fd is not None:
                            self.ats_isos_fd[fval][b] = iso
                            if self.config.ats_blend_learn:
                                self.ats_blend_fd[fval][b] = 0.0
                        elif self.ats_isos is not None:
                            self.ats_isos[b] = iso
                            if self.config.ats_blend_learn and self.ats_blend is not None:
                                self.ats_blend[b] = 0.0

        return CalibrationResult(
            isotonic=self.isotonic,
            temperature=self.temperature,
            blender=self.blender,
        )

    def calibrate_margin_cdf(self, margin_cdf: np.ndarray, spreads: Optional[np.ndarray] = None) -> np.ndarray:
        # Convert CDF back to PMF, apply temperature (global or per-bin), then re-CDF and apply isotonic
        pmf = np.diff(
            np.concatenate([np.zeros((len(margin_cdf), 1)), margin_cdf], axis=1),
            axis=1,
        )
        if spreads is not None and self.margin_temperature_bins is not None:
            # Row-wise temperature by |spread| bins
            spreads = np.asarray(spreads)
            abs_spreads = np.abs(spreads)
            bins = np.asarray(self.ats_bin_edges)
            num_bins = len(bins) - 1
            pmf_t = np.zeros_like(pmf)
            for b in range(num_bins):
                lo, hi = bins[b], bins[b + 1]
                mask = (abs_spreads >= lo) & (abs_spreads < hi)
                if not np.any(mask):
                    continue
                t_b = self.margin_temperature_bins[b]
                pmf_t[mask] = self._apply_temperature_to_pmf(pmf[mask], t_b)
            # Any rows not covered fall back to global
            remaining = ~np.isfinite(pmf_t).any(axis=1)
            if np.any(remaining):
                pmf_t[remaining] = self._apply_temperature_to_pmf(pmf[remaining], self.margin_temperature)
        else:
            pmf_t = self._apply_temperature_to_pmf(pmf, self.margin_temperature)
        cdf_t = np.cumsum(pmf_t, axis=1)
        
        # Check for NaN/inf values before isotonic transformation
        if np.any(~np.isfinite(cdf_t)):
            nan_count = np.sum(~np.isfinite(cdf_t))
            import warnings
            warnings.warn(
                f"Found {nan_count} non-finite values in margin CDFs. "
                f"Replacing with sensible defaults to prevent calibration crash."
            )
            # Replace NaN/inf with uniform CDF (uninformative prior)
            bad_rows = ~np.isfinite(cdf_t).all(axis=1)
            if np.any(bad_rows):
                # Create uniform CDF for bad rows
                uniform_cdf = np.linspace(0, 1, cdf_t.shape[1])
                cdf_t[bad_rows] = uniform_cdf
        
        if self.isotonic is None:
            return cdf_t
        return self.isotonic.transform(cdf_t)

    def calibrate_joint_probs(self, joint_probs: np.ndarray) -> np.ndarray:
        if self.temperature is None:
            return joint_probs
        home_pmfs = joint_probs.sum(axis=2)
        away_pmfs = joint_probs.sum(axis=1)
        stacked_pmfs = np.stack([home_pmfs, away_pmfs], axis=1)
        transformed = self.temperature.transform(stacked_pmfs)
        adjusted_joint = np.zeros_like(joint_probs)
        for idx in range(joint_probs.shape[0]):
            target_home = transformed[idx, 0, :]
            target_away = transformed[idx, 1, :]
            joint = joint_probs[idx].copy()
            for _ in range(5):
                row_sum = joint.sum(axis=1, keepdims=True)
                row_sum[row_sum == 0] = 1e-8
                joint *= (target_home[:, None] / row_sum)
                col_sum = joint.sum(axis=0, keepdims=True)
                col_sum[col_sum == 0] = 1e-8
                joint *= (target_away[None, :] / col_sum)
            total_sum = joint.sum()
            if not np.isfinite(total_sum) or total_sum <= 0:
                joint = np.outer(target_home, target_away)
                total_sum = joint.sum()
            adjusted_joint[idx] = joint / total_sum
        return adjusted_joint

    def blend_expectations(
        self,
        model_home: np.ndarray,
        model_away: np.ndarray,
        market_home: np.ndarray,
        market_away: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.blender.blend(model_home, model_away, market_home, market_away)

    # -------- Scalar post-calibration helpers --------
    def calibrate_win_prob(self, win_probs: np.ndarray) -> np.ndarray:
        win_iso = getattr(self, "win_iso", None)
        if win_iso is None:
            return win_probs
        return np.clip(win_iso.transform(win_probs), 0.0, 1.0)

    def calibrate_cover_prob(self, cover_probs: np.ndarray, spreads: np.ndarray) -> np.ndarray:
        ats_isos_fd = getattr(self, "ats_isos_fd", None)
        ats_isos = getattr(self, "ats_isos", None)
        if ats_isos_fd is None and ats_isos is None:
            return cover_probs
        spreads = np.asarray(spreads)
        abs_spreads = np.abs(spreads)
        bins = np.asarray(getattr(self, "ats_bin_edges", (0.0, 2.0, 4.0, 6.0, 10.0, 40.0)))
        # digitize returns indices 1..len(bins)-1; convert to 0-based
        bin_idx = np.clip(np.digitize(abs_spreads, bins) - 1, 0, len(bins) - 2)
        calibrated = np.empty_like(cover_probs, dtype=float)
        if ats_isos_fd is not None:
            fav_mask = spreads < 0.0
            num_bins = len(bins) - 1
            for b in range(num_bins):
                for fval in (0, 1):
                    mask = (bin_idx == b) & (fav_mask == bool(fval))
                    if not np.any(mask):
                        continue
                    iso = ats_isos_fd[fval][b]
                    p_raw = cover_probs[mask]
                    p_iso = iso.transform(p_raw)
                    w = 0.5
                    if hasattr(self, "ats_blend_fd") and self.ats_blend_fd is not None:
                        w = float(self.ats_blend_fd[fval][b])
                    p_blend = p_raw + w * (p_iso - p_raw)
                    calibrated[mask] = np.clip(p_blend, self.config.ats_clip_low, self.config.ats_clip_high)
        elif ats_isos is not None:
            for b in range(len(ats_isos)):
                mask = bin_idx == b
                if not np.any(mask):
                    continue
                iso = ats_isos[b]
                p_raw = cover_probs[mask]
                p_iso = iso.transform(p_raw)
                w = 0.5
                if hasattr(self, "ats_blend") and self.ats_blend is not None:
                    w = float(self.ats_blend[b])
                p_blend = p_raw + w * (p_iso - p_raw)
                calibrated[mask] = np.clip(p_blend, self.config.ats_clip_low, self.config.ats_clip_high)
        # Any bins with no iso (shouldn't happen) fall back
        calibrated[~np.isfinite(calibrated)] = cover_probs[~np.isfinite(calibrated)]
        return calibrated

    # -------- Internal helpers --------
    def _apply_temperature_to_pmf(self, pmf: np.ndarray, temperature: float) -> np.ndarray:
        t = max(float(temperature), 1e-6)
        scaled = np.power(np.clip(pmf, 1e-18, None), 1.0 / t)
        row_sum = scaled.sum(axis=1, keepdims=True)
        row_sum = np.clip(row_sum, 1e-18, None)
        return scaled / row_sum

    def _crps_margin(self, cdf: np.ndarray, outcomes: np.ndarray) -> float:
        # Discrete CRPS over integer margin bins
        n, width = cdf.shape
        m_low = self.margin_range[0]
        # Map outcome to nearest bin index
        idx = np.clip((outcomes - m_low).astype(int), 0, width - 1)
        # Build target step CDF in vectorized fashion
        arange = np.arange(width)[None, :]
        step = (arange >= idx[:, None]).astype(float)
        diff = cdf - step
        return float(np.mean(np.sum(diff * diff, axis=1)))


__all__ = [
    "Calibrator",
    "CalibrationConfig",
    "CalibrationResult",
]

