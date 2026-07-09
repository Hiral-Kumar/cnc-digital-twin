"""
metrics.py
─────────────────────────────────────────────────────────────────────────────
Evaluation Metrics for PC-NDT

All four metrics used in the research paper, implemented correctly
and verified against the original sources.

METRIC 1: RMSE — standard prediction accuracy
METRIC 2: PHM 2012 Score — official PRONOSTIA benchmark metric
METRIC 3: Delta RMSE — cross-condition generalisation drop
METRIC 4: Pearson ρ — physics constraint effectiveness
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
from scipy import stats
from typing import Union
import logging

logger = logging.getLogger(__name__)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error.

    WHAT IT MEASURES:
        Average magnitude of prediction error in the same units as RUL.
        Sensitive to large errors (squares them before averaging),
        which is appropriate for safety-critical predictions where
        catastrophic mispredictions are worse than small ones.

    WHEN IS IT "GOOD ENOUGH":
        For IMS normalized RUL [0,1]: published LSTM baselines achieve
        ~0.12–0.18. Your target is to beat your own unconstrained ablation
        (PC-NDT no physics) by a statistically meaningful margin.

    Args:
        y_true: np.ndarray of any shape — true RUL values
        y_pred: np.ndarray of same shape — predicted RUL values

    Returns:
        float — RMSE value (lower is better)
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def phm2012_score(y_true: np.ndarray, y_pred: np.ndarray,
                  early_denom: float = 20.0,
                  late_denom: float = 5.0) -> float:
    """
    Official IEEE PHM 2012 Challenge Scoring Function.

    SOURCE: Nectoux et al. (2012), PRONOSTIA dataset paper.
            This is the NATIVE metric for PRONOSTIA results.
            Do not use the C-MAPSS score for PRONOSTIA — they are
            calibrated differently and produce non-comparable numbers.

    FORMULA:
        %Er_i = 100 × (RUL_true,i − RUL_pred,i) / RUL_true,i

        A_i = exp(−ln(0.5) × (Er_i / 5))   if Er_i ≤ 0  [late — dangerous]
            = exp(+ln(0.5) × (Er_i / 20))  if Er_i > 0  [early — safe]

        Score = mean(A_i)   [higher is better, max = 1.0]

    ASYMMETRY EXPLAINED:
        Er ≤ 0: you overestimated RUL — told engineer bearing is healthier
                than it is. DANGEROUS. Penalized with denominator 5 (steep).
        Er > 0: you underestimated RUL — predicted failure too early.
                Over-conservative, safe. Penalized with denominator 20 (gentle).

        Same |error|, but late prediction scores ~4× worse than early.
        This asymmetry encodes the real cost structure of maintenance:
        a missed failure is far worse than an unnecessary early inspection.

    Args:
        y_true: true RUL values (raw cycle counts OR normalized — must match y_pred scale)
        y_pred: predicted RUL values
        early_denom: denominator for early predictions (Er > 0), default 20
        late_denom:  denominator for late predictions (Er ≤ 0), default 5

    Returns:
        float — PHM 2012 score in [0, 1] (higher is better)
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    # Avoid division by zero for bearings at the very end of life (RUL_true ≈ 0)
    # Clip to a small epsilon to prevent NaN in the percentage computation
    y_true_safe = np.where(np.abs(y_true) < 1e-6, 1e-6, y_true)

    # Percentage error: positive = predicted too early (safe), negative = too late (dangerous)
    Er = 100.0 * (y_true_safe - y_pred) / y_true_safe

    # Apply asymmetric exponential penalty
    # np.where applies vectorized conditional:
    # where Er ≤ 0 → late prediction penalty
    # where Er > 0 → early prediction penalty
    A = np.where(
        Er <= 0,
        np.exp(-np.log(0.5) * (Er / late_denom)),   # late: steeper decay
        np.exp(np.log(0.5) * (Er / early_denom))    # early: gentler decay
    )

    score = float(np.mean(A))
    return score


def delta_rmse(rmse_source: float, rmse_target: float) -> float:
    """
    Cross-Condition RMSE Drop (ΔRMSEdrop).

    WHAT IT MEASURES:
        The percentage increase in RMSE when the same model is evaluated
        on a new operating condition it was never trained on.

        Smaller ΔRMSEdrop = better generalisation = stronger evidence
        that your physics constraints provide condition-invariant knowledge.

    THIS IS YOUR DIRECT RQ3 ANSWER:
        If PC-NDT (full) ΔRMSEdrop = 15%
        and PC-NDT (no physics) ΔRMSEdrop = 40%
        → Physics constraints improve generalisation by 25 percentage points.
        That gap is the quantitative evidence for your research claim.

    Args:
        rmse_source: RMSE on the training condition (e.g., IMS Test 3)
        rmse_target: RMSE on the new condition (e.g., PRONOSTIA Condition 2)

    Returns:
        float — percentage increase in RMSE (lower is better)
    """
    if rmse_source <= 0:
        raise ValueError(f"rmse_source must be positive, got {rmse_source}")

    return float(100.0 * (rmse_target - rmse_source) / rmse_source)


def pearson_rho(predicted_rates: np.ndarray,
                reference_rates: np.ndarray) -> float:
    """
    Pearson Correlation Coefficient between predicted and physics-reference rates.

    WHAT IT MEASURES:
        How well the Neural ODE's predicted degradation rates (dh/dt)
        correlate with what Archard's Law / Paris' Law says the rate
        should be, given the measured operating conditions.

        ρ close to 1.0 → model learned physically correct dynamics
        ρ close to 0.0 → model learned something unrelated to physics
        ρ < 0.0        → model learned dynamics that contradict physics

    THIS IS YOUR DIRECT RQ2 ANSWER:
        A high ρ for PC-NDT (full) vs. low ρ for PC-NDT (no physics)
        proves the physics constraints are teaching the model something
        physically real, not just adding noise to the loss function.

    Args:
        predicted_rates: np.ndarray — wear/crack rates from Neural ODE
        reference_rates: np.ndarray — rates computed from Archard/Paris formulas

    Returns:
        float — Pearson r in [-1, 1] (closer to 1 is better for RQ2)
    """
    predicted_rates = np.asarray(predicted_rates, dtype=np.float64).ravel()
    reference_rates = np.asarray(reference_rates, dtype=np.float64).ravel()

    if len(predicted_rates) < 2:
        logger.warning("Pearson ρ requires at least 2 samples. Returning 0.0")
        return 0.0

    r, p_value = stats.pearsonr(predicted_rates, reference_rates)

    logger.debug(f"  Pearson ρ = {r:.4f} (p = {p_value:.4e})")
    return float(r)


def compute_all_metrics(y_true: np.ndarray,
                        y_pred: np.ndarray,
                        rmse_source: float = None,
                        predicted_rates: np.ndarray = None,
                        reference_rates: np.ndarray = None,
                        config: dict = None) -> dict:
    """
    Convenience function: compute all applicable metrics in one call.

    Args:
        y_true:           true RUL values
        y_pred:           predicted RUL values
        rmse_source:      RMSE on source condition (for ΔRMSEdrop, optional)
        predicted_rates:  Neural ODE degradation rates (for Pearson ρ, optional)
        reference_rates:  physics reference rates (for Pearson ρ, optional)
        config:           config dict (for PHM 2012 score denominators)

    Returns:
        dict of metric_name → value
    """
    results = {}

    # Always compute RMSE
    results['rmse'] = rmse(y_true, y_pred)

    # PHM 2012 Score (always compute — valid for normalized or raw RUL)
    early_d = config['evaluation']['phm2012']['early_denominator'] if config else 20
    late_d  = config['evaluation']['phm2012']['late_denominator']  if config else 5
    results['phm2012_score'] = phm2012_score(y_true, y_pred, early_d, late_d)

    # Delta RMSE (only if source RMSE is provided)
    if rmse_source is not None:
        results['delta_rmse'] = delta_rmse(rmse_source, results['rmse'])

    # Pearson rho (only if rate arrays are provided)
    if predicted_rates is not None and reference_rates is not None:
        results['pearson_rho'] = pearson_rho(predicted_rates, reference_rates)

    return results
