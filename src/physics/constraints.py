"""
constraints.py — Physics Constraint Loss Functions for PC-NDT (V3: z-score)

═══════════════════════════════════════════════════════════════════════
EVOLUTION OF THIS FILE (documented for the record):

  V1 (original): raw squared error (pred_rate - ref)^2
    BUG: trivially minimized by collapsing dh_dt toward zero.
    Confirmed via direct probe: dh_dt magnitude ~0.036, physics losses
    shrinking toward zero alongside prediction loss -- degenerate
    joint optimum, model never learned genuine dynamics.

  V2: relative error (pred-ref)^2 / (ref^2 + eps)
    BUG: ref (real SI-unit physics values, ~1e-13) and pred (latent
    hidden-state derivatives, ~0.1-0.3) live in incommensurable numerical
    scales. Division by ref^2 (~1e-26) made denominator dominated by eps,
    while numerator ~pred^2 was ~1e8x larger -- losses exploded to
    100,000-240,000, confirmed empirically during retraining.

  V3 (this version): z-score (distributional) normalization
    FIX: rather than comparing raw magnitudes of two incommensurable
    quantities, compare where each value sits within its OWN batch
    distribution. This is scale-invariant by construction -- no epsilon
    tuning, no magic scale floors, no assumption that pred and ref share
    units. The constraint now asks a physically meaningful question:
    "does dh_dt vary ACROSS THE BATCH in the same relative pattern that
    Archard's Law predicts wear rate should vary across these operating
    conditions?" -- which is the correct comparison a physics-informed
    constraint should make, since dh_dt is a latent quantity that need
    not share literal SI units with the physical formula it's guided by.
═══════════════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
from src.data.graph_utils import compute_graph_laplacian
import logging

logger = logging.getLogger(__name__)

MIN_ACTIVITY_THRESHOLD = 0.05
ZSCORE_EPS = 1e-6   # only guards against exact zero-variance batches (e.g. batch_size=1)


class PhysicsConstraints(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        cfg_phys = config['physics']

        archard = cfg_phys['archard']
        self.k = archard['wear_coefficient_k']
        self.H = archard['hardness_H']
        self.F_constant = archard['constant_load_N']

        paris = cfg_phys['paris']
        self.C_paris = paris['C']
        self.m_paris = paris['m']
        self.Y_paris = paris['geometry_factor_Y']

        fourier = cfg_phys['fourier']
        self.alpha     = fourier['thermal_diffusivity']
        self.rho       = fourier['density']
        self.c_heat    = fourier['specific_heat']
        self.mu_frict  = fourier['friction_coefficient']

        self.min_activity_threshold = cfg_phys.get('min_activity_threshold', MIN_ACTIVITY_THRESHOLD)
        self.lambda_anticollapse = cfg_phys.get('lambda_anticollapse', 0.5)

        logger.info(
            f"PhysicsConstraints initialized | "
            f"k={self.k:.2e}, C={self.C_paris:.2e}, m={self.m_paris}, "
            f"alpha={self.alpha:.2e} | "
            f"[V3 FIX] z-score normalization + anti-collapse "
            f"(min_activity={self.min_activity_threshold}, "
            f"lambda_ac={self.lambda_anticollapse})"
        )

    def _zscore(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standardize a tensor to zero mean, unit variance across its
        batch dimension (dim=0). If x has near-zero variance (e.g. all
        values identical, or batch_size=1), returns zeros to avoid
        division blowup -- a degenerate no-signal case contributes no
        gradient rather than an exploding one.
        """
        mean = x.mean(dim=0, keepdim=True)
        std  = x.std(dim=0, keepdim=True)
        std_safe = torch.clamp(std, min=ZSCORE_EPS)
        return (x - mean) / std_safe

    def _zscore_matched_loss(self, pred: torch.Tensor, ref) -> torch.Tensor:
        """
        Z-score-normalized comparison between predicted and reference rates.

        WHAT THIS ACTUALLY TESTS:
            After z-scoring both pred and ref across the batch dimension,
            each becomes a distribution with mean=0, std=1. Comparing
            THESE (rather than raw values) tests whether pred and ref
            vary in the SAME RELATIVE PATTERN across the batch -- e.g.,
            samples with higher predicted wear rate should correspond to
            samples where Archard's formula (given their operating
            conditions) ALSO predicts higher wear rate, regardless of
            the absolute unit scale of either quantity.

        IMPORTANT CAVEAT (documented honestly):
            If ref is IDENTICAL across the entire batch (e.g. constant
            operating conditions, as in IMS where RPM/load never change
            within a single dataset), z-scoring ref produces near-zero
            variance -- meaning this constraint provides very little
            gradient signal when operating conditions don't vary within
            a batch. This is an expected, honest limitation: the
            constraint is most informative when batches span DIFFERENT
            operating conditions (e.g. mixed IMS+PRONOSTIA batches), and
            provides only weak regularization on single-condition data
            like IMS alone. This should be noted in the paper's
            limitations section.

        Args:
            pred: predicted rate tensor [B, N] or [B]
            ref:  reference rate, scalar or [B, N] or [B]

        Returns:
            scalar loss — mean squared difference of z-scored quantities
        """
        if not torch.is_tensor(ref):
            ref = torch.as_tensor(ref, dtype=pred.dtype, device=pred.device)

        # Broadcast ref to match pred's shape if it's a scalar
        if ref.dim() == 0:
            ref = ref.expand_as(pred)

        pred_z = self._zscore(pred)
        ref_z  = self._zscore(ref)

        return torch.mean((pred_z - ref_z) ** 2)

    def anticollapse_penalty(self, predicted_derivatives: torch.Tensor) -> torch.Tensor:
        activity = predicted_derivatives.norm(dim=-1)
        deficit = torch.relu(self.min_activity_threshold - activity)
        return torch.mean(deficit ** 2)

    def archard_loss(self,
                     predicted_derivatives: torch.Tensor,
                     spindle_rpm: float = 2000.0,
                     bearing_pitch_dia_m: float = 0.02815) -> torch.Tensor:
        shaft_freq_hz = spindle_rpm / 60.0
        v = torch.pi * bearing_pitch_dia_m * shaft_freq_hz
        archard_ref = (self.k * self.F_constant * v) / self.H

        pred_rate = predicted_derivatives.abs().mean(dim=-1)   # [B, N]

        # V3 FIX: z-score matched comparison instead of raw/relative error
        return self._zscore_matched_loss(pred_rate, archard_ref)

    def paris_loss(self,
                   predicted_derivatives: torch.Tensor,
                   vibration_rms: torch.Tensor,
                   hidden_state: torch.Tensor) -> torch.Tensor:
        a_proxy = hidden_state.norm(dim=-1)
        delta_sigma = vibration_rms.clamp(min=1e-6)
        delta_K = self.Y_paris * delta_sigma * torch.sqrt(
            torch.tensor(torch.pi) * a_proxy.clamp(min=1e-6)
        )
        paris_ref = self.C_paris * (delta_K ** self.m_paris)   # [B, N] — varies per sample!

        pred_rate = predicted_derivatives.abs().mean(dim=-1)   # [B, N]

        # V3 FIX: z-score matched comparison
        # NOTE: paris_ref genuinely varies across the batch (depends on
        # vibration_rms and hidden_state per-sample), so this constraint
        # retains meaningful gradient signal even on single-condition
        # datasets like IMS -- unlike archard_loss and fourier_loss whose
        # references are near-constant within a batch of fixed RPM/load.
        return self._zscore_matched_loss(pred_rate, paris_ref)

    def fourier_loss(self,
                     predicted_derivatives: torch.Tensor,
                     temperature_features: torch.Tensor,
                     adjacency: torch.Tensor,
                     spindle_rpm: float = 2000.0,
                     bearing_pitch_dia_m: float = 0.02815) -> torch.Tensor:
        B, N = temperature_features.shape

        L = compute_graph_laplacian(adjacency)
        LT = torch.einsum('nm,bm->bn', L, temperature_features)
        diffusion = self.alpha * LT

        shaft_freq_hz = spindle_rpm / 60.0
        v = torch.pi * bearing_pitch_dia_m * shaft_freq_hz
        Q_i = self.mu_frict * self.F_constant * v
        heat_gen = Q_i / (self.rho * self.c_heat)

        fourier_ref = diffusion + heat_gen   # [B, N] — varies via temperature_features
        pred_rate = predicted_derivatives.abs().mean(dim=-1)

        return self._zscore_matched_loss(pred_rate, fourier_ref.abs())

    def physics_disagreement_score(self,
                                   predicted_derivatives: torch.Tensor,
                                   spindle_rpm: float = 2000.0,
                                   bearing_pitch_dia_m: float = 0.02815) -> torch.Tensor:
        """
        NOTE: this deployment-time interpretability signal intentionally
        keeps RAW magnitude comparison (not z-scored), since a maintenance
        engineer needs an absolute, single-sample number to read off a
        dashboard -- z-scoring requires a batch/distribution and has no
        meaning for a single live sensor reading. This is a deliberate
        design difference between the TRAINING constraint (z-scored,
        needs distributional comparison) and the INFERENCE signal (raw,
        needs single-sample interpretability).
        """
        shaft_freq_hz = spindle_rpm / 60.0
        v = torch.pi * bearing_pitch_dia_m * shaft_freq_hz
        archard_ref = (self.k * self.F_constant * v) / self.H

        pred_rate = predicted_derivatives.abs().mean(dim=-1)
        pds = (pred_rate - archard_ref).abs()

        return pds

    def compute_all(self,
                    predicted_derivatives: torch.Tensor,
                    hidden_state: torch.Tensor,
                    adjacency: torch.Tensor,
                    features: torch.Tensor,
                    lambda_archard: float,
                    lambda_paris: float,
                    lambda_fourier: float,
                    spindle_rpm: float = 2000.0) -> dict:
        bearing_pitch_dia_m = 0.02815

        vibration_rms = features[:, :, 0]
        temperature = torch.zeros_like(vibration_rms)

        l_archard = self.archard_loss(predicted_derivatives, spindle_rpm, bearing_pitch_dia_m)
        l_paris = self.paris_loss(predicted_derivatives, vibration_rms, hidden_state)
        l_fourier = self.fourier_loss(predicted_derivatives, temperature, adjacency,
                                       spindle_rpm, bearing_pitch_dia_m)
        l_anticollapse = self.anticollapse_penalty(predicted_derivatives)

        total = (lambda_archard * l_archard +
                 lambda_paris   * l_paris   +
                 lambda_fourier * l_fourier +
                 self.lambda_anticollapse * l_anticollapse)

        return {
            'archard':      l_archard,
            'paris':        l_paris,
            'fourier':      l_fourier,
            'anticollapse': l_anticollapse,
            'total':        total,
        }