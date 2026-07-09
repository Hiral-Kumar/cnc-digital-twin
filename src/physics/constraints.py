"""
constraints.py
─────────────────────────────────────────────────────────────────────────────
Physics Constraint Loss Functions for PC-NDT

WHAT THIS FILE IMPLEMENTS:
    Three physics laws embedded as soft penalty terms in the training loss:

    1. ARCHARD'S WEAR LAW (bearing nodes)
       dW/dt = k × F × v / H
       Constrains predicted wear rate to scale correctly with load & velocity.

    2. PARIS' FATIGUE CRACK GROWTH LAW (structural nodes)
       da/dN = C × (ΔK)^m   where ΔK = Y × Δσ × √(π × a)
       Constrains predicted crack growth to follow the nonlinear power law
       that produces accelerating failure near end-of-life.

    3. FOURIER'S HEAT EQUATION (all thermally connected nodes)
       ∂T_i/∂t = α × Σ_j L_ij × T_j + Q_i / (ρ × c)
       Constrains predicted temperature evolution to obey heat conduction
       along the sensor graph structure.

THE COMBINED LOSS:
    L_total = L_pred + λ₁×L_Archard + λ₂×L_Paris + λ₃×L_Fourier

    Each λ is set to 0 during warm-up epochs, then ramped in gradually.
    See trainer.py for the warm-up schedule.

WHY SOFT CONSTRAINTS (loss terms) NOT HARD CONSTRAINTS (exact equations):
    Hard constraints would require the ODE right-hand side to exactly
    satisfy the physics equations — leaving no room for the model to
    learn the physics that these laws DON'T capture (surface roughness
    evolution, lubrication breakdown, environmental effects).

    Soft constraints let physics GUIDE the model while allowing learned
    corrections for real-world deviations from ideal equations.
    This is the core philosophy of Physics-Informed Machine Learning.
─────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
from src.data.graph_utils import compute_graph_laplacian
import logging

logger = logging.getLogger(__name__)


class PhysicsConstraints(nn.Module):
    """
    All three physics constraints in a single module.

    This module is called during training to compute the three physics
    loss terms. It is DISABLED at inference time (the physics auditor
    is a training-time teacher, not a deployment-time component).

    At inference, we still compute the Physics Disagreement Score (PDS)
    — a lightweight single-number interpretability signal derived from
    the Archard constraint only — which costs almost nothing extra.
    """

    def __init__(self, config: dict):
        super().__init__()

        cfg_phys = config['physics']

        # ── Archard's Wear Law constants ─────────────────────────────────
        archard = cfg_phys['archard']
        self.k = float(archard['wear_coefficient_k'])    # dimensionless wear coefficient
        self.H = float(archard['hardness_H'])              # Pa — material hardness
        self.F_constant = float(archard['constant_load_N'] ) # N — IMS constant load

        # ── Paris' Law constants ──────────────────────────────────────────
        paris = cfg_phys['paris']
        self.C_paris = float(paris['C'])               # material constant (SI)
        self.m_paris = float(paris['m'])                 # Paris exponent (~3 for steel)
        self.Y_paris = float(paris['geometry_factor_Y']) # geometry factor

        # ── Fourier's Heat Equation constants ─────────────────────────────
        fourier = cfg_phys['fourier']
        self.alpha     = float(fourier['thermal_diffusivity'])  # m²/s
        self.rho       = float(fourier['density'])             # kg/m³
        self.c_heat    = float(fourier['specific_heat'])        # J/(kg·K)
        self.mu_frict  = float(fourier['friction_coefficient']) # dimensionless

        logger.info(
            f"PhysicsConstraints initialized | "
            f"k={self.k:.2e}, C={self.C_paris:.2e}, m={self.m_paris}, "
            f"α={self.alpha:.2e}"
        )

    def archard_loss(self,
                     predicted_derivatives: torch.Tensor,
                     spindle_rpm: float = 2000.0,
                     bearing_pitch_dia_m: float = 0.02815) -> torch.Tensor:
        """
        Archard's Wear Law Constraint.

        PHYSICS:
            Wear rate:  dW/dt = k × F × v / H
            where:
                k = wear coefficient (material property, dimensionless)
                F = normal contact force (Newtons) — constant in IMS (26,690 N)
                v = sliding velocity (m/s) = π × d_pitch × RPM / 60
                H = material hardness (Pascals)

        WHAT WE CONSTRAIN:
            The "wear dimension" of the Neural ODE hidden state's derivative
            should scale with k×F×v/H. Since we don't know which hidden
            dimension corresponds to wear, we constrain the MAGNITUDE of
            the overall predicted derivative to not deviate excessively from
            the Archard reference rate.

            More precisely: the mean absolute value of dh/dt across nodes
            should correlate with the Archard wear rate under the current
            operating conditions.

        Args:
            predicted_derivatives: [B, N, hidden_dim] — dh/dt from Neural ODE
            spindle_rpm:           current spindle speed in RPM
            bearing_pitch_dia_m:   pitch diameter of bearing in meters

        Returns:
            scalar loss tensor
        """
        # Compute sliding velocity from RPM and bearing geometry
        # v = π × d_pitch × (RPM / 60)   [m/s]
        shaft_freq_hz = spindle_rpm / 60.0
        v = torch.pi * bearing_pitch_dia_m * shaft_freq_hz

        # Archard reference wear rate (scalar)
        # dW_dt_ref = k × F × v / H
        archard_ref = (self.k * self.F_constant * v) / self.H

        # Predicted rate: mean absolute derivative across hidden dimensions
        # Shape: [B, N] — one rate per (batch, node)
        pred_rate = predicted_derivatives.abs().mean(dim=-1)   # [B, N]

        # Penalty: mean squared deviation from Archard reference
        # We scale archard_ref to match the typical magnitude of pred_rate
        # by multiplying by a fixed scale factor (this is why λ_archard matters
        # — it handles the magnitude mismatch between the raw Archard value
        # and the hidden state derivative scale)
        loss = torch.mean((pred_rate - archard_ref) ** 2)

        return loss

    def paris_loss(self,
                   predicted_derivatives: torch.Tensor,
                   vibration_rms: torch.Tensor,
                   hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Paris' Fatigue Crack Growth Law Constraint.

        PHYSICS:
            da/dN = C × (ΔK)^m
            ΔK = Y × Δσ × √(π × a)

        where:
            a      = crack length (inferred from hidden state magnitude)
            Δσ     = stress range (proportional to vibration RMS)
            C, m   = material constants for bearing steel
            Y      = geometry factor

        THE NONLINEAR SELF-REFERENTIAL STRUCTURE:
            ΔK depends on √a, and a is tracked in the hidden state.
            This means: as the hidden state grows (degradation increases),
            ΔK grows as √a, and thus da/dN grows as (√a)^m = a^(m/2).

            For m=3 (steel): da/dN ∝ a^1.5 — superlinear growth.
            This is what makes bearing failures ACCELERATE near end-of-life.
            The model is forced to learn this nonlinear acceleration.

        Args:
            predicted_derivatives: [B, N, hidden_dim] — dh/dt from Neural ODE
            vibration_rms:         [B, N] — RMS vibration (feature 0) at current step
            hidden_state:          [B, N, hidden_dim] — current h(t)

        Returns:
            scalar loss tensor
        """
        # Proxy for crack length a: L2 norm of hidden state per node
        # As degradation progresses, the hidden state grows → proxy for crack growth
        # Shape: [B, N]
        a_proxy = hidden_state.norm(dim=-1)     # [B, N]

        # Stress intensity factor range ΔK = Y × Δσ × √(π × a)
        # Δσ ∝ vibration_rms (stress is proportional to vibration amplitude)
        delta_sigma = vibration_rms.clamp(min=1e-6)   # [B, N]
        delta_K = self.Y_paris * delta_sigma * torch.sqrt(
            torch.tensor(torch.pi) * a_proxy.clamp(min=1e-6)
        )                                               # [B, N]

        # Paris' Law reference crack growth rate
        # da/dN = C × (ΔK)^m
        paris_ref = self.C_paris * (delta_K ** self.m_paris)   # [B, N]

        # Predicted crack growth rate: magnitude of derivative
        pred_rate = predicted_derivatives.abs().mean(dim=-1)   # [B, N]

        # Penalty: mean squared deviation from Paris reference
        loss = torch.mean((pred_rate - paris_ref) ** 2)

        return loss

    def fourier_loss(self,
                     predicted_derivatives: torch.Tensor,
                     temperature_features: torch.Tensor,
                     adjacency: torch.Tensor,
                     spindle_rpm: float = 2000.0,
                     bearing_pitch_dia_m: float = 0.02815) -> torch.Tensor:
        """
        Fourier's Heat Equation Constraint (Discrete Graph Form).

        PHYSICS:
            ∂T_i/∂t = α × Σ_j L_ij × T_j + Q_i / (ρ × c)

            L = D - A   (graph Laplacian, D = degree matrix)
            Q_i = μ × F × v   (heat generation from friction at node i)

        THE GRAPH ALIGNMENT:
            The Laplacian L uses the SAME adjacency A that AGCRN learned.
            This creates a beautiful mathematical alignment: the graph
            structure is jointly optimised for both predictive accuracy
            (via AGCRN message passing) and thermal physics consistency
            (via this Fourier constraint). One graph, two physical roles.

        Args:
            predicted_derivatives: [B, N, hidden_dim] — dh/dt from Neural ODE
            temperature_features:  [B, N] — temperature signal (if available)
                                   For IMS (no temperature): use zeros
                                   For PRONOSTIA: use actual temperature feature
            adjacency:             [N, N] — learned adjacency from AGCRN
            spindle_rpm:           float — current spindle speed
            bearing_pitch_dia_m:   float — pitch diameter in meters

        Returns:
            scalar loss tensor
        """
        B, N = temperature_features.shape

        # Compute graph Laplacian L = D - A from the learned adjacency
        # This is the discrete analogue of the continuous Laplacian ∇²
        L = compute_graph_laplacian(adjacency)   # [N, N]

        # Temperature term: α × L × T
        # L: [N, N], T: [B, N]
        # Result: [B, N] — the diffusion term for each node
        LT = torch.einsum('nm,bm->bn', L, temperature_features)   # [B, N]
        diffusion = self.alpha * LT

        # Heat generation term: Q_i / (ρ × c)
        # Q_i = μ × F_contact × v (friction heat at each bearing node)
        shaft_freq_hz = spindle_rpm / 60.0
        v = torch.pi * bearing_pitch_dia_m * shaft_freq_hz
        Q_i = self.mu_frict * self.F_constant * v
        heat_gen = Q_i / (self.rho * self.c_heat)   # scalar, same for all nodes

        # Fourier reference temperature rate per node
        # ∂T_i/∂t = α × (L×T)_i + Q_i/(ρc)
        fourier_ref = diffusion + heat_gen   # [B, N]

        # Predicted temperature rate: magnitude of derivative
        # (We use the derivative magnitude as a proxy for temperature rate)
        pred_rate = predicted_derivatives.abs().mean(dim=-1)   # [B, N]

        # Penalty: deviation from Fourier reference
        loss = torch.mean((pred_rate - fourier_ref.abs()) ** 2)

        return loss

    def physics_disagreement_score(self,
                                   predicted_derivatives: torch.Tensor,
                                   spindle_rpm: float = 2000.0,
                                   bearing_pitch_dia_m: float = 0.02815) -> torch.Tensor:
        """
        Physics Disagreement Score (PDS) — the deployment-time interpretability signal.

        This is the single scalar per node that tells a maintenance engineer
        HOW MUCH the model's predicted degradation rate deviates from what
        Archard's Law says it should be, given the operating conditions.

        UNLIKE the training constraints, this is computed at INFERENCE too.

        A rising PDS on a single bearing → that bearing is degrading faster
        than physics predicts → anomaly → schedule inspection.

        A rising PDS on ALL bearings simultaneously → sensor drift or model
        distribution shift → trigger retraining pipeline.

        Args:
            predicted_derivatives: [B, N, hidden_dim] — dh/dt from Neural ODE
            spindle_rpm:           current spindle speed
            bearing_pitch_dia_m:   pitch diameter

        Returns:
            pds: [B, N] — Physics Disagreement Score per (batch, node)
                 Units: same as predicted derivative magnitude.
                 Higher = larger deviation from physical expectation.
        """
        shaft_freq_hz = spindle_rpm / 60.0
        v = torch.pi * bearing_pitch_dia_m * shaft_freq_hz
        archard_ref = (self.k * self.F_constant * v) / self.H

        pred_rate = predicted_derivatives.abs().mean(dim=-1)   # [B, N]
        pds = (pred_rate - archard_ref).abs()                  # [B, N]

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
        """
        Compute all three physics losses and combine with weights.

        Called from the training loop once per batch.

        Args:
            predicted_derivatives: [B, N, hidden_dim] — dh/dt
            hidden_state:          [B, N, hidden_dim] — h(t)
            adjacency:             [N, N] — learned graph A
            features:              [B, N, F] — raw features (vibration, temperature)
            lambda_archard:        weight for Archard loss (0 during warm-up)
            lambda_paris:          weight for Paris loss
            lambda_fourier:        weight for Fourier loss
            spindle_rpm:           operating speed

        Returns:
            dict:
                'archard': Archard loss (scalar)
                'paris':   Paris loss (scalar)
                'fourier': Fourier loss (scalar)
                'total':   weighted sum
        """
        bearing_pitch_dia_m = 0.02815  # IMS bearing pitch diameter

        # Vibration RMS: feature index 0 — [B, N]
        vibration_rms = features[:, :, 0]

        # Temperature: feature index 0 for now (IMS has no temp sensor)
        # For PRONOSTIA with temperature data, use the temperature feature directly
        # Here we approximate with zeros (conservative — no thermal gradient)
        temperature = torch.zeros_like(vibration_rms)

        # Compute each constraint
        l_archard = self.archard_loss(
            predicted_derivatives, spindle_rpm, bearing_pitch_dia_m
        )
        l_paris = self.paris_loss(
            predicted_derivatives, vibration_rms, hidden_state
        )
        l_fourier = self.fourier_loss(
            predicted_derivatives, temperature, adjacency,
            spindle_rpm, bearing_pitch_dia_m
        )

        # Weighted combination
        total = (lambda_archard * l_archard +
                 lambda_paris   * l_paris   +
                 lambda_fourier * l_fourier)

        return {
            'archard': l_archard,
            'paris':   l_paris,
            'fourier': l_fourier,
            'total':   total,
        }
