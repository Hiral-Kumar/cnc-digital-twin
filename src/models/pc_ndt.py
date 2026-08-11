"""
pc_ndt.py — Unified PC-NDT Model
Physics-Constrained Neural Digital Twin

Ties AGCRN (spatial encoder) + Neural ODE (temporal propagator)
+ Linear readout into one nn.Module.

Input:  X_seq [B, T, N, F]  — sliding window of sensor features
Output: RUL   [B, N]        — Remaining Useful Life per node
        A     [N, N]        — learned adjacency (for logging/viz)
        pds   [B, N]        — Physics Disagreement Score (interpretability)
        dh_dt [B, N, D]     — ODE derivatives (for physics constraint losses)
        h     [B, N, D]     — final hidden state
"""

import torch
import torch.nn as nn
from .agcrn import AGCRN
from .neural_ode import NeuralODE
import logging

logger = logging.getLogger(__name__)


class PCNDT(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        self.spatial_encoder = AGCRN(config)
        self.temporal_propagator = NeuralODE(config)

        hidden_dim = config['model']['neural_ode']['hidden_dim']
        self.readout = nn.Linear(hidden_dim, 1)  # [B,N,D] -> [B,N,1]

        nn.init.xavier_uniform_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"PC-NDT | Total trainable parameters: {n_params:,}")

    def forward(self, X_seq: torch.Tensor,
                t_span: torch.Tensor = None) -> dict:
        """
        Args:
            X_seq:  [B, T, N, F]
            t_span: 1-D tensor of ODE evaluation times, e.g. tensor([0.0, 1.0])

        Returns dict with keys: rul, adjacency, pds, dh_dt, h_final
        """
        device = X_seq.device

        if t_span is None:
            t_span = torch.tensor([0.0, 1.0], dtype=X_seq.dtype, device=device)

        # ── Stage 1: Spatial encoding ─────────────────────────────────────
        h0, A = self.spatial_encoder(X_seq)      # h0: [B,N,D], A: [N,N]

        # ── Stage 2: Temporal propagation (Neural ODE) ────────────────────
        _, h_final = self.temporal_propagator(h0, t_span)   # h_final: [B,N,D]

        # ── Stage 3: RUL readout ──────────────────────────────────────────
        rul = self.readout(h_final).squeeze(-1)   # [B,N,1] -> [B,N]
        rul = torch.sigmoid(rul)                  # constrain to (0,1)

        # ── Derivatives for physics constraints ───────────────────────────
        dh_dt = self.temporal_propagator.get_derivatives(h_final, t=0.0)

        # ── Physics Disagreement Score (always computed) ──────────────────
        from src.physics.constraints import PhysicsConstraints
        physics = PhysicsConstraints(self.config)
        pds = physics.physics_disagreement_score(dh_dt)

        return {
            'rul':       rul,        # [B, N]
            'adjacency': A,          # [N, N]
            'pds':       pds,        # [B, N]
            'dh_dt':     dh_dt,      # [B, N, D]
            'h_final':   h_final,    # [B, N, D]
        }
