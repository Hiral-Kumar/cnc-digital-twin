"""
pc_ndt.py — Unified PC-NDT Model (V2: elapsed-time-scaled ODE horizon)

═══════════════════════════════════════════════════════════════════════
WHY THIS CHANGED:
    Confirmed via direct evidence (rul_curves.png from the elapsed-time
    fix): Bearings 2-4 now track true RUL closely from file 0 to ~70%
    of the lifecycle, but PLATEAU near RUL≈0.33 rather than continuing
    to 0. The model tracks well but hits a ceiling on how far it can
    predict degradation.

    Root cause hypothesis: every window integrates the Neural ODE over
    the SAME fixed t_span=[0,1], regardless of how much real operational
    time the window represents. Since dh_dt magnitude is roughly
    constant (confirmed: activity stays 0.13-0.43 throughout training),
    a FIXED integration horizon gives h_final a FIXED maximum possible
    drift from h0 -- capping how far into "failed" territory the hidden
    state can ever represent, regardless of how late in the bearing's
    life the window actually is.

THE FIX:
    Scale t_span's endpoint by the window's elapsed-time feature (the
    same 6th feature added in ims_loader.py). Early-life windows
    (elapsed_time near 0) integrate over a short horizon. Late-life
    windows (elapsed_time near 1) integrate over a LONGER horizon --
    directly encoding "more real time has passed, so more drift should
    have accumulated in the degradation state."

    t_span_end = 1.0 + elapsed_time_scale * elapsed_time
    (elapsed_time_scale is a tunable multiplier, default 3.0, meaning
    late-life windows integrate up to 4x longer than early-life ones)
═══════════════════════════════════════════════════════════════════════
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
        self.readout = nn.Linear(hidden_dim, 1)

        nn.init.xavier_uniform_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

        # ── NEW: elapsed-time-scaled integration horizon ──────────────
        cfg_ode = config['model']['neural_ode']
        self.elapsed_time_scale = cfg_ode.get('elapsed_time_scale', 3.0)
        # Index of the elapsed-time feature within the feature vector
        # (it's always the LAST feature — see ims_loader.py, which
        # concatenates it after the 5 statistical features)
        self.elapsed_time_feature_idx = -1

        logger.info(
            f"  PC-NDT: elapsed-time-scaled ODE horizon enabled "
            f"(scale={self.elapsed_time_scale})"
        )

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"PC-NDT | Total trainable parameters: {n_params:,}")

    def _compute_adaptive_t_span(self, X_seq: torch.Tensor) -> torch.Tensor:
        """
        Compute a per-batch integration horizon based on the AVERAGE
        elapsed-time value across the batch's most recent timestep.

        WHY AVERAGE ACROSS BATCH (not per-sample):
            torchdiffeq's odeint integrates a single shared t_span for
            the entire batch in one call -- it does not support a
            different t_span per batch element natively. Using the
            batch's mean elapsed-time is a reasonable approximation:
            since DataLoader shuffles training data, batches naturally
            mix various elapsed-time values, and this still provides a
            meaningfully different signal batch-to-batch compared to a
            fixed constant. At evaluation (shuffle=False), consecutive
            batches will have very similar elapsed-time, which is
            actually appropriate -- consecutive windows in inference
            ARE at similar points in the machine's life.

        Args:
            X_seq: [B, T, N, F] — input window, F's last feature is elapsed-time

        Returns:
            t_span: 1D tensor [0.0, t_end] where t_end scales with
                    elapsed time
        """
        # Extract elapsed-time feature from the LAST timestep of the window
        # (most recent, most relevant to "how far along are we right now")
        elapsed_time_values = X_seq[:, -1, :, self.elapsed_time_feature_idx]  # [B, N]
        mean_elapsed = elapsed_time_values.mean().item()
        mean_elapsed = max(0.0, min(1.0, mean_elapsed))  # safety clamp to [0,1]

        t_end = 1.0 + self.elapsed_time_scale * mean_elapsed
        return torch.tensor([0.0, t_end], dtype=X_seq.dtype, device=X_seq.device)

    def forward(self, X_seq: torch.Tensor,
                t_span: torch.Tensor = None) -> dict:
        device = X_seq.device

        if t_span is None:
            # NEW: adaptive horizon instead of fixed [0, 1]
            t_span = self._compute_adaptive_t_span(X_seq)

        h0, A = self.spatial_encoder(X_seq)
        _, h_final = self.temporal_propagator(h0, t_span)

        rul = self.readout(h_final).squeeze(-1)
        rul = torch.sigmoid(rul)

        dh_dt = self.temporal_propagator.get_derivatives(h_final, t=0.0)

        from src.physics.constraints import PhysicsConstraints
        physics = PhysicsConstraints(self.config)
        pds = physics.physics_disagreement_score(dh_dt)

        return {
            'rul':       rul,
            'adjacency': A,
            'pds':       pds,
            'dh_dt':     dh_dt,
            'h_final':   h_final,
        }