"""
neural_ode.py
─────────────────────────────────────────────────────────────────────────────
Neural ODE Temporal Propagator for PC-NDT

WHAT THIS FILE IMPLEMENTS:
    Chen et al., NeurIPS 2018 (Best Paper Award)
    "Neural Ordinary Differential Equations"

    A neural network f_θ parameterises the DERIVATIVE of the hidden
    degradation state h(t):

        dh(t)/dt = f_θ(h(t), t)

    Starting from the AGCRN's output H_T as the initial condition h(t₀),
    we integrate f_θ forward to produce the future degradation trajectory.

    The integration is done by a numerical ODE solver (dopri5, adaptive step).
    Gradients are computed via the ADJOINT SENSITIVITY METHOD, which requires
    O(1) memory regardless of how many solver steps were taken.

WHAT f_θ LOOKS LIKE:
    A 3-layer MLP with tanh activations.

    WHY tanh (not ReLU):
        Mechanical degradation is a smooth, continuous process.
        tanh is infinitely differentiable — its output varies smoothly
        as h(t) changes, which is necessary for the ODE solver's adaptive
        step size mechanism (it estimates local error using derivatives
        of the RHS). ReLU's kink at 0 can cause step-size instability.

HOW THE PHYSICS CONSTRAINTS CONNECT:
    The physics constraint losses (Archard, Paris, Fourier) compute the
    expected rate of change based on physics laws. They compare this to
    the Neural ODE's predicted dh/dt, and penalize divergence.

    Because the Neural ODE directly parameterises dh/dt = f_θ(h, t),
    the physics constraints operate in exactly the right mathematical space:
    they constrain the derivative, which is what differential equations
    (Archard, Paris, Fourier) specify. This alignment is why Neural ODE
    is the correct architecture for physics-constrained degradation modeling.
─────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class ODEFunction(nn.Module):
    """
    The neural network f_θ that parameterises dh/dt.

    Input:  [h(t), t] — current hidden state + time
    Output: dh/dt     — predicted rate of change

    The output has the SAME shape as the input hidden state h(t).
    This is a constraint of the ODE formulation: the derivative must
    live in the same space as the state itself.

    WHY INCLUDE TIME t AS AN INPUT:
        The rate of mechanical degradation is not stationary — it changes
        as the bearing ages. Early life: slow degradation (Paris region I).
        Near failure: rapidly accelerating (Paris region III).
        Providing time t as an input allows f_θ to learn time-dependent
        dynamics without being restricted to a fixed degradation rate.
    """

    def __init__(self, hidden_dim: int, mlp_hidden: int,
                 n_layers: int, n_nodes: int):
        """
        Args:
            hidden_dim: dimension of h(t) per node
            mlp_hidden: width of hidden MLP layers
            n_layers:   depth of MLP (3 recommended)
            n_nodes:    number of sensor nodes (4 for IMS)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_nodes    = n_nodes

        # Input to MLP: flattened h(t) + scalar time
        # h(t) shape: [B, N, hidden_dim] → flattened per node: [B, N, hidden_dim]
        # We process each node independently with shared weights
        # (the AGCRN already handled spatial mixing — ODE handles temporal evolution)
        input_dim = hidden_dim + 1  # +1 for time

        layers = []
        current_dim = input_dim
        for i in range(n_layers - 1):
            layers.append(nn.Linear(current_dim, mlp_hidden))
            layers.append(nn.Tanh())
            current_dim = mlp_hidden

        # Final layer: output dimension matches hidden state
        layers.append(nn.Linear(current_dim, hidden_dim))

        self.net = nn.Sequential(*layers)

        # Initialize final layer weights near zero
        # This ensures the ODE starts with nearly-zero derivatives (stability)
        # A large initial dh/dt can cause the solver to take very small steps,
        # making training very slow
        nn.init.zeros_(self.net[-1].bias)
        nn.init.uniform_(self.net[-1].weight, -0.01, 0.01)

        logger.debug(
            f"ODEFunction: input_dim={input_dim}, "
            f"hidden={mlp_hidden}, layers={n_layers}, "
            f"output_dim={hidden_dim}"
        )

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Compute dh/dt = f_θ(h, t).

        NOTE ON ARGUMENT ORDER:
            torchdiffeq calls this as f(t, h) — time FIRST, then state.
            This is the standard ODE convention: dy/dt = f(t, y).
            Do not change the argument order.

        Args:
            t: scalar tensor — current time in the integration
            h: [B, N, hidden_dim] — current hidden state

        Returns:
            dh_dt: [B, N, hidden_dim] — time derivative of hidden state
        """
        B, N, D = h.shape

        # Broadcast scalar time t to match hidden state shape
        # t is a scalar; we need [B, N, 1] to concatenate with h: [B, N, D]
        t_broadcast = t.expand(B, N, 1)

        # Concatenate hidden state and time along feature dimension
        h_t = torch.cat([h, t_broadcast], dim=-1)  # [B, N, D+1]

        # Apply MLP per node (same weights for all nodes — spatial context
        # was already encoded by AGCRN; the ODE just needs to evolve it)
        dh_dt = self.net(h_t)   # [B, N, D]

        return dh_dt


class NeuralODE(nn.Module):
    """
    Neural ODE temporal propagator.

    Takes the AGCRN output H_T as initial condition h(t₀),
    integrates the ODE forward, and returns the predicted
    degradation trajectory.

    MEMORY EFFICIENCY — THE ADJOINT METHOD:
        Standard backpropagation through an ODE solver requires storing
        ALL intermediate states (one per solver step). For 200 integration
        steps, that's 200× the hidden state memory — prohibitive.

        The adjoint sensitivity method solves a SECOND ODE backwards in time
        to compute gradients directly, using only the final state.
        Memory = O(1) regardless of solver steps.

        `odeint_adjoint` from torchdiffeq implements this automatically.
        You do not need to implement the adjoint — just use this wrapper.

    INFERENCE VS TRAINING:
        Training:  odeint_adjoint (memory-efficient, adjoint backprop)
        Inference: odeint (simpler, slightly faster, no adjoint needed
                   since we don't compute gradients at inference)
    """

    def __init__(self, config: dict):
        super().__init__()

        cfg_ode   = config['model']['neural_ode']
        cfg_graph = config['graph']

        self.hidden_dim = cfg_ode['hidden_dim']
        self.solver     = cfg_ode['solver']         # 'dopri5'
        self.rtol       = cfg_ode['rtol']           # 1e-3
        self.atol       = cfg_ode['atol']           # 1e-4
        self.use_adjoint = cfg_ode['adjoint']       # True for training

        # The ODE function f_θ
        self.ode_func = ODEFunction(
            hidden_dim = cfg_ode['hidden_dim'],
            mlp_hidden = cfg_ode['mlp_hidden'],
            n_layers   = cfg_ode['mlp_layers'],
            n_nodes    = cfg_graph['n_nodes'],
        )

        logger.info(
            f"NeuralODE initialized | "
            f"solver={self.solver}, rtol={self.rtol}, atol={self.atol}, "
            f"adjoint={self.use_adjoint}"
        )

    def forward(self,
                h0: torch.Tensor,
                t_span: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate the ODE from t₀ to t₁ (or multiple time points).

        Args:
            h0: [B, N, hidden_dim] — initial condition (AGCRN output H_T)
            t_span: 1D tensor of time points to evaluate at.
                    Minimum: [t₀, t₁] for start and end.
                    Extended: [t₀, t₀+Δ, t₀+2Δ, ..., t₁] for trajectory.

                    UNITS: relative time in "windows".
                    t₀ = 0.0 (end of input window)
                    t₁ = 1.0 (one step ahead)
                    This keeps values in a numerically stable range for the solver.

        Returns:
            trajectory: [len(t_span), B, N, hidden_dim]
                        Hidden state evaluated at each time point in t_span.
                        trajectory[0] = h(t₀) = h0 (by definition)
                        trajectory[-1] = h(t₁) = prediction horizon state

            h_final: [B, N, hidden_dim]
                     The hidden state at the last time point t₁.
                     This is what gets passed to the readout head for RUL.
        """
        # Choose between adjoint (training) and standard (inference) integration
        if self.use_adjoint and self.training:
            # odeint_adjoint: memory-efficient, O(1) memory, adjoint backprop
            trajectory = odeint(
                func        = self.ode_func,
                y0          = h0,
                t           = t_span,
                method      = self.solver,
                rtol        = self.rtol,
                atol        = self.atol,
                adjoint_params = list(self.ode_func.parameters()),
            )
        else:
            # Standard odeint for inference (no gradient needed, slightly faster)
            from torchdiffeq import odeint as odeint_standard
            with torch.no_grad():
                trajectory = odeint_standard(
                    func   = self.ode_func,
                    y0     = h0,
                    t      = t_span,
                    method = self.solver,
                    rtol   = self.rtol,
                    atol   = self.atol,
                )

        # trajectory shape: [T_points, B, N, hidden_dim]
        h_final = trajectory[-1]   # [B, N, hidden_dim]

        return trajectory, h_final

    def get_derivatives(self, h: torch.Tensor,
                        t: float = 0.0) -> torch.Tensor:
        """
        Compute dh/dt at a specific (h, t) without running the full ODE.

        Used by the physics constraint module to get the predicted
        degradation rate at each training timestep for comparison
        against Archard/Paris/Fourier reference rates.

        Args:
            h: [B, N, hidden_dim] — hidden state at which to evaluate derivative
            t: float — time value (default 0.0 for comparison purposes)

        Returns:
            dh_dt: [B, N, hidden_dim] — predicted derivative
        """
        t_tensor = torch.tensor(t, dtype=h.dtype, device=h.device)
        return self.ode_func(t_tensor, h)
