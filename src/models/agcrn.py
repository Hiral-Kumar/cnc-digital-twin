"""
agcrn.py
─────────────────────────────────────────────────────────────────────────────
Adaptive Graph Convolutional Recurrent Network (AGCRN)
Spatial Encoder for PC-NDT

WHAT THIS FILE IMPLEMENTS:
    AGCRN: Bai et al., NeurIPS 2020
    "Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting"

    We adapt AGCRN from traffic forecasting to CNC bearing degradation.
    The core innovation: instead of requiring a pre-specified sensor graph
    (adjacency matrix A), AGCRN learns A end-to-end from data via
    learnable node embeddings E:

        A = softmax( ReLU( E · Eᵀ ) )

    Combined with node-adaptive parameters (each sensor type gets its own
    transformation matrix derived from E), AGCRN's output H_T encodes each
    sensor's current state in the context of its learned neighborhood.

    This H_T becomes the initial condition h(t₀) for the Neural ODE.

ARCHITECTURE:
    Input: [Batch, TimeWindow, N_nodes, N_features]
           X(t₁), X(t₂), ..., X(t_W)

    ┌─────────────────────────────────────────┐
    │  Node Embeddings E ∈ ℝ^{N × d}         │
    │  (learnable, updated during training)   │
    └──────────────────┬──────────────────────┘
                       │
                       ▼
    ┌─────────────────────────────────────────┐
    │  Adaptive Adjacency                     │
    │  A = softmax(ReLU(E · Eᵀ)) ∈ ℝ^{N×N} │
    └──────────────────┬──────────────────────┘
                       │
                       ▼
    ┌─────────────────────────────────────────┐
    │  Graph-Convolved GRU (per timestep)     │
    │  Standard GRU gates, but each matrix    │
    │  multiplication replaced by             │
    │  graph-convolved operation A⊛(·)        │
    └──────────────────┬──────────────────────┘
                       │
                       ▼
    Output H_T: [Batch, N_nodes, hidden_dim]
    (spatially-contextualised node representations)
─────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class NodeEmbedding(nn.Module):
    """
    Learnable node embedding matrix E ∈ ℝ^{N × d}.

    Each of the N sensor nodes gets a d-dimensional learnable vector.
    During training, these embeddings are updated by backpropagation
    so that nodes with correlated degradation behaviors end up with
    similar embedding vectors (and thus stronger edges in A).

    WHY EMBEDDINGS INSTEAD OF FIXED FEATURES:
        Fixed features (like bearing position on shaft) are prior knowledge
        that constrains what the graph can learn. Learnable embeddings let
        the model discover unexpected coupling relationships — for example,
        a bearing far from the failed one might develop a strong edge if
        it consistently shows early thermal signatures before others.
    """

    def __init__(self, n_nodes: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Parameter(
            torch.randn(n_nodes, embedding_dim) * 0.1
        )
        # Initialize small (×0.1) to prevent large initial A values
        # which would make the softmax output nearly uniform (vanishing gradients)

    def forward(self) -> torch.Tensor:
        """Return the embedding matrix E ∈ ℝ^{N × d}."""
        return self.embedding


class AdaptiveAdjacency(nn.Module):
    """
    Compute the learned adjacency matrix from node embeddings.

        A = softmax( ReLU( E · Eᵀ ) )

    WHY THIS FORMULA:
        E · Eᵀ ∈ ℝ^{N×N}  → similarity score between all node pairs.
                              High dot product = similar embeddings = strong edge.
        ReLU(·)             → zeroes out negative similarities.
                              Negative dot products have no physical meaning
                              as coupling strengths.
        softmax(·)          → row-normalizes into a valid attention distribution.
                              Each node distributes 100% of "attention" across
                              its neighbors, interpretable as coupling fractions.

    WHY NOT USE A FIXED GRAPH (like DCRNN):
        In CNC machines, sensor coupling strengths depend on machine geometry,
        material properties, and operating state — quantities that cannot be
        reliably hand-engineered. Learning A from data handles all of this
        automatically and generalises across machine configurations.
    """

    def __init__(self):
        super().__init__()

    def forward(self, E: torch.Tensor) -> torch.Tensor:
        """
        Args:
            E: [N, d] — node embedding matrix

        Returns:
            A: [N, N] — learned adjacency matrix (row-normalized)
        """
        # [N, d] × [d, N] = [N, N] — pairwise similarity matrix
        similarity = torch.matmul(E, E.transpose(0, 1))

        # Remove negative values (no negative coupling strengths physically)
        similarity = F.relu(similarity)

        # Row-wise softmax: each row is a probability distribution over neighbors
        # dim=1 means we normalize across columns (neighbors) for each row (node)
        A = F.softmax(similarity, dim=1)

        return A


class NodeAdaptiveGraphConv(nn.Module):
    """
    Graph convolution with node-adaptive (per-node) transformation matrices.

    Standard graph convolution: H_new = A × H × W
        One shared weight matrix W — ALL nodes use the SAME transformation.

    Node-adaptive graph convolution: H_new_i = Σ_j A_ij × H_j × W_i
        Each node i has its OWN transformation matrix W_i.

    WHY NODE-ADAPTIVE MATTERS:
        In a CNC machine, accelerometers and thermocouples process signals
        with completely different statistical properties:
        - Accelerometers: high-frequency vibration, RMS in m/s²
        - Thermocouples: slow thermal drift, values in °C

        A single shared W would be forced to compromise between these
        very different signal types. Node-adaptive W_i lets each sensor
        type learn the transformation that best extracts its contribution
        to the shared hidden state.

    IMPLEMENTATION (following Bai et al. 2020):
        W_i = E_i · W_pool   where E_i ∈ ℝ^d is node i's embedding
                              and W_pool ∈ ℝ^{d × (hidden × input)} is shared

        This gives N different W_i matrices while only storing one W_pool
        — parameter-efficient and fully differentiable.
    """

    def __init__(self, in_dim: int, out_dim: int,
                 embedding_dim: int, cheb_k: int = 2):
        """
        Args:
            in_dim:       input feature dimension per node
            out_dim:      output hidden dimension per node
            embedding_dim: node embedding dimension d
            cheb_k:       Chebyshev polynomial order (approximation depth)
                          cheb_k=2 uses A⁰ (identity) and A¹ (one-hop)
                          cheb_k=3 adds A² (two-hop neighbors)
        """
        super().__init__()
        self.in_dim  = in_dim
        self.out_dim = out_dim
        self.cheb_k  = cheb_k

        # W_pool ∈ ℝ^{d × (cheb_k × in_dim × out_dim)}
        # From this, each node derives its own transformation matrices
        self.W_pool = nn.Parameter(
            torch.randn(embedding_dim, cheb_k * in_dim * out_dim) * 0.01
        )
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, X: torch.Tensor,
                A: torch.Tensor,
                E: torch.Tensor) -> torch.Tensor:
        """
        Apply node-adaptive graph convolution.

        Args:
            X: [B, N, in_dim]   — node features (batch of graphs)
            A: [N, N]            — adjacency matrix (shared across batch)
            E: [N, embedding_dim] — node embeddings

        Returns:
            H: [B, N, out_dim]  — transformed node features
        """
        B, N, _ = X.shape

        # Compute Chebyshev basis: [A⁰X, A¹X, ..., A^{k-1}X]
        # A⁰X = X (identity — each node's own features)
        # A¹X = A×X (one-hop neighborhood aggregation)
        # A²X = A×A×X (two-hop — neighbors of neighbors)
        support_list = [X]  # A⁰X = X
        Ax = X
        for _ in range(1, self.cheb_k):
            # A is [N, N], Ax is [B, N, in_dim]
            # Einsum: for each batch item, multiply [N,N] × [N,in_dim]
            Ax = torch.einsum('nm,bmd->bnd', A, Ax)
            support_list.append(Ax)

        # Concatenate Chebyshev basis along feature dim
        # Each: [B, N, in_dim] → cat → [B, N, cheb_k × in_dim]
        support = torch.cat(support_list, dim=-1)

        # Compute node-specific weight matrices from embeddings
        # E: [N, d] × W_pool: [d, cheb_k×in×out] → [N, cheb_k×in×out]
        node_weights = torch.matmul(E, self.W_pool)
        # Reshape to [N, cheb_k×in_dim, out_dim]
        node_weights = node_weights.view(N, self.cheb_k * self.in_dim, self.out_dim)

        # Apply per-node transformation
        # support: [B, N, cheb_k×in] → bmm with [N, cheb_k×in, out]
        # Result: [B, N, out_dim]
        output = torch.einsum('bni,nio->bno', support, node_weights)
        output = output + self.bias

        return output


class AGCRNCell(nn.Module):
    """
    A single AGCRN cell: one timestep of the graph-convolved GRU.

    Standard GRU update equations:
        r_t = σ(W_r × [X_t, H_{t-1}] + b_r)          ← reset gate
        u_t = σ(W_u × [X_t, H_{t-1}] + b_u)          ← update gate
        C_t = tanh(W_c × [X_t, r_t ⊙ H_{t-1}] + b_c) ← candidate state
        H_t = u_t ⊙ H_{t-1} + (1-u_t) ⊙ C_t          ← new hidden state

    AGCRN replaces each W × (·) with NodeAdaptiveGraphConv(A, E, ·):
        r_t = σ( GraphConv([X_t, H_{t-1}], A, E, W_r) )
        u_t = σ( GraphConv([X_t, H_{t-1}], A, E, W_u) )
        C_t = tanh( GraphConv([X_t, r_t ⊙ H_{t-1}], A, E, W_c) )
        H_t = u_t ⊙ H_{t-1} + (1-u_t) ⊙ C_t

    Each gate's GraphConv uses the SAME adjacency A and embeddings E,
    but DIFFERENT W_pool matrices (W_r, W_u, W_c — separate parameters).
    """

    def __init__(self, in_features: int, hidden_dim: int,
                 embedding_dim: int, cheb_k: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim

        # GRU gates — each is a separate NodeAdaptiveGraphConv
        # Input to each gate: [X_t || H_{t-1}] → in_features + hidden_dim
        gate_input_dim = in_features + hidden_dim

        self.reset_conv  = NodeAdaptiveGraphConv(gate_input_dim, hidden_dim,
                                                  embedding_dim, cheb_k)
        self.update_conv = NodeAdaptiveGraphConv(gate_input_dim, hidden_dim,
                                                  embedding_dim, cheb_k)
        self.cand_conv   = NodeAdaptiveGraphConv(gate_input_dim, hidden_dim,
                                                  embedding_dim, cheb_k)

    def forward(self,
                X: torch.Tensor,
                H: torch.Tensor,
                A: torch.Tensor,
                E: torch.Tensor) -> torch.Tensor:
        """
        One timestep of the graph-convolved GRU.

        Args:
            X: [B, N, in_features]  — input features at current timestep
            H: [B, N, hidden_dim]   — previous hidden state
            A: [N, N]               — learned adjacency
            E: [N, embedding_dim]   — node embeddings

        Returns:
            H_new: [B, N, hidden_dim] — updated hidden state
        """
        # Concatenate input and previous hidden state along feature dim
        XH = torch.cat([X, H], dim=-1)   # [B, N, in_features + hidden_dim]

        # Reset gate: how much of the previous state to "forget"
        r = torch.sigmoid(self.reset_conv(XH, A, E))        # [B, N, hidden_dim]

        # Update gate: how much to update toward the candidate state
        u = torch.sigmoid(self.update_conv(XH, A, E))       # [B, N, hidden_dim]

        # Candidate state: the proposed new hidden state content
        # Uses reset gate to selectively incorporate previous state
        X_r = torch.cat([X, r * H], dim=-1)   # [B, N, in_features + hidden_dim]
        C = torch.tanh(self.cand_conv(X_r, A, E))           # [B, N, hidden_dim]

        # New hidden state: interpolate between old state and candidate
        # When u=0: keep old state (nothing to learn at this timestep)
        # When u=1: completely replace with candidate (major change detected)
        H_new = u * H + (1 - u) * C                         # [B, N, hidden_dim]

        return H_new


class AGCRN(nn.Module):
    """
    Full Adaptive Graph Convolutional Recurrent Network.

    Processes a sequence of T timesteps through stacked AGCRN cells,
    returning the final hidden state H_T as the spatial encoding of
    the machine's collective sensor state.

    H_T serves as h(t₀) — the initial condition for the Neural ODE.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: full config dict from config.yaml
        """
        super().__init__()

        cfg = config['model']['agcrn']
        n_nodes   = config['graph']['n_nodes']
        n_features = config['preprocessing']['n_features']

        self.n_nodes    = n_nodes
        self.hidden_dim = cfg['hidden_dim']
        self.n_layers   = cfg['n_layers']

        # Learnable node embeddings shared across all layers and all cells
        self.node_embeddings = NodeEmbedding(n_nodes, cfg['embedding_dim'])

        # Adaptive adjacency module (stateless — recomputed from embeddings)
        self.adaptive_adj = AdaptiveAdjacency()

        # Stack of AGCRN cells (one per layer)
        # First layer: input = n_features
        # Deeper layers: input = hidden_dim (output of previous layer)
        self.cells = nn.ModuleList()
        for layer in range(cfg['n_layers']):
            in_dim = n_features if layer == 0 else cfg['hidden_dim']
            self.cells.append(
                AGCRNCell(
                    in_features  = in_dim,
                    hidden_dim   = cfg['hidden_dim'],
                    embedding_dim= cfg['embedding_dim'],
                    cheb_k       = cfg['cheb_k'],
                )
            )

        logger.info(
            f"AGCRN initialized | N={n_nodes}, "
            f"hidden={cfg['hidden_dim']}, "
            f"layers={cfg['n_layers']}, "
            f"cheb_k={cfg['cheb_k']}"
        )

    def forward(self, X_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a full input sequence through all AGCRN layers.

        Args:
            X_seq: [B, T, N, F] — sequence of T timesteps
                   B = batch size
                   T = window size (50 by default)
                   N = n_nodes (4)
                   F = n_features (5)

        Returns:
            H_T:  [B, N, hidden_dim] — final hidden state after T timesteps
                  This is the spatially-contextualised encoding of the
                  machine's collective sensor state — becomes h(t₀) for ODE
            A:    [N, N] — the learned adjacency matrix
                  Returned for logging, visualization, and Fourier constraint
        """
        B, T, N, F = X_seq.shape

        # Compute shared graph structure (same for all timesteps in this batch)
        E = self.node_embeddings()          # [N, d]
        A = self.adaptive_adj(E)            # [N, N]

        # Initialize hidden states for all layers
        # Hidden state starts at zero (no prior history at window start)
        H_layers = [
            torch.zeros(B, N, self.hidden_dim, device=X_seq.device)
            for _ in range(self.n_layers)
        ]

        # Process each timestep in the window
        for t in range(T):
            X_t = X_seq[:, t, :, :]          # [B, N, F] — current timestep

            # Pass through each AGCRN layer
            for layer_idx, cell in enumerate(self.cells):
                # First layer receives raw features
                # Deeper layers receive previous layer's hidden state
                layer_input = X_t if layer_idx == 0 else H_layers[layer_idx - 1]
                H_layers[layer_idx] = cell(layer_input,
                                           H_layers[layer_idx],
                                           A, E)

        # Return the FINAL hidden state of the LAST layer
        H_T = H_layers[-1]   # [B, N, hidden_dim]

        return H_T, A

    def get_adjacency(self) -> torch.Tensor:
        """
        Return the current learned adjacency matrix (detached from graph).
        Used for visualization and comparison with physical prior.
        """
        E = self.node_embeddings()
        A = self.adaptive_adj(E)
        return A.detach()
