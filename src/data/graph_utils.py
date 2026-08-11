"""
graph_utils.py
─────────────────────────────────────────────────────────────────────────────
Graph Utility Functions for PC-NDT

WHAT THIS FILE DOES:
    1. Builds a physical proximity adjacency matrix from known shaft geometry.
       This is the BASELINE (prior) graph — NOT used in training.
       It serves ONLY as a sanity check to validate the AGCRN-learned graph.

    2. Computes the Graph Laplacian L = D - A, which appears directly in
       the Fourier heat constraint loss.

    3. Provides a comparison utility: how similar is the learned adjacency
       to the physical prior? This quantitative comparison is your
       qualitative RQ1 evidence.

WHY THE GRAPH LAPLACIAN MATTERS FOR FOURIER:
    Fourier's heat equation on a continuous domain: ∂T/∂t = α∇²T + Q/(ρc)

    On a DISCRETE graph, the Laplace operator ∇² is replaced by the
    graph Laplacian L:
        ∂T_i/∂t ≈ α × Σ_j L_ij × T_j + Q_i/(ρ·c)

    where L_ij = (D - A)_ij:
        D_ii = Σ_j A_ij  (degree matrix — sum of edge weights for node i)
        A_ij = adjacency weight between nodes i and j

    The beauty of your framework: the SAME adjacency A learned by AGCRN
    appears in the Laplacian L used for the Fourier constraint.
    The graph is jointly optimised for predictive accuracy AND thermal
    physics consistency — they share one learned structure.
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import torch
from typing import Union
import logging

logger = logging.getLogger(__name__)

ArrayLike = Union[np.ndarray, torch.Tensor]


def build_proximity_adjacency(shaft_distances: list,
                               sigma: float = 50.0) -> np.ndarray:
    """
    Build a physical proximity adjacency matrix from shaft geometry.

    WHAT IT REPRESENTS:
        Bearings mounted closer together on the shaft have stronger
        mechanical coupling (vibration propagates with less attenuation
        over shorter distances). This is encoded as a Gaussian kernel
        on the shaft distance between each pair of bearings.

    FORMULA:
        A_ij = exp(−dist(i,j)² / (2σ²))     if i ≠ j
        A_ii = 0                              (no self-loops)

    The resulting matrix is symmetric and normalized row-wise so that
    each bearing's influence distribution sums to 1.

    WHY THIS IS NOT USED IN TRAINING:
        AGCRN learns A from data — it should discover this structure
        without being told. If we provided A_prior as input, we would
        be answering RQ1 ("can the model discover the topology?")
        before it has a chance to try. We keep this as a validation
        reference only.

    Args:
        shaft_distances: 2D list [N×N] of inter-bearing shaft distances (mm).
                         From config.yaml → graph.shaft_distances
        sigma:           Gaussian kernel bandwidth (mm). Controls how quickly
                         coupling strength decays with distance.
                         50mm ≈ half-decay at 50mm shaft separation.

    Returns:
        np.ndarray [N, N] — symmetric, row-normalized proximity adjacency
    """
    distances = np.array(shaft_distances, dtype=np.float64)
    n = distances.shape[0]

    # Apply Gaussian kernel to distances
    # exp(-d²/2σ²) gives 1.0 for d=0 (same bearing) and decays with distance
    A = np.exp(-distances ** 2 / (2 * sigma ** 2))

    # Remove self-loops: diagonal = 0
    # Self-loops would mean a bearing is coupled to itself, which has no
    # physical meaning in the graph convolution context
    np.fill_diagonal(A, 0.0)

    # Row-normalize: each row sums to 1
    # This makes A_ij interpretable as "what fraction of node i's information
    # comes from node j" — a probability distribution over neighbors
    row_sums = A.sum(axis=1, keepdims=True)
    # Avoid division by zero (isolated nodes have zero row sum)
    row_sums = np.maximum(row_sums, 1e-10)
    A_normalized = A / row_sums

    logger.debug(f"Proximity adjacency built for N={n} nodes")
    logger.debug(f"  Max coupling: {A_normalized.max():.4f}")
    logger.debug(f"  Min coupling: {A_normalized[A_normalized > 0].min():.4f}")

    return A_normalized.astype(np.float32)


def compute_graph_laplacian(adjacency: ArrayLike) -> Union[np.ndarray, torch.Tensor]:
    """
    Compute the normalized graph Laplacian L = D - A.

    WHAT THE LAPLACIAN DOES:
        The Laplacian operator measures "how different a node is from
        its neighbors". Multiplying a signal vector T by L gives:
            (L·T)_i = Σ_j A_ij × (T_i - T_j)
                    = D_ii × T_i - Σ_j A_ij × T_j

        In the Fourier heat equation context:
            ∂T_i/∂t = −α × (L·T)_i + Q_i/(ρ·c)

        This means heat flows OUT from hot nodes (positive L·T term
        pulls temperature down) toward cooler neighbors, with the rate
        proportional to the temperature DIFFERENCE — exactly Fourier's law.

    WHY THE GRAPH LAPLACIAN IS THE RIGHT DISCRETE ANALOGUE:
        The continuous Laplacian ∇²T measures the second spatial
        derivative (curvature) of the temperature field. On a graph,
        the "spatial" structure is the adjacency. The graph Laplacian
        is the canonical discrete analogue of the continuous Laplacian,
        proven to converge to the continuous operator as the graph
        becomes denser.

    Args:
        adjacency: [N, N] adjacency matrix (numpy or torch tensor)

    Returns:
        Graph Laplacian of same type and device as input
    """
    is_torch = isinstance(adjacency, torch.Tensor)

    if is_torch:
        A = adjacency
        # Degree matrix D: diagonal matrix of row sums
        degree = A.sum(dim=-1)            # [N]
        D = torch.diag(degree)            # [N, N]
        L = D - A                          # [N, N]
    else:
        A = np.asarray(adjacency, dtype=np.float32)
        degree = A.sum(axis=-1)
        D = np.diag(degree)
        L = D - A

    return L


def compare_adjacency_matrices(A_learned: np.ndarray,
                                A_prior: np.ndarray) -> dict:
    """
    Quantify how similar the learned adjacency is to the physical prior.

    This is your RQ1 evidence: if A_learned correlates strongly with A_prior,
    the AGCRN has discovered the physically expected sensor coupling structure
    without being told what it should be.

    We use three complementary metrics:
        1. Pearson correlation — linear relationship between off-diagonal entries
        2. Rank correlation (Spearman) — preserves only the ordering
        3. Mean absolute difference — raw numerical distance

    Args:
        A_learned: [N, N] adjacency learned by AGCRN (numpy array)
        A_prior:   [N, N] physical proximity adjacency from build_proximity_adjacency()

    Returns:
        dict with comparison metrics
    """
    from scipy import stats

    n = A_learned.shape[0]

    # Extract off-diagonal elements only (diagonal = 0 in both, trivially correlated)
    mask = ~np.eye(n, dtype=bool)
    learned_flat = A_learned[mask].ravel()
    prior_flat   = A_prior[mask].ravel()

    pearson_r,  pearson_p  = stats.pearsonr(learned_flat, prior_flat)
    spearman_r, spearman_p = stats.spearmanr(learned_flat, prior_flat)
    mae = float(np.mean(np.abs(learned_flat - prior_flat)))

    results = {
        'pearson_r':    float(pearson_r),
        'pearson_p':    float(pearson_p),
        'spearman_r':   float(spearman_r),
        'spearman_p':   float(spearman_p),
        'mae':          mae,
        'n_nodes':      n,
    }

    logger.info(
        f"Adjacency comparison → "
        f"Pearson r={pearson_r:.3f} (p={pearson_p:.3e}), "
        f"Spearman r={spearman_r:.3f}, MAE={mae:.4f}"
    )

    return results


def adjacency_to_edge_index(adjacency: np.ndarray,
                              threshold: float = 0.01) -> np.ndarray:
    """
    Convert a dense adjacency matrix to a sparse edge list.

    Used for visualization and debugging — not in the main training loop.

    Args:
        adjacency: [N, N] dense adjacency matrix
        threshold: edges with weight below this are considered absent

    Returns:
        np.ndarray [2, E] — edge index in COO format (source, target pairs)
    """
    rows, cols = np.where(adjacency > threshold)
    edge_index = np.stack([rows, cols], axis=0)
    return edge_index
