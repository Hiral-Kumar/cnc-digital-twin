"""
src/data/__init__.py
────────────────────────────────────────────────────────────────
Data package for PC-NDT.

Exposes the public API for all data loading and preprocessing.
Users of this package import from here, not from individual modules.

Example:
    from src.data import IMSLoader, PronostiaLoader, build_datasets
"""

from .ims_loader       import IMSLoader
from .pronostia_loader import PronostiaLoader
from .preprocessing    import (
    MinMaxNormalizer,
    chronological_split,
    create_sliding_windows,
    BearingRULDataset,
    build_datasets,
)
from .graph_utils      import (
    build_proximity_adjacency,
    compute_graph_laplacian,
    compare_adjacency_matrices,
)

__all__ = [
    "IMSLoader",
    "PronostiaLoader",
    "MinMaxNormalizer",
    "chronological_split",
    "create_sliding_windows",
    "BearingRULDataset",
    "build_datasets",
    "build_proximity_adjacency",
    "compute_graph_laplacian",
    "compare_adjacency_matrices",
]
