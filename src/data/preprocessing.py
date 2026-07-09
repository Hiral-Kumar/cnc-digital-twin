"""
preprocessing.py
─────────────────────────────────────────────────────────────────────────────
Preprocessing Pipeline for PC-NDT

WHAT THIS FILE DOES:
    Takes the raw feature matrices from IMSLoader / PronostiaLoader and
    applies all the ML-specific transformations needed before the model
    can use the data:

    1. MinMax Normalization — fitted on training data ONLY
    2. Chronological train/val split within a single run
    3. Sliding window segmentation with temporal oversampling
    4. PyTorch Dataset wrapper for use with DataLoader

WHY THE ORDERING MATTERS:
    Fit normalization BEFORE creating windows. If you fit after, the
    normalization statistics will be computed on the test-contaminated
    full feature matrix — data leakage.

KEY DESIGN DECISION — TEMPORAL OVERSAMPLING:
    Near-failure timesteps (last 20% of run) are sampled with stride=1.
    Healthy timesteps (first 80%) are sampled with stride=5.
    This creates a more balanced dataset without fabricating data or
    duplicating signals — just sliding the window more densely where
    the model needs the most examples.
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class MinMaxNormalizer:
    """
    Feature-wise MinMax normalization: scales each feature to [0, 1].

    CRITICAL RULE: Always call .fit() on TRAINING data only.
    Then call .transform() on training, validation, and test sets
    using the statistics learned from training alone.

    This prevents data leakage — test statistics must never influence
    the normalization applied during training.
    """

    def __init__(self):
        self.feat_min = None   # [N_nodes, N_features] — learned from training set
        self.feat_max = None   # [N_nodes, N_features] — learned from training set
        self._fitted = False

    def fit(self, features: np.ndarray) -> 'MinMaxNormalizer':
        """
        Compute min and max statistics from training features.

        Args:
            features: np.ndarray [N_files, N_nodes, N_features]
                      This must be the TRAINING split only.

        Returns:
            self (for method chaining: normalizer.fit(X).transform(X))
        """
        # Compute min/max across the time dimension (axis=0)
        # Result shape: [N_nodes, N_features]
        # This gives each (node, feature) pair its own scale —
        # bearing 1 RMS and bearing 3 RMS may have very different ranges
        self.feat_min = features.min(axis=0)  # [N_nodes, N_features]
        self.feat_max = features.max(axis=0)  # [N_nodes, N_features]
        self._fitted = True

        logger.info(f"  Normalizer fitted | features shape: {features.shape}")
        logger.debug(f"  Feature mins: {self.feat_min.mean(axis=0)}")
        logger.debug(f"  Feature maxs: {self.feat_max.mean(axis=0)}")

        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Apply normalization to features using TRAINING statistics.

        Args:
            features: np.ndarray [N_files, N_nodes, N_features]

        Returns:
            np.ndarray [N_files, N_nodes, N_features] scaled to [0, 1]
        """
        if not self._fitted:
            raise RuntimeError(
                "MinMaxNormalizer has not been fitted yet. "
                "Call .fit(training_features) before .transform()."
            )

        # Broadcast: feat_min/max are [N_nodes, N_features]
        # features are [N_files, N_nodes, N_features]
        # Subtraction broadcasts correctly along axis 0 (time dimension)
        denom = self.feat_max - self.feat_min
        # Prevent division by zero for constant features
        denom = np.where(denom < 1e-10, 1.0, denom)

        normalized = (features - self.feat_min) / denom

        # Clip to [0, 1] — test data may occasionally fall outside training range
        normalized = np.clip(normalized, 0.0, 1.0)

        return normalized.astype(np.float32)

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Convenience method: fit on features, then transform them."""
        return self.fit(features).transform(features)


def chronological_split(features: np.ndarray,
                         rul: np.ndarray,
                         train_frac: float = 0.60,
                         val_frac: float = 0.20) -> dict:
    """
    Split a run-to-failure dataset chronologically into train/val/mini-test.

    WHY NOT RANDOM SPLIT:
        If you randomly pick 80% of timesteps for training, some will come
        from near the END of the run (near failure). The test set then
        contains timesteps from BEFORE those — the model has "seen the future".
        This is one of the most common mistakes in time-series ML papers.
        Always split chronologically for any sequential degradation dataset.

    Args:
        features: np.ndarray [N_files, N_nodes, N_features]
        rul:      np.ndarray [N_files, N_nodes]
        train_frac: fraction of timeline for training (e.g., 0.60)
        val_frac:   fraction of timeline for validation (e.g., 0.20)
                    Remaining (1 - train_frac - val_frac) = mini-test

    Returns:
        dict with 'train', 'val', 'mini_test' each containing
        {'features': ..., 'rul': ...}
    """
    n_files = features.shape[0]
    train_end = int(n_files * train_frac)
    val_end = int(n_files * (train_frac + val_frac))

    return {
        'train': {
            'features': features[:train_end],
            'rul': rul[:train_end],
        },
        'val': {
            'features': features[train_end:val_end],
            'rul': rul[train_end:val_end],
        },
        'mini_test': {
            'features': features[val_end:],
            'rul': rul[val_end:],
        },
    }


def create_sliding_windows(features: np.ndarray,
                            rul: np.ndarray,
                            window_size: int = 50,
                            stride_healthy: int = 5,
                            stride_degraded: int = 1,
                            degraded_threshold: float = 0.80) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window samples with temporal oversampling near failure.

    WHAT A SLIDING WINDOW IS:
        Given a timeline of 984 timesteps, a window of size 50 means:
        Sample 1: timesteps [0..49]  → predict RUL at timestep 50
        Sample 2: timesteps [1..50]  → predict RUL at timestep 51
        ... and so on.

        The stride controls how many timesteps we advance between samples.
        stride=1 → very dense (lots of overlapping windows)
        stride=5 → sparser (faster training, less redundancy in healthy phase)

    TEMPORAL OVERSAMPLING:
        Problem: 80%+ of timesteps represent healthy operation.
                 Only 20% represent the critical near-failure regime.
        Solution: Use stride=1 (dense) in the last 20% of each run.
                  Use stride=5 (sparse) in the first 80%.
        This increases the proportion of near-failure examples in each
        training batch without fabricating data or duplicating signals.

    Args:
        features: [N_files, N_nodes, N_features]
        rul:      [N_files, N_nodes]
        window_size: number of timesteps per sample
        stride_healthy: stride during first (1-degraded_threshold) of run
        stride_degraded: stride during last degraded_threshold of run
        degraded_threshold: fraction of run considered "healthy"

    Returns:
        X: np.ndarray [N_windows, window_size, N_nodes, N_features]
        y: np.ndarray [N_windows, N_nodes] — RUL at the timestep AFTER window
    """
    n_files, n_nodes, n_features = features.shape
    degraded_start = int(n_files * degraded_threshold)

    X_windows = []
    y_windows = []

    t = 0
    while t + window_size < n_files:
        # Extract window: features[t : t+window_size]
        X_windows.append(features[t: t + window_size])  # [W, N, F]
        y_windows.append(rul[t + window_size])           # [N] — label is NEXT timestep

        # Advance by appropriate stride
        if t < degraded_start:
            t += stride_healthy
        else:
            t += stride_degraded

    if len(X_windows) == 0:
        raise ValueError(
            f"No windows created. Check that your dataset has more than "
            f"{window_size} files and your stride settings are valid."
        )

    X = np.stack(X_windows, axis=0)  # [N_windows, W, N, F]
    y = np.stack(y_windows, axis=0)  # [N_windows, N]

    logger.info(
        f"  Created {len(X_windows)} windows | "
        f"Input shape: {X.shape} | Target shape: {y.shape}"
    )

    return X, y


class BearingRULDataset(Dataset):
    """
    PyTorch Dataset wrapping the sliding window bearing data.

    WHY A PYTORCH DATASET:
        PyTorch's DataLoader requires a Dataset object. The Dataset
        handles indexing and type conversion; the DataLoader handles
        batching, shuffling (within a split), and parallel loading.

        IMPORTANT: Even though we have chronological data, we CAN shuffle
        windows within the training set — shuffling at the window level
        doesn't cause leakage because each window's label is already
        correctly tied to its input. What you must NEVER shuffle is the
        split boundary itself (i.e., which timesteps go into train vs. test).

    Usage:
        dataset = BearingRULDataset(X, y)
        loader  = DataLoader(dataset, batch_size=32, shuffle=True)
        for X_batch, y_batch in loader:
            # X_batch: [32, 50, 4, 5] — torch.float32
            # y_batch: [32, 4]         — torch.float32
            pass
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: np.ndarray [N_windows, window_size, N_nodes, N_features]
            y: np.ndarray [N_windows, N_nodes]
        """
        # Convert to tensors once during initialization (not during __getitem__)
        # This avoids repeated numpy→tensor conversion overhead during training
        self.X = torch.from_numpy(X).float()  # [N_windows, W, N, F]
        self.y = torch.from_numpy(y).float()  # [N_windows, N]

        assert len(self.X) == len(self.y), (
            f"Feature and label arrays must have same length. "
            f"Got X: {len(self.X)}, y: {len(self.y)}"
        )

    def __len__(self) -> int:
        """Return total number of windows in this dataset."""
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return one (input_window, target_rul) pair.

        Args:
            idx: window index

        Returns:
            X_window: torch.Tensor [window_size, N_nodes, N_features]
            y_rul:    torch.Tensor [N_nodes]
        """
        return self.X[idx], self.y[idx]

    @property
    def input_shape(self) -> tuple:
        """Shape of a single input window (without batch dimension)."""
        return tuple(self.X.shape[1:])  # [W, N, F]

    @property
    def n_nodes(self) -> int:
        return self.X.shape[2]

    @property
    def n_features(self) -> int:
        return self.X.shape[3]

    @property
    def window_size(self) -> int:
        return self.X.shape[1]


def build_datasets(loaded_data: dict, config: dict,
                   normalizer: Optional[MinMaxNormalizer] = None,
                   is_training_run: bool = True) -> dict:
    """
    Full preprocessing pipeline from loaded data to ready-to-use PyTorch Datasets.

    This function ties together normalization, splitting, windowing, and
    Dataset creation into a single call. Use this from your training script.

    Args:
        loaded_data: dict from IMSLoader.load_test() or PronostiaLoader.load_run()
        config:      full config dict from config.yaml
        normalizer:  if None and is_training_run=True, fits a new normalizer.
                     if provided, applies existing normalizer (for val/test sets).
        is_training_run: if True, fits normalizer on train split.
                         if False, requires normalizer to be provided.

    Returns:
        dict with keys 'train', 'val', 'mini_test' each containing a
        BearingRULDataset, plus 'normalizer' for use on subsequent sets.
    """
    cfg_prep  = config['preprocessing']
    cfg_split = config['splits']

    features = loaded_data['features']  # [N_files, N_nodes, N_features]
    rul      = loaded_data['rul']       # [N_files, N_nodes]

    # Step 1: Chronological split (before normalization)
    splits = chronological_split(
        features, rul,
        train_frac=cfg_split['train_fraction'],
        val_frac=cfg_split['val_fraction'],
    )

    # Step 2: Normalization (fit on training split ONLY)
    if is_training_run:
        normalizer = MinMaxNormalizer()
        splits['train']['features'] = normalizer.fit_transform(
            splits['train']['features']
        )
        logger.info("  Fitted and applied normalizer to training split")
    else:
        if normalizer is None:
            raise ValueError(
                "For non-training runs, a pre-fitted normalizer must be provided."
            )

    # Apply training normalizer to val and mini_test splits
    splits['val']['features'] = normalizer.transform(splits['val']['features'])
    splits['mini_test']['features'] = normalizer.transform(
        splits['mini_test']['features']
    )

    # Step 3: Create sliding windows for each split
    datasets = {}
    for split_name, split_data in splits.items():
        # Use degraded stride for all splits except training
        # (val/test don't need oversampling — we want all timesteps)
        stride_h = cfg_prep['stride_healthy'] if split_name == 'train' else 1
        stride_d = cfg_prep['stride_degraded']

        X_win, y_win = create_sliding_windows(
            features=split_data['features'],
            rul=split_data['rul'],
            window_size=cfg_prep['window_size'],
            stride_healthy=stride_h,
            stride_degraded=stride_d,
            degraded_threshold=cfg_prep['degraded_threshold'],
        )

        datasets[split_name] = BearingRULDataset(X_win, y_win)
        logger.info(f"  {split_name}: {len(datasets[split_name])} windows")

    datasets['normalizer'] = normalizer
    return datasets
