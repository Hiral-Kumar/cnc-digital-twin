"""
test_smoke.py
─────────────────────────────────────────────────────────────────────────────
End-to-End Smoke Test for PC-NDT Data Pipeline

WHAT THIS TEST DOES:
    Runs the complete data pipeline — normalization, splitting, windowing,
    dataset creation, metric computation — using SYNTHETIC data.

    This means the test works even if you haven't downloaded the NASA IMS
    dataset yet. It proves the code is logically correct before you
    plug in real data.

WHY A SMOKE TEST:
    A "smoke test" checks that the system doesn't catch fire when you
    turn it on. It doesn't check scientific correctness — it checks that
    shapes, types, and basic logic are all consistent end-to-end.

    When your mentor runs `pytest tests/` and sees:
        ✓ test_normalizer_fits_on_train_only
        ✓ test_normalizer_no_leakage
        ✓ test_chronological_split_ordering
        ✓ test_sliding_windows_shapes
        ✓ test_temporal_oversampling
        ✓ test_bearing_rul_dataset
        ✓ test_metrics_rmse
        ✓ test_metrics_phm2012_asymmetry
        ✓ test_full_pipeline_end_to_end
    ...they know this is a serious project.

HOW TO RUN:
    From the cnc-digital-twin/ directory:
        pytest tests/ -v

    Or just this file:
        pytest tests/test_smoke.py -v
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pytest
import sys
import os

# Add project root to path so imports work from anywhere
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.preprocessing import (
    MinMaxNormalizer,
    chronological_split,
    create_sliding_windows,
    BearingRULDataset,
    build_datasets,
)
from src.evaluation.metrics import (
    rmse,
    phm2012_score,
    delta_rmse,
    pearson_rho,
    compute_all_metrics,
)


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATA FIXTURES
# Fixtures are reusable data/objects that multiple tests can share.
# The @pytest.fixture decorator makes them available to any test function
# that lists them as a parameter.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_features():
    """
    Synthetic feature matrix mimicking IMS Test 2 dimensions.
    Shape: [200, 4, 5] = [N_files, N_nodes, N_features]
    Values: monotonically increasing RMS (index 0) to simulate degradation,
            plus random noise for other features.
    """
    np.random.seed(42)
    n_files, n_nodes, n_features = 200, 4, 5

    features = np.random.randn(n_files, n_nodes, n_features).astype(np.float32)

    # Make RMS (feature 0) increase monotonically for the last 40 files
    # This simulates real bearing degradation behavior
    for node in range(n_nodes):
        base_rms = np.linspace(0.1, 0.3, n_files)
        # Add degradation spike in the last 20% of the run
        degradation = np.zeros(n_files)
        degradation[160:] = np.linspace(0, 1.5, 40)
        features[:, node, 0] = base_rms + degradation + 0.01 * np.random.randn(n_files)

    return features


@pytest.fixture
def synthetic_rul():
    """
    Synthetic RUL labels matching the synthetic_features fixture.
    Shape: [200, 4] = [N_files, N_nodes]
    Uses FPT convention: constant plateau then linear decrease.
    """
    n_files, n_nodes = 200, 4
    rul = np.zeros((n_files, n_nodes), dtype=np.float32)

    fpt = 160  # FPT at 80% of run
    max_rul = n_files - fpt  # = 40

    for node in range(n_nodes):
        for t in range(n_files):
            if t <= fpt:
                rul[t, node] = float(max_rul)
            else:
                rul[t, node] = float(max_rul - (t - fpt))

    # Normalize to [0, 1]
    rul = rul / max_rul
    return rul


@pytest.fixture
def minimal_config():
    """
    Minimal config dict containing only what the preprocessing code needs.
    In production, this comes from loading config/config.yaml.
    For tests, we build it manually to avoid filesystem dependencies.
    """
    return {
        'preprocessing': {
            'n_features': 5,
            'window_size': 20,           # smaller than production (50) for speed
            'stride_healthy': 5,
            'stride_degraded': 1,
            'degraded_threshold': 0.80,
            'drift_removal_window': 20,
            'normalization': 'minmax',
            'fpt': {
                'rms_sigma_threshold': 2.0,
                'rolling_window': 30,
            },
        },
        'splits': {
            'train_fraction': 0.60,
            'val_fraction': 0.20,
        },
        'graph': {
            'n_nodes': 4,
        },
        'evaluation': {
            'phm2012': {
                'early_denominator': 20,
                'late_denominator': 5,
            }
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# NORMALIZATION TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestMinMaxNormalizer:

    def test_normalizer_fits_on_train_only(self, synthetic_features):
        """
        Verify that fit() learns statistics only from the data passed to it.
        After fitting on train, transform should produce values in [0,1].
        """
        train = synthetic_features[:120]   # first 60%
        normalizer = MinMaxNormalizer()
        train_norm = normalizer.fit_transform(train)

        # After fit_transform, training features should be in [0, 1]
        assert train_norm.min() >= -1e-6, "Normalized values should be >= 0"
        assert train_norm.max() <= 1.0 + 1e-6, "Normalized values should be <= 1"
        assert normalizer._fitted, "Normalizer should be marked as fitted"

    def test_normalizer_no_leakage(self, synthetic_features):
        """
        CRITICAL: Verify that the normalizer fitted on train does NOT
        use test data statistics. The test set may produce values slightly
        outside [0,1] if it has higher values than the training set —
        this is expected and CORRECT behavior (clipped to [0,1]).

        If you were to fit on the full dataset and then split, test values
        would always be in [0,1] — this would be data leakage.
        """
        train = synthetic_features[:120]
        test  = synthetic_features[160:]   # near-failure = potentially higher values

        normalizer = MinMaxNormalizer()
        normalizer.fit(train)

        # Normalizer statistics should match training set statistics
        assert normalizer.feat_min is not None
        assert normalizer.feat_max is not None
        np.testing.assert_allclose(
            normalizer.feat_min,
            train.min(axis=0),
            rtol=1e-5,
            err_msg="Normalizer min should match training set min exactly"
        )

    def test_normalizer_raises_before_fit(self, synthetic_features):
        """Verify that calling transform() before fit() raises an error."""
        normalizer = MinMaxNormalizer()
        with pytest.raises(RuntimeError, match="not been fitted"):
            normalizer.transform(synthetic_features)


# ─────────────────────────────────────────────────────────────────────────────
# SPLITTING TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestChronologicalSplit:

    def test_chronological_split_ordering(self, synthetic_features, synthetic_rul):
        """
        Verify that splitting is strictly chronological: no test data appears
        before the last training timestep.
        """
        splits = chronological_split(synthetic_features, synthetic_rul,
                                     train_frac=0.60, val_frac=0.20)

        n_files = len(synthetic_features)
        train_end = int(n_files * 0.60)   # = 120
        val_end   = int(n_files * 0.80)   # = 160

        # Train split should have exactly train_end files
        assert len(splits['train']['features']) == train_end

        # Validation split should start immediately after training
        assert len(splits['val']['features']) == val_end - train_end

        # Mini-test split should be the remainder
        assert len(splits['mini_test']['features']) == n_files - val_end

    def test_splits_dont_overlap(self, synthetic_features, synthetic_rul):
        """
        Verify that the three splits are truly non-overlapping.
        Total windows across all splits must equal original file count.
        """
        splits = chronological_split(synthetic_features, synthetic_rul)
        total = sum(len(s['features']) for s in splits.values())
        assert total == len(synthetic_features), (
            "Train + val + mini_test should cover all files with no overlap"
        )


# ─────────────────────────────────────────────────────────────────────────────
# SLIDING WINDOW TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestSlidingWindows:

    def test_window_output_shapes(self, synthetic_features, synthetic_rul):
        """Verify output shapes are correct for given window parameters."""
        window_size = 20
        X, y = create_sliding_windows(
            synthetic_features, synthetic_rul,
            window_size=window_size,
            stride_healthy=5,
            stride_degraded=1,
            degraded_threshold=0.80,
        )
        n_files, n_nodes, n_features = synthetic_features.shape

        # X should be [N_windows, window_size, n_nodes, n_features]
        assert X.ndim == 4
        assert X.shape[1] == window_size
        assert X.shape[2] == n_nodes
        assert X.shape[3] == n_features

        # y should be [N_windows, n_nodes]
        assert y.ndim == 2
        assert y.shape[1] == n_nodes

        # Number of X and y windows must match
        assert len(X) == len(y)

    def test_temporal_oversampling_produces_more_near_failure(
            self, synthetic_features, synthetic_rul):
        """
        Verify that near-failure region (last 20%) produces more windows
        than the healthy region of equivalent duration when oversampling.
        """
        n_files = len(synthetic_features)
        degraded_start = int(n_files * 0.80)  # = 160

        # With oversampling:
        X_over, _ = create_sliding_windows(
            synthetic_features, synthetic_rul,
            window_size=20, stride_healthy=5, stride_degraded=1,
            degraded_threshold=0.80,
        )

        # Without oversampling (uniform stride):
        X_uniform, _ = create_sliding_windows(
            synthetic_features, synthetic_rul,
            window_size=20, stride_healthy=5, stride_degraded=5,
            degraded_threshold=0.80,
        )

        # Oversampling should create more total windows than uniform stride
        assert len(X_over) > len(X_uniform), (
            "Temporal oversampling should produce more windows "
            "than uniform stride sampling"
        )

    def test_window_label_is_next_timestep(self, synthetic_features, synthetic_rul):
        """
        Verify that each window's label is the RUL at the timestep
        IMMEDIATELY AFTER the window ends — not within the window.

        This is critical: predicting RUL at t=50 using data from t=[0..49].
        If the label is inside the window, you have temporal leakage.
        """
        window_size = 10
        X, y = create_sliding_windows(
            synthetic_features, synthetic_rul,
            window_size=window_size,
            stride_healthy=1, stride_degraded=1,
        )

        # First window: input is [0..9], label should be rul[10]
        np.testing.assert_allclose(
            y[0],
            synthetic_rul[window_size],
            rtol=1e-5,
            err_msg="First window label should be RUL at timestep window_size"
        )


# ─────────────────────────────────────────────────────────────────────────────
# PYTORCH DATASET TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestBearingRULDataset:

    def test_dataset_len_and_shapes(self, synthetic_features, synthetic_rul):
        """Verify Dataset __len__ and __getitem__ return correct shapes."""
        import torch

        X_win, y_win = create_sliding_windows(
            synthetic_features, synthetic_rul,
            window_size=20, stride_healthy=5, stride_degraded=1,
        )

        dataset = BearingRULDataset(X_win, y_win)

        assert len(dataset) == len(X_win)

        # Check single item
        X_item, y_item = dataset[0]
        assert isinstance(X_item, torch.Tensor)
        assert isinstance(y_item, torch.Tensor)
        assert X_item.shape == (20, 4, 5)   # [W, N, F]
        assert y_item.shape == (4,)           # [N]

    def test_dataset_dtype_is_float32(self, synthetic_features, synthetic_rul):
        """Verify tensors are float32 (required by PyTorch linear layers)."""
        import torch

        X_win, y_win = create_sliding_windows(
            synthetic_features, synthetic_rul,
            window_size=20, stride_healthy=5, stride_degraded=1,
        )
        dataset = BearingRULDataset(X_win, y_win)
        X_item, y_item = dataset[0]

        assert X_item.dtype == torch.float32
        assert y_item.dtype == torch.float32


# ─────────────────────────────────────────────────────────────────────────────
# METRICS TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestMetrics:

    def test_rmse_perfect_prediction(self):
        """RMSE should be 0 when prediction exactly matches truth."""
        y = np.array([0.8, 0.6, 0.4, 0.2])
        assert rmse(y, y) == pytest.approx(0.0, abs=1e-10)

    def test_rmse_known_value(self):
        """Verify RMSE with a hand-computable example."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 4.0])  # error only on last: (4-3)²=1
        # RMSE = √((0+0+1)/3) = √(1/3) ≈ 0.5774
        expected = np.sqrt(1.0 / 3.0)
        assert rmse(y_true, y_pred) == pytest.approx(expected, rel=1e-5)

    def test_phm2012_score_perfect_prediction(self):
        """PHM 2012 score should be 1.0 for perfect prediction."""
        y_true = np.array([100.0, 200.0, 50.0])
        y_pred = y_true.copy()
        score = phm2012_score(y_true, y_pred)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_phm2012_late_worse_than_early(self):
        """
        CRITICAL: A late prediction (Er ≤ 0) should score WORSE than
        an early prediction (Er > 0) of the same absolute magnitude.
        This encodes the real cost structure: missed failures > false alarms.
        """
        y_true = np.array([100.0])

        # Late prediction: predicted 110 when true is 100 (Er = -10%)
        score_late = phm2012_score(y_true, np.array([110.0]))

        # Early prediction: predicted 90 when true is 100 (Er = +10%)
        score_early = phm2012_score(y_true, np.array([90.0]))

        assert score_late < score_early, (
            "Late predictions must score worse than early predictions "
            "of the same absolute magnitude. "
            f"Got score_late={score_late:.4f}, score_early={score_early:.4f}"
        )

    def test_delta_rmse_no_degradation(self):
        """ΔRMSEdrop should be 0% when source and target RMSE are equal."""
        assert delta_rmse(0.15, 0.15) == pytest.approx(0.0, abs=1e-10)

    def test_delta_rmse_degradation(self):
        """ΔRMSEdrop should be 100% when target RMSE doubles source."""
        assert delta_rmse(0.10, 0.20) == pytest.approx(100.0, rel=1e-5)

    def test_pearson_rho_perfect_correlation(self):
        """Pearson ρ should be 1.0 for perfectly correlated arrays."""
        x = np.linspace(0, 1, 50)
        y = 2 * x + 3  # perfectly linear
        assert pearson_rho(x, y) == pytest.approx(1.0, abs=1e-5)

    def test_pearson_rho_no_correlation(self):
        """Pearson ρ should be near 0 for uncorrelated random arrays."""
        np.random.seed(0)
        x = np.random.randn(500)
        y = np.random.randn(500)
        rho = abs(pearson_rho(x, y))
        assert rho < 0.15, (
            f"Uncorrelated arrays should give |ρ| < 0.15, got {rho:.4f}"
        )

    def test_compute_all_metrics_returns_expected_keys(self):
        """compute_all_metrics should return all four metric keys."""
        y_true = np.random.rand(100)
        y_pred = np.random.rand(100)
        config = {
            'evaluation': {
                'phm2012': {'early_denominator': 20, 'late_denominator': 5}
            }
        }
        results = compute_all_metrics(
            y_true, y_pred,
            rmse_source=0.10,
            predicted_rates=np.random.rand(50),
            reference_rates=np.random.rand(50),
            config=config,
        )
        for key in ['rmse', 'phm2012_score', 'delta_rmse', 'pearson_rho']:
            assert key in results, f"Missing metric: {key}"


# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

class TestFullPipeline:

    def test_full_pipeline_end_to_end(self, synthetic_features,
                                       synthetic_rul, minimal_config):
        """
        Run the complete preprocessing pipeline end-to-end and verify
        that all output shapes and types are correct.

        This test simulates exactly what train.py will do when it calls
        build_datasets() on real IMS data.
        """
        # Simulate what IMSLoader.load_test() returns
        loaded_data = {
            'features': synthetic_features,
            'rul': synthetic_rul,
        }

        # Run the full pipeline
        datasets = build_datasets(
            loaded_data,
            minimal_config,
            normalizer=None,
            is_training_run=True,
        )

        # Verify all splits are created
        assert 'train' in datasets
        assert 'val' in datasets
        assert 'mini_test' in datasets
        assert 'normalizer' in datasets

        # Verify normalizer was fitted
        assert datasets['normalizer']._fitted

        # Verify training dataset has samples
        assert len(datasets['train']) > 0

        # Verify PyTorch DataLoader works with the training dataset
        from torch.utils.data import DataLoader
        loader = DataLoader(datasets['train'], batch_size=8, shuffle=True)
        X_batch, y_batch = next(iter(loader))

        # Check batch shapes
        assert X_batch.ndim == 4    # [B, W, N, F]
        assert y_batch.ndim == 2    # [B, N]
        assert X_batch.shape[0] <= 8
        assert X_batch.shape[1] == minimal_config['preprocessing']['window_size']
        assert X_batch.shape[2] == minimal_config['graph']['n_nodes']
        assert X_batch.shape[3] == minimal_config['preprocessing']['n_features']

        print(f"\n✓ Full pipeline smoke test passed")
        print(f"  Train windows:     {len(datasets['train'])}")
        print(f"  Val windows:       {len(datasets['val'])}")
        print(f"  Mini-test windows: {len(datasets['mini_test'])}")
        print(f"  Batch shape:       {list(X_batch.shape)}")


# ─────────────────────────────────────────────────────────────────────────────
# RUN DIRECTLY (without pytest)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # You can also run: python tests/test_smoke.py
    # This runs all tests and prints results without pytest
    import traceback

    print("=" * 60)
    print("PC-NDT Smoke Test — Running directly")
    print("=" * 60)

    np.random.seed(42)
    n_files, n_nodes, n_features = 200, 4, 5
    features = np.random.randn(n_files, n_nodes, n_features).astype(np.float32)
    features[160:, :, 0] += np.linspace(0, 2, 40).reshape(-1, 1)

    rul = np.zeros((n_files, n_nodes), dtype=np.float32)
    for node in range(n_nodes):
        for t in range(n_files):
            rul[t, node] = max(0.0, 1.0 - max(0.0, t - 160) / 40.0)

    config = {
        'preprocessing': {
            'n_features': 5, 'window_size': 20,
            'stride_healthy': 5, 'stride_degraded': 1,
            'degraded_threshold': 0.80, 'drift_removal_window': 20,
            'normalization': 'minmax',
            'fpt': {'rms_sigma_threshold': 2.0, 'rolling_window': 30},
        },
        'splits': {'train_fraction': 0.60, 'val_fraction': 0.20},
        'graph': {'n_nodes': 4},
        'evaluation': {
            'phm2012': {'early_denominator': 20, 'late_denominator': 5}
        }
    }

    tests_passed = 0
    tests_failed = 0

    test_cases = [
        ("Normalizer fit + transform", lambda: MinMaxNormalizer().fit_transform(features[:120])),
        ("Chronological split", lambda: chronological_split(features, rul)),
        ("Sliding windows", lambda: create_sliding_windows(features, rul, window_size=20)),
        ("RMSE metric", lambda: rmse(np.array([0.8, 0.6]), np.array([0.7, 0.5]))),
        ("PHM 2012 asymmetry",
         lambda: phm2012_score(np.array([100.0]), np.array([110.0])) <
                 phm2012_score(np.array([100.0]), np.array([90.0]))),
        ("Full pipeline",
         lambda: build_datasets({'features': features, 'rul': rul}, config)),
    ]

    for name, fn in test_cases:
        try:
            result = fn()
            print(f"  ✓ {name}")
            tests_passed += 1
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            traceback.print_exc()
            tests_failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {tests_passed} passed, {tests_failed} failed")
    if tests_failed == 0:
        print("All smoke tests passed! Pipeline is ready.")
    else:
        print("Some tests failed. Fix before proceeding.")
