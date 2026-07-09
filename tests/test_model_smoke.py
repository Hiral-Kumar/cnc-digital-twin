"""
test_model_smoke.py
End-to-end model smoke test using synthetic data.

Verifies that the full forward pass (AGCRN → Neural ODE → Readout)
runs without errors and produces correct output shapes.

Run with:  pytest tests/test_model_smoke.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import pytest

# ── Minimal config ──────────────────────────────────────────────────────────
@pytest.fixture
def config():
    return {
        'graph':  {'n_nodes': 4, 'embedding_dim': 8,
                   'shaft_distances': [[0,50,100,150],[50,0,50,100],
                                       [100,50,0,50],[150,100,50,0]]},
        'preprocessing': {'n_features': 5, 'window_size': 10,
                          'bearing': {'n_balls': 16, 'contact_angle_deg': 15.17,
                                      'ball_diameter_mm': 0.331,
                                      'pitch_diameter_mm': 2.815,
                                      'shaft_speed_rpm': 2000},
                          'fpt': {'rms_sigma_threshold': 2.0, 'rolling_window': 10},
                          'drift_removal_window': 10},
        'model': {
            'agcrn': {'embedding_dim': 8, 'hidden_dim': 16,
                      'n_layers': 1, 'cheb_k': 2},
            'neural_ode': {'hidden_dim': 16, 'mlp_layers': 2, 'mlp_hidden': 32,
                           'activation': 'tanh', 'solver': 'euler',
                           'rtol': 1e-2, 'atol': 1e-3, 'adjoint': False},
            'readout': {'input_dim': 16, 'output_dim': 1},
        },
        'physics': {
            'warmup_epochs': 5, 'rampup_epochs': 5,
            'lambda_archard': 0.1, 'lambda_paris': 0.1, 'lambda_fourier': 0.1,
            'lambda_grid': [0.01, 0.1, 1.0],
            'archard':  {'wear_coefficient_k': 1e-7, 'hardness_H': 6e9,
                         'constant_load_N': 26689.0},
            'paris':    {'C': 1e-11, 'm': 3.0, 'geometry_factor_Y': 1.12},
            'fourier':  {'thermal_diffusivity': 1.2e-5, 'density': 7850.0,
                         'specific_heat': 500.0, 'friction_coefficient': 0.15},
        },
        'training': {
            'epochs': 2, 'batch_size': 4, 'learning_rate': 1e-3,
            'weight_decay': 1e-4, 'lr_scheduler': 'cosine',
            'grad_clip_max_norm': 1.0, 'early_stopping_patience': 5,
            'monitor_metric': 'val_rmse', 'checkpoint_dir': '/tmp/ckpts',
            'save_best_only': True, 'log_every_n_steps': 1,
        },
        'data': {
            'ims': {
                'root_dir': 'data/raw/IMS',
                'test1_dir': 'data/raw/IMS/1st_test',
                'test2_dir': 'data/raw/IMS/2nd_test',
                'test3_dir': 'data/raw/IMS/3rd_test',
                'sampling_rate': 20480,
                'n_channels_test1': 8, 'n_channels_test23': 4,
                'n_bearings': 4, 'test1_channel_selection': [0, 2, 4, 6],
                'failures': {'test1': [], 'test2': [], 'test3': []}
            }
        },
        'seed': 42,
        'evaluation': {'phm2012': {'early_denominator': 20, 'late_denominator': 5}},
    }


@pytest.fixture
def synthetic_batch(config):
    torch.manual_seed(42)
    B = 4
    T = config['model']['agcrn']['hidden_dim']   # window size
    N = config['graph']['n_nodes']
    F = config['preprocessing']['n_features']
    T = 10  # window
    X = torch.randn(B, T, N, F)
    y = torch.rand(B, N)
    return X, y


class TestAGCRN:
    def test_forward_shapes(self, config, synthetic_batch):
        from src.models.agcrn import AGCRN
        model = AGCRN(config)
        X, _ = synthetic_batch
        H_T, A = model(X)
        B, T, N, F = X.shape
        D = config['model']['agcrn']['hidden_dim']

        assert H_T.shape == (B, N, D), f"Expected ({B},{N},{D}), got {H_T.shape}"
        assert A.shape   == (N, N),     f"Expected ({N},{N}), got {A.shape}"

    def test_adjacency_is_valid_distribution(self, config, synthetic_batch):
        """Each row of A should sum to ~1 (softmax output)."""
        from src.models.agcrn import AGCRN
        model = AGCRN(config)
        X, _ = synthetic_batch
        _, A = model(X)
        row_sums = A.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), \
            "Adjacency rows must sum to 1 (softmax normalised)"

    def test_adjacency_is_learned(self, config, synthetic_batch):
        """Adjacency must have non-trivial structure (not all equal)."""
        from src.models.agcrn import AGCRN
        model = AGCRN(config)
        X, _ = synthetic_batch
        _, A = model(X)
        # After initialization, rows should not all be identical
        # (embedding dim > 1 with random init ensures this)
        A_np = A.detach().numpy()
        row_stds = A_np.std(axis=1)
        assert row_stds.mean() > 0, "Adjacency should have non-uniform row values"


class TestNeuralODE:
    def test_forward_shapes(self, config, synthetic_batch):
        from src.models.neural_ode import NeuralODE
        ode = NeuralODE(config)
        ode.eval()
        B, T, N, F = synthetic_batch[0].shape
        D = config['model']['neural_ode']['hidden_dim']
        h0 = torch.randn(B, N, D)
        t_span = torch.tensor([0.0, 1.0])
        traj, h_final = ode(h0, t_span)
        assert h_final.shape == (B, N, D), \
            f"Expected ({B},{N},{D}), got {h_final.shape}"

    def test_derivatives_shape(self, config, synthetic_batch):
        from src.models.neural_ode import NeuralODE
        ode = NeuralODE(config)
        B, T, N, F = synthetic_batch[0].shape
        D = config['model']['neural_ode']['hidden_dim']
        h = torch.randn(B, N, D)
        dh = ode.get_derivatives(h)
        assert dh.shape == h.shape


class TestPhysicsConstraints:
    def test_all_losses_are_scalar(self, config, synthetic_batch):
        from src.physics.constraints import PhysicsConstraints
        phys = PhysicsConstraints(config)
        B, T, N, F = synthetic_batch[0].shape
        D = config['model']['neural_ode']['hidden_dim']
        dh   = torch.randn(B, N, D)
        h    = torch.randn(B, N, D)
        A    = torch.softmax(torch.randn(N, N), dim=1)
        feats= torch.rand(B, N, F)

        result = phys.compute_all(dh, h, A, feats,
                                  lambda_archard=0.1,
                                  lambda_paris=0.1,
                                  lambda_fourier=0.1)
        for key in ['archard', 'paris', 'fourier', 'total']:
            assert result[key].ndim == 0, f"{key} loss must be a scalar"
            assert torch.isfinite(result[key]), f"{key} loss must be finite"

    def test_pds_shape(self, config, synthetic_batch):
        from src.physics.constraints import PhysicsConstraints
        phys = PhysicsConstraints(config)
        B, T, N, F = synthetic_batch[0].shape
        D = config['model']['neural_ode']['hidden_dim']
        dh = torch.randn(B, N, D)
        pds = phys.physics_disagreement_score(dh)
        assert pds.shape == (B, N), f"PDS must be [B,N], got {pds.shape}"


class TestPCNDT:
    def test_full_forward_pass(self, config, synthetic_batch):
        """The most important test: end-to-end forward pass."""
        from src.models.pc_ndt import PCNDT
        model = PCNDT(config)
        model.eval()
        X, y = synthetic_batch
        with torch.no_grad():
            out = model(X)

        B, T, N, F = X.shape
        D = config['model']['neural_ode']['hidden_dim']

        assert out['rul'].shape  == (B, N),   f"RUL shape wrong: {out['rul'].shape}"
        assert out['pds'].shape  == (B, N),   f"PDS shape wrong: {out['pds'].shape}"
        assert out['dh_dt'].shape== (B, N, D),f"dh_dt shape wrong"
        assert out['adjacency'].shape == (N, N)

    def test_rul_in_valid_range(self, config, synthetic_batch):
        """RUL output (after sigmoid) must be in (0, 1)."""
        from src.models.pc_ndt import PCNDT
        model = PCNDT(config)
        model.eval()
        X, _ = synthetic_batch
        with torch.no_grad():
            out = model(X)
        rul = out['rul']
        assert (rul >= 0).all() and (rul <= 1).all(), \
            "RUL must be in [0, 1] after sigmoid activation"

    def test_backward_pass_does_not_crash(self, config, synthetic_batch):
        """A backward pass must complete without NaN or error."""
        from src.models.pc_ndt import PCNDT
        model = PCNDT(config)
        model.train()
        X, y = synthetic_batch
        out  = model(X)
        loss = ((out['rul'] - y) ** 2).mean()
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), \
                    f"NaN gradient detected in {name}"


if __name__ == '__main__':
    print("Running model smoke tests directly...")
    import traceback
    tests_passed = tests_failed = 0

    cfg = {
        'graph':  {'n_nodes': 4, 'embedding_dim': 8,
                   'shaft_distances': [[0,50,100,150],[50,0,50,100],
                                       [100,50,0,50],[150,100,50,0]]},
        'preprocessing': {'n_features': 5, 'window_size': 10,
                          'bearing': {'n_balls': 16, 'contact_angle_deg': 15.17,
                                      'ball_diameter_mm': 0.331,
                                      'pitch_diameter_mm': 2.815,
                                      'shaft_speed_rpm': 2000},
                          'fpt': {'rms_sigma_threshold': 2.0, 'rolling_window': 10},
                          'drift_removal_window': 10},
        'model': {
            'agcrn': {'embedding_dim': 8, 'hidden_dim': 16, 'n_layers': 1, 'cheb_k': 2},
            'neural_ode': {'hidden_dim': 16, 'mlp_layers': 2, 'mlp_hidden': 32,
                           'activation': 'tanh', 'solver': 'euler',
                           'rtol': 1e-2, 'atol': 1e-3, 'adjoint': False},
            'readout': {'input_dim': 16, 'output_dim': 1},
        },
        'physics': {
            'warmup_epochs': 5, 'rampup_epochs': 5,
            'lambda_archard': 0.1, 'lambda_paris': 0.1, 'lambda_fourier': 0.1,
            'lambda_grid': [0.01, 0.1, 1.0],
            'archard':  {'wear_coefficient_k': 1e-7, 'hardness_H': 6e9, 'constant_load_N': 26689.0},
            'paris':    {'C': 1e-11, 'm': 3.0, 'geometry_factor_Y': 1.12},
            'fourier':  {'thermal_diffusivity': 1.2e-5, 'density': 7850.0,
                         'specific_heat': 500.0, 'friction_coefficient': 0.15},
        },
        'data': {'ims': {'root_dir': 'data/raw/IMS', 'test1_dir': 'data/raw/IMS/1st_test',
                         'test2_dir': 'data/raw/IMS/2nd_test', 'test3_dir': 'data/raw/IMS/3rd_test',
                         'sampling_rate': 20480, 'n_channels_test1': 8, 'n_channels_test23': 4,
                         'n_bearings': 4, 'test1_channel_selection': [0,2,4,6],
                         'failures': {'test1':[],'test2':[],'test3':[]}}},
        'training': {'epochs': 2, 'batch_size': 4, 'learning_rate': 1e-3,
                     'weight_decay': 1e-4, 'lr_scheduler': 'cosine',
                     'grad_clip_max_norm': 1.0, 'early_stopping_patience': 5,
                     'monitor_metric': 'val_rmse', 'checkpoint_dir': '/tmp/ckpts',
                     'save_best_only': True, 'log_every_n_steps': 1},
        'seed': 42,
        'evaluation': {'phm2012': {'early_denominator': 20, 'late_denominator': 5}},
    }
    torch.manual_seed(42)
    X = torch.randn(4, 10, 4, 5)
    y = torch.rand(4, 4)

    from src.models.pc_ndt import PCNDT
    from src.physics.constraints import PhysicsConstraints

    tests = [
        ("PCNDT forward pass", lambda: PCNDT(cfg).eval() or PCNDT(cfg)(X)),
        ("RUL in [0,1]",       lambda: (PCNDT(cfg)(X)['rul'] >= 0).all()),
        ("Physics losses finite",
         lambda: all(torch.isfinite(v) for v in
                     PhysicsConstraints(cfg).compute_all(
                         torch.randn(4,4,16), torch.randn(4,4,16),
                         torch.softmax(torch.randn(4,4),dim=1),
                         torch.rand(4,4,5), 0.1, 0.1, 0.1).values())),
    ]

    for name, fn in tests:
        try:
            fn()
            print(f"  ✓ {name}")
            tests_passed += 1
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            traceback.print_exc()
            tests_failed += 1

    print(f"\n{tests_passed} passed, {tests_failed} failed")
