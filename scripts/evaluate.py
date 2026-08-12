"""
evaluate.py — Held-Out Evaluation for PC-NDT
═══════════════════════════════════════════════════════════════════════
Loads the best saved checkpoint and evaluates on IMS Test 3
(the held-out set never seen during training).

Produces:
  1. Console: clean metrics table (RMSE, PHM2012 score)
  2. results/rul_curves.png       — true vs predicted RUL per bearing
  3. results/adjacency_heatmap.png — learned sensor graph visualised
  4. results/pds_trend.png        — Physics Disagreement Score over time
  5. results/training_history.png — val_rmse curve from training
  6. results/evaluation_results.json — all numbers in one file

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
═══════════════════════════════════════════════════════════════════════
"""

import sys, os, argparse, logging, yaml, json
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.ims_loader      import IMSLoader
from src.data.preprocessing   import (MinMaxNormalizer, create_sliding_windows,
                                       BearingRULDataset)
from src.models.pc_ndt        import PCNDT
from src.physics.constraints  import PhysicsConstraints
from src.evaluation.metrics   import rmse, phm2012_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# ── Plot styling ────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi':       150,
    'font.family':      'DejaVu Sans',
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        True,
    'grid.alpha':       0.3,
})
COLORS = ['#1F4E8C', '#2E75B6', '#70AD47', '#FF6B35']
BEARING_NAMES = ['Bearing 1', 'Bearing 2', 'Bearing 3', 'Bearing 4']


# ═══════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_and_normalize_test3(config: dict):
    """
    Load IMS Test 3 with the SAME normalization as training.

    WHY WE REFIT NORMALIZER ON TEST 1:
        The normalizer was fitted on IMS Test 1 during training.
        We never saved it to disk (a fix for the next iteration).
        Since MinMaxNormalizer is deterministic — same input always
        gives same output — we can refit it on Test 1 right now and
        get the EXACT same scaling factors used during training.
        No data leakage: Test 3 is only transformed, never fitted on.
    """
    loader = IMSLoader(config)

    # Step 1: Load Test 1 to refit the normalizer
    logger.info("Loading IMS Test 1 to refit normalizer (deterministic)...")
    data_test1 = loader.load_test(test_id=1)
    features_t1 = data_test1['features']

    # Fit ONLY on the training fraction of Test 1 (same as training script)
    train_frac = config['splits']['train_fraction']
    train_end  = int(len(features_t1) * train_frac)
    normalizer = MinMaxNormalizer()
    normalizer.fit(features_t1[:train_end])
    logger.info(f"  Normalizer fitted on {train_end} files from Test 1")

    # Step 2: Load Test 3 (held-out) and apply the training normalizer
    logger.info("Loading IMS Test 3 (held-out evaluation set)...")
    data_test3 = loader.load_test(test_id=3)
    features_t3 = normalizer.transform(data_test3['features'])
    rul_t3      = data_test3['rul']

    logger.info(f"  Test 3: {data_test3['n_files']} files | "
                f"FPT indices: {data_test3['fpt_idx']}")

    return features_t3, rul_t3, data_test3, normalizer


def build_test3_loader(features: np.ndarray,
                       rul: np.ndarray,
                       config: dict,
                       batch_size: int = 32) -> DataLoader:
    """Create a DataLoader for Test 3 — no shuffling, stride=1 everywhere."""
    cfg = config['preprocessing']
    X, y = create_sliding_windows(
        features, rul,
        window_size     = cfg['window_size'],
        stride_healthy  = 1,    # evaluate EVERY timestep, no skipping
        stride_degraded = 1,
        degraded_threshold = cfg['degraded_threshold'],
    )
    dataset = BearingRULDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


# ═══════════════════════════════════════════════════════════════════════
# SECTION 2 — INFERENCE
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_inference(model: PCNDT,
                  loader: DataLoader,
                  device: str) -> dict:
    """
    Run the full forward pass on every batch in the loader.

    Returns arrays of:
        predictions  [N_windows, N_nodes]
        targets      [N_windows, N_nodes]
        pds          [N_windows, N_nodes]  — Physics Disagreement Score
        adjacency    [N_nodes,   N_nodes]  — learned graph (from last batch)
    """
    model.eval()
    all_preds  = []
    all_targets= []
    all_pds    = []
    A_final    = None

    for X_batch, y_batch in tqdm(loader, desc="  Running inference"):
        X_batch = X_batch.to(device)
        out = model(X_batch)

        all_preds.append(out['rul'].cpu().numpy())
        all_targets.append(y_batch.numpy())
        all_pds.append(out['pds'].cpu().numpy())
        A_final = out['adjacency'].cpu().numpy()

    return {
        'predictions': np.concatenate(all_preds,   axis=0),  # [N_win, N]
        'targets':     np.concatenate(all_targets, axis=0),  # [N_win, N]
        'pds':         np.concatenate(all_pds,     axis=0),  # [N_win, N]
        'adjacency':   A_final,                               # [N, N]
    }


# ═══════════════════════════════════════════════════════════════════════
# SECTION 3 — METRICS
# ═══════════════════════════════════════════════════════════════════════

def compute_metrics(results: dict, config: dict) -> dict:
    """
    Compute all evaluation metrics across all bearings.

    Reports per-bearing AND aggregated numbers so you can see
    which bearing the model handles best and worst.
    """
    preds   = results['predictions']   # [N_win, N_nodes]
    targets = results['targets']       # [N_win, N_nodes]
    n_nodes = preds.shape[1]

    cfg_phm = config['evaluation']['phm2012']
    early_d = cfg_phm['early_denominator']
    late_d  = cfg_phm['late_denominator']

    per_bearing = {}
    for node_idx in range(n_nodes):
        p = preds[:, node_idx]
        t = targets[:, node_idx]
        per_bearing[BEARING_NAMES[node_idx]] = {
            'rmse':         round(rmse(t, p), 6),
            'phm2012_score':round(phm2012_score(t, p, early_d, late_d), 6),
        }

    # Aggregate across all nodes
    agg_rmse  = rmse(targets.ravel(), preds.ravel())
    agg_score = phm2012_score(targets.ravel(), preds.ravel(), early_d, late_d)

    return {
        'per_bearing':   per_bearing,
        'aggregate': {
            'rmse':          round(agg_rmse,  6),
            'phm2012_score': round(agg_score, 6),
        }
    }


# ═══════════════════════════════════════════════════════════════════════
# SECTION 4 — VISUALISATION
# ═══════════════════════════════════════════════════════════════════════

def plot_rul_curves(results: dict,
                    data_test3: dict,
                    save_path: str):
    """
    Plot true vs. predicted RUL for each bearing.

    This is the most visually important plot in your paper.
    The x-axis is time (file index = operational cycle).
    The y-axis is normalised RUL in [0, 1].
    A good model: predicted curve tracks the true curve closely,
    especially in the final 20% where failure is imminent.
    """
    preds   = results['predictions']   # [N_win, N]
    targets = results['targets']       # [N_win, N]
    n_nodes = preds.shape[1]
    window  = 50                       # window_size from config

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()
    fig.suptitle('PC-NDT: True vs Predicted RUL — IMS Test 3 (Held-Out)',
                 fontsize=13, fontweight='bold', y=1.01)

    for node_idx in range(n_nodes):
        ax   = axes[node_idx]
        p    = preds[:, node_idx]
        t    = targets[:, node_idx]
        time = np.arange(len(p)) + window    # offset by window size

        ax.plot(time, t, color='#1F4E8C', linewidth=2,
                label='True RUL', alpha=0.9)
        ax.plot(time, p, color='#FF6B35', linewidth=1.5,
                linestyle='--', label='Predicted RUL', alpha=0.85)

        # Mark FPT (First Prediction Time) if available
        fpt = data_test3['fpt_idx'][node_idx] if node_idx < len(data_test3['fpt_idx']) else None
        if fpt and fpt < len(p) + window:
            ax.axvline(x=fpt, color='#70AD47', linestyle=':',
                       linewidth=1.5, label=f'FPT (t={fpt})', alpha=0.8)

        node_rmse  = rmse(t, p)
        ax.set_title(f'{BEARING_NAMES[node_idx]} | RMSE = {node_rmse:.4f}',
                     fontsize=10)
        ax.set_xlabel('File index (operational cycle)', fontsize=9)
        ax.set_ylabel('Normalised RUL', fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {save_path}")


def plot_adjacency_heatmap(adjacency: np.ndarray, save_path: str):
    """
    Visualise the learned sensor dependency graph as a heatmap.

    WHAT TO LOOK FOR (your RQ1 answer):
        Diagonal blocks should be lighter (self-similarity not plotted).
        Bearings 1↔2 and 3↔4 should have stronger edges (adjacent on shaft).
        Bearings 1↔4 should have weaker edges (opposite ends of shaft).
        If the learned graph shows this structure, AGCRN has correctly
        discovered the physical coupling without being told.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    mask = np.eye(adjacency.shape[0], dtype=bool)   # mask diagonal
    sns.heatmap(
        adjacency,
        annot=True,
        fmt='.3f',
        cmap='Blues',
        mask=mask,
        linewidths=0.5,
        linecolor='white',
        ax=ax,
        vmin=0, vmax=adjacency.max(),
        annot_kws={'size': 11, 'weight': 'bold'},
    )

    ax.set_title('Learned Sensor Dependency Graph (AGCRN Adjacency A)',
                 fontsize=11, fontweight='bold', pad=12)
    ax.set_xticklabels(BEARING_NAMES, rotation=30, ha='right', fontsize=9)
    ax.set_yticklabels(BEARING_NAMES, rotation=0, fontsize=9)
    ax.set_xlabel('Target Node (influence receiver)', fontsize=9)
    ax.set_ylabel('Source Node (influence sender)', fontsize=9)

    # Add interpretation note
    fig.text(0.5, -0.04,
             'Higher value = stronger learned coupling between sensor pair.\n'
             'Compare against physical prior: adjacent bearings should couple strongly.',
             ha='center', fontsize=8, color='#555555', style='italic')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {save_path}")


def plot_pds_trend(results: dict, save_path: str):
    """
    Plot the Physics Disagreement Score (PDS) over time for each bearing.

    This is your deployment-time interpretability signal — the thing
    a maintenance engineer actually looks at on the dashboard.

    WHAT TO LOOK FOR:
        PDS should be low and stable during healthy operation.
        PDS should RISE sharply as the bearing enters the failure regime.
        A rising PDS on ONE bearing = that bearing is degrading.
        Rising PDS on ALL bearings = possible sensor drift (investigate).
    """
    pds     = results['pds']     # [N_win, N]
    n_nodes = pds.shape[1]
    window  = 50

    fig, ax = plt.subplots(figsize=(12, 5))

    for node_idx in range(n_nodes):
        # Smooth PDS with a rolling mean for readability
        pds_node   = pds[:, node_idx]
        smooth_win = max(1, len(pds_node) // 50)
        pds_smooth = np.convolve(
            pds_node,
            np.ones(smooth_win) / smooth_win,
            mode='valid'
        )
        time = np.arange(len(pds_smooth)) + window

        ax.plot(time, pds_smooth,
                color=COLORS[node_idx],
                linewidth=1.8,
                label=BEARING_NAMES[node_idx],
                alpha=0.85)

    ax.set_title(
        'Physics Disagreement Score (PDS) Over Time\n'
        'Rising PDS = model predicts degradation rate diverging from Archard\'s Law',
        fontsize=10, fontweight='bold'
    )
    ax.set_xlabel('File index (operational cycle)', fontsize=9)
    ax.set_ylabel('Physics Disagreement Score', fontsize=9)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {save_path}")


def plot_training_history(history: list, save_path: str):
    """
    Plot the val_rmse curve from training history stored in the checkpoint.

    This shows your mentors that training was stable (smooth decrease)
    and that early stopping fired at the right point.
    """
    epochs   = [h['epoch']    for h in history]
    val_rmse = [h['val_rmse'] for h in history]
    train_loss=[h['total']    for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle('Training History', fontsize=12, fontweight='bold')

    ax1.plot(epochs, val_rmse, color='#1F4E8C', linewidth=2)
    best_epoch = epochs[int(np.argmin(val_rmse))]
    best_rmse  = min(val_rmse)
    ax1.axvline(x=best_epoch, color='#FF6B35', linestyle='--',
                linewidth=1.5, label=f'Best (epoch {best_epoch})')
    ax1.scatter([best_epoch], [best_rmse], color='#FF6B35', s=60, zorder=5)
    ax1.set_title(f'Validation RMSE | Best = {best_rmse:.5f}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('RMSE')
    ax1.legend(fontsize=9)

    ax2.plot(epochs, train_loss, color='#70AD47', linewidth=2)
    ax2.set_title('Total Training Loss (pred + physics)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════════
# SECTION 5 — MAIN
# ═══════════════════════════════════════════════════════════════════════

def print_results_table(metrics: dict, checkpoint_info: dict):
    """Print a clean, readable results summary to the console."""
    print()
    print("=" * 60)
    print("  PC-NDT EVALUATION RESULTS — IMS Test 3 (Held-Out)")
    print("=" * 60)
    print(f"  Checkpoint: epoch {checkpoint_info['epoch']} | "
          f"val_rmse = {checkpoint_info['val_rmse']:.5f}")
    print()
    print(f"  {'Bearing':<15} {'RMSE':>10} {'PHM2012 Score':>15}")
    print(f"  {'-'*40}")

    for name, vals in metrics['per_bearing'].items():
        print(f"  {name:<15} {vals['rmse']:>10.5f} {vals['phm2012_score']:>15.5f}")

    print(f"  {'-'*40}")
    agg = metrics['aggregate']
    print(f"  {'AGGREGATE':<15} {agg['rmse']:>10.5f} {agg['phm2012_score']:>15.5f}")
    print()
    print("  Interpretation:")
    print(f"    RMSE: lower is better | PHM2012 Score: higher is better (max=1.0)")
    print(f"    PHM2012 penalises late predictions more than early ones")
    print("=" * 60)
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        default='config/config.yaml')
    parser.add_argument('--checkpoint',
                        default='checkpoints/best_model.pt')
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────
    with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")

    # ── Create results directory ─────────────────────────────────────
    os.makedirs('results', exist_ok=True)

    # ── Load checkpoint ──────────────────────────────────────────────
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found at '{args.checkpoint}'.\n"
            f"Run 'python scripts/train.py' first to generate one."
        )

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    checkpoint_info = {
        'epoch':    ckpt['epoch'],
        'val_rmse': ckpt['val_rmse'],
    }
    logger.info(f"  Saved at epoch {ckpt['epoch']} | val_rmse = {ckpt['val_rmse']:.5f}")

    # ── Load Test 3 data ─────────────────────────────────────────────
    features_t3, rul_t3, data_test3, normalizer = load_and_normalize_test3(config)
    test3_loader = build_test3_loader(features_t3, rul_t3, config)
    logger.info(f"Test 3 loader: {len(test3_loader)} batches")

    # ── Load model ───────────────────────────────────────────────────
    model = PCNDT(config).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    logger.info("Model loaded from checkpoint")

    # ── Run inference ────────────────────────────────────────────────
    logger.info("Running inference on IMS Test 3...")
    results = run_inference(model, test3_loader, device)

    # ── Compute metrics ──────────────────────────────────────────────
    metrics = compute_metrics(results, config)

    # ── Print results table ──────────────────────────────────────────
    print_results_table(metrics, checkpoint_info)

    # ── Generate all plots ───────────────────────────────────────────
    logger.info("Generating plots...")

    plot_rul_curves(
        results, data_test3,
        save_path='results/rul_curves.png'
    )
    plot_adjacency_heatmap(
        results['adjacency'],
        save_path='results/adjacency_heatmap.png'
    )
    plot_pds_trend(
        results,
        save_path='results/pds_trend.png'
    )
    if 'history' in ckpt and ckpt['history']:
        plot_training_history(
            ckpt['history'],
            save_path='results/training_history.png'
        )
    else:
        logger.warning("No training history in checkpoint — skipping history plot")

    # ── Save JSON results ─────────────────────────────────────────────
    json_output = {
        'checkpoint':     checkpoint_info,
        'dataset':        'NASA IMS Test 3 (held-out)',
        'metrics':        metrics,
        'notes': {
            'normalization':  'fitted on IMS Test 1 training split (deterministic)',
            'evaluation_set': 'IMS Test 3 — never seen during training',
            'model':          'PC-NDT (AGCRN + Neural ODE + Physics Constraints)',
        }
    }
    json_path = 'results/evaluation_results.json'
    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=2)
    logger.info(f"  Saved: {json_path}")

    print(f"All outputs saved to results/")
    print(f"  results/rul_curves.png")
    print(f"  results/adjacency_heatmap.png")
    print(f"  results/pds_trend.png")
    print(f"  results/training_history.png")
    print(f"  results/evaluation_results.json")
    print()
    print("Next step: compare these numbers against your baseline models")
    print("(LSTM, GRU, AGCRN-only) to build Table 1 of your research paper.")


if __name__ == '__main__':
    main()
