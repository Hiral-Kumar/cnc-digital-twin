"""
baselines.py — Train LSTM + GRU Baselines and Build Table 1

Trains simple baselines on the SAME data as PC-NDT (IMS Test 1 → Test 3),
so the comparison is fair and apples-to-apples.

Usage:
    python scripts/baselines.py
"""

import sys, os, logging, yaml, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.ims_loader     import IMSLoader
from src.data.preprocessing  import build_datasets, MinMaxNormalizer, create_sliding_windows, BearingRULDataset
from src.baselines.lstm_gru  import LSTMBaseline, GRUBaseline
from src.evaluation.metrics  import rmse, phm2012_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


def train_baseline(model, train_loader, val_loader, epochs, device, name):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    best_val_rmse = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out['rul'], y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validation
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                out = model(X)
                preds.append(out['rul'].cpu().numpy())
                targets.append(y.numpy())
        val_rmse = rmse(np.concatenate(targets), np.concatenate(preds))

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == epochs - 1:
            logger.info(f"  [{name}] Epoch {epoch:03d} | val_rmse={val_rmse:.5f}")

    model.load_state_dict(best_state)
    return model, best_val_rmse


@torch.no_grad()
def evaluate_on_test(model, test_loader, device, config):
    model.eval()
    preds, targets = [], []
    for X, y in test_loader:
        X = X.to(device)
        out = model(X)
        preds.append(out['rul'].cpu().numpy())
        targets.append(y.numpy())
    preds   = np.concatenate(preds)
    targets = np.concatenate(targets)

    cfg = config['evaluation']['phm2012']
    return {
        'rmse':          round(rmse(targets, preds), 6),
        'phm2012_score': round(phm2012_score(targets, preds,
                                              cfg['early_denominator'],
                                              cfg['late_denominator']), 6),
    }


def main():
    with open('config/config.yaml', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # ── Load data (same as PC-NDT training) ──────────────────────────
    loader = IMSLoader(config)
    logger.info("Loading IMS Test 1 (training)...")
    data_t1 = loader.load_test(1)
    datasets = build_datasets(data_t1, config, is_training_run=True)
    normalizer = datasets['normalizer']

    logger.info("Loading IMS Test 3 (held-out test)...")
    data_t3 = loader.load_test(3)
    features_t3 = normalizer.transform(data_t3['features'])

    cfg_prep = config['preprocessing']
    X_test, y_test = create_sliding_windows(
        features_t3, data_t3['rul'],
        window_size=cfg_prep['window_size'],
        stride_healthy=1, stride_degraded=1,
    )
    test_dataset = BearingRULDataset(X_test, y_test)

    batch_size = config['training']['batch_size']
    train_loader = DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(datasets['val'],   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,      batch_size=batch_size, shuffle=False)

    # ── Train baselines ───────────────────────────────────────────────
    results = {}
    epochs = 50   # baselines converge faster than PC-NDT — fewer epochs needed

    logger.info("=" * 50)
    logger.info("Training LSTM baseline...")
    lstm = LSTMBaseline(config).to(device)
    lstm, lstm_val_rmse = train_baseline(lstm, train_loader, val_loader, epochs, device, "LSTM")
    results['LSTM'] = evaluate_on_test(lstm, test_loader, device, config)
    results['LSTM']['val_rmse'] = round(lstm_val_rmse, 6)

    logger.info("=" * 50)
    logger.info("Training GRU baseline...")
    gru = GRUBaseline(config).to(device)
    gru, gru_val_rmse = train_baseline(gru, train_loader, val_loader, epochs, device, "GRU")
    results['GRU'] = evaluate_on_test(gru, test_loader, device, config)
    results['GRU']['val_rmse'] = round(gru_val_rmse, 6)

    # ── Load PC-NDT results if already evaluated ──────────────────────
    pcndt_path = 'results/evaluation_results.json'
    if os.path.exists(pcndt_path):
        with open(pcndt_path) as f:
            pcndt_results = json.load(f)
        results['PC-NDT (full)'] = pcndt_results['metrics']['aggregate']
        logger.info("Loaded existing PC-NDT results for comparison")

    # ── Print Table 1 ──────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  TABLE 1 — Model Comparison on IMS Test 3 (Held-Out)")
    print("=" * 60)
    print(f"  {'Model':<20} {'RMSE':>10} {'PHM2012 Score':>15}")
    print(f"  {'-'*47}")
    for name, vals in results.items():
        print(f"  {name:<20} {vals['rmse']:>10.5f} {vals['phm2012_score']:>15.5f}")
    print("=" * 60)
    print()

    # ── Save results ────────────────────────────────────────────────────
    os.makedirs('results', exist_ok=True)
    with open('results/baseline_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    logger.info("Saved: results/baseline_comparison.json")


if __name__ == '__main__':
    main()
