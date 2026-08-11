"""
evaluate_pronostia.py — Cross-Condition Generalisation Test (RQ3)

Loads the PC-NDT model trained on NASA IMS (2000 RPM) and evaluates it
ZERO-SHOT on PRONOSTIA bearings (1500-1800 RPM, different loads).

This answers RQ3: do physics constraints improve generalisation to
operating conditions never seen during training?

IMPORTANT: PRONOSTIA has N=2 nodes (horizontal + vertical accelerometer
on ONE bearing) vs IMS's N=4 nodes (4 bearings). We handle this by
evaluating PC-NDT's readout per-node independently — the AGCRN graph
structure adapts because it's learned from embeddings sized to N=2
for this evaluation. We use a SEPARATE small AGCRN sized for N=2,
loading only the Neural ODE + physics-trained readout weights that
transfer (the ODE dynamics function f_theta has no dependency on N).

Usage:
    python scripts/evaluate_pronostia.py
"""

import sys, os, argparse, logging, yaml, json
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.pronostia_loader import PronostiaLoader
from src.data.preprocessing    import MinMaxNormalizer, create_sliding_windows, BearingRULDataset
from src.models.pc_ndt         import PCNDT
from src.evaluation.metrics    import rmse, phm2012_score, delta_rmse

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


def build_pronostia_config(base_config: dict) -> dict:
    """
    Create a modified config for PRONOSTIA's N=2 node structure.
    Everything else (hidden dims, physics constants) stays the same —
    only the graph size changes.
    """
    import copy
    cfg = copy.deepcopy(base_config)
    cfg['graph']['n_nodes'] = 2   # PRONOSTIA: horizontal + vertical only
    cfg['graph']['shaft_distances'] = [[0, 10], [10, 0]]  # single bearing, 2 channels
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/config.yaml')
    parser.add_argument('--bearing', default='Training_set/Bearing1_1',
                        help='Which PRONOSTIA bearing folder to evaluate')
    parser.add_argument('--checkpoint', default='checkpoints/best_model.pt',
                        help='Trained PC-NDT checkpoint to evaluate')
    parser.add_argument('--ims-test-rmse', type=float, default=None,
                        help='RMSE on IMS Test 3 (for delta_rmse calc). '
                             'If not provided, reads from results/evaluation_results.json')
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as f:
        base_config = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Load PRONOSTIA bearing data ──────────────────────────────────
    logger.info(f"Loading PRONOSTIA bearing: {args.bearing}")
    pronostia_loader = PronostiaLoader(base_config)

    try:
        data = pronostia_loader.load_bearing(args.bearing)
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.error(
            "\nMake sure PRONOSTIA is downloaded and the path in "
            "config.yaml -> data.pronostia.root_dir is correct.\n"
            "Expected structure: PRONOSTIA/Training_set/Bearing1_1/acc_*.csv"
        )
        return

    logger.info(f"  Condition: {data['condition_id']} "
                f"({pronostia_loader.CONDITIONS[data['condition_id']]})")
    logger.info(f"  Bursts loaded: {data['n_bursts']}")

    # ── Build a fresh normalizer for PRONOSTIA (different sensor scale) ──
    # NOTE: We normalize PRONOSTIA independently since raw g-force scales
    # differ from IMS. What transfers is the LEARNED DYNAMICS (f_theta),
    # not the raw feature statistics.
    cfg_split = base_config['splits']
    n_bursts = data['n_bursts']
    calib_end = int(n_bursts * 0.3)   # use first 30% (healthy) to calibrate scale

    normalizer = MinMaxNormalizer()
    normalizer.fit(data['features'][:calib_end])
    features_norm = normalizer.transform(data['features'])

    # ── Build sliding windows ────────────────────────────────────────
    pronostia_config = build_pronostia_config(base_config)
    cfg_prep = pronostia_config['preprocessing']
    window_size = min(cfg_prep['window_size'], max(5, n_bursts // 4))

    X, y = create_sliding_windows(
        features_norm, data['rul'],
        window_size=window_size,
        stride_healthy=1, stride_degraded=1,
    )
    dataset = BearingRULDataset(X, y)
    loader  = DataLoader(dataset, batch_size=16, shuffle=False)

    logger.info(f"  Created {len(dataset)} evaluation windows (window_size={window_size})")

    # ── Load PC-NDT model — reinitialize with N=2 graph, load compatible weights ──
    pronostia_config['preprocessing']['window_size'] = window_size
    model = PCNDT(pronostia_config).to(device)

    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        logger.error(
            f"No checkpoint found at {checkpoint_path}. "
            "Train the model first or pass --checkpoint to a valid .pt file."
        )
        return

    ckpt = torch.load(checkpoint_path, map_location=device)
    # Transfer every tensor whose shape still matches. The only expected
    # mismatch is the learned node embeddings, which depend on N=4 for IMS
    # but N=2 for PRONOSTIA.
    model_dict = model.state_dict()
    pretrained_dict = {
        k: v for k, v in ckpt['model_state'].items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    skipped_keys = sorted(set(ckpt['model_state'].keys()) - set(pretrained_dict.keys()))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    logger.info(
        f"  Loaded {len(pretrained_dict)} transferable weight tensors; "
        f"skipped {len(skipped_keys)} mismatched tensors"
    )
    logger.info(
        "  NOTE: only node embeddings are re-initialized here because PRONOSTIA "
        "uses N=2 channels instead of IMS's N=4 nodes. That is the remaining "
        "partial-transfer caveat."
    )

    # ── Run inference ─────────────────────────────────────────────────
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            out = model(X_batch)
            preds.append(out['rul'].cpu().numpy())
            targets.append(y_batch.numpy())

    preds   = np.concatenate(preds)
    targets = np.concatenate(targets)

    # ── Metrics ────────────────────────────────────────────────────────
    cfg_phm = base_config['evaluation']['phm2012']
    pronostia_rmse  = rmse(targets.ravel(), preds.ravel())
    pronostia_score = phm2012_score(targets.ravel(), preds.ravel(),
                                     cfg_phm['early_denominator'],
                                     cfg_phm['late_denominator'])

    # ── Delta RMSE (cross-condition generalisation, RQ3) ────────────────
    ims_rmse = args.ims_test_rmse
    if ims_rmse is None and os.path.exists('results/evaluation_results.json'):
        with open('results/evaluation_results.json') as f:
            ims_results = json.load(f)
        ims_rmse = ims_results['metrics']['aggregate']['rmse']

    delta = None
    if ims_rmse is not None:
        delta = delta_rmse(ims_rmse, pronostia_rmse)

    # ── Print results ────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  RQ3 — CROSS-CONDITION GENERALISATION TEST")
    print(f"  Bearing: {args.bearing} | Condition {data['condition_id']}")
    print("=" * 60)
    print(f"  PRONOSTIA RMSE:        {pronostia_rmse:.5f}")
    print(f"  PRONOSTIA PHM Score:   {pronostia_score:.5f}")
    if ims_rmse is not None:
        print(f"  IMS Test 3 RMSE:       {ims_rmse:.5f}  (source condition)")
        print(f"  ΔRMSEdrop:             {delta:+.2f}%")
        print()
        print(f"  Interpretation: model trained at 2000 RPM (IMS) evaluated")
        print(f"  zero-shot at {pronostia_loader.CONDITIONS[data['condition_id']]['rpm']} RPM (PRONOSTIA).")
        print(f"  Lower ΔRMSEdrop = better generalisation.")
    else:
        print(f"  (Run scripts/evaluate.py first to get IMS RMSE for ΔRMSEdrop)")
    print("=" * 60)
    print()

    # ── Save results ────────────────────────────────────────────────────
    os.makedirs('results', exist_ok=True)
    output = {
        'bearing':          args.bearing,
        'condition_id':     data['condition_id'],
        'condition_details':pronostia_loader.CONDITIONS[data['condition_id']],
        'n_windows':        len(dataset),
        'pronostia_rmse':   round(pronostia_rmse, 6),
        'pronostia_phm2012_score': round(pronostia_score, 6),
        'ims_test3_rmse':   ims_rmse,
        'delta_rmse_drop_pct': round(delta, 2) if delta is not None else None,
        'note': 'Partial weight transfer with only node embeddings reinitialized due to N=2 vs N=4 graph mismatch. '
            'For strongest RQ3 claim, retrain jointly or replace node-specific embeddings with a graph-size-agnostic encoder.'
    }
    out_path = f'results/pronostia_{args.bearing.replace("/", "_")}.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
