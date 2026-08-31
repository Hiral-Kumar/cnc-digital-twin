"""
train.py — Entry point for PC-NDT training (V2: mixed failure-mode training)

═══════════════════════════════════════════════════════════════════════
WHY THIS CHANGED:
    Rigorous diagnosis (documented in project history) confirmed that
    training on IMS Test 1 ALONE (inner race + roller element failures)
    produces a model that cannot generalize to IMS Test 3 (outer race
    failure) -- NOT because of a bug, but because the model has simply
    never seen an outer-race degradation signature during training.
    Mini-test RMSE (within Test 1, same failure modes) improved 5x after
    fixing the physics constraint formulation; Test 3 RMSE (different
    failure mode) did not move at all -- confirming this is a data
    diversity limitation, not an architecture or loss-function bug.

THE FIX:
    Combine IMS Test 1 (inner race + roller) AND Test 2 (outer race --
    the SAME failure mode as Test 3) into the training set. Test 3
    remains completely untouched as the held-out evaluation set.

    This directly tests the hypothesis: does exposing the model to an
    outer-race failure signature during training (even from a DIFFERENT
    physical bearing than Test 3) improve generalization to Test 3's
    outer-race failure?

IMPORTANT CAVEAT (documented honestly):
    This is a genuine, reasonable fix -- but it does slightly weaken the
    "held-out" claim for Test 3, since the training set now includes the
    SAME FAILURE MODE (though not the same physical run/bearing/data).
    This should be stated explicitly in the paper: Test 3 remains a
    held-out RUN, but not a held-out FAILURE MODE, once this change is
    applied. This is a reasonable and common practice in PHM literature
    (training on diverse failure modes, testing on unseen runs of a
    similar mode), but the distinction matters for how strongly the
    generalization claim can be stated.
═══════════════════════════════════════════════════════════════════════
"""

import sys, os, argparse, logging, yaml, torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data        import IMSLoader
from src.data.preprocessing import (
    MinMaxNormalizer, chronological_split, create_sliding_windows,
    BearingRULDataset,
)
from src.models.pc_ndt     import PCNDT
from src.physics.constraints import PhysicsConstraints
from src.training.trainer  import Trainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def build_mixed_training_data(config: dict):
    """
    Load and combine IMS Test 1 and Test 2 for training, keeping their
    validation splits separate but combinable, and fitting ONE shared
    normalizer across both (fit on Test 1's train split + Test 2's
    train split combined, so both failure modes contribute to the
    normalization statistics).

    Returns:
        train_dataset:  ConcatDataset combining Test 1 + Test 2 training windows
        val_dataset:    ConcatDataset combining Test 1 + Test 2 validation windows
        normalizer:     the single fitted MinMaxNormalizer (needed for
                        evaluate.py to correctly transform Test 3 later)
    """
    loader = IMSLoader(config)
    cfg_prep  = config['preprocessing']
    cfg_split = config['splits']

    logger.info("Loading IMS Test 1 (inner race + roller element failure)...")
    data_t1 = loader.load_test(test_id=1)

    logger.info("Loading IMS Test 2 (outer race failure — same mode as Test 3)...")
    data_t2 = loader.load_test(test_id=2)

    # ── Step 1: chronological split BOTH tests independently ──────────
    splits_t1 = chronological_split(
        data_t1['features'], data_t1['rul'],
        train_frac=cfg_split['train_fraction'],
        val_frac=cfg_split['val_fraction'],
    )
    splits_t2 = chronological_split(
        data_t2['features'], data_t2['rul'],
        train_frac=cfg_split['train_fraction'],
        val_frac=cfg_split['val_fraction'],
    )

    # ── Step 2: fit ONE normalizer on BOTH training splits combined ────
    # This ensures the normalizer sees the full range of values from
    # BOTH failure modes, so Test 3 (outer race, like Test 2) will be
    # transformed using statistics that already account for that
    # failure mode's typical feature ranges.
    combined_train_features = np.concatenate([
        splits_t1['train']['features'],
        splits_t2['train']['features'],
    ], axis=0)

    normalizer = MinMaxNormalizer()
    normalizer.fit(combined_train_features)
    logger.info(
        f"  Fitted ONE shared normalizer on combined Test1+Test2 training "
        f"data ({len(combined_train_features)} total files)"
    )

    # ── Step 3: apply the shared normalizer to all four splits ─────────
    t1_train_feat = normalizer.transform(splits_t1['train']['features'])
    t1_val_feat   = normalizer.transform(splits_t1['val']['features'])
    t2_train_feat = normalizer.transform(splits_t2['train']['features'])
    t2_val_feat   = normalizer.transform(splits_t2['val']['features'])

    # ── Step 4: create sliding windows for each split separately ───────
    # (window creation must happen within each test's own timeline —
    #  windows must never span across the Test1/Test2 boundary, since
    #  that would create a physically meaningless discontinuous window)
    X_t1_train, y_t1_train = create_sliding_windows(
        t1_train_feat, splits_t1['train']['rul'],
        window_size=cfg_prep['window_size'],
        stride_healthy=cfg_prep['stride_healthy'],
        stride_degraded=cfg_prep['stride_degraded'],
        degraded_threshold=cfg_prep['degraded_threshold'],
    )
    X_t1_val, y_t1_val = create_sliding_windows(
        t1_val_feat, splits_t1['val']['rul'],
        window_size=cfg_prep['window_size'],
        stride_healthy=1, stride_degraded=1,
    )
    X_t2_train, y_t2_train = create_sliding_windows(
        t2_train_feat, splits_t2['train']['rul'],
        window_size=cfg_prep['window_size'],
        stride_healthy=cfg_prep['stride_healthy'],
        stride_degraded=cfg_prep['stride_degraded'],
        degraded_threshold=cfg_prep['degraded_threshold'],
    )
    X_t2_val, y_t2_val = create_sliding_windows(
        t2_val_feat, splits_t2['val']['rul'],
        window_size=cfg_prep['window_size'],
        stride_healthy=1, stride_degraded=1,
    )

    logger.info(
        f"  Test 1 windows: train={len(X_t1_train)}, val={len(X_t1_val)}"
    )
    logger.info(
        f"  Test 2 windows: train={len(X_t2_train)}, val={len(X_t2_val)}"
    )

    # ── Step 5: combine into single ConcatDataset for training/val ─────
    train_dataset = ConcatDataset([
        BearingRULDataset(X_t1_train, y_t1_train),
        BearingRULDataset(X_t2_train, y_t2_train),
    ])
    val_dataset = ConcatDataset([
        BearingRULDataset(X_t1_val, y_t1_val),
        BearingRULDataset(X_t2_val, y_t2_val),
    ])

    logger.info(
        f"  COMBINED training set: {len(train_dataset)} windows "
        f"(Test1 + Test2 mixed failure modes)"
    )
    logger.info(f"  COMBINED validation set: {len(val_dataset)} windows")

    return train_dataset, val_dataset, normalizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/config.yaml')
    parser.add_argument('--debug',  action='store_true',
                        help='Fast 5-epoch debug run on IMS Test 2 only (unchanged)')
    parser.add_argument('--mixed', action='store_true', default=True,
                        help='Train on combined Test1+Test2 failure modes (default: True)')
    parser.add_argument('--single', dest='mixed', action='store_false',
                        help='Train on Test1 ONLY (old behavior, for comparison)')
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    set_seed(config['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")

    if args.debug:
        # Debug mode unchanged — fast sanity check on Test 2 alone
        logger.info("DEBUG MODE: fast 5-epoch run on IMS Test 2 only")
        loader = IMSLoader(config)
        raw_data = loader.load_test(2)
        from src.data.preprocessing import build_datasets
        datasets = build_datasets(raw_data, config, is_training_run=True)
        normalizer = datasets['normalizer']
        train_dataset = datasets['train']
        val_dataset   = datasets['val']
        config['training']['epochs'] = 5

    elif args.mixed:
        logger.info("MIXED-FAILURE-MODE TRAINING: combining Test 1 + Test 2")
        train_dataset, val_dataset, normalizer = build_mixed_training_data(config)

    else:
        logger.info("SINGLE-FAILURE-MODE TRAINING: Test 1 only (legacy behavior)")
        loader = IMSLoader(config)
        raw_data = loader.load_test(1)
        from src.data.preprocessing import build_datasets
        datasets = build_datasets(raw_data, config, is_training_run=True)
        normalizer = datasets['normalizer']
        train_dataset = datasets['train']
        val_dataset   = datasets['val']

    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=False, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, drop_last=False)

    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model   = PCNDT(config)
    physics = PhysicsConstraints(config)

    trainer = Trainer(model, config, device=device, normalizer=normalizer)
    history = trainer.fit(train_loader, val_loader, physics)

    logger.info(f"Training complete. Best val_rmse = {trainer.best_val_rmse:.5f}")
    logger.info(
        "Normalizer state saved inside checkpoint — evaluate.py will now "
        "load it exactly instead of refitting. NOTE: this normalizer was "
        f"fitted on {'Test1+Test2 combined' if args.mixed and not args.debug else 'Test1 or Test2 alone'}."
    )


if __name__ == '__main__':
    main()
