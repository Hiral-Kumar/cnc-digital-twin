"""
train.py — Entry point for PC-NDT training

Usage:
    python scripts/train.py
    python scripts/train.py --debug      # fast 5-epoch run on IMS Test 2
"""

import sys, os, argparse, logging, yaml, torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data        import IMSLoader, build_datasets
from src.models.pc_ndt     import PCNDT
from src.physics.constraints import PhysicsConstraints
from src.training.trainer  import Trainer
from src.evaluation.metrics import compute_all_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/config.yaml')
    parser.add_argument('--debug',  action='store_true',
                        help='Fast 5-epoch debug run on IMS Test 2')
    args = parser.parse_args()

    # FIXED — forces UTF-8 reading on Windows
    with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    set_seed(config['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")

    # ── Data ────────────────────────────────────────────────────────────
    loader = IMSLoader(config)
    test_id = 2 if args.debug else 1
    logger.info(f"Loading IMS Test {test_id} ({'debug' if args.debug else 'training'} mode)")
    raw_data = loader.load_test(test_id)

    datasets = build_datasets(raw_data, config, is_training_run=True)

    batch_size = config['training']['batch_size']
    train_loader = DataLoader(datasets['train'], batch_size=batch_size,
                              shuffle=True,  drop_last=True)
    val_loader   = DataLoader(datasets['val'],   batch_size=batch_size,
                              shuffle=False, drop_last=False)

    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── Model ───────────────────────────────────────────────────────────
    model   = PCNDT(config)
    physics = PhysicsConstraints(config)

    if args.debug:
        config['training']['epochs'] = 5

    # ── Training ─────────────────────────────────────────────────────────
    trainer = Trainer(model, config, device=device)
    history = trainer.fit(train_loader, val_loader, physics)

    logger.info(f"Training complete. Best val_rmse = {trainer.best_val_rmse:.5f}")

    # ── Quick evaluation on mini-test ────────────────────────────────────
    mini_loader = DataLoader(datasets['mini_test'], batch_size=batch_size,
                             shuffle=False)
    mini_rmse = trainer.evaluate(mini_loader)
    logger.info(f"Mini-test RMSE = {mini_rmse:.5f}")


if __name__ == '__main__':
    main()
