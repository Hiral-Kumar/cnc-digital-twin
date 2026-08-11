"""
inspect_milling.py — quick NASA Milling dataset inspection.

Prints a compact summary of the dataset so you can verify the loader,
the wear measurements, and the available metadata from the command line.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.milling_loader import MillingLoader


logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


def main():
    config_path = PROJECT_ROOT / 'config' / 'config.yaml'
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    logger.info("Loading NASA Milling dataset...")
    loader = MillingLoader(config)
    data = loader.load_all_runs()

    vb = data['vb_measurements']
    vibration_rms = data['vibration_rms']
    cases = data['case_metadata']
    finite_vibration_rms = vibration_rms[np.isfinite(vibration_rms)]

    print()
    print("=" * 72)
    print("NASA MILLING DATASET INSPECTION")
    print("=" * 72)
    print(f"Dataset root:            {config['data']['milling']['root_dir']}")
    print(f"Valid runs loaded:       {len(vb)}")
    print(f"VB range (mm):           [{vb.min():.4f}, {vb.max():.4f}]")
    print(f"VB mean / std (mm):      {vb.mean():.4f} / {vb.std():.4f}")
    if len(finite_vibration_rms) > 0:
        p05, p50, p95 = np.percentile(finite_vibration_rms, [5, 50, 95])
        print(f"Vibration RMS range:     [{finite_vibration_rms.min():.6f}, {finite_vibration_rms.max():.6f}]")
        print(f"Vibration RMS p05/p50/p95:{p05:.6f} / {p50:.6f} / {p95:.6f}")
    else:
        print("Vibration RMS range:     no finite vibration samples found")
    print()
    print("First 5 runs:")
    for idx, case in enumerate(cases[:5]):
        print(
            f"  {idx + 1:>2}. run_index={case.get('run_index')} | "
            f"material={case.get('material')} | feed={case.get('feed_rate')} | doc={case.get('doc')}"
        )

    output = {
        'valid_runs_loaded': int(len(vb)),
        'vb_min_mm': float(vb.min()),
        'vb_max_mm': float(vb.max()),
        'vb_mean_mm': float(vb.mean()),
        'vb_std_mm': float(vb.std()),
    }

    results_path = PROJECT_ROOT / 'results' / 'milling_inspection.json'
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    print()
    print(f"Saved summary: {results_path}")
    print()


if __name__ == '__main__':
    main()