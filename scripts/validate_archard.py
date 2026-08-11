"""
validate_archard.py - Archard Constraint Validation (RQ2) - V4

ASCII-only version to avoid Windows default encoding issues when the file is
read with Python's default text encoding.
"""

import sys
import os
import logging
import yaml
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.physics.constraints import PhysicsConstraints

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

EXCLUDED_RUNS = {17}

MATERIAL_HARDNESS_PA = {
    1.0: 6.0e9,
    2.0: 2.2e9,
}


def load_milling_runs(root_dir: str) -> list:
    root_path = Path(root_dir)
    if not root_path.is_absolute():
        root_path = PROJECT_ROOT / root_path

    mat_files = sorted([p for p in root_path.glob('*.mat')]) if root_path.exists() else []
    if not mat_files and root_path.exists():
        mat_files = sorted([p for p in root_path.rglob('*.mat')])
    if len(mat_files) > 1:
        logger.warning(
            f"Multiple .mat files found in {root_dir}: {[p.name for p in mat_files]}. "
            "Using the first sorted entry."
        )
    if not mat_files:
        raise FileNotFoundError(
            f"No .mat file found in {root_dir}\n"
            "Download from: https://phm-datasets.s3.amazonaws.com/NASA/3.+Milling.zip"
        )

    mat_path = mat_files[0]
    data = loadmat(str(mat_path), simplify_cells=True)
    struct_key = [k for k in data if not k.startswith('__')][0]
    raw_runs = data[struct_key]
    if not isinstance(raw_runs, (list, np.ndarray)):
        raw_runs = [raw_runs]

    parsed = []
    for i, run in enumerate(raw_runs):
        if i in EXCLUDED_RUNS or not isinstance(run, dict):
            continue

        vb = run.get('VB', None)
        if vb is None or (hasattr(vb, '__len__') and len(vb) == 0):
            continue

        feed = run.get('feed', None)
        doc = run.get('DOC', None)
        material = run.get('material', None)
        if feed is None or doc is None or material is None:
            continue

        sig = run.get('vib_spindle', None)
        if sig is None:
            continue

        sig = np.asarray(sig).ravel()
        rms = float(np.sqrt(np.mean(sig ** 2)))
        if not np.isfinite(rms) or rms > 100:
            continue

        vb_val = float(np.atleast_1d(vb)[0])
        if not np.isfinite(vb_val):
            continue

        parsed.append({
            'run_index': i,
            'vb': vb_val,
            'feed': float(np.atleast_1d(feed)[0]) if hasattr(feed, '__len__') else float(feed),
            'doc': float(np.atleast_1d(doc)[0]) if hasattr(doc, '__len__') else float(doc),
            'material': float(np.atleast_1d(material)[0]) if hasattr(material, '__len__') else float(material),
            'vib_rms': rms,
        })

    return parsed


def main():
    with open(PROJECT_ROOT / 'config' / 'config.yaml', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    root_dir = config['data']['milling']['root_dir']
    logger.info(f"Loading Milling dataset from {root_dir}...")
    runs = load_milling_runs(root_dir)
    logger.info(f"  Loaded {len(runs)} valid runs (excluded: {EXCLUDED_RUNS})")

    groups = {}
    for r in runs:
        key = (r['material'], r['feed'], r['doc'])
        groups.setdefault(key, []).append(r)

    physics = PhysicsConstraints(config)
    rpm = config['preprocessing']['bearing']['shaft_speed_rpm']
    shaft_freq_hz = rpm / 60.0

    group_summary = []
    for (material, feed, doc), group_runs in groups.items():
        if len(group_runs) < 2:
            continue

        group_runs = sorted(group_runs, key=lambda r: r['run_index'])
        vb_series = np.array([r['vb'] for r in group_runs])
        vib_series = np.array([r['vib_rms'] for r in group_runs])

        finite_vb = vb_series[np.isfinite(vb_series)]
        finite_vib = vib_series[np.isfinite(vib_series)]
        if len(finite_vb) < 2 or len(finite_vib) < 1:
            continue

        mean_wear_rate = float(np.mean(np.diff(finite_vb)))
        mean_vib_rms = float(np.mean(finite_vib))

        v = np.pi * 0.02815 * shaft_freq_hz * feed
        F_effective = physics.F_constant * doc
        H_material = MATERIAL_HARDNESS_PA.get(material, physics.H)
        predicted_rate = (physics.k * F_effective * v) / H_material

        group_summary.append({
            'material': material,
            'feed': feed,
            'doc': doc,
            'n_runs': len(group_runs),
            'mean_wear_rate': mean_wear_rate,
            'mean_vib_rms': mean_vib_rms,
            'archard_predicted': predicted_rate,
        })

    predicted = np.array([g['archard_predicted'] for g in group_summary])
    measured = np.array([g['mean_wear_rate'] for g in group_summary])
    vib_rms = np.array([g['mean_vib_rms'] for g in group_summary])
    n_groups = len(group_summary)

    finite_mask = np.isfinite(predicted) & np.isfinite(measured) & np.isfinite(vib_rms)
    predicted = predicted[finite_mask]
    measured = measured[finite_mask]
    vib_rms = vib_rms[finite_mask]
    group_summary = [g for g, keep in zip(group_summary, finite_mask) if keep]
    n_groups = len(group_summary)

    if n_groups < 2:
        raise ValueError('Not enough finite group-level data points to compute correlations.')

    pearson_archard, p_pearson_a = stats.pearsonr(predicted, measured)
    spearman_archard, p_spearman_a = stats.spearmanr(predicted, measured)
    pearson_vib, p_pearson_v = stats.pearsonr(vib_rms, measured)
    spearman_vib, p_spearman_v = stats.spearmanr(vib_rms, measured)

    m1_wear = [g['mean_wear_rate'] for g in group_summary if g['material'] == 1.0]
    m2_wear = [g['mean_wear_rate'] for g in group_summary if g['material'] == 2.0]
    ratio_observed = np.mean(m2_wear) / np.mean(m1_wear)
    ratio_predicted = MATERIAL_HARDNESS_PA[1.0] / MATERIAL_HARDNESS_PA[2.0]

    print()
    print('=' * 70)
    print('  RQ2 - ARCHARD CONSTRAINT PHYSICAL VALIDATION (V4, MATERIAL-AWARE)')
    print('=' * 70)
    print(f'  Runs analyzed:      {len(runs)} (excluded corrupted run 17)')
    print(f'  Condition groups:   {n_groups}')
    print()
    print(f"  {'Group':<28} {'n':>3} {'MeanWear':>10} {'ArchardPred':>13}")
    print(f"  {'-' * 58}")
    for g in sorted(group_summary, key=lambda x: x['archard_predicted']):
        label = f"m={g['material']} f={g['feed']} d={g['doc']}"
        print(f"  {label:<28} {g['n_runs']:>3} {g['mean_wear_rate']:>10.5f} {g['archard_predicted']:>13.3e}")
    print()
    print('  MATERIAL-ISOLATED CHECK (from V3 finding):')
    print(f"    Material 1.0 mean wear: {np.mean(m1_wear):.5f} mm  (n={len(m1_wear)} groups)")
    print(f"    Material 2.0 mean wear: {np.mean(m2_wear):.5f} mm  (n={len(m2_wear)} groups)")
    print(f"    Ratio (m2/m1): {ratio_observed:.2f}x")
    print(f"    Our hardness assumption predicts m2/m1 ratio: {ratio_predicted:.2f}x (inverse hardness ratio)")
    print()
    print('  Archard formula (material-aware) vs mean wear rate:')
    print(f"    Pearson r  = {pearson_archard:+.4f}  (p={p_pearson_a:.3f})")
    print(f"    Spearman rho = {spearman_archard:+.4f}  (p={p_spearman_a:.3f})")
    print()
    print('  Vibration RMS vs mean wear rate:')
    print(f"    Pearson r  = {pearson_vib:+.4f}  (p={p_pearson_v:.3f})")
    print(f"    Spearman rho = {spearman_vib:+.4f}  (p={p_spearman_v:.3f})")
    print()
    print('  Verified: 0 NaN in group-level means after filtering.')
    print()

    if (ratio_observed > 1) != (ratio_predicted > 1):
        print('  WARNING: observed material 2 wears MORE, but the hardness values')
        print('    predict material 2 should wear LESS.')
        print('    This means the material labels may be reversed, or the material')
        print('    field may encode something other than base hardness.')
    print('=' * 70)
    print()

    os.makedirs(PROJECT_ROOT / 'results', exist_ok=True)
    output = {
        'n_runs_analyzed': len(runs),
        'excluded_runs': list(EXCLUDED_RUNS),
        'n_condition_groups': n_groups,
        'group_level_summary': group_summary,
        'material_wear_ratio_observed': round(float(ratio_observed), 3),
        'material_hardness_ratio_assumed': round(float(ratio_predicted), 3),
        'direction_match': bool((ratio_observed > 1) == (ratio_predicted > 1)),
        'pearson_archard': round(float(pearson_archard), 4),
        'pearson_archard_pvalue': round(float(p_pearson_a), 4),
        'spearman_archard': round(float(spearman_archard), 4),
        'pearson_vibration': round(float(pearson_vib), 4),
        'spearman_vibration': round(float(spearman_vib), 4),
        'caveat': f'n={n_groups} groups - low statistical power. Material hardness values are literature estimates, not fitted to this dataset.',
        'verified_nan_filter': True,
        'verified_nan_count': int(np.isnan(np.array([g['mean_wear_rate'] for g in group_summary])).sum()),
    }
    with open(PROJECT_ROOT / 'results' / 'archard_validation.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    logger.info('Saved: results/archard_validation.json')


if __name__ == '__main__':
    main()
