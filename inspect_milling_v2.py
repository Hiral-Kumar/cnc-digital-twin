"""
inspect_milling_v2.py — deeper NASA Milling dataset inspection.

Reports two diagnostics:
1. How many distinct (material, feed, DOC) combinations exist.
2. Which run(s) produce impossible vibration RMS values and why.
"""

import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy.io import loadmat

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


def extract_field(run, field_name):
    if isinstance(run, dict):
        return run.get(field_name, None)
    try:
        return getattr(run, field_name)
    except AttributeError:
        return None


def as_numeric_array(signal):
    arr = np.asarray(signal)
    if arr.dtype == object:
        parts = []
        for item in arr.ravel():
            try:
                parts.append(np.asarray(item, dtype=np.float64).ravel())
            except Exception:
                continue
        if not parts:
            return np.array([], dtype=np.float64)
        arr = np.concatenate(parts)
    else:
        arr = arr.astype(np.float64, copy=False).ravel()
    return arr


def signal_summary(signal):
    arr = as_numeric_array(signal)
    finite = arr[np.isfinite(arr)]
    summary = {
        'raw_type': type(signal).__name__,
        'raw_shape': tuple(np.shape(signal)),
        'raw_size': int(np.size(signal)),
        'finite_count': int(len(finite)),
    }
    if len(finite) > 0:
        summary.update({
            'min': float(np.min(finite)),
            'max': float(np.max(finite)),
            'mean': float(np.mean(finite)),
            'rms': float(np.sqrt(np.mean(finite ** 2))),
        })
    else:
        summary.update({'min': None, 'max': None, 'mean': None, 'rms': None})
    return summary


def main():
    config_path = PROJECT_ROOT / 'config' / 'config.yaml'
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    dataset_root = Path(config['data']['milling']['root_dir'])
    candidate_paths = [
        dataset_root,
        Path(str(dataset_root).replace('Milling', 'milling')),
        dataset_root / 'mill',
        Path(str(dataset_root).replace('Milling', 'milling')) / 'mill',
    ]

    mat_path = None
    for root in candidate_paths:
        if root.exists():
            files = list(root.glob('*.mat'))
            if files:
                mat_path = files[0]
                break
            nested = list(root.rglob('*.mat'))
            if nested:
                mat_path = nested[0]
                break

    if mat_path is None:
        raise FileNotFoundError(f'Could not find a .mat file under any candidate root: {candidate_paths}')

    logger.info(f'Loading Milling dataset: {mat_path}')
    mat_data = loadmat(str(mat_path), simplify_cells=True)

    struct_key = None
    for key in mat_data.keys():
        if not key.startswith('__'):
            struct_key = key
            break
    if struct_key is None:
        raise ValueError(f'Could not find a data struct in {mat_path}')

    runs = mat_data[struct_key]
    if not isinstance(runs, (list, np.ndarray)):
        runs = [runs]

    conditions = Counter()
    run_records = []
    suspect_runs = []

    for i, run in enumerate(runs):
        vb = extract_field(run, 'VB')
        if vb is None:
            continue

        try:
            vb_value = float(np.atleast_1d(vb)[0])
        except Exception:
            continue
        if not np.isfinite(vb_value):
            continue

        material = extract_field(run, 'material')
        feed = extract_field(run, 'feed')
        doc = extract_field(run, 'DOC')
        condition_key = (
            None if material is None else float(np.atleast_1d(material)[0]) if np.size(material) else material,
            None if feed is None else float(np.atleast_1d(feed)[0]) if np.size(feed) else feed,
            None if doc is None else float(np.atleast_1d(doc)[0]) if np.size(doc) else doc,
        )
        conditions[condition_key] += 1

        vib_spindle = extract_field(run, 'vib_spindle')
        vib_table = extract_field(run, 'vib_table')
        source = 'vib_spindle' if vib_spindle is not None else 'vib_table'
        raw_signal = vib_spindle if vib_spindle is not None else vib_table
        summary = signal_summary(raw_signal) if raw_signal is not None else {
            'raw_type': None, 'raw_shape': None, 'raw_size': 0, 'finite_count': 0,
            'min': None, 'max': None, 'mean': None, 'rms': None,
        }

        run_records.append({
            'run_index': i,
            'vb': vb_value,
            'condition': condition_key,
            'source': source,
            'summary': summary,
        })

        rms = summary['rms']
        if rms is None or not np.isfinite(rms) or rms > 100.0:
            suspect_runs.append({
                'run_index': i,
                'vb': vb_value,
                'source': source,
                'summary': summary,
                'reason': (
                    'non-finite RMS' if rms is None or not np.isfinite(rms)
                    else 'extreme RMS (> 100)'
                ),
            })

    print()
    print('=' * 88)
    print('NASA MILLING DEEP INSPECTION')
    print('=' * 88)
    print(f'MAT file: {mat_path}')
    print(f'Valid VB runs: {len(run_records)}')
    print()

    print('PART 1 - DISTINCT CUTTING CONDITIONS')
    print('-' * 88)
    print(f'Distinct (material, feed, DOC) combinations: {len(conditions)}')
    print('Condition counts:')
    for condition, count in conditions.most_common():
        print(f'  material={condition[0]} | feed={condition[1]} | doc={condition[2]} -> {count} runs')

    print()
    print('PART 2 - SUSPECT VIBRATION RMS RUNS')
    print('-' * 88)
    if not suspect_runs:
        print('No suspect runs found using the current threshold.')
    else:
        print(f'Suspect runs found: {len(suspect_runs)}')
        for item in sorted(suspect_runs, key=lambda x: (x['reason'], x['run_index'])):
            s = item['summary']
            print(
                f"  run_index={item['run_index']} | reason={item['reason']} | "
                f"source={item['source']} | raw_type={s['raw_type']} | raw_shape={s['raw_shape']} | "
                f"raw_size={s['raw_size']} | finite_count={s['finite_count']} | "
                f"min={s['min']} | max={s['max']} | rms={s['rms']}"
            )

    print()
    print('Top 5 RMS values:')
    top_by_rms = sorted(
        [r for r in run_records if r['summary']['rms'] is not None and np.isfinite(r['summary']['rms'])],
        key=lambda x: x['summary']['rms'],
        reverse=True,
    )[:5]
    for record in top_by_rms:
        s = record['summary']
        print(
            f"  run_index={record['run_index']} | rms={s['rms']:.6e} | "
            f"source={record['source']} | raw_type={s['raw_type']} | raw_shape={s['raw_shape']}"
        )


if __name__ == '__main__':
    main()