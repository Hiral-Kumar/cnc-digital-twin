"""
milling_loader.py — NASA Milling Dataset Loader

WHAT THIS FILE DOES:
    Loads the NASA/BEST-Lab Milling dataset, which contains DIRECT
    flank wear measurements (VB) — actual ground-truth physical wear,
    not inferred from vibration statistics.

    This dataset is used ONLY to validate the Archard constraint —
    it is never part of the RUL training pipeline.

FILE FORMAT:
    NASA Milling data is distributed as a single MATLAB .mat file
    containing a struct array. Each entry represents one cutting run:
        - VB:    flank wear measurement (mm) — the ground truth
        - AE_spindle, AE_table: acoustic emission signals
        - smcAC, smcDC: current signals
        - vib_spindle, vib_table: vibration signals
        - case: metadata (material, feed rate, DOC)

USAGE:
    from src.data.milling_loader import MillingLoader
    loader = MillingLoader(config)
    data = loader.load_all_runs()
    # data['vb_measurements']  : list of wear values per run
    # data['vibration_rms']    : list of RMS vibration per run
"""

import os
from pathlib import Path
import numpy as np
from scipy.io import loadmat
import logging

logger = logging.getLogger(__name__)


class MillingLoader:
    """
    Loader for the NASA Milling Dataset (.mat format).

    Used exclusively for Archard constraint validation (RQ2) —
    comparing the model's predicted wear rate against REAL measured
    flank wear, not just vibration-derived proxies.
    """

    def __init__(self, config: dict):
        self.cfg_data = config['data']['milling']
        self.root_dir = Path(self.cfg_data['root_dir'])

    def _candidate_roots(self):
        """Return possible dataset roots for common workspace layouts."""
        candidates = [self.root_dir]

        lower_root = Path(str(self.root_dir).replace('Milling', 'milling'))
        if lower_root not in candidates:
            candidates.append(lower_root)

        for base in (self.root_dir, lower_root):
            if base.name.lower() == 'milling':
                nested = base / 'mill'
                if nested not in candidates:
                    candidates.append(nested)

        return candidates

    def load_all_runs(self) -> dict:
        """
        Load the NASA Milling .mat file and extract wear + vibration data.

        Returns:
            dict:
                'vb_measurements' : np.ndarray [N_runs] — flank wear (mm)
                'vibration_rms'   : np.ndarray [N_runs] — RMS vibration
                'run_indices'     : np.ndarray [N_runs] — sequential run number
                'case_metadata'   : list of dicts — material, feed, DOC per run
        """
        mat_path = None
        tried_roots = []
        for root in self._candidate_roots():
            tried_roots.append(str(root))
            if not root.exists():
                continue

            direct_files = list(root.glob('*.mat'))
            if direct_files:
                mat_path = direct_files[0]
                break

            nested_files = list(root.rglob('*.mat'))
            if nested_files:
                mat_path = nested_files[0]
                break

        if mat_path is None:
            raise FileNotFoundError(
                f"No .mat file found in any of: {', '.join(tried_roots)}\n"
                f"Download from: https://phm-datasets.s3.amazonaws.com/NASA/3.+Milling.zip"
            )

        logger.info(f"Loading Milling dataset: {mat_path}")

        mat_data = loadmat(str(mat_path), simplify_cells=True)

        # The struct is usually named 'mill' in NASA's distribution
        struct_key = None
        for key in mat_data.keys():
            if not key.startswith('__'):
                struct_key = key
                break

        if struct_key is None:
            raise ValueError(f"Could not find data struct in {mat_path}")

        runs = mat_data[struct_key]
        if not isinstance(runs, (list, np.ndarray)):
            runs = [runs]

        vb_measurements = []
        vibration_rms   = []
        case_metadata    = []

        for i, run in enumerate(runs):
            try:
                vb = self._extract_field(run, 'VB')
                if vb is None or (isinstance(vb, np.ndarray) and vb.size == 0):
                    continue   # skip runs without wear measurement

                vb_value = float(np.atleast_1d(vb)[0])
                if not np.isfinite(vb_value):
                    continue   # skip invalid wear measurements

                vib_signal = self._extract_field(run, 'vib_spindle')
                if vib_signal is None:
                    vib_signal = self._extract_field(run, 'vib_table')

                if vib_signal is not None and np.size(vib_signal) > 0:
                    rms = float(np.sqrt(np.mean(np.asarray(vib_signal).ravel() ** 2)))
                else:
                    rms = np.nan

                vb_measurements.append(vb_value)
                vibration_rms.append(rms)
                case_metadata.append({
                    'run_index': i,
                    'material':  self._extract_field(run, 'material'),
                    'feed_rate': self._extract_field(run, 'feed'),
                    'doc':       self._extract_field(run, 'DOC'),
                })
            except Exception as e:
                logger.debug(f"  Skipping run {i}: {e}")
                continue

        logger.info(f"  Loaded {len(vb_measurements)} runs with valid VB measurements")

        return {
            'vb_measurements': np.array(vb_measurements, dtype=np.float64),
            'vibration_rms':   np.array(vibration_rms,   dtype=np.float64),
            'run_indices':     np.arange(len(vb_measurements)),
            'case_metadata':   case_metadata,
        }

    @staticmethod
    def _extract_field(run, field_name):
        """Safely extract a field from the MATLAB struct, handling nesting."""
        if isinstance(run, dict):
            return run.get(field_name, None)
        try:
            return getattr(run, field_name)
        except AttributeError:
            return None

    def compute_wear_rate(self, data: dict) -> np.ndarray:
        """
        Compute the wear RATE (dVB/d_run) from consecutive measurements.

        This is what gets compared against the model's predicted
        dW/dt in the Archard validation script.

        Returns:
            np.ndarray [N_runs - 1] — wear rate between consecutive runs
        """
        vb = data['vb_measurements']
        # Sort by run index to ensure chronological order
        order = np.argsort(data['run_indices'])
        vb_sorted = vb[order]

        wear_rate = np.diff(vb_sorted)   # simple first-difference rate
        return wear_rate
