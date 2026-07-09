"""
pronostia_loader.py
─────────────────────────────────────────────────────────────────────────────
PRONOSTIA / FEMTO-ST PHM 2012 Dataset Loader

WHAT THIS FILE DOES:
    Loads the PRONOSTIA accelerated bearing degradation dataset used
    as the cross-condition generalisation test (RQ3).

    This dataset is NEVER used for training. It is evaluated zero-shot:
    the model trained on NASA IMS is applied directly to PRONOSTIA
    to test whether physics constraints improve generalisation
    across different operating speeds and loads.

PRONOSTIA vs IMS — KEY STRUCTURAL DIFFERENCES:
    ┌─────────────────────┬──────────────────┬──────────────────────┐
    │ Property            │ NASA IMS         │ PRONOSTIA            │
    ├─────────────────────┼──────────────────┼──────────────────────┤
    │ Run duration        │ Days–weeks       │ 28 min – 7 hours     │
    │ Sampling rate       │ 20,480 Hz (cont) │ 25,600 Hz (burst)    │
    │ Burst structure     │ None (continuous)│ 0.1s every 10s       │
    │ Temperature data    │ No               │ Yes (10 Hz)          │
    │ Conditions          │ 1 (2000 RPM)     │ 3 (1500–1800 RPM)    │
    │ Bearings per run    │ 4 simultaneously │ 1 per run            │
    │ Graph size          │ N=4 natural      │ N=2 (h+v channels)   │
    └─────────────────────┴──────────────────┴──────────────────────┘

FOLDER STRUCTURE (what your downloaded PRONOSTIA should look like):
    PRONOSTIA/
    ├── Training_set/
    │   ├── Bearing1_1/
    │   │   ├── acc_00001.csv   ← vibration burst files
    │   │   ├── acc_00002.csv
    │   │   └── ...
    │   ├── Bearing1_2/
    │   └── ...
    └── Test_set/
        ├── Bearing3_3/
        └── ...

USAGE:
    from src.data.pronostia_loader import PronostiaLoader
    loader = PronostiaLoader(config)
    data = loader.load_bearing("Training_set/Bearing1_1")
    # data['features'] shape: [N_bursts, 2, N_features]  (N=2: horiz + vert)
    # data['rul']      shape: [N_bursts, 2]
─────────────────────────────────────────────────────────────────────────────
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft
from typing import Optional
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class PronostiaLoader:
    """
    Loader for the PRONOSTIA/FEMTO-ST PHM 2012 Bearing Dataset.

    This dataset serves one role in PC-NDT: cross-condition generalisation
    evaluation. It is never used during training.

    Node layout (N=2 for PRONOSTIA, vs N=4 for IMS):
        Node 0 → horizontal accelerometer
        Node 1 → vertical accelerometer

    WHY N=2 INSTEAD OF N=4:
        PRONOSTIA monitors one bearing at a time, with two accelerometers
        mounted orthogonally. There is no shared-shaft multi-bearing setup
        like IMS. Using N=2 here is honest to the data structure.

        In cross-condition evaluation, we evaluate bearing-by-bearing
        rather than requiring the same graph size as IMS. The RUL
        prediction is per-node (per accelerometer direction), and we
        average across nodes to get a single bearing-level RUL estimate.
    """

    # Operating conditions for the three PRONOSTIA test conditions
    # Used for physics constraint calculations
    CONDITIONS = {
        1: {'rpm': 1800, 'load_n': 4000},
        2: {'rpm': 1650, 'load_n': 4200},
        3: {'rpm': 1500, 'load_n': 5000},
    }

    # Vibration sampling rate (Hz) — bursts are at this rate
    VIBRATION_RATE = 25600

    # Burst duration and interval
    BURST_DURATION_S = 0.1     # each burst is 0.1 seconds
    BURST_INTERVAL_S = 10.0    # one burst collected every 10 seconds
    SAMPLES_PER_BURST = int(VIBRATION_RATE * BURST_DURATION_S)  # 2560

    def __init__(self, config: dict):
        """
        Args:
            config: Full config dictionary from config.yaml.
        """
        self.cfg_data = config['data']['pronostia']
        self.cfg_prep = config['preprocessing']
        self.root_dir = self.cfg_data['root_dir']

        # Compute BPFO for each condition (varies with RPM)
        # BPFO = (n_balls/2) × (1 − (d/D)cos φ) × shaft_freq_hz
        b = self.cfg_prep['bearing']
        cos_angle = np.cos(np.radians(b['contact_angle_deg']))
        ball_to_pitch = b['ball_diameter_mm'] / b['pitch_diameter_mm']
        base_factor = (b['n_balls'] / 2) * (1 - ball_to_pitch * cos_angle)

        self.bpfo_per_condition = {}
        for cond_id, cond_vals in self.CONDITIONS.items():
            shaft_freq = cond_vals['rpm'] / 60.0
            self.bpfo_per_condition[cond_id] = base_factor * shaft_freq

        logger.info(f"PRONOSTIA BPFO per condition: {self.bpfo_per_condition}")

    def load_bearing(self,
                     bearing_subdir: str,
                     condition_id: int = None) -> dict:
        """
        Load all burst files for a single bearing run.

        Args:
            bearing_subdir: subdirectory name, e.g. "Training_set/Bearing1_1"
                           The naming convention "BearingX_Y" means:
                           X = condition number (1, 2, or 3)
                           Y = bearing index within that condition
            condition_id:  Override condition detection. If None, auto-detected
                          from directory name (first digit after 'Bearing').

        Returns:
            dict:
                'features'    : np.ndarray [N_bursts, 2, N_features]
                'rul'         : np.ndarray [N_bursts, 2] normalized [0,1]
                'rul_raw'     : np.ndarray [N_bursts, 2] in burst counts
                'fpt_idx'     : np.ndarray [2] FPT per channel
                'condition_id': int
                'bearing_dir' : str
                'n_bursts'    : int
        """
        bearing_dir = os.path.join(self.root_dir, bearing_subdir)

        if not os.path.exists(bearing_dir):
            raise FileNotFoundError(
                f"Bearing directory not found: {bearing_dir}\n"
                f"Update config/config.yaml → data.pronostia.root_dir"
            )

        # Auto-detect condition from directory name if not specified
        if condition_id is None:
            condition_id = self._detect_condition(bearing_subdir)

        bpfo = self.bpfo_per_condition.get(condition_id, self.bpfo_per_condition[1])

        # Load vibration burst files (acc_XXXXX.csv format)
        acc_files = sorted([
            f for f in os.listdir(bearing_dir)
            if f.startswith('acc_') and f.endswith('.csv')
        ])

        if len(acc_files) == 0:
            raise ValueError(
                f"No acceleration files (acc_*.csv) found in {bearing_dir}\n"
                f"Ensure you have the correct PRONOSTIA folder structure."
            )

        logger.info(
            f"Loading {bearing_subdir}: {len(acc_files)} bursts, "
            f"Condition {condition_id} ({self.CONDITIONS[condition_id]})"
        )

        n_nodes = 2   # horizontal (col 0) + vertical (col 1)
        n_features = self.cfg_prep['n_features']
        features_list = []

        for fname in tqdm(acc_files, desc=f"  {os.path.basename(bearing_subdir)}"):
            fpath = os.path.join(bearing_dir, fname)
            burst = self._load_burst_file(fpath)
            # burst shape: [2560, 2]

            feats = self._extract_features(burst, bpfo)
            # feats shape: [2, n_features]
            features_list.append(feats)

        features = np.stack(features_list, axis=0)
        # features shape: [N_bursts, 2, N_features]

        # Baseline drift removal (same as IMS loader)
        drift_window = min(
            self.cfg_prep['drift_removal_window'],
            len(features) // 4  # at most 25% of bursts (runs are short)
        )
        baseline = features[:drift_window].mean(axis=0, keepdims=True)
        features = features - baseline

        # FPT computation
        fpt_indices = self._compute_fpt(features)

        # RUL assignment
        n_bursts = len(acc_files)
        rul_raw, rul_normalized = self._assign_rul(n_bursts, n_nodes, fpt_indices)

        return {
            'features':     features,
            'rul':          rul_normalized,
            'rul_raw':      rul_raw,
            'fpt_idx':      fpt_indices,
            'condition_id': condition_id,
            'bearing_dir':  bearing_subdir,
            'n_bursts':     n_bursts,
            'bpfo_hz':      bpfo,
        }

    def load_all_bearings(self, subset: str = 'Training_set') -> list:
        """
        Load all bearings in a given subset (Training_set or Test_set).

        Args:
            subset: 'Training_set' or 'Test_set'

        Returns:
            list of dicts, one per bearing run
        """
        subset_dir = os.path.join(self.root_dir, subset)
        if not os.path.exists(subset_dir):
            logger.warning(f"Subset directory not found: {subset_dir}")
            return []

        bearing_dirs = sorted([
            d for d in os.listdir(subset_dir)
            if d.startswith('Bearing') and
            os.path.isdir(os.path.join(subset_dir, d))
        ])

        all_data = []
        for bdir in bearing_dirs:
            try:
                data = self.load_bearing(f"{subset}/{bdir}")
                all_data.append(data)
            except Exception as e:
                logger.warning(f"Could not load {bdir}: {e}")

        logger.info(f"Loaded {len(all_data)} bearings from {subset}")
        return all_data

    def _detect_condition(self, bearing_subdir: str) -> int:
        """
        Auto-detect operating condition from directory name.
        'Training_set/Bearing1_1' → condition 1
        'Training_set/Bearing2_3' → condition 2
        """
        basename = os.path.basename(bearing_subdir)
        # BearingX_Y — X is the condition number
        try:
            condition_char = basename.replace('Bearing', '')[0]
            return int(condition_char)
        except (IndexError, ValueError):
            logger.warning(
                f"Could not auto-detect condition from '{basename}'. "
                f"Defaulting to condition 1."
            )
            return 1

    def _load_burst_file(self, fpath: str) -> np.ndarray:
        """
        Load one PRONOSTIA vibration burst file.

        PRONOSTIA acc_*.csv format (no header):
            Column 0: time (seconds within burst)
            Column 1: horizontal acceleration (g)
            Column 2: vertical acceleration (g)

        Returns:
            np.ndarray [2560, 2] — horizontal and vertical channels
        """
        try:
            df = pd.read_csv(fpath, header=None)
            # Skip time column (col 0), take horiz (col 1) and vert (col 2)
            raw = df.values[:, 1:3].astype(np.float32)

            # Pad or trim to exactly SAMPLES_PER_BURST (2560) samples
            # Burst files should all be identical length, but defensive coding
            if len(raw) < self.SAMPLES_PER_BURST:
                pad = np.zeros(
                    (self.SAMPLES_PER_BURST - len(raw), 2), dtype=np.float32
                )
                raw = np.vstack([raw, pad])
            elif len(raw) > self.SAMPLES_PER_BURST:
                raw = raw[:self.SAMPLES_PER_BURST]

            return raw

        except Exception as e:
            logger.warning(f"Could not load burst file {fpath}: {e}")
            return np.zeros((self.SAMPLES_PER_BURST, 2), dtype=np.float32)

    def _extract_features(self, burst: np.ndarray, bpfo: float) -> np.ndarray:
        """
        Extract 5 statistical features from a 0.1-second vibration burst.

        Same 5 features as IMS loader for consistency:
        [RMS, Kurtosis, Crest Factor, Peak-to-Peak, Spectral Amplitude @ BPFO]

        This consistency is deliberate: the same feature extraction applied
        to both datasets means the Neural ODE sees inputs with the same
        semantic meaning, even though the raw signals come from different
        experiments. Consistent features are essential for cross-dataset
        generalisation — if IMS features mean one thing and PRONOSTIA
        features mean something different, zero-shot transfer is impossible.

        Args:
            burst: np.ndarray [2560, 2] — one burst, two channels
            bpfo:  float — Ball Pass Frequency Outer Race for this condition

        Returns:
            np.ndarray [2, 5] — features per channel
        """
        n_samples, n_channels = burst.shape
        features = np.zeros((n_channels, 5), dtype=np.float32)

        for ch in range(n_channels):
            signal = burst[:, ch]

            rms = np.sqrt(np.mean(signal ** 2))
            kurt = float(stats.kurtosis(signal, fisher=True))
            peak = np.max(np.abs(signal))
            crest_factor = peak / (rms + 1e-10)
            peak_to_peak = float(np.max(signal) - np.min(signal))

            # Spectral amplitude at BPFO
            fft_mag = np.abs(fft(signal))[:n_samples // 2]
            freqs = np.fft.fftfreq(n_samples, d=1.0 / self.VIBRATION_RATE)
            freqs = freqs[:n_samples // 2]
            bpfo_idx = np.argmin(np.abs(freqs - bpfo))
            spectral_amp = float(fft_mag[bpfo_idx])

            features[ch] = [rms, kurt, crest_factor, peak_to_peak, spectral_amp]

        return features

    def _compute_fpt(self, features: np.ndarray) -> np.ndarray:
        """
        Compute FPT indices for PRONOSTIA runs.
        Same logic as IMS, but PRONOSTIA runs are shorter so we use
        a smaller rolling window (capped at 25% of total bursts).
        """
        n_bursts, n_nodes, _ = features.shape
        fpt_indices = np.zeros(n_nodes, dtype=int)

        rolling_window = min(
            self.cfg_prep['fpt']['rolling_window'],
            max(5, n_bursts // 4)
        )
        sigma_thresh = self.cfg_prep['fpt']['rms_sigma_threshold']
        rms_series = features[:, :, 0]

        for node_idx in range(n_nodes):
            rms = rms_series[:, node_idx]
            baseline_mean = rms[:rolling_window].mean()
            baseline_std = rms[:rolling_window].std()
            threshold = baseline_mean + sigma_thresh * baseline_std

            above = np.where(rms > threshold)[0]
            if len(above) > 0:
                fpt_indices[node_idx] = above[0]
            else:
                fpt_indices[node_idx] = int(0.80 * n_bursts)

        return fpt_indices

    def _assign_rul(self, n_bursts: int, n_nodes: int,
                    fpt_indices: np.ndarray) -> tuple:
        """
        Assign RUL labels using the FPT convention.
        Identical logic to IMS loader — consistent labeling across datasets.
        """
        rul_raw = np.zeros((n_bursts, n_nodes), dtype=np.float32)

        for node_idx in range(n_nodes):
            fpt = fpt_indices[node_idx]
            max_rul = n_bursts - fpt

            for t in range(n_bursts):
                if t <= fpt:
                    rul_raw[t, node_idx] = float(max_rul)
                else:
                    rul_raw[t, node_idx] = float(max_rul - (t - fpt))

            rul_raw[:, node_idx] = np.clip(rul_raw[:, node_idx], 0, max_rul)

        rul_max = rul_raw.max(axis=0, keepdims=True)
        rul_max = np.maximum(rul_max, 1.0)
        rul_normalized = rul_raw / rul_max

        return rul_raw, rul_normalized
