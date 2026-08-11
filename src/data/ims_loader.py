"""
ims_loader.py
─────────────────────────────────────────────────────────────────────────────
NASA IMS Bearing Dataset Loader

WHAT THIS FILE DOES:
    Loads the three run-to-failure experiments from the NASA IMS bearing
    dataset. Each experiment runs four bearings simultaneously on a shared
    shaft at 2000 RPM under 6000 lb radial load until one or more bearings
    fail. This file handles all the dataset-specific quirks documented
    through our dataset study.

KEY DESIGN DECISIONS:
    1. Test 1 has 8 channels; Tests 2 & 3 have 4. We reduce Test 1 to 4
       channels by selecting one accelerometer per bearing (horizontal).
       This gives a consistent graph size N=4 across all three tests.

    2. Files are sorted chronologically by filename (filenames ARE timestamps)
       — never alphabetically, which would give wrong order.

    3. RUL labels use the First Prediction Time (FPT) convention: RUL stays
       at its maximum value until the first detectable degradation onset,
       then decreases linearly to 0 at failure. This is more physically
       honest than a pure linear ramp from day 1.

    4. Baseline drift removal: subtract mean of first `drift_window` files
       from all subsequent files. This is causal (uses only past data) and
       deployable at inference time.

USAGE:
    from src.data.ims_loader import IMSLoader
    loader = IMSLoader(config)
    data = loader.load_test(test_id=1)
    # data['features'] shape: [N_files, N_nodes, N_features]
    # data['rul']      shape: [N_files, N_nodes]
    # data['fpt_idx']  shape: [N_nodes]  — index where FPT occurs per bearing
─────────────────────────────────────────────────────────────────────────────
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft
from tqdm import tqdm
import yaml
import logging

# Set up module-level logger
# All print statements in this codebase use logging, not print()
# This way logs can be redirected to files in production
logger = logging.getLogger(__name__)


class IMSLoader:
    """
    Loader for the NASA IMS Bearing Run-to-Failure Dataset.

    The IMS dataset has three experiments (Test 1, Test 2, Test 3).
    This loader handles all three with a consistent interface.
    """

    # Channel layout for IMS Test 1 (8 channels)
    # Each bearing has 2 accelerometers: ch_1 (horizontal), ch_2 (vertical)
    CHANNELS_TEST1 = {
        'bearing1_ch1': 0, 'bearing1_ch2': 1,
        'bearing2_ch1': 2, 'bearing2_ch2': 3,
        'bearing3_ch1': 4, 'bearing3_ch2': 5,
        'bearing4_ch1': 6, 'bearing4_ch2': 7,
    }

    # Channel layout for Tests 2 & 3 (4 channels)
    CHANNELS_TEST23 = {
        'bearing1': 0,
        'bearing2': 1,
        'bearing3': 2,
        'bearing4': 3,
    }

    # Documented failure modes per test (from IMS documentation)
    FAILURE_INFO = {
        1: {'bearing3': 'inner_race', 'bearing4': 'roller_element'},
        2: {'bearing1': 'outer_race'},
        3: {'bearing3': 'outer_race'},
    }

    def __init__(self, config: dict):
        """
        Args:
            config: The full config dictionary loaded from config.yaml.
                    We access config['data']['ims'] and
                    config['preprocessing'] here.
        """
        self.cfg_data = config['data']['ims']
        self.cfg_prep = config['preprocessing']
        self.cfg_graph = config['graph']

        # Bearing geometry for characteristic defect frequency calculation
        # Ball Pass Frequency Outer Race (BPFO):
        # BPFO = (n_balls / 2) × (1 - ball_dia/pitch_dia × cos(contact_angle)) × shaft_freq
        b = self.cfg_prep['bearing']
        shaft_freq_hz = b['shaft_speed_rpm'] / 60.0
        cos_angle = np.cos(np.radians(b['contact_angle_deg']))
        ball_to_pitch = b['ball_diameter_mm'] / b['pitch_diameter_mm']
        self.bpfo = (b['n_balls'] / 2) * (1 - ball_to_pitch * cos_angle) * shaft_freq_hz
        logger.info(f"Computed BPFO = {self.bpfo:.2f} Hz for IMS bearings")

    def load_test(self, test_id: int) -> dict:
        """
        Load a complete run-to-failure experiment.

        Args:
            test_id: 1, 2, or 3 — which IMS test to load.

        Returns:
            dict with keys:
                'features'  : np.ndarray [N_files, N_nodes, N_features]
                'rul'       : np.ndarray [N_files, N_nodes] — normalized [0,1]
                'rul_raw'   : np.ndarray [N_files, N_nodes] — raw cycle counts
                'fpt_idx'   : np.ndarray [N_nodes] — FPT index per bearing
                'n_files'   : int
                'test_id'   : int
                'filenames' : list of str — sorted filenames (= timestamps)
        """
        logger.info(f"Loading IMS Test {test_id}...")

        # Step 1: Get the directory and sort files chronologically
        dir_map = {1: self.cfg_data['test1_dir'],
                   2: self.cfg_data['test2_dir'],
                   3: self.cfg_data['test3_dir']}
        test_dir = dir_map[test_id]

        if not os.path.exists(test_dir):
            raise FileNotFoundError(
                f"IMS Test {test_id} directory not found at: {test_dir}\n"
                f"Please update config/config.yaml → data.ims.test{test_id}_dir "
                f"to point to your unzipped IMS folder."
            )

        # Sort filenames chronologically — filenames ARE timestamps in IMS
        # e.g., "2003.10.22.12.06.24" — sorted alphabetically = chronologically
        filenames = sorted([
            f for f in os.listdir(test_dir)
            if not f.startswith('.')   # skip hidden files
        ])

        if len(filenames) == 0:
            raise ValueError(f"No files found in {test_dir}")

        logger.info(f"  Found {len(filenames)} files in Test {test_id}")

        # Step 2: Determine which columns to use
        # Test 1 → select horizontal accelerometers only (4-node reduction)
        # Tests 2/3 → use all 4 columns as-is
        if test_id == 1:
            col_selection = self.cfg_data['test1_channel_selection']  # [0, 2, 4, 6]
        else:
            col_selection = [0, 1, 2, 3]

        n_nodes = self.cfg_graph['n_nodes']  # always 4
        n_features = self.cfg_prep['n_features']

        # Step 3: Load all files and extract features
        # We extract features file-by-file to avoid loading the entire
        # dataset into RAM simultaneously (each file = 20480 samples × 4/8 channels)
        features_list = []

        for fname in tqdm(filenames, desc=f"  Loading Test {test_id}"):
            fpath = os.path.join(test_dir, fname)
            raw = self._load_single_file(fpath, col_selection)
            # raw shape: [20480, 4]

            feats = self._extract_features(raw)
            # feats shape: [4, 5] = [n_nodes, n_features]
            features_list.append(feats)

        # Stack into [N_files, N_nodes, N_features]
        features = np.stack(features_list, axis=0)
        logger.info(f"  Feature matrix shape: {features.shape}")

        # Step 4: Baseline drift removal
        # Subtract mean of first `drift_window` files from all files
        # This is a causal operation (uses only the early-life baseline)
        drift_window = self.cfg_prep['drift_removal_window']
        baseline = features[:drift_window].mean(axis=0, keepdims=True)
        # baseline shape: [1, N_nodes, N_features]
        features = features - baseline
        # Note: after subtraction, healthy early-life features are near zero.
        # Degradation shows as increasing positive deviation from baseline.
        logger.info(f"  Applied baseline drift removal (window={drift_window})")

        # Step 5: Compute FPT (First Prediction Time) per bearing
        fpt_indices = self._compute_fpt(features)
        logger.info(f"  FPT indices per bearing: {fpt_indices}")

        # Step 6: Assign RUL labels using FPT convention
        n_files = len(filenames)
        rul_raw, rul_normalized = self._assign_rul(n_files, n_nodes, fpt_indices)

        return {
            'features':   features,          # [N_files, N_nodes, N_features]
            'rul':        rul_normalized,    # [N_files, N_nodes] in [0, 1]
            'rul_raw':    rul_raw,           # [N_files, N_nodes] in cycles
            'fpt_idx':    fpt_indices,       # [N_nodes]
            'n_files':    n_files,
            'test_id':    test_id,
            'filenames':  filenames,
            'bpfo_hz':    self.bpfo,
        }

    def _load_single_file(self, fpath: str, col_selection: list) -> np.ndarray:
        """
        Load one IMS data file and return selected channels.

        IMS files are tab-separated text files, no header.
        Each file = exactly 20480 rows × 4 or 8 columns.

        Returns:
            np.ndarray of shape [20480, n_nodes] — raw vibration samples
        """
        try:
            # Using pandas for robust parsing (handles varied whitespace)
            df = pd.read_csv(fpath, sep='\t', header=None)
            raw = df.values[:, col_selection].astype(np.float32)
            return raw
        except Exception as e:
            logger.warning(f"Could not load file {fpath}: {e}")
            # Return zeros for corrupted files rather than crashing the pipeline
            # You can track which files were corrupted via the warning logs
            return np.zeros((20480, len(col_selection)), dtype=np.float32)

    def _extract_features(self, raw: np.ndarray) -> np.ndarray:
        """
        Extract 5 statistical features from raw vibration signal.

        WHY THESE 5 FEATURES:
            RMS           → overall energy, rises monotonically in Stage 3-4
            Kurtosis      → impulsiveness, spikes in Stage 2-3 (non-monotonic!)
            Crest Factor  → peak/RMS, sensitive to early isolated impacts
            Peak-to-Peak  → total amplitude range, good for shock events
            Spectral Amp  → amplitude at BPFO frequency, bearing-specific indicator

        Args:
            raw: np.ndarray [20480, n_nodes] — one file of raw vibration data

        Returns:
            np.ndarray [n_nodes, 5] — feature vector per bearing
        """
        n_samples, n_nodes = raw.shape
        features = np.zeros((n_nodes, 5), dtype=np.float32)

        for node_idx in range(n_nodes):
            signal = raw[:, node_idx]

            # Feature 0: RMS (Root Mean Square)
            # √(mean(x²)) — captures overall vibration energy
            rms = np.sqrt(np.mean(signal ** 2))

            # Feature 1: Kurtosis
            # Measures "impulsiveness" — how often large spikes occur relative to
            # the distribution's spread. Healthy bearing ≈ 3.0 (Gaussian).
            # Faulty bearing: kurtosis spikes as defect impacts appear, then
            # DROPS near failure as vibration becomes chaotic. This non-monotonic
            # behavior is important — kurtosis alone is insufficient as a health index.
            # scipy.stats.kurtosis returns excess kurtosis (Gaussian = 0 by default)
            kurt = float(stats.kurtosis(signal, fisher=True))

            # Feature 2: Crest Factor
            # peak / RMS — ratio of peak amplitude to energy content.
            # Good for detecting early-stage isolated defect impulses before RMS rises.
            # If RMS is near zero (all-zero signal from corrupted file), clip to 0.
            peak = np.max(np.abs(signal))
            crest_factor = peak / (rms + 1e-10)  # 1e-10 prevents division by zero

            # Feature 3: Peak-to-Peak
            # Total amplitude range = max − min.
            # Useful for shock events (tool crash, sudden overload) more than
            # gradual wear. Complements RMS which can miss brief spikes.
            peak_to_peak = float(np.max(signal) - np.min(signal))

            # Feature 4: Spectral Amplitude at BPFO
            # FFT of the signal → find the frequency bin closest to BPFO
            # BPFO (Ball Pass Frequency Outer Race) is the characteristic frequency
            # at which a defect on the outer race produces impacts as balls roll over it.
            # Amplitude at this exact frequency is a bearing-specific diagnostic indicator.
            sampling_rate = self.cfg_data['sampling_rate']  # 20480 Hz
            fft_mag = np.abs(fft(signal))[:n_samples // 2]
            freqs = np.fft.fftfreq(n_samples, d=1.0 / sampling_rate)[:n_samples // 2]

            # Find the index closest to BPFO
            bpfo_idx = np.argmin(np.abs(freqs - self.bpfo))
            spectral_amp = float(fft_mag[bpfo_idx])

            # Stack into feature vector for this node
            features[node_idx] = [rms, kurt, crest_factor, peak_to_peak, spectral_amp]

        return features

    def _compute_fpt(self, features: np.ndarray) -> np.ndarray:
        """
        Compute First Prediction Time (FPT) index per bearing.

        FPT = first timestep where RMS vibration crosses 2σ above its
        rolling mean computed over the first `rolling_window` timesteps.

        WHY FPT MATTERS:
            Without FPT, the RUL label would decrease linearly from day 1 —
            implying the bearing is already degrading from the very first file.
            This is physically wrong: bearings run healthy for most of their life.
            FPT ensures the model learns that early-life constant RUL is normal,
            and only after FPT does it learn a decreasing RUL trend.

        Args:
            features: np.ndarray [N_files, N_nodes, N_features]

        Returns:
            np.ndarray [N_nodes] — FPT file index per bearing (0-indexed)
                                    If no FPT detected, defaults to 80% of run.
        """
        n_files, n_nodes, _ = features.shape
        fpt_indices = np.zeros(n_nodes, dtype=int)

        rolling_window = self.cfg_prep['fpt']['rolling_window']
        sigma_thresh = self.cfg_prep['fpt']['rms_sigma_threshold']

        # RMS is feature index 0
        rms_series = features[:, :, 0]  # [N_files, N_nodes]

        for node_idx in range(n_nodes):
            rms = rms_series[:, node_idx]  # [N_files]

            # Compute rolling mean and std using the first `rolling_window` files
            # as the reference (healthy) baseline
            baseline_mean = rms[:rolling_window].mean()
            baseline_std = rms[:rolling_window].std()
            threshold = baseline_mean + sigma_thresh * baseline_std

            # Find first file where RMS exceeds the threshold
            above_threshold = np.where(rms > threshold)[0]

            if len(above_threshold) > 0:
                fpt_indices[node_idx] = above_threshold[0]
            else:
                # No detectable degradation onset — set FPT to 80% of run
                # This happens for bearings that don't fail in this experiment
                # (e.g., Bearing 1 and 2 in Test 1 where only 3 and 4 fail)
                fpt_indices[node_idx] = int(0.80 * n_files)
                logger.debug(
                    f"  No FPT detected for node {node_idx} — "
                    f"defaulting to 80% of run ({fpt_indices[node_idx]})"
                )

        return fpt_indices

    def _assign_rul(self, n_files: int, n_nodes: int,
                    fpt_indices: np.ndarray) -> tuple:
        """
        Assign RUL labels using the FPT convention.

        LABEL SCHEME:
            For each bearing (node):
            - Files 0 to FPT: RUL = (n_files - FPT) [constant — healthy plateau]
            - Files FPT to n_files: RUL linearly decreases from (n_files - FPT) to 0

        This creates a "hockey stick" RUL curve that is more physically honest
        than a straight linear ramp from file 1 to 0.

        Returns:
            rul_raw:        np.ndarray [N_files, N_nodes] — raw RUL in file counts
            rul_normalized: np.ndarray [N_files, N_nodes] — normalized to [0, 1]
        """
        rul_raw = np.zeros((n_files, n_nodes), dtype=np.float32)

        for node_idx in range(n_nodes):
            fpt = fpt_indices[node_idx]
            max_rul = n_files - fpt  # maximum RUL value (at the FPT point)

            for t in range(n_files):
                if t <= fpt:
                    # Healthy plateau: RUL is constant at max_rul
                    rul_raw[t, node_idx] = float(max_rul)
                else:
                    # Degradation phase: linear decrease from max_rul to 0
                    rul_raw[t, node_idx] = float(max_rul - (t - fpt))

            # Clip any negative values (shouldn't happen, but safety check)
            rul_raw[:, node_idx] = np.clip(rul_raw[:, node_idx], 0, max_rul)

        # Normalize to [0, 1]
        # Each node has its own max_rul, so normalize per node
        rul_max_per_node = rul_raw.max(axis=0, keepdims=True)  # [1, N_nodes]
        rul_max_per_node = np.maximum(rul_max_per_node, 1.0)   # prevent div by zero
        rul_normalized = rul_raw / rul_max_per_node

        return rul_raw, rul_normalized
