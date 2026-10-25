import os
import logging
import yaml
import joblib
import numpy as np
import pandas as pd

from src.data.loader import OhioT1DMLoader
from src.data.interpolation import linear_interpolation
from src.data.synchronization import synchronize_modalities
from src.data.windowing import create_sliding_windows
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


class Preprocessor:
    """Orchestrates the full preprocessing pipeline for all patients."""

    def __init__(self, config_path: str = "config/reproduction.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.data_dir = self.config['dataset']['data_dir']
        self.window_size = self.config['preprocessing']['window_size']
        self.horizon = self.config['preprocessing']['prediction_horizon']
        self.stride = self.config['preprocessing']['stride']
        self.train_split = self.config['preprocessing']['train_split']
        self.val_split = self.config['preprocessing']['val_split']

    def process_all_patients(self) -> dict:
        loader = OhioT1DMLoader(self.data_dir)
        patients_raw = loader.load_all_patients()

        if not patients_raw:
            logger.error("No patient data loaded. Aborting.")
            return {}

        os.makedirs("artifacts/scalers", exist_ok=True)
        processed = {}

        for pid, raw in patients_raw.items():
            logger.info(f"Processing patient {pid} ...")

            # 1. Multi-modal synchronization
            sync_df = synchronize_modalities(raw, freq='5min')
            if sync_df.empty or 'glucose_level' not in sync_df.columns:
                logger.warning(f"Patient {pid}: empty after sync. Skipping.")
                continue

            # 2. Interpolation
            sync_df, stats = linear_interpolation(sync_df, glucose_col='glucose_level')
            logger.info(f"  Interpolation stats: {stats}")

            # Drop remaining NaNs in glucose
            sync_df = sync_df.dropna(subset=['glucose_level'])

            if len(sync_df) < self.window_size + self.horizon + 10:
                logger.warning(f"Patient {pid}: insufficient data ({len(sync_df)} rows). Skipping.")
                continue

            # 3. Chronological split
            n = len(sync_df)
            train_end = int(n * self.train_split)
            val_end = train_end + int(n * self.val_split)

            train_df = sync_df.iloc[:train_end].copy()
            val_df = sync_df.iloc[train_end:val_end].copy()
            test_df = sync_df.iloc[val_end:].copy()

            # 4. Min-Max normalization — fit on TRAIN only
            scaler = MinMaxScaler()
            train_scaled = pd.DataFrame(
                scaler.fit_transform(train_df), index=train_df.index, columns=train_df.columns)
            val_scaled = pd.DataFrame(
                scaler.transform(val_df), index=val_df.index, columns=val_df.columns)
            test_scaled = pd.DataFrame(
                scaler.transform(test_df), index=test_df.index, columns=test_df.columns)

            # Save scaler for inverse transform later
            joblib.dump(scaler, f"artifacts/scalers/scaler_patient_{pid}.joblib")

            # 5. Sliding windows
            X_train, Y_train = create_sliding_windows(train_scaled, self.window_size, self.horizon, self.stride)
            X_val, Y_val = create_sliding_windows(val_scaled, self.window_size, self.horizon, self.stride)
            X_test, Y_test = create_sliding_windows(test_scaled, self.window_size, self.horizon, self.stride)

            if X_train.size == 0:
                logger.warning(f"Patient {pid}: no valid windows in train. Skipping.")
                continue

            logger.info(f"  Patient {pid}: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

            processed[pid] = {
                'X_train': X_train, 'Y_train': Y_train,
                'X_val': X_val,   'Y_val': Y_val,
                'X_test': X_test,  'Y_test': Y_test,
                'feature_names': list(sync_df.columns),
                'n_features': X_train.shape[2] if X_train.ndim == 3 else 1
            }

        return processed
