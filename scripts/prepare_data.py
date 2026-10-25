"""
prepare_data.py
---------------
Orchestrates the full data pipeline:
  1. Optionally generate synthetic data (if use_synthetic=true)
  2. Load and parse patient XML files
  3. Synchronize multi-modal signals
  4. Interpolate missing CGM readings
  5. Normalize and window
  6. Save pickle for downstream scripts
"""

import os
import sys
import pickle
import logging
import argparse
import yaml
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.data.preprocessing import Preprocessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # -- If synthetic mode, generate the data first --
    if config['dataset'].get('use_synthetic', False):
        syn_dir = config['dataset'].get('data_dir', 'data')
        syn_cfg = config.get('synthetic_data', {})
        n_patients = syn_cfg.get('n_patients', 6)
        n_days = syn_cfg.get('n_days', 14)
        seed = syn_cfg.get('seed', 42)
        syn_out = os.path.join(syn_dir, 'synthetic')
        logger.info(f"Generating synthetic data -> {syn_out}")
        subprocess.run([
            sys.executable, "scripts/generate_synthetic_data.py",
            "--output_dir", syn_out,
            "--n_patients", str(n_patients),
            "--n_days", str(n_days),
            "--seed", str(seed)
        ], check=True)
        # Point config to the synthetic subfolder for loading
        config['dataset']['data_dir'] = syn_out

    preprocessor = Preprocessor.__new__(Preprocessor)
    preprocessor.config = config
    preprocessor.data_dir = config['dataset']['data_dir']
    preprocessor.window_size = config['preprocessing']['window_size']
    preprocessor.horizon = config['preprocessing']['prediction_horizon']
    preprocessor.stride = config['preprocessing']['stride']
    preprocessor.train_split = config['preprocessing']['train_split']
    preprocessor.val_split = config['preprocessing']['val_split']

    logger.info("Preprocessing all patients...")
    processed = preprocessor.process_all_patients()

    if not processed:
        logger.error("No patients processed. Check data directory.")
        sys.exit(1)

    logger.info(f"Processed {len(processed)} patients: {list(processed.keys())}")

    os.makedirs("artifacts", exist_ok=True)
    out_path = args.output or "artifacts/processed_data.pkl"
    with open(out_path, 'wb') as f:
        pickle.dump(processed, f)
    logger.info(f"[OK] Saved to {out_path}")

    # Print summary
    for pid, d in processed.items():
        print(f"  Patient {pid}: train={d['X_train'].shape}, val={d['X_val'].shape}, "
              f"test={d['X_test'].shape}, features={d['n_features']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/reproduction.yaml")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    main(args)
