"""
evaluate.py
-----------
Loads saved model weights + test data and computes:
  - RMSE, MAE, MARD, R² per patient and overall
  - Clarke Error Grid (Zone A–E)
  - Clinical decision support analysis
  - Comparison table vs paper-reported values
"""

import os
import sys
import json
import pickle
import logging
import argparse
import yaml
import numpy as np
import joblib
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.models.lightweight_lglm import LightweightLGLM
from src.models.centralized_lstm import CentralizedLSTM
from src.evaluation.metrics import calculate_metrics
from src.clinical.clarke_error_grid import clarke_error_grid
from src.clinical.decision_support import ClinicalDecisionSupport

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# Paper-reported reference values (Table from the paper)
PAPER_RESULTS = {
    'LGLM_30min': {'RMSE': 0.181, 'MAE': 0.140, 'MARD': 8.52, 'R2': 0.923},
    'LGLM_60min': {'RMSE': 0.243, 'MAE': 0.193, 'MARD': 10.87, 'R2': 0.881},
    'Centralized_30min': {'RMSE': 0.198, 'MAE': 0.152, 'MARD': 9.15, 'R2': 0.911},
}


def inverse_transform_glucose(y_scaled: np.ndarray, scaler_path: str,
                               glucose_col_idx: int = 0) -> np.ndarray:
    """Inverse-transform normalized glucose back to mg/dL."""
    try:
        scaler = joblib.load(scaler_path)
        dummy = np.zeros((len(y_scaled), scaler.n_features_in_))
        dummy[:, glucose_col_idx] = y_scaled
        inv = scaler.inverse_transform(dummy)
        return inv[:, glucose_col_idx]
    except Exception as e:
        logger.warning(f"Could not inverse transform: {e}. Returning scaled values.")
        return y_scaled


def evaluate_all(processed: dict, config: dict, run_tag: str = "lglm") -> dict:
    gl_cfg = config['gossip_learning']
    window_size = config['preprocessing']['window_size']
    patient_ids = list(processed.keys())
    n_features = processed[patient_ids[0]]['n_features']

    os.makedirs("artifacts/plots", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # -- Load gossip models --
    models = {}
    for pid in patient_ids:
        model_path = f"artifacts/models/{run_tag}_patient_{pid}.pt"
        if not os.path.exists(model_path):
            logger.warning(f"Model not found for patient {pid} at {model_path}")
            continue
        m = LightweightLGLM(input_features=n_features,
                            hidden_units=gl_cfg['hidden_units'],
                            window_size=window_size)
        m.load_state_dict(torch.load(model_path, map_location='cpu'))
        m.eval()
        models[pid] = m

    if not models:
        logger.error("No models loaded. Run train_lglm.py first.")
        return {}

    cds = ClinicalDecisionSupport(low_threshold=70.0, high_threshold=180.0)
    all_true_mg, all_pred_mg = [], []
    per_patient_results = {}

    for pid in models:
        pdata = processed[pid]
        if pdata['X_test'].size == 0:
            continue

        with torch.no_grad():
            preds_scaled = models[pid](torch.FloatTensor(pdata['X_test'])).numpy()
        true_scaled = pdata['Y_test']

        # Inverse transform to mg/dL if scaler exists
        scaler_path = f"artifacts/scalers/scaler_patient_{pid}.joblib"
        true_mg = inverse_transform_glucose(true_scaled, scaler_path)
        pred_mg = inverse_transform_glucose(preds_scaled, scaler_path)

        metrics = calculate_metrics(true_mg, pred_mg)
        clinical = cds.analyze_predictions(true_mg, pred_mg)
        per_patient_results[pid] = {**metrics, **clinical}
        all_true_mg.extend(true_mg.tolist())
        all_pred_mg.extend(pred_mg.tolist())
        logger.info(f"Patient {pid}: RMSE={metrics['RMSE']:.2f} mg/dL | "
                    f"MAE={metrics['MAE']:.2f} | MARD={metrics['MARD']:.2f}%")

    all_true_mg = np.array(all_true_mg)
    all_pred_mg = np.array(all_pred_mg)

    # -- Clarke Error Grid --
    ceg_zones = clarke_error_grid(
        all_true_mg, all_pred_mg,
        title=f"Clarke Error Grid - {run_tag}",
        save_path=f"artifacts/plots/{run_tag}_clarke.png"
    )
    logger.info(f"Clarke Error Grid zones: {ceg_zones}")

    # -- Overall metrics --
    overall = calculate_metrics(all_true_mg, all_pred_mg)

    # -- Prediction vs Actual plot --
    _plot_predictions(all_true_mg, all_pred_mg, run_tag)

    # -- Residuals plot --
    _plot_residuals(all_true_mg, all_pred_mg, run_tag)

    results = {
        'per_patient': per_patient_results,
        'overall': overall,
        'clarke_zones': ceg_zones,
        'n_patients': len(models),
        'n_test_samples': len(all_true_mg)
    }

    # Save results
    with open(f"results/{run_tag}_evaluation.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Print comparison table
    _print_comparison(overall, run_tag)
    return results


def _plot_predictions(y_true, y_pred, tag, n_plot=300):
    idx = np.random.choice(len(y_true), min(n_plot, len(y_true)), replace=False)
    idx = np.sort(idx)
    fig, ax = plt.subplots(figsize=(14, 5), facecolor='#1a1a2e')
    ax.set_facecolor('#16213e')
    ax.plot(y_true[idx], color='#a8ff78', linewidth=1.5, label='Actual Glucose', alpha=0.9)
    ax.plot(y_pred[idx], color='#00d2ff', linewidth=1.5, linestyle='--', label='Predicted', alpha=0.9)
    ax.axhline(y=70, color='#ff6b6b', linestyle=':', linewidth=1, alpha=0.7, label='Hypo threshold')
    ax.axhline(y=180, color='#f7971e', linestyle=':', linewidth=1, alpha=0.7, label='Hyper threshold')
    ax.set_xlabel("Sample Index", color='white', fontsize=12)
    ax.set_ylabel("Glucose (mg/dL)", color='white', fontsize=12)
    ax.set_title(f"Predicted vs Actual Glucose - {tag}", color='white', fontsize=14, fontweight='bold')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444466')
    ax.legend(facecolor='#2d2d4e', edgecolor='#555577', labelcolor='white', fontsize=10)
    plt.tight_layout()
    plt.savefig(f"artifacts/plots/{tag}_predictions.png", dpi=150, bbox_inches='tight')
    plt.close()


def _plot_residuals(y_true, y_pred, tag):
    residuals = y_pred - y_true
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='#1a1a2e')
    for ax in axes:
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444466')

    axes[0].scatter(y_true, residuals, s=4, alpha=0.4, color='#00d2ff')
    axes[0].axhline(0, color='#a8ff78', linewidth=1.5, linestyle='--')
    axes[0].set_xlabel("Actual Glucose (mg/dL)", color='white')
    axes[0].set_ylabel("Residual (pred - actual)", color='white')
    axes[0].set_title("Residual Plot", color='white', fontweight='bold')

    axes[1].hist(residuals, bins=50, color='#7b2ff7', edgecolor='#555577', alpha=0.8)
    axes[1].axvline(0, color='#a8ff78', linewidth=1.5, linestyle='--')
    axes[1].set_xlabel("Residual (mg/dL)", color='white')
    axes[1].set_ylabel("Count", color='white')
    axes[1].set_title("Residual Distribution", color='white', fontweight='bold')

    plt.suptitle(f"Residual Analysis - {tag}", color='white', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"artifacts/plots/{tag}_residuals.png", dpi=150, bbox_inches='tight')
    plt.close()


def _print_comparison(overall: dict, run_tag: str):
    print("\n" + "=" * 65)
    print(f"  EVALUATION RESULTS: {run_tag.upper()}")
    print("=" * 65)
    print(f"  {'Metric':<12} {'Reproduced':>12} {'Paper (30-min)':>16} {'Delta':>10}")
    print("-" * 65)
    paper = PAPER_RESULTS.get('LGLM_30min', {})
    for metric in ['RMSE', 'MAE', 'MARD', 'R2']:
        rep = overall.get(metric, float('nan'))
        ref = paper.get(metric, float('nan'))
        delta = rep - ref if not (np.isnan(rep) or np.isnan(ref)) else float('nan')
        unit = "%" if metric == 'MARD' else ""
        delta_str = f"{delta:+.4f}" if not np.isnan(delta) else "N/A"
        print(f"  {metric:<12} {rep:>12.4f}{unit}  {ref:>12.4f}{unit}  {delta_str:>10}")
    print("=" * 65)
    print("\n  NOTE: Delta reflects difference vs paper's synthetic/OhioT1DM values.")
    print("  Differences are EXPECTED when using synthetic data; this is normal.\n")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/reproduction.yaml")
    parser.add_argument("--data_pkl", default="artifacts/processed_data.pkl")
    parser.add_argument("--run_tag", default="lglm")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    with open(args.data_pkl, 'rb') as f:
        processed = pickle.load(f)

    evaluate_all(processed, config, run_tag=args.run_tag)
