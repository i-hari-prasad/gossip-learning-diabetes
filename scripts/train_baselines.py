"""
train_baselines.py
------------------
Trains centralized LSTM baseline on pooled data from all patients.
"""

import os
import sys
import json
import pickle
import logging
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.models.centralized_lstm import CentralizedLSTM
from src.evaluation.metrics import calculate_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


def train_centralized(processed: dict, config: dict, run_tag: str = "centralized_lstm") -> dict:
    cfg = config['centralized_baseline']
    epochs = cfg['epochs']
    hidden = cfg['hidden_units']
    n_layers = cfg['num_layers']
    lr = cfg['learning_rate']
    bs = cfg['batch_size']

    os.makedirs("artifacts/models", exist_ok=True)
    os.makedirs("artifacts/plots", exist_ok=True)
    os.makedirs("artifacts/metrics", exist_ok=True)

    # Pool all patient data
    X_train_all = np.concatenate([d['X_train'] for d in processed.values() if d['X_train'].size > 0])
    Y_train_all = np.concatenate([d['Y_train'] for d in processed.values() if d['Y_train'].size > 0])
    X_val_all = np.concatenate([d['X_val'] for d in processed.values() if d['X_val'].size > 0])
    Y_val_all = np.concatenate([d['Y_val'] for d in processed.values() if d['Y_val'].size > 0])

    n_features = X_train_all.shape[2] if X_train_all.ndim == 3 else 1
    logger.info(f"Centralized training: {X_train_all.shape[0]} train, {X_val_all.shape[0]} val samples")

    model = CentralizedLSTM(input_features=n_features, hidden_units=hidden, num_layers=n_layers)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    train_ds = TensorDataset(torch.FloatTensor(X_train_all), torch.FloatTensor(Y_train_all))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=False)

    train_losses, val_losses = [], []
    best_val = float('inf')
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = []
        for bx, by in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
        train_losses.append(float(np.mean(epoch_loss)))

        model.eval()
        with torch.no_grad():
            val_preds = model(torch.FloatTensor(X_val_all)).numpy()
        val_mse = float(((Y_val_all - val_preds) ** 2).mean())
        val_losses.append(val_mse)
        scheduler.step(val_mse)

        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch:3d}/{epochs} | train_MSE={train_losses[-1]:.5f} | val_MSE={val_mse:.5f}")

    # Restore best weights
    if best_state:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), f"artifacts/models/{run_tag}.pt")

    # -- Per-patient test evaluation --
    test_results = {}
    all_true, all_pred = [], []
    model.eval()
    for pid, pdata in processed.items():
        if pdata['X_test'].size == 0:
            continue
        with torch.no_grad():
            preds = model(torch.FloatTensor(pdata['X_test'])).numpy()
        m = calculate_metrics(pdata['Y_test'], preds)
        test_results[pid] = m
        all_true.extend(pdata['Y_test'].tolist())
        all_pred.extend(preds.tolist())

    results = {
        'per_patient': test_results,
        'overall': calculate_metrics(np.array(all_true), np.array(all_pred)),
        'model_params': model.count_parameters(),
        'train_losses': train_losses,
        'val_losses': val_losses
    }

    # -- Training curve plot --
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#1a1a2e')
    ax.set_facecolor('#16213e')
    ax.plot(train_losses, color='#00d2ff', linewidth=1.5, label='Train MSE')
    ax.plot(val_losses, color='#f7971e', linewidth=1.5, label='Val MSE')
    ax.set_xlabel("Epoch", color='white', fontsize=12)
    ax.set_ylabel("MSE (normalized)", color='white', fontsize=12)
    ax.set_title("Centralized LSTM Training Curve", color='white', fontsize=14, fontweight='bold')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444466')
    ax.legend(facecolor='#2d2d4e', edgecolor='#555577', labelcolor='white')
    plt.tight_layout()
    plt.savefig(f"artifacts/plots/{run_tag}_training.png", dpi=150, bbox_inches='tight')
    plt.close()

    out_path = f"artifacts/metrics/{run_tag}_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"[OK] Results saved to {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/reproduction.yaml")
    parser.add_argument("--data_pkl", default="artifacts/processed_data.pkl")
    parser.add_argument("--run_tag", default="centralized_lstm")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if not os.path.exists(args.data_pkl):
        logger.error(f"Data pickle not found: {args.data_pkl}. Run prepare_data.py first.")
        sys.exit(1)

    with open(args.data_pkl, 'rb') as f:
        processed = pickle.load(f)

    results = train_centralized(processed, config, run_tag=args.run_tag)
    overall = results['overall']
    print(f"\nCentralized LSTM - RMSE: {overall['RMSE']:.4f} | MAE: {overall['MAE']:.4f} | "
          f"MARD: {overall['MARD']:.2f}% | R²: {overall['R2']:.4f}")
