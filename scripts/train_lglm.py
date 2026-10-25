"""
train_lglm.py
-------------
Federated Gossip Learning training loop for the Lightweight LGLM model.

Protocol:
  For each gossip round t:
    1. Each patient node i trains locally: W_i <- W_i - η ∇L(W_i, D_i)
    2. Energy-aware check: can node i communicate?
    3. If yes: W_i <- α*W_i + (1-α)/|N| * Σ_{j∈N} W_j
"""

import os
import sys
import copy
import json
import logging
import argparse
import yaml
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.models.lightweight_lglm import LightweightLGLM
from src.training.local_training import train_local_model, evaluate_model
from src.communication.gossip import aggregate_models
from src.communication.topology import create_topology, get_neighbors
from src.energy.battery_simulator import BatterySimulator
from src.evaluation.metrics import calculate_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


def train_gossip(processed_patients: dict, config: dict,
                 topology_type: str = None, energy_aware: bool = True,
                 run_tag: str = "lglm") -> dict:
    """
    Full gossip learning training loop.

    Args:
        processed_patients: dict of {pid: {X_train, Y_train, X_val, Y_val, X_test, Y_test, n_features}}
        config: reproduction.yaml loaded dict
        topology_type: override topology (uses config value if None)
        energy_aware: whether to enforce energy threshold
        run_tag: prefix for saved artifacts

    Returns:
        results dict with per-patient metrics + energy stats
    """
    gl_cfg = config['gossip_learning']
    e_cfg = config['energy']
    rounds = gl_cfg['rounds']
    local_epochs = gl_cfg['local_epochs']
    lr = gl_cfg['learning_rate']
    alpha = gl_cfg['aggregation_alpha']
    hidden_units = gl_cfg['hidden_units']
    sparsity = gl_cfg.get('sparsity_ratio', 0.0)
    topo = topology_type or gl_cfg['topology']

    patient_ids = list(processed_patients.keys())
    n_features = processed_patients[patient_ids[0]]['n_features']
    window_size = config['preprocessing']['window_size']

    os.makedirs("artifacts/models", exist_ok=True)
    os.makedirs("artifacts/plots", exist_ok=True)
    os.makedirs("artifacts/metrics", exist_ok=True)

    # -- Initialize one model + battery per patient --
    models = {}
    batteries = {}
    for pid in patient_ids:
        m = LightweightLGLM(input_features=n_features, hidden_units=hidden_units,
                            window_size=window_size, sparsity_ratio=sparsity)
        models[pid] = m
        batteries[pid] = BatterySimulator(
            initial_battery=e_cfg['initial_battery_mJ'],
            threshold=e_cfg['threshold_mJ'],
            tx_power=e_cfg['tx_power_mW'],
            tx_time=e_cfg['tx_time_ms']
        )

    # -- Build topology graph --
    G = create_topology(patient_ids, topology_type=topo)
    logger.info(f"Topology: {topo} | Nodes: {len(patient_ids)} | Edges: {G.number_of_edges()}")

    # -- Round history for convergence plot --
    round_val_losses = {pid: [] for pid in patient_ids}
    energy_log = []

    # -- Main gossip loop --
    for rnd in range(1, rounds + 1):
        round_energy = {}

        # 1. Local training
        for pid in patient_ids:
            pdata = processed_patients[pid]
            if pdata['X_train'].size == 0:
                continue
            models[pid], _ = train_local_model(
                models[pid], pdata['X_train'], pdata['Y_train'],
                epochs=local_epochs, batch_size=gl_cfg['batch_size'], lr=lr
            )
            batteries[pid].consume_compute_energy(epochs=local_epochs)

        # 2. Gossip aggregation (energy-aware)
        new_states = {}
        for pid in patient_ids:
            if not batteries[pid].can_communicate(energy_aware=energy_aware):
                new_states[pid] = models[pid].state_dict()  # keep local
                continue
            neighbors = get_neighbors(G, pid)
            alive_neighbors = [pid2 for pid2 in neighbors if batteries[pid2].is_alive]
            neighbor_models = [models[nb] for nb in alive_neighbors]
            new_states[pid] = aggregate_models(models[pid], neighbor_models, alpha=alpha)
            batteries[pid].consume_communication_energy()
            round_energy[pid] = batteries[pid].current_battery

        # Load aggregated weights
        for pid in patient_ids:
            models[pid].load_state_dict(new_states[pid])

        # 3. Validation loss tracking (every 5 rounds)
        if rnd % 5 == 0 or rnd == rounds:
            val_losses = {}
            for pid in patient_ids:
                pdata = processed_patients[pid]
                if pdata['X_val'].size > 0:
                    mse, _ = evaluate_model(models[pid], pdata['X_val'], pdata['Y_val'])
                    round_val_losses[pid].append(mse)
                    val_losses[pid] = mse
            energy_snapshot = {pid: batteries[pid].current_battery for pid in patient_ids}
            energy_log.append({'round': rnd, 'val_losses': val_losses,
                                'battery': energy_snapshot})
            avg_val = np.mean(list(val_losses.values())) if val_losses else 0
            logger.info(f"Round {rnd:3d}/{rounds} | avg_val_MSE={avg_val:.5f}")

    # -- Final evaluation on test set --
    test_results = {}
    all_true, all_pred = [], []
    for pid in patient_ids:
        pdata = processed_patients[pid]
        if pdata['X_test'].size == 0:
            continue
        _, preds = evaluate_model(models[pid], pdata['X_test'], pdata['Y_test'])
        m = calculate_metrics(pdata['Y_test'], preds)
        m['energy'] = batteries[pid].get_stats()
        test_results[pid] = m
        all_true.extend(pdata['Y_test'].tolist())
        all_pred.extend(preds.tolist())

        # Save model
        torch.save(models[pid].state_dict(),
                   f"artifacts/models/{run_tag}_patient_{pid}.pt")

    # -- Convergence plot --
    _plot_convergence(round_val_losses, rounds, run_tag)

    # -- Energy plot --
    _plot_energy(energy_log, patient_ids, rounds, run_tag)

    # -- Save metrics --
    results = {
        'per_patient': test_results,
        'overall': calculate_metrics(np.array(all_true), np.array(all_pred)),
        'energy_log': energy_log,
        'topology': topo,
        'energy_aware': energy_aware,
        'model_params': models[patient_ids[0]].count_parameters() if patient_ids else {}
    }
    out_path = f"artifacts/metrics/{run_tag}_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {out_path}")
    return results


def _plot_convergence(round_val_losses: dict, rounds: int, tag: str):
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#1a1a2e')
    ax.set_facecolor('#16213e')
    palette = plt.cm.cool(np.linspace(0.2, 0.9, len(round_val_losses)))
    for (pid, losses), color in zip(round_val_losses.items(), palette):
        if losses:
            xs = np.linspace(5, rounds, len(losses))
            ax.plot(xs, losses, marker='o', markersize=3, label=f"Patient {pid}", color=color, linewidth=1.5)
    ax.set_xlabel("Gossip Round", color='white', fontsize=12)
    ax.set_ylabel("Validation MSE (normalized)", color='white', fontsize=12)
    ax.set_title("Gossip Learning Convergence", color='white', fontsize=14, fontweight='bold')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444466')
    ax.legend(facecolor='#2d2d4e', edgecolor='#555577', labelcolor='white', fontsize=8)
    plt.tight_layout()
    plt.savefig(f"artifacts/plots/{tag}_convergence.png", dpi=150, bbox_inches='tight')
    plt.close()


def _plot_energy(energy_log: list, patient_ids: list, rounds: int, tag: str):
    if not energy_log:
        return
    fig, ax = plt.subplots(figsize=(10, 4), facecolor='#1a1a2e')
    ax.set_facecolor('#16213e')
    palette = plt.cm.plasma(np.linspace(0.1, 0.9, len(patient_ids)))
    for pid, color in zip(patient_ids, palette):
        xs = [e['round'] for e in energy_log if pid in e.get('battery', {})]
        ys = [e['battery'][pid] for e in energy_log if pid in e.get('battery', {})]
        if xs:
            ax.plot(xs, ys, marker='s', markersize=3, label=f"P-{pid}", color=color, linewidth=1.5)
    ax.axhline(y=10.0, color='#ff6b6b', linestyle='--', linewidth=1, label='Threshold')
    ax.set_xlabel("Gossip Round", color='white', fontsize=12)
    ax.set_ylabel("Battery (mJ)", color='white', fontsize=12)
    ax.set_title("Energy Consumption Over Rounds", color='white', fontsize=14, fontweight='bold')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444466')
    ax.legend(facecolor='#2d2d4e', edgecolor='#555577', labelcolor='white', fontsize=8)
    plt.tight_layout()
    plt.savefig(f"artifacts/plots/{tag}_energy.png", dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/reproduction.yaml")
    parser.add_argument("--topology", default=None)
    parser.add_argument("--no_energy_aware", action="store_true")
    parser.add_argument("--run_tag", default="lglm")
    parser.add_argument("--data_pkl", default="artifacts/processed_data.pkl",
                        help="Path to preprocessed data pickle")
    args = parser.parse_args()

    import pickle
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if not os.path.exists(args.data_pkl):
        logger.error(f"Preprocessed data not found at {args.data_pkl}. Run prepare_data.py first.")
        sys.exit(1)

    with open(args.data_pkl, 'rb') as f:
        processed = pickle.load(f)

    results = train_gossip(
        processed, config,
        topology_type=args.topology,
        energy_aware=not args.no_energy_aware,
        run_tag=args.run_tag
    )

    print("\n=== RESULTS ===")
    overall = results['overall']
    print(f"Overall RMSE: {overall.get('RMSE', 'N/A'):.4f}")
    print(f"Overall MAE:  {overall.get('MAE', 'N/A'):.4f}")
    print(f"Overall MARD: {overall.get('MARD', 'N/A'):.2f}%")
    print(f"Overall R²:   {overall.get('R2', 'N/A'):.4f}")
